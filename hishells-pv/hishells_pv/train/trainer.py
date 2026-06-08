"""PyTorch training loop for the PV-shell U-Net.

Saves three checkpoints under ``runs/<run_name>/``:

- ``best_model.pt``         - lowest validation loss
- ``high_recall_model.pt``  - highest validation patch-recall at 0.075
- ``final_model.pt``        - last epoch

Each checkpoint stores ``{"model_state_dict", "cfg", "history"}`` so inference
can rebuild the architecture from the embedded config.
"""
from __future__ import annotations

import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from hishells_pv.data.dataset import PVPatchDataset
from hishells_pv.models.losses import BCETverskyLoss
from hishells_pv.models.metrics import evaluate
from hishells_pv.models.unet import UNetPV
from hishells_pv.utils.io import load_yaml


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a device string, preferring MPS then CUDA when ``auto``."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _default_run_name() -> str:
    return f"pv_unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(
    config: str,
    *,
    run_name: str | None = None,
    device_name: str = "auto",
    runs_root: str | Path = "runs",
    epochs_override: int | None = None,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    num_workers: int = 0,
    smoke: bool = False,
) -> Path:
    """Train the U-Net and return the run output directory."""
    cfg = load_yaml(config)
    device = resolve_device(device_name)
    _seed_everything(42)

    out = Path(runs_root) / (run_name or _default_run_name())
    out.mkdir(parents=True, exist_ok=True)
    (out / "torch_config.json").write_text(
        json.dumps(
            {
                "config": str(Path(config).resolve()),
                "device": str(device),
                "mps_available": torch.backends.mps.is_available(),
                "cuda_available": torch.cuda.is_available(),
                "torch_version": torch.__version__,
            },
            indent=2,
        )
    )

    batch_size = int(cfg["optim"]["batch_size"])
    epochs = int(epochs_override or cfg["optim"]["epochs"])
    if smoke:
        epochs = min(epochs, 1)
        max_train_steps = max_train_steps or 30
        max_val_steps = max_val_steps or 10

    train_ds = PVPatchDataset(cfg, "train", seed=42)
    val_ds = PVPatchDataset(cfg, "val", seed=4242)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model = UNetPV.from_config(cfg.get("model", {})).to(device)
    loss_fn = BCETverskyLoss.from_config(cfg.get("loss", {}))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"].get("weight_decay", 0.0)),
    )
    thresholds = [float(t) for t in cfg.get("metrics", {}).get("thresholds", [0.05, 0.075, 0.1])]

    history_path = out / "history_torch.csv"
    fieldnames = [
        "epoch",
        "learning_rate",
        "loss",
        "seconds",
        "steps",
        "samples_per_second",
        "val_loss",
        "val_patch_precision_0p075",
        "val_patch_recall_0p075",
        "val_patch_f1_0p075",
        "val_pixel_precision_0p075",
        "val_pixel_recall_0p075",
    ]
    best_val_loss = float("inf")
    best_recall = -1.0
    history: list[dict[str, Any]] = []
    with history_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            losses: list[float] = []
            n_samples = 0
            t0 = time.perf_counter()
            for step, (x, y) in enumerate(train_loader, start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
                n_samples += int(x.shape[0])
                if max_train_steps is not None and step >= int(max_train_steps):
                    break
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            val = evaluate(model, val_loader, loss_fn, device, thresholds=thresholds, max_steps=max_val_steps)
            row = {
                "epoch": epoch,
                "learning_rate": opt.param_groups[0]["lr"],
                "loss": float(np.mean(losses)) if losses else None,
                "seconds": seconds,
                "steps": len(losses),
                "samples_per_second": n_samples / max(seconds, 1e-9),
                "val_loss": val["loss"],
                "val_patch_precision_0p075": val.get("patch_precision_0p075"),
                "val_patch_recall_0p075": val.get("patch_recall_0p075"),
                "val_patch_f1_0p075": val.get("patch_f1_0p075"),
                "val_pixel_precision_0p075": val.get("pixel_precision_0p075"),
                "val_pixel_recall_0p075": val.get("pixel_recall_0p075"),
            }
            writer.writerow(row)
            fh.flush()
            history.append(row)
            _print_progress(epoch, device, row)

            if row["val_loss"] is not None and float(row["val_loss"]) < best_val_loss:
                best_val_loss = float(row["val_loss"])
                _save_ckpt(out / "best_model.pt", model, cfg, history)
            recall = row.get("val_patch_recall_0p075")
            if recall is not None and float(recall) > best_recall:
                best_recall = float(recall)
                _save_ckpt(out / "high_recall_model.pt", model, cfg, history)

    _save_ckpt(out / "final_model.pt", model, cfg, history)
    (out / "history_torch.json").write_text(json.dumps(history, indent=2))
    print(f"[torch-done] saved run -> {out.resolve()}", flush=True)
    return out


def _save_ckpt(path: Path, model: torch.nn.Module, cfg: dict[str, Any], history: list[dict[str, Any]]) -> None:
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg, "history": history}, path)


def _print_progress(epoch: int, device: torch.device, row: dict[str, Any]) -> None:
    loss = row["loss"] if row["loss"] is not None else float("nan")
    val_loss = row["val_loss"] if row["val_loss"] is not None else float("nan")
    recall = row["val_patch_recall_0p075"] if row["val_patch_recall_0p075"] is not None else float("nan")
    print(
        "[torch-progress] "
        f"epoch={epoch} device={device} loss={loss:.4f} "
        f"val_loss={val_loss:.4f} val_patch_recall_0p075={recall:.4f} "
        f"samples_per_second={row['samples_per_second']:.2f}",
        flush=True,
    )
