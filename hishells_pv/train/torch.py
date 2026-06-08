"""PyTorch training entry point for the PV detector."""

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
import yaml

from hishells_pv.data.dataset import PVPatchDataset
from hishells_pv.models.losses import BCETverskyLoss
from hishells_pv.models.metrics import evaluate
from hishells_pv.models.unet import UNetPV


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _default_run_name() -> str:
    return f"pv_unet_torch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def train(
    config: str,
    *,
    run_name: str | None,
    device_name: str = "auto",
    epochs_override: int | None = None,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    num_workers: int = 0,
    smoke: bool = False,
) -> Path:
    """Train the PyTorch PV-shell U-Net from a generated dataset config."""
    cfg = _load_yaml(config)
    device = _device(device_name)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    out = Path("runs") / (run_name or _default_run_name())
    out.mkdir(parents=True, exist_ok=True)
    (out / "torch_config.json").write_text(
        json.dumps(
            {
                "config": str(Path(config).resolve()),
                "device": str(device),
                "mps_available": torch.backends.mps.is_available(),
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

    model_cfg = cfg.get("model", {})
    model = UNetPV(
        base_filters=int(model_cfg.get("base_filters", 24)),
        depth=int(model_cfg.get("depth", 3)),
        dilation_rate=int(model_cfg.get("dilation_rate", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    loss_cfg = cfg.get("loss", {})
    loss_fn = BCETverskyLoss(
        alpha=float(loss_cfg.get("tversky_alpha", 0.3)),
        beta=float(loss_cfg.get("tversky_beta", 0.7)),
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        tversky_weight=float(loss_cfg.get("tversky_weight", 0.5)),
    )
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
            print(
                "[torch-progress] "
                f"epoch={epoch} device={device} loss={row['loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} "
                f"val_patch_recall_0p075={row['val_patch_recall_0p075']:.4f} "
                f"samples_per_second={row['samples_per_second']:.2f}",
                flush=True,
            )
            if row["val_loss"] is not None and float(row["val_loss"]) < best_val_loss:
                best_val_loss = float(row["val_loss"])
                torch.save({"model_state_dict": model.state_dict(), "cfg": cfg, "history": history}, out / "best_model.pt")
            recall = row.get("val_patch_recall_0p075")
            if recall is not None and float(recall) > best_recall:
                best_recall = float(recall)
                torch.save(
                    {"model_state_dict": model.state_dict(), "cfg": cfg, "history": history},
                    out / "high_recall_model.pt",
                )

    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg, "history": history}, out / "final_model.pt")
    (out / "history_torch.json").write_text(json.dumps(history, indent=2))
    print(f"[torch-done] saved run -> {out.resolve()}", flush=True)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--device", default="auto", help="auto, mps, cpu, or cuda")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-train-steps", type=int, default=None)
    ap.add_argument("--max-val-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    train(
        args.config,
        run_name=args.run,
        device_name=args.device,
        epochs_override=args.epochs,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        num_workers=args.num_workers,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "train"]

