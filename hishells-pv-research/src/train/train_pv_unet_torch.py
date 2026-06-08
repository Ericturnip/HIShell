from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _read_manifest(root: Path, split: str) -> list[str]:
    path = root / "splits" / f"{split}_manifest.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _zscore_finite(pv: np.ndarray) -> np.ndarray:
    finite = pv[np.isfinite(pv)]
    if finite.size == 0:
        return np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mu = float(np.mean(finite))
    sigma = float(np.std(finite) + 1e-6)
    return np.nan_to_num((pv - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _normalize(pv: np.ndarray, method: str) -> np.ndarray:
    if method == "zscore_galaxy_only":
        return _zscore_finite(pv)
    if method in ("none", None):
        return np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raise ValueError(f"Unknown norm_method: {method}")


def _pad_to(pv: np.ndarray, lab: np.ndarray, ph: int, pw: int) -> tuple[np.ndarray, np.ndarray]:
    v, s = pv.shape
    dv = max(0, ph - v)
    ds = max(0, pw - s)
    if dv == 0 and ds == 0:
        return pv, lab
    finite = pv[np.isfinite(pv)]
    pad_value = float(np.nanmedian(finite)) if finite.size else 0.0
    pad = ((dv // 2, dv - dv // 2), (ds // 2, ds - ds // 2))
    return (
        np.pad(pv, pad, mode="constant", constant_values=pad_value),
        np.pad(lab, pad, mode="constant", constant_values=0),
    )


class PVPatchDataset(Dataset):
    def __init__(self, cfg: dict[str, Any], split: str, *, seed: int = 1337) -> None:
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg["output_root"])
        files = [f for f in _read_manifest(self.root, split) if not f.endswith("_posxy.npy")]
        if split == "train":
            rng = random.Random(seed)
            rng.shuffle(files)
        samples_per_pv = cfg.get("train", {}).get("samples_per_pv")
        repeat = max(1, int(samples_per_pv)) if samples_per_pv is not None else 1
        self.files = [fname for fname in files for _ in range(repeat)]
        self.norm = cfg["train"].get("norm_method", "zscore_galaxy_only")
        self.ph = int(cfg["train"]["patch_vel"])
        self.pw = int(cfg["train"]["patch_pos"])
        self.strict_shape = bool(cfg.get("standardized_pv", {}).get("enabled")) and bool(
            cfg["train"].get("strict_fixed_shape", True)
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        pv = np.load(self.root / "pv" / fname)
        lab = np.load(self.root / "labels" / fname)
        pv = _normalize(pv, self.norm)
        pv, lab = _pad_to(pv, lab, self.ph, self.pw)
        if self.strict_shape and pv.shape != (self.ph, self.pw):
            raise ValueError(f"{fname} has shape {pv.shape}; expected {(self.ph, self.pw)}")
        x = torch.from_numpy(pv.astype(np.float32, copy=False)).unsqueeze(0)
        y = torch.from_numpy(lab.astype(np.float32, copy=False)).unsqueeze(0)
        return x, y


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        padding = int(dilation)
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(float(dropout)))
        layers.extend(
            [
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetPV(nn.Module):
    def __init__(self, *, base_filters: int = 24, depth: int = 3, dilation_rate: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = 1
        filters = int(base_filters)
        for d in range(int(depth)):
            self.down_blocks.append(
                ConvBlock(in_ch, filters, dilation=int(dilation_rate), dropout=float(dropout) if d > 0 else 0.0)
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2))
            in_ch = filters
            filters *= 2
        self.bottleneck = ConvBlock(in_ch, filters, dilation=int(dilation_rate), dropout=float(dropout))

        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for d in reversed(range(int(depth))):
            filters //= 2
            self.up_convs.append(nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2))
            self.up_blocks.append(
                ConvBlock(filters * 2, filters, dilation=1, dropout=float(dropout) if d > 0 else 0.0)
            )
        self.out = nn.Conv2d(filters, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.out(x)


def tversky_loss_from_logits(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    y_true = y_true.float()
    y_pred = torch.sigmoid(logits)
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1.0 - y_true) * y_pred)
    fn = torch.sum(y_true * (1.0 - y_pred))
    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - score


class BCETverskyLoss(nn.Module):
    def __init__(self, *, alpha: float, beta: float, bce_weight: float, tversky_weight: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.bce_weight = float(bce_weight)
        self.tversky_weight = float(tversky_weight)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, y_true) + self.tversky_weight * tversky_loss_from_logits(
            logits,
            y_true,
            alpha=self.alpha,
            beta=self.beta,
        )


@dataclass
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, tp: int, fp: int, fn: int) -> None:
        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)

    def result(self) -> dict[str, float | int]:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1, "tp": self.tp, "fp": self.fp, "fn": self.fn}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    thresholds: list[float],
    max_steps: int | None = None,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    pixel_counts = {t: BinaryCounts() for t in thresholds}
    patch_counts = {t: BinaryCounts() for t in thresholds}
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.detach().cpu()))
        prob = torch.sigmoid(logits)
        y_bool = y > 0.5
        y_patch = torch.any(y_bool.flatten(1), dim=1)
        patch_scores = torch.amax(prob.flatten(1), dim=1)
        for t in thresholds:
            pred = prob >= float(t)
            pixel_counts[t].update(
                int(torch.logical_and(pred, y_bool).sum().item()),
                int(torch.logical_and(pred, ~y_bool).sum().item()),
                int(torch.logical_and(~pred, y_bool).sum().item()),
            )
            pred_patch = patch_scores >= float(t)
            patch_counts[t].update(
                int(torch.logical_and(pred_patch, y_patch).sum().item()),
                int(torch.logical_and(pred_patch, ~y_patch).sum().item()),
                int(torch.logical_and(~pred_patch, y_patch).sum().item()),
            )
        if max_steps is not None and step >= int(max_steps):
            break
    out: dict[str, Any] = {"loss": float(np.mean(losses)) if losses else None}
    for t in thresholds:
        tag = str(t).replace(".", "p")
        out[f"pixel_precision_{tag}"] = pixel_counts[t].result()["precision"]
        out[f"pixel_recall_{tag}"] = pixel_counts[t].result()["recall"]
        out[f"patch_precision_{tag}"] = patch_counts[t].result()["precision"]
        out[f"patch_recall_{tag}"] = patch_counts[t].result()["recall"]
        out[f"patch_f1_{tag}"] = patch_counts[t].result()["f1"]
    return out


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


if __name__ == "__main__":
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
