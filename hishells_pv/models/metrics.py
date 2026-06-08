"""Patch-level and pixel-level detector metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class BinaryCounts:
    """Accumulate true-positive, false-positive, and false-negative counts."""

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
    """Evaluate pixel and patch metrics at one or more probability thresholds."""
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


__all__ = ["BinaryCounts", "evaluate"]

