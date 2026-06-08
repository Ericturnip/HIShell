"""Pixel- and patch-level segmentation metrics for PV-shell evaluation (PyTorch)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class BinaryCounts:
    """Running confusion counts for a single threshold."""

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


def threshold_tag(t: float) -> str:
    """Stable metric-name suffix for a threshold, e.g. 0.075 -> '0p075'."""
    return str(t).replace(".", "p")


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
    """Compute mean loss plus pixel/patch precision-recall-F1 at each threshold.

    A "patch" is positive iff any label pixel is set; its predicted score is the
    max probability over the patch. This mirrors the detection-style objective:
    we care whether a shell is flagged anywhere in the cut, not pixel-perfect masks.
    """
    model.eval()
    losses: list[float] = []
    pixel_counts = {t: BinaryCounts() for t in thresholds}
    patch_counts = {t: BinaryCounts() for t in thresholds}
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        losses.append(float(loss_fn(logits, y).detach().cpu()))
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
        tag = threshold_tag(t)
        pix = pixel_counts[t].result()
        pat = patch_counts[t].result()
        out[f"pixel_precision_{tag}"] = pix["precision"]
        out[f"pixel_recall_{tag}"] = pix["recall"]
        out[f"patch_precision_{tag}"] = pat["precision"]
        out[f"patch_recall_{tag}"] = pat["recall"]
        out[f"patch_f1_{tag}"] = pat["f1"]
    return out
