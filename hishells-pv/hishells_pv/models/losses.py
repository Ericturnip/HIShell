"""Segmentation losses for PV-shell U-Net training (PyTorch).

All losses operate on raw logits (the model output), applying sigmoid
internally where needed.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def tversky_loss_from_logits(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft Tversky loss. ``alpha`` weights false positives, ``beta`` weights false negatives.

    With ``beta > alpha`` the loss penalizes missed shell pixels more than spurious
    ones, which biases the model toward high recall (the project objective).
    """
    y_true = y_true.float()
    y_pred = torch.sigmoid(logits)
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1.0 - y_true) * y_pred)
    fn = torch.sum(y_true * (1.0 - y_pred))
    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - score


class BCETverskyLoss(nn.Module):
    """Weighted sum of ``BCEWithLogits`` and Tversky loss."""

    def __init__(
        self,
        *,
        alpha: float = 0.3,
        beta: float = 0.7,
        bce_weight: float = 0.5,
        tversky_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.bce_weight = float(bce_weight)
        self.tversky_weight = float(tversky_weight)
        self.smooth = float(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "BCETverskyLoss":
        cfg = cfg or {}
        return cls(
            alpha=float(cfg.get("tversky_alpha", 0.3)),
            beta=float(cfg.get("tversky_beta", 0.7)),
            bce_weight=float(cfg.get("bce_weight", 0.5)),
            tversky_weight=float(cfg.get("tversky_weight", 0.5)),
            smooth=float(cfg.get("smooth", 1.0)),
        )

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, y_true.float())
        tversky = tversky_loss_from_logits(
            logits,
            y_true,
            alpha=self.alpha,
            beta=self.beta,
            smooth=self.smooth,
        )
        return self.bce_weight * bce + self.tversky_weight * tversky
