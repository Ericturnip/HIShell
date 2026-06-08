"""Loss functions for the PyTorch PV detector."""

from __future__ import annotations

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
    """Compute recall-weighted Tversky loss from raw model logits."""
    y_true = y_true.float()
    y_pred = torch.sigmoid(logits)
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1.0 - y_true) * y_pred)
    fn = torch.sum(y_true * (1.0 - y_pred))
    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - score


class BCETverskyLoss(nn.Module):
    """Blend BCE stability with Tversky's stronger false-negative penalty."""

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


__all__ = ["BCETverskyLoss", "tversky_loss_from_logits"]

