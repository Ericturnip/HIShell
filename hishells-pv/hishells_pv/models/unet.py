"""PyTorch U-Net for position-velocity (PV) shell segmentation.

The network outputs raw logits (no final sigmoid); apply ``torch.sigmoid``
downstream for probabilities. This matches :class:`hishells_pv.models.losses.BCETverskyLoss`,
which expects logits.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two dilated conv layers with BatchNorm + ReLU and optional dropout."""

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
    """Encoder/decoder U-Net operating on single-channel PV patches.

    Parameters mirror the YAML ``model`` section so a config dict can be
    splatted directly via :meth:`from_config`.
    """

    def __init__(
        self,
        *,
        base_filters: int = 24,
        depth: int = 3,
        dilation_rate: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = {
            "base_filters": int(base_filters),
            "depth": int(depth),
            "dilation_rate": int(dilation_rate),
            "dropout": float(dropout),
        }

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

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "UNetPV":
        cfg = cfg or {}
        return cls(
            base_filters=int(cfg.get("base_filters", 24)),
            depth=int(cfg.get("depth", 3)),
            dilation_rate=int(cfg.get("dilation_rate", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
        )

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
