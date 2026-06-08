"""Training entry points."""

from __future__ import annotations


def train_torch(*args, **kwargs):
    """Run PyTorch training, importing the trainer only when requested."""
    from .torch import train

    return train(*args, **kwargs)


__all__ = ["train_torch"]
