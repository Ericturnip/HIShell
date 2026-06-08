"""Torch dataset and array preprocessing for PV-shell training.

Expected on-disk layout under ``cfg['output_root']``::

    output_root/
      pv/<name>.npy        # (V, S) float PV cut
      labels/<name>.npy    # (V, S) binary/float shell mask
      splits/<split>_manifest.txt   # one .npy filename per line

Sidecar files ending in ``_posxy.npy`` are ignored by the dataset.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def zscore_finite(pv: np.ndarray) -> np.ndarray:
    """Z-score using only finite pixels; non-finite values map to 0."""
    finite = pv[np.isfinite(pv)]
    if finite.size == 0:
        return np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mu = float(np.mean(finite))
    sigma = float(np.std(finite) + 1e-6)
    return np.nan_to_num((pv - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize(pv: np.ndarray, method: str | None) -> np.ndarray:
    if method == "zscore_galaxy_only":
        return zscore_finite(pv)
    if method in ("none", None):
        return np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raise ValueError(f"Unknown norm_method: {method}")


def pad_to(pv: np.ndarray, lab: np.ndarray, ph: int, pw: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric constant-pad PV (to its finite median) and label (to 0) up to (ph, pw)."""
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


def read_manifest(root: Path, split: str) -> list[str]:
    path = root / "splits" / f"{split}_manifest.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


class PVPatchDataset(Dataset):
    """Yields ``(x, y)`` tensors of shape ``(1, patch_vel, patch_pos)``."""

    def __init__(self, cfg: dict[str, Any], split: str, *, seed: int = 1337) -> None:
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg["output_root"])
        files = [f for f in read_manifest(self.root, split) if not f.endswith("_posxy.npy")]
        if split == "train":
            random.Random(seed).shuffle(files)
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
        pv = normalize(pv, self.norm)
        pv, lab = pad_to(pv, lab, self.ph, self.pw)
        if self.strict_shape and pv.shape != (self.ph, self.pw):
            raise ValueError(f"{fname} has shape {pv.shape}; expected {(self.ph, self.pw)}")
        x = torch.from_numpy(pv.astype(np.float32, copy=False)).unsqueeze(0)
        y = torch.from_numpy(lab.astype(np.float32, copy=False)).unsqueeze(0)
        return x, y
