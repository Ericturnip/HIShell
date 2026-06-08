"""Shared pytest fixtures: a tiny synthetic standardized-PV dataset on disk."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PH, PW = 32, 64  # small patch (vel x pos) to keep tests fast


def _write_split(root: Path, split: str, n: int, rng: np.random.Generator) -> None:
    pv_dir = root / "pv"
    lab_dir = root / "labels"
    split_dir = root / "splits"
    for d in (pv_dir, lab_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(n):
        name = f"{split}_{i:03d}.npy"
        pv = rng.normal(size=(PH, PW)).astype(np.float32)
        lab = np.zeros((PH, PW), dtype=np.float32)
        # Embed a positive blob in half the samples + a matching bright PV patch.
        if i % 2 == 0:
            v0, s0 = PH // 4, PW // 4
            lab[v0 : v0 + 6, s0 : s0 + 10] = 1.0
            pv[v0 : v0 + 6, s0 : s0 + 10] += 6.0
        np.save(pv_dir / name, pv)
        np.save(lab_dir / name, lab)
        names.append(name)
    (split_dir / f"{split}_manifest.txt").write_text("\n".join(names) + "\n")


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> dict:
    """Create a small synthetic dataset and return a training config dict."""
    root = tmp_path / "data"
    rng = np.random.default_rng(0)
    _write_split(root, "train", 8, rng)
    _write_split(root, "val", 4, rng)
    _write_split(root, "test", 4, rng)
    return {
        "output_root": str(root),
        "standardized_pv": {"enabled": True},
        "train": {
            "patch_vel": PH,
            "patch_pos": PW,
            "norm_method": "zscore_galaxy_only",
            "strict_fixed_shape": True,
        },
        "model": {"base_filters": 8, "depth": 3, "dilation_rate": 1, "dropout": 0.0},
        "loss": {"tversky_alpha": 0.3, "tversky_beta": 0.7, "bce_weight": 0.5, "tversky_weight": 0.5},
        "optim": {"batch_size": 2, "epochs": 1, "lr": 1e-3, "weight_decay": 0.0},
        "metrics": {"thresholds": [0.05, 0.075, 0.1]},
    }


GAL = "tinygal"
IMG = 40  # tiny sky image side (pixels)


def _write_tiny_cube(path: Path) -> None:
    """A minimal 2D FITS image with a valid celestial WCS + beam keywords."""
    from astropy.io import fits

    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = IMG
    hdr["NAXIS2"] = IMG
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = IMG / 2.0
    hdr["CRPIX2"] = IMG / 2.0
    hdr["CRVAL1"] = 150.0
    hdr["CRVAL2"] = 2.0
    hdr["CDELT1"] = -1.0 / 3600.0  # 1 arcsec/pix
    hdr["CDELT2"] = 1.0 / 3600.0
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["BMAJ"] = 6.0 / 3600.0  # 6 arcsec beam
    hdr["BMIN"] = 6.0 / 3600.0
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=np.zeros((IMG, IMG), dtype=np.float32), header=hdr).writeto(path, overwrite=True)


@pytest.fixture()
def tiny_agg_dataset(tmp_path: Path) -> dict:
    """A synthetic dataset wired for the aggregate stage.

    Returns a training config (for producing a checkpoint) and a per-galaxy
    aggregate config plus its split label.
    """
    rng = np.random.default_rng(7)
    root = tmp_path / "data"
    _write_split(root, "train", 8, rng)
    _write_split(root, "val", 4, rng)

    # Per-galaxy aggregation input dir: pv (npy + _posxy.npy + .json) + manifest.
    agg_root = tmp_path / "agg" / GAL
    pv_dir = agg_root / "pv"
    splits_dir = agg_root / "splits"
    pv_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    label = f"{GAL}_test"
    names = []
    for i in range(4):
        stem = f"{GAL}__std_{i:06d}"
        pv = rng.normal(size=(PH, PW)).astype(np.float32)
        pv[PH // 4 : PH // 4 + 6, PW // 4 : PW // 4 + 10] += 6.0
        np.save(pv_dir / f"{stem}.npy", pv)
        posxy = np.stack(
            [rng.integers(0, IMG, size=PW), rng.integers(0, IMG, size=PW)], axis=1
        ).astype(np.float32)
        np.save(pv_dir / f"{stem}_posxy.npy", posxy)
        (pv_dir / f"{stem}.json").write_text("{}")
        names.append(f"{stem}.npy")
    (splits_dir / f"{label}_manifest.txt").write_text("".join(f"{n}\n" for n in names))

    cube_path = tmp_path / "cube.fits"
    _write_tiny_cube(cube_path)

    train_config = {
        "output_root": str(root),
        "standardized_pv": {"enabled": True},
        "train": {
            "patch_vel": PH,
            "patch_pos": PW,
            "norm_method": "zscore_galaxy_only",
            "strict_fixed_shape": True,
        },
        "model": {"base_filters": 8, "depth": 3, "dilation_rate": 1, "dropout": 0.0},
        "loss": {"tversky_alpha": 0.3, "tversky_beta": 0.7, "bce_weight": 0.5, "tversky_weight": 0.5},
        "optim": {"batch_size": 2, "epochs": 1, "lr": 1e-3, "weight_decay": 0.0},
        "metrics": {"thresholds": [0.05, 0.075, 0.1]},
    }
    agg_config = {
        "cube_path": str(cube_path),
        "output_root": str(agg_root),
        "galaxy": {"name": GAL, "beam_fwhm_arcsec": 6.0},
        "train": {"patch_vel": PH, "patch_pos": PW, "norm_method": "zscore_galaxy_only"},
        "post": {"min_component_area_pv": [6, 5]},
        "aggregate": {
            "vote_sigma_pix": 3,
            "nms_radius_pix": 5,
            "min_radius_beams": 2.5,
            "max_thickness_frac": 0.4,
        },
    }
    return {
        "train_config": train_config,
        "agg_config": agg_config,
        "label": label,
        "galaxy_id": GAL,
    }
