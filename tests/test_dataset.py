from pathlib import Path

import numpy as np
import yaml

from hishells_pv.data import PVPatchDataset, zscore_finite


def _write_synthetic_dataset(root: Path) -> dict:
    for sub in ("pv", "labels", "splits"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    pv = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
    label = np.zeros((32, 64), dtype=np.uint8)
    label[10:14, 20:24] = 1
    np.save(root / "pv" / "cut_000.npy", pv)
    np.save(root / "labels" / "cut_000.npy", label)
    (root / "splits" / "train_manifest.txt").write_text("cut_000.npy\n")
    return {
        "output_root": str(root),
        "standardized_pv": {"enabled": True},
        "train": {
            "norm_method": "zscore_galaxy_only",
            "patch_vel": 32,
            "patch_pos": 64,
            "strict_fixed_shape": True,
            "samples_per_pv": 1,
        },
    }


def test_pv_patch_dataset_loads_generated_arrays(tmp_path):
    cfg = _write_synthetic_dataset(tmp_path)
    dataset = PVPatchDataset(cfg, "train")

    x, y = dataset[0]

    assert tuple(x.shape) == (1, 32, 64)
    assert tuple(y.shape) == (1, 32, 64)
    assert y.max().item() == 1.0


def test_zscore_finite_handles_nan():
    pv = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)

    out = zscore_finite(pv)

    assert np.isfinite(out).all()

