"""
Load generated PV cuts and masks as TensorFlow datasets.
Training uses this module after standardized-data scripts have already written
fixed-shape arrays and split manifests.
"""

from __future__ import annotations
import random
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from src.utils.io import load_yaml

AUTOTUNE = tf.data.AUTOTUNE

def _read_manifest(path: Path) -> List[str]:
    """Read one legacy split manifest containing PV filenames."""
    assert path.exists(), f"Manifest missing: {path}"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _resolve_cfg(cfg_path: str) -> Dict:
    """
    Load the training config in the lightweight training environment.
    Generation uses the full FITS/WCS resolver, but training only needs the
    output, model, and optimizer settings.
    """
    cfg = Path(cfg_path)
    resolved = cfg.resolve().with_name(cfg.stem + "._resolved.yaml")
    if resolved.exists():
        return load_yaml(resolved)
    try:
        from src.utils.config import resolve_config
        return resolve_config(str(cfg), write_resolved=True)
    except Exception as exc:
        print(f"[dataset] warning: using raw training config; full resolver unavailable ({exc})")
        return load_yaml(cfg)


def _load_pair(root: Path, fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one PV cut and its label mask from the generated dataset."""
    pv_path = root / "pv" / fname
    lab_path = root / "labels" / fname
    if not pv_path.exists() or not lab_path.exists():
        raise FileNotFoundError(f"Missing PV/label pair: {pv_path.name}")
    pv = np.load(pv_path)
    lab = np.load(lab_path)
    assert pv.shape == lab.shape, f"shape mismatch for {fname}: {pv.shape} vs {lab.shape}"
    return pv, lab

def _zscore_galaxy_only(pv: np.ndarray) -> np.ndarray:
    """Normalize a PV cut using the finite values inside that cut only."""
    x = pv[np.isfinite(pv)]
    if x.size == 0:
        return np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mu = float(np.mean(x))
    sigma = float(np.std(x) + 1e-6)
    z = (pv - mu) / sigma
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _normalize(pv: np.ndarray, method: str) -> np.ndarray:
    """Apply the normalization method selected in the training config."""
    if method == "zscore_galaxy_only":
        return _zscore_galaxy_only(pv)
    if method in ("none", None):
        return np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raise ValueError(f"Unknown norm_method: {method}")

def _pad_to(pv: np.ndarray, lab: np.ndarray, ph: int, pw: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep patch shape fixed when a cut reaches the cube edge."""
    v, s = pv.shape
    dv = max(0, ph - v)
    ds = max(0, pw - s)
    if dv == 0 and ds == 0:
        return pv, lab
    finite = pv[np.isfinite(pv)]
    pad_value = float(np.nanmedian(finite)) if finite.size else 0.0
    pv = np.pad(
        pv,
        ((dv // 2, dv - dv // 2), (ds // 2, ds - ds // 2)),
        mode="constant",
        constant_values=pad_value,
    )
    lab = np.pad(
        lab,
        ((dv // 2, dv - dv // 2), (ds // 2, ds - ds // 2)),
        mode="constant",
        constant_values=0,
    )
    return pv, lab


def _choose_patch(
    v: int, s: int,
    pv: np.ndarray, lab: np.ndarray,
    pos_frac: float, ph: int, pw: int,
    rng: random.Random
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose one training crop, optionally centering on a labeled shell pixel."""
    want_pos = rng.random() < pos_frac
    if want_pos and lab.any():
        ys, xs = np.where(lab > 0)
        k = rng.randrange(len(ys))
        cy, cx = int(ys[k]), int(xs[k])
        y0 = max(0, min(cy - ph // 2, v - ph))
        x0 = max(0, min(cx - pw // 2, s - pw))
    else:
        y0 = rng.randrange(0, max(1, v - ph + 1))
        x0 = rng.randrange(0, max(1, s - pw + 1))
    y1, x1 = y0 + ph, x0 + pw
    return pv[y0:y1, x0:x1], lab[y0:y1, x0:x1]


def _gen_samples(cfg: Dict, split: str, seed: int = 1337):
    """Yield PV and label tensors for TensorFlow's generator dataset."""
    rng = random.Random(seed)
    root = Path(cfg["output_root"])
    man_path = root / "splits" / f"{split}_manifest.txt"
    files = _read_manifest(man_path)
    if split == "train":
        rng.shuffle(files)

    norm = cfg["train"]["norm_method"]
    pos_frac = float(cfg["train"]["pos_fraction"])
    ph = int(cfg["train"]["patch_vel"])
    pw = int(cfg["train"]["patch_pos"])
    fixed_samples = cfg["train"].get("samples_per_pv")
    strict_shape = bool(cfg.get("standardized_pv", {}).get("enabled")) and bool(
        cfg["train"].get("strict_fixed_shape", True)
    )

    for fname in files:
        if fname.endswith("_posxy.npy"):
            continue
        try:
            pv, lab = _load_pair(root, fname)
        except FileNotFoundError as e:
            print(f"[dataset] warning: {e} ; skipping")
            continue

        pv = _normalize(pv, norm)
        pv, lab = _pad_to(pv, lab, ph, pw)

        v, s = pv.shape
        if strict_shape and (v, s) != (ph, pw):
            raise ValueError(f"{fname} has shape {(v, s)} after padding; expected fixed standardized shape {(ph, pw)}")
        # samples_per_pv controls whether a run sees one crop or several from each PV.
        if fixed_samples is not None:
            n_per = max(1, int(fixed_samples))
        else:
            n_per = max(4, (v * s) // (ph * pw))
        for _ in range(n_per):
            x, y = _choose_patch(v, s, pv, lab, pos_frac, ph, pw, rng)
            x = x[..., np.newaxis].astype(np.float32)
            y = y[..., np.newaxis].astype(np.float32)
            yield x, y

def build_dataset(
    cfg_path: str,
    split: str,
    batch_size: int,
    seed: int = 1337,
    repeat: bool = False
) -> tf.data.Dataset:
    """
    Build the TensorFlow dataset used by training and evaluation.
    Training can repeat and shuffle samples.
    Validation and test splits stay deterministic.
    """
    cfg = _resolve_cfg(cfg_path)
    ph = int(cfg["train"]["patch_vel"])
    pw = int(cfg["train"]["patch_pos"])

    elemspec = (
        tf.TensorSpec(shape=(ph, pw, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(ph, pw, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: _gen_samples(cfg, split=split, seed=seed),
        output_signature=elemspec
    )

    if split == "train":
        if repeat:
            ds = ds.repeat()
        ds = ds.shuffle(buffer_size=max(batch_size * 8, 64), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def _selftest(cfg_path: str):
    """Print a quick dataset sanity check for local training setup."""
    cfg = _resolve_cfg(cfg_path)
    bs = max(1, int(cfg["optim"]["batch_size"]))
    ds = build_dataset(cfg_path, split="train", batch_size=bs, seed=42, repeat=False)
    it = iter(ds)
    x, y = next(it)
    print("[dataset.selftest] batch shapes:", x.shape, y.shape)
    print("[dataset.selftest] x mean/std:",
          tf.math.reduce_mean(x).numpy(),
          tf.math.reduce_std(x).numpy())
    print("[dataset.selftest] y unique:",
          tf.unique(tf.reshape(tf.cast(y > 0.5, tf.int32), [-1]))[0].numpy())
    print("[dataset.selftest] OK")


def estimate_num_patches(cfg_path: str, split: str) -> int:
    """Deterministically estimate how many patches the generator will yield for a split."""
    cfg = _resolve_cfg(cfg_path)
    root = Path(cfg["output_root"])
    ph = int(cfg["train"]["patch_vel"])
    pw = int(cfg["train"]["patch_pos"])
    fixed_samples = cfg["train"].get("samples_per_pv")
    files = _read_manifest(root / "splits" / f"{split}_manifest.txt")

    total = 0
    for fname in files:
        if fname.endswith("_posxy.npy"):  # skip meta sidecars
            continue
        pv_path = root / "pv" / fname
        if not pv_path.exists():
            continue
        v, s = np.load(pv_path, mmap_mode="r").shape  # no RAM spike
        # account for padding to at least (ph, pw)
        v = max(v, ph); s = max(s, pw)
        if fixed_samples is not None:
            n_per = max(1, int(fixed_samples))
        else:
            n_per = max(4, (v * s) // (ph * pw))
        total += n_per
    return total


def estimate_steps(cfg_path: str, split: str, batch_size: int) -> int:
    """Convert the deterministic patch estimate into batched training steps."""
    n = estimate_num_patches(cfg_path, split)
    return max(1, int(ceil(n / max(1, batch_size))))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.config)
    else:
        print("Use --selftest to run a quick dataset check.")
