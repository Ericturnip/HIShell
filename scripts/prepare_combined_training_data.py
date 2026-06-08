#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


DEFAULT_VAL_GALAXIES = ("ho_i", "ngc_2366", "ngc_4449", "ngc_4736")
DEFAULT_TEST_GALAXIES = ("ddo53", "ho_ii", "ngc_3184", "ngc_7793")


def _is_base_npy(path: Path) -> bool:
    return path.suffix == ".npy" and not path.name.endswith("_posxy.npy")


def _safe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _split_for(galaxy: str, val: set[str], test: set[str]) -> str:
    if galaxy in val:
        return "val"
    if galaxy in test:
        return "test"
    return "train"


def _read_source(label_json: Path) -> str:
    if not label_json.exists():
        return "unknown"
    try:
        return json.loads(label_json.read_text()).get("source", "unknown")
    except Exception:
        return "unknown"


def _write_manifest(names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in sorted(names)))


def build_combined(
    data_root: Path,
    out_root: Path,
    *,
    val_galaxies: set[str],
    test_galaxies: set[str],
    min_finite_fraction: float,
    force: bool,
) -> dict:
    data_root = data_root.resolve()
    out_root = out_root.resolve()
    if out_root.exists() and force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for sub in ("pv", "labels", "label_types", "splits"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    manifests: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_counts: dict[str, Counter[str]] = {k: Counter() for k in manifests}
    source_counts: dict[str, Counter[str]] = {k: Counter() for k in manifests}
    galaxy_counts: dict[str, Counter[str]] = {k: Counter() for k in manifests}
    skipped_counts: Counter[str] = Counter()
    label_pixels: dict[str, int] = defaultdict(int)
    total_pixels: dict[str, int] = defaultdict(int)

    galaxy_dirs = sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and p.name != out_root.name and (p / "pv").exists() and (p / "labels").exists()
    )

    for gal_dir in galaxy_dirs:
        split = _split_for(gal_dir.name, val_galaxies, test_galaxies)
        for pv_path in sorted(p for p in (gal_dir / "pv").glob("*.npy") if _is_base_npy(p)):
            lab_path = gal_dir / "labels" / pv_path.name
            if not lab_path.exists():
                continue
            pv = np.load(pv_path, mmap_mode="r")
            finite_fraction = float(np.isfinite(pv).mean())
            if finite_fraction < min_finite_fraction:
                skipped_counts["low_finite_fraction"] += 1
                skipped_counts[f"low_finite_fraction:{gal_dir.name}"] += 1
                continue
            out_name = f"{gal_dir.name}__{pv_path.name}"
            _safe_link(pv_path, out_root / "pv" / out_name)
            _safe_link(lab_path, out_root / "labels" / out_name)

            type_path = gal_dir / "label_types" / pv_path.name
            if type_path.exists():
                _safe_link(type_path, out_root / "label_types" / out_name)

            lab = np.load(lab_path, mmap_mode="r")
            is_pos = bool(np.asarray(lab).any())
            manifests[split].append(out_name)
            split_counts[split]["total"] += 1
            split_counts[split]["positive" if is_pos else "negative"] += 1
            galaxy_counts[split][gal_dir.name] += 1
            source = _read_source(lab_path.with_suffix(".json"))
            source_counts[split][source] += 1
            label_pixels[split] += int(np.count_nonzero(lab))
            total_pixels[split] += int(lab.size)

    for split, names in manifests.items():
        _write_manifest(names, out_root / "splits" / f"{split}_manifest.txt")

    summary = {
        "data_root": str(data_root),
        "output_root": str(out_root),
        "split_policy": "galaxy-held-out",
        "val_galaxies": sorted(val_galaxies),
        "test_galaxies": sorted(test_galaxies),
        "min_finite_fraction": min_finite_fraction,
        "skipped": dict(skipped_counts),
        "splits": {
            split: {
                "counts": dict(split_counts[split]),
                "galaxies": dict(galaxy_counts[split]),
                "sources": dict(source_counts[split]),
                "label_pixel_fraction": (
                    label_pixels[split] / total_pixels[split] if total_pixels[split] else 0.0
                ),
            }
            for split in ("train", "val", "test")
        },
    }
    (out_root / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_training_config(out_root: Path, config_path: Path, *, epochs: int, batch_size: int) -> None:
    cfg = {
        "output_root": str(out_root.resolve()),
        "train": {
            "pos_fraction": 0.5,
            "patch_pos": 256,
            "patch_vel": 96,
            "norm_method": "zscore_galaxy_only",
            "samples_per_pv": 1,
            "max_steps_per_epoch": 400,
            "max_validation_steps": 100,
        },
        "model": {
            "base_filters": 24,
            "depth": 3,
            "dilation_rate": 1,
            "dropout": 0.10,
        },
        "optim": {
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": batch_size,
            "epochs": epochs,
            "loss": "binary_crossentropy",
        },
        "notes": {
            "model_task": "PV mask segmentation with a 2D CNN/U-Net; slice-level classification can be derived from masks later.",
            "split_policy": "Whole galaxies are held out so repeated cuts through the same shells do not leak across train/val/test.",
            "sampling": "The dataset generator samples positive-centered and random PV patches with pos_fraction=0.5.",
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="training_data")
    ap.add_argument("--out-root", default="training_data/combined")
    ap.add_argument("--config", default="training_data/combined_train.yaml")
    ap.add_argument("--val-galaxies", nargs="*", default=list(DEFAULT_VAL_GALAXIES))
    ap.add_argument("--test-galaxies", nargs="*", default=list(DEFAULT_TEST_GALAXIES))
    ap.add_argument("--min-finite-fraction", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    summary = build_combined(
        data_root,
        out_root,
        val_galaxies=set(args.val_galaxies),
        test_galaxies=set(args.test_galaxies),
        min_finite_fraction=args.min_finite_fraction,
        force=args.force,
    )
    write_training_config(out_root, Path(args.config), epochs=args.epochs, batch_size=args.batch_size)

    print(f"[combined] wrote symlinked dataset -> {out_root.resolve()}")
    print(f"[combined] wrote config -> {Path(args.config).resolve()}")
    for split, info in summary["splits"].items():
        counts = info["counts"]
        print(
            f"[combined] {split}: total={counts.get('total', 0)} "
            f"positive={counts.get('positive', 0)} negative={counts.get('negative', 0)} "
            f"galaxies={len(info['galaxies'])}"
        )


if __name__ == "__main__":
    main()
