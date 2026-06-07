#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.diagnostic_utils import (
    load_json,
    patch_prediction,
    pixel_metrics,
    predict_full_pv,
    read_manifest,
    source_paths,
    write_csv,
)


THRESHOLDS = (0.05, 0.075, 0.1)


def _load_cfg(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def _category(meta: dict, has_label: bool) -> str:
    typ = meta.get("type")
    if typ == "catalog_shell":
        off = float(meta.get("offset_fraction", 999.0))
        orient = str(meta.get("orientation", "unknown"))
        if abs(off) < 1e-9:
            return f"centered_positive__{orient}" if has_label else f"centered_unlabeled__{orient}"
        return f"spatial_offset_{abs(off):.1f}radius__{orient}" if has_label else f"spatial_offset_unlabeled_{abs(off):.1f}radius__{orient}"
    if typ == "grid":
        return "fine_grid_deployment_like_positive" if has_label else "fine_grid_deployment_like_negative"
    if typ == "background_negative":
        return "random_background_positive" if has_label else "random_background_negative"
    return "unknown_positive" if has_label else "unknown_negative"


def _empty_stats() -> dict:
    return {
        "n_patches": 0,
        "positive_patches": 0,
        "negative_patches": 0,
        "pixel": {str(t): {"tp": 0, "fp": 0, "fn": 0} for t in THRESHOLDS},
        "patch": {str(t): {"tp": 0, "fp": 0, "fn": 0} for t in THRESHOLDS},
        "missed_positive_patches": {str(t): [] for t in THRESHOLDS},
        "false_positive_patches": {str(t): [] for t in THRESHOLDS},
        "max_probabilities": [],
    }


def _add_example(stats: dict, threshold: float, key: str, record: dict) -> None:
    bucket = stats[key][str(threshold)]
    if len(bucket) < 100:
        bucket.append(record)


def _update(stats: dict, *, name: str, galaxy: str, category: str, prob: np.ndarray, label: np.ndarray, meta: dict) -> None:
    y_patch = bool(np.asarray(label > 0.5).any())
    stats["n_patches"] += 1
    stats["positive_patches"] += int(y_patch)
    stats["negative_patches"] += int(not y_patch)
    max_prob = float(np.nanmax(prob)) if prob.size else 0.0
    stats["max_probabilities"].append(max_prob)
    rec = {
        "name": name,
        "galaxy": galaxy,
        "category": category,
        "max_probability": max_prob,
        "type": meta.get("type"),
        "orientation": meta.get("orientation"),
        "offset_fraction": meta.get("offset_fraction"),
        "target_shell_id": meta.get("target_shell_id"),
    }
    for t in THRESHOLDS:
        pm = pixel_metrics(prob, label, t)
        pstats = stats["pixel"][str(t)]
        for key in ("tp", "fp", "fn"):
            pstats[key] += int(pm[key])

        pred_patch = patch_prediction(prob, t)
        pst = stats["patch"][str(t)]
        if pred_patch and y_patch:
            pst["tp"] += 1
        elif pred_patch and not y_patch:
            pst["fp"] += 1
            _add_example(stats, t, "false_positive_patches", rec)
        elif (not pred_patch) and y_patch:
            pst["fn"] += 1
            _add_example(stats, t, "missed_positive_patches", rec)


def _finalize(stats: dict) -> dict:
    out = dict(stats)
    for section in ("pixel", "patch"):
        for t, values in out[section].items():
            tp, fp, fn = values["tp"], values["fp"], values["fn"]
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            values["precision"] = precision
            values["recall"] = recall
            values["f1"] = 2 * precision * recall / max(1e-12, precision + recall)
            if section == "patch":
                values["false_positive_patch_count"] = fp
                values["missed_positive_patch_count"] = fn
    maxp = np.asarray(out.pop("max_probabilities", []), dtype=float)
    out["max_probability_summary"] = {
        "mean": float(np.nanmean(maxp)) if maxp.size else None,
        "median": float(np.nanmedian(maxp)) if maxp.size else None,
        "p90": float(np.nanpercentile(maxp, 90)) if maxp.size else None,
    }
    return out


def _velocity_shift_channels(meta: dict, shift_kms: float) -> int:
    vel = meta.get("vel_kms") or []
    if len(vel) < 3:
        return 0
    step = abs(float(vel[2]))
    if step > 100.0:
        step /= 1000.0
    return int(round(float(shift_kms) / max(step, 1e-6)))


def evaluate(args: argparse.Namespace) -> dict:
    cfg = _load_cfg(Path(args.config))
    combined_root = Path(cfg["output_root"])
    data_root = Path(args.data_root) if args.data_root else combined_root.parent
    patch_vel = int(cfg["train"]["patch_vel"])
    patch_pos = int(cfg["train"]["patch_pos"])

    from tensorflow import keras

    model = keras.models.load_model(args.model, compile=False)
    categories: dict[str, dict] = defaultdict(_empty_stats)

    names = read_manifest(combined_root, args.split)
    per_category_seen: dict[str, int] = defaultdict(int)
    for idx, name in enumerate(names, start=1):
        paths = source_paths(combined_root, data_root, name)
        if not paths["pv"].exists() or not paths["label"].exists():
            continue
        meta = load_json(paths["meta"])
        galaxy = name.split("__", 1)[0] if "__" in name else str(meta.get("galaxy", "unknown"))
        pv = np.load(paths["pv"])
        label = np.load(paths["label"]).astype(np.uint8)
        has_label = bool(label.any())
        cat = _category(meta, has_label)
        if args.max_per_category and per_category_seen[cat] >= args.max_per_category:
            continue
        per_category_seen[cat] += 1

        prob = predict_full_pv(
            model,
            pv,
            patch_vel=patch_vel,
            patch_pos=patch_pos,
            batch_size=args.batch_size,
            normalize=True,
        )
        _update(categories[cat], name=name, galaxy=galaxy, category=cat, prob=prob, label=label, meta=meta)

        if args.spectral_offsets and meta.get("type") == "catalog_shell" and has_label and abs(float(meta.get("offset_fraction", 999))) < 1e-9:
            for shift in args.spectral_offsets:
                ch = _velocity_shift_channels(meta, float(shift))
                if ch == 0:
                    continue
                shifted = np.roll(pv, shift=ch, axis=0)
                shifted_prob = predict_full_pv(
                    model,
                    shifted,
                    patch_vel=patch_vel,
                    patch_pos=patch_pos,
                    batch_size=args.batch_size,
                    normalize=True,
                )
                scat = f"spectral_offset_{shift:+g}kms_centered_positive"
                _update(categories[scat], name=name, galaxy=galaxy, category=scat, prob=shifted_prob, label=label, meta=meta)

        if idx % 100 == 0:
            print(f"[robustness] processed {idx}/{len(names)} manifest rows")

    return {
        "config": str(Path(args.config).resolve()),
        "model": str(Path(args.model).resolve()),
        "split": args.split,
        "thresholds": list(THRESHOLDS),
        "notes": [
            "Categories are derived from existing PV metadata; no new training data was created.",
            "Spatial offsets use the existing catalog offset_fraction values, which are fractions of shell projected radius, not beam widths.",
            "Spectral offsets are synthetic rolls of centered positive PV inputs; labels remain fixed for sensitivity testing.",
        ],
        "categories": {cat: _finalize(stats) for cat, stats in sorted(categories.items())},
    }


def flatten_csv(result: dict) -> list[dict]:
    rows = []
    for cat, stats in result["categories"].items():
        for level in ("pixel", "patch"):
            for threshold, values in stats[level].items():
                rows.append(
                    {
                        "category": cat,
                        "metric_level": level,
                        "threshold": threshold,
                        "n_patches": stats["n_patches"],
                        "positive_patches": stats["positive_patches"],
                        "negative_patches": stats["negative_patches"],
                        **{k: v for k, v in values.items() if not isinstance(v, list)},
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_data/combined_train_overnight.yaml")
    ap.add_argument("--model", default="runs/pv_unet_overnight_20260520_015732/best_model.keras")
    ap.add_argument("--run-dir", default="runs/pv_unet_overnight_20260520_015732")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-per-category", type=int, default=0)
    ap.add_argument("--spectral-offsets", nargs="*", type=float, default=[-30.0, -15.0, 15.0, 30.0])
    args = ap.parse_args()

    result = evaluate(args)
    run_dir = Path(args.run_dir)
    out_json = run_dir / "cut_offset_robustness_eval.json"
    out_csv = run_dir / "cut_offset_robustness_eval.csv"
    out_json.write_text(json.dumps(result, indent=2))
    write_csv(flatten_csv(result), out_csv)
    print(f"[robustness] wrote {out_json.resolve()}")
    print(f"[robustness] wrote {out_csv.resolve()}")


if __name__ == "__main__":
    main()
