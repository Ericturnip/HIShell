#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.diagnostic_utils import component_features, load_json, predict_full_pv, read_manifest, source_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_data/combined_train_overnight.yaml")
    ap.add_argument("--model", default="runs/pv_unet_overnight_20260520_015732/best_model.keras")
    ap.add_argument("--run-dir", default="runs/pv_unet_overnight_20260520_015732")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.05, 0.075])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-candidates", type=int, default=500)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    combined_root = Path(cfg["output_root"])
    data_root = Path(args.data_root) if args.data_root else combined_root.parent
    patch_vel = int(cfg["train"]["patch_vel"])
    patch_pos = int(cfg["train"]["patch_pos"])

    from tensorflow import keras

    model = keras.models.load_model(args.model, compile=False)
    candidates = []

    for name in read_manifest(combined_root, args.split):
        paths = source_paths(combined_root, data_root, name)
        if not paths["pv"].exists() or not paths["label"].exists():
            continue
        label = np.load(paths["label"])
        if bool(label.any()):
            continue
        meta = load_json(paths["meta"])
        pv = np.load(paths["pv"])
        prob = predict_full_pv(model, pv, patch_vel=patch_vel, patch_pos=patch_pos, batch_size=args.batch_size)
        max_prob = float(np.nanmax(prob)) if prob.size else 0.0
        triggered = [float(t) for t in args.thresholds if max_prob >= float(t)]
        if not triggered:
            continue
        threshold = min(triggered)
        mask = prob >= threshold
        comps = component_features(prob, mask, threshold=threshold, meta=meta, min_area=1)
        largest = max(comps, key=lambda r: r["area_pix"], default={})
        vals = prob[mask]
        candidates.append(
            {
                "source_galaxy": name.split("__", 1)[0] if "__" in name else meta.get("galaxy"),
                "patch_path": str(paths["pv"]),
                "patch_id": Path(name).stem,
                "thresholds_triggered": triggered,
                "max_predicted_probability": max_prob,
                "mean_predicted_probability_inside_predicted_mask": float(np.nanmean(vals)) if vals.size else 0.0,
                "total_predicted_probability_mass": float(np.nansum(prob * mask)),
                "predicted_mask_area": int(mask.sum()),
                "connected_component_count": len(comps),
                "largest_connected_component_area": largest.get("area_pix", 0),
                "pv_cut_center": meta.get("center_pix"),
                "pv_cut_angle": meta.get("theta_rad", meta.get("pa_eff_deg", meta.get("orientation"))),
                "local_velocity_center": meta.get("local_velocity_kms"),
                "velocity_window": meta.get("velocity_window_kms", meta.get("vel_kms")),
                "source_type": meta.get("source", meta.get("type")),
                "metadata_type": meta.get("type"),
            }
        )
        candidates.sort(key=lambda r: (r["max_predicted_probability"], r["predicted_mask_area"]), reverse=True)
        if len(candidates) > args.max_candidates:
            candidates = candidates[: args.max_candidates]

    out = {
        "config": str(Path(args.config).resolve()),
        "model": str(Path(args.model).resolve()),
        "split": args.split,
        "thresholds": args.thresholds,
        "notes": [
            "This is a review list only. It is not added to any training YAML or manifest.",
            "Held-out test false positives should not be injected into future training unless a new clean holdout is defined.",
        ],
        "candidates": candidates,
    }
    out_path = Path(args.run_dir) / "hard_negative_candidates.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[hard-negatives] wrote {len(candidates)} candidates -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
