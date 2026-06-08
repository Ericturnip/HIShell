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

from hishells_pv.eval.diagnostic_utils import component_features, load_json, predict_full_pv, read_manifest, source_paths, write_csv


def threshold_tag(t: float) -> str:
    return str(t).replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_data/combined_train_overnight.yaml")
    ap.add_argument("--model", default="runs/pv_unet_overnight_20260520_015732/best_model.keras")
    ap.add_argument("--run-dir", default="runs/pv_unet_overnight_20260520_015732")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.05, 0.075])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-area-pix", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    combined_root = Path(cfg["output_root"])
    data_root = Path(args.data_root) if args.data_root else combined_root.parent
    patch_vel = int(cfg["train"]["patch_vel"])
    patch_pos = int(cfg["train"]["patch_pos"])

    from tensorflow import keras

    model = keras.models.load_model(args.model, compile=False)
    rows_by_threshold = {float(t): [] for t in args.thresholds}

    for i, name in enumerate(read_manifest(combined_root, args.split), start=1):
        paths = source_paths(combined_root, data_root, name)
        if not paths["pv"].exists():
            continue
        meta = load_json(paths["meta"])
        prob = predict_full_pv(
            model,
            np.load(paths["pv"]),
            patch_vel=patch_vel,
            patch_pos=patch_pos,
            batch_size=args.batch_size,
        )
        galaxy = name.split("__", 1)[0] if "__" in name else meta.get("galaxy")
        for threshold in rows_by_threshold:
            comps = component_features(prob, prob >= threshold, threshold=threshold, meta=meta, min_area=args.min_area_pix)
            for comp in comps:
                comp.update(
                    {
                        "galaxy": galaxy,
                        "patch_id": Path(name).stem,
                        "patch_name": name,
                        "source_type": meta.get("type"),
                        "source": meta.get("source"),
                        "shell_id": meta.get("target_shell_id"),
                        "orientation": meta.get("orientation"),
                        "offset_fraction": meta.get("offset_fraction"),
                        "pv_cut_center": meta.get("center_pix"),
                        "pv_cut_angle": meta.get("theta_rad", meta.get("pa_eff_deg", meta.get("orientation"))),
                    }
                )
                # Soft physical warnings. Existing metadata lacks beam/kpc scales, so these are flags, not cuts.
                vel_extent = comp.get("velocity_extent_kms")
                comp["velocity_extent_plausibility_warning"] = bool(vel_extent is not None and vel_extent > 120.0)
                comp["edge_artifact_warning"] = bool(comp.get("edge_touching"))
                rows_by_threshold[threshold].append(comp)
        if i % 100 == 0:
            print(f"[components] processed {i} patches")

    run_dir = Path(args.run_dir)
    for threshold, rows in rows_by_threshold.items():
        tag = threshold_tag(threshold)
        csv_path = run_dir / f"candidate_components_threshold_{tag}.csv"
        json_path = run_dir / f"candidate_components_threshold_{tag}.json"
        write_csv(rows, csv_path)
        json_path.write_text(json.dumps({"threshold": threshold, "components": rows}, indent=2))
        print(f"[components] threshold={threshold} rows={len(rows)} -> {csv_path.resolve()}")


if __name__ == "__main__":
    main()
