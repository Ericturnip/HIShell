#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.diagnostic_utils import load_json, read_manifest, source_paths
from src.utils.io import load_yaml


def audit(config: Path, run_dir: Path, data_root: Path | None = None) -> dict:
    cfg = load_yaml(config)
    combined_root = Path(cfg["output_root"])
    data_root = data_root or combined_root.parent

    split_summary = load_json(combined_root / "split_summary.json")
    result: dict = {
        "config": str(config.resolve()),
        "run_dir": str(run_dir.resolve()),
        "combined_root": str(combined_root.resolve()),
        "split_summary": split_summary,
        "splits": {},
        "metadata_coverage": {
            "has_galaxy": True,
            "has_shell_id": True,
            "has_radec": False,
            "has_pixel_position": True,
            "has_pv_cut_center": True,
            "has_pv_cut_angle": "partial",
            "has_spatial_offset": True,
            "has_velocity_center": "catalog labels only; no local moment1 center",
            "has_velocity_window": "native full spectral axis only",
            "has_source_cube": "config level",
            "has_moment_map_source": False,
        },
        "risks": [],
    }

    for split in ("train", "val", "test"):
        names = read_manifest(combined_root, split)
        galaxies = Counter()
        types = Counter()
        sources = Counter()
        orientations = Counter()
        offsets = Counter()
        positive_by_type = Counter()
        negative_by_type = Counter()
        missing_meta = 0
        vel_ranges = []
        shapes = Counter()
        centered_catalog_shell = 0
        catalog_shell_total = 0

        for name in names:
            galaxy = name.split("__", 1)[0] if "__" in name else "unknown"
            galaxies[galaxy] += 1
            paths = source_paths(combined_root, data_root, name)
            meta = load_json(paths["meta"])
            label_meta = load_json(paths["label_json"])
            if not meta:
                missing_meta += 1
            typ = str(meta.get("type", "missing"))
            src = str(meta.get("source", "none"))
            types[typ] += 1
            sources[src] += 1
            if meta.get("orientation") is not None:
                orientations[str(meta["orientation"])] += 1
            if meta.get("offset_fraction") is not None:
                offsets[str(meta["offset_fraction"])] += 1
            if typ == "catalog_shell":
                catalog_shell_total += 1
                if abs(float(meta.get("offset_fraction", 999))) < 1e-9:
                    centered_catalog_shell += 1
            objects = label_meta.get("objects") or []
            if objects:
                positive_by_type[typ] += 1
            else:
                negative_by_type[typ] += 1
            if meta.get("vel_kms"):
                vel_ranges.append(meta["vel_kms"])
            if meta.get("nv") and meta.get("npos"):
                shapes[f"{meta['nv']}x{meta['npos']}"] += 1

        result["splits"][split] = {
            "n": len(names),
            "galaxies": dict(galaxies),
            "pv_types": dict(types),
            "sources": dict(sources),
            "orientations": dict(orientations),
            "offset_fractions": dict(offsets),
            "positive_slices_by_type": dict(positive_by_type),
            "negative_slices_by_type": dict(negative_by_type),
            "missing_meta": missing_meta,
            "centered_catalog_shell_cuts": centered_catalog_shell,
            "catalog_shell_cuts": catalog_shell_total,
            "shape_counts_top20": dict(shapes.most_common(20)),
            "example_velocity_ranges": vel_ranges[:5],
        }

    test = result["splits"].get("test", {})
    test_types = test.get("pv_types", {})
    if test_types.get("catalog_shell", 0) > test_types.get("grid", 0):
        result["risks"].append(
            "Held-out test galaxies are disjoint from training galaxies, but the test manifest is dominated by catalog_shell cuts, including centered and near-centered Bagetakos cuts. Metrics can still be optimistic for blind deployment."
        )
    result["risks"].append(
        "Current PV arrays use the native/full spectral axis and metadata does not record local Moment-1 velocity centering."
    )
    result["risks"].append(
        "Current spatial windows are configured in arcsec/pixels/radius-scaled catalog cuts, not a fixed kpc input window."
    )

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_data/combined_train_overnight.yaml")
    ap.add_argument("--run-dir", default="runs/pv_unet_overnight_20260520_015732")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    result = audit(Path(args.config), run_dir, Path(args.data_root) if args.data_root else None)
    out = Path(args.out) if args.out else run_dir / "dataset_split_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[audit] wrote {out.resolve()}")


if __name__ == "__main__":
    main()
