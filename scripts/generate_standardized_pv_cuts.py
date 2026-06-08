#!/usr/bin/env python3
"""Generate or dry-run fixed-kpc, local-velocity-centered PV cuts.

This is a preparation utility for the next training/inference design. It does
not launch model training and does not modify existing manifests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hishells_pv.pv.standardized_cuts import (
    StandardCutSpec,
    deployment_grid_specs,
    moment1_velocity_map,
    physical_velocity_axis_kms,
    sample_standardized_pv,
)
from hishells_pv.utils.config import resolve_config
from hishells_pv.utils.wcs_tools import open_cube, pixel_scales_arcsec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["deployment-grid", "single-center"], default="deployment-grid")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--spatial-window-kpc", type=float, default=5.0)
    ap.add_argument("--velocity-window-kms", type=float, default=200.0)
    ap.add_argument("--angles-deg", nargs="*", type=float, default=[0.0, 45.0, 90.0, 135.0])
    ap.add_argument("--stride-pix", type=float, default=32.0)
    ap.add_argument("--max-cuts", type=int, default=100)
    ap.add_argument("--center-pix", nargs=2, type=float, default=None)
    ap.add_argument("--angle-deg", type=float, default=0.0)
    args = ap.parse_args()

    cfg = resolve_config(args.config, write_resolved=False)
    cube, hdr, wcs, _ = open_cube(cfg["cube_path"])
    velocity_kms, velocity_meta = physical_velocity_axis_kms(hdr)
    m0 = np.nansum(np.clip(np.nan_to_num(cube), 0.0, None), axis=0)
    m1 = moment1_velocity_map(cube, velocity_kms)
    ax, ay = pixel_scales_arcsec(wcs)
    pix_scale = 0.5 * (ax + ay)
    galaxy = str(cfg.get("galaxy", {}).get("name") or Path(args.config).stem)
    distance = float(cfg["galaxy"]["distance_mpc"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single-center":
        if args.center_pix is None:
            raise SystemExit("--center-pix X Y is required for single-center mode")
        specs = [
            StandardCutSpec(
                galaxy=galaxy,
                center_pix=(float(args.center_pix[0]), float(args.center_pix[1])),
                angle_deg=float(args.angle_deg),
                spatial_window_kpc=args.spatial_window_kpc,
                velocity_window_kms=args.velocity_window_kms,
                source="single_center",
            )
        ]
    else:
        specs = list(
            deployment_grid_specs(
                galaxy,
                m0,
                angles_deg=args.angles_deg,
                stride_pix=args.stride_pix,
                max_specs=args.max_cuts,
            )
        )
        specs = [
            StandardCutSpec(
                galaxy=s.galaxy,
                center_pix=s.center_pix,
                angle_deg=s.angle_deg,
                spatial_window_kpc=args.spatial_window_kpc,
                velocity_window_kms=args.velocity_window_kms,
                source=s.source,
            )
            for s in specs
        ]

    written = []
    for i, spec in enumerate(specs):
        stem = f"std_{i:05d}"
        pv, meta = sample_standardized_pv(
            cube,
            hdr,
            m1,
            spec,
            distance_mpc=distance,
            pixel_scale_arcsec=pix_scale,
            fallback_velocity_kms=cfg.get("galaxy", {}).get("vsys_kms"),
        )
        meta["dry_run"] = bool(args.dry_run)
        if not args.dry_run:
            np.save(out_dir / f"{stem}.npy", pv)
            (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        written.append({"stem": stem, **meta})

    manifest = {
        "config": str(Path(args.config).resolve()),
        "out_dir": str(out_dir.resolve()),
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "n_specs": len(specs),
        "velocity_axis_meta": velocity_meta,
        "cuts": written,
    }
    (out_dir / "standardized_cut_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[standardized] {'planned' if args.dry_run else 'wrote'} {len(written)} cuts -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
