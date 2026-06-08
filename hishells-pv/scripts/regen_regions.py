#!/usr/bin/env python3
"""Regenerate DS9/CARTA .reg files for already-aggregated galaxies.

The aggregation runs wrote vote maps + detections JSON, but `_write_regions`
previously failed on THINGS cubes whose WCS carries 3-4 axes (RA, Dec, velocity,
Stokes). That bug is fixed (it now reduces to the 2D celestial sub-WCS), so we
can rebuild the .reg files directly from the existing detections JSON + each
galaxy's cube header, without re-running the model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hishells_pv.infer.aggregate import _write_regions  # noqa: E402

RUN_DIR = ROOT / "runs" / "pv_unet_real"
DATA = ROOT / "data" / "THINGS"

# aggregate dir -> cube FITS for the matching galaxy.
JOBS = {
    "aggregate_stress": DATA / "NGC_3031_NA_CUBE_THINGS.FITS",
    "aggregate_ddo53_test": DATA / "DDO53_NA_CUBE_THINGS.FITS",
    "aggregate_ho_ii_test": DATA / "HO_II_NA_CUBE_THINGS.FITS",
    "aggregate_ngc_3184_test": DATA / "NGC_3184_NA_CUBE_THINGS.FITS",
    "aggregate_ngc_7793_test": DATA / "NGC_7793_NA_CUBE_THINGS.FITS",
}


def main() -> None:
    for agg_name, cube in JOBS.items():
        agg_dir = RUN_DIR / agg_name
        det_files = list(agg_dir.glob("detections_*.json"))
        if not det_files:
            print(f"[regen] skip {agg_name}: no detections JSON")
            continue
        det_path = det_files[0]
        data = json.loads(det_path.read_text())
        split = data["split"]
        dets = data["detections"]
        peaks = [(d["y_pix"], d["x_pix"], d["score"]) for d in dets]

        wcs = WCS(fits.getheader(str(cube)))
        out_reg = agg_dir / f"detections_{split}.reg"
        _write_regions(peaks, wcs, out_reg, color="cyan")
        print(f"[regen] {agg_name}: wrote {len(peaks)} regions -> {out_reg}")


if __name__ == "__main__":
    main()
