#!/usr/bin/env python3
"""Run sky-plane aggregation per test galaxy.

Thin wrapper retained for back-compat: the per-galaxy isolation + config
synthesis now lives in :func:`hishells_pv.infer.aggregate_galaxies.aggregate_all`,
which is also exposed as ``hishells-pv aggregate-all``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hishells_pv.infer.aggregate_galaxies import aggregate_all  # noqa: E402

STD = ROOT / "training_data" / "standardized_5kpc_200kms"
RUN_DIR = ROOT / "runs" / "pv_unet_real"
THRESH = 0.4


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    aggregate_all(
        output_root=STD,
        run_dir=RUN_DIR,
        splits=("test",),
        thresh=THRESH,
        device_name="cpu",
        write_regions=True,
        configs_dir=ROOT / "training_data" / "configs",
    )


if __name__ == "__main__":
    main()
