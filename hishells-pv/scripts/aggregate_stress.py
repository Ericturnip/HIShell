#!/usr/bin/env python3
"""Run sky-plane aggregation on the held-out NGC 3031 stress galaxy.

NGC 3031 is the single isolated stress galaxy, so (unlike the 4-galaxy test
split) it maps cleanly onto one cube's image plane. We point the aggregation at
the existing standardized dataset (its stress cuts + ``splits/stress_manifest.txt``
are already on disk) via a static per-galaxy config, then call ``aggregate()``
with ``split="stress"`` so outputs land in ``runs/pv_unet_real/aggregate_stress``.

This avoids editing the ``aggregate`` CLI, whose ``--split`` choices are limited
to train/val/test; the underlying ``aggregate()`` function does not restrict the
split label.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hishells_pv.infer.aggregate import aggregate  # noqa: E402

CONFIG = ROOT / "configs" / "pv_config_NGC3031.yaml"
RUN_DIR = ROOT / "runs" / "pv_unet_real"
SPLIT = "stress"
THRESH = 0.4  # calibrated on val (training_data/.../calib/threshold.txt)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    print(f"\n===== aggregate {SPLIT} (NGC 3031) =====", flush=True)
    aggregate(
        cfg_path=str(CONFIG),
        run_dir=str(RUN_DIR),
        split=SPLIT,
        thresh=THRESH,
        device_name="cpu",
        write_regions=True,
    )


if __name__ == "__main__":
    main()
