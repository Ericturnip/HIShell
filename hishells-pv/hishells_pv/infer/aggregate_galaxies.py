"""Drive sky-plane aggregation across every galaxy in a standardized dataset.

The trained model and standardized dataset are galaxy-mixed, but aggregation is
inherently per-galaxy (it projects PV votes onto a single cube's image plane via
the ``*_posxy.npy`` sidecars). This module reproduces, as a first-class CLI step,
what the older ``scripts/run_aggregate_test.py`` and ``scripts/aggregate_stress.py``
did by hand: for each galaxy in the requested splits it ensures a per-galaxy
aggregation input dir + config exist (reusing generate-emitted ones when present),
then calls :func:`hishells_pv.infer.aggregate.aggregate` with a galaxy-specific
split label so outputs do not collide.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from hishells_pv.datagen.emit_agg import build_galaxy_agg_input, split_label, write_agg_config
from hishells_pv.infer.aggregate import aggregate
from hishells_pv.utils.config import resolve_config

DEFAULT_SPLITS = ("test", "stress")


def _galaxy_id(filename: str) -> str:
    return filename.split("__", 1)[0]


def _read_split_manifest(output_root: Path, split: str) -> list[str]:
    path = output_root / "splits" / f"{split}_manifest.txt"
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.strip().endswith("_posxy.npy")]


def _ensure_galaxy_config(
    output_root: Path,
    galaxy_id: str,
    filenames: list[str],
    split: str,
    *,
    configs_dir: Path,
    patch_vel: int,
    patch_pos: int,
) -> tuple[Path, str]:
    """Return (config_path, label), building inputs/config if not already present."""
    label = split_label(galaxy_id, split)
    agg_root = output_root / "agg" / galaxy_id
    cfg_path = agg_root / f"{galaxy_id}_agg_config.yaml"
    manifest = agg_root / "splits" / f"{label}_manifest.txt"

    if cfg_path.exists() and manifest.exists():
        return cfg_path, label

    src_yaml = configs_dir / f"{galaxy_id}.yaml"
    if not src_yaml.exists():
        raise FileNotFoundError(
            f"No agg config for {galaxy_id} and no source config at {src_yaml}; "
            f"re-run `generate` (it emits agg configs) or pass --configs-dir."
        )
    source_cfg = resolve_config(str(src_yaml), write_resolved=False)
    agg_root = build_galaxy_agg_input(output_root, output_root / "pv", galaxy_id, filenames, label)
    cfg_path = write_agg_config(
        agg_root, galaxy_id, source_cfg, patch_vel=patch_vel, patch_pos=patch_pos
    )
    return cfg_path, label


def aggregate_all(
    output_root: str | Path,
    run_dir: str | Path,
    *,
    splits: Sequence[str] = DEFAULT_SPLITS,
    thresh: float = 0.4,
    device_name: str = "auto",
    write_regions: bool = True,
    configs_dir: str | Path | None = None,
    patch_vel: int = 96,
    patch_pos: int = 256,
) -> list[Path]:
    """Aggregate every galaxy present in ``output_root`` for the given splits.

    Returns the list of per-galaxy aggregate output dirs that were produced.
    """
    output_root = Path(output_root)
    run_dir = Path(run_dir)
    configs_dir = Path(configs_dir) if configs_dir is not None else output_root.parent / "configs"

    produced: list[Path] = []
    for split in splits:
        names = _read_split_manifest(output_root, split)
        if not names:
            print(f"[aggregate-all] split={split}: no manifest entries, skipping")
            continue
        files_by_galaxy: dict[str, list[str]] = {}
        for name in names:
            files_by_galaxy.setdefault(_galaxy_id(name), []).append(name)

        for galaxy_id, filenames in sorted(files_by_galaxy.items()):
            print(f"\n===== aggregate {galaxy_id} ({split}) =====", flush=True)
            cfg_path, label = _ensure_galaxy_config(
                output_root,
                galaxy_id,
                filenames,
                split,
                configs_dir=configs_dir,
                patch_vel=patch_vel,
                patch_pos=patch_pos,
            )
            aggregate(
                cfg_path=str(cfg_path),
                run_dir=str(run_dir),
                split=label,
                thresh=float(thresh),
                device_name=device_name,
                write_regions=write_regions,
            )
            produced.append(run_dir / f"aggregate_{label}")
    return produced
