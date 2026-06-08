"""Build per-galaxy aggregation inputs and configs (torch-free).

Aggregation is inherently per-galaxy: it projects PV votes onto a single cube's
image plane via the ``*_posxy.npy`` sidecars. The standardized dataset, however,
is galaxy-mixed (all galaxies share one ``pv/`` dir and galaxy-held-out
``splits/<split>_manifest.txt`` files). This module isolates one galaxy's PV
cuts into ``<out_root>/agg/<galaxy>/`` and writes a config that carries the
``cube_path`` + ``galaxy`` block that :func:`hishells_pv.infer.aggregate.aggregate`
requires.

It is intentionally free of heavy ML imports (no torch) so it can run as part of
the data-generation half of the library.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hishells_pv.utils.io import save_yaml

__all__ = ["split_label", "build_galaxy_agg_input", "write_agg_config"]

DEFAULT_POST = {"min_component_area_pv": [6, 5]}
DEFAULT_AGGREGATE = {
    "vote_sigma_pix": 5,
    "nms_radius_pix": 12,
    "min_radius_beams": 2.5,
    "max_thickness_frac": 0.4,
}


def split_label(galaxy_id: str, split: str) -> str:
    """Per-galaxy split label, e.g. ``ngc_7793_test``.

    Used as both the manifest stem (``<label>_manifest.txt``) and the
    ``aggregate`` split argument so outputs land in ``aggregate_<label>`` and do
    not collide across galaxies sharing one run dir.
    """
    return f"{galaxy_id}_{split}"


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        # Filesystems without symlink support: fall back to a hard copy.
        import shutil

        shutil.copy2(src, dst)


def build_galaxy_agg_input(
    out_root: Path,
    dataset_pv_dir: Path,
    galaxy_id: str,
    filenames: list[str],
    label: str,
) -> Path:
    """Isolate one galaxy's PV cuts into ``out_root/agg/<galaxy>/``.

    Symlinks (or copies) each ``{stem}.npy``, ``{stem}_posxy.npy`` and
    ``{stem}.json`` from ``dataset_pv_dir`` and writes
    ``splits/<label>_manifest.txt``. Returns the per-galaxy agg root.
    """
    agg_root = Path(out_root) / "agg" / galaxy_id
    pv_out = agg_root / "pv"
    splits_out = agg_root / "splits"
    pv_out.mkdir(parents=True, exist_ok=True)
    splits_out.mkdir(parents=True, exist_ok=True)

    present: list[str] = []
    for fname in filenames:
        stem = fname[:-4] if fname.endswith(".npy") else fname
        pv_src = Path(dataset_pv_dir) / f"{stem}.npy"
        if not pv_src.exists():
            continue
        for suffix in (".npy", "_posxy.npy", ".json"):
            src = Path(dataset_pv_dir) / f"{stem}{suffix}"
            if src.exists():
                _link_or_copy(src, pv_out / f"{stem}{suffix}")
        present.append(f"{stem}.npy")

    (splits_out / f"{label}_manifest.txt").write_text("".join(f"{f}\n" for f in present))
    return agg_root


def write_agg_config(
    agg_root: Path,
    galaxy_id: str,
    source_cfg: dict[str, Any],
    *,
    patch_vel: int,
    patch_pos: int,
    norm_method: str = "zscore_galaxy_only",
    post: dict[str, Any] | None = None,
    aggregate: dict[str, Any] | None = None,
) -> Path:
    """Write ``<galaxy>_agg_config.yaml`` carrying the fields aggregate needs.

    ``source_cfg`` must provide ``cube_path`` and a ``galaxy`` block (typically a
    resolved per-galaxy config, so ``galaxy.beam_fwhm_arcsec`` is FITS-filled).
    """
    agg_root = Path(agg_root)
    cfg = {
        "cube_path": source_cfg["cube_path"],
        "output_root": str(agg_root.resolve()),
        "galaxy": source_cfg.get("galaxy", {}),
        "train": {
            "patch_vel": int(patch_vel),
            "patch_pos": int(patch_pos),
            "norm_method": norm_method,
        },
        "post": dict(post or DEFAULT_POST),
        "aggregate": dict(aggregate or DEFAULT_AGGREGATE),
    }
    cfg_path = agg_root / f"{galaxy_id}_agg_config.yaml"
    save_yaml(cfg, cfg_path)
    return cfg_path
