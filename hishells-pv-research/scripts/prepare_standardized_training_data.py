#!/usr/bin/env python3
"""
Create the clean physical baseline dataset for the final project.
The script writes standardized PV arrays, beam-audited label masks, split
manifests, and the training config consumed by the U-Net scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pv.label_pv import build_labels_for_grid_pv
from src.pv.shell_catalog import catalog_to_pixel_shells, load_bagetakos_table7
from src.pv.standardized_cuts import (
    StandardCutSpec,
    kpc_to_arcsec,
    moment1_velocity_map,
    physical_velocity_axis_kms,
    sample_standardized_pv,
)
from src.utils.config import resolve_config
from src.utils.io import dumps_json
from src.utils.wcs_tools import open_cube, pixel_scales_arcsec, radec_to_xy, synthesized_beam_arcsec, unit_vectors_for_pa


DEFAULT_VAL_GALAXIES = ("ho_i", "ngc_2366", "ngc_4449", "ngc_4736")
DEFAULT_TEST_GALAXIES = ("ddo53", "ho_ii", "ngc_3184", "ngc_7793")


def _split_for(galaxy_id: str, val_galaxies: set[str], test_galaxies: set[str]) -> str:
    """Assign one whole galaxy to the train, validation, or test split."""
    if galaxy_id in val_galaxies:
        return "val"
    if galaxy_id in test_galaxies:
        return "test"
    return "train"


def _canonical_aliases(item: dict[str, Any], cfg_path: Path) -> set[str]:
    """Collect the names that can identify one galaxy in CLI split options."""
    galaxy_id = Path(item.get("output_root", cfg_path.stem)).name
    return {
        galaxy_id.casefold(),
        str(item.get("name", "")).casefold(),
        Path(str(item.get("stem", ""))).name.casefold(),
    }


def _angle_from_vec(vec: tuple[float, float]) -> float:
    """Convert a 2D direction vector into the 0 to 180 degree PV angle frame."""
    return float((math.degrees(math.atan2(vec[1], vec[0])) + 360.0) % 180.0)


def _galaxy_frame_grid_angles(galaxy_major: tuple[float, float], step_deg: float) -> list[float]:
    """
    Build deployment-grid angles anchored to the galaxy major axis.
    The held-out grid tests use these angles to mimic how a new galaxy is searched.
    """
    step = max(1e-6, float(step_deg))
    n_steps = max(1, int(round(180.0 / step)))
    base = _angle_from_vec(galaxy_major)
    angles = [float((base + i * step) % 180.0) for i in range(n_steps)]
    # Collapse possible duplicates from awkward non-divisor step choices.
    return sorted({round(angle, 6) for angle in angles})


def _galaxy_frame_deployment_specs(
    *,
    galaxy_name: str,
    moment0: np.ndarray,
    center_pix: tuple[float, float],
    galaxy_major: tuple[float, float],
    galaxy_minor: tuple[float, float],
    angles_deg: list[float],
    stride_pix: float,
    spatial_window_kpc: float,
    velocity_window_kms: float,
    mask_percentile: float = 40.0,
    max_specs: int | None = None,
    quality_flags: tuple[str, ...] = (),
) -> list[StandardCutSpec]:
    """
    Place blind-search PV cuts on a major/minor-axis grid inside the HI disk.
    The grid is physical in intent but is evaluated in pixel coordinates after WCS setup.
    """
    finite = moment0[np.isfinite(moment0) & (moment0 > 0)]
    if finite.size == 0:
        return []
    floor = max(float(np.nanpercentile(finite, mask_percentile)), float(np.nanmin(finite)))
    ys, xs = np.where(moment0 > floor)
    if xs.size == 0:
        return []

    cx, cy = center_pix
    mx, my = galaxy_major
    nx, ny = galaxy_minor
    dx = xs.astype(float) - float(cx)
    dy = ys.astype(float) - float(cy)
    major_coord = dx * mx + dy * my
    minor_coord = dx * nx + dy * ny
    stride = max(1.0, float(stride_pix))
    major_values = np.arange(float(np.nanmin(major_coord)), float(np.nanmax(major_coord)) + 0.5 * stride, stride)
    minor_values = np.arange(float(np.nanmin(minor_coord)), float(np.nanmax(minor_coord)) + 0.5 * stride, stride)

    centers: list[tuple[float, float]] = []
    height, width = moment0.shape
    for major in major_values:
        for minor in minor_values:
            x = float(cx + major * mx + minor * nx)
            y = float(cy + major * my + minor * ny)
            xi = int(round(x))
            yi = int(round(y))
            if not (0 <= xi < width and 0 <= yi < height):
                continue
            if not np.isfinite(moment0[yi, xi]) or moment0[yi, xi] < floor:
                continue
            centers.append((x, y))

    if max_specs is not None and centers:
        max_centers = max(1, int(max_specs) // max(1, len(angles_deg)))
        if len(centers) > max_centers:
            take = np.linspace(0, len(centers) - 1, max_centers).round().astype(int)
            centers = [centers[int(i)] for i in take]

    specs: list[StandardCutSpec] = []
    for x, y in centers:
        for angle in angles_deg:
            specs.append(
                StandardCutSpec(
                    galaxy=galaxy_name,
                    center_pix=(float(x), float(y)),
                    angle_deg=float(angle),
                    spatial_window_kpc=spatial_window_kpc,
                    velocity_window_kms=velocity_window_kms,
                    source="deployment_grid_standardized",
                    cut_category="fine_grid_deployment_like",
                    quality_flags=quality_flags,
                )
            )
            if max_specs is not None and len(specs) >= int(max_specs):
                return specs
    return specs


def _ellipse_support_radius(direction, major_vec, minor_vec, a_pix, b_pix):
    """
    Estimate how far an elliptical catalog shell extends along one PV direction.
    Grazing cuts use this value to touch shell edges without deleting them later.
    """
    ux, uy = direction
    mx, my = major_vec
    nx, ny = minor_vec
    denom = ((ux * mx + uy * my) / max(a_pix, 1e-6)) ** 2
    denom += ((ux * nx + uy * ny) / max(b_pix, 1e-6)) ** 2
    if denom <= 0 or not np.isfinite(denom):
        return max(a_pix, b_pix)
    return float(1.0 / np.sqrt(denom))


def _orientation(name: str, shell: dict[str, Any], galaxy_major, galaxy_minor, pa_convention: str):
    """Return the PV direction used for a named catalog or galaxy orientation."""
    shell_major, shell_minor = unit_vectors_for_pa(float(shell["pa_deg"]), convention=pa_convention)
    if name == "shell_major":
        return shell_major, shell_minor
    if name == "shell_minor":
        return shell_minor, (-shell_major[0], -shell_major[1])
    if name == "galaxy_major":
        return galaxy_major, galaxy_minor
    if name == "galaxy_minor":
        return galaxy_minor, (-galaxy_major[0], -galaxy_major[1])
    raise ValueError(f"unknown orientation {name}")


def _shell_base_kwargs(shell: dict[str, Any]) -> dict[str, Any]:
    """Copy catalog-shell identity fields into every generated cut metadata record."""
    return {
        "source_shell_id": int(shell["shell_id"]) if shell.get("shell_id") is not None else None,
        "source_shell_type": int(shell["type"]) if shell.get("type") is not None else None,
        "source_shell_center_pix": (float(shell["xc"]), float(shell["yc"])),
        "source_shell_radius_pix": (float(shell["a_pix"]), float(shell["b_pix"])),
    }


def _catalog_specs(
    *,
    galaxy_name: str,
    shells: list[dict[str, Any]],
    galaxy_major,
    galaxy_minor,
    pa_convention: str,
    spatial_window_kpc: float,
    velocity_window_kms: float,
    beam_pix: float | None,
    pixel_offsets_when_no_beam: list[float],
) -> list[StandardCutSpec]:
    """
    Generate positive cuts around catalog shells.
    The set includes centered, spatial-offset, angle-offset, velocity-offset, and grazing cuts.
    """
    specs: list[StandardCutSpec] = []
    orientations = ["shell_major", "shell_minor", "galaxy_major", "galaxy_minor"]
    for shell in shells:
        shell_major, shell_minor = unit_vectors_for_pa(float(shell["pa_deg"]), convention=pa_convention)
        base = _shell_base_kwargs(shell)

        for orient_name in orientations:
            dvec, _ = _orientation(orient_name, shell, galaxy_major, galaxy_minor, pa_convention)
            specs.append(
                StandardCutSpec(
                    galaxy=galaxy_name,
                    center_pix=(float(shell["xc"]), float(shell["yc"])),
                    angle_deg=_angle_from_vec(dvec),
                    spatial_window_kpc=spatial_window_kpc,
                    velocity_window_kms=velocity_window_kms,
                    source="catalog_standardized",
                    cut_category=f"centered_positive__{orient_name}",
                    **base,
                )
            )

        for orient_name in ["shell_major", "shell_minor"]:
            dvec, pvec = _orientation(orient_name, shell, galaxy_major, galaxy_minor, pa_convention)
            if beam_pix:
                offsets = [(0.5 * beam_pix, 0.5), (1.0 * beam_pix, 1.0), (2.0 * beam_pix, 2.0)]
                flags: tuple[str, ...] = ()
            else:
                offsets = [(float(x), None) for x in pixel_offsets_when_no_beam]
                flags = ("beam_missing_used_pixel_offsets",)
            for offset_pix, offset_beam in offsets:
                for sign in (-1.0, 1.0):
                    signed = sign * offset_pix
                    specs.append(
                        StandardCutSpec(
                            galaxy=galaxy_name,
                            center_pix=(float(shell["xc"]) + signed * pvec[0], float(shell["yc"]) + signed * pvec[1]),
                            angle_deg=_angle_from_vec(dvec),
                            spatial_window_kpc=spatial_window_kpc,
                            velocity_window_kms=velocity_window_kms,
                            source="catalog_standardized",
                            cut_category=f"spatial_offset_{abs(offset_pix):g}pix__{orient_name}",
                            spatial_offset_pix=signed,
                            spatial_offset_beam=None if offset_beam is None else sign * offset_beam,
                            quality_flags=flags,
                            **base,
                        )
                    )

        dvec, pvec = shell_major, shell_minor
        shell_angle = _angle_from_vec(dvec)
        for delta in (-45.0, -22.5, 22.5, 45.0):
            specs.append(
                StandardCutSpec(
                    galaxy=galaxy_name,
                    center_pix=(float(shell["xc"]), float(shell["yc"])),
                    angle_deg=(shell_angle + delta) % 180.0,
                    spatial_window_kpc=spatial_window_kpc,
                    velocity_window_kms=velocity_window_kms,
                    source="catalog_standardized",
                    cut_category=f"angle_offset_{delta:+g}deg",
                    angle_offset_deg=delta,
                    **base,
                )
            )

        for voff in (-30.0, -15.0, 15.0, 30.0):
            specs.append(
                StandardCutSpec(
                    galaxy=galaxy_name,
                    center_pix=(float(shell["xc"]), float(shell["yc"])),
                    angle_deg=shell_angle,
                    spatial_window_kpc=spatial_window_kpc,
                    velocity_window_kms=velocity_window_kms,
                    source="catalog_standardized",
                    cut_category=f"velocity_offset_{voff:+g}kms",
                    velocity_offset_kms=voff,
                    **base,
                )
            )

        for sign in (-1.0, 1.0):
            # A near-edge cut: intentionally small real masks can occur here.
            radius = _ellipse_support_radius(pvec, shell_major, shell_minor, float(shell["a_pix"]), float(shell["b_pix"]))
            signed = sign * 0.9 * radius
            specs.append(
                StandardCutSpec(
                    galaxy=galaxy_name,
                    center_pix=(float(shell["xc"]) + signed * pvec[0], float(shell["yc"]) + signed * pvec[1]),
                    angle_deg=shell_angle,
                    spatial_window_kpc=spatial_window_kpc,
                    velocity_window_kms=velocity_window_kms,
                    source="catalog_standardized",
                    cut_category="random_nearby_grazing",
                    spatial_offset_pix=signed,
                    quality_flags=("designed_grazing_cut",),
                    **base,
                )
            )
    return specs


def _far_from_shells(x: float, y: float, shells: list[dict[str, Any]], min_sep_pix: float, sep_radius_factor: float) -> bool:
    """Check whether a candidate background center is safely away from catalog shells."""
    for shell in shells:
        radius = max(float(shell.get("a_pix", 0.0)), float(shell.get("b_pix", 0.0)))
        safe = max(min_sep_pix, sep_radius_factor * radius)
        if (x - float(shell["xc"])) ** 2 + (y - float(shell["yc"])) ** 2 < safe**2:
            return False
    return True


def _background_specs(
    *,
    galaxy_name: str,
    moment0: np.ndarray,
    shells: list[dict[str, Any]],
    spatial_window_kpc: float,
    velocity_window_kms: float,
    n_target: int,
    seed: int,
    min_sep_pix: float,
    sep_radius_factor: float,
) -> list[StandardCutSpec]:
    """
    Sample random negative cuts from real HI emission away from catalog shells.
    These are ordinary background examples, not mined hard negatives.
    """
    finite = moment0[np.isfinite(moment0) & (moment0 > 0)]
    if finite.size == 0 or n_target <= 0:
        return []
    lo = float(np.nanpercentile(finite, 45.0))
    hi = float(np.nanpercentile(finite, 99.5))
    ys, xs = np.where((moment0 >= lo) & (moment0 <= hi))
    if xs.size == 0:
        return []

    rng = np.random.default_rng(seed)
    specs: list[StandardCutSpec] = []
    attempts = 0
    while len(specs) < n_target and attempts < 100 * n_target:
        attempts += 1
        idx = int(rng.integers(0, xs.size))
        x = float(xs[idx])
        y = float(ys[idx])
        if not _far_from_shells(x, y, shells, min_sep_pix, sep_radius_factor):
            continue
        specs.append(
            StandardCutSpec(
                galaxy=galaxy_name,
                center_pix=(x, y),
                angle_deg=float(rng.uniform(0, 180.0)),
                spatial_window_kpc=spatial_window_kpc,
                velocity_window_kms=velocity_window_kms,
                source="random_background_standardized",
                cut_category="background_random_negative",
            )
        )
    return specs


def _load_galaxy_manifest(data_root: Path) -> list[dict[str, Any]]:
    """Load galaxy configs from the prepared training-data directory."""
    manifest = data_root / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["galaxies"]
    configs = sorted((data_root / "configs").glob("*.yaml"))
    return [{"name": p.stem, "config": str(p)} for p in configs if not p.name.endswith("._resolved.yaml")]


def _center_radec(wcs, center_pix: tuple[float, float]) -> tuple[float, float] | None:
    """Attach sky coordinates to a PV cut when the cube WCS can provide them."""
    try:
        ra, dec = wcs.celestial.pixel_to_world_values(float(center_pix[0]), float(center_pix[1]))
        return float(ra), float(dec)
    except Exception:
        return None


def _write_cut(
    *,
    spec: StandardCutSpec,
    stem: str,
    split: str,
    cfg: dict[str, Any],
    cube: np.ndarray,
    hdr,
    wcs,
    moment1: np.ndarray,
    moment1_path: Path,
    pixel_scale_arcsec: float,
    velocity_axis: np.ndarray,
    velocity_meta: dict[str, Any],
    shells: list[dict[str, Any]],
    label_opts: dict[str, Any],
    out_root: Path,
    target_velocity_bins: int,
    target_spatial_pixels: int,
    beam_major_arcsec: float | None,
    beam_minor_arcsec: float | None,
) -> dict[str, Any] | None:
    """
    Write one standardized PV cut, its label mask, and its metadata sidecars.
    This is the point where physical sampling, beam-aware label cleanup, and split rows meet.
    """
    pv, meta = sample_standardized_pv(
        cube,
        hdr,
        moment1,
        spec,
        distance_mpc=float(cfg["galaxy"]["distance_mpc"]),
        pixel_scale_arcsec=pixel_scale_arcsec,
        slit_width_pix=int((cfg.get("pv") or {}).get("standardized", {}).get("slit_width_pix", 3)),
        pos_step_pix=float((cfg.get("pv") or {}).get("standardized", {}).get("pos_step_pix", 1.0)),
        fallback_velocity_kms=cfg.get("galaxy", {}).get("vsys_kms"),
        source_cube_path=str(Path(cfg["cube_path"]).resolve()),
        moment1_path=str(moment1_path.resolve()),
        center_radec=_center_radec(wcs, spec.center_pix),
        velocity_axis_kms_override=velocity_axis,
        velocity_meta_override=velocity_meta,
        target_velocity_bins=target_velocity_bins,
        target_spatial_pixels=target_spatial_pixels,
    )
    if pv.shape != (int(target_velocity_bins), int(target_spatial_pixels)):
        raise ValueError(f"standardized PV has unexpected shape {pv.shape}; expected {(target_velocity_bins, target_spatial_pixels)}")
    if pv.shape[0] < 2 or pv.shape[1] < 8:
        return None
    fallback_beam = cfg["galaxy"].get("beam_fwhm_arcsec")
    if beam_major_arcsec is None and fallback_beam is not None:
        beam_major_arcsec = beam_minor_arcsec = float(fallback_beam)
    meta["beam_major_fwhm_arcsec"] = None if beam_major_arcsec is None else float(beam_major_arcsec)
    meta["beam_minor_fwhm_arcsec"] = None if beam_minor_arcsec is None else float(beam_minor_arcsec)
    meta["beam_fwhm_arcsec"] = meta["beam_major_fwhm_arcsec"]
    meta["beam_fwhm_pix"] = None if beam_major_arcsec is None else float(beam_major_arcsec) / max(pixel_scale_arcsec, 1e-9)
    beam_major_model_cols = None
    beam_minor_model_cols = None
    if beam_major_arcsec is not None:
        beam_major_model_cols = float(beam_major_arcsec) / max(float(meta["angular_window_arcsec"]), 1e-9) * int(target_spatial_pixels)
    if beam_minor_arcsec is not None:
        beam_minor_model_cols = float(beam_minor_arcsec) / max(float(meta["angular_window_arcsec"]), 1e-9) * int(target_spatial_pixels)
    meta["beam_major_model_cols"] = beam_major_model_cols
    meta["beam_minor_model_cols"] = beam_minor_model_cols

    posxy = np.asarray(meta["posxy_pix"], dtype=np.float32)
    v_axis = np.asarray(meta["velocity_axis_kms"], dtype=np.float32)
    opts = dict(label_opts)
    opts["hv_scale"] = 1.0
    lab, type_mask, objects, warnings = build_labels_for_grid_pv(posxy, v_axis, shells, opts)
    lab, type_mask, objects, beam_audit = _erase_isolated_subbeam_components(
        lab,
        type_mask,
        objects,
        beam_major_model_cols=beam_major_model_cols,
        beam_minor_model_cols=beam_minor_model_cols,
    )
    if beam_audit["removed_components"]:
        warnings.append({"reason": "beam_aware_subbeam_components_erased", **beam_audit})

    fname = f"{stem}.npy"
    np.save(out_root / "pv" / fname, pv.astype(np.float32))
    np.save(out_root / "labels" / fname, lab.astype(np.uint8))
    np.save(out_root / "label_types" / fname, type_mask.astype(np.uint8))
    np.save(out_root / "pv" / f"{stem}_posxy.npy", posxy.astype(np.float32))
    dumps_json(meta, out_root / "pv" / f"{stem}.json")
    label_sidecar = {
        "pv_file": fname,
        "label_shape": list(lab.shape),
        "n_objects": len(objects),
        "objects": objects,
        "warnings": warnings,
        "beam_audit": beam_audit,
        "source_shell_id": meta.get("source_shell_id"),
        "cut_category": meta.get("cut_category"),
    }
    dumps_json(label_sidecar, out_root / "labels" / f"{stem}.json")

    positive = bool(lab.any())
    return {
        "filename": fname,
        "image_path": str((out_root / "pv" / fname).resolve()),
        "mask_path": str((out_root / "labels" / fname).resolve()),
        "metadata_path": str((out_root / "pv" / f"{stem}.json").resolve()),
        "galaxy": stem.split("__", 1)[0],
        "split": split,
        "cut_type": meta.get("source"),
        "cut_category": meta.get("cut_category"),
        "shell_id": meta.get("source_shell_id"),
        "positive": int(positive),
        "centered_offset_grid_background_category": meta.get("cut_category"),
        "local_velocity_kms": meta.get("local_velocity_kms"),
        "velocity_center_kms": meta.get("velocity_center_kms"),
        "velocity_offset_kms": meta.get("velocity_offset_kms"),
        "spatial_window_kpc": meta.get("spatial_window_kpc"),
        "velocity_window_kms": meta.get("velocity_window_kms"),
        "quality_flags": ";".join(meta.get("quality_flags") or []),
        "n_label_objects": len(objects),
        "mask_pixels": int(lab.sum()),
        "beam_audit_removed_components": int(beam_audit["removed_components"]),
        "beam_audit_removed_pixels": int(beam_audit["removed_pixels"]),
        "nv": int(pv.shape[0]),
        "npos": int(pv.shape[1]),
        "velocity_resampled": meta.get("velocity_resampled"),
        "target_velocity_bins": meta.get("target_velocity_bins"),
        "velocity_edge_policy": meta.get("velocity_edge_policy"),
        "velocity_bin_width_kms": meta.get("velocity_bin_width_kms"),
    }


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected component coordinates as ``(N, 2)`` arrays."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for y0, x0 in zip(*np.where(mask & ~seen)):
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    if yy == y and xx == x:
                        continue
                    if mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        components.append(np.asarray(coords, dtype=np.int32))
    return components


def _object_still_present(obj: dict[str, Any], lab: np.ndarray) -> bool:
    """Check whether a catalog object still has label pixels after beam cleanup."""
    box = obj.get("bbox_vpos")
    if not box:
        return True
    v0, p0, v1, p1 = [int(x) for x in box]
    v0 = max(0, min(v0, lab.shape[0] - 1))
    v1 = max(0, min(v1, lab.shape[0] - 1))
    p0 = max(0, min(p0, lab.shape[1] - 1))
    p1 = max(0, min(p1, lab.shape[1] - 1))
    return bool(np.any(lab[v0 : v1 + 1, p0 : p1 + 1] > 0))


def _erase_isolated_subbeam_components(
    lab: np.ndarray,
    type_mask: np.ndarray,
    objects: list[dict[str, Any]],
    *,
    beam_major_model_cols: float | None,
    beam_minor_model_cols: float | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """
    Remove standalone sub-beam mask speckles while preserving grazing edges.

    The telescope beam constrains spatial resolution, so the audit is applied
    along the PV position axis. Components touching the PV boundary are kept to
    preserve large shells that only graze the cut edge.
    """
    audit = {
        "enabled": bool(beam_major_model_cols and beam_major_model_cols > 0),
        "beam_major_model_cols": None if beam_major_model_cols is None else float(beam_major_model_cols),
        "beam_minor_model_cols": None if beam_minor_model_cols is None else float(beam_minor_model_cols),
        "removed_components": 0,
        "removed_pixels": 0,
        "preserved_boundary_components": 0,
    }
    if not audit["enabled"] or not np.any(lab):
        return lab, type_mask, objects, audit

    cleaned = lab.copy()
    cleaned_types = type_mask.copy()
    min_spatial_width = max(1, int(np.ceil(float(beam_major_model_cols or 1.0))))
    min_area = max(1, int(np.ceil(float(beam_minor_model_cols or beam_major_model_cols or 1.0))))
    for coords in _connected_components(lab > 0):
        ys = coords[:, 0]
        xs = coords[:, 1]
        touches_boundary = (
            ys.min() == 0
            or ys.max() == lab.shape[0] - 1
            or xs.min() == 0
            or xs.max() == lab.shape[1] - 1
        )
        if touches_boundary:
            audit["preserved_boundary_components"] += 1
            continue
        spatial_width = int(xs.max() - xs.min() + 1)
        area = int(coords.shape[0])
        if spatial_width < min_spatial_width or area < min_area:
            cleaned[ys, xs] = 0
            cleaned_types[ys, xs] = 0
            audit["removed_components"] += 1
            audit["removed_pixels"] += area

    kept_objects = [obj for obj in objects if _object_still_present(obj, cleaned)]
    return cleaned, cleaned_types, kept_objects, audit


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a manifest table while preserving the first-seen column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_training_config(
    out_root: Path,
    config_path: Path,
    *,
    epochs: int,
    batch_size: int,
    target_velocity_bins: int,
    target_spatial_pixels: int,
    excluded_galaxies: list[str],
    stress_galaxies: list[str],
) -> None:
    """
    Write the training config tied to this generated dataset.
    The config records the clean-baseline choices so training can be reproduced later.
    """
    cfg = {
        "output_root": str(out_root.resolve()),
        "manifests": {
            "train_csv": str((out_root / "train_manifest.csv").resolve()),
            "val_csv": str((out_root / "val_manifest.csv").resolve()),
            "test_csv": str((out_root / "test_manifest.csv").resolve()),
            "stress_csv": str((out_root / "stress_manifest.csv").resolve()),
            "legacy_split_dir": str((out_root / "splits").resolve()),
        },
        "standardized_pv": {
            "enabled": True,
            "spatial_window_kpc": 5.0,
            "velocity_window_kms": 200.0,
            "velocity_center": "local_moment1",
            "velocity_range": "200 km/s total centered on the local Moment-1 velocity; padded where native channels are missing",
            "target_velocity_bins": target_velocity_bins,
            "velocity_bin_width_kms": 200.0 / max(int(target_velocity_bins), 1),
            "target_spatial_pixels": target_spatial_pixels,
            "spatial_bin_width_kpc": 5.0 / max(int(target_spatial_pixels), 1),
            "velocity_edge_policy": "center_local_moment1_window_then_median_pad_missing_native_channels",
            "padding_policy": "local valid median; no Gaussian-noise padding",
        },
        "excluded_galaxies": excluded_galaxies,
        "stress_validation_galaxies": stress_galaxies,
        "label_cleaning_mode": "beam_aware_subbeam_component_erasure",
        "hard_negative_injection": {
            "enabled": False,
            "note": "Clean Physical Baseline intentionally excludes mined hard negatives.",
        },
        "train": {
            "pos_fraction": 0.65,
            "patch_pos": target_spatial_pixels,
            "patch_vel": target_velocity_bins,
            "norm_method": "zscore_galaxy_only",
            "strict_fixed_shape": True,
            "samples_per_pv": 2,
            "max_steps_per_epoch": 900,
            "max_validation_steps": 220,
        },
        "model": {
            "base_filters": 24,
            "depth": 3,
            "dilation_rate": 1,
            "dropout": 0.10,
        },
        "loss": {
            "name": "bce_tversky",
            "tversky_alpha": 0.3,
            "tversky_beta": 0.7,
            "bce_weight": 0.5,
            "tversky_weight": 0.5,
            "positive_weight": 1.0,
        },
        "metrics": {
            "thresholds": [0.05, 0.075, 0.1],
            "model_selection": {
                "primary_monitor": "val_pr_auc",
                "high_recall_monitor": "val_patch_recall_0p075",
                "mode": "max",
            },
        },
        "segmented_validation": {
            "enabled": True,
            "split": "val",
            "every": 5,
            "max_cuts": None,
            "primary_deployment_metric": "Fine-Grid patch F1/recall tracked separately from aggregate validation.",
        },
        "optim": {
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": batch_size,
            "epochs": epochs,
            "loss": "bce_tversky",
        },
        "notes": {
            "model_task": "High-recall PV shell localization, not final clean catalog generation.",
            "split_policy": "Whole galaxies are held out. ngc_3031 is isolated to the stress split. Hard negatives from old non-standardized runs are not included.",
            "thresholds": "Monitor 0.05, 0.075, and 0.1; favor recall at 0.05/0.075.",
            "velocity_resampling": "PV cuts are interpolated to a fixed number of velocity bins so the model patch spans the configured physical velocity window.",
        },
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    """Run full clean-physical-baseline data generation from CLI arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("training_data"))
    ap.add_argument("--out-root", type=Path, default=Path("training_data/standardized_5kpc_200kms"))
    ap.add_argument("--val-galaxies", nargs="*", default=list(DEFAULT_VAL_GALAXIES))
    ap.add_argument("--test-galaxies", nargs="*", default=list(DEFAULT_TEST_GALAXIES))
    ap.add_argument("--spatial-window-kpc", type=float, default=5.0)
    ap.add_argument("--velocity-window-kms", type=float, default=200.0)
    ap.add_argument("--target-velocity-bins", type=int, default=96)
    ap.add_argument("--target-spatial-pixels", type=int, default=256)
    ap.add_argument("--grid-max-per-galaxy", type=int, default=96)
    ap.add_argument("--grid-stride-pix", type=float, default=96.0)
    ap.add_argument(
        "--grid-stride-kpc",
        type=float,
        default=None,
        help="Physical center spacing for deployment-like grid cuts. Overrides --grid-stride-pix.",
    )
    ap.add_argument("--grid-angle-step-deg", type=float, default=45.0)
    ap.add_argument("--background-per-shell", type=int, default=4)
    ap.add_argument("--background-max-per-galaxy", type=int, default=200)
    ap.add_argument("--pixel-offsets-when-no-beam", nargs="*", type=float, default=[4.0, 8.0, 16.0])
    ap.add_argument("--only", nargs="*", default=None, help="Optional galaxy ids, names, or stems for a limited generation pass.")
    ap.add_argument("--exclude-galaxies", nargs="*", default=[], help="Galaxy ids/names/stems to leave out of all splits.")
    ap.add_argument("--stress-galaxies", nargs="*", default=["ngc_3031"], help="Galaxy ids/names/stems reserved for standalone stress validation.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = args.out_root.resolve()
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    for sub in ("pv", "labels", "label_types", "moment1", "splits"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    val_galaxies = set(args.val_galaxies)
    test_galaxies = set(args.test_galaxies)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": [], "stress": []}
    skipped = Counter()

    only = {x.casefold() for x in args.only} if args.only else None
    exclude = {x.casefold() for x in args.exclude_galaxies}
    stress_galaxies = {x.casefold() for x in args.stress_galaxies}
    for item in _load_galaxy_manifest(args.data_root):
        cfg_path = Path(item["config"])
        galaxy_id = Path(item.get("output_root", cfg_path.stem)).name
        aliases = _canonical_aliases(item, cfg_path)
        if exclude & aliases:
            skipped[f"{galaxy_id}:excluded"] += 1
            print(f"[standardized-data] skipping excluded galaxy {galaxy_id}")
            continue
        if only and not (only & aliases):
            continue
        split = "stress" if (stress_galaxies & aliases) else _split_for(galaxy_id, val_galaxies, test_galaxies)
        cfg = resolve_config(str(cfg_path), write_resolved=False)
        galaxy_name = str(cfg["galaxy"].get("name") or item.get("name") or galaxy_id)
        print(f"[standardized-data] {galaxy_id} ({split})")

        cube, hdr, wcs, _ = open_cube(cfg["cube_path"])
        velocity_axis, velocity_meta = physical_velocity_axis_kms(hdr)
        moment0 = np.nansum(np.clip(np.nan_to_num(cube), 0.0, None), axis=0)
        moment1 = moment1_velocity_map(cube, velocity_axis)
        moment1_path = out_root / "moment1" / f"{galaxy_id}_moment1.npy"
        np.save(moment1_path, moment1.astype(np.float32))
        ax, ay = pixel_scales_arcsec(wcs)
        pixel_scale = 0.5 * (ax + ay)
        if args.grid_stride_kpc is not None:
            grid_stride_arcsec = kpc_to_arcsec(float(args.grid_stride_kpc), float(cfg["galaxy"]["distance_mpc"]))
            grid_stride_pix = grid_stride_arcsec / max(pixel_scale, 1e-9)
        else:
            grid_stride_pix = float(args.grid_stride_pix)
        header_beam = synthesized_beam_arcsec(hdr)
        beam_major_arcsec = header_beam[0] if header_beam is not None else None
        beam_minor_arcsec = header_beam[1] if header_beam is not None else None
        fallback_beam = cfg["galaxy"].get("beam_fwhm_arcsec")
        if beam_major_arcsec is None and fallback_beam is not None:
            beam_major_arcsec = beam_minor_arcsec = float(fallback_beam)
        beam_pix = None if beam_major_arcsec is None else float(beam_major_arcsec) / max(pixel_scale, 1e-9)
        if beam_pix is None:
            skipped["beam_missing_used_pixel_offsets"] += 1

        holes_dat = cfg["catalogs"]["holes_dat"]
        target = cfg["catalogs"].get("target_galaxy") or galaxy_name
        keep_types = cfg.get("pv", {}).get("label", {}).get("keep_types", [1, 2, 3])
        catalog = load_bagetakos_table7(holes_dat, target_galaxy=target, keep_types=keep_types)
        shells, diag = catalog_to_pixel_shells(catalog, wcs=wcs, distance_mpc=float(cfg["galaxy"]["distance_mpc"]))
        nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
        shells = [s for s in shells if 0 <= s["xc"] < nx and 0 <= s["yc"] < ny]
        if diag.get("warnings"):
            skipped[f"{galaxy_id}:catalog_warnings"] += len(diag["warnings"])

        grid = cfg.get("pv", {}).get("grid", {})
        pa_convention = str(grid.get("pa_convention", "astro")).lower()
        galaxy_major, galaxy_minor = unit_vectors_for_pa(float(cfg["galaxy"]["pa_deg"]), convention=pa_convention)
        grid_angles = _galaxy_frame_grid_angles(galaxy_major, args.grid_angle_step_deg)
        try:
            galaxy_center_pix = radec_to_xy(
                wcs,
                float(cfg["galaxy"]["ra_deg"]),
                float(cfg["galaxy"]["dec_deg"]),
            )
        except Exception:
            good = moment0[np.isfinite(moment0) & (moment0 > 0)]
            floor = float(np.nanpercentile(good, 40.0)) if good.size else 0.0
            yy, xx = np.where(moment0 > floor)
            galaxy_center_pix = (
                float(np.nanmedian(xx)) if xx.size else 0.5 * nx,
                float(np.nanmedian(yy)) if yy.size else 0.5 * ny,
            )
        specs = _catalog_specs(
            galaxy_name=galaxy_name,
            shells=shells,
            galaxy_major=galaxy_major,
            galaxy_minor=galaxy_minor,
            pa_convention=pa_convention,
            spatial_window_kpc=args.spatial_window_kpc,
            velocity_window_kms=args.velocity_window_kms,
            beam_pix=beam_pix,
            pixel_offsets_when_no_beam=args.pixel_offsets_when_no_beam,
        )

        grid_flags = (
            "galaxy_frame_center_grid",
            f"physical_grid_stride_kpc={float(args.grid_stride_kpc):.6g}",
            f"physical_grid_stride_pix={float(grid_stride_pix):.6g}",
        ) if args.grid_stride_kpc is not None else ("galaxy_frame_center_grid",)
        specs.extend(
            _galaxy_frame_deployment_specs(
                galaxy_name=galaxy_name,
                moment0=moment0,
                center_pix=galaxy_center_pix,
                galaxy_major=galaxy_major,
                galaxy_minor=galaxy_minor,
                angles_deg=grid_angles,
                stride_pix=grid_stride_pix,
                spatial_window_kpc=args.spatial_window_kpc,
                velocity_window_kms=args.velocity_window_kms,
                max_specs=args.grid_max_per_galaxy,
                quality_flags=grid_flags,
            )
        )

        n_bg = min(args.background_max_per_galaxy, max(0, args.background_per_shell * len(shells)))
        specs.extend(
            _background_specs(
                galaxy_name=galaxy_name,
                moment0=moment0,
                shells=shells,
                spatial_window_kpc=args.spatial_window_kpc,
                velocity_window_kms=args.velocity_window_kms,
                n_target=n_bg,
                seed=12345 + sum(ord(ch) for ch in galaxy_id),
                min_sep_pix=20.0,
                sep_radius_factor=1.5,
            )
        )

        label_opts = cfg.get("pv", {}).get("label", {})
        for idx, spec in enumerate(specs):
            stem = f"{galaxy_id}__std_{idx:06d}"
            try:
                row = _write_cut(
                    spec=spec,
                    stem=stem,
                    split=split,
                    cfg=cfg,
                    cube=cube,
                    hdr=hdr,
                    wcs=wcs,
                    moment1=moment1,
                    moment1_path=moment1_path,
                    pixel_scale_arcsec=pixel_scale,
                    velocity_axis=velocity_axis,
                    velocity_meta=velocity_meta,
                    shells=shells,
                    label_opts=label_opts,
                    out_root=out_root,
                    target_velocity_bins=args.target_velocity_bins,
                    target_spatial_pixels=args.target_spatial_pixels,
                    beam_major_arcsec=beam_major_arcsec,
                    beam_minor_arcsec=beam_minor_arcsec,
                )
            except Exception as exc:
                skipped[f"{galaxy_id}:write_failed"] += 1
                if skipped[f"{galaxy_id}:write_failed"] <= 5:
                    print(f"[standardized-data] warning: {galaxy_id} {idx} failed: {exc}")
                continue
            if row is None:
                skipped[f"{galaxy_id}:too_short"] += 1
                continue
            row["velocity_axis_unit_corrected"] = velocity_meta.get("velocity_axis_unit_corrected")
            rows_by_split[split].append(row)

    summary = {
        "output_root": str(out_root),
        "split_policy": "galaxy-held-out",
        "val_galaxies": sorted(val_galaxies),
        "test_galaxies": sorted(test_galaxies),
        "excluded_galaxies": sorted(args.exclude_galaxies),
        "stress_galaxies": sorted(args.stress_galaxies),
        "standardized_representation": {
            "spatial_window_kpc": args.spatial_window_kpc,
            "target_spatial_pixels": args.target_spatial_pixels,
            "velocity_window_kms": args.velocity_window_kms,
            "velocity_center": "local Moment-1 velocity",
            "target_velocity_bins": args.target_velocity_bins,
            "velocity_edge_policy": "center_local_moment1_window_then_median_pad_missing_native_channels",
            "padding_policy": "local valid median; no Gaussian-noise padding",
            "deployment_grid": {
                "grid_stride_pix": args.grid_stride_pix,
                "grid_stride_kpc": args.grid_stride_kpc,
                "grid_angle_step_deg": args.grid_angle_step_deg,
                "grid_max_per_galaxy": args.grid_max_per_galaxy,
                "angle_frame": "galaxy_major_axis",
            },
        },
        "skipped": dict(skipped),
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        _write_csv(rows, out_root / f"{split}_manifest.csv")
        (out_root / "splits" / f"{split}_manifest.txt").write_text("".join(f"{r['filename']}\n" for r in rows))
        summary["splits"][split] = {
            "count": len(rows),
            "positives": int(sum(r["positive"] for r in rows)),
            "negatives": int(sum(1 - int(r["positive"]) for r in rows)),
            "galaxies": dict(Counter(r["galaxy"] for r in rows)),
            "cut_categories": dict(Counter(r["cut_category"] for r in rows)),
            "positives_by_category": dict(Counter(r["cut_category"] for r in rows if int(r["positive"]))),
            "negatives_by_category": dict(Counter(r["cut_category"] for r in rows if not int(r["positive"]))),
        }

    dumps_json(summary, out_root / "standardized_split_summary.json")
    _write_training_config(
        out_root,
        out_root / "train_standardized_high_recall.yaml",
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_velocity_bins=args.target_velocity_bins,
        target_spatial_pixels=args.target_spatial_pixels,
        excluded_galaxies=sorted(args.exclude_galaxies),
        stress_galaxies=sorted(args.stress_galaxies),
    )
    print(f"[standardized-data] wrote dataset -> {out_root}")
    for split, info in summary["splits"].items():
        print(f"[standardized-data] {split}: n={info['count']} pos={info['positives']} neg={info['negatives']}")
    print(f"[standardized-data] wrote config -> {out_root / 'train_standardized_high_recall.yaml'}")


if __name__ == "__main__":
    main()
