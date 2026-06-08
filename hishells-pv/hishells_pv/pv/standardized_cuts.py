from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from hishells_pv.utils.wcs_tools import open_cube, pixel_scales_arcsec, velocity_axis_kms


@dataclass(frozen=True)
class StandardCutSpec:
    galaxy: str
    center_pix: tuple[float, float]
    angle_deg: float
    spatial_window_kpc: float = 5.0
    velocity_window_kms: float = 200.0
    local_velocity_kms: float | None = None
    source: str = "deployment_grid"
    cut_category: str = "deployment_grid"
    source_shell_id: int | None = None
    source_shell_type: int | None = None
    source_shell_center_pix: tuple[float, float] | None = None
    source_shell_radius_pix: tuple[float, float] | None = None
    spatial_offset_pix: float | None = None
    spatial_offset_beam: float | None = None
    angle_offset_deg: float | None = None
    velocity_offset_kms: float | None = None
    quality_flags: tuple[str, ...] = ()


def _fixed_velocity_axis(
    velocity_kms: np.ndarray,
    *,
    center_kms: float,
    window_kms: float,
    target_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a fixed-width velocity axis centered on the requested local velocity."""
    finite = velocity_kms[np.isfinite(velocity_kms)]
    if finite.size == 0:
        raise ValueError("velocity axis has no finite values")
    native_min = float(np.nanmin(finite))
    native_max = float(np.nanmax(finite))
    native_descending = bool(float(velocity_kms[0]) > float(velocity_kms[-1]))

    half = 0.5 * float(window_kms)
    requested_min = float(center_kms - half)
    requested_max = float(center_kms + half)
    target_min = requested_min
    target_max = requested_max
    shifted_to_zero_floor = False
    if target_min < 0.0:
        target_min = 0.0
        target_max = float(window_kms)
        shifted_to_zero_floor = True
    edge_policy = "zero_floor_shift_then_median_padding" if shifted_to_zero_floor else "centered_on_local_moment1_with_median_padding"

    n_bins = max(2, int(target_bins))
    step = float(window_kms) / float(n_bins)
    if native_descending:
        axis = np.linspace(target_max - 0.5 * step, target_min + 0.5 * step, n_bins, dtype=np.float32)
    else:
        axis = np.linspace(target_min + 0.5 * step, target_max - 0.5 * step, n_bins, dtype=np.float32)
    meta = {
        "target_velocity_min_kms": target_min,
        "target_velocity_max_kms": target_max,
        "requested_velocity_min_kms": requested_min,
        "requested_velocity_max_kms": requested_max,
        "velocity_window_shifted_to_zero_floor": shifted_to_zero_floor,
        "native_velocity_min_kms": native_min,
        "native_velocity_max_kms": native_max,
        "velocity_edge_policy": edge_policy,
        "velocity_resampled": True,
        "target_velocity_bins": int(n_bins),
        "velocity_bin_width_kms": step,
        "velocity_padding_low_kms": max(0.0, native_min - min(float(axis.min()), float(axis.max()))),
        "velocity_padding_high_kms": max(0.0, max(float(axis.min()), float(axis.max())) - native_max),
    }
    return axis, meta


def _interp_spectrum_to_axis(
    velocity_kms: np.ndarray,
    spectrum: np.ndarray,
    target_axis_kms: np.ndarray,
    *,
    pad_value: float = 0.0,
) -> np.ndarray:
    finite = np.isfinite(velocity_kms) & np.isfinite(spectrum)
    if finite.sum() < 2:
        return np.full(target_axis_kms.shape, float(pad_value), dtype=np.float32)
    xp = np.asarray(velocity_kms[finite], dtype=np.float64)
    fp = np.asarray(spectrum[finite], dtype=np.float64)
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    # Collapse duplicate velocity samples if present.
    uniq, inv = np.unique(xp, return_inverse=True)
    if uniq.size != xp.size:
        sums = np.zeros(uniq.shape, dtype=np.float64)
        counts = np.zeros(uniq.shape, dtype=np.float64)
        np.add.at(sums, inv, fp)
        np.add.at(counts, inv, 1.0)
        xp = uniq
        fp = sums / np.maximum(counts, 1.0)
    if xp.size < 2:
        return np.full(target_axis_kms.shape, float(pad_value), dtype=np.float32)
    out = np.interp(
        np.asarray(target_axis_kms, dtype=np.float64),
        xp,
        fp,
        left=float(pad_value),
        right=float(pad_value),
    )
    return out.astype(np.float32)


def physical_velocity_axis_kms(hdr: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a velocity axis in km/s, correcting common unlabeled m/s FELO cubes."""
    axis = velocity_axis_kms(hdr).astype(np.float32)
    ctype = " ".join(str(hdr.get(f"CTYPE{i}", "")).upper() for i in range(1, int(hdr.get("NAXIS", 0)) + 1))
    cunit = " ".join(str(hdr.get(f"CUNIT{i}", "")).lower() for i in range(1, int(hdr.get("NAXIS", 0)) + 1))
    corrected = False
    reason = None
    if ("FELO" in ctype or "VELO" in ctype or "VRAD" in ctype) and "km" not in cunit:
        finite = axis[np.isfinite(axis)]
        step = abs(float(np.nanmedian(np.diff(finite)))) if finite.size > 1 else 0.0
        if finite.size and (abs(float(np.nanmedian(finite))) > 2_000.0 or step > 100.0):
            axis = axis / 1000.0
            corrected = True
            reason = "velocity CUNIT is blank/non-km and FELO-like values look like m/s"
    return axis, {"velocity_axis_unit_corrected": corrected, "velocity_axis_correction_reason": reason}


def kpc_to_arcsec(window_kpc: float, distance_mpc: float) -> float:
    if distance_mpc <= 0:
        raise ValueError("distance_mpc must be positive")
    return float(window_kpc) / (float(distance_mpc) * 1000.0) * 206265.0


def spatial_window_pixels(window_kpc: float, distance_mpc: float, pixel_scale_arcsec: float) -> tuple[float, float]:
    angular = kpc_to_arcsec(window_kpc, distance_mpc)
    return angular, angular / max(float(pixel_scale_arcsec), 1e-9)


def moment1_velocity_map(
    cube: np.ndarray,
    velocity_kms: np.ndarray,
    *,
    min_moment0_percentile: float = 20.0,
) -> np.ndarray:
    weights = np.clip(np.nan_to_num(cube.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    moment0 = np.nansum(weights, axis=0)
    finite = moment0[np.isfinite(moment0) & (moment0 > 0)]
    out = np.full(moment0.shape, np.nan, dtype=np.float32)
    if finite.size == 0:
        return out
    floor = float(np.nanpercentile(finite, min_moment0_percentile))
    good = moment0 > floor
    weighted_v = np.nansum(weights * velocity_kms[:, None, None], axis=0)
    out[good] = (weighted_v[good] / np.maximum(moment0[good], 1e-12)).astype(np.float32)
    return out


def sample_line_coordinates(
    center_pix: tuple[float, float],
    angle_deg: float,
    half_length_pix: float,
    *,
    pos_step_pix: float = 1.0,
    target_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    theta = np.deg2rad(float(angle_deg))
    direction = (float(np.cos(theta)), float(np.sin(theta)))
    perp = (-direction[1], direction[0])
    if target_samples is not None:
        n = max(2, int(target_samples))
        t = np.linspace(-half_length_pix, half_length_pix, n, dtype=np.float32)
    else:
        t = np.arange(-half_length_pix, half_length_pix + 0.5 * pos_step_pix, pos_step_pix, dtype=np.float32)
    xs = float(center_pix[0]) + t * direction[0]
    ys = float(center_pix[1]) + t * direction[1]
    return xs, ys, t, direction, perp


def local_velocity_for_cut(moment1: np.ndarray, xs: np.ndarray, ys: np.ndarray, *, fallback: float | None = None) -> tuple[float | None, str]:
    nx = moment1.shape[1]
    ny = moment1.shape[0]
    vals: list[float] = []
    for x, y in zip(xs, ys):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < nx and 0 <= yi < ny and np.isfinite(moment1[yi, xi]):
            vals.append(float(moment1[yi, xi]))
    if vals:
        return float(np.nanmedian(vals)), "moment1_median_along_cut"
    return fallback, "fallback" if fallback is not None else "missing"


def sample_standardized_pv(
    cube: np.ndarray,
    hdr: Any,
    moment1: np.ndarray,
    spec: StandardCutSpec,
    *,
    distance_mpc: float,
    pixel_scale_arcsec: float | None = None,
    slit_width_pix: int = 3,
    pos_step_pix: float = 1.0,
    fallback_velocity_kms: float | None = None,
    source_cube_path: str | None = None,
    moment1_path: str | None = None,
    center_radec: tuple[float, float] | None = None,
    velocity_axis_kms_override: np.ndarray | None = None,
    velocity_meta_override: dict[str, Any] | None = None,
    target_velocity_bins: int | None = 96,
    target_spatial_pixels: int | None = 256,
    velocity_pad_value: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if target_velocity_bins is None:
        raise ValueError(
            "standardized PV cuts require target_velocity_bins so the full requested "
            "velocity window can be resampled without native-channel truncation"
        )
    if velocity_axis_kms_override is None:
        velocity_kms, velocity_meta = physical_velocity_axis_kms(hdr)
    else:
        velocity_kms = np.asarray(velocity_axis_kms_override, dtype=np.float32)
        velocity_meta = dict(velocity_meta_override or {})
    if (
        fallback_velocity_kms is not None
        and velocity_meta.get("velocity_axis_unit_corrected")
        and abs(float(fallback_velocity_kms)) > 2_000.0
    ):
        fallback_velocity_kms = float(fallback_velocity_kms) / 1000.0
    if pixel_scale_arcsec is None:
        raise ValueError("pixel_scale_arcsec is required")
    angular_arcsec, window_pix = spatial_window_pixels(spec.spatial_window_kpc, distance_mpc, pixel_scale_arcsec)
    half_len = 0.5 * window_pix
    xs, ys, pos_axis, direction, perp = sample_line_coordinates(
        spec.center_pix,
        spec.angle_deg,
        half_len,
        pos_step_pix=pos_step_pix,
        target_samples=target_spatial_pixels,
    )

    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    in_bounds = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
    local_v, local_method = local_velocity_for_cut(moment1, xs, ys, fallback=fallback_velocity_kms)
    if spec.local_velocity_kms is not None:
        local_v = float(spec.local_velocity_kms)
        local_method = "provided"
    if local_v is None:
        raise ValueError("Could not estimate local velocity and no fallback was provided")

    velocity_offset = float(spec.velocity_offset_kms or 0.0)
    velocity_center = float(local_v + velocity_offset)
    output_velocity_axis, fixed_velocity_meta = _fixed_velocity_axis(
        velocity_kms,
        center_kms=velocity_center,
        window_kms=float(spec.velocity_window_kms),
        target_bins=int(target_velocity_bins),
    )
    vmin = float(fixed_velocity_meta["target_velocity_min_kms"])
    vmax = float(fixed_velocity_meta["target_velocity_max_kms"])
    vmask = (velocity_kms >= min(vmin, vmax)) & (velocity_kms <= max(vmin, vmax))
    channel_idx = np.where(vmask)[0]
    if channel_idx.size == 0:
        nearest = int(np.argmin(np.abs(velocity_kms - local_v)))
        channel_idx = np.asarray([nearest], dtype=int)

    native_vmin = float(np.nanmin(velocity_kms))
    native_vmax = float(np.nanmax(velocity_kms))
    target_vmin = np.minimum(float(output_velocity_axis.min()), float(output_velocity_axis.max()))
    target_vmax = np.maximum(float(output_velocity_axis.min()), float(output_velocity_axis.max()))
    velocity_in_bounds = (output_velocity_axis >= native_vmin) & (output_velocity_axis <= native_vmax)

    slit_half = max(0, int(slit_width_pix) // 2)
    pv = np.full((output_velocity_axis.size, xs.size), np.nan, dtype=np.float32)
    validity_mask = np.zeros_like(pv, dtype=bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        spectra = []
        for offset in range(-slit_half, slit_half + 1):
            xi = int(round(float(x + offset * perp[0])))
            yi = int(round(float(y + offset * perp[1])))
            if 0 <= xi < nx and 0 <= yi < ny:
                spectra.append(cube[:, yi, xi])
        if spectra:
            native_spectrum = np.nanmean(np.stack(spectra, axis=0), axis=0)
            finite_native = native_spectrum[np.isfinite(native_spectrum)]
            local_pad = float(np.nanmedian(finite_native)) if finite_native.size else 0.0
            pad_value = local_pad if velocity_pad_value is None else float(velocity_pad_value)
            pv[:, i] = _interp_spectrum_to_axis(
                velocity_kms,
                native_spectrum,
                output_velocity_axis,
                pad_value=pad_value,
            )
            validity_mask[:, i] = velocity_in_bounds

    column_valid = np.any(validity_mask, axis=0)
    if np.any(~np.isfinite(pv)):
        finite_vals = pv[np.isfinite(pv)]
        global_fill = float(np.nanmedian(finite_vals)) if finite_vals.size else 0.0
        valid_cols = np.where(column_valid)[0]
        for i in np.where(~column_valid)[0]:
            if valid_cols.size:
                nearest = int(valid_cols[np.argmin(np.abs(valid_cols - i))])
                nearest_vals = pv[:, nearest]
                finite_nearest = nearest_vals[np.isfinite(nearest_vals)]
                fill = float(np.nanmedian(finite_nearest)) if finite_nearest.size else global_fill
            else:
                fill = global_fill
            pv[:, i] = np.where(np.isfinite(pv[:, i]), pv[:, i], fill)
        pv = np.where(np.isfinite(pv), pv, global_fill).astype(np.float32)

    if pv.shape[0] != int(target_velocity_bins):
        raise AssertionError(f"standardized PV velocity shape drifted: {pv.shape[0]} != {target_velocity_bins}")
    if target_spatial_pixels is not None and pv.shape[1] != int(target_spatial_pixels):
        raise AssertionError(f"standardized PV spatial shape drifted: {pv.shape[1]} != {target_spatial_pixels}")

    meta = {
        "standardization_version": "fixed-physical-window-local-velocity-median-padded-v4",
        "galaxy": spec.galaxy,
        "source_cube_path": source_cube_path,
        "moment1_path": moment1_path,
        "source": spec.source,
        "cut_category": spec.cut_category,
        "adopted_distance_mpc": float(distance_mpc),
        "spatial_window_kpc": float(spec.spatial_window_kpc),
        "angular_window_arcsec": float(angular_arcsec),
        "pixel_window_size": float(window_pix),
        "pixel_scale_arcsec": float(pixel_scale_arcsec),
        "pv_cut_center_pix": [float(spec.center_pix[0]), float(spec.center_pix[1])],
        "pv_cut_center_radec": None if center_radec is None else [float(center_radec[0]), float(center_radec[1])],
        "pv_cut_angle_deg": float(spec.angle_deg),
        "dir_pix": [float(direction[0]), float(direction[1])],
        "perp_pix": [float(perp[0]), float(perp[1])],
        "posxy_pix": [[float(x), float(y)] for x, y in zip(xs, ys)],
        "pos_step_pix": float(pos_step_pix),
        "target_spatial_pixels": None if target_spatial_pixels is None else int(target_spatial_pixels),
        "spatial_bin_width_kpc": float(spec.spatial_window_kpc) / max(int(target_spatial_pixels or len(xs)), 1),
        "pos_axis_pix": [float(x) for x in pos_axis],
        "slit_width_pix": int(slit_width_pix),
        "local_velocity_kms": float(local_v),
        "local_velocity_method": local_method,
        "velocity_center_kms": velocity_center,
        "velocity_offset_kms": velocity_offset,
        "velocity_window_kms": float(spec.velocity_window_kms),
        "velocity_window_policy": "fixed_width_shift_negative_lower_bound_to_0_200_no_native_truncation",
        "velocity_window_truncated": False,
        "velocity_min_kms": vmin,
        "velocity_max_kms": vmax,
        "channel_indices_used": [int(x) for x in channel_idx],
        "velocity_axis_kms": [float(x) for x in output_velocity_axis],
        "velocity_pad_value": None if velocity_pad_value is None else float(velocity_pad_value),
        "padding_policy": "local_valid_spectrum_median_and_nearest_valid_spatial_column_median",
        "validity_mask_available": True,
        "valid_spatial_columns": int(column_valid.sum()),
        "padded_spatial_columns": int((~column_valid).sum()),
        "validity_mask": validity_mask.astype(np.uint8).tolist(),
        "target_velocity_min_kms_centered": float(target_vmin),
        "target_velocity_max_kms_centered": float(target_vmax),
        "moment1_map_source": "derived_intensity_weighted_from_cube",
        "fallback_velocity_kms": fallback_velocity_kms,
        "source_shell_id": spec.source_shell_id,
        "source_shell_type": spec.source_shell_type,
        "source_shell_center_pix": None if spec.source_shell_center_pix is None else [
            float(spec.source_shell_center_pix[0]),
            float(spec.source_shell_center_pix[1]),
        ],
        "source_shell_radius_pix": None if spec.source_shell_radius_pix is None else [
            float(spec.source_shell_radius_pix[0]),
            float(spec.source_shell_radius_pix[1]),
        ],
        "spatial_offset_pix": None if spec.spatial_offset_pix is None else float(spec.spatial_offset_pix),
        "spatial_offset_beam": None if spec.spatial_offset_beam is None else float(spec.spatial_offset_beam),
        "angle_offset_deg": None if spec.angle_offset_deg is None else float(spec.angle_offset_deg),
        "quality_flags": list(spec.quality_flags),
        "nv": int(pv.shape[0]),
        "npos": int(pv.shape[1]),
    }
    meta.update(velocity_meta)
    meta.update(fixed_velocity_meta)
    return pv, meta


def deployment_grid_specs(
    galaxy: str,
    moment0: np.ndarray,
    *,
    angles_deg: list[float],
    stride_pix: float,
    mask_percentile: float = 40.0,
    max_specs: int | None = None,
) -> Iterator[StandardCutSpec]:
    finite = moment0[np.isfinite(moment0) & (moment0 > 0)]
    if finite.size == 0:
        return
    floor = max(float(np.nanpercentile(finite, mask_percentile)), float(np.nanmin(finite)))
    ys, xs = np.where(moment0 > floor)
    if xs.size == 0:
        return
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    centers: list[tuple[float, float]] = []
    for y in np.arange(ymin, ymax + 1, max(1.0, stride_pix)):
        for x in np.arange(xmin, xmax + 1, max(1.0, stride_pix)):
            if not np.isfinite(moment0[int(round(y)), int(round(x))]) or moment0[int(round(y)), int(round(x))] < floor:
                continue
            centers.append((float(x), float(y)))
    if max_specs is not None and centers:
        max_centers = max(1, int(max_specs) // max(1, len(angles_deg)))
        if len(centers) > max_centers:
            take = np.linspace(0, len(centers) - 1, max_centers).round().astype(int)
            centers = [centers[int(i)] for i in take]
    n = 0
    for x, y in centers:
            for angle in angles_deg:
                yield StandardCutSpec(galaxy=galaxy, center_pix=(x, y), angle_deg=float(angle))
                n += 1
                if max_specs is not None and n >= int(max_specs):
                    return


def write_standardized_cut(
    cfg: dict[str, Any],
    spec: StandardCutSpec,
    out_dir: Path,
    *,
    stem: str,
    target_velocity_bins: int | None = 96,
) -> dict[str, Any]:
    cube, hdr, wcs, _ = open_cube(cfg["cube_path"])
    v_axis, _ = physical_velocity_axis_kms(hdr)
    m1 = moment1_velocity_map(cube, v_axis)
    ax, ay = pixel_scales_arcsec(wcs)
    pix_scale = 0.5 * (ax + ay)
    pv, meta = sample_standardized_pv(
        cube,
        hdr,
        m1,
        spec,
        distance_mpc=float(cfg["galaxy"]["distance_mpc"]),
        pixel_scale_arcsec=pix_scale,
        slit_width_pix=int((cfg.get("pv") or {}).get("standardized", {}).get("slit_width_pix", 3)),
        pos_step_pix=float((cfg.get("pv") or {}).get("standardized", {}).get("pos_step_pix", 1.0)),
        fallback_velocity_kms=cfg.get("galaxy", {}).get("vsys_kms"),
        target_velocity_bins=target_velocity_bins,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}.npy", pv)
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return meta
