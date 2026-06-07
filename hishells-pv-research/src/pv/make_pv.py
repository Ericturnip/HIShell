# Grid-aligned PV slicer for galaxy major/minor axes.
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.pv.shell_catalog import catalog_to_pixel_shells, load_bagetakos_table7
from src.utils.io import load_yaml, dumps_json
from src.utils.config import resolve_config
from src.utils.wcs_tools import (
    open_cube, pixel_scales_arcsec, radec_to_xy,
    velocity_axis_kms,
    unit_vectors_for_pa, rotate_xy
)

__VERSION__ = "make_pv.py@hybrid-grid+catalog-2.0"

def load_cfg(cfg_path: str):
    return resolve_config(cfg_path, write_resolved=True)


def _moment0(cube):
    m0 = np.nansum(cube, axis=0).astype(np.float32)
    mx = np.nanmax(m0) if np.isfinite(m0).any() else 0.0
    if mx > 0:
        m0 = m0 / mx
    return np.nan_to_num(m0, nan=0.0, posinf=0.0, neginf=0.0)

def _safe_sample_spectrum(cube, x, y, slit_half, nx, ny, perp_vec):
    """
    Average spectra across a thin slit centered on one line sample.
    Nearest-neighbor sampling keeps this generation step fast and stable.
    """
    vdim = cube.shape[0]
    sx, sy = perp_vec
    acc = []
    for k in range(-slit_half, slit_half + 1):
        xi = int(round(x + k * sx))
        yi = int(round(y + k * sy))
        if 0 <= xi < nx and 0 <= yi < ny:
            acc.append(cube[:, yi, xi])
    if not acc:
        return np.zeros(vdim, dtype=np.float32)
    return np.mean(acc, axis=0)

def _draw_endcap(ax, x0, y0, sx, sy, slit_half, lw=1.0):
    x1, y1 = x0 - sy * slit_half, y0 + sx * slit_half
    x2, y2 = x0 + sy * slit_half, y0 - sx * slit_half
    ax.plot([x1, x2], [y1, y2], lw=lw)

def _ellipse_support_radius(direction, major_vec, minor_vec, a_pix, b_pix):
    """Distance from ellipse center to boundary along a unit direction."""
    ux, uy = direction
    mx, my = major_vec
    nx, ny = minor_vec
    denom = ((ux * mx + uy * my) / max(a_pix, 1e-6)) ** 2
    denom += ((ux * nx + uy * ny) / max(b_pix, 1e-6)) ** 2
    if denom <= 0 or not np.isfinite(denom):
        return max(a_pix, b_pix)
    return 1.0 / np.sqrt(denom)

def _save_line_pv(
    *,
    stem_base,
    cube,
    hdr,
    cfg,
    out_dir: Path,
    overlays_dir: Path,
    m0,
    v_axis,
    x0,
    y0,
    dir_x,
    dir_y,
    perp_x,
    perp_y,
    half_len_pix,
    slit_width_pix,
    pos_step_pix,
    meta_extra,
    qa_enabled=True,
):
    """Save one PV slice, its sampled sky pixels, metadata, and optional QA."""
    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    slit_half = max(0, int(slit_width_pix) // 2)
    npos = int(2 * half_len_pix / pos_step_pix) + 1
    if npos < 9:
        return False

    ts = np.linspace(-half_len_pix, half_len_pix, npos, dtype=np.float32)
    xs = x0 + ts * dir_x
    ys = y0 + ts * dir_y
    in_bounds = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
    ts = ts[in_bounds]
    xs, ys = xs[in_bounds], ys[in_bounds]
    if xs.size < 8:
        return False

    pv = np.zeros((len(v_axis), xs.size), dtype=np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        pv[:, i] = _safe_sample_spectrum(cube, x, y, slit_half, nx, ny, (perp_x, perp_y))

    np.save(out_dir / f"{stem_base}.npy", pv)
    np.save(out_dir / f"{stem_base}_posxy.npy", np.stack([xs, ys], axis=1).astype(np.float32))

    meta = {
        "script": __VERSION__,
        "cfg_hash": cfg["_meta"]["_hash"],
        "center_pix": [float(x0), float(y0)],
        "dir_pix": [float(dir_x), float(dir_y)],
        "perp_pix": [float(perp_x), float(perp_y)],
        "slit_width_pix": int(slit_width_pix),
        "pos_step_pix": float(pos_step_pix),
        "pos_pix": [float(ts[0]), float(ts[-1]), float(pos_step_pix)],
        "pos_axis_pix": [float(t) for t in ts],
        "npos": int(pv.shape[1]),
        "nv": int(pv.shape[0]),
        "vel_kms": [float(v_axis[0]), float(v_axis[-1]), float(v_axis[1] - v_axis[0])],
    }
    meta.update(meta_extra)
    dumps_json(meta, out_dir / f"{stem_base}.json")

    if qa_enabled:
        plt.figure(figsize=(6, 5))
        plt.imshow(m0, origin="lower", cmap="gray")
        plt.plot(xs, ys, lw=1.0)
        if xs.size >= 2:
            _draw_endcap(plt.gca(), xs[0], ys[0], perp_x, perp_y, slit_half, lw=1.0)
            _draw_endcap(plt.gca(), xs[-1], ys[-1], perp_x, perp_y, slit_half, lw=1.0)
        plt.title(stem_base)
        plt.tight_layout()
        plt.savefig(overlays_dir / f"{stem_base}_spatial.png", dpi=140)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.imshow(pv, origin="lower", aspect="auto")
        plt.colorbar()
        plt.title(f"PV (velocity x position): {stem_base}")
        plt.tight_layout()
        plt.savefig(overlays_dir / f"{stem_base}_pv.png", dpi=140)
        plt.close()

    return True

def make_grid_pv(cube, hdr, wcs, cfg, out_dir: Path, overlays_dir: Path):
    """
    Write a galaxy-aligned grid of PV slices.
    Major cuts run along the galaxy major axis.
    Minor cuts run along the galaxy minor axis.
    """
    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    v = velocity_axis_kms(hdr)
    ax_as, ay_as = pixel_scales_arcsec(wcs)  # arcsec/pix (x,y)
    ra0, dec0 = cfg["galaxy"]["ra_deg"], cfg["galaxy"]["dec_deg"]
    cx, cy = radec_to_xy(wcs, ra0, dec0)

    grid = cfg["pv"]["grid"]
    frame = str(grid.get("frame", "galaxy")).lower()

    # NEW: orientation controls
    conv = str(grid.get("pa_convention", "astro")).lower()  # "astro" | "image"
    pa_delta = float(grid.get("pa_delta_deg", 0.0))
    pa_eff = float(cfg["galaxy"]["pa_deg"]) + pa_delta

    # extents & steps (arcsec)
    x_extent_as = float(grid["x_extent_arcsec"])
    y_extent_as = float(grid["y_extent_arcsec"])
    x_step_as   = float(grid["x_step_arcsec"])
    y_step_as   = float(grid["y_step_arcsec"])
    margin_as   = float(grid.get("margin_arcsec", 0.0))
    slit_w      = int(grid.get("slit_width_pix", 3))
    slit_half   = max(0, slit_w // 2)

    # sampling stride along the slice in pixels
    pos_step_pix = float(cfg["pv"]["grid"].get("pos_step_pix", 1.0))

    # major and minor directions in image-pixel space
    if frame == "galaxy":
        dvec_major, nvec_major = unit_vectors_for_pa(pa_eff, convention=conv)
        ux, uy = dvec_major
        vx, vy = nvec_major
        x_pix_per_as = 1.0 / ax_as
        y_pix_per_as = 1.0 / ay_as
    elif frame == "radec":
        # Align with image axes when the config asks for image-frame cuts.
        ux, uy = 1.0, 0.0
        vx, vy = 0.0, 1.0
        x_pix_per_as = 1.0 / ax_as
        y_pix_per_as = 1.0 / ay_as
    else:
        raise ValueError(f"pv.grid.frame must be 'galaxy' or 'radec', got {frame!r}")

    # extents/steps in pixels (shrink by margin)
    x_extent_pix = max(0.0, (x_extent_as - margin_as) * x_pix_per_as)
    y_extent_pix = max(0.0, (y_extent_as - margin_as) * y_pix_per_as)
    x_step_pix   = max(1.0, x_step_as * x_pix_per_as)
    y_step_pix   = max(1.0, y_step_as * y_pix_per_as)

    m0 = _moment0(cube)

    def _save_line_pv(
        stem_base,
        x0,
        y0,
        dir_x,
        dir_y,
        perp_x,
        perp_y,
        half_len_pix,
        *,
        axis_name,
        offset_arcsec,
    ):
        # sample positions along the line
        npos = int(2 * half_len_pix / pos_step_pix) + 1
        if npos < 9:
            return False
        ts = np.linspace(-half_len_pix, +half_len_pix, npos, dtype=np.float32)
        xs = x0 + ts * dir_x
        ys = y0 + ts * dir_y

        # clip to in-bounds
        mask = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
        ts = ts[mask]
        xs, ys = xs[mask], ys[mask]
        if xs.size < 8:
            return False

        # build PV
        pv = np.zeros((len(v), xs.size), dtype=np.float32)
        for i, (x, y) in enumerate(zip(xs, ys)):
            pv[:, i] = _safe_sample_spectrum(cube, x, y, slit_half, nx, ny, (perp_x, perp_y))

        # save arrays + sidecars
        np.save(out_dir / f"{stem_base}.npy", pv)
        np.save(out_dir / f"{stem_base}_posxy.npy", np.stack([xs, ys], axis=1).astype(np.float32))
        meta = {
            "script": __VERSION__, "cfg_hash": cfg["_meta"]["_hash"],
            "type": "grid",
            "frame": frame,
            "grid_axis": axis_name,
            "offset_arcsec": float(offset_arcsec),
            "pa_convention": conv,
            "pa_eff_deg": float(pa_eff),
            "center_pix": [float(x0), float(y0)],
            "dir_pix": [float(dir_x), float(dir_y)],     # along the line
            "perp_pix": [float(perp_x), float(perp_y)],  # across the slit
            "slit_width_pix": int(slit_w),
            "pos_step_pix": float(pos_step_pix),
            "pos_pix": [float(ts[0]), float(ts[-1]), float(pos_step_pix)],
            "pos_axis_pix": [float(t) for t in ts],
            "npos": int(pv.shape[1]),
            "nv": int(pv.shape[0]),
            "vel_kms": [float(v[0]), float(v[-1]), float(v[1]-v[0])]
        }
        dumps_json(meta, out_dir / f"{stem_base}.json")

        # spatial overlay
        plt.figure(figsize=(6, 5))
        plt.imshow(m0, origin="lower", cmap="gray")
        plt.plot(xs, ys, lw=1.0)
        # endcaps (show slit width at both ends)
        if xs.size >= 2:
            _draw_endcap(plt.gca(), xs[0],  ys[0],  perp_x, perp_y, slit_half, lw=1.0)
            _draw_endcap(plt.gca(), xs[-1], ys[-1], perp_x, perp_y, slit_half, lw=1.0)
        plt.title(stem_base)
        plt.tight_layout()
        plt.savefig(overlays_dir / f"{stem_base}_spatial.png", dpi=140)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.imshow(pv, origin="lower", aspect="auto")
        plt.colorbar()
        plt.title(f"PV (velocity x position): {stem_base}")
        plt.tight_layout()
        plt.savefig(overlays_dir / f"{stem_base}_pv.png", dpi=140)
        plt.close()
        return True

    n_minor, n_major = 0, 0

    # Vertical cuts use constant major-axis offset and run along the minor axis.
    x_positions_pix = np.arange(-x_extent_pix, x_extent_pix + 1e-6, x_step_pix, dtype=np.float32)
    for xprime in x_positions_pix:
        # Point on the central line at the requested major-axis offset.
        x0 = cx + xprime * ux
        y0 = cy + xprime * uy
        # human-friendly arcsec value for naming (approx via x scaling)
        xprime_as = xprime / x_pix_per_as if x_pix_per_as != 0 else 0.0
        stem = f"grid_xp_{int(round(xprime_as))}as"
        if _save_line_pv(
            stem,
            x0,
            y0,
            dir_x=vx,
            dir_y=vy,
            perp_x=ux,
            perp_y=uy,
            half_len_pix=y_extent_pix,
            axis_name="minor",
            offset_arcsec=xprime_as,
        ):
            n_minor += 1

    # Horizontal cuts use constant minor-axis offset and run along the major axis.
    y_positions_pix = np.arange(-y_extent_pix, y_extent_pix + 1e-6, y_step_pix, dtype=np.float32)
    for yprime in y_positions_pix:
        x0 = cx + yprime * vx
        y0 = cy + yprime * vy
        yprime_as = yprime / y_pix_per_as if y_pix_per_as != 0 else 0.0
        stem = f"grid_yp_{int(round(yprime_as))}as"
        if _save_line_pv(
            stem,
            x0,
            y0,
            dir_x=ux,
            dir_y=uy,
            perp_x=vx,
            perp_y=vy,
            half_len_pix=x_extent_pix,
            axis_name="major",
            offset_arcsec=yprime_as,
        ):
            n_major += 1

    return {"major": n_major, "minor": n_minor, "total": n_major + n_minor}

def _orientation_vectors(name, shell, galaxy_major, galaxy_minor, pa_convention):
    if name == "shell_major":
        return unit_vectors_for_pa(float(shell["pa_deg"]), convention=pa_convention)
    if name == "shell_minor":
        major, minor = unit_vectors_for_pa(float(shell["pa_deg"]), convention=pa_convention)
        return minor, (-major[0], -major[1])
    if name == "galaxy_major":
        return galaxy_major, galaxy_minor
    if name == "galaxy_minor":
        return galaxy_minor, (-galaxy_major[0], -galaxy_major[1])
    raise ValueError(f"unknown shell cut orientation: {name!r}")

def make_catalog_shell_pv(cube, hdr, wcs, cfg, out_dir: Path, overlays_dir: Path):
    """
    Add catalog-centered PV cuts so every known shell can appear in multiple,
    non-identical training/validation views.

    The blind grid remains the inference-like/background sampling path. These
    cuts are explicitly tagged as ``type='catalog_shell'`` and
    ``source='catalog_augmented'`` in their JSON sidecars so downstream code can
    split, down-weight, or exclude them if needed.
    """
    scfg = (cfg.get("pv") or {}).get("shell_cuts", {})
    if not scfg.get("enabled", False):
        return {"shells": 0, "cuts": 0, "by_type": {}}

    holes_dat = (cfg.get("catalogs") or {}).get("holes_dat")
    if not holes_dat or not Path(holes_dat).exists():
        print("[make_pv] WARNING: pv.shell_cuts enabled but catalogs.holes_dat is missing.")
        return {"shells": 0, "cuts": 0, "by_type": {}}
    distance_mpc = (cfg.get("galaxy") or {}).get("distance_mpc")
    if distance_mpc is None:
        print("[make_pv] WARNING: pv.shell_cuts needs galaxy.distance_mpc for shell radii.")
        return {"shells": 0, "cuts": 0, "by_type": {}}

    target = (cfg.get("catalogs") or {}).get("target_galaxy") or (cfg.get("galaxy") or {}).get("name")
    keep_types = scfg.get("keep_types", (cfg.get("pv") or {}).get("label", {}).get("keep_types", [1, 2, 3]))
    catalog = load_bagetakos_table7(holes_dat, target_galaxy=target, keep_types=keep_types)
    shells, diag = catalog_to_pixel_shells(
        catalog,
        wcs=wcs,
        distance_mpc=float(distance_mpc),
        hv_scale=1.0,
        hv_offset=0.0,
    )
    if diag.get("warnings"):
        print(f"[make_pv] shell_cuts catalog warnings: {len(diag['warnings'])}")

    max_shells = scfg.get("max_shells")
    if max_shells is not None:
        shells = shells[: int(max_shells)]

    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    shells = [s for s in shells if 0 <= s["xc"] < nx and 0 <= s["yc"] < ny]
    if not shells:
        return {"shells": 0, "cuts": 0, "by_type": {}}

    v_axis = velocity_axis_kms(hdr)
    m0 = _moment0(cube)
    grid = cfg["pv"].get("grid", {})
    pa_convention = str(scfg.get("pa_convention", (cfg.get("pv") or {}).get("label", {}).get("catalog_pa_convention", "astro"))).lower()
    galaxy_conv = str(grid.get("pa_convention", "astro")).lower()
    galaxy_pa = float(cfg["galaxy"]["pa_deg"]) + float(grid.get("pa_delta_deg", 0.0))
    galaxy_major, galaxy_minor = unit_vectors_for_pa(galaxy_pa, convention=galaxy_conv)

    orientations = list(scfg.get("orientations", ["shell_major", "shell_minor", "galaxy_major", "galaxy_minor"]))
    offset_fracs = [float(x) for x in scfg.get("offset_fractions", [-0.5, 0.0, 0.5])]
    length_scale = float(scfg.get("length_scale_radius", 2.5))
    min_half_len = float(scfg.get("min_half_length_pix", 12.0))
    max_half_len = scfg.get("max_half_length_pix")
    max_half_len = None if max_half_len is None else float(max_half_len)
    slit_width = int(scfg.get("slit_width_pix", grid.get("slit_width_pix", 3)))
    pos_step = float(scfg.get("pos_step_pix", grid.get("pos_step_pix", 1.0)))
    qa_max = scfg.get("qa_max_plots", 40)
    qa_max = None if qa_max is None else int(qa_max)

    written = 0
    by_type = {}
    for shell in shells:
        shell_major, shell_minor = unit_vectors_for_pa(float(shell["pa_deg"]), convention=pa_convention)
        for orient in orientations:
            try:
                dvec, pvec = _orientation_vectors(orient, shell, galaxy_major, galaxy_minor, pa_convention)
            except ValueError as exc:
                print(f"[make_pv] WARNING: {exc}")
                continue
            dx, dy = dvec
            px, py = pvec
            half_len = _ellipse_support_radius(
                (dx, dy), shell_major, shell_minor, float(shell["a_pix"]), float(shell["b_pix"])
            ) * length_scale
            half_len = max(half_len, min_half_len)
            if max_half_len is not None:
                half_len = min(half_len, max_half_len)
            perp_radius = _ellipse_support_radius(
                (px, py), shell_major, shell_minor, float(shell["a_pix"]), float(shell["b_pix"])
            )

            for off in offset_fracs:
                x0 = float(shell["xc"]) + off * perp_radius * px
                y0 = float(shell["yc"]) + off * perp_radius * py
                off_tag = f"{off:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
                stem = f"cat_T{shell['type']}_S{int(shell['shell_id']):04d}_{orient}_o{off_tag}"
                qa_enabled = qa_max is None or written < qa_max
                ok = _save_line_pv(
                    stem_base=stem,
                    cube=cube,
                    hdr=hdr,
                    cfg=cfg,
                    out_dir=out_dir,
                    overlays_dir=overlays_dir,
                    m0=m0,
                    v_axis=v_axis,
                    x0=x0,
                    y0=y0,
                    dir_x=dx,
                    dir_y=dy,
                    perp_x=px,
                    perp_y=py,
                    half_len_pix=half_len,
                    slit_width_pix=slit_width,
                    pos_step_pix=pos_step,
                    qa_enabled=qa_enabled,
                    meta_extra={
                        "type": "catalog_shell",
                        "source": "catalog_augmented",
                        "frame": "shell_catalog",
                        "orientation": orient,
                        "offset_fraction": float(off),
                        "offset_pix": float(off * perp_radius),
                        "target_shell_id": int(shell["shell_id"]),
                        "target_shell_type": int(shell["type"]),
                        "target_shell_center_pix": [float(shell["xc"]), float(shell["yc"])],
                        "target_shell_radius_pix": [float(shell["a_pix"]), float(shell["b_pix"])],
                        "target_shell_pa_deg": float(shell["pa_deg"]),
                        "pa_convention": pa_convention,
                        "length_scale_radius": float(length_scale),
                    },
                )
                if ok:
                    written += 1
                    by_type[int(shell["type"])] = by_type.get(int(shell["type"]), 0) + 1

    print(
        f"[make_pv] shell_cuts: wrote {written} catalog-augmented cuts "
        f"from {len(shells)} shells; by_type={by_type}"
    )
    return {"shells": len(shells), "cuts": written, "by_type": by_type}

def _load_catalog_pixel_shells_for_cfg(cfg, wcs):
    holes_dat = (cfg.get("catalogs") or {}).get("holes_dat")
    distance_mpc = (cfg.get("galaxy") or {}).get("distance_mpc")
    if not holes_dat or not Path(holes_dat).exists() or distance_mpc is None:
        return [], {}
    target = (cfg.get("catalogs") or {}).get("target_galaxy") or (cfg.get("galaxy") or {}).get("name")
    keep_types = (cfg.get("pv") or {}).get("label", {}).get("keep_types", [1, 2, 3])
    catalog = load_bagetakos_table7(holes_dat, target_galaxy=target, keep_types=keep_types)
    return catalog_to_pixel_shells(
        catalog,
        wcs=wcs,
        distance_mpc=float(distance_mpc),
        hv_scale=1.0,
        hv_offset=0.0,
    )

def _far_from_shells(x, y, shells, min_sep_pix, sep_radius_factor):
    for sh in shells:
        radius = max(float(sh.get("a_pix", 0.0)), float(sh.get("b_pix", 0.0)))
        safe = max(float(min_sep_pix), float(sep_radius_factor) * radius)
        if (x - float(sh["xc"])) ** 2 + (y - float(sh["yc"])) ** 2 < safe ** 2:
            return False
    return True

def make_background_negative_pv(cube, hdr, wcs, cfg, out_dir: Path, overlays_dir: Path):
    """Generate random background cuts inside the HI disk and away from catalog shells."""
    neg = (cfg.get("pv") or {}).get("negatives", {})
    if not neg.get("enabled", False):
        return {"cuts": 0}

    shells, diag = _load_catalog_pixel_shells_for_cfg(cfg, wcs)
    if diag.get("warnings"):
        print(f"[make_pv] negatives catalog warnings: {len(diag['warnings'])}")

    n_shells = max(1, len(shells))
    n_target = int(neg.get("n_per_shell", 12)) * n_shells
    max_total = neg.get("max_total")
    if max_total is not None:
        n_target = min(n_target, int(max_total))
    if n_target <= 0:
        return {"cuts": 0}

    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    v_axis = velocity_axis_kms(hdr)
    m0 = _moment0(cube)
    finite = m0[np.isfinite(m0)]
    if finite.size == 0:
        return {"cuts": 0}

    galaxy_pct = float(neg.get("galaxy_mask_percentile", 45.0))
    upper_pct = float(neg.get("avoid_brightest_percentile", 99.5))
    lo = np.percentile(finite, galaxy_pct)
    hi = np.percentile(finite, upper_pct)
    mask = (m0 >= lo) & (m0 <= hi)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return {"cuts": 0}

    ax_as, ay_as = pixel_scales_arcsec(wcs)
    pix_per_as = 1.0 / max(ax_as, ay_as, 1e-9)
    min_sep_pix = float(neg.get("min_sep_arcsec", 20.0)) * pix_per_as
    sep_radius_factor = float(neg.get("sep_radius_factor", 1.5))
    half_len = float(neg.get("half_length_pix", 64.0))
    slit_width = int(neg.get("slit_width_pix", (cfg.get("pv") or {}).get("grid", {}).get("slit_width_pix", 3)))
    pos_step = float(neg.get("pos_step_pix", (cfg.get("pv") or {}).get("grid", {}).get("pos_step_pix", 1.0)))
    qa_max = neg.get("qa_max_plots", 30)
    qa_max = None if qa_max is None else int(qa_max)
    seed = int(neg.get("seed", 12345))
    max_attempts = int(neg.get("max_attempts_factor", 100)) * n_target
    rng = np.random.default_rng(seed)

    written = 0
    attempts = 0
    while written < n_target and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, xs.size))
        x0 = float(xs[i])
        y0 = float(ys[i])
        if not _far_from_shells(x0, y0, shells, min_sep_pix, sep_radius_factor):
            continue

        theta = float(rng.uniform(0.0, np.pi))
        dx, dy = np.cos(theta), np.sin(theta)
        px, py = -dy, dx
        stem = f"neg_bg_{written:05d}"
        ok = _save_line_pv(
            stem_base=stem,
            cube=cube,
            hdr=hdr,
            cfg=cfg,
            out_dir=out_dir,
            overlays_dir=overlays_dir,
            m0=m0,
            v_axis=v_axis,
            x0=x0,
            y0=y0,
            dir_x=dx,
            dir_y=dy,
            perp_x=px,
            perp_y=py,
            half_len_pix=half_len,
            slit_width_pix=slit_width,
            pos_step_pix=pos_step,
            qa_enabled=qa_max is None or written < qa_max,
            meta_extra={
                "type": "background_negative",
                "source": "random_background",
                "frame": "image",
                "theta_rad": theta,
                "min_sep_pix": float(min_sep_pix),
                "sep_radius_factor": float(sep_radius_factor),
            },
        )
        if ok:
            written += 1

    print(
        f"[make_pv] negatives: wrote {written}/{n_target} random background cuts "
        f"(attempts={attempts}, shells={len(shells)})"
    )
    return {"cuts": written, "target": n_target, "attempts": attempts}

def main(cfg):
    outdir = Path(cfg["output_root"]) / "pv"
    outdir.mkdir(parents=True, exist_ok=True)
    overlays = Path(cfg["output_root"]) / "qa_pv"
    overlays.mkdir(parents=True, exist_ok=True)

    cube, hdr, wcs, _ = open_cube(cfg["cube_path"])

    counts = make_grid_pv(cube, hdr, wcs, cfg, outdir, overlays)
    shell_counts = make_catalog_shell_pv(cube, hdr, wcs, cfg, outdir, overlays)
    negative_counts = make_background_negative_pv(cube, hdr, wcs, cfg, outdir, overlays)

    if isinstance(counts, int):
        counts = {"major": 0, "minor": 0, "total": counts}

    print(
        f"[make_pv] wrote {counts.get('major',0)} major-axis and "
        f"{counts.get('minor',0)} minor-axis cuts "
        f"(total={counts.get('total', counts.get('major',0)+counts.get('minor',0))}) to {outdir}"
    )
    if shell_counts.get("cuts", 0):
        print(
            f"[make_pv] wrote {shell_counts['cuts']} catalog-augmented shell cuts "
            f"from {shell_counts['shells']} shells"
        )
    if negative_counts.get("cuts", 0):
        print(f"[make_pv] wrote {negative_counts['cuts']} random background negative cuts")
    print(f"[make_pv] overlays saved to {overlays}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    # sanity: require PA & INC for meaningful major/minor axes
    if cfg["galaxy"].get("pa_deg") is None and str(cfg["pv"]["grid"].get("frame","galaxy")).lower()=="galaxy":
        raise SystemExit("galaxy.pa_deg must be set in YAML for grid slicing with frame='galaxy'.")
    if "pv" not in cfg or "grid" not in cfg["pv"]:
        raise SystemExit("pv.grid block missing in YAML.")
    main(cfg)
