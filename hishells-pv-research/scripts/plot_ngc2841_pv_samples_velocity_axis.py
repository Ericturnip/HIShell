#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_meta(pv_path: Path) -> dict:
    meta_path = pv_path.with_suffix(".json")
    with meta_path.open("r") as f:
        return json.load(f)


def _asinh_stretch(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.nanpercentile(finite, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return np.arcsinh(6.0 * scaled) / np.arcsinh(6.0)


def _spatial_axis_kpc(meta: dict, npos: int) -> np.ndarray:
    pos_axis = np.asarray(meta.get("pos_axis_pix", []), dtype=np.float32)
    if pos_axis.size == npos:
        pixel_scale = float(meta.get("pixel_scale_arcsec", 1.0))
        distance = float(meta.get("adopted_distance_mpc", 0.0))
        kpc_per_pix = pixel_scale * distance * 1000.0 / 206265.0
        return pos_axis * kpc_per_pix
    window = float(meta.get("spatial_window_kpc", 5.0))
    return np.linspace(-0.5 * window, 0.5 * window, npos, dtype=np.float32)


def _image_extent(meta: dict, pv: np.ndarray) -> tuple[float, float, float, float]:
    nv, npos = pv.shape
    x = _spatial_axis_kpc(meta, npos)
    if x.size > 1:
        dx = float(np.nanmedian(np.diff(x)))
    else:
        dx = float(meta.get("spatial_window_kpc", 5.0))
    x0 = float(x[0] - 0.5 * dx)
    x1 = float(x[-1] + 0.5 * dx)

    v = np.asarray(meta.get("velocity_axis_kms", []), dtype=np.float32)
    if v.size == nv:
        dv = float(np.nanmedian(np.diff(v))) if nv > 1 else float(meta.get("velocity_bin_width_kms", 1.0))
        y0 = float(v[0] - 0.5 * dv)
        y1 = float(v[-1] + 0.5 * dv)
        return x0, x1, y0, y1

    vmin = float(meta.get("target_velocity_min_kms", meta.get("velocity_min_kms", 0.0)))
    vmax = float(meta.get("target_velocity_max_kms", meta.get("velocity_max_kms", nv)))
    return x0, x1, vmin, vmax


def _draw_velocity_markers(ax: plt.Axes, meta: dict, catalog_hv: float | None) -> None:
    local_v = meta.get("local_velocity_kms")
    if local_v is not None and np.isfinite(float(local_v)):
        ax.axhline(float(local_v), color="#00bcd4", lw=1.4, ls="-", label="local M1")
    requested_min = meta.get("requested_velocity_min_kms")
    requested_max = meta.get("requested_velocity_max_kms")
    if requested_min is not None and requested_max is not None:
        ax.axhline(float(requested_min), color="#fdd835", lw=0.9, ls=":", label="requested +/-100")
        ax.axhline(float(requested_max), color="#fdd835", lw=0.9, ls=":")
    if catalog_hv is None:
        catalog_hv = meta.get("catalog_hv_kms")
    if catalog_hv is not None and np.isfinite(float(catalog_hv)):
        ax.axhline(float(catalog_hv), color="#ff4db8", lw=1.2, ls="--", label="catalog HV")


def _panel_text(meta: dict, row: pd.Series) -> str:
    catalog_hv = row.get("catalog_hv_kms")
    try:
        catalog_hv_text = f"{float(catalog_hv):.1f}"
    except (TypeError, ValueError):
        catalog_hv_text = "n/a"
    return "\n".join(
        [
            f"{meta.get('galaxy', row.get('galaxy', 'ngc_2841'))}  {meta.get('cut_category', row.get('cut_category', ''))}",
            f"shell={meta.get('source_shell_id', row.get('shell_id', 'n/a'))}  positive={row.get('positive', 'n/a')}  mask_px={row.get('mask_pixels', 'n/a')}",
            (
                "target window "
                f"{float(meta.get('target_velocity_min_kms', meta.get('velocity_min_kms'))):.1f}"
                " to "
                f"{float(meta.get('target_velocity_max_kms', meta.get('velocity_max_kms'))):.1f} km/s"
            ),
            f"local M1={float(meta.get('local_velocity_kms')):.1f} km/s  catalog HV={catalog_hv_text} km/s",
            f"edge policy={meta.get('velocity_edge_policy', 'n/a')}  dv={float(meta.get('velocity_bin_width_kms', np.nan)):.3f} km/s/bin",
        ]
    )


def plot_sample(row: pd.Series, data_dir: Path, out_dir: Path, index: int) -> dict:
    source = str(row.get("source_filename") or row.get("filename"))
    pv_path = data_dir / "pv" / source
    if not pv_path.exists() and Path(source).exists():
        pv_path = Path(source)
    meta = _load_meta(pv_path)
    pv = np.load(pv_path)

    mask_path = data_dir / "labels" / source
    if mask_path.exists():
        mask = np.load(mask_path)
    else:
        mask = np.zeros_like(pv, dtype=np.uint8)

    if mask.shape != pv.shape:
        mask = np.zeros_like(pv, dtype=np.uint8)

    extent = _image_extent(meta, pv)
    catalog_hv = row.get("catalog_hv_kms")
    try:
        catalog_hv = float(catalog_hv)
    except (TypeError, ValueError):
        catalog_hv = None

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]},
    )

    stretched = _asinh_stretch(pv)
    im0 = axes[0].imshow(stretched, origin="lower", aspect="auto", cmap="gray", extent=extent)
    if np.any(mask > 0):
        axes[0].contour(mask > 0, levels=[0.5], colors=["#00ff8a"], linewidths=1.1, extent=extent, origin="lower")
    _draw_velocity_markers(axes[0], meta, catalog_hv)
    axes[0].set_title("PV intensity + label contour")
    axes[0].set_xlabel("Spatial offset along cut (kpc)")
    axes[0].set_ylabel("Velocity (km/s)")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.75)

    axes[1].imshow(mask, origin="lower", aspect="auto", cmap="Greens", vmin=0, vmax=1, extent=extent)
    _draw_velocity_markers(axes[1], meta, catalog_hv)
    axes[1].set_title("Ground-truth mask")
    axes[1].set_xlabel("Spatial offset along cut (kpc)")
    axes[1].set_ylabel("Velocity (km/s)")

    axes[2].imshow(stretched, origin="lower", aspect="auto", cmap="gray", extent=extent)
    if np.any(mask > 0):
        overlay = np.ma.masked_where(mask <= 0, mask)
        axes[2].imshow(overlay, origin="lower", aspect="auto", cmap="spring", alpha=0.38, extent=extent)
    _draw_velocity_markers(axes[2], meta, catalog_hv)
    axes[2].set_title("PV + transparent mask")
    axes[2].set_xlabel("Spatial offset along cut (kpc)")
    axes[2].set_ylabel("Velocity (km/s)")

    fig.suptitle(_panel_text(meta, row), fontsize=11)
    filename = f"{index:03d}_{row.get('sample_group', 'sample')}_{pv_path.stem}.png"
    safe_filename = filename.replace("/", "_").replace(" ", "_")
    out_path = out_dir / safe_filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    out_row = dict(row)
    out_row.update(
        {
            "filename": str(out_path),
            "source_filename": pv_path.name,
            "target_velocity_min_kms": meta.get("target_velocity_min_kms", meta.get("velocity_min_kms")),
            "target_velocity_max_kms": meta.get("target_velocity_max_kms", meta.get("velocity_max_kms")),
            "requested_velocity_min_kms": meta.get("requested_velocity_min_kms"),
            "requested_velocity_max_kms": meta.get("requested_velocity_max_kms"),
            "local_velocity_kms": meta.get("local_velocity_kms"),
            "velocity_axis_first_row_kms": float(np.asarray(meta.get("velocity_axis_kms"))[0]),
            "velocity_axis_last_row_kms": float(np.asarray(meta.get("velocity_axis_kms"))[-1]),
        }
    )
    return out_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NGC 2841 QA PV samples with actual velocity axes.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("training_data/standardized_5kpc_200kms_resampled96_no3031"),
    )
    parser.add_argument(
        "--input-index",
        type=Path,
        default=None,
        help="Existing ngc2841 sample index. Defaults to qa_ngc2841_pv_samples/ngc2841_pv_samples_index.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to qa_ngc2841_pv_samples_velocity_axis.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    input_index = args.input_index or data_dir / "qa_ngc2841_pv_samples" / "ngc2841_pv_samples_index.csv"
    out_dir = args.out_dir or data_dir / "qa_ngc2841_pv_samples_velocity_axis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(input_index)
    out_rows = []
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        out_rows.append(plot_sample(row, data_dir, out_dir, i))

    out_index = out_dir / "ngc2841_pv_samples_velocity_axis_index.csv"
    pd.DataFrame(out_rows).to_csv(out_index, index=False)
    print(f"Wrote {len(out_rows)} panels to {out_dir}")
    print(f"Wrote index to {out_index}")


if __name__ == "__main__":
    main()
