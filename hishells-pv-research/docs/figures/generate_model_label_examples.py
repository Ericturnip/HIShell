"""
Generate example panels comparing catalog labels with U-Net probabilities.
The selected panels show both good pixel overlap and patch-level hits with messy
pixel overlap for shell types 1, 2, and 3.
"""

from __future__ import annotations

import json
import os
import argparse
import csv
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

plt = None


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "training_data" / "standardized_5kpc_200kms_clean_physical_baseline"
RUN_DIR = ROOT / "docs" / "results" / "clean_physical_baseline"
MODEL_PATH = RUN_DIR / "high_recall_model.keras"
OUT = ROOT / "docs" / "model_label_examples"

THRESHOLD = 0.075
TYPE_NAMES = {
    1: "type 1: both sides stalled",
    2: "type 2: one side expanding",
    3: "type 3: both sides expanding",
}
TYPE_COLORS = {
    1: "tab:orange",
    2: "tab:blue",
    3: "tab:green",
}


def setup() -> None:
    """Prepare the output directory and shared Matplotlib style."""
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def ensure_matplotlib():
    """Import Matplotlib lazily so score-only runs do not need plotting setup."""
    global plt
    if plt is None:
        import matplotlib.pyplot as _plt

        plt = _plt
    return plt


def zscore_finite(pv: np.ndarray) -> np.ndarray:
    """Apply the same finite-value z-score normalization used at evaluation time."""
    finite = pv[np.isfinite(pv)]
    if finite.size == 0:
        return np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mu = float(np.mean(finite))
    sigma = float(np.std(finite) + 1e-6)
    return np.nan_to_num((pv - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def robust_display(pv: np.ndarray) -> np.ndarray:
    """Scale PV intensity values into a display range for grayscale panels."""
    finite = pv[np.isfinite(pv)]
    if finite.size == 0:
        return np.zeros_like(pv, dtype=float)
    lo, hi = np.nanpercentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return np.clip((pv - lo) / (hi - lo), 0, 1)


def read_rows() -> list[dict]:
    """Read held-out positive patches from validation, test, and stress manifests."""
    rows: list[dict] = []
    for split in ["val", "test", "stress"]:
        with (DATA_ROOT / f"{split}_manifest.csv").open(newline="") as fh:
            for row in csv.DictReader(fh):
                if int(float(row.get("positive", 0) or 0)) != 1:
                    continue
                row["split"] = split
                rows.append(row)
    return rows


def shell_ids_for_type(label_sidecar: Path, shell_type: int) -> str:
    """List catalog shell ids of one type inside a selected patch."""
    try:
        sidecar = json.loads(label_sidecar.read_text())
    except Exception:
        return ""
    ids = [str(obj.get("shell_id")) for obj in sidecar.get("objects", []) if int(obj.get("type", 0) or 0) == shell_type]
    return ",".join(sorted(set(ids)))


def sample_metrics(row: dict, prob: np.ndarray) -> list[dict]:
    """Compute per-shell-type overlap metrics for one probability map."""
    type_mask = np.load(DATA_ROOT / "label_types" / row["filename"])
    pred = prob >= THRESHOLD
    out: list[dict] = []
    for shell_type in [1, 2, 3]:
        truth = type_mask == shell_type
        label_pixels = int(truth.sum())
        if label_pixels == 0:
            continue
        tp = int(np.logical_and(pred, truth).sum())
        fp = int(np.logical_and(pred, ~truth).sum())
        fn = int(np.logical_and(~pred, truth).sum())
        pixel_precision = tp / max(1, tp + fp)
        pixel_recall = tp / max(1, tp + fn)
        dice = 2 * tp / max(1, 2 * tp + fp + fn)
        patch_hit = bool(float(np.nanmax(prob)) >= THRESHOLD)
        single_type = int(len(set(np.unique(type_mask[type_mask > 0]).astype(int))) == 1)
        out.append(
            {
                "split": row["split"],
                "galaxy": row["galaxy"],
                "filename": row["filename"],
                "cut_category": row.get("cut_category", ""),
                "manifest_shell_id": row.get("shell_id", ""),
                "shell_type": shell_type,
                "shell_ids_for_type": shell_ids_for_type(DATA_ROOT / "labels" / f"{Path(row['filename']).stem}.json", shell_type),
                "label_pixels": label_pixels,
                "single_label_type_in_patch": single_type,
                "max_probability": float(np.nanmax(prob)),
                "mean_probability_on_label": float(np.nanmean(prob[truth])),
                "integrated_probability_on_label": float(np.nansum(prob[truth])),
                "predicted_pixels": int(pred.sum()),
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                "pixel_precision": pixel_precision,
                "pixel_recall": pixel_recall,
                "dice": dice,
                "patch_hit": int(patch_hit),
            }
        )
    return out


def score_examples(metrics: list[dict]) -> dict[tuple[int, str], dict]:
    """Select one clean and one messy patch-level hit for each shell type."""
    selected: dict[tuple[int, str], dict] = {}
    for shell_type in [1, 2, 3]:
        type_rows = [m for m in metrics if int(m["shell_type"]) == shell_type and int(m["patch_hit"]) == 1]
        if not type_rows:
            continue
        good_pool = [m for m in type_rows if float(m["pixel_recall"]) >= 0.45] or type_rows
        for m in good_pool:
            m["good_score"] = (
                3.0 * float(m["dice"])
                + 1.5 * float(m["pixel_recall"])
                + 0.6 * float(m["pixel_precision"])
                + 0.25 * int(m["single_label_type_in_patch"])
                + 0.0002 * int(m["label_pixels"])
            )
        selected[(shell_type, "good_pixel_overlap")] = sorted(good_pool, key=lambda m: float(m["good_score"]), reverse=True)[0]

        dice_values = np.asarray([float(m["dice"]) for m in type_rows], dtype=float)
        dice_cut = max(0.35, float(np.quantile(dice_values, 0.35)))
        poor_pool = [
            m
            for m in type_rows
            if float(m["max_probability"]) >= 0.20 and float(m["pixel_recall"]) > 0.0 and float(m["dice"]) <= dice_cut
        ]
        if not poor_pool:
            poor_pool = sorted(type_rows, key=lambda m: float(m["dice"]))[: max(10, min(50, len(type_rows)))]
        for m in poor_pool:
            m["poor_score"] = (
                1.8 * float(m["max_probability"])
                + 0.8 * float(m["pixel_recall"])
                - 2.5 * float(m["dice"])
                - 0.4 * float(m["pixel_precision"])
            )
        selected[(shell_type, "patch_hit_messy_pixels")] = sorted(poor_pool, key=lambda m: float(m["poor_score"]), reverse=True)[0]
    return selected


def extent_for(row: dict, pv: np.ndarray) -> tuple[list[float], str]:
    """Return physical plot extents from the PV metadata sidecar."""
    meta_path = DATA_ROOT / "pv" / f"{Path(row['filename']).stem}.json"
    meta = json.loads(meta_path.read_text())
    v_axis = np.asarray(meta.get("velocity_axis_kms", np.arange(pv.shape[0])), dtype=float)
    spatial = float(meta.get("spatial_window_kpc", 5.0))
    extent = [0, spatial, float(v_axis[-1]), float(v_axis[0])]
    return extent, f"{spatial:g} kpc x {float(meta.get('velocity_window_kms', 200.0)):g} km/s"


def contour_array(ax, mask: np.ndarray, *, extent: list[float], **kwargs) -> None:
    """Draw label or prediction contours in the same orientation as the PV image."""
    # imshow(origin="upper") places row 0 at the top velocity. contour maps
    # row 0 to the lower extent by default, so flip only for display.
    ax.contour(np.flipud(mask.astype(float)), levels=[0.5], extent=extent, **kwargs)


def plot_example(row: dict, quality: str, prob: np.ndarray) -> tuple[Path, Path]:
    """Draw one three-panel label, probability, and overlay example."""
    ensure_matplotlib()
    pv = np.load(DATA_ROOT / "pv" / row["filename"])
    type_mask = np.load(DATA_ROOT / "label_types" / row["filename"])
    shell_type = int(row["shell_type"])
    truth = type_mask == shell_type
    pred = prob >= THRESHOLD
    extent, physical_window = extent_for(row, pv)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    axes[0].imshow(robust_display(pv), cmap="gray_r", aspect="auto", extent=extent, interpolation="nearest", origin="upper")
    contour_array(axes[0], truth, extent=extent, colors=[TYPE_COLORS[shell_type]], linewidths=1.0)
    axes[0].set_title("catalog label")

    im = axes[1].imshow(
        prob,
        cmap="magma",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        origin="upper",
        vmin=0,
        vmax=max(0.25, float(np.nanpercentile(prob, 99))),
    )
    contour_array(axes[1], truth, extent=extent, colors=["cyan"], linewidths=0.8)
    axes[1].set_title("model probability")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cbar.set_label("probability")

    axes[2].imshow(robust_display(pv), cmap="gray_r", aspect="auto", extent=extent, interpolation="nearest", origin="upper")
    contour_array(axes[2], truth, extent=extent, colors=[TYPE_COLORS[shell_type]], linewidths=1.2)
    if np.any(pred):
        contour_array(axes[2], pred, extent=extent, colors=["tab:red"], linewidths=1.0)
    axes[2].set_title("label vs prediction")
    axes[2].plot([], [], color=TYPE_COLORS[shell_type], label="label")
    axes[2].plot([], [], color="tab:red", label=f"pred. >= {THRESHOLD:g}")
    axes[2].legend(frameon=False, loc="lower right")

    for ax in axes:
        ax.set_xlabel("position (kpc)")
    axes[0].set_ylabel("velocity (km s$^{-1}$)")

    quality_title = "good pixel overlap" if quality == "good_pixel_overlap" else "patch hit, messy pixels"
    shell_ids = row.get("shell_ids_for_type") or row.get("manifest_shell_id")
    fig.suptitle(
        f"{TYPE_NAMES[shell_type]} | {row['split']} | {row['galaxy']} shell {shell_ids} | {quality_title}\n"
        f"Dice={float(row['dice']):.3f}, pixel recall={float(row['pixel_recall']):.3f}, "
        f"pixel precision={float(row['pixel_precision']):.3f}, "
        f"max P={float(row['max_probability']):.3f} | {physical_window}",
        y=1.12,
        fontsize=9,
    )
    stem = f"type{shell_type}_{quality}_{row['split']}_{row['galaxy']}_{Path(row['filename']).stem}"
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_grid(selected_rows: list[dict], probs: dict[str, np.ndarray]) -> tuple[Path, Path]:
    """Draw the compact six-row overview of all selected shell examples."""
    ensure_matplotlib()
    fig, axes = plt.subplots(6, 3, figsize=(7.4, 12.4), constrained_layout=True)
    for r, row in enumerate(selected_rows):
        prob = probs[str(row["filename"])]
        pv = np.load(DATA_ROOT / "pv" / row["filename"])
        type_mask = np.load(DATA_ROOT / "label_types" / row["filename"])
        shell_type = int(row["shell_type"])
        truth = type_mask == shell_type
        pred = prob >= THRESHOLD
        extent, _ = extent_for(row, pv)
        quality_title = "good pixels" if row["quality"] == "good_pixel_overlap" else "patch hit / poor pixels"
        titles = [
            f"{TYPE_NAMES[shell_type]}\n{quality_title}",
            "probability",
            f"Dice {float(row['dice']):.2f}, recall {float(row['pixel_recall']):.2f}",
        ]
        axes[r, 0].imshow(robust_display(pv), cmap="gray_r", aspect="auto", extent=extent, interpolation="nearest", origin="upper")
        contour_array(axes[r, 0], truth, extent=extent, colors=[TYPE_COLORS[shell_type]], linewidths=0.9)
        axes[r, 1].imshow(
            prob,
            cmap="magma",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            origin="upper",
            vmin=0,
            vmax=max(0.25, float(np.nanpercentile(prob, 99))),
        )
        contour_array(axes[r, 1], truth, extent=extent, colors=["cyan"], linewidths=0.7)
        axes[r, 2].imshow(robust_display(pv), cmap="gray_r", aspect="auto", extent=extent, interpolation="nearest", origin="upper")
        contour_array(axes[r, 2], truth, extent=extent, colors=[TYPE_COLORS[shell_type]], linewidths=0.9)
        if np.any(pred):
            contour_array(axes[r, 2], pred, extent=extent, colors=["tab:red"], linewidths=0.8)
        for c in range(3):
            axes[r, c].set_title(titles[c], fontsize=8)
            axes[r, c].tick_params(labelsize=6)
            if r == 5:
                axes[r, c].set_xlabel("position (kpc)", fontsize=7)
            if c == 0:
                axes[r, c].set_ylabel("velocity\n(km s$^{-1}$)", fontsize=7)
    png = OUT / "all_type_examples_grid.png"
    pdf = OUT / "all_type_examples_grid.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def score_examples_with_model() -> None:
    """Run the saved model on held-out positives and save selected examples."""
    import tensorflow as tf

    OUT.mkdir(parents=True, exist_ok=True)
    tf.get_logger().setLevel("ERROR")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    rows = read_rows()
    print(f"Scoring {len(rows)} held-out positive PV cuts...")
    all_metrics: list[dict] = []
    probs_by_file: dict[str, np.ndarray] = {}
    batch_size = 64
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        xs = []
        kept = []
        for row in batch:
            pv = np.load(DATA_ROOT / "pv" / row["filename"])
            xs.append(zscore_finite(pv)[..., None])
            kept.append(row)
        pred = model.predict(np.stack(xs, axis=0).astype(np.float32), verbose=0, batch_size=batch_size)[..., 0]
        for row, prob in zip(kept, pred):
            probs_by_file[str(row["filename"])] = prob.astype(np.float32)
            all_metrics.extend(sample_metrics(row, prob))
        if start and start % 1024 == 0:
            print(f"  processed {start}/{len(rows)}")

    metrics_path = OUT / "selected_example_scoring_table.csv"
    write_csv(all_metrics, metrics_path)
    selected = score_examples(all_metrics)

    prob_dir = OUT / "selected_probabilities"
    prob_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    for shell_type in [1, 2, 3]:
        for quality in ["good_pixel_overlap", "patch_hit_messy_pixels"]:
            row = selected[(shell_type, quality)].copy()
            row["quality"] = quality
            selected_rows.append(row)
            np.save(prob_dir / f"{Path(str(row['filename'])).stem}.npy", probs_by_file[str(row["filename"])])

    write_csv(selected_rows, OUT / "selected_examples.csv")
    print("Scoring complete:")
    print(metrics_path)
    print(OUT / "selected_examples.csv")
    print(prob_dir)


def plot_selected_examples() -> None:
    """Plot examples from saved selections and probability maps."""
    setup()
    selected_path = OUT / "selected_examples.csv"
    if not selected_path.exists():
        raise SystemExit(f"Missing selected examples table: {selected_path}. Run --score-only first.")
    selected_rows_raw = read_csv(selected_path)
    prob_dir = OUT / "selected_probabilities"
    probs_by_file = {}
    outputs = []
    selected_rows = []
    for raw_row in selected_rows_raw:
        row = dict(raw_row)
        selected_rows.append(row)
        stem = Path(str(row["filename"])).stem
        prob_path = prob_dir / f"{stem}.npy"
        if not prob_path.exists():
            raise SystemExit(f"Missing saved probability map: {prob_path}. Run --score-only first.")
        prob = np.load(prob_path)
        probs_by_file[str(row["filename"])] = prob
        outputs.extend(plot_example(row, str(row["quality"]), prob))
    outputs.extend(plot_grid(selected_rows, probs_by_file))

    notes = OUT / "MODEL_LABEL_EXAMPLE_NOTES.md"
    notes.write_text(
        "# Model vs Original Label Examples\n\n"
        "Each shell type has two held-out examples: one with strong pixel-level overlap "
        "and one where the patch is detected but the predicted pixels are messy.\n\n"
        "Contours: catalog label is colored by shell type; prediction contour is red at threshold 0.075. "
        "The probability panel shows the raw U-Net output.\n\n"
        "Use `all_type_examples_grid.png` for a compact overview. "
        "Use the individual PNG/PDF files for larger slide panels.\n\n"
        "Selected examples are listed in `selected_examples.csv`; all scored candidates are in `selected_example_scoring_table.csv`.\n"
    )
    print("Generated outputs:")
    for out in outputs:
        print(out)
    print(OUT / "selected_examples.csv")
    print(notes)


def write_csv(rows: list[dict], path: Path) -> None:
    """Write selected-example tables with stable column order."""
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    """Read one generated CSV table as dictionaries."""
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    """Score and/or plot the model-label comparison examples."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true", help="Run TensorFlow model scoring and save selected probability maps.")
    ap.add_argument("--plot-only", action="store_true", help="Plot from saved selected_examples.csv and probability maps.")
    args = ap.parse_args()
    if args.score_only and args.plot_only:
        raise SystemExit("Choose only one of --score-only or --plot-only.")
    if args.plot_only:
        plot_selected_examples()
    elif args.score_only:
        score_examples_with_model()
    else:
        score_examples_with_model()
        plot_selected_examples()


if __name__ == "__main__":
    main()
