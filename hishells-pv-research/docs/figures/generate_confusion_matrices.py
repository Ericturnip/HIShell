"""
Generate the patch-level and pixel-level confusion matrices used in the report.
The script reads saved evaluation JSON files, rebuilds the count matrices, and
exports matching PNG/PDF figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "docs" / "results" / "clean_physical_baseline"
OUT = Path(__file__).resolve().parent
THRESHOLD = "0.075"
PATCH_SHAPE = (96, 256)
SPLITS = ("val", "test", "stress")


def load_counts(metric_key: str) -> dict[str, int]:
    """Combine confusion counts across validation, test, and stress splits."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}
    pixels_per_patch = PATCH_SHAPE[0] * PATCH_SHAPE[1]
    for split in SPLITS:
        path = RUN_DIR / f"eval_{split}_high_recall_model.json"
        data = json.loads(path.read_text())
        metrics = data[metric_key][THRESHOLD]
        tp = int(metrics["tp"])
        fp = int(metrics["fp"])
        fn = int(metrics["fn"])
        if metric_key == "patch_detection_metrics":
            total = int(data["patches_seen"])
        else:
            total = int(data["patches_seen"]) * pixels_per_patch
        tn = total - tp - fp - fn
        if tn < 0:
            raise ValueError(f"Negative TN for {split}/{metric_key}: {tn}")
        counts["tp"] += tp
        counts["fp"] += fp
        counts["fn"] += fn
        counts["tn"] += tn
        counts["total"] += total
    return counts


def as_matrix(counts: dict[str, int]) -> np.ndarray:
    """Arrange counts as true-label rows and predicted-label columns."""
    return np.asarray([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=np.int64)


def counts_from_metrics(metrics: dict[str, int | float]) -> dict[str, int]:
    """Normalize one saved metrics block into the shared count dictionary."""
    return {
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tn": int(metrics["tn"]),
        "total": int(metrics["tp"]) + int(metrics["fp"]) + int(metrics["fn"]) + int(metrics["tn"]),
    }


def format_count(value: int) -> str:
    """Format large count annotations so they fit inside matrix cells."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def plot_confusion(matrix: np.ndarray, title: str, out_stem: str) -> tuple[Path, Path]:
    """Draw one paper-style confusion matrix and save PNG/PDF outputs."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sums, 1)
    recall = matrix[1, 1] / max(1, matrix[1].sum())
    precision = matrix[1, 1] / max(1, matrix[:, 1].sum())
    specificity = matrix[0, 0] / max(1, matrix[0].sum())

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
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

    fig, ax = plt.subplots(figsize=(3.9, 3.4), constrained_layout=True)
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["negative", "positive"])
    ax.set_yticks([0, 1], labels=["negative", "positive"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")

    for i in range(2):
        for j in range(2):
            color = "white" if normalized[i, j] > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{normalized[i, j]:.3f}\n{format_count(int(matrix[i, j]))}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    ax.text(
        0.5,
        -0.34,
        f"threshold={THRESHOLD}; recall={recall:.3f}; precision={precision:.3f}; specificity={specificity:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row-normalized fraction")

    png = OUT / f"{out_stem}.png"
    pdf = OUT / f"{out_stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    """Generate all confusion-matrix figures and their JSON summary."""
    OUT.mkdir(parents=True, exist_ok=True)
    patch_counts = load_counts("patch_detection_metrics")
    pixel_counts = load_counts("pixel_threshold_metrics")
    patch_matrix = as_matrix(patch_counts)
    pixel_matrix = as_matrix(pixel_counts)
    ddo53_path = RUN_DIR / "ddo53_physical_galaxy_grid_shell_recall.json"
    ddo53 = json.loads(ddo53_path.read_text())
    ddo53_patch_counts = counts_from_metrics(ddo53["patch"])
    ddo53_pixel_counts = counts_from_metrics(ddo53["pixel"])
    ddo53_patch_matrix = as_matrix(ddo53_patch_counts)
    ddo53_pixel_matrix = as_matrix(ddo53_pixel_counts)

    outputs = []
    outputs.extend(
        plot_confusion(
            patch_matrix,
            "Patch-level confusion matrix",
            "fig08_patch_level_confusion_matrix",
        )
    )
    outputs.extend(
        plot_confusion(
            pixel_matrix,
            "Pixel-level confusion matrix",
            "fig09_pixel_level_confusion_matrix",
        )
    )
    outputs.extend(
        plot_confusion(
            ddo53_patch_matrix,
            "DDO 53 physical-grid patch matrix",
            "fig10_ddo53_physical_grid_patch_confusion_matrix",
        )
    )
    outputs.extend(
        plot_confusion(
            ddo53_pixel_matrix,
            "DDO 53 physical-grid pixel matrix",
            "fig11_ddo53_physical_grid_pixel_confusion_matrix",
        )
    )

    summary = {
        "threshold": float(THRESHOLD),
        "splits": list(SPLITS),
        "patch_shape": list(PATCH_SHAPE),
        "patch_counts": patch_counts,
        "patch_matrix_true_rows_pred_cols": patch_matrix.tolist(),
        "pixel_counts": pixel_counts,
        "pixel_matrix_true_rows_pred_cols": pixel_matrix.tolist(),
        "ddo53_physical_grid": {
            "source": str(ddo53_path),
            "fine_grid_rows": int(ddo53["fine_grid_rows"]),
            "shell_level_grid_coverage": float(ddo53["shell_level_grid_coverage"]),
            "shell_level_detection_recall": float(ddo53["shell_level_detection_recall"]),
            "patch_counts": ddo53_patch_counts,
            "patch_matrix_true_rows_pred_cols": ddo53_patch_matrix.tolist(),
            "pixel_counts": ddo53_pixel_counts,
            "pixel_matrix_true_rows_pred_cols": ddo53_pixel_matrix.tolist(),
        },
    }
    summary_path = OUT / "confusion_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    for out in outputs:
        print(out)
    print(summary_path)


if __name__ == "__main__":
    main()
