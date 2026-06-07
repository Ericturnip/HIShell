"""
Compute segmented validation metrics during and after training.
These metrics keep catalog-centered, offset, velocity-offset, fine-grid, and
background cuts separate for the report.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.pv.dataset import _normalize
from src.utils.io import load_yaml


SEGMENTED_CATEGORIES = (
    "Catalog-Centered",
    "Offset / Grazing",
    "Velocity-Offset",
    "Fine-Grid",
    "Background / Random Negatives",
)


def canonical_cut_category(raw: str | None) -> str:
    """Map detailed manifest cut names onto the five report categories."""
    text = str(raw or "").casefold()
    if "fine_grid" in text or "deployment" in text:
        return "Fine-Grid"
    if "background" in text or "random_negative" in text:
        return "Background / Random Negatives"
    if "velocity_offset" in text:
        return "Velocity-Offset"
    if "spatial_offset" in text or "angle_offset" in text or "grazing" in text:
        return "Offset / Grazing"
    if "centered" in text or "catalog" in text:
        return "Catalog-Centered"
    return "Background / Random Negatives" if "negative" in text else "Catalog-Centered"


def _manifest_csv_for_split(cfg: dict[str, Any], split: str) -> Path:
    """Find the CSV manifest path for a named dataset split."""
    manifests = cfg.get("manifests") or {}
    explicit = manifests.get(f"{split}_csv")
    if explicit:
        return Path(explicit)
    return Path(cfg["output_root"]) / f"{split}_manifest.csv"


def _empty_stats() -> dict[str, Any]:
    """Create the mutable counters used before metric finalization."""
    return {
        "cuts": 0,
        "positive_cuts": 0,
        "negative_cuts": 0,
        "pixel": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        "patch": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
    }


def _finalize_counter(counter: dict[str, int]) -> dict[str, float | int]:
    """Convert true-positive, false-positive, and false-negative counts to metrics."""
    tp = int(counter["tp"])
    fp = int(counter["fp"])
    fn = int(counter["fn"])
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _finalize_stats(stats: dict[str, Any], thresholds: list[float]) -> dict[str, Any]:
    """Build the JSON-ready metric block for one cut category."""
    return {
        "cuts": int(stats["cuts"]),
        "positive_cuts": int(stats["positive_cuts"]),
        "negative_cuts": int(stats["negative_cuts"]),
        "pixel_threshold_metrics": {
            str(t): _finalize_counter(stats["pixel"][str(t)])
            for t in thresholds
        },
        "patch_detection_metrics": {
            str(t): _finalize_counter(stats["patch"][str(t)])
            for t in thresholds
        },
    }


def evaluate_model_by_category(
    model: keras.Model,
    config: str | Path | dict[str, Any],
    *,
    split: str = "val",
    batch_size: int | None = None,
    thresholds: list[float] | None = None,
    max_cuts: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate a model while keeping the five cut categories separate.
    Fine-Grid is returned as the primary deployment-readiness category.
    """
    cfg = load_yaml(config) if isinstance(config, (str, Path)) else dict(config)
    thresholds = [float(t) for t in (thresholds or cfg.get("metrics", {}).get("thresholds") or [0.05, 0.075, 0.1])]
    bs = int(batch_size or cfg.get("optim", {}).get("batch_size", 8))
    norm = cfg.get("train", {}).get("norm_method", "zscore_galaxy_only")
    ph = int(cfg["train"]["patch_vel"])
    pw = int(cfg["train"]["patch_pos"])
    manifest_csv = _manifest_csv_for_split(cfg, split)

    rows: list[dict[str, str]] = []
    if manifest_csv.exists():
        with manifest_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    if max_cuts is not None:
        rows = rows[: max(0, int(max_cuts))]

    category_stats = {name: _empty_stats() for name in SEGMENTED_CATEGORIES}
    overall = _empty_stats()

    pending_x: list[np.ndarray] = []
    pending_y: list[np.ndarray] = []
    pending_categories: list[str] = []

    def flush() -> None:
        if not pending_x:
            return
        x_batch = np.stack(pending_x, axis=0).astype(np.float32)
        y_batch = np.stack(pending_y, axis=0).astype(bool)
        pred = model(x_batch, training=False).numpy()
        for idx, category in enumerate(pending_categories):
            y_true = y_batch[idx, ..., 0]
            y_pred = pred[idx, ..., 0]
            label_patch = bool(np.any(y_true))
            pred_score = float(np.nanmax(y_pred)) if y_pred.size else 0.0
            for stats in (overall, category_stats[category]):
                stats["cuts"] += 1
                stats["positive_cuts"] += int(label_patch)
                stats["negative_cuts"] += int(not label_patch)
                for threshold in thresholds:
                    key = str(threshold)
                    pred_bin = y_pred >= threshold
                    stats["pixel"][key]["tp"] += int(np.logical_and(pred_bin, y_true).sum())
                    stats["pixel"][key]["fp"] += int(np.logical_and(pred_bin, ~y_true).sum())
                    stats["pixel"][key]["fn"] += int(np.logical_and(~pred_bin, y_true).sum())
                    pred_patch = pred_score >= threshold
                    stats["patch"][key]["tp"] += int(pred_patch and label_patch)
                    stats["patch"][key]["fp"] += int(pred_patch and not label_patch)
                    stats["patch"][key]["fn"] += int((not pred_patch) and label_patch)
        pending_x.clear()
        pending_y.clear()
        pending_categories.clear()

    for row in rows:
        pv_path = Path(row.get("image_path") or "")
        mask_path = Path(row.get("mask_path") or "")
        if not pv_path.exists() or not mask_path.exists():
            continue
        pv = _normalize(np.load(pv_path), norm)
        lab = np.load(mask_path).astype(np.float32)
        if pv.shape != (ph, pw) or lab.shape != (ph, pw):
            raise ValueError(f"{pv_path.name} has PV/label shapes {pv.shape}/{lab.shape}; expected {(ph, pw)}")
        pending_x.append(pv[..., np.newaxis])
        pending_y.append(lab[..., np.newaxis])
        pending_categories.append(canonical_cut_category(row.get("cut_category") or row.get("cut_type")))
        if len(pending_x) >= bs:
            flush()
    flush()

    categories = {name: _finalize_stats(stats, thresholds) for name, stats in category_stats.items()}
    primary = categories["Fine-Grid"]
    return {
        "split": split,
        "manifest_csv": str(manifest_csv),
        "thresholds": thresholds,
        "deployment_primary_category": "Fine-Grid",
        "overall": _finalize_stats(overall, thresholds),
        "categories": categories,
        "deployment_primary": primary,
    }


class SegmentedValidationCallback(keras.callbacks.Callback):
    """
    Write per-cut-category validation metrics to JSONL and TensorBoard.
    Training uses this callback to avoid hiding fine-grid performance in aggregates.
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any],
        out_dir: str | Path,
        *,
        split: str = "val",
        every: int = 5,
        max_cuts: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.out_dir = Path(out_dir)
        self.split = split
        self.every = max(1, int(every))
        self.max_cuts = max_cuts
        self.writer = tf.summary.create_file_writer(str(self.out_dir / "tensorboard_segmented"))
        self.jsonl_path = self.out_dir / f"segmented_{split}_metrics.jsonl"

    def on_epoch_end(self, epoch, logs=None):
        """Run segmented validation on the configured epoch schedule."""
        e = epoch + 1
        if e != 1 and e % self.every != 0:
            return
        result = evaluate_model_by_category(
            self.model,
            self.config,
            split=self.split,
            max_cuts=self.max_cuts,
        )
        record = {"epoch": e, **result}
        with self.jsonl_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        with self.writer.as_default():
            for category, payload in result["categories"].items():
                tag_base = category.replace(" / ", "_").replace(" ", "_")
                for threshold, metrics in payload["pixel_threshold_metrics"].items():
                    tag = threshold.replace(".", "p")
                    tf.summary.scalar(f"segmented/{tag_base}/pixel_precision_{tag}", metrics["precision"], step=e)
                    tf.summary.scalar(f"segmented/{tag_base}/pixel_recall_{tag}", metrics["recall"], step=e)
                    tf.summary.scalar(f"segmented/{tag_base}/pixel_f1_{tag}", metrics["f1"], step=e)
                for threshold, metrics in payload["patch_detection_metrics"].items():
                    tag = threshold.replace(".", "p")
                    tf.summary.scalar(f"segmented/{tag_base}/patch_precision_{tag}", metrics["precision"], step=e)
                    tf.summary.scalar(f"segmented/{tag_base}/patch_recall_{tag}", metrics["recall"], step=e)
                    tf.summary.scalar(f"segmented/{tag_base}/patch_f1_{tag}", metrics["f1"], step=e)
            self.writer.flush()
