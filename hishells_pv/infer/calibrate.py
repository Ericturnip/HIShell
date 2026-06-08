#!/usr/bin/env python3
"""
Evaluate a trained PV U-Net on one dataset split.
The output JSON records pixel metrics, patch metrics, and segmented metrics by
cut category so report tables can point back to this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hishells_pv.data.tensorflow_dataset import build_dataset, estimate_steps
from hishells_pv.train.callbacks import evaluate_model_by_category
from hishells_pv.utils.io import load_yaml


def _safe_float(x) -> float:
    """Convert TensorFlow scalar outputs into JSON-safe Python floats."""
    return float(x.numpy() if hasattr(x, "numpy") else x)


def evaluate(config: str, model_path: str, split: str, batch_size: int | None = None) -> dict:
    """
    Run model inference and collect pixel, patch, and segmented validation metrics.
    This is the main evaluation entry point used by the clean baseline reports.
    """
    cfg = load_yaml(config)
    bs = int(batch_size or cfg["optim"]["batch_size"])
    steps = estimate_steps(config, split, bs)
    if cfg.get("train", {}).get(f"max_{split}_steps") is not None:
        steps = min(steps, int(cfg["train"][f"max_{split}_steps"]))

    ds = build_dataset(config, split=split, batch_size=bs, seed=2026, repeat=False)
    model = keras.models.load_model(model_path, compile=False)
    segmented = None
    try:
        segmented = evaluate_model_by_category(model, cfg, split=split, batch_size=bs)
    except Exception as exc:
        segmented = {"error": str(exc)}

    bce = keras.losses.BinaryCrossentropy(from_logits=False)
    pr_auc = keras.metrics.AUC(curve="PR", name="pr_auc")
    precision = keras.metrics.Precision(name="precision")
    recall = keras.metrics.Recall(name="recall")

    thresholds = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    tp = {t: 0 for t in thresholds}
    fp = {t: 0 for t in thresholds}
    fn = {t: 0 for t in thresholds}
    patch_scores: list[float] = []
    patch_labels: list[int] = []
    losses: list[float] = []
    batches = 0

    for x, y in ds.take(steps):
        pred = model(x, training=False)
        losses.append(_safe_float(bce(y, pred)))
        pr_auc.update_state(y, pred)
        precision.update_state(y, pred)
        recall.update_state(y, pred)

        y_np = y.numpy() > 0.5
        p_np = pred.numpy()
        patch_scores.extend(p_np.reshape((p_np.shape[0], -1)).max(axis=1).tolist())
        patch_labels.extend(y_np.reshape((y_np.shape[0], -1)).any(axis=1).astype(int).tolist())
        for t in thresholds:
            pred_bin = p_np >= t
            tp[t] += int(np.logical_and(pred_bin, y_np).sum())
            fp[t] += int(np.logical_and(pred_bin, ~y_np).sum())
            fn[t] += int(np.logical_and(~pred_bin, y_np).sum())
        batches += 1

    threshold_metrics = {}
    for t in thresholds:
        prec = tp[t] / max(1, tp[t] + fp[t])
        rec = tp[t] / max(1, tp[t] + fn[t])
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        threshold_metrics[str(t)] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp[t],
            "fp": fp[t],
            "fn": fn[t],
        }

    patch_scores_np = np.asarray(patch_scores, dtype=np.float32)
    patch_labels_np = np.asarray(patch_labels, dtype=np.int32)
    patch_metrics = {}
    for t in thresholds:
        pred_patch = patch_scores_np >= t
        y_patch = patch_labels_np.astype(bool)
        ptp = int(np.logical_and(pred_patch, y_patch).sum())
        pfp = int(np.logical_and(pred_patch, ~y_patch).sum())
        pfn = int(np.logical_and(~pred_patch, y_patch).sum())
        pprec = ptp / max(1, ptp + pfp)
        precall = ptp / max(1, ptp + pfn)
        patch_metrics[str(t)] = {
            "precision": pprec,
            "recall": precall,
            "f1": 2 * pprec * precall / max(1e-12, pprec + precall),
            "tp": ptp,
            "fp": pfp,
            "fn": pfn,
        }

    return {
        "config": str(Path(config).resolve()),
        "model": str(Path(model_path).resolve()),
        "split": split,
        "batch_size": bs,
        "batches": batches,
        "loss": float(np.mean(losses)) if losses else None,
        "pixel_pr_auc": _safe_float(pr_auc.result()),
        "pixel_precision_at_0p5": _safe_float(precision.result()),
        "pixel_recall_at_0p5": _safe_float(recall.result()),
        "pixel_threshold_metrics": threshold_metrics,
        "patch_detection_metrics": patch_metrics,
        "segmented_validation_metrics": segmented,
        "deployment_primary_metric": None
        if not isinstance(segmented, dict) or "deployment_primary" not in segmented
        else {
            "category": segmented.get("deployment_primary_category"),
            "metrics": segmented.get("deployment_primary"),
        },
        "patches_seen": int(len(patch_labels)),
        "positive_patches": int(patch_labels_np.sum()) if len(patch_labels_np) else 0,
        "negative_patches": int(len(patch_labels_np) - patch_labels_np.sum()) if len(patch_labels_np) else 0,
    }


def main() -> None:
    """Evaluate one split from CLI arguments and write the JSON result file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = evaluate(args.config, args.model, args.split, args.batch_size)
    out = Path(args.out) if args.out else Path(args.model).with_name(f"eval_{args.split}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"[eval] wrote {out.resolve()}")
    print(
        f"[eval] split={result['split']} loss={result['loss']:.4f} "
        f"pixel_pr_auc={result['pixel_pr_auc']:.4f} "
        f"precision@0.5={result['pixel_precision_at_0p5']:.4f} "
        f"recall@0.5={result['pixel_recall_at_0p5']:.4f}"
    )
    deployment = result.get("deployment_primary_metric")
    if deployment and deployment.get("metrics"):
        fine = deployment["metrics"]["patch_detection_metrics"].get("0.075")
        if fine:
            print(
                "[eval] deployment Fine-Grid patch@0.075 "
                f"precision={fine['precision']:.4f} recall={fine['recall']:.4f} f1={fine['f1']:.4f}"
            )


if __name__ == "__main__":
    main()
