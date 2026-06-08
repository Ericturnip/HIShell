"""Calibrate a global probability threshold against labelled PV cuts.

Sweeps a grid of thresholds over a labelled split and selects the one that
maximizes patch-level F1 (subject to an optional minimum recall), then writes
``<output_root>/calib/threshold.txt`` and a JSON report.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hishells_pv.data.dataset import read_manifest
from hishells_pv.infer.predict import load_checkpoint, predict_pv
from hishells_pv.train.trainer import resolve_device
from hishells_pv.utils.io import load_yaml


def _patch_pixel_counts(prob: np.ndarray, lab: np.ndarray, t: float) -> tuple[int, int, int, int, int, int]:
    pred = prob >= t
    truth = lab > 0.5
    pix_tp = int(np.logical_and(pred, truth).sum())
    pix_fp = int(np.logical_and(pred, ~truth).sum())
    pix_fn = int(np.logical_and(~pred, truth).sum())
    patch_true = bool(truth.any())
    patch_pred = bool((prob.max() if prob.size else 0.0) >= t)
    pat_tp = int(patch_pred and patch_true)
    pat_fp = int(patch_pred and not patch_true)
    pat_fn = int((not patch_pred) and patch_true)
    return pix_tp, pix_fp, pix_fn, pat_tp, pat_fp, pat_fn


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def calibrate_threshold(
    config: str,
    model_path: str | Path,
    *,
    split: str = "val",
    thresholds: list[float] | None = None,
    min_recall: float = 0.0,
    device_name: str = "auto",
    out_dir: str | Path | None = None,
) -> dict:
    """Return the calibration report and write ``threshold.txt`` + report JSON."""
    device = resolve_device(device_name)
    model, ckpt_cfg = load_checkpoint(model_path, device)
    cfg = load_yaml(config)
    root = Path(cfg["output_root"])
    norm_method = cfg.get("train", {}).get("norm_method", "zscore_galaxy_only")

    if thresholds is None:
        grid = np.round(np.linspace(0.02, 0.6, 30), 4)
        thresholds = [float(t) for t in grid]

    files = [f for f in read_manifest(root, split) if not f.endswith("_posxy.npy")]
    pix = {t: [0, 0, 0] for t in thresholds}
    pat = {t: [0, 0, 0] for t in thresholds}
    for fname in files:
        pv = np.load(root / "pv" / fname)
        lab = np.load(root / "labels" / fname)
        prob = predict_pv(model, pv, device, norm_method=norm_method)
        for t in thresholds:
            ptp, pfp, pfn, atp, afp, afn = _patch_pixel_counts(prob, lab, t)
            pix[t][0] += ptp; pix[t][1] += pfp; pix[t][2] += pfn
            pat[t][0] += atp; pat[t][1] += afp; pat[t][2] += afn

    rows = []
    for t in thresholds:
        patch = _prf(*pat[t])
        pixel = _prf(*pix[t])
        rows.append({"threshold": t, "patch": patch, "pixel": pixel})

    eligible = [r for r in rows if r["patch"]["recall"] >= float(min_recall)] or rows
    best = max(eligible, key=lambda r: r["patch"]["f1"])

    out = Path(out_dir) if out_dir else (root / "calib")
    out.mkdir(parents=True, exist_ok=True)
    (out / "threshold.txt").write_text(f"{best['threshold']}\n")
    report = {
        "model": str(Path(model_path).resolve()),
        "split": split,
        "n_pv": len(files),
        "min_recall": float(min_recall),
        "best_threshold": best["threshold"],
        "best_patch": best["patch"],
        "best_pixel": best["pixel"],
        "grid": rows,
    }
    (out / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"[calibrate] split={split} n={len(files)} best_threshold={best['threshold']} "
        f"patch_f1={best['patch']['f1']:.3f} patch_recall={best['patch']['recall']:.3f} -> {out.resolve()}"
    )
    return report
