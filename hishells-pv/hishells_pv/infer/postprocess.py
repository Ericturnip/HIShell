"""Post-process probability maps into ranked PV-shell candidate components.

For each PV cut we threshold the model probability, extract connected
components (``scipy.ndimage.label``), compute per-component features, apply
geometry/size filters, and emit a ranked candidate list as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from hishells_pv.data.dataset import read_manifest
from hishells_pv.infer.predict import load_checkpoint, predict_pv
from hishells_pv.train.trainer import resolve_device
from hishells_pv.utils.io import load_yaml


def _component_features(prob: np.ndarray, mask: np.ndarray, label_id: int, labels: np.ndarray) -> dict[str, Any]:
    sel = labels == label_id
    ys, xs = np.where(sel)
    area = int(sel.sum())
    v0, v1 = int(ys.min()), int(ys.max())
    s0, s1 = int(xs.min()), int(xs.max())
    height = v1 - v0 + 1  # velocity extent
    width = s1 - s0 + 1  # spatial extent
    comp_prob = prob[sel]
    h, w = prob.shape
    touches_edge = bool(v0 == 0 or s0 == 0 or v1 == h - 1 or s1 == w - 1)
    return {
        "area_pix": area,
        "mean_prob": float(comp_prob.mean()),
        "max_prob": float(comp_prob.max()),
        "prob_mass": float(comp_prob.sum()),
        "centroid_vel": float(ys.mean()),
        "centroid_pos": float(xs.mean()),
        "vel_extent": int(height),
        "pos_extent": int(width),
        "bbox": [v0, s0, v1, s1],
        "touches_edge": touches_edge,
    }


def extract_candidates(
    prob: np.ndarray,
    *,
    threshold: float,
    min_area_pix: int = 6,
    drop_edge: bool = False,
    max_vel_extent: int | None = None,
) -> list[dict[str, Any]]:
    """Return filtered candidate components for one probability map."""
    mask = prob >= float(threshold)
    labels, n = ndimage.label(mask)
    candidates: list[dict[str, Any]] = []
    for label_id in range(1, n + 1):
        feats = _component_features(prob, mask, label_id, labels)
        if feats["area_pix"] < int(min_area_pix):
            continue
        if drop_edge and feats["touches_edge"]:
            continue
        if max_vel_extent is not None and feats["vel_extent"] > int(max_vel_extent):
            continue
        candidates.append(feats)
    candidates.sort(key=lambda c: c["prob_mass"], reverse=True)
    return candidates


def postprocess_run(
    config: str | None,
    model_path: str | Path,
    *,
    threshold: float | None = None,
    split: str | None = None,
    min_area_pix: int = 6,
    drop_edge: bool = False,
    device_name: str = "auto",
    out_dir: str | Path | None = None,
) -> Path:
    """Run inference + component extraction over a split (or all PVs) and save candidates.

    If ``threshold`` is None, reads ``<output_root>/calib/threshold.txt`` when present,
    else defaults to 0.075.
    """
    device = resolve_device(device_name)
    model, ckpt_cfg = load_checkpoint(model_path, device)
    cfg = load_yaml(config) if config else ckpt_cfg
    root = Path(cfg["output_root"])
    norm_method = cfg.get("train", {}).get("norm_method", "zscore_galaxy_only")

    if threshold is None:
        calib = root / "calib" / "threshold.txt"
        threshold = float(calib.read_text().strip()) if calib.exists() else 0.075

    if split is not None:
        names = [f for f in read_manifest(root, split) if not f.endswith("_posxy.npy")]
        pv_paths = [root / "pv" / n for n in names]
    else:
        pv_paths = sorted(p for p in (root / "pv").glob("*.npy") if not p.name.endswith("_posxy.npy"))

    out = Path(out_dir) if out_dir else (root / "candidates")
    out.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total = 0
    for pv_path in pv_paths:
        pv = np.load(pv_path)
        prob = predict_pv(model, pv, device, norm_method=norm_method)
        cands = extract_candidates(
            prob,
            threshold=float(threshold),
            min_area_pix=int(min_area_pix),
            drop_edge=bool(drop_edge),
        )
        total += len(cands)
        results.append({"pv": pv_path.name, "n_candidates": len(cands), "candidates": cands})

    report = {
        "model": str(Path(model_path).resolve()),
        "threshold": float(threshold),
        "split": split or "all",
        "min_area_pix": int(min_area_pix),
        "drop_edge": bool(drop_edge),
        "n_pv": len(pv_paths),
        "n_candidates_total": total,
        "results": results,
    }
    out_path = out / f"candidates_{split or 'all'}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(
        f"[postprocess] split={split or 'all'} n_pv={len(pv_paths)} "
        f"threshold={threshold} candidates={total} -> {out_path.resolve()}"
    )
    return out_path
