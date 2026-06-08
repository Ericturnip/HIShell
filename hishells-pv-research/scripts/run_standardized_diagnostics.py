#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.diagnostic_utils import component_features, load_json, pixel_metrics, predict_full_pv, write_csv


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _zscore_finite(pv: np.ndarray) -> np.ndarray:
    finite = pv[np.isfinite(pv)]
    if finite.size == 0:
        return np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mu = float(np.mean(finite))
    sigma = float(np.std(finite) + 1e-6)
    return np.nan_to_num((pv - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _pad_to(pv: np.ndarray, label: np.ndarray, patch_vel: int, patch_pos: int) -> tuple[np.ndarray, np.ndarray]:
    v, s = pv.shape
    dv = max(0, patch_vel - v)
    ds = max(0, patch_pos - s)
    if dv == 0 and ds == 0:
        return pv, label
    pv = np.pad(pv, ((dv // 2, dv - dv // 2), (ds // 2, ds - ds // 2)), mode="edge")
    label = np.pad(label, ((dv // 2, dv - dv // 2), (ds // 2, ds - ds // 2)), mode="constant", constant_values=0)
    return pv, label


def _choose_patch(
    pv: np.ndarray,
    label: np.ndarray,
    *,
    pos_frac: float,
    patch_vel: int,
    patch_pos: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    v, s = pv.shape
    want_pos = rng.random() < pos_frac
    if want_pos and bool(np.any(label > 0)):
        ys, xs = np.where(label > 0)
        k = rng.randrange(len(ys))
        cy, cx = int(ys[k]), int(xs[k])
        y0 = max(0, min(cy - patch_vel // 2, v - patch_vel))
        x0 = max(0, min(cx - patch_pos // 2, s - patch_pos))
    else:
        y0 = rng.randrange(0, max(1, v - patch_vel + 1))
        x0 = rng.randrange(0, max(1, s - patch_pos + 1))
    return pv[y0 : y0 + patch_vel, x0 : x0 + patch_pos], label[y0 : y0 + patch_vel, x0 : x0 + patch_pos], y0, x0


def _load_pv_label(row: dict[str, str], patch_vel: int, patch_pos: int) -> tuple[np.ndarray, np.ndarray]:
    pv = np.load(row["image_path"])
    label = np.load(row["mask_path"]) if row.get("mask_path") and Path(row["mask_path"]).exists() else np.zeros_like(pv)
    pv = _zscore_finite(pv)
    return _pad_to(pv, label, patch_vel, patch_pos)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _write_png_rgb(path: Path, image: np.ndarray) -> None:
    image = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _stretch_image(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.nanpercentile(x, [1.0, 99.5]) if x.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return np.arcsinh(4.0 * y) / np.arcsinh(4.0)


def _nearest_resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = np.linspace(0, img.shape[0] - 1, height).astype(int)
    xs = np.linspace(0, img.shape[1] - 1, width).astype(int)
    return img[ys][:, xs]


def _gray_rgb(x: np.ndarray) -> np.ndarray:
    g = (255.0 * _stretch_image(x)).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def _mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    out[mask > 0] = color
    return out


def _prob_rgb(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), 0.0, 1.0)
    return np.stack(
        [
            (255.0 * p).astype(np.uint8),
            (255.0 * np.sqrt(p) * 0.75).astype(np.uint8),
            (255.0 * (1.0 - p) * 0.15).astype(np.uint8),
        ],
        axis=2,
    )


def _overlay_rgb(pv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = _gray_rgb(pv).astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    alpha = 0.45 * (mask > 0)[..., None]
    return ((1 - alpha) * base + alpha * red).astype(np.uint8)


def _panel_image(pv: np.ndarray, label: np.ndarray, prob: np.ndarray, *, scale_h: int = 192, scale_w: int = 256) -> np.ndarray:
    tiles = [
        _gray_rgb(pv),
        _mask_rgb(label > 0, (80, 180, 255)),
        _prob_rgb(prob),
        _mask_rgb(prob >= 0.05, (255, 80, 70)),
        _mask_rgb(prob >= 0.075, (255, 160, 60)),
        _overlay_rgb(pv, prob >= 0.075),
    ]
    tiles = [_nearest_resize(tile, scale_h, scale_w) for tile in tiles]
    gap = np.full((scale_h, 4, 3), 30, dtype=np.uint8)
    row_img = tiles[0]
    for tile in tiles[1:]:
        row_img = np.concatenate([row_img, gap, tile], axis=1)
    return row_img


def _predict_patch(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict(x[None, ..., None].astype(np.float32), verbose=0)[0, ..., 0]


def _row_context(row: dict[str, str]) -> dict[str, Any]:
    meta = load_json(Path(row.get("metadata_path", "")))
    return {
        "filename": row.get("filename"),
        "galaxy": row.get("galaxy"),
        "split": row.get("split"),
        "cut_type": row.get("cut_type"),
        "cut_category": row.get("cut_category"),
        "shell_id": row.get("shell_id"),
        "positive_manifest": _safe_int(row.get("positive")),
        "local_velocity_kms": _safe_float(row.get("local_velocity_kms")),
        "velocity_center_kms": _safe_float(row.get("velocity_center_kms")),
        "velocity_offset_kms": _safe_float(row.get("velocity_offset_kms")),
        "spatial_window_kpc": _safe_float(row.get("spatial_window_kpc")),
        "velocity_window_kms": _safe_float(row.get("velocity_window_kms")),
        "mask_pixels_manifest": _safe_int(row.get("mask_pixels")),
        "quality_flags": row.get("quality_flags", ""),
        "pv_cut_center": meta.get("pv_cut_center_pix"),
        "pv_cut_angle_deg": meta.get("pv_cut_angle_deg"),
        "spatial_offset_pix": meta.get("spatial_offset_pix"),
        "spatial_offset_beam": meta.get("spatial_offset_beam"),
        "angle_offset_deg": meta.get("angle_offset_deg"),
        "source": meta.get("source"),
        "source_shell_id": meta.get("source_shell_id"),
    }


def _patch_sample_predictions(
    model: Any,
    rows_by_split: dict[str, list[dict[str, str]]],
    *,
    splits: list[str],
    patch_vel: int,
    patch_pos: int,
    pos_frac: float,
    samples_per_pv: int,
    batch_size: int,
    seed: int,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    sample_rows: list[dict[str, Any]] = []
    batch_x: list[np.ndarray] = []
    batch_y: list[np.ndarray] = []
    batch_meta: list[dict[str, Any]] = []

    def flush() -> None:
        if not batch_x:
            return
        preds = model.predict(np.stack(batch_x, axis=0)[..., None].astype(np.float32), verbose=0)[..., 0]
        for pred, label, meta in zip(preds, batch_y, batch_meta):
            true_patch = bool(np.any(label > 0.5))
            max_prob = float(np.nanmax(pred)) if pred.size else 0.0
            out = dict(meta)
            out.update(
                {
                    "true_patch": int(true_patch),
                    "label_pixels_in_patch": int(np.sum(label > 0.5)),
                    "max_probability": max_prob,
                    "mean_probability": float(np.nanmean(pred)) if pred.size else 0.0,
                    "probability_mass": float(np.nansum(pred)),
                }
            )
            for threshold in thresholds:
                tag = str(threshold).replace(".", "p")
                pred_patch = max_prob >= threshold
                pm = pixel_metrics(pred, label, threshold)
                out[f"pred_patch_{tag}"] = int(pred_patch)
                out[f"patch_tp_{tag}"] = int(pred_patch and true_patch)
                out[f"patch_fp_{tag}"] = int(pred_patch and not true_patch)
                out[f"patch_fn_{tag}"] = int((not pred_patch) and true_patch)
                out[f"patch_tn_{tag}"] = int((not pred_patch) and not true_patch)
                for key, value in pm.items():
                    out[f"pixel_{key}_{tag}"] = value
            sample_rows.append(out)
        batch_x.clear()
        batch_y.clear()
        batch_meta.clear()

    for split in splits:
        rng = random.Random(seed)
        for row in rows_by_split[split]:
            pv, label = _load_pv_label(row, patch_vel, patch_pos)
            for sample_index in range(samples_per_pv):
                x, y, y0, x0 = _choose_patch(
                    pv,
                    label,
                    pos_frac=pos_frac,
                    patch_vel=patch_vel,
                    patch_pos=patch_pos,
                    rng=rng,
                )
                meta = _row_context(row)
                meta.update({"sample_index": sample_index, "crop_v0": y0, "crop_pos0": x0})
                batch_x.append(x)
                batch_y.append(y)
                batch_meta.append(meta)
                if len(batch_x) >= batch_size:
                    flush()
    flush()
    return sample_rows


def _category_metrics(sample_rows: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    for row in sample_rows:
        for threshold in thresholds:
            tag = str(threshold).replace(".", "p")
            key = (row["split"], row["galaxy"], row["cut_category"], threshold)
            group = groups.setdefault(
                key,
                {
                    "split": row["split"],
                    "galaxy": row["galaxy"],
                    "cut_category": row["cut_category"],
                    "threshold": threshold,
                    "patches": 0,
                    "positive_patches": 0,
                    "negative_patches": 0,
                    "patch_tp": 0,
                    "patch_fp": 0,
                    "patch_fn": 0,
                    "patch_tn": 0,
                    "pixel_tp": 0,
                    "pixel_fp": 0,
                    "pixel_fn": 0,
                },
            )
            group["patches"] += 1
            group["positive_patches"] += int(row["true_patch"])
            group["negative_patches"] += int(not row["true_patch"])
            for name in ("tp", "fp", "fn", "tn"):
                group[f"patch_{name}"] += int(row[f"patch_{name}_{tag}"])
            group["pixel_tp"] += int(row[f"pixel_tp_{tag}"])
            group["pixel_fp"] += int(row[f"pixel_fp_{tag}"])
            group["pixel_fn"] += int(row[f"pixel_fn_{tag}"])

    rows = []
    for group in groups.values():
        ptp, pfp, pfn = group["patch_tp"], group["patch_fp"], group["patch_fn"]
        px_tp, px_fp, px_fn = group["pixel_tp"], group["pixel_fp"], group["pixel_fn"]
        patch_precision = ptp / max(1, ptp + pfp)
        patch_recall = ptp / max(1, ptp + pfn)
        pixel_precision = px_tp / max(1, px_tp + px_fp)
        pixel_recall = px_tp / max(1, px_tp + px_fn)
        group["patch_precision"] = patch_precision
        group["patch_recall"] = patch_recall
        group["patch_f1"] = 2 * patch_precision * patch_recall / max(1e-12, patch_precision + patch_recall)
        group["pixel_precision"] = pixel_precision
        group["pixel_recall"] = pixel_recall
        group["pixel_f1"] = 2 * pixel_precision * pixel_recall / max(1e-12, pixel_precision + pixel_recall)
        rows.append(group)
    rows.sort(key=lambda r: (r["split"], r["galaxy"], r["cut_category"], float(r["threshold"])))
    return rows


def _select_panel_rows(sample_rows: list[dict[str, Any]], max_per_category: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        true_patch = bool(row["true_patch"])
        pred005 = bool(row["pred_patch_0p05"])
        pred0075 = bool(row["pred_patch_0p075"])
        cut_category = row.get("cut_category", "")
        if true_patch and pred0075:
            buckets["true_positive_0p075"].append(row)
        if true_patch and pred005 and not pred0075:
            buckets["detected_0p05_missed_0p075"].append(row)
        if true_patch and not pred0075:
            buckets[f"{row['split']}_missed_positive_0p075"].append(row)
        if (not true_patch) and pred0075:
            buckets["false_positive_0p075"].append(row)
        if (not true_patch) and pred005 and not pred0075:
            buckets["false_positive_0p05_only"].append(row)
        if cut_category == "fine_grid_deployment_like" and (not true_patch) and pred0075:
            buckets["fine_grid_false_positive_0p075"].append(row)
        if cut_category == "background_random_negative" and (not true_patch) and pred0075:
            buckets["background_false_positive_0p075"].append(row)

    selected: list[dict[str, Any]] = []
    for category, rows in buckets.items():
        if "missed_positive" in category:
            rows = sorted(rows, key=lambda r: (r["max_probability"], -r["label_pixels_in_patch"]))
        else:
            rows = sorted(rows, key=lambda r: r["max_probability"], reverse=True)
        for rank, row in enumerate(rows[:max_per_category], start=1):
            item = dict(row)
            item["panel_category"] = category
            item["panel_rank"] = rank
            selected.append(item)
    return selected


def _write_review_panels(
    model: Any,
    selected_rows: list[dict[str, Any]],
    rows_by_filename: dict[str, dict[str, str]],
    *,
    run_dir: Path,
    patch_vel: int,
    patch_pos: int,
) -> None:
    panel_dir = run_dir / "review_panels_standardized"
    index_rows = []
    for row in selected_rows:
        source = rows_by_filename[row["filename"]]
        pv, label = _load_pv_label(source, patch_vel, patch_pos)
        y0 = int(row["crop_v0"])
        x0 = int(row["crop_pos0"])
        x = pv[y0 : y0 + patch_vel, x0 : x0 + patch_pos]
        y = label[y0 : y0 + patch_vel, x0 : x0 + patch_pos]
        prob = _predict_patch(model, x)
        stem = Path(row["filename"]).stem
        category = row["panel_category"]
        filename = f"{category}_{int(row['panel_rank']):03d}_{row['split']}_{stem}_s{row['sample_index']}.png"
        _write_png_rgb(panel_dir / filename, _panel_image(x, y, prob))
        mask = prob >= 0.075
        vals = prob[mask]
        index_rows.append(
            {
                "filename": filename,
                "category": category,
                "split": row["split"],
                "galaxy": row["galaxy"],
                "patch_id": stem,
                "sample_index": row["sample_index"],
                "shell_id": row.get("shell_id"),
                "cut_category": row.get("cut_category"),
                "threshold": 0.075,
                "max_probability": float(np.nanmax(prob)) if prob.size else 0.0,
                "mean_predicted_probability_inside_mask": float(np.nanmean(vals)) if vals.size else 0.0,
                "predicted_mask_area": int(mask.sum()),
                "label_pixels_in_patch": int(np.sum(y > 0.5)),
                "crop_v0": y0,
                "crop_pos0": x0,
                "notes": f"local_velocity={row.get('local_velocity_kms')}; velocity_offset={row.get('velocity_offset_kms')}",
            }
        )
    write_csv(index_rows, run_dir / "review_panels_standardized_index.csv")


def _full_pv_components(
    model: Any,
    rows_by_split: dict[str, list[dict[str, str]]],
    *,
    splits: list[str],
    run_dir: Path,
    patch_vel: int,
    patch_pos: int,
    batch_size: int,
    thresholds: list[float],
    min_area_pix: int,
    save_probability_maps: bool,
) -> dict[str, int]:
    rows_by_threshold: dict[float, list[dict[str, Any]]] = {threshold: [] for threshold in thresholds}
    processed_by_split: dict[str, int] = {}
    for split in splits:
        processed = 0
        prob_dir = run_dir / "probability_maps" / "best_model" / split
        for row in rows_by_split[split]:
            pv = np.load(row["image_path"])
            label_exists = bool(row.get("mask_path") and Path(row["mask_path"]).exists())
            label_positive = False
            if label_exists:
                label_positive = bool(np.any(np.load(row["mask_path"]) > 0.5))
            meta = load_json(Path(row["metadata_path"]))
            prob = predict_full_pv(model, pv, patch_vel=patch_vel, patch_pos=patch_pos, batch_size=batch_size)
            if save_probability_maps:
                prob_dir.mkdir(parents=True, exist_ok=True)
                np.save(prob_dir / row["filename"], prob.astype(np.float16))
            for threshold in thresholds:
                comps = component_features(prob, prob >= threshold, threshold=threshold, meta=meta, min_area=min_area_pix)
                for comp in comps:
                    comp.update(_row_context(row))
                    comp.update(
                        {
                            "split": split,
                            "patch_id": Path(row["filename"]).stem,
                            "patch_name": row["filename"],
                            "label_positive_full_pv": int(label_positive),
                        }
                    )
                    comp["velocity_extent_plausibility_warning"] = bool(
                        comp.get("velocity_extent_kms") is not None and comp["velocity_extent_kms"] > 120.0
                    )
                    comp["edge_artifact_warning"] = bool(comp.get("edge_touching"))
                    rows_by_threshold[threshold].append(comp)
            processed += 1
            if processed % 500 == 0:
                print(f"[components] {split}: processed {processed} PV cuts")
        processed_by_split[split] = processed

    for threshold, rows in rows_by_threshold.items():
        tag = str(threshold).replace(".", "p")
        csv_path = run_dir / f"candidate_components_threshold_{tag}.csv"
        json_path = run_dir / f"candidate_components_threshold_{tag}.json"
        write_csv(rows, csv_path)
        json_path.write_text(json.dumps({"threshold": threshold, "components": rows}, indent=2))
        print(f"[components] threshold={threshold} rows={len(rows)} -> {csv_path}")
    return processed_by_split


def _mine_hard_negatives(
    model: Any,
    rows: list[dict[str, str]],
    *,
    run_dir: Path,
    split: str,
    patch_vel: int,
    patch_pos: int,
    batch_size: int,
    thresholds: list[float],
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    negative_rows = [row for row in rows if _safe_int(row.get("positive")) == 0 or _safe_int(row.get("mask_pixels")) == 0]
    for i, row in enumerate(negative_rows, start=1):
        pv = np.load(row["image_path"])
        prob = predict_full_pv(model, pv, patch_vel=patch_vel, patch_pos=patch_pos, batch_size=batch_size)
        max_prob = float(np.nanmax(prob)) if prob.size else 0.0
        triggered = [float(t) for t in thresholds if max_prob >= float(t)]
        if not triggered:
            continue
        threshold = min(triggered)
        mask = prob >= threshold
        meta = load_json(Path(row["metadata_path"]))
        comps = component_features(prob, mask, threshold=threshold, meta=meta, min_area=1)
        largest = max(comps, key=lambda r: r["area_pix"], default={})
        vals = prob[mask]
        item = _row_context(row)
        item.update(
            {
                "source_split": split,
                "patch_path": row["image_path"],
                "patch_id": Path(row["filename"]).stem,
                "thresholds_triggered": triggered,
                "max_predicted_probability": max_prob,
                "mean_predicted_probability_inside_predicted_mask": float(np.nanmean(vals)) if vals.size else 0.0,
                "total_predicted_probability_mass": float(np.nansum(prob * mask)),
                "predicted_mask_area": int(mask.sum()),
                "connected_component_count": len(comps),
                "largest_connected_component_area": largest.get("area_pix", 0),
                "largest_component_edge_touching": largest.get("edge_touching"),
                "hard_negative_source": "standardized_train_negative",
            }
        )
        candidates.append(item)
        candidates.sort(key=lambda r: (r["max_predicted_probability"], r["predicted_mask_area"]), reverse=True)
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
        if i % 500 == 0:
            print(f"[hard-negatives] processed {i}/{len(negative_rows)} train negatives")

    out = {
        "model": str((run_dir / "best_model.keras").resolve()),
        "source_split": split,
        "thresholds": thresholds,
        "notes": [
            "Fresh hard negatives mined in the standardized 5 kpc / 200 km/s coordinate system.",
            "This is a review list only. It was not added to any training manifest or YAML.",
            "Only the training fold is mined here to avoid held-out test contamination.",
        ],
        "candidates": candidates,
    }
    json_path = run_dir / "hard_negative_candidates_standardized_train.json"
    csv_path = run_dir / "hard_negative_candidates_standardized_train.csv"
    json_path.write_text(json.dumps(out, indent=2))
    write_csv(candidates, csv_path)
    print(f"[hard-negatives] wrote {len(candidates)} candidates -> {json_path}")
    return candidates


def _summarize_inspection(sample_rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    fine_grid_fp = [
        r
        for r in sample_rows
        if r["cut_category"] == "fine_grid_deployment_like" and not r["true_patch"] and r["pred_patch_0p075"]
    ]
    background_fp = [
        r
        for r in sample_rows
        if r["cut_category"] == "background_random_negative" and not r["true_patch"] and r["pred_patch_0p075"]
    ]
    missed = [r for r in sample_rows if r["true_patch"] and not r["pred_patch_0p075"]]
    fine_grid_fp.sort(key=lambda r: r["max_probability"], reverse=True)
    background_fp.sort(key=lambda r: r["max_probability"], reverse=True)
    missed.sort(key=lambda r: (r["split"], r["max_probability"], -r["label_pixels_in_patch"]))

    write_csv(fine_grid_fp, run_dir / "fine_grid_false_positives_0p075.csv")
    write_csv(background_fp, run_dir / "background_false_positives_0p075.csv")
    write_csv(missed, run_dir / "missed_positive_patches_0p075.csv")

    summary = {
        "fine_grid_false_positives_0p075": len(fine_grid_fp),
        "background_false_positives_0p075": len(background_fp),
        "missed_positive_patches_0p075": len(missed),
        "missed_positive_patches_by_split": dict(
            sorted(
                {
                    split: sum(1 for r in missed if r["split"] == split)
                    for split in {r["split"] for r in sample_rows}
                }.items()
            )
        ),
    }
    (run_dir / "standardized_diagnostics_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_markdown_report(
    run_dir: Path,
    *,
    sample_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    inspection_summary: dict[str, Any],
    component_counts: dict[str, int],
    hard_negatives: list[dict[str, Any]],
) -> None:
    def threshold_totals(threshold: float) -> dict[str, float | int]:
        tag = str(threshold).replace(".", "p")
        tp = sum(int(r[f"patch_tp_{tag}"]) for r in sample_rows)
        fp = sum(int(r[f"patch_fp_{tag}"]) for r in sample_rows)
        fn = sum(int(r[f"patch_fn_{tag}"]) for r in sample_rows)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}

    top_categories = [
        r
        for r in category_rows
        if float(r["threshold"]) == 0.075 and (r["cut_category"] in {"fine_grid_deployment_like", "background_random_negative"})
    ]
    lines = [
        "# Standardized Run Diagnostics",
        "",
        "Checkpoint: `best_model.keras`",
        "",
        "No training was launched during this diagnostics pass.",
        "",
        "## Patch-Sampled Metrics",
        "",
        "| threshold | patch precision | patch recall | TP | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in (0.05, 0.075):
        totals = threshold_totals(threshold)
        lines.append(
            f"| {threshold} | {totals['precision']:.4f} | {totals['recall']:.4f} | {totals['tp']} | {totals['fp']} | {totals['fn']} |"
        )
    lines.extend(
        [
            "",
            "## Inspection Targets",
            "",
            f"- Fine-grid false-positive patch samples at 0.075: {inspection_summary['fine_grid_false_positives_0p075']}",
            f"- Background false-positive patch samples at 0.075: {inspection_summary['background_false_positives_0p075']}",
            f"- Missed positive patch samples at 0.075: {inspection_summary['missed_positive_patches_0p075']}",
            f"- Missed positives by split: {inspection_summary['missed_positive_patches_by_split']}",
            "",
            "Detailed CSVs:",
            "",
            "- `fine_grid_false_positives_0p075.csv`",
            "- `background_false_positives_0p075.csv`",
            "- `missed_positive_patches_0p075.csv`",
            "",
            "## Deployment-Like Categories at 0.075",
            "",
            "| split | galaxy | cut category | patches | positives | negatives | patch precision | patch recall | patch FP | patch FN |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_categories:
        lines.append(
            "| {split} | {galaxy} | {cut_category} | {patches} | {positive_patches} | {negative_patches} | {patch_precision:.4f} | {patch_recall:.4f} | {patch_fp} | {patch_fn} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Component Extraction",
            "",
            f"Full-PV probability maps/components were generated for: {component_counts}.",
            "",
            "Outputs:",
            "",
            "- `probability_maps/best_model/{split}/`",
            "- `candidate_components_threshold_0p05.csv` / `.json`",
            "- `candidate_components_threshold_0p075.csv` / `.json`",
            "",
            "## Fresh Standardized Hard Negatives",
            "",
            f"Fresh train-fold hard-negative candidates: {len(hard_negatives)}",
            "",
            "Outputs:",
            "",
            "- `hard_negative_candidates_standardized_train.json`",
            "- `hard_negative_candidates_standardized_train.csv`",
            "",
            "These candidates are review-only and were not added to any training manifest or config. They were mined from the standardized training fold only, so the held-out test fold remains isolated.",
            "",
            "## Review Panels",
            "",
            "Review PNGs were written to `review_panels_standardized/`, with index `review_panels_standardized_index.csv`.",
        ]
    )
    (run_dir / "standardized_diagnostics_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_data/standardized_5kpc_200kms/train_standardized_high_recall.yaml")
    ap.add_argument("--model", default="runs/pv_unet_standardized_5kpc_200kms_20260521_015917/best_model.keras")
    ap.add_argument("--run-dir", default="runs/pv_unet_standardized_5kpc_200kms_20260521_015917")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--hard-negative-split", default="train")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.075])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--panel-max-per-category", type=int, default=12)
    ap.add_argument("--component-min-area-pix", type=int, default=4)
    ap.add_argument("--max-hard-negatives", type=int, default=500)
    ap.add_argument("--skip-probability-map-save", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    root = Path(cfg["output_root"])
    patch_vel = int(cfg["train"]["patch_vel"])
    patch_pos = int(cfg["train"]["patch_pos"])
    pos_frac = float(cfg["train"]["pos_fraction"])
    samples_per_pv = int(cfg["train"].get("samples_per_pv") or 1)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {
        split: _read_csv(root / f"{split}_manifest.csv")
        for split in sorted(set(args.splits + [args.hard_negative_split]))
    }
    rows_by_filename = {
        row["filename"]: row
        for rows in rows_by_split.values()
        for row in rows
    }

    from tensorflow import keras

    model = keras.models.load_model(args.model, compile=False)

    sample_rows = _patch_sample_predictions(
        model,
        rows_by_split,
        splits=args.splits,
        patch_vel=patch_vel,
        patch_pos=patch_pos,
        pos_frac=pos_frac,
        samples_per_pv=samples_per_pv,
        batch_size=args.batch_size,
        seed=args.seed,
        thresholds=args.thresholds,
    )
    write_csv(sample_rows, run_dir / "patch_sample_predictions_val_test.csv")
    category_rows = _category_metrics(sample_rows, args.thresholds)
    write_csv(category_rows, run_dir / "category_metrics_by_cut_category.csv")
    (run_dir / "category_metrics_by_cut_category.json").write_text(json.dumps({"rows": category_rows}, indent=2))
    inspection_summary = _summarize_inspection(sample_rows, run_dir)

    selected_rows = _select_panel_rows(sample_rows, args.panel_max_per_category)
    _write_review_panels(
        model,
        selected_rows,
        rows_by_filename,
        run_dir=run_dir,
        patch_vel=patch_vel,
        patch_pos=patch_pos,
    )

    component_counts = _full_pv_components(
        model,
        rows_by_split,
        splits=args.splits,
        run_dir=run_dir,
        patch_vel=patch_vel,
        patch_pos=patch_pos,
        batch_size=args.batch_size,
        thresholds=args.thresholds,
        min_area_pix=args.component_min_area_pix,
        save_probability_maps=not args.skip_probability_map_save,
    )

    hard_negatives = _mine_hard_negatives(
        model,
        rows_by_split[args.hard_negative_split],
        run_dir=run_dir,
        split=args.hard_negative_split,
        patch_vel=patch_vel,
        patch_pos=patch_pos,
        batch_size=args.batch_size,
        thresholds=args.thresholds,
        max_candidates=args.max_hard_negatives,
    )

    _write_markdown_report(
        run_dir,
        sample_rows=sample_rows,
        category_rows=category_rows,
        inspection_summary=inspection_summary,
        component_counts=component_counts,
        hard_negatives=hard_negatives,
    )
    print(f"[diagnostics] wrote patch predictions: {run_dir / 'patch_sample_predictions_val_test.csv'}")
    print(f"[diagnostics] wrote category metrics: {run_dir / 'category_metrics_by_cut_category.csv'}")
    print(f"[diagnostics] wrote summary report: {run_dir / 'standardized_diagnostics_report.md'}")


if __name__ == "__main__":
    main()
