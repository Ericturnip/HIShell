#!/usr/bin/env python3
"""
Evaluate connected-component post-processing on saved U-Net probability maps.
The script reports patch precision and recall before filtering, after beam
filtering, after velocity checks, and after probability-mass ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
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


def _load_json(path: Path) -> dict[str, Any]:
    """Load optional metadata sidecars without stopping evaluation on one bad file."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_manifest_csv(root: Path, split: str) -> list[dict[str, str]]:
    """Read the split manifest that links PV arrays, labels, and metadata."""
    path = root / f"{split}_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest CSV: {path}")
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _zscore_finite(pv: np.ndarray) -> np.ndarray:
    """Normalize one PV cut using only finite sampled data values."""
    finite = pv[np.isfinite(pv)]
    if finite.size == 0:
        return np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mu = float(np.mean(finite))
    sigma = float(np.std(finite) + 1e-6)
    return np.nan_to_num((pv - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _truth_positive(label: np.ndarray) -> bool:
    """Convert a pixel mask into the patch-level target used by recall reports."""
    return bool(np.any(label > 0.5))


def _is_eval_exception(row: dict[str, str], meta: dict[str, Any]) -> bool:
    """Identify catalog or grazing cuts where sub-beam labels should be preserved."""
    text = " ".join(
        str(x or "")
        for x in (
            row.get("cut_type"),
            row.get("cut_category"),
            row.get("quality_flags"),
            meta.get("source"),
            meta.get("cut_category"),
            " ".join(meta.get("quality_flags") or []),
        )
    ).casefold()
    return "catalog" in text or "grazing" in text or "centered_positive" in text


def _velocity_step_kms(meta: dict[str, Any], shape: tuple[int, int]) -> float:
    """Recover the physical velocity scale represented by one model row."""
    if meta.get("velocity_bin_width_kms"):
        return abs(float(meta["velocity_bin_width_kms"]))
    axis = meta.get("velocity_axis_kms") or []
    if len(axis) > 1:
        return abs(float(np.nanmedian(np.diff(np.asarray(axis, dtype=np.float64)))))
    return float(meta.get("velocity_window_kms", 200.0)) / max(int(shape[0]), 1)


def _spatial_step_kpc(meta: dict[str, Any], shape: tuple[int, int]) -> float:
    """Recover the physical spatial scale represented by one model column."""
    return float(meta.get("spatial_window_kpc", 5.0)) / max(int(meta.get("target_spatial_pixels") or shape[1]), 1)


def _parse_fits_value(raw: str) -> float | None:
    """Parse one numeric FITS card value from a raw 80-character header line."""
    value = raw.split("/", 1)[0].strip()
    if not value:
        return None
    if value.startswith("'"):
        return None
    try:
        return float(value.replace("D", "E"))
    except ValueError:
        return None


_FITS_BEAM_CACHE: dict[str, dict[str, float | str] | None] = {}


def _read_primary_fits_header_values(path: Path, keys: set[str]) -> dict[str, float]:
    """
    Read primary FITS header values without requiring Astropy at evaluation time.
    This keeps beam-area recovery available in lightweight TensorFlow environments.
    """
    out: dict[str, float] = {}
    history_text: list[str] = []
    with path.open("rb") as fh:
        while True:
            block = fh.read(2880)
            if not block:
                break
            text = block.decode("ascii", errors="ignore")
            for i in range(0, len(text), 80):
                card = text[i : i + 80]
                key = card[:8].strip()
                if key == "END":
                    _parse_aips_clean_beam_history(history_text, out)
                    return out
                if key == "HISTORY":
                    history_text.append(card[8:].strip())
                if key in keys and len(card) > 10 and card[8] == "=":
                    val = _parse_fits_value(card[10:])
                    if val is not None:
                        out[key] = val
    _parse_aips_clean_beam_history(history_text, out)
    return out


def _parse_aips_clean_beam_history(history_text: list[str], out: dict[str, float]) -> None:
    """Recover AIPS CLEAN beam values when BMAJ and BMIN are stored in HISTORY."""
    if out.get("BMAJ") and out.get("BMIN"):
        return
    joined = "\n".join(history_text)
    match = re.search(
        r"BMAJ\s*=\s*([+-]?\d+(?:\.\d*)?(?:[ED][+-]?\d+)?)\s+"
        r"BMIN\s*=\s*([+-]?\d+(?:\.\d*)?(?:[ED][+-]?\d+)?)",
        joined,
        flags=re.IGNORECASE,
    )
    if not match:
        return
    out.setdefault("BMAJ", float(match.group(1).replace("D", "E")))
    out.setdefault("BMIN", float(match.group(2).replace("D", "E")))


def _true_beam_from_fits(path_value: str | None) -> dict[str, float | str] | None:
    """
    Convert FITS BMAJ, BMIN, and CDELT into Gaussian beam area in pixels.
    The value is cached because the same source cube appears in many PV cuts.
    """
    if not path_value:
        return None
    if path_value in _FITS_BEAM_CACHE:
        return _FITS_BEAM_CACHE[path_value]
    path = Path(path_value)
    if not path.exists():
        _FITS_BEAM_CACHE[path_value] = None
        return None
    try:
        values = _read_primary_fits_header_values(path, {"BMAJ", "BMIN", "CDELT1", "CDELT2"})
    except Exception:
        _FITS_BEAM_CACHE[path_value] = None
        return None
    bmaj = values.get("BMAJ")
    bmin = values.get("BMIN")
    cdelt = values.get("CDELT1") or values.get("CDELT2")
    if not (bmaj and bmin and cdelt):
        _FITS_BEAM_CACHE[path_value] = None
        return None
    bmaj_pix = float(bmaj) / abs(float(cdelt))
    bmin_pix = float(bmin) / abs(float(cdelt))
    area = 1.133 * bmaj_pix * bmin_pix
    if not np.isfinite(area) or area <= 0:
        _FITS_BEAM_CACHE[path_value] = None
        return None
    result = {
        "beam_area_pixels": float(np.ceil(area)),
        "beam_area_pixels_raw": float(area),
        "beam_major_pixels": float(bmaj_pix),
        "beam_minor_pixels": float(bmin_pix),
        "bmaj_deg": float(bmaj),
        "bmin_deg": float(bmin),
        "cdelt_deg_per_pixel": float(cdelt),
        "source": "fits_primary_header_bmaj_bmin_cdelt",
    }
    _FITS_BEAM_CACHE[path_value] = result
    return result


def _beam_area_pixels(meta: dict[str, Any], fallback_beam_area_pixels: float) -> tuple[float, str, dict[str, Any]]:
    """
    Choose the best beam-area estimate available for one PV cut.
    FITS headers are preferred, metadata is second, and the CLI fallback is last.
    """
    fits_beam = _true_beam_from_fits(meta.get("source_cube_path"))
    if fits_beam is not None:
        return float(fits_beam["beam_area_pixels"]), str(fits_beam["source"]), dict(fits_beam)
    major = meta.get("beam_major_model_cols")
    minor = meta.get("beam_minor_model_cols")
    if major is not None and minor is not None:
        area = (np.pi / 4.0) * float(major) * float(minor)
        if np.isfinite(area) and area > 0:
            return float(np.ceil(area)), "metadata_beam_major_minor_model_cols", {
                "beam_area_pixels_raw": float(area),
                "beam_major_pixels": float(major),
                "beam_minor_pixels": float(minor),
            }
    beam_pix = meta.get("beam_fwhm_pix")
    if beam_pix is not None:
        area = (np.pi / 4.0) * float(beam_pix) ** 2
        if np.isfinite(area) and area > 0:
            return float(np.ceil(area)), "metadata_beam_fwhm_pix", {
                "beam_area_pixels_raw": float(area),
                "beam_major_pixels": float(beam_pix),
                "beam_minor_pixels": float(beam_pix),
            }
    return float(fallback_beam_area_pixels), "fallback_cli_beam_area_pixels", {
        "beam_area_pixels_raw": float(fallback_beam_area_pixels),
    }


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    """Group thresholded probability pixels into 8-connected components."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    unvisited = mask.ravel().copy()
    components: list[np.ndarray] = []
    neighbor_offsets = (-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1)

    while True:
        seeds = np.flatnonzero(unvisited)
        if seeds.size == 0:
            break
        seed = int(seeds[0])
        stack = [seed]
        unvisited[seed] = False
        comp: list[int] = []
        while stack:
            idx = stack.pop()
            comp.append(idx)
            y, x = divmod(idx, w)
            for off in neighbor_offsets:
                nb = idx + off
                if nb < 0 or nb >= h * w:
                    continue
                ny, nx = divmod(nb, w)
                if abs(ny - y) > 1 or abs(nx - x) > 1:
                    continue
                if unvisited[nb]:
                    unvisited[nb] = False
                    stack.append(nb)
        components.append(np.asarray(comp, dtype=np.int64))
    return components


def _component_features(
    *,
    prob: np.ndarray,
    label: np.ndarray,
    threshold: float,
    row: dict[str, str],
    meta: dict[str, Any],
    fallback_beam_area_pixels: float,
    beam_area_scale: float,
    velocity_flag_kms: float,
    eval_mode: bool,
) -> list[dict[str, Any]]:
    """
    Measure every connected component in one probability map.
    The returned rows are the candidate table used for filtering and ranking.
    """
    vel_step = _velocity_step_kms(meta, prob.shape)
    spatial_step = _spatial_step_kpc(meta, prob.shape)
    beam_area, beam_source, beam_details = _beam_area_pixels(meta, fallback_beam_area_pixels)
    eval_exception = bool(eval_mode and _is_eval_exception(row, meta))
    label_bool = label > 0.5

    rows: list[dict[str, Any]] = []
    for comp_id, flat in enumerate(_component_coords(prob >= float(threshold)), start=1):
        yy, xx = np.divmod(flat, prob.shape[1])
        vals = prob[yy, xx]
        y0, y1 = int(yy.min()), int(yy.max())
        x0, x1 = int(xx.min()), int(xx.max())
        velocity_extent_pixels = int(y1 - y0 + 1)
        spatial_extent_pixels = int(x1 - x0 + 1)
        bbox_area = int(velocity_extent_pixels * spatial_extent_pixels)
        area = int(flat.size)
        mass = float(np.nansum(vals))
        edge_touching = y0 == 0 or x0 == 0 or y1 == prob.shape[0] - 1 or x1 == prob.shape[1] - 1
        velocity_min_edge_touching = y0 == 0
        velocity_max_edge_touching = y1 == prob.shape[0] - 1
        velocity_edge_touching = velocity_min_edge_touching or velocity_max_edge_touching
        velocity_both_edges_touching = velocity_min_edge_touching and velocity_max_edge_touching
        velocity_extent_kms = float(velocity_extent_pixels * vel_step)
        overlaps_label = bool(np.any(label_bool[yy, xx]))
        area_over_beam = float(area / max(beam_area, 1e-9))
        beam_pass = bool(area >= float(beam_area_scale) * beam_area or eval_exception)
        velocity_flag = bool(velocity_extent_kms > float(velocity_flag_kms))
        final_pass = bool(beam_pass and not velocity_edge_touching)
        row_out = {
            "split": row.get("split"),
            "galaxy": row.get("galaxy"),
            "filename": row.get("filename"),
            "patch_id": Path(row.get("filename", "")).stem,
            "cut_type": row.get("cut_type"),
            "cut_category": row.get("cut_category"),
            "shell_id": row.get("shell_id"),
            "patch_positive": int(_truth_positive(label)),
            "component_overlaps_label": int(overlaps_label),
            "threshold": float(threshold),
            "component_id": int(comp_id),
            "area_pixels": area,
            "beam_area_pixels": beam_area,
            "beam_area_source": beam_source,
            "beam_area_pixels_raw": beam_details.get("beam_area_pixels_raw"),
            "beam_major_pixels": beam_details.get("beam_major_pixels"),
            "beam_minor_pixels": beam_details.get("beam_minor_pixels"),
            "beam_bmaj_deg": beam_details.get("bmaj_deg"),
            "beam_bmin_deg": beam_details.get("bmin_deg"),
            "beam_cdelt_deg_per_pixel": beam_details.get("cdelt_deg_per_pixel"),
            "area_over_beam": area_over_beam,
            "velocity_extent_pixels": velocity_extent_pixels,
            "velocity_extent_kms": velocity_extent_kms,
            "spatial_extent_pixels": spatial_extent_pixels,
            "spatial_extent_kpc": float(spatial_extent_pixels * spatial_step),
            "max_probability": float(np.nanmax(vals)),
            "mean_probability": float(np.nanmean(vals)),
            "integrated_probability_mass": mass,
            "bounding_box_area": bbox_area,
            "component_area_over_bounding_box_area": float(area / max(bbox_area, 1)),
            "probability_mass_over_bounding_box_area": float(mass / max(bbox_area, 1)),
            "edge_touching": int(edge_touching),
            "velocity_min_edge_touching": int(velocity_min_edge_touching),
            "velocity_max_edge_touching": int(velocity_max_edge_touching),
            "velocity_edge_touching": int(velocity_edge_touching),
            "velocity_both_edges_touching": int(velocity_both_edges_touching),
            "velocity_extent_gt_limit_flag": int(velocity_flag),
            "edge_touching_flag": int(edge_touching),
            "eval_mode_beam_exception": int(eval_exception),
            "beam_area_pass": int(beam_pass),
            "beam_velocity_extent_hard_pass": int(beam_pass and not velocity_flag),
            "beam_velocity_edge_hard_pass": int(final_pass),
            "beam_velocity_extent_edge_hard_pass": int(beam_pass and not velocity_flag and not velocity_edge_touching),
            "final_candidate_pass": int(final_pass),
            "probability_mass_rank_score": mass,
        }
        rows.append(row_out)
    return rows


def _metrics_from_patch_predictions(truth: dict[str, bool], pred: dict[str, bool]) -> dict[str, Any]:
    """Compute patch precision, recall, and F1 from patch-level detections."""
    tp = sum(1 for key, val in truth.items() if val and pred.get(key, False))
    fp = sum(1 for key, val in truth.items() if not val and pred.get(key, False))
    fn = sum(1 for key, val in truth.items() if val and not pred.get(key, False))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _rank_metrics(
    *,
    truth: dict[str, bool],
    components: list[dict[str, Any]],
    top_ns: list[int],
) -> dict[str, Any]:
    """
    Score the human-review candidate list after probability-mass ranking.
    This reports how recall and precision change when only the top N components are kept.
    """
    ranked = sorted(
        [c for c in components if c["final_candidate_pass"]],
        key=lambda c: c["probability_mass_rank_score"],
        reverse=True,
    )
    out: dict[str, Any] = {}
    for n in top_ns:
        chosen = ranked[: max(0, int(n))]
        pred_by_patch: dict[str, bool] = defaultdict(bool)
        for comp in chosen:
            pred_by_patch[str(comp["filename"])] = True
        patch_metrics = _metrics_from_patch_predictions(truth, pred_by_patch)
        cand_tp = sum(1 for comp in chosen if int(comp["component_overlaps_label"]))
        cand_fp = len(chosen) - cand_tp
        out[str(n)] = {
            "candidate_count": len(chosen),
            "candidate_precision": cand_tp / max(1, len(chosen)),
            "candidate_tp": cand_tp,
            "candidate_fp": cand_fp,
            "patch_precision": patch_metrics["precision"],
            "patch_recall": patch_metrics["recall"],
            "patch_f1": patch_metrics["f1"],
            "patch_tp": patch_metrics["tp"],
            "patch_fp": patch_metrics["fp"],
            "patch_fn": patch_metrics["fn"],
        }
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write component candidates with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _threshold_tag(threshold: float) -> str:
    """Convert a numeric threshold into a filename-safe tag."""
    return str(threshold).replace(".", "p")


def evaluate_split(
    *,
    cfg: dict[str, Any],
    model_path: Path,
    split: str,
    out_dir: Path,
    thresholds: list[float],
    batch_size: int,
    fallback_beam_area_pixels: float,
    beam_area_scale: float,
    velocity_flag_kms: float,
    top_ns: list[int],
    eval_mode: bool,
) -> dict[str, Any]:
    """
    Run component filtering for one split and write its candidate CSV files.
    The returned metrics form the before-and-after table used in the report.
    """
    from tensorflow import keras

    root = Path(cfg["output_root"])
    rows = _read_manifest_csv(root, split)
    model = keras.models.load_model(model_path, compile=False)

    patch_truth: dict[str, bool] = {}
    patch_preds = {
        threshold: {
            "before_filtering": {},
            "after_beam_area_filter": {},
            "after_beam_velocity_extent_filter": {},
            "after_beam_velocity_edge_filter": {},
            "after_beam_velocity_extent_edge_filter": {},
        }
        for threshold in thresholds
    }
    candidates_by_threshold: dict[float, list[dict[str, Any]]] = {threshold: [] for threshold in thresholds}

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        pvs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []
        kept_rows: list[dict[str, str]] = []
        for row in batch_rows:
            pv_path = Path(row["image_path"])
            label_path = Path(row["mask_path"])
            meta_path = Path(row["metadata_path"])
            if not pv_path.exists() or not label_path.exists():
                continue
            pv = np.load(pv_path)
            label = np.load(label_path)
            pvs.append(_zscore_finite(pv)[..., None])
            labels.append(label.astype(np.uint8))
            metas.append(_load_json(meta_path))
            kept_rows.append(row)
        if not pvs:
            continue
        probs = model.predict(np.stack(pvs, axis=0).astype(np.float32), batch_size=batch_size, verbose=0)[..., 0]
        for row, meta, label, prob in zip(kept_rows, metas, labels, probs):
            fname = row["filename"]
            patch_truth[fname] = _truth_positive(label)
            for threshold in thresholds:
                comps = _component_features(
                    prob=prob,
                    label=label,
                    threshold=threshold,
                    row=row,
                    meta=meta,
                    fallback_beam_area_pixels=fallback_beam_area_pixels,
                    beam_area_scale=beam_area_scale,
                    velocity_flag_kms=velocity_flag_kms,
                    eval_mode=eval_mode,
                )
                candidates_by_threshold[threshold].extend(comps)
                patch_preds[threshold]["before_filtering"][fname] = bool(comps)
                patch_preds[threshold]["after_beam_area_filter"][fname] = any(c["beam_area_pass"] for c in comps)
                patch_preds[threshold]["after_beam_velocity_extent_filter"][fname] = any(
                    c["beam_velocity_extent_hard_pass"] for c in comps
                )
                patch_preds[threshold]["after_beam_velocity_edge_filter"][fname] = any(
                    c["beam_velocity_edge_hard_pass"] for c in comps
                )
                patch_preds[threshold]["after_beam_velocity_extent_edge_filter"][fname] = any(
                    c["beam_velocity_extent_edge_hard_pass"] for c in comps
                )
        if (start // max(1, batch_size)) % 25 == 0:
            print(f"[postprocess] split={split} processed {min(start + batch_size, len(rows))}/{len(rows)} patches", flush=True)

    split_result: dict[str, Any] = {
        "split": split,
        "model": str(model_path.resolve()),
        "output_root": str(root.resolve()),
        "patches": len(patch_truth),
        "positive_patches": int(sum(patch_truth.values())),
        "negative_patches": int(len(patch_truth) - sum(patch_truth.values())),
        "thresholds": {},
    }

    for threshold in thresholds:
        candidates = candidates_by_threshold[threshold]
        tag = _threshold_tag(threshold)
        ranked = sorted(candidates, key=lambda c: c["probability_mass_rank_score"], reverse=True)
        for rank, comp in enumerate(ranked, start=1):
            comp["probability_mass_rank"] = rank
        csv_path = out_dir / f"component_candidates_{split}_threshold_{tag}.csv"
        _write_csv(ranked, csv_path)

        metrics = {
            stage: _metrics_from_patch_predictions(patch_truth, preds)
            for stage, preds in patch_preds[threshold].items()
        }
        metrics["after_beam_probability_mass_top_n"] = _rank_metrics(
            truth=patch_truth,
            components=candidates,
            top_ns=top_ns,
        )
        split_result["thresholds"][str(threshold)] = {
            "candidate_csv": str(csv_path.resolve()),
            "candidate_count_before_filtering": len(candidates),
            "candidate_count_after_beam_area_filter": int(sum(c["beam_area_pass"] for c in candidates)),
            "candidate_count_after_beam_velocity_extent_filter": int(sum(c["beam_velocity_extent_hard_pass"] for c in candidates)),
            "candidate_count_after_beam_velocity_edge_filter": int(sum(c["beam_velocity_edge_hard_pass"] for c in candidates)),
            "candidate_count_after_beam_velocity_extent_edge_filter": int(sum(c["beam_velocity_extent_edge_hard_pass"] for c in candidates)),
            "candidate_count_final_candidate_pass": int(sum(c["final_candidate_pass"] for c in candidates)),
            "velocity_extent_gt_limit_flag_count": int(sum(c["velocity_extent_gt_limit_flag"] for c in candidates)),
            "edge_touching_flag_count": int(sum(c["edge_touching_flag"] for c in candidates)),
            "velocity_edge_touching_flag_count": int(sum(c["velocity_edge_touching"] for c in candidates)),
            "velocity_both_edges_touching_flag_count": int(sum(c["velocity_both_edges_touching"] for c in candidates)),
            "beam_area_source_counts": dict(
                (source, sum(1 for c in candidates if c["beam_area_source"] == source))
                for source in sorted({c["beam_area_source"] for c in candidates})
            ),
            "metrics": metrics,
        }
    return split_result


def main() -> None:
    """Run post-processing evaluation for validation, test, and stress splits."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--splits", nargs="*", default=["val", "test", "stress"])
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.075, 0.05])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fallback-beam-area-pixels", type=float, default=35.0)
    ap.add_argument("--beam-area-scale", type=float, default=1.2)
    ap.add_argument("--velocity-flag-kms", type=float, default=70.0)
    ap.add_argument("--top-n", nargs="*", type=int, default=[50, 100, 250, 500, 1000, 2000])
    ap.add_argument("--no-eval-mode-exceptions", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    out_dir = Path(args.run_dir)
    all_results = {
        "config": str(Path(args.config).resolve()),
        "model": str(Path(args.model).resolve()),
        "thresholds": args.thresholds,
        "fallback_beam_area_pixels": args.fallback_beam_area_pixels,
        "beam_area_scale": args.beam_area_scale,
        "velocity_flag_kms": args.velocity_flag_kms,
        "eval_mode_beam_exceptions": not args.no_eval_mode_exceptions,
        "splits": {},
    }
    for split in args.splits:
        result = evaluate_split(
            cfg=cfg,
            model_path=Path(args.model),
            split=split,
            out_dir=out_dir,
            thresholds=[float(t) for t in args.thresholds],
            batch_size=int(args.batch_size),
            fallback_beam_area_pixels=float(args.fallback_beam_area_pixels),
            beam_area_scale=float(args.beam_area_scale),
            velocity_flag_kms=float(args.velocity_flag_kms),
            top_ns=[int(n) for n in args.top_n],
            eval_mode=not args.no_eval_mode_exceptions,
        )
        all_results["splits"][split] = result

    out_json = out_dir / "component_filter_metrics.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"[postprocess] wrote {out_json.resolve()}")


if __name__ == "__main__":
    main()
