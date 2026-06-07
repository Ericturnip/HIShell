#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pv.shell_catalog import load_bagetakos_table7


TYPE_LABELS = {
    1: "type_1_both_sides_stalled_or_no_velocity_caps",
    2: "type_2_one_side_expanding",
    3: "type_3_both_sides_expanding",
}

TYPE_REVIEW_HINTS = {
    1: "Look for a PV cavity/gap where neither side shows a clear expansion cap.",
    2: "Look for a cavity with one visible expanding velocity side.",
    3: "Look for both expanding sides, usually the cleanest closed/elliptical PV signature.",
}


def _norm_galaxy(name: str) -> str:
    key = str(name).strip().casefold().replace(" ", "_")
    aliases = {
        "holmberg_i": "ho_i",
        "holmberg_ii": "ho_ii",
        "ddo_53": "ddo53",
        "ddo_154": "ddo154",
        "ic_2574": "ic_2574",
    }
    return aliases.get(key, key)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
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


def _reference_catalog(table7: Path) -> list[dict[str, Any]]:
    df = load_bagetakos_table7(table7)
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        shell_type = _int(rec.get("shell_type"), -1)
        galaxy_key = _norm_galaxy(str(rec.get("Name", "")))
        rows.append(
            {
                "galaxy": galaxy_key,
                "source_galaxy_name": rec.get("Name"),
                "shell_id": _int(rec.get("shell_id")),
                "paper_type": shell_type,
                "paper_type_label": TYPE_LABELS.get(shell_type, "unknown"),
                "review_hint": TYPE_REVIEW_HINTS.get(shell_type, ""),
                "ra_deg": rec.get("ra_deg"),
                "dec_deg": rec.get("dec_deg"),
                "velocity_center_kms": rec.get("vel_center_kms"),
                "diameter_pc": rec.get("d_pc"),
                "expansion_velocity_kms": rec.get("vexp_kms"),
                "position_angle_deg": rec.get("pa_deg"),
                "axis_ratio": rec.get("axis_ratio"),
                "galactocentric_radius_kpc": rec.get("R_kpc"),
                "nHI": rec.get("nHI"),
                "kinematic_age_myr": rec.get("tkin"),
                "log_energy_erg": rec.get("logE"),
                "log_HI_mass": rec.get("logMHI"),
                "recommended_pv_center_velocity_kms": rec.get("vel_center_kms"),
                "recommended_pv_spatial_window_kpc": 5.0,
                "recommended_pv_velocity_window_kms": 200.0,
                "recommended_major_axis_pa_deg": rec.get("pa_deg"),
                "recommended_minor_axis_pa_deg": (_float(rec.get("pa_deg")) + 90.0) % 180.0,
            }
        )
    return sorted(rows, key=lambda r: (str(r["galaxy"]), int(r["shell_id"])))


def _manifest_shells(training_root: Path, splits: list[str]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for split in splits:
        for row in _read_csv(training_root / f"{split}_manifest.csv"):
            if not _truthy(row.get("positive")) or not row.get("shell_id"):
                continue
            key = (_norm_galaxy(row.get("galaxy", "")), _int(row.get("shell_id")))
            item = out.setdefault(
                key,
                {
                    "galaxy": key[0],
                    "shell_id": key[1],
                    "generated_splits": set(),
                    "generated_cut_categories": Counter(),
                    "generated_patch_count": 0,
                },
            )
            item["generated_splits"].add(split)
            item["generated_cut_categories"][row.get("cut_category", "")] += 1
            item["generated_patch_count"] += 1
    return out


def _candidate_rows(run_dir: Path, splits: list[str], threshold_tag: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in splits:
        path = run_dir / f"component_candidates_{split}_threshold_{threshold_tag}.csv"
        rows.extend(_read_csv(path))
    return rows


def _aggregate_known_shell_candidates(candidates: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        if not row.get("shell_id"):
            continue
        if not _truthy(row.get("final_candidate_pass")):
            continue
        key = (_norm_galaxy(row.get("galaxy", "")), _int(row.get("shell_id")))
        grouped[key].append(row)

    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        best = max(rows, key=lambda r: _float(r.get("probability_mass_rank_score")))
        supporting_patches = {r.get("filename") for r in rows if r.get("filename")}
        overlapping = [r for r in rows if _truthy(r.get("component_overlaps_label"))]
        out[key] = {
            "detected_after_filters": 1,
            "best_split": best.get("split"),
            "best_patch_filename": best.get("filename"),
            "best_cut_category": best.get("cut_category"),
            "best_component_id": best.get("component_id"),
            "best_probability_mass": _float(best.get("integrated_probability_mass")),
            "best_max_probability": _float(best.get("max_probability")),
            "best_mean_probability": _float(best.get("mean_probability")),
            "best_area_over_beam": _float(best.get("area_over_beam")),
            "best_velocity_extent_kms": _float(best.get("velocity_extent_kms")),
            "best_spatial_extent_kpc": _float(best.get("spatial_extent_kpc")),
            "best_fill_factor": _float(best.get("component_area_over_bounding_box_area")),
            "best_probability_mass_density": _float(best.get("probability_mass_over_bounding_box_area")),
            "best_edge_touching": _int(best.get("edge_touching")),
            "best_velocity_edge_touching": _int(best.get("velocity_edge_touching")),
            "best_velocity_extent_gt_70kms": _int(best.get("velocity_extent_gt_limit_flag")),
            "supporting_candidate_components": len(rows),
            "supporting_patch_count": len(supporting_patches),
            "overlapping_label_component_count": len(overlapping),
            "supporting_cut_categories": ";".join(
                f"{name}:{count}" for name, count in Counter(r.get("cut_category", "") for r in rows).most_common()
            ),
        }
    return out


def _candidate_quality_class(row: dict[str, Any]) -> str:
    if not _truthy(row.get("detected_after_filters")):
        return "not_recovered_in_current_candidates"
    if _int(row.get("best_velocity_edge_touching")):
        return "reject_or_low_priority_velocity_edge"
    if _int(row.get("best_velocity_extent_gt_70kms")):
        return "review_velocity_broad"
    mass = _float(row.get("best_probability_mass"))
    area_over_beam = _float(row.get("best_area_over_beam"))
    fill = _float(row.get("best_fill_factor"))
    if mass >= 500 and area_over_beam >= 5 and fill >= 0.25:
        return "high_confidence_review"
    if mass >= 100 and area_over_beam >= 1.2:
        return "medium_confidence_review"
    return "low_confidence_review"


def _blind_review_candidates(candidates: list[dict[str, str]], max_rows: int) -> list[dict[str, Any]]:
    rows = [r for r in candidates if _truthy(r.get("final_candidate_pass")) and not _truthy(r.get("component_overlaps_label"))]
    rows = sorted(rows, key=lambda r: _float(r.get("probability_mass_rank_score")), reverse=True)
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows[:max_rows], start=1):
        out.append(
            {
                "review_rank": rank,
                "split": row.get("split"),
                "galaxy": _norm_galaxy(row.get("galaxy", "")),
                "filename": row.get("filename"),
                "cut_category": row.get("cut_category"),
                "component_id": row.get("component_id"),
                "probability_mass": _float(row.get("integrated_probability_mass")),
                "max_probability": _float(row.get("max_probability")),
                "mean_probability": _float(row.get("mean_probability")),
                "area_over_beam": _float(row.get("area_over_beam")),
                "velocity_extent_kms": _float(row.get("velocity_extent_kms")),
                "spatial_extent_kpc": _float(row.get("spatial_extent_kpc")),
                "edge_touching": _int(row.get("edge_touching")),
                "velocity_edge_touching": _int(row.get("velocity_edge_touching")),
                "velocity_extent_gt_70kms": _int(row.get("velocity_extent_gt_limit_flag")),
                "review_priority": _candidate_quality_class(
                    {
                        "detected_after_filters": 1,
                        "best_probability_mass": row.get("integrated_probability_mass"),
                        "best_area_over_beam": row.get("area_over_beam"),
                        "best_fill_factor": row.get("component_area_over_bounding_box_area"),
                        "best_velocity_edge_touching": row.get("velocity_edge_touching"),
                        "best_velocity_extent_gt_70kms": row.get("velocity_extent_gt_limit_flag"),
                    }
                ),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table7", default="../J_AJ_141_23_table7.dat.txt")
    ap.add_argument("--training-root", default="training_data/standardized_5kpc_200kms_clean_physical_baseline")
    ap.add_argument("--run-dir", default="runs/pv_unet_clean_physical_baseline_20260522_022751")
    ap.add_argument("--splits", nargs="*", default=["val", "test", "stress"])
    ap.add_argument("--threshold-tag", default="0p075")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-blind-review", type=int, default=1000)
    args = ap.parse_args()

    table7 = Path(args.table7)
    training_root = Path(args.training_root)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "review_catalogs"
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = _reference_catalog(table7)
    by_ref = {(row["galaxy"], int(row["shell_id"])): row for row in reference}
    generated = _manifest_shells(training_root, args.splits)
    candidates = _candidate_rows(run_dir, args.splits, args.threshold_tag)
    detected = _aggregate_known_shell_candidates(candidates)

    finished: list[dict[str, Any]] = []
    for key, gen in sorted(generated.items()):
        ref = by_ref.get(key, {"galaxy": key[0], "shell_id": key[1]})
        row = dict(ref)
        row["generated_patch_count"] = gen["generated_patch_count"]
        row["generated_splits"] = ";".join(sorted(gen["generated_splits"]))
        row["generated_cut_categories"] = ";".join(
            f"{name}:{count}" for name, count in gen["generated_cut_categories"].most_common()
        )
        row.update(
            detected.get(
                key,
                {
                    "detected_after_filters": 0,
                    "supporting_candidate_components": 0,
                    "supporting_patch_count": 0,
                    "overlapping_label_component_count": 0,
                },
            )
        )
        row["candidate_quality_class"] = _candidate_quality_class(row)
        finished.append(row)

    blind = _blind_review_candidates(candidates, int(args.max_blind_review))
    type_counts = Counter(row.get("paper_type") for row in finished)
    detection_by_type: dict[str, dict[str, int]] = {}
    for shell_type in sorted(k for k in type_counts if k not in ("", None)):
        typed = [r for r in finished if r.get("paper_type") == shell_type]
        detection_by_type[str(shell_type)] = {
            "total_generated_shells": len(typed),
            "detected_after_filters": sum(_truthy(r.get("detected_after_filters")) for r in typed),
        }

    examples = [
        r for r in reference
        if (r["galaxy"], int(r["shell_id"])) in {("ic_2574", 21), ("ic_2574", 23), ("ngc_2403", 90)}
    ]

    _write_csv(reference, out_dir / "bagetakos_reference_shell_catalog.csv")
    _write_csv(finished, out_dir / f"finished_known_shell_catalog_threshold_{args.threshold_tag}.csv")
    _write_csv(blind, out_dir / f"blind_review_candidates_threshold_{args.threshold_tag}.csv")
    _write_csv(examples, out_dir / "figure2_type_examples.csv")

    summary = {
        "reference_shells": len(reference),
        "generated_shells_in_requested_splits": len(finished),
        "detected_known_shells_after_filters": sum(_truthy(r.get("detected_after_filters")) for r in finished),
        "blind_review_candidates_written": len(blind),
        "type_detection_counts": detection_by_type,
        "outputs": {
            "reference_catalog": str((out_dir / "bagetakos_reference_shell_catalog.csv").resolve()),
            "finished_known_shell_catalog": str((out_dir / f"finished_known_shell_catalog_threshold_{args.threshold_tag}.csv").resolve()),
            "blind_review_candidates": str((out_dir / f"blind_review_candidates_threshold_{args.threshold_tag}.csv").resolve()),
            "figure2_examples": str((out_dir / "figure2_type_examples.csv").resolve()),
        },
    }
    (out_dir / f"review_catalog_summary_threshold_{args.threshold_tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
