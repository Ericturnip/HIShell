#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.diagnostic_utils import connected_components, load_json, write_csv


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def _component_rows(mask: np.ndarray) -> list[dict[str, Any]]:
    labels, n = connected_components(mask > 0)
    rows = []
    for comp_id in range(1, n + 1):
        yy, xx = np.where(labels == comp_id)
        if yy.size == 0:
            continue
        rows.append(
            {
                "component_id": comp_id,
                "component_area_pix": int(yy.size),
                "bbox_v0": int(yy.min()),
                "bbox_pos0": int(xx.min()),
                "bbox_v1": int(yy.max()),
                "bbox_pos1": int(xx.max()),
                "spatial_extent_pix": int(xx.max() - xx.min() + 1),
                "velocity_extent_pix": int(yy.max() - yy.min() + 1),
                "centroid_v": float(np.mean(yy)),
                "centroid_pos": float(np.mean(xx)),
            }
        )
    return rows


def _classification(comp: dict[str, Any], row: dict[str, str], meta: dict[str, Any], total_components: int) -> tuple[str, str]:
    area = int(comp["component_area_pix"])
    spatial_extent = int(comp["spatial_extent_pix"])
    velocity_extent = int(comp["velocity_extent_pix"])
    beam_pix = meta.get("beam_fwhm_pix")
    if beam_pix:
        beam_area = float(beam_pix) ** 2
        resolvable_area = max(4.0, beam_area)
        speck_area = max(2.0, 0.25 * beam_area)
    else:
        beam_area = None
        resolvable_area = 12.0
        speck_area = 3.0

    category = str(row.get("cut_category") or meta.get("cut_category") or "")
    shell_id = row.get("shell_id") or meta.get("source_shell_id")
    associated = shell_id not in ("", None, "None")
    grazing_context = any(token in category for token in ("grazing", "spatial_offset", "angle_offset", "velocity_offset", "fine_grid"))

    if area >= resolvable_area or (spatial_extent >= 3 and velocity_extent >= 3 and area >= 8):
        return "resolvable_positive", "component is larger than the beam/proxy area or spatially and spectrally coherent"
    if associated and grazing_context:
        return "grazing_positive", "small component is associated with a catalog shell in an offset/grazing/deployment-like cut"
    if area <= speck_area and not associated and total_components <= 2:
        return "isolated_subbeam_speck", "tiny component without catalog-shell association"
    return "ambiguous_small_positive", "small component needs visual review before cleaning"


def audit(out_root: Path) -> dict[str, Any]:
    manifest_paths = [out_root / f"{split}_manifest.csv" for split in ("train", "val", "test")]
    rows = []
    summary = Counter()
    masks_seen = 0
    positive_masks = 0
    changed_labels = 0

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        for manifest_row in _read_manifest(manifest_path):
            masks_seen += 1
            mask_path = Path(manifest_row["mask_path"])
            if not mask_path.exists():
                continue
            mask = np.load(mask_path)
            if not bool(mask.any()):
                continue
            positive_masks += 1
            meta = load_json(Path(manifest_row["metadata_path"]))
            comps = _component_rows(mask)
            largest = max((c["component_area_pix"] for c in comps), default=0)
            total_area = int(mask.sum())
            for comp in comps:
                cls, reason = _classification(comp, manifest_row, meta, len(comps))
                summary[cls] += 1
                vel_axis = meta.get("velocity_axis_kms") or []
                if len(vel_axis) > 1:
                    dv = abs(float(np.nanmedian(np.diff(np.asarray(vel_axis, dtype=float)))))
                    velocity_extent_kms = comp["velocity_extent_pix"] * dv
                else:
                    velocity_extent_kms = None
                spatial_extent_kpc = None
                pix_window = meta.get("pixel_window_size")
                if pix_window:
                    spatial_extent_kpc = comp["spatial_extent_pix"] * float(meta.get("spatial_window_kpc", 5.0)) / float(pix_window)
                beam_area_units = None
                if meta.get("beam_fwhm_pix"):
                    beam_area_units = comp["component_area_pix"] / max(float(meta["beam_fwhm_pix"]) ** 2, 1e-9)

                dist_shell = None
                center = meta.get("source_shell_center_pix")
                if center and meta.get("posxy_pix"):
                    posxy = np.asarray(meta["posxy_pix"], dtype=float)
                    pos_idx = min(max(int(round(comp["centroid_pos"])), 0), len(posxy) - 1)
                    dist_shell = float(np.hypot(posxy[pos_idx, 0] - center[0], posxy[pos_idx, 1] - center[1]))

                rows.append(
                    {
                        "split": manifest_row.get("split"),
                        "galaxy": manifest_row.get("galaxy"),
                        "filename": manifest_row.get("filename"),
                        "cut_category": manifest_row.get("cut_category"),
                        "shell_id": manifest_row.get("shell_id"),
                        "total_positive_mask_area_pix": total_area,
                        "connected_component_count": len(comps),
                        "largest_component_area_pix": largest,
                        **comp,
                        "spatial_extent_kpc": spatial_extent_kpc,
                        "velocity_extent_kms": velocity_extent_kms,
                        "area_in_beam_units": beam_area_units,
                        "distance_from_expected_shell_center_pix": dist_shell,
                        "known_catalog_shell_associated": manifest_row.get("shell_id") not in ("", None, "None"),
                        "classification": cls,
                        "classification_reason": reason,
                        "label_changed": False,
                    }
                )

    csv_path = out_root / "label_component_audit.csv"
    json_path = out_root / "label_component_audit.json"
    write_csv(rows, csv_path)
    result = {
        "masks_seen": masks_seen,
        "positive_masks": positive_masks,
        "component_count": len(rows),
        "classification_counts": dict(summary),
        "label_cleaning_mode": "audit_only",
        "labels_changed": changed_labels,
        "csv": str(csv_path.resolve()),
    }
    json_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("training_data/standardized_5kpc_200kms"))
    args = ap.parse_args()
    result = audit(args.out_root)
    print(f"[label-audit] masks={result['masks_seen']} positives={result['positive_masks']} components={result['component_count']}")
    print(f"[label-audit] classifications={result['classification_counts']}")
    print(f"[label-audit] wrote {result['csv']}")


if __name__ == "__main__":
    main()
