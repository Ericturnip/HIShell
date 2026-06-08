#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _is_base_npy(path: Path) -> bool:
    return path.suffix == ".npy" and not path.name.endswith("_posxy.npy")


def validate_galaxy(gal_dir: Path, *, max_examples: int = 5) -> dict:
    pv_dir = gal_dir / "pv"
    lab_dir = gal_dir / "labels"
    type_dir = gal_dir / "label_types"
    summary_path = gal_dir / "label_summary.json"
    problems: list[str] = []
    examples: list[str] = []
    source_counts: Counter[str] = Counter()
    source_pos_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    pv_files = sorted(p for p in pv_dir.glob("*.npy") if _is_base_npy(p)) if pv_dir.exists() else []
    n_pairs = n_pos = n_neg = n_nan = n_shape_bad = 0
    label_pixels = 0
    total_pixels = 0

    for pv_path in pv_files:
        lab_path = lab_dir / pv_path.name
        if not lab_path.exists():
            problems.append(f"missing label for {pv_path.name}")
            continue

        pv = np.load(pv_path, mmap_mode="r")
        lab = np.load(lab_path, mmap_mode="r")
        n_pairs += 1

        if pv.shape != lab.shape:
            n_shape_bad += 1
            problems.append(f"shape mismatch {pv_path.name}: pv={pv.shape}, label={lab.shape}")
            continue
        if not np.isfinite(pv).all():
            n_nan += 1
            problems.append(f"non-finite PV values in {pv_path.name}")
        if not np.isfinite(lab).all():
            n_nan += 1
            problems.append(f"non-finite label values in {pv_path.name}")

        lab_any = bool(np.asarray(lab).any())
        n_pos += int(lab_any)
        n_neg += int(not lab_any)
        label_pixels += int(np.count_nonzero(lab))
        total_pixels += int(lab.size)

        meta_path = lab_path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            src = meta.get("source", "unknown")
            source_counts[src] += 1
            if lab_any:
                source_pos_counts[src] += 1
            for obj in meta.get("labels", []):
                typ = str(obj.get("type", "unknown"))
                type_counts[typ] += 1

        typ_path = type_dir / pv_path.name
        if typ_path.exists():
            typ = np.load(typ_path, mmap_mode="r")
            if typ.shape != pv.shape:
                problems.append(f"type-mask shape mismatch {pv_path.name}: {typ.shape} vs {pv.shape}")

        if lab_any and len(examples) < max_examples:
            examples.append(str((gal_dir / "qa_labels" / f"{pv_path.stem}_label_overlay.png").resolve()))

    catalog_warnings = []
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        catalog_warnings = summary.get("warnings", [])

    return {
        "galaxy": gal_dir.name,
        "pv_files": len(pv_files),
        "pairs": n_pairs,
        "positive": n_pos,
        "negative": n_neg,
        "shape_bad": n_shape_bad,
        "nonfinite": n_nan,
        "label_pixel_fraction": (label_pixels / total_pixels) if total_pixels else 0.0,
        "sources": dict(source_counts),
        "positive_sources": dict(source_pos_counts),
        "label_objects_by_type": dict(type_counts),
        "catalog_warnings": catalog_warnings,
        "problems": problems[:25],
        "qa_examples": examples,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="training_data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.data_root).resolve()
    galaxies = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name != "combined" and (p / "pv").exists() and (p / "labels").exists()
    )
    results = [validate_galaxy(g) for g in galaxies]

    totals = defaultdict(int)
    source_totals: Counter[str] = Counter()
    type_totals: Counter[str] = Counter()
    warnings = []
    problems = []
    for r in results:
        for key in ("pv_files", "pairs", "positive", "negative", "shape_bad", "nonfinite"):
            totals[key] += int(r[key])
        source_totals.update(r["sources"])
        type_totals.update(r["label_objects_by_type"])
        warnings.extend((r["galaxy"], w) for w in r["catalog_warnings"])
        problems.extend((r["galaxy"], p) for p in r["problems"])

    report = {
        "data_root": str(root),
        "galaxies": len(results),
        "totals": dict(totals),
        "sources": dict(source_totals),
        "label_objects_by_type": dict(type_totals),
        "catalog_warning_count": len(warnings),
        "problem_count": len(problems),
        "galaxy_results": results,
    }

    out = Path(args.out).resolve() if args.out else root / "training_readiness_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[validate] wrote {out}")
    print(
        "[validate] pairs={pairs} positive={positive} negative={negative} "
        "shape_bad={shape_bad} nonfinite={nonfinite} warnings={warnings} problems={problems}".format(
            **totals,
            warnings=len(warnings),
            problems=len(problems),
        )
    )
    if problems:
        raise SystemExit("[validate] problems found; inspect report before training")


if __name__ == "__main__":
    main()
