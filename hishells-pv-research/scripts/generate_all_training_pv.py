#!/usr/bin/env python
"""Generate hybrid PV slices and labels for every catalog galaxy with a cube.

This is intentionally a data-generation/diagnostics script only. It does not
start CNN training.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


NAME_TO_CUBE_STEM = {
    "NGC 628": "NGC_628",
    "NGC 2366": "NGC_2366",
    "NGC 2403": "NGC_2403",
    "Holmberg II": "HO_II",
    "DDO 53": "DDO53",
    "NGC 2841": "NGC_2841",
    "Holmberg I": "HO_I",
    "NGC 2976": "NGC_2976",
    "NGC 3031": "NGC_3031",
    "NGC 3184": "NGC_3184",
    "IC 2574": "IC_2574",
    "NGC 3521": "NGC_3521",
    "NGC 3627": "NGC_3627",
    "NGC 4214": "NGC_4214",
    "NGC 4449": "NGC_4449",
    "NGC 4736": "NGC_4736",
    "DDO 154": "DDO154",
    "NGC 5194": "NGC_5194",
    "NGC 6946": "NGC_6946",
    "NGC 7793": "NGC_7793",
}


def hms_to_deg(h: str, m: str, s: str) -> float:
    return 15.0 * (float(h) + float(m) / 60.0 + float(s) / 3600.0)


def dms_to_deg(sign: str, d: str, m: str, s: str) -> float:
    mult = -1.0 if sign == "-" else 1.0
    return mult * (abs(float(d)) + float(m) / 60.0 + float(s) / 3600.0)


def parse_table2(table2: Path) -> list[dict]:
    rows: list[dict] = []
    for line in table2.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-") or stripped.startswith("Name"):
            continue
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 12:
            continue
        coord = parts[2].split()
        if len(coord) != 6:
            continue
        try:
            dec_sign = "-" if coord[3].startswith("-") else "+"
            dec_d = coord[3].lstrip("+-")
            rows.append(
                {
                    "name": parts[0],
                    "stem": NAME_TO_CUBE_STEM.get(parts[0], parts[0].replace(" ", "_")),
                    "ra_deg": hms_to_deg(coord[0], coord[1], coord[2]),
                    "dec_deg": dms_to_deg(dec_sign, dec_d, coord[4], coord[5]),
                    "distance_mpc": float(parts[4]),
                    "inc_deg": float(parts[5]),
                    "pa_deg": float(parts[6]),
                }
            )
        except Exception:
            continue
    return rows


def find_cube(root: Path, stem: str) -> Path | None:
    matches = sorted(root.glob(f"{stem}_NA_CUBE_THINGS.FITS")) + sorted(root.glob(f"{stem}_NA_CUBE_THINGS.fits"))
    return matches[0] if matches else None


def make_config(gal: dict, cube: Path, args: argparse.Namespace, out_root: Path) -> dict:
    gal_id = gal["stem"].lower()
    return {
        "cube_path": str(cube.resolve()),
        "output_root": str((out_root / gal_id).resolve()),
        "catalogs": {
            "holes_dat": str(args.table7.resolve()),
            "kin_dat": None,
            "target_galaxy": gal["name"],
            "use_circular": True,
            "hv_scale": "auto",
            "hv_offset_kms": 0.0,
        },
        "galaxy": {
            "name": gal["name"],
            "ra_deg": gal["ra_deg"],
            "dec_deg": gal["dec_deg"],
            "pa_deg": gal["pa_deg"],
            "inc_deg": gal["inc_deg"],
            "vsys_kms": None,
            "distance_mpc": gal["distance_mpc"],
            "beam_fwhm_arcsec": None,
        },
        "pv": {
            "grid": {
                "frame": "galaxy",
                "pa_convention": "astro",
                "pa_delta_deg": 0,
                "x_extent_arcsec": args.grid_extent_arcsec,
                "y_extent_arcsec": args.grid_extent_arcsec,
                "x_step_arcsec": args.grid_step_arcsec,
                "y_step_arcsec": args.grid_step_arcsec,
                "slit_width_pix": 3,
                "margin_arcsec": 0,
                "pos_step_pix": 1.0,
            },
            "axes": {"include_major_minor": True},
            "shell_cuts": {
                "enabled": True,
                "keep_types": [1, 2, 3],
                "orientations": ["shell_major", "shell_minor", "galaxy_major", "galaxy_minor"],
                "offset_fractions": [-0.5, 0.0, 0.5],
                "length_scale_radius": 4.0,
                "min_half_length_pix": 32,
                "max_half_length_pix": 160,
                "slit_width_pix": 3,
                "pos_step_pix": 1.0,
                "qa_max_plots": args.qa_pv_shell_plots,
            },
            "negatives": {
                "enabled": True,
                "n_per_shell": args.negatives_per_shell,
                "max_total": args.max_negatives_per_galaxy,
                "galaxy_mask_percentile": 45.0,
                "avoid_brightest_percentile": 99.5,
                "min_sep_arcsec": 20.0,
                "sep_radius_factor": 1.5,
                "half_length_pix": 64,
                "slit_width_pix": 3,
                "pos_step_pix": 1.0,
                "seed": 12345,
                "max_attempts_factor": 100,
                "qa_max_plots": args.qa_pv_negative_plots,
            },
            "label": {
                "keep_types": [1, 2, 3],
                "catalog_pa_convention": "astro",
                "min_axis_ratio_circular": 0.80,
                "type1_velocity_half_width_kms": 10.0,
                "type1_velocity_half_width_channels": 2,
                "vexp_fallback_channels": 3,
                "dilate_vel_channels": 1,
                "qa_max_plots": args.qa_label_plots,
            },
        },
    }


def run_module(module: str, config: Path, cwd: Path) -> None:
    cmd = [sys.executable, "-m", module, "--config", str(config)]
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--table2", type=Path, default=None)
    ap.add_argument("--table7", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--grid-step-arcsec", type=float, default=30.0)
    ap.add_argument("--grid-extent-arcsec", type=float, default=420.0)
    ap.add_argument("--qa-pv-shell-plots", type=int, default=30)
    ap.add_argument("--qa-pv-negative-plots", type=int, default=30)
    ap.add_argument("--qa-label-plots", type=int, default=80)
    ap.add_argument("--negatives-per-shell", type=int, default=12)
    ap.add_argument("--max-negatives-per-galaxy", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="Optional catalog names or cube stems to run.")
    ap.add_argument("--force", action="store_true", help="Delete each galaxy output before regenerating it.")
    ap.add_argument("--configs-only", action="store_true", help="Write configs and manifest but do not generate PV data.")
    args = ap.parse_args()

    args.root = args.root.resolve()
    args.table2 = (args.table2 or args.root / "J_AJ_141_23_table2.dat.txt").resolve()
    args.table7 = (args.table7 or args.root / "J_AJ_141_23_table7.dat.txt").resolve()
    out_root = (args.out_root or args.root / "training_data").resolve()
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    requested = {x.casefold() for x in args.only} if args.only else None
    manifest = []
    skipped = []
    for gal in parse_table2(args.table2):
        if requested and gal["name"].casefold() not in requested and gal["stem"].casefold() not in requested:
            continue
        cube = find_cube(args.root, gal["stem"])
        if cube is None:
            skipped.append({"name": gal["name"], "stem": gal["stem"], "reason": "missing_cube"})
            continue
        cfg = make_config(gal, cube, args, out_root)
        cfg_path = cfg_dir / f"{gal['stem'].lower()}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        manifest.append({"name": gal["name"], "stem": gal["stem"], "cube": str(cube), "config": str(cfg_path), "output_root": cfg["output_root"]})

    (out_root / "manifest.json").write_text(json.dumps({"galaxies": manifest, "skipped": skipped}, indent=2), encoding="utf-8")
    print(f"[batch] configs written: {len(manifest)}; skipped: {len(skipped)}; manifest={out_root / 'manifest.json'}")

    if args.configs_only:
        return

    summaries = []
    failed = []
    for item in manifest:
        name = item["name"]
        cfg_path = Path(item["config"])
        gal_out = Path(item["output_root"])
        if args.force and gal_out.exists():
            shutil.rmtree(gal_out)
        print(f"\n[batch] === {name} ===")
        try:
            run_module("src.pv.make_pv", cfg_path, args.root)
            run_module("src.pv.label_pv", cfg_path, args.root)
            summary_path = gal_out / "label_summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            summaries.append({"name": name, "stem": item["stem"], "output_root": item["output_root"], **summary})
        except subprocess.CalledProcessError as exc:
            failed.append({"name": name, "stem": item["stem"], "returncode": exc.returncode})
            print(f"[batch] ERROR: {name} failed with return code {exc.returncode}")

    batch_summary = {"generated": summaries, "failed": failed, "skipped": skipped}
    (out_root / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(f"\n[batch] complete: generated={len(summaries)} failed={len(failed)} skipped={len(skipped)}")
    print(f"[batch] summary={out_root / 'batch_summary.json'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
