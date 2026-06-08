#!/usr/bin/env python3
"""Render scientifically-styled figures from a trained PV-shell run.

Reads only the lightweight metric artifacts produced by training and
sky-plane aggregation - no torch, no checkpoints - so it runs in any
environment with numpy/pandas/matplotlib:

- ``<run>/history_torch.csv``                  -> training/validation curves
- ``<run>/aggregate_*_*/detections_*.json``    -> per-galaxy candidate catalogs

All figures are written as 300 dpi PNGs to ``--out`` (default ``plots/runs``).

Usage::

    python scripts/plot_run_metrics.py
    python scripts/plot_run_metrics.py --run runs/pv_unet_real --out plots/runs
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Colorblind-safe palette (Dark2), matching analysis.ipynb's TYPE_COLORS.
PALETTE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]
SAVE_KW = dict(dpi=300, bbox_inches="tight", facecolor="white")


def _apply_style() -> None:
    """Set a consistent, publication-friendly matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def _pretty_galaxy(split: str) -> str:
    """``ddo53_test`` -> ``DDO 53``; ``ngc_3184_test`` -> ``NGC 3184``."""
    name = re.sub(r"_(test|val|train|stress)$", "", split)
    name = name.replace("_", " ").upper()
    name = re.sub(r"\b(NGC|DDO|HO|UGC|IC)\s*", r"\1 ", name).strip()
    return re.sub(r"\s+", " ", name)


# --------------------------------------------------------------------------- #
# Training-history figures
# --------------------------------------------------------------------------- #
def plot_training_history(history_csv: Path, out_dir: Path) -> None:
    if not history_csv.exists():
        print(f"[plot] skip training curves: {history_csv} not found")
        return
    df = pd.read_csv(history_csv)
    epochs = df["epoch"]

    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if "loss" in df:
        ax.plot(epochs, df["loss"], "o-", color=PALETTE[0], lw=2, label="train loss")
    if "val_loss" in df:
        ax.plot(epochs, df["val_loss"], "s--", color=PALETTE[1], lw=2, label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE-Tversky loss")
    ax.set_title("Training and validation loss")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "training_loss.png")

    # 2. Patch-level precision / recall / F1 at 0.075
    patch_cols = {
        "val_patch_precision_0p075": ("precision", PALETTE[0], "o-"),
        "val_patch_recall_0p075": ("recall", PALETTE[1], "s--"),
        "val_patch_f1_0p075": ("F1", PALETTE[2], "^-"),
    }
    if any(c in df for c in patch_cols):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for col, (label, color, style) in patch_cols.items():
            if col in df:
                ax.plot(epochs, df[col], style, color=color, lw=2, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("score at threshold = 0.075")
        ax.set_ylim(0, 1.02)
        ax.set_title("Validation patch-level detection metrics")
        ax.legend(loc="best")
        fig.tight_layout()
        _save(fig, out_dir, "patch_metrics.png")

    # 3. Pixel-level precision / recall at 0.075
    pixel_cols = {
        "val_pixel_precision_0p075": ("precision", PALETTE[0], "o-"),
        "val_pixel_recall_0p075": ("recall", PALETTE[1], "s--"),
    }
    if any(c in df for c in pixel_cols):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for col, (label, color, style) in pixel_cols.items():
            if col in df:
                ax.plot(epochs, df[col], style, color=color, lw=2, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("score at threshold = 0.075")
        ax.set_ylim(0, 1.02)
        ax.set_title("Validation pixel-level segmentation metrics")
        ax.legend(loc="best")
        fig.tight_layout()
        _save(fig, out_dir, "pixel_metrics.png")


# --------------------------------------------------------------------------- #
# Detection-catalog figures
# --------------------------------------------------------------------------- #
def _load_detections(run_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict] = []
    thresholds: dict[str, float] = {}
    for jpath in sorted(run_dir.glob("aggregate_*/detections_*.json")):
        data = json.loads(jpath.read_text())
        split = data.get("split", jpath.stem.replace("detections_", ""))
        galaxy = _pretty_galaxy(split)
        thr = data.get("params", {}).get("thresh")
        if thr is not None:
            thresholds[galaxy] = float(thr)
        for det in data.get("detections", []):
            rows.append(
                {
                    "galaxy": galaxy,
                    "score": float(det.get("score", np.nan)),
                    "r_pix": float(det.get("r_pix", np.nan)),
                    "x_pix": det.get("x_pix"),
                    "y_pix": det.get("y_pix"),
                }
            )
    return pd.DataFrame(rows), thresholds


def plot_detections(run_dir: Path, out_dir: Path) -> None:
    df, thresholds = _load_detections(run_dir)
    if df.empty:
        print("[plot] skip detection figures: no detections_*.json found")
        return

    galaxies = sorted(df["galaxy"].unique())
    color_for = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(galaxies)}

    # 4. Candidate score distributions per galaxy (box + jittered points)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rng = np.random.default_rng(0)
    for i, g in enumerate(galaxies, start=1):
        scores = df.loc[df["galaxy"] == g, "score"].dropna().values
        if scores.size == 0:
            continue
        ax.boxplot(
            scores,
            positions=[i],
            widths=0.55,
            patch_artist=True,
            boxprops=dict(facecolor=color_for[g], alpha=0.35, color=color_for[g]),
            medianprops=dict(color="black"),
            whiskerprops=dict(color=color_for[g]),
            capprops=dict(color=color_for[g]),
            flierprops=dict(marker="", alpha=0),
        )
        jitter = rng.uniform(-0.18, 0.18, size=scores.size)
        ax.scatter(
            np.full(scores.size, i) + jitter,
            scores,
            color=color_for[g],
            edgecolor="black",
            linewidth=0.3,
            s=24,
            zorder=3,
        )
    thr_vals = sorted(set(thresholds.values()))
    for thr in thr_vals:
        ax.axhline(thr, color="grey", ls=":", lw=1.2)
    if thr_vals:
        ax.axhline(
            thr_vals[0], color="grey", ls=":", lw=1.2,
            label=f"aggregation threshold = {thr_vals[0]:g}",
        )
        ax.legend(loc="best")
    ax.set_xticks(range(1, len(galaxies) + 1))
    ax.set_xticklabels(galaxies, rotation=30, ha="right")
    ax.set_ylabel("candidate vote score")
    ax.set_title("Per-galaxy distribution of candidate scores (test split)")
    fig.tight_layout()
    _save(fig, out_dir, "detection_scores.png")

    # 5. Candidate radius distribution per galaxy
    fig, ax = plt.subplots(figsize=(9, 4.5))
    r_all = df["r_pix"].dropna()
    bins = np.linspace(r_all.min(), r_all.max(), 16) if not r_all.empty else 10
    for g in galaxies:
        radii = df.loc[df["galaxy"] == g, "r_pix"].dropna().values
        if radii.size == 0:
            continue
        ax.hist(
            radii,
            bins=bins,
            histtype="step",
            lw=1.8,
            color=color_for[g],
            label=f"{g}  (n = {radii.size})",
        )
    ax.set_xlabel("candidate radius (pixels)")
    ax.set_ylabel("count")
    ax.set_title("Per-galaxy distribution of candidate radii (test split)")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "detection_radii.png")

    # 6. Score vs radius scatter + per-galaxy detection counts
    fig, (ax_sc, ax_ct) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]}
    )
    for g in galaxies:
        sub = df[df["galaxy"] == g]
        ax_sc.scatter(
            sub["r_pix"],
            sub["score"],
            color=color_for[g],
            edgecolor="black",
            linewidth=0.3,
            s=40,
            alpha=0.85,
            label=g,
        )
    ax_sc.set_xlabel("candidate radius (pixels)")
    ax_sc.set_ylabel("candidate vote score")
    ax_sc.set_title("Candidate score vs radius")
    ax_sc.legend(loc="best")

    counts = df.groupby("galaxy").size().reindex(galaxies, fill_value=0)
    ax_ct.barh(
        range(len(galaxies)),
        counts.values,
        color=[color_for[g] for g in galaxies],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_ct.set_yticks(range(len(galaxies)))
    ax_ct.set_yticklabels(galaxies)
    ax_ct.invert_yaxis()
    ax_ct.set_xlabel("number of candidates")
    ax_ct.set_title("Candidates per galaxy")
    fig.tight_layout()
    _save(fig, out_dir, "score_vs_radius.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        default=str(ROOT / "runs" / "pv_unet_real"),
        help="Run directory containing history_torch.csv and aggregate_*/ outputs",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "plots" / "runs"),
        help="Output directory for the PNG figures",
    )
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()
    plot_training_history(run_dir / "history_torch.csv", out_dir)
    plot_detections(run_dir, out_dir)
    print(f"[plot] done -> {out_dir}")


if __name__ == "__main__":
    main()
