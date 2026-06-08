from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PV_ROOT = ROOT / "pv_shells"
DATA_ROOT = PV_ROOT / "training_data" / "standardized_5kpc_200kms_clean_physical_baseline"
RUN_DIR = PV_ROOT / "runs" / "pv_unet_clean_physical_baseline_20260522_022751"
OUT = ROOT / "presentation_visuals"

COLORS = {
    "ink": "#17212b",
    "muted": "#64748b",
    "blue": "#2563eb",
    "teal": "#0f766e",
    "green": "#16a34a",
    "amber": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "gray": "#e5e7eb",
}

TYPE_NAMES = {
    1: "Type 1: both sides stalled",
    2: "Type 2: one side expanding",
    3: "Type 3: both sides expanding",
}

TYPE_COLORS = {
    1: "#f59e0b",
    2: "#2563eb",
    3: "#16a34a",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "figure.facecolor": "white",
        }
    )


def savefig(fig: plt.Figure, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def read_manifest(split: str) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / f"{split}_manifest.csv")


def load_metrics() -> dict:
    return json.loads((RUN_DIR / "component_filter_metrics.json").read_text())


def wrap_text(text: str, width: int = 24) -> str:
    return "\n".join(wrap(text, width))


def plot_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_axis_off()
    steps = [
        ("Raw THINGS cubes", "HI data cubes plus Bagetakos shell catalog"),
        ("Physical PV cuts", "5 kpc spatial window, 200 km/s velocity window, 96 x 256 tensor"),
        ("Beam-aware labels", "Erase isolated sub-beam speckles; preserve real boundary intersections"),
        ("High-recall U-Net", "BCE + Tversky loss, tuned to avoid missed shells"),
        ("Component filtering", "Connected components, beam area, edge and velocity sanity checks"),
        ("Review catalog", "Ranked shell candidates with physical measurements and shell type hints"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    y = 0.55
    for i, ((title, subtitle), x) in enumerate(zip(steps, xs)):
        color = [COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["purple"], COLORS["amber"], COLORS["red"]][i]
        ax.text(
            x,
            y + 0.11,
            str(i + 1),
            ha="center",
            va="center",
            color="white",
            fontsize=15,
            fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.45", fc=color, ec="none"),
            transform=ax.transAxes,
        )
        ax.text(
            x,
            y,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.08", fc="#f8fafc", ec="#cbd5e1"),
            transform=ax.transAxes,
        )
        ax.text(
            x,
            y - 0.16,
            wrap_text(subtitle, 22),
            ha="center",
            va="center",
            fontsize=9.5,
            color=COLORS["muted"],
            transform=ax.transAxes,
        )
        if i < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.055, y),
                xytext=(x + 0.055, y),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2),
            )
    ax.text(
        0.5,
        0.88,
        "Project Pipeline: From HI Cubes to Reviewable Shell Candidates",
        ha="center",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.12,
        "Main idea: the neural network is used as a high-recall detector; deterministic physics filters and human review clean the candidate list.",
        ha="center",
        fontsize=12,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )
    return savefig(fig, "01_pipeline_overview.png")


def plot_shell_type_schematic() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.2), sharex=True, sharey=True)
    x = np.linspace(-1, 1, 240)
    for typ, ax in zip([1, 2, 3], axes):
        ax.set_facecolor("#f8fafc")
        ax.axhline(0, color="#94a3b8", lw=2, alpha=0.9)
        ax.plot(x, 0.08 * np.sin(3 * x), color="#475569", lw=4, alpha=0.32)
        theta = np.linspace(0, 2 * np.pi, 300)
        if typ == 1:
            ax.plot(0.55 * np.cos(theta), 0.22 * np.sin(theta), color=TYPE_COLORS[typ], lw=3)
            ax.text(0, -0.52, "cavity/gap, no clear velocity caps", ha="center", color=COLORS["muted"], fontsize=10)
        elif typ == 2:
            ax.plot(0.55 * np.cos(theta), 0.22 * np.sin(theta), color="#94a3b8", lw=2, ls="--")
            ax.plot(0.55 * np.cos(theta[:150]), 0.22 * np.sin(theta[:150]), color=TYPE_COLORS[typ], lw=4)
            ax.annotate("", xy=(0.35, 0.25), xytext=(0.15, 0.05), arrowprops=dict(arrowstyle="->", color=TYPE_COLORS[typ], lw=3))
            ax.text(0, -0.52, "one side shows expansion", ha="center", color=COLORS["muted"], fontsize=10)
        else:
            ax.plot(0.55 * np.cos(theta), 0.22 * np.sin(theta), color=TYPE_COLORS[typ], lw=4)
            ax.annotate("", xy=(0.35, 0.25), xytext=(0.15, 0.05), arrowprops=dict(arrowstyle="->", color=TYPE_COLORS[typ], lw=3))
            ax.annotate("", xy=(-0.35, -0.25), xytext=(-0.15, -0.05), arrowprops=dict(arrowstyle="->", color=TYPE_COLORS[typ], lw=3))
            ax.text(0, -0.52, "both sides show expansion", ha="center", color=COLORS["muted"], fontsize=10)
        ax.set_title(TYPE_NAMES[typ], color=TYPE_COLORS[typ])
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlabel("position offset")
        if typ == 1:
            ax.set_ylabel("velocity offset")
        ax.grid(color="white", lw=1.5)
    fig.suptitle("Catalog Shell Types in PV Space", fontsize=20, fontweight="bold", color=COLORS["ink"], y=1.02)
    return savefig(fig, "02_shell_type_schematic.png")


def select_type_examples() -> dict[int, dict]:
    candidates: dict[int, list[tuple[float, dict]]] = defaultdict(list)
    labels_dir = DATA_ROOT / "labels"
    for sidecar in labels_dir.glob("*.json"):
        try:
            obj = json.loads(sidecar.read_text())
        except Exception:
            continue
        pv_file = obj.get("pv_file")
        if not pv_file or not (DATA_ROOT / "pv" / pv_file).exists():
            continue
        objects = obj.get("objects", [])
        types_here = {int(o.get("type", 0) or 0) for o in objects}
        for item in obj.get("objects", []):
            typ = int(item.get("type", 0) or 0)
            if typ not in (1, 2, 3):
                continue
            galaxy = str(pv_file).split("__", 1)[0]
            if galaxy == "ngc_3031":
                continue
            v0, p0, v1, p1 = [int(x) for x in item.get("bbox_vpos", [0, 0, 95, 255])]
            edge_penalty = 250 if (v0 <= 4 or v1 >= 91 or p0 <= 4 or p1 >= 251) else 0
            purity_bonus = 140 if types_here == {typ} else 0
            single_bonus = 80 if len(objects) == 1 else 0
            centered_bonus = 60 if "centered_positive" in str(obj.get("cut_category")) else 0
            score = float(item.get("mask_pixels", 0)) + purity_bonus + single_bonus + centered_bonus - edge_penalty
            candidates[typ].append((score, {"sidecar": sidecar, "object": item, "pv_file": pv_file}))
    return {typ: sorted(candidates[typ], key=lambda pair: pair[0], reverse=True)[0][1] for typ in (1, 2, 3)}


def robust_image(pv: np.ndarray) -> np.ndarray:
    finite = pv[np.isfinite(pv)]
    if finite.size == 0:
        return np.zeros_like(pv, dtype=float)
    lo, hi = np.nanpercentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return np.clip((pv - lo) / (hi - lo), 0, 1)


def plot_pv_examples() -> Path:
    examples = select_type_examples()
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.5), sharex=True, sharey=False)
    for typ, ax in zip([1, 2, 3], axes):
        ex = examples[typ]
        pv_path = DATA_ROOT / "pv" / ex["pv_file"]
        lab_path = DATA_ROOT / "labels" / ex["pv_file"]
        type_path = DATA_ROOT / "label_types" / ex["pv_file"]
        meta = json.loads((DATA_ROOT / "pv" / f"{pv_path.stem}.json").read_text())
        pv = np.load(pv_path)
        lab = np.load(lab_path)
        type_mask = np.load(type_path) if type_path.exists() else lab * typ
        v_axis = np.asarray(meta.get("velocity_axis_kms", np.arange(pv.shape[0])), dtype=float)
        extent = [0, float(meta.get("spatial_window_kpc", 5.0)), float(v_axis[-1]), float(v_axis[0])]
        ax.imshow(robust_image(pv), aspect="auto", cmap="gray_r", extent=extent)
        for t in [1, 2, 3]:
            mask = (type_mask == t).astype(float)
            if np.any(mask):
                ax.contour(
                    mask,
                    levels=[0.5],
                    colors=[TYPE_COLORS[t]],
                    linewidths=1.8,
                    extent=extent,
                )
        item = ex["object"]
        title = f"{TYPE_NAMES[typ]}\n{meta.get('galaxy')} shell {item.get('shell_id')}"
        ax.set_title(title, color=TYPE_COLORS[typ], fontsize=12)
        ax.set_xlabel("spatial distance along cut (kpc)")
        if typ == 1:
            ax.set_ylabel("velocity (km/s)")
        ax.text(
            0.02,
            0.04,
            f"{pv.shape[0]} x {pv.shape[1]} model input",
            transform=ax.transAxes,
            fontsize=9,
            color=COLORS["ink"],
            bbox=dict(fc="white", ec="none", alpha=0.75),
        )
    fig.suptitle("What the Model Sees: Standardized PV Cuts with Catalog Labels", fontsize=18, fontweight="bold", y=1.02)
    return savefig(fig, "03_standardized_pv_label_examples.png")


def plot_dataset_composition() -> Path:
    dfs = []
    for split in ["train", "val", "test", "stress"]:
        df = read_manifest(split)
        df["split"] = split
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.5))

    split_stats = all_df.groupby("split")["positive"].agg(["sum", "count"]).loc[["train", "val", "test", "stress"]]
    split_stats["negative"] = split_stats["count"] - split_stats["sum"]
    x = np.arange(len(split_stats))
    axes[0].bar(x, split_stats["sum"], color=COLORS["green"], label="label-containing")
    axes[0].bar(x, split_stats["negative"], bottom=split_stats["sum"], color="#cbd5e1", label="background / empty")
    axes[0].set_xticks(x, split_stats.index)
    axes[0].set_ylabel("PV cuts")
    axes[0].set_title("Generated Data Split")
    axes[0].legend(frameon=False)
    for i, row in enumerate(split_stats.itertuples()):
        axes[0].text(i, row.count + max(split_stats["count"]) * 0.02, f"{int(row.count):,}", ha="center", fontsize=9, color=COLORS["muted"])

    top_gal = all_df["galaxy"].value_counts().head(12).sort_values()
    axes[1].barh(top_gal.index, top_gal.values, color=COLORS["blue"])
    axes[1].set_title("Most Represented Galaxies")
    axes[1].set_xlabel("PV cuts")
    axes[1].tick_params(axis="y", labelsize=9)
    fig.suptitle("Dataset Composition", fontsize=18, fontweight="bold", y=1.02)
    return savefig(fig, "04_dataset_composition.png")


def plot_training_curves() -> Path:
    df = pd.read_csv(RUN_DIR / "history.csv")
    df["epoch_display"] = df["epoch"] + 1
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.4))
    axes[0].plot(df["epoch_display"], df["loss"], label="train loss", lw=2.5, color=COLORS["blue"])
    axes[0].plot(df["epoch_display"], df["val_loss"], label="validation loss", lw=2.5, color=COLORS["red"])
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Optimization")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["epoch_display"], df["patch_recall_0p075"], label="train recall", lw=2.5, color=COLORS["green"])
    axes[1].plot(df["epoch_display"], df["val_patch_recall_0p075"], label="validation recall", lw=2.5, color=COLORS["teal"])
    axes[1].plot(df["epoch_display"], df["patch_precision_0p075"], label="train precision", lw=2, color=COLORS["amber"], ls="--")
    axes[1].plot(df["epoch_display"], df["val_patch_precision_0p075"], label="validation precision", lw=2, color=COLORS["purple"], ls="--")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0.65, 1.02)
    axes[1].set_title("Patch-Level Metrics at Threshold 0.075")
    axes[1].legend(frameon=False, ncol=2, fontsize=9)
    axes[1].grid(alpha=0.25)
    fig.suptitle("Clean Physical Baseline Training Behavior", fontsize=18, fontweight="bold", y=1.02)
    return savefig(fig, "05_training_curves.png")


def plot_postprocessing_metrics() -> Path:
    metrics = load_metrics()
    stages = [
        ("before_filtering", "Raw\nU-Net"),
        ("after_beam_area_filter", "Beam\narea"),
        ("after_beam_velocity_extent_filter", "Beam +\nvelocity"),
        ("after_beam_velocity_edge_filter", "Beam +\nedge"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.5), sharey=True)
    for ax, split in zip(axes, ["val", "test", "stress"]):
        vals_p, vals_r = [], []
        for key, _ in stages:
            row = metrics["splits"][split]["thresholds"]["0.075"]["metrics"][key]
            vals_p.append(row["precision"])
            vals_r.append(row["recall"])
        x = np.arange(len(stages))
        width = 0.38
        ax.bar(x - width / 2, vals_p, width, color=COLORS["blue"], label="precision")
        ax.bar(x + width / 2, vals_r, width, color=COLORS["green"], label="recall")
        ax.set_xticks(x, [label for _, label in stages])
        ax.set_ylim(0.68, 1.02)
        ax.set_title(split.upper())
        ax.grid(axis="y", alpha=0.2)
        for i, v in enumerate(vals_p):
            ax.text(i - width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=COLORS["muted"])
        for i, v in enumerate(vals_r):
            ax.text(i + width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=COLORS["muted"])
    axes[0].set_ylabel("patch-level score")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Post-Processing Improves Precision While Protecting Recall", fontsize=18, fontweight="bold", y=1.02)
    return savefig(fig, "06_postprocessing_metrics.png")


def plot_ranking_curves() -> Path:
    metrics = load_metrics()
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    for split, color in [("val", COLORS["blue"]), ("test", COLORS["green"]), ("stress", COLORS["red"])]:
        top = metrics["splits"][split]["thresholds"]["0.075"]["metrics"]["after_beam_probability_mass_top_n"]
        ns = sorted(int(k) for k in top.keys())
        recall = [top[str(n)]["patch_recall"] for n in ns]
        precision = [top[str(n)]["patch_precision"] for n in ns]
        ax.plot(recall, precision, marker="o", lw=2.5, color=color, label=split)
        for n, r, p in zip(ns, recall, precision):
            if n in {100, 1000, 5000}:
                ax.text(r, p + 0.012, f"top {n}", fontsize=8, color=color, ha="center")
    ax.set_xlabel("recall among top-N ranked candidates")
    ax.set_ylabel("precision among top-N ranked candidates")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.75, 1.01)
    ax.set_title("Probability-Mass Ranking: Review Load vs Completeness")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return savefig(fig, "07_probability_ranking_curve.png")


def plot_ngc3031_stress() -> Path:
    metrics = load_metrics()
    splits = ["val", "test", "stress"]
    rows = []
    for split in splits:
        d = metrics["splits"][split]["thresholds"]["0.075"]
        raw = d["candidate_count_before_filtering"]
        final = d["candidate_count_final_candidate_pass"]
        edge = d["edge_touching_flag_count"]
        vedge = d["velocity_edge_touching_flag_count"]
        m = d["metrics"]["after_beam_area_filter"]
        rows.append((split, raw, final, edge, vedge, m["precision"], m["recall"]))
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.4))
    x = np.arange(len(rows))
    axes[0].bar(x - 0.2, [r[3] for r in rows], width=0.4, color=COLORS["amber"], label="any edge-touching")
    axes[0].bar(x + 0.2, [r[4] for r in rows], width=0.4, color=COLORS["red"], label="velocity-edge touching")
    axes[0].set_xticks(x, [r[0] for r in rows])
    axes[0].set_ylabel("component count")
    axes[0].set_title("NGC 3031 Stress Set Creates Many Edge-Like Components")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - 0.2, [r[5] for r in rows], width=0.4, color=COLORS["blue"], label="precision")
    axes[1].bar(x + 0.2, [r[6] for r in rows], width=0.4, color=COLORS["green"], label="recall")
    axes[1].set_xticks(x, [r[0] for r in rows])
    axes[1].set_ylim(0.7, 1.02)
    axes[1].set_title("Stress-Test Performance After Beam Filter")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Why NGC 3031 Is a Stress Test, Not Normal Training Data", fontsize=18, fontweight="bold", y=1.02)
    return savefig(fig, "08_ngc3031_stress_test.png")


def write_notes(paths: list[Path]) -> Path:
    metrics = load_metrics()
    test = metrics["splits"]["test"]["thresholds"]["0.075"]["metrics"]
    stress = metrics["splits"]["stress"]["thresholds"]["0.075"]["metrics"]
    notes = f"""# Presentation Visuals: HIShells PV Project

Generated figures live in this folder. Each PNG is slide-ready.

## 01_pipeline_overview.png

Shows the end-to-end idea: raw HI cubes are converted into physically standardized PV cuts, labeled with beam-aware masks, passed through a high-recall U-Net, then cleaned with deterministic connected-component filters.

Main talking point: the neural network is not the final catalog. It is the sensitive detection layer; physics-aware filtering and human review turn detections into credible candidates.

## 02_shell_type_schematic.png

Explains the three catalog shell types in simple PV-space language:

- Type 1: both sides stalled, or no clear velocity caps.
- Type 2: one side expanding.
- Type 3: both sides expanding.

Main talking point: type classification is a natural last-stage helper for reviewers because it tells them what PV signature to expect.

## 03_standardized_pv_label_examples.png

Shows actual standardized model inputs: each PV image is resampled to a fixed physical shape, with catalog label contours overlaid.

Main talking point: every input has the same physical meaning, not just the same pixel dimensions: 5 kpc across and 200 km/s in velocity, mapped into a 96 x 256 tensor.

## 04_dataset_composition.png

Summarizes how many PV cuts are in each split and which galaxies contribute the most cuts.

Main talking point: the model is trained on many orientations and offsets around catalog shells, plus background/random cuts, while NGC 3031 is isolated as a stress split.

## 05_training_curves.png

Shows the clean physical baseline training run.

Main talking point: the model is intentionally optimized for recall. That means it is allowed to be over-inclusive at the pixel level because later post-processing removes obvious false positives.

## 06_postprocessing_metrics.png

Shows before/after patch precision and recall at threshold 0.075.

Key numbers:

- Test after beam-area filtering: precision = {test['after_beam_area_filter']['precision']:.3f}, recall = {test['after_beam_area_filter']['recall']:.3f}.
- Stress/NGC 3031 after beam-area filtering: precision = {stress['after_beam_area_filter']['precision']:.3f}, recall = {stress['after_beam_area_filter']['recall']:.3f}.

Main talking point: beam-aware filtering improves precision while preserving near-perfect recall on validation/test.

## 07_probability_ranking_curve.png

Shows what happens when candidates are sorted by integrated probability mass and reviewers inspect only the top N.

Main talking point: ranking gives a tunable review workload. You can choose a smaller high-confidence list or a larger high-recall list.

## 08_ngc3031_stress_test.png

Shows why NGC 3031/M81 behaves differently: it produces far more edge-touching components than normal validation/test splits.

Main talking point: NGC 3031 is not just a file-format problem. It contains line-of-sight confusion and tidal structures, so it is best treated as a stress test for post-processing.

## Suggested slide order

1. Problem: detecting HI shells in PV space.
2. `01_pipeline_overview.png`
3. `02_shell_type_schematic.png`
4. `03_standardized_pv_label_examples.png`
5. `04_dataset_composition.png`
6. `05_training_curves.png`
7. `06_postprocessing_metrics.png`
8. `07_probability_ranking_curve.png`
9. `08_ngc3031_stress_test.png`
10. Conclusion: high recall first, physics-aware cleanup second, human review last.

## Generated Files

{chr(10).join(f'- `{p.name}`' for p in paths)}
"""
    path = OUT / "VISUALIZATION_NOTES.md"
    path.write_text(notes)
    return path


def main() -> None:
    setup()
    paths = [
        plot_pipeline(),
        plot_shell_type_schematic(),
        plot_pv_examples(),
        plot_dataset_composition(),
        plot_training_curves(),
        plot_postprocessing_metrics(),
        plot_ranking_curves(),
        plot_ngc3031_stress(),
    ]
    notes = write_notes(paths)
    print("Generated:")
    for path in paths:
        print(path)
    print(notes)


if __name__ == "__main__":
    main()
