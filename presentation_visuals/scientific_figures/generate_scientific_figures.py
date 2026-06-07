from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PV_ROOT = ROOT / "pv_shells"
DATA_ROOT = PV_ROOT / "training_data" / "standardized_5kpc_200kms_clean_physical_baseline"
RUN_DIR = PV_ROOT / "runs" / "pv_unet_clean_physical_baseline_20260522_022751"
OUT = Path(__file__).resolve().parent

TYPE_NAMES = {
    1: "type 1",
    2: "type 2",
    3: "type 3",
}
TYPE_COLORS = {
    1: "tab:orange",
    2: "tab:blue",
    3: "tab:green",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def panel_label(ax, label: str) -> None:
    ax.text(
        0.03,
        0.95,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )


def robust_image(pv: np.ndarray) -> np.ndarray:
    finite = pv[np.isfinite(pv)]
    if finite.size == 0:
        return np.zeros_like(pv, dtype=float)
    lo, hi = np.nanpercentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((pv - lo) / (hi - lo), 0, 1)


def load_metrics() -> dict:
    return json.loads((RUN_DIR / "component_filter_metrics.json").read_text())


def read_manifest(split: str) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / f"{split}_manifest.csv")


def select_type_examples() -> dict[int, dict]:
    candidates: dict[int, list[tuple[float, dict]]] = defaultdict(list)
    for sidecar in (DATA_ROOT / "labels").glob("*.json"):
        try:
            meta_label = json.loads(sidecar.read_text())
        except Exception:
            continue
        pv_file = meta_label.get("pv_file")
        if not pv_file or not (DATA_ROOT / "pv" / pv_file).exists():
            continue
        objects = meta_label.get("objects", [])
        types_here = {int(o.get("type", 0) or 0) for o in objects}
        galaxy = str(pv_file).split("__", 1)[0]
        if galaxy == "ngc_3031":
            continue
        for obj in objects:
            typ = int(obj.get("type", 0) or 0)
            if typ not in (1, 2, 3):
                continue
            v0, p0, v1, p1 = [int(x) for x in obj.get("bbox_vpos", [0, 0, 95, 255])]
            edge_penalty = 250 if (v0 <= 4 or v1 >= 91 or p0 <= 4 or p1 >= 251) else 0
            purity_bonus = 140 if types_here == {typ} else 0
            single_bonus = 80 if len(objects) == 1 else 0
            centered_bonus = 60 if "centered_positive" in str(meta_label.get("cut_category")) else 0
            score = float(obj.get("mask_pixels", 0)) + purity_bonus + single_bonus + centered_bonus - edge_penalty
            candidates[typ].append((score, {"sidecar": sidecar, "object": obj, "pv_file": pv_file}))
    return {typ: sorted(candidates[typ], key=lambda pair: pair[0], reverse=True)[0][1] for typ in (1, 2, 3)}


def figure_pv_examples() -> tuple[Path, Path]:
    examples = select_type_examples()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), constrained_layout=True)
    for typ, ax, letter in zip([1, 2, 3], axes, ["a", "b", "c"]):
        ex = examples[typ]
        pv_path = DATA_ROOT / "pv" / ex["pv_file"]
        lab_path = DATA_ROOT / "labels" / ex["pv_file"]
        type_path = DATA_ROOT / "label_types" / ex["pv_file"]
        meta = json.loads((DATA_ROOT / "pv" / f"{pv_path.stem}.json").read_text())
        pv = np.load(pv_path)
        label = np.load(lab_path)
        type_mask = np.load(type_path) if type_path.exists() else label * typ
        v_axis = np.asarray(meta.get("velocity_axis_kms", np.arange(pv.shape[0])), dtype=float)
        extent = [0, float(meta.get("spatial_window_kpc", 5.0)), float(v_axis[-1]), float(v_axis[0])]
        ax.imshow(robust_image(pv), aspect="auto", cmap="gray_r", extent=extent, interpolation="nearest")
        for t in [1, 2, 3]:
            mask = type_mask == t
            if np.any(mask):
                ax.contour(mask.astype(float), levels=[0.5], colors=[TYPE_COLORS[t]], linewidths=1.0, extent=extent)
        panel_label(ax, letter)
        item = ex["object"]
        ax.set_title(f"{TYPE_NAMES[typ]}, {meta.get('galaxy')} #{item.get('shell_id')}")
        ax.set_xlabel("position (kpc)")
        if typ == 1:
            ax.set_ylabel("velocity (km s$^{-1}$)")
    return save(fig, "fig01_pv_examples")


def figure_dataset_and_types() -> tuple[Path, Path]:
    manifests = []
    for split in ["train", "val", "test", "stress"]:
        df = read_manifest(split)
        df["split"] = split
        manifests.append(df)
    df_all = pd.concat(manifests, ignore_index=True)

    type_counts = Counter()
    for sidecar in (DATA_ROOT / "labels").glob("*.json"):
        try:
            obj = json.loads(sidecar.read_text())
        except Exception:
            continue
        for item in obj.get("objects", []):
            typ = int(item.get("type", 0) or 0)
            if typ in (1, 2, 3):
                type_counts[typ] += 1

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    split_order = ["train", "val", "test", "stress"]
    split_stats = df_all.groupby("split")["positive"].agg(["sum", "count"]).reindex(split_order)
    split_stats["negative"] = split_stats["count"] - split_stats["sum"]
    x = np.arange(len(split_order))
    axes[0].bar(x, split_stats["sum"], color="0.25", label="positive")
    axes[0].bar(x, split_stats["negative"], bottom=split_stats["sum"], color="0.75", label="negative")
    axes[0].set_xticks(x, split_order, rotation=20)
    axes[0].set_ylabel("PV cuts")
    axes[0].legend(frameon=False)
    panel_label(axes[0], "a")

    tx = np.arange(3)
    axes[1].bar(tx, [type_counts[t] for t in [1, 2, 3]], color=[TYPE_COLORS[t] for t in [1, 2, 3]])
    axes[1].set_xticks(tx, ["type 1", "type 2", "type 3"], rotation=20)
    axes[1].set_ylabel("labeled shell objects")
    panel_label(axes[1], "b")
    return save(fig, "fig02_dataset_type_distribution")


def figure_training_history() -> tuple[Path, Path]:
    hist = pd.read_csv(RUN_DIR / "history.csv")
    epoch = hist["epoch"] + 1
    epoch_ticks = np.arange(1, int(epoch.max()) + 1, 2)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    axes[0].plot(epoch, hist["loss"], color="0.15", lw=1.5, label="train")
    axes[0].plot(epoch, hist["val_loss"], color="tab:red", lw=1.5, label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_xlim(1, int(epoch.max()))
    axes[0].set_xticks(epoch_ticks)
    axes[0].legend(frameon=False)
    panel_label(axes[0], "a")

    axes[1].plot(epoch, hist["val_patch_recall_0p075"], color="tab:green", lw=1.5, label="recall")
    axes[1].plot(epoch, hist["val_patch_precision_0p075"], color="tab:blue", lw=1.5, label="precision")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation patch metric")
    axes[1].set_ylim(0.75, 1.02)
    axes[1].set_xlim(1, int(epoch.max()))
    axes[1].set_xticks(epoch_ticks)
    axes[1].legend(frameon=False)
    panel_label(axes[1], "b")
    return save(fig, "fig03_training_history")


def figure_postprocessing_metrics() -> tuple[Path, Path]:
    metrics = load_metrics()
    stage_keys = [
        "before_filtering",
        "after_beam_area_filter",
        "after_beam_velocity_extent_filter",
        "after_beam_velocity_edge_filter",
    ]
    stage_labels = ["raw", "beam", "beam+vel", "beam+edge"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True, sharex=True)
    x = np.arange(len(stage_keys))
    for split, marker in [("val", "o"), ("test", "s"), ("stress", "^")]:
        vals = [metrics["splits"][split]["thresholds"]["0.075"]["metrics"][k]["precision"] for k in stage_keys]
        axes[0].plot(x, vals, marker=marker, lw=1.4, label=split)
        vals = [metrics["splits"][split]["thresholds"]["0.075"]["metrics"][k]["recall"] for k in stage_keys]
        axes[1].plot(x, vals, marker=marker, lw=1.4, label=split)
    for ax, ylabel, letter in zip(axes, ["precision", "recall",], ["a", "b"]):
        ax.set_xticks(x, stage_labels, rotation=25)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.70, 1.02)
        ax.legend(frameon=False)
        panel_label(ax, letter)
    return save(fig, "fig04_postprocessing_precision_recall")


def figure_candidate_features() -> tuple[Path, Path]:
    df = pd.read_csv(RUN_DIR / "component_candidates_test_threshold_0p075.csv")
    df = df[df["final_candidate_pass"] == 1].copy()
    df["truth"] = np.where(df["component_overlaps_label"] == 1, "overlaps label", "candidate only")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), constrained_layout=True)
    specs = [
        ("area_over_beam", "area / beam area", np.logspace(-1, 2.2, 32), True),
        ("velocity_extent_kms", "velocity extent (km s$^{-1}$)", np.linspace(0, 120, 31), False),
        ("spatial_extent_kpc", "spatial extent (kpc)", np.linspace(0, 5, 31), False),
        ("integrated_probability_mass", "integrated probability mass", np.logspace(0, 3.6, 35), True),
    ]
    for ax, (col, label, bins, logx), letter in zip(axes.flat, specs, ["a", "b", "c", "d"]):
        for truth, color in [("overlaps label", "0.2"), ("candidate only", "tab:red")]:
            values = df.loc[df["truth"] == truth, col].astype(float)
            values = values[np.isfinite(values)]
            if logx:
                values = values[values > 0]
            ax.hist(values, bins=bins, histtype="step", lw=1.3, density=True, color=color, label=truth)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(frameon=False)
        panel_label(ax, letter)
    return save(fig, "fig05_candidate_feature_distributions")


def figure_probability_ranking() -> tuple[Path, Path]:
    metrics = load_metrics()
    fig, ax = plt.subplots(figsize=(3.8, 3.2), constrained_layout=True)
    for split, marker in [("val", "o"), ("test", "s"), ("stress", "^")]:
        top = metrics["splits"][split]["thresholds"]["0.075"]["metrics"]["after_beam_probability_mass_top_n"]
        ns = sorted(int(k) for k in top.keys())
        recall = [top[str(n)]["patch_recall"] for n in ns]
        precision = [top[str(n)]["patch_precision"] for n in ns]
        ax.plot(recall, precision, marker=marker, lw=1.4, ms=4, label=split)
    ax.set_xlabel("patch recall among top-$N$")
    ax.set_ylabel("patch precision among top-$N$")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.75, 1.02)
    ax.legend(frameon=False)
    panel_label(ax, "a")
    return save(fig, "fig06_probability_mass_ranking")


def figure_beam_and_stress() -> tuple[Path, Path]:
    metrics = load_metrics()
    df = pd.concat(
        [
            pd.read_csv(RUN_DIR / f"component_candidates_{split}_threshold_0p075.csv").assign(split=split)
            for split in ["val", "test", "stress"]
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    beam = df[["galaxy", "beam_area_pixels"]].drop_duplicates()
    by_gal = beam.groupby("galaxy")["beam_area_pixels"].median().sort_values()
    axes[0].barh(np.arange(len(by_gal)), by_gal.values, color="0.35")
    axes[0].set_yticks(np.arange(len(by_gal)), by_gal.index)
    axes[0].set_xlabel("beam area (model pixels)")
    panel_label(axes[0], "a")

    splits = ["val", "test", "stress"]
    edge = [metrics["splits"][s]["thresholds"]["0.075"]["edge_touching_flag_count"] for s in splits]
    vedge = [metrics["splits"][s]["thresholds"]["0.075"]["velocity_edge_touching_flag_count"] for s in splits]
    x = np.arange(len(splits))
    axes[1].bar(x - 0.18, edge, width=0.36, color="0.55", label="any edge")
    axes[1].bar(x + 0.18, vedge, width=0.36, color="tab:red", label="velocity edge")
    axes[1].set_xticks(x, splits)
    axes[1].set_ylabel("component count")
    axes[1].legend(frameon=False)
    panel_label(axes[1], "b")
    return save(fig, "fig07_beam_and_stress_diagnostics")


def write_captions(files: list[tuple[Path, Path]]) -> Path:
    metrics = load_metrics()
    test = metrics["splits"]["test"]["thresholds"]["0.075"]["metrics"]["after_beam_area_filter"]
    stress = metrics["splits"]["stress"]["thresholds"]["0.075"]["metrics"]["after_beam_area_filter"]
    text = f"""# Scientific Figure Captions

These figures are styled like compact paper figures. Each is exported as PNG and PDF.

**Figure 1. PV examples and catalog masks.** Three standardized position-velocity cuts used as U-Net inputs. Grayscale shows HI intensity after robust display scaling; colored contours show catalog-derived masks for shell types 1, 2, and 3.

**Figure 2. Dataset and label composition.** Panel (a) shows positive and negative PV cuts in each split. Panel (b) shows labeled shell-object counts by catalog type.

**Figure 3. Training history.** Panel (a) shows train and validation loss for the clean physical baseline. Panel (b) shows validation patch precision and recall at the working threshold of 0.075.

**Figure 4. Post-processing metrics.** Patch-level precision and recall before filtering and after deterministic connected-component filters. On the test split, beam-area filtering gives precision = {test['precision']:.3f} and recall = {test['recall']:.3f}. On the NGC 3031 stress split, beam-area filtering gives precision = {stress['precision']:.3f} and recall = {stress['recall']:.3f}.

**Figure 5. Candidate component feature distributions.** Distributions of connected-component morphology and probability features for test-set candidates that survive the final filter. Black curves overlap a catalog label; red curves are candidate-only components.

**Figure 6. Probability-mass ranking.** Precision-recall tradeoff when candidates are ranked by integrated probability mass and only the top N are retained.

**Figure 7. Beam and stress diagnostics.** Panel (a) shows synthesized beam areas in standardized model pixels. Panel (b) compares edge-touching components across validation, test, and the NGC 3031 stress split.

## Files

{chr(10).join(f'- `{png.name}` / `{pdf.name}`' for png, pdf in files)}
"""
    path = OUT / "SCIENTIFIC_FIGURE_CAPTIONS.md"
    path.write_text(text)
    return path


def main() -> None:
    setup()
    files = [
        figure_pv_examples(),
        figure_dataset_and_types(),
        figure_training_history(),
        figure_postprocessing_metrics(),
        figure_candidate_features(),
        figure_probability_ranking(),
        figure_beam_and_stress(),
    ]
    captions = write_captions(files)
    print("Generated scientific figures:")
    for png, pdf in files:
        print(png)
        print(pdf)
    print(captions)


if __name__ == "__main__":
    main()
