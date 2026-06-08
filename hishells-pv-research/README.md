# HI Shell Detection With Standardized PV Slices

This repository contains the paper-facing code and model artifact for detecting HI shells in position-velocity (PV) slices from THINGS galaxy cubes.

The project takes HI data cubes, cuts them into standardized PV images, labels shell-like structure from the Bagetakos et al. catalog, and trains a 2D U-Net/CNN segmentation model. The model predicts a probability mask over each PV slice. Downstream scripts can turn those masks into candidate shell components for review.

This is not a dump of every experiment. Historical runs, review panels, generated arrays, TensorBoard logs, and scratch figures have been left out. What remains is the finished pipeline, the published checkpoint, its condensed metric record, and the shell-type analysis notebook.

## What Is In This Repo

```text
hishells-pv-research/
  artifacts/
    model_metadata.json
    models/
      hi_shell_pv_unet_clean_physical_baseline.keras
  notebooks/
    shell_type_selection_effects.ipynb
  scripts/
  src/
  training_data/
    configs/
    manifest.json
  J_AJ_141_23_table2.dat.txt
  J_AJ_141_23_table7.dat.txt
```

The main pieces are:

- `src/`: Python modules for catalog loading, WCS handling, PV generation, mask labeling, dataset loading, model training, inference, evaluation, and post-processing.
- `scripts/`: Command-line entry points for the full training workflow and smaller one-off tasks.
- `training_data/configs/`: Per-galaxy YAML configs. These point to local FITS files under `data/raw/`.
- `artifacts/models/hi_shell_pv_unet_clean_physical_baseline.keras`: the published Keras checkpoint.
- `artifacts/model_metadata.json`: checksum, model settings, recommended threshold, and headline metrics.
- `notebooks/shell_type_selection_effects.ipynb`: the preserved shell-type analysis notebook, with outputs cleared.
- `J_AJ_141_23_table2.dat.txt` and `J_AJ_141_23_table7.dat.txt`: the small catalog tables used by the labeling code and shell-type notebook.

## What Is Not Included

The repo intentionally does not store:

- THINGS FITS cubes
- generated PV arrays and label masks
- historical `runs/` directories
- raw eval JSON files
- review panels and probability-map dumps
- TensorBoard logs
- old presentation figures
- extra checkpoints from abandoned or smoke-test runs

That keeps the repository small enough to read. The model metrics cited by this repo are condensed into `artifacts/model_metadata.json`.

## Published Model

The checkpoint used for the paper-facing result is:

```text
artifacts/models/hi_shell_pv_unet_clean_physical_baseline.keras
```

Metadata:

```text
artifacts/model_metadata.json
```

Current metadata records:

- architecture: 2D U-Net / CNN mask segmenter
- input shape: `(96, 256, 1)`
- spatial window: `5.0 kpc`
- velocity window: `200 km/s`
- recommended probability threshold: `0.075`
- checkpoint SHA-256: `7e6b3a00378e5d76c72a1954b95cab508e5f799f5def0b589146a751ce9ff6e0`
- source run: `pv_unet_clean_physical_baseline_20260522_022751`

Headline metrics at threshold `0.075`:

| Split | Patch precision | Patch recall | Patch F1 | Pixel precision | Pixel recall | Pixel F1 | Pixel PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.778 | 0.998 | 0.874 | 0.200 | 0.577 | 0.296 | 0.252 |
| test | 0.815 | 1.000 | 0.898 | 0.268 | 0.690 | 0.386 | 0.352 |
| stress | 0.749 | 0.977 | 0.848 | 0.032 | 0.282 | 0.057 | 0.018 |

The stress split holds out NGC 3031. It is deliberately hard because M81-group tidal structure creates line-of-sight features that can look shell-like in PV space.

The DDO 53 physical-grid check in the metadata records full shell-level coverage and detection recall for catalog shells `[1, 2, 3]` at the same threshold.

## Data Layout

Put raw THINGS cubes under:

```text
data/raw/
```

The config files expect paths like:

```text
data/raw/NGC_2403_NA_CUBE_THINGS.FITS
data/raw/NGC_3031_NA_CUBE_THINGS.FITS
```

Generated datasets are written under `training_data/`. Training and evaluation outputs are written under `runs/`. Both are local products, not source files.

## Install

The intended environment is Python 3.11. The project includes both Conda and pip-style dependency files.

Conda:

```bash
cd hishells-pv-research
conda env create -f environment.yml
conda activate hishells
pip install -e .
```

Pip:

```bash
cd hishells-pv-research
python -m pip install -e ".[dev]"
```

TensorFlow and astronomy packages can be fussy on Apple Silicon. If your local setup splits FITS/WCS work and TensorFlow work into separate environments, the main training script supports separate Python executables through `PREP_PYTHON` and `TRAIN_PYTHON`.

## Quick Checks

Check the model file against its metadata:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path

root = Path(".")
meta = json.loads((root / "artifacts/model_metadata.json").read_text())
model = root / meta["model"]["file"]
print(model.exists())
print(hashlib.sha256(model.read_bytes()).hexdigest() == meta["model"]["sha256"])
PY
```

Check that the catalog table loads:

```bash
python - <<'PY'
from src.pv.shell_catalog import load_bagetakos_table7

catalog = load_bagetakos_table7("J_AJ_141_23_table7.dat.txt")
print(len(catalog))
print(sorted(catalog["shell_type"].dropna().astype(int).unique()))
PY
```

Expected catalog output is `1046` rows and shell types `[1, 2, 3]`.

## Run The Published Model

Inference needs generated PV arrays. If you already have a resolved training/inference config and a directory of PV `.npy` files, run:

```bash
python -m src.infer.infer_pv \
  --config training_data/standardized_5kpc_200kms_clean_physical_baseline/train_standardized_high_recall.yaml \
  --model artifacts/models/hi_shell_pv_unet_clean_physical_baseline.keras \
  --out runs/published_model_inference
```

The shell wrapper is:

```bash
scripts/infer.sh \
  training_data/standardized_5kpc_200kms_clean_physical_baseline/train_standardized_high_recall.yaml \
  artifacts/models/hi_shell_pv_unet_clean_physical_baseline.keras \
  runs/published_model_inference
```

The output is one probability map per input PV slice.

## Reproduce The Clean Physical Baseline

The clean physical baseline is the workflow that produced the published checkpoint. It:

1. builds standardized PV slices and masks,
2. runs a dataset self-test,
3. trains the high-recall U-Net,
4. evaluates validation, test, and stress splits.

Run it from the repository root:

```bash
PREP_PYTHON=python TRAIN_PYTHON=python scripts/run_clean_physical_baseline.sh
```

Useful environment overrides:

```bash
DATA_ROOT=training_data
GRID_STRIDE_PIX=32
GRID_MAX_PER_GALAXY=2000
GRID_ANGLE_STEP_DEG=22.5
EVERY=5
```

If your TensorFlow environment is separate:

```bash
PREP_PYTHON=/path/to/astro/python \
TRAIN_PYTHON=/path/to/tensorflow/python \
scripts/run_clean_physical_baseline.sh
```

The generated config is written to:

```text
training_data/standardized_5kpc_200kms_clean_physical_baseline/train_standardized_high_recall.yaml
```

The new run is written to:

```text
runs/<run-name>/
```

## Main Scripts

Common entry points:

- `scripts/run_clean_physical_baseline.sh`: end-to-end published workflow.
- `scripts/prepare_standardized_training_data.py`: build fixed-shape PV cuts, labels, split manifests, and training config.
- `scripts/train.sh`: train from an existing training config.
- `scripts/evaluate_pv_unet.py`: evaluate a model on one split.
- `scripts/infer.sh`: resolve config and run probability-map inference.
- `scripts/extract_candidate_components.py`: extract connected candidate regions from probability maps.
- `scripts/postprocess_component_filter_eval.py`: evaluate component filtering and ranking.
- `scripts/build_shell_review_catalog.py`: build review catalogs from candidate outputs.
- `scripts/validate_pv_training_data.py`: check generated PV and label arrays.

Older helper scripts are still present where they support the workflow, but the repo no longer carries their historical outputs.

## Source Map

The most useful modules are:

- `src/pv/shell_catalog.py`: load and normalize the Bagetakos shell catalog.
- `src/pv/standardized_cuts.py`: create standardized spatial and velocity windows.
- `src/pv/label_pv.py`: convert catalog shells into PV mask labels.
- `src/pv/dataset.py`: load generated PV arrays and masks as TensorFlow datasets.
- `src/train/models_unet.py`: define the U-Net used by the clean baseline.
- `src/train/losses.py`: define the recall-weighted segmentation losses and metrics.
- `src/train/train_pv_unet.py`: train the Keras U-Net.
- `src/infer/infer_pv.py`: run a saved model over PV arrays.
- `src/eval/eval_pv_unet.py`: evaluate pixel-level and patch-level behavior.
- `src/post/aggregate.py`: aggregate predictions back toward shell candidates.

## Shell-Type Analysis

The notebook:

```text
notebooks/shell_type_selection_effects.ipynb
```

asks whether Bagetakos shell Types 1, 2, and 3 behave like distinct physical populations or mostly reflect detectability. Its outputs and execution counts are cleared. It uses the two checked-in catalog tables, so it can be rerun without raw FITS cubes.

In short, the analysis treats the Type labels cautiously. Type 1 is dominated by the absence of a measured expansion velocity. Types 2 and 3 carry more velocity information, but the notebook tests how much of that separation is recoverable from non-velocity features.

## Notes For Paper Readers

If you only want to inspect or download the trained model, start with:

```text
artifacts/models/hi_shell_pv_unet_clean_physical_baseline.keras
artifacts/model_metadata.json
```

If you want to reproduce the dataset and training run, start with:

```text
scripts/run_clean_physical_baseline.sh
scripts/prepare_standardized_training_data.py
src/train/train_pv_unet.py
```

If you want the shell-type side analysis, start with:

```text
notebooks/shell_type_selection_effects.ipynb
```

## Caveats

The checkpoint is included, but the original FITS cubes are not. You need local THINGS cubes to regenerate PV cuts or rerun training.

The metric JSON files from the source run are not included. The metrics that matter for the paper-facing model are copied into `artifacts/model_metadata.json`.

Generated files can become large quickly. Keep local `data/raw/`, generated `training_data/` products, and `runs/` outputs out of commits unless you are intentionally publishing a specific small artifact.
