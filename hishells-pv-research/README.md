# HI Shell Detection With Standardized PV Slices

This repository contains the working code for my HI shell detection project. It uses position-velocity cuts from THINGS galaxies and trains a high-recall U-Net to find shell-like structure.

Each PV slice is resampled into the same physical frame: 5 kpc across the spatial axis and 200 km/s across the velocity axis. The code keeps catalog-centered cuts, offset cuts, velocity-offset cuts, fine-grid cuts, and background cuts separate so the validation numbers say what kind of case the model handled.

## Repository Contents

- `src/`: PV extraction, shell labeling, dataset loading, training, evaluation, and post-processing.
- `scripts/`: Command-line entrypoints for generating cuts, training models, evaluating runs, and cleaning candidates.
- `training_data/configs/`: Per-galaxy config files. They point to `data/raw/*.FITS`, which Git ignores.
- `docs/figures/`: Paper-style figures and captions for the project presentation.
- `docs/model_label_examples/`: Example PV panels showing labels and model probabilities.
- `docs/results/clean_physical_baseline/`: Small JSON/CSV summaries from the clean physical baseline run. Model weights are left out.
- `J_AJ_141_23_table2.dat.txt` and `J_AJ_141_23_table7.dat.txt`: Bagetakos et al. catalog tables used to build shell labels.

## Data

Raw THINGS FITS cubes are too large for this repo. Generated PV arrays, labels, model checkpoints, and TensorBoard logs are local outputs too. Put local cubes under:

```bash
data/raw/
```

The configs expect names such as:

```bash
data/raw/NGC_2403_NA_CUBE_THINGS.FITS
data/raw/NGC_3031_NA_CUBE_THINGS.FITS
```

Install the Git hook before committing:

```bash
bash scripts/install_hooks.sh
```

The hook blocks FITS files, generated numpy arrays, and model checkpoints.

## Environment

```bash
conda env create -f environment.yml
conda activate hishells
pip install -e .
```

For a lighter install:

```bash
python -m pip install -e ".[dev]"
```

## Clean Physical Baseline Run

Generate standardized cuts, train the high-recall model, and evaluate validation, test, and stress splits:

```bash
PREP_PYTHON=python TRAIN_PYTHON=python scripts/run_clean_physical_baseline.sh
```

This run uses:

- PV arrays with shape `(96, 256)`.
- A 5 kpc spatial window.
- A 200 km/s velocity window.
- Local-median padding at cube boundaries.
- Beam-aware label cleanup.
- No hard-negative injection.
- Tversky-style loss, with false negatives weighted more heavily than false positives.
- NGC 3031 held out as a stress-test galaxy because M81-group tidal structure creates severe line-of-sight confusion.

## Evaluation

The evaluation code reports pixel and patch precision, recall, and F1 by cut category:

- catalog-centered
- offset / grazing
- velocity-offset
- fine-grid deployment
- background / random negatives

The fine-grid score is the deployment benchmark. It is closest to how the model would search a new galaxy without catalog-centered help.

For report writing and grading:

- `docs/METHODS_CODE_MAP.md` maps each method step to the named functions and modules that implement it.
- `docs/FIGURE_REPRODUCIBILITY.md` lists the script and function used to produce each report figure.

## Tests

```bash
pytest
```

The tests focus on configuration loading, dataset behavior, loss numerics, and the synthetic PV pipeline.
