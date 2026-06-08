# HIShells PV — hunting HI shells in PV space

Galaxies blow holes in their neutral hydrogen. Supernovae and stellar winds
sweep up the gas into expanding **HI shells**, and in position-velocity (PV)
cuts those shells show up as tell-tale ring/arc signatures. `hishells_pv` is a
small research pipeline that goes looking for them in THINGS HI cubes.

The idea: take a cube + a shell catalog, slice standardized PV cuts, train a
high-recall U-Net to flag shell-like structure, then turn the network's
probability maps into ranked candidate catalogs you can actually follow up on.

Works either as a library (`import hishells_pv`) or the `hishells-pv` CLI.

## Quickstart

```bash
conda create -n hishells python=3.11
conda activate hishells
pip install -e .
```

Run the whole thing end-to-end:

```bash
# 1. standardized PV cuts + labels
hishells-pv generate --data-root training_data --out-root training_data/standardized

# 2. train the U-Net (auto-picks MPS / CUDA / CPU)
hishells-pv train --config training_data/standardized/train_standardized_high_recall.yaml --run pv_unet_baseline

# 3. probability maps
hishells-pv infer --model runs/pv_unet_baseline/best_model.pt --config training_data/standardized/train_standardized_high_recall.yaml

# 4. calibrate a threshold on the val split
hishells-pv calibrate --model runs/pv_unet_baseline/best_model.pt --config training_data/standardized/train_standardized_high_recall.yaml --split val

# 5. ranked candidate components
hishells-pv postprocess --model runs/pv_unet_baseline/best_model.pt --config training_data/standardized/train_standardized_high_recall.yaml --split test

# 6. project PV votes back onto the sky, per galaxy
hishells-pv aggregate-all --output-root training_data/standardized --run-dir runs/pv_unet_baseline --splits test stress
```

`hishells-pv --help` lists everything. The device is auto-detected (MPS → CUDA →
CPU); override with `--device`.

Optional extras:

```bash
pip install -e ".[analysis]"   # statsmodels + scikit-learn for analysis.ipynb
pip install -e ".[dev]"        # pytest, jupyter, etc.
```

## The commands

| Command | What it does |
|---|---|
| `hishells-pv generate` | Make standardized PV cuts + labels |
| `hishells-pv generate-all` | Per-galaxy PV/label generation driver |
| `hishells-pv validate` | Sanity-check the generated training data |
| `hishells-pv combine` | Combine per-galaxy training data |
| `hishells-pv train` | Train the PyTorch U-Net |
| `hishells-pv infer` | Write per-PV probability maps |
| `hishells-pv calibrate` | Calibrate a global probability threshold |
| `hishells-pv postprocess` | Pull out ranked candidate components |
| `hishells-pv aggregate` | Aggregate one galaxy's PV predictions into sky candidates |
| `hishells-pv aggregate-all` | Aggregate every galaxy in a standardized dataset |
| `hishells-pv resolve-config` | Resolve a config (FITS inference + defaults) |

## A note on sky-plane aggregation

Aggregation is per-galaxy: it projects PV votes back onto a single cube's image
plane via the `*_posxy.npy` sidecars, but the standardized dataset is
galaxy-mixed. So `generate` also emits, per evaluation galaxy, an isolated
`agg/<galaxy>/` input dir and an aggregate-ready `<galaxy>_agg_config.yaml` (with
the `cube_path` and `galaxy` block). `aggregate-all` walks every galaxy in the
requested splits and writes to `runs/<run>/aggregate_<galaxy>_<split>`. The split
label is `<galaxy>_<split>`, so per-galaxy outputs never collide. To run just one:

```bash
hishells-pv aggregate \
  --config training_data/standardized/agg/ngc_7793/ngc_7793_agg_config.yaml \
  --run-dir runs/pv_unet_baseline --split ngc_7793_test
```

## As a library

```python
from hishells_pv import load_bagetakos_table7, UNetPV
from hishells_pv.train.trainer import train
from hishells_pv.infer.predict import predict_run

holes = load_bagetakos_table7("catalogs/J_AJ_141_23_table7.dat.txt")
```

## Selection-effects analysis

`analysis.ipynb` (repo root) asks a fun question: does the Bagetakos HI-hole
*Type* taxonomy reflect real physics, or is it mostly a detectability cut? It
reuses this package's catalog loader:

```python
from hishells_pv.catalog.shell_catalog import load_bagetakos_table7
```

## What's in here

- `hishells_pv/` — the installable package:
  - `catalog/` — loads the Bagetakos/CDS shell catalog.
  - `pv/` — makes and labels PV cuts (`make_pv`, `label_pv`, `standardized_cuts`).
  - `data/` — the PyTorch dataset and array prep.
  - `models/` — `UNetPV`, losses, metrics.
  - `train/` — the training loop.
  - `infer/` — inference, threshold calibration, post-processing, sky aggregation.
  - `datagen/` — end-to-end data-generation drivers.
  - `qa/`, `utils/` — config resolution, FITS/WCS helpers, IO.
  - `cli.py` — the `hishells-pv` entry point.
- `configs/` — example galaxy/config YAMLs.
- `catalogs/` — small Bagetakos/CDS catalog text tables used for labels.
- `tests/` — regression and smoke tests (including a synthetic end-to-end run).
- `scripts/` — safety helpers (large-data guard, git hooks) plus aggregation, QA, and plotting scripts.
- `data/`, `training_data/`, `runs/` — local-only output dirs, ignored by Git.

## Safety hooks

```bash
scripts/install_git_hooks.sh        # install pre-commit / pre-push guards
scripts/check_no_large_data.sh all  # block raw FITS, checkpoints, large arrays
```

## Tests

```bash
pytest
```

Covers catalog parsing, the U-Net forward pass, losses, the dataset, config
helpers, and a synthetic end-to-end `train → infer → calibrate → postprocess`
smoke test. No external data needed.
