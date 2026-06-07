#!/usr/bin/env bash
set -euo pipefail

# End-to-end Clean Physical Baseline. Run from the repository root.
# The local conda envs may split astronomy and TensorFlow dependencies. Override
# independently when needed, e.g.
# PREP_PYTHON=python TRAIN_PYTHON=/opt/homebrew/anaconda3/envs/tf/bin/python scripts/run_clean_physical_baseline.sh

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

PREP_PYTHON="${PREP_PYTHON:-python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${PYTHON:-python}}"
DATA_ROOT="${DATA_ROOT:-training_data}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/standardized_5kpc_200kms_clean_physical_baseline}"
CONFIG="$OUT_ROOT/train_standardized_high_recall.yaml"
RUN="${RUN:-pv_unet_clean_physical_baseline_$(date +%Y%m%d_%H%M%S)}"
EVERY="${EVERY:-5}"

echo "== 1. Build clean standardized physical dataset =="
"$PREP_PYTHON" scripts/prepare_standardized_training_data.py \
  --data-root "$DATA_ROOT" \
  --out-root "$OUT_ROOT" \
  --spatial-window-kpc 5.0 \
  --velocity-window-kms 200.0 \
  --target-velocity-bins 96 \
  --target-spatial-pixels 256 \
  --grid-stride-pix "${GRID_STRIDE_PIX:-32}" \
  --grid-max-per-galaxy "${GRID_MAX_PER_GALAXY:-2000}" \
  --grid-angle-step-deg "${GRID_ANGLE_STEP_DEG:-22.5}" \
  --stress-galaxies ngc_3031 \
  --force

echo "== 2. Dataset selftest =="
"$TRAIN_PYTHON" -m src.pv.dataset --config "$CONFIG" --selftest

echo "== 3. Train high-recall clean baseline =="
"$TRAIN_PYTHON" -m src.train.train_pv_unet --config "$CONFIG" --run "$RUN" --quiet --every "$EVERY"
echo "$RUN" > "$DATA_ROOT/current_clean_physical_baseline_run.txt"

echo "== 4. Evaluate held-out validation/test/stress splits =="
for SPLIT in val test stress; do
  "$TRAIN_PYTHON" scripts/evaluate_pv_unet.py \
    --config "$CONFIG" \
    --model "runs/$RUN/high_recall_model.keras" \
    --split "$SPLIT" \
    --out "runs/$RUN/eval_${SPLIT}_high_recall_model.json"
done

echo "== done =="
echo "dataset: $OUT_ROOT"
echo "run: runs/$RUN"
