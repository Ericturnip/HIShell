#!/usr/bin/env bash
set -euo pipefail

# End-to-end PV-shell training pipeline. Run from the repository root.
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-training_data}"
CONFIG="${CONFIG:-training_data/combined_train.yaml}"
RUN="${RUN:-pv_unet_combined_$(date +%Y%m%d_%H%M%S)}"

echo "== 1. Generate PV slices and labels =="
"$PYTHON" scripts/generate_all_training_pv.py

echo "== 2. Validate generated PV/label arrays =="
"$PYTHON" scripts/validate_pv_training_data.py --data-root "$DATA_ROOT"

echo "== 3. Build combined galaxy-held-out training dataset =="
"$PYTHON" scripts/prepare_combined_training_data.py \
  --data-root "$DATA_ROOT" \
  --out-root "$DATA_ROOT/combined" \
  --config "$CONFIG" \
  --force

echo "== 4. Dataset selftest =="
"$PYTHON" -m hishells_pv.data.tensorflow_dataset --config "$CONFIG" --selftest

echo "== 5. Train 2D CNN/U-Net =="
"$PYTHON" -m hishells_pv.train.tensorflow --config "$CONFIG" --run "$RUN" --quiet --every 1
echo "$RUN" > "$DATA_ROOT/current_training_run.txt"

echo "== 6. Evaluate best checkpoint on held-out test galaxies =="
"$PYTHON" scripts/evaluate_pv_unet.py \
  --config "$CONFIG" \
  --model "runs/$RUN/best_model.keras" \
  --split test \
  --out "runs/$RUN/eval_test.json"

echo "== done =="
echo "run: runs/$RUN"
