#!/usr/bin/env bash
set -euo pipefail

# Quiet TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=3

# Tweaks (can override via env: CONFIG=..., RUN=..., EVERY=...)
CONFIG="${CONFIG:-training_data/combined_train.yaml}"
RUN="${RUN:-pv_unet_$(date +%Y%m%d_%H%M)}"
EVERY="${EVERY:-1}"
PYTHON="${PYTHON:-python}"

echo "== dataset selftest =="
"$PYTHON" -m src.pv.dataset --config "$CONFIG" --selftest

echo "== training =="
"$PYTHON" -m src.train.train_pv_unet --config "$CONFIG" --run "$RUN" --quiet --every "$EVERY"

echo "== outputs =="
echo "  - best model: runs/$RUN/best_model.keras"
echo "  - final model: runs/$RUN/final_model.keras"
echo "  - logs: runs/$RUN/history.csv"
