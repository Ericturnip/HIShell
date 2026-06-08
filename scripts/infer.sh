#!/usr/bin/env bash
set -euo pipefail

config="${1:-}"
model="${2:-}"
out_dir="${3:-}"

if [[ -z "${config}" || -z "${model}" ]]; then
  echo "Usage: scripts/infer.sh <config.yaml> <model.keras> [output_dir]"
  exit 2
fi

echo "== Resolve config =="
python -m hishells_pv.qa.print_resolved_config --config "${config}"

echo "== Infer PV probability maps =="
if [[ -n "${out_dir}" ]]; then
  python -m hishells_pv.infer.infer_pv --config "${config}" --model "${model}" --out "${out_dir}"
else
  python -m hishells_pv.infer.infer_pv --config "${config}" --model "${model}"
fi

echo "Inference complete."
