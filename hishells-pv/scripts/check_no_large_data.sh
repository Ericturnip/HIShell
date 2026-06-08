#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-staged}"
MAX_BYTES="${MAX_GIT_BYTES:-99614720}" # 95 MiB, below GitHub's hard 100 MB limit.

usage() {
  cat <<'USAGE'
Usage: scripts/check_no_large_data.sh [staged|tracked|all]

Blocks raw FITS cubes, generated model/data artifacts, and files larger than
MAX_GIT_BYTES. Defaults to staged files for pre-commit use.
USAGE
}

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

case "$MODE" in
  staged)
    files_cmd=(git diff --cached --name-only -z --diff-filter=ACMR)
    ;;
  tracked)
    files_cmd=(git ls-files -z)
    ;;
  all)
    files_cmd=(git ls-files -z --cached --others --exclude-standard)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "[data-guard] unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

blocked=0
blocked_ext='fits|fit|fts|fits.gz|npy|npz|h5|hdf5|keras|ckpt|pt|pth|onnx|pkl|pickle'

while IFS= read -r -d '' file; do
  [ -e "$file" ] || continue
  lower="$(printf '%s' "$file" | tr '[:upper:]' '[:lower:]')"

  case "$lower" in
    *.fits|*.fit|*.fts|*.fits.gz|*.npy|*.npz|*.h5|*.hdf5|*.keras|*.ckpt|*.pt|*.pth|*.onnx|*.pkl|*.pickle)
      echo "[data-guard] blocked generated/raw artifact: $file" >&2
      blocked=1
      ;;
  esac

  size="$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)"
  if [ "$size" -gt "$MAX_BYTES" ]; then
    mib=$(( (size + 1048575) / 1048576 ))
    echo "[data-guard] blocked large file (${mib} MiB): $file" >&2
    blocked=1
  fi
done < <("${files_cmd[@]}")

if [ "$blocked" -ne 0 ]; then
  cat >&2 <<EOF

[data-guard] Refusing to continue.
Move large data/checkpoints outside the repo, or keep them in ignored folders
such as data/, training_data/, runs/, checkpoints/, or models/.
EOF
  exit 1
fi

echo "[data-guard] OK: no FITS/generated/large files detected in $MODE set."
