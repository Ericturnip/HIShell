#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
git_root="$(git -C "$repo_dir" rev-parse --show-toplevel)"
hooks_path="$(python - "$git_root" "$repo_dir/.githooks" <<'PY'
import os
import sys

print(os.path.relpath(sys.argv[2], sys.argv[1]))
PY
)"

git -C "$git_root" config core.hooksPath "$hooks_path"
echo "Git hooks installed from $hooks_path"
