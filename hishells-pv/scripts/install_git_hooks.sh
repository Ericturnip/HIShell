#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

mkdir -p .git/hooks

cat > .git/hooks/pre-commit <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail
scripts/check_no_large_data.sh staged
HOOK

cat > .git/hooks/pre-push <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail
scripts/check_no_large_data.sh tracked
HOOK

chmod +x .git/hooks/pre-commit .git/hooks/pre-push scripts/check_no_large_data.sh
echo "[hooks] installed pre-commit and pre-push data guards."
