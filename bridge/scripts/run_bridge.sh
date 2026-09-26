#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRIDGE_PYTHON="${WBWAM_BRIDGE_PYTHON:-${WBWAM_BRIDGE_VENV:-$REPO_ROOT/bridge/.venv-wam}/bin/python}"
if [[ ! -x "$BRIDGE_PYTHON" ]]; then
  echo 'Run bridge/scripts/setup_env.sh first, or set WBWAM_BRIDGE_PYTHON.' >&2
  exit 1
fi
cd "$REPO_ROOT"
exec "$BRIDGE_PYTHON" -m bridge "$@"
