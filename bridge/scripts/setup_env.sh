#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRIDGE_VENV="${WBWAM_BRIDGE_VENV:-$REPO_ROOT/bridge/.venv-wam}"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
uv venv --allow-existing "$BRIDGE_VENV" --python "${WBWAM_PYTHON_VERSION:-3.10}"
if [[ "$(uname -s)" == Darwin ]]; then
  # CPU smoke checks only; real deployment uses Linux and CUDA.
  uv pip install --python "$BRIDGE_VENV/bin/python" torch==2.7.1 torchvision==0.22.1
else
  uv pip install --python "$BRIDGE_VENV/bin/python" torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
fi
uv pip install --python "$BRIDGE_VENV/bin/python" -r "$REPO_ROOT/bridge/requirements.txt"
uv pip install --python "$BRIDGE_VENV/bin/python" --no-deps -e "$REPO_ROOT/training"
echo "Ready: $BRIDGE_VENV/bin/python -m bridge --help (from repository root)"
