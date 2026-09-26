#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

check_env() {
    local name="$1"
    local python_bin="$2"
    local py_path="${3:-}"
    shift 3 || true
    local modules=("$@")

    echo "== $name =="
    if [[ ! -x "$python_bin" ]]; then
        echo "missing: $python_bin"
        return 0
    fi
    PYTHONPATH="$py_path${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" - "${modules[@]}" <<'PY'
import importlib
import sys

print("python", sys.version.split()[0])
for name in sys.argv[1:]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(name, "ERR", type(exc).__name__, exc)
        continue
    print(name, "OK", getattr(module, "__version__", "installed"))
PY
}

check_env "teleop (.venv_teleop)" \
    "$REPO_ROOT/.venv_teleop/bin/python" \
    "$REPO_ROOT/tracker/sonic/.deps/manus_sdk:$REPO_ROOT/tracker/sonic:$REPO_ROOT/third_party/wuji_retargeting" \
    numpy torch cv2 zmq msgpack gear_sonic wuji_retargeting xrobotoolkit_sdk ManusServer

check_env "sim (.venv_sim)" \
    "$REPO_ROOT/.venv_sim/bin/python" \
    "$REPO_ROOT/tracker/sonic" \
    numpy torch cv2 zmq mujoco gear_sonic unitree_sdk2py

check_env "wam bridge (bridge/.venv-wam)" \
    "$REPO_ROOT/bridge/.venv-wam/bin/python" \
    "$REPO_ROOT/training/src:$REPO_ROOT" \
    numpy torch torchvision cv2 zmq onnxruntime transformers wbwam.runtime
