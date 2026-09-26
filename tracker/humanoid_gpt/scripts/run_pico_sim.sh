#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GMR_ROOT="${PROJECT_ROOT}/.deps/GMR"
XRT_ROOT="${PROJECT_ROOT}/.deps/xrobotoolkit_sdk"

if [[ "$(uname -m)" == "aarch64" ]]; then
  XRT_LIB_DIR="${XRT_ROOT}/lib/aarch64"
else
  XRT_LIB_DIR="${XRT_ROOT}/lib"
fi

export PYTHONPATH="${PROJECT_ROOT}:${GMR_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${XRT_LIB_DIR}:/opt/apps/roboticsservice/lib:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="/opt/apps/roboticsservice/plugins:${QT_PLUGIN_PATH:-}"
export QML2_IMPORT_PATH="/opt/apps/roboticsservice/qml:${QML2_IMPORT_PATH:-}"
export PYTHONUNBUFFERED=1

# XRobo uses a localhost gRPC channel; socks5h proxy URIs are unsupported.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

if [[ -n "${HUMANOID_GPT_PYTHON:-}" ]]; then
  PYTHON_CMD=("${HUMANOID_GPT_PYTHON}")
elif [[ "${CONDA_DEFAULT_ENV:-}" == "h-gpt" ]]; then
  PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n h-gpt python)
else
  PYTHON_CMD=(python3)
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_CMD[@]}" -m deploy.play_track \
  --mocap-type pico \
  --buffer-ms 0 \
  "$@"
