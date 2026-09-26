#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
COLLECTION_ENV="${REPO_ROOT}/collector/humanoid_gpt/scripts/hgpt_collection.env"
GMR_ROOT="${PROJECT_ROOT}/.deps/GMR"
XRT_ROOT="${PROJECT_ROOT}/.deps/xrobotoolkit_sdk"
MANUS_ROOT="${PROJECT_ROOT}/.deps/manus_sdk"

# shellcheck disable=SC1090
source "${COLLECTION_ENV}"

has_option() {
  local requested="$1"
  shift
  local raw option
  for raw in "$@"; do
    option="${raw%%=*}"
    option="${option//_/-}"
    if [[ "${option}" == "${requested}" ]]; then
      return 0
    fi
  done
  return 1
}

DEFAULT_ARGS=()
if ! has_option --hand-source "$@"; then
  DEFAULT_ARGS+=(--hand-source "${HGPT_HAND_SOURCE}")
fi
if ! has_option --pico-service-mode "$@"; then
  DEFAULT_ARGS+=(--pico-service-mode "${HGPT_PICO_SERVICE_MODE}")
fi
if ! has_option --hand-status-endpoint "$@"; then
  DEFAULT_ARGS+=(--hand-status-endpoint "${HGPT_HAND_STATUS_ENDPOINT}")
fi
if ! has_option --publish-wuji-hand "$@" && \
  ! has_option --no-publish-wuji-hand "$@"; then
  case "${HGPT_PUBLISH_WUJI_HAND,,}" in
    1|true|yes|on)
      DEFAULT_ARGS+=(--publish-wuji-hand)
      ;;
    0|false|no|off)
      DEFAULT_ARGS+=(--no-publish-wuji-hand)
      ;;
    *)
      printf 'Invalid HGPT_PUBLISH_WUJI_HAND: %s\n' \
        "${HGPT_PUBLISH_WUJI_HAND}" >&2
      exit 2
      ;;
  esac
fi

if [[ "$(uname -m)" == "aarch64" ]]; then
  XRT_LIB_DIR="${XRT_ROOT}/lib/aarch64"
else
  XRT_LIB_DIR="${XRT_ROOT}/lib"
fi

export PYTHONPATH="${PROJECT_ROOT}:${GMR_ROOT}:${MANUS_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${XRT_LIB_DIR}:${MANUS_ROOT}/ManusSDK/lib:/opt/apps/roboticsservice/lib:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="/opt/apps/roboticsservice/plugins:${QT_PLUGIN_PATH:-}"
export QML2_IMPORT_PATH="/opt/apps/roboticsservice/qml:${QML2_IMPORT_PATH:-}"
export PYTHONUNBUFFERED=1
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
exec "${PYTHON_CMD[@]}" -m deploy.pose_manager "${DEFAULT_ARGS[@]}" "$@"
