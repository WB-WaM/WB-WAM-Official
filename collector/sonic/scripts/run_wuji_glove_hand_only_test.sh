#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$COLLECTOR_DIR/../.." && pwd)"

DEFAULT_LAUNCHER_CONFIG_FILE="$SCRIPT_DIR/collector_pc.env"
source "$SCRIPT_DIR/load_launcher_config.sh"

PYTHON_BIN="${COLLECTOR_PYTHON:-$REPO_ROOT/.venv_teleop/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${COLLECTOR_PYTHON_FALLBACK:-python3}"
fi

prepend_path_var() {
    local var_name="$1"
    shift
    local current="${!var_name:-}"
    local prefix=""
    local path=""

    for path in "$@"; do
        if [[ -d "$path" ]]; then
            prefix="${prefix:+$prefix:}$path"
        fi
    done

    if [[ -n "$prefix" ]]; then
        export "$var_name=$prefix${current:+:$current}"
    fi
}

prepend_path_var PYTHONPATH     "$REPO_ROOT/tracker/sonic"     "$REPO_ROOT/third_party/wuji_sdk"     "$REPO_ROOT/third_party/wuji_retargeting"

if [[ "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
    require_wuji_glove_addresses "$@"
fi

HAND_ONLY_BIND="${WUJI_HAND_ONLY_BIND:-tcp://*:5556}"
HAND_ONLY_FPS="${WUJI_HAND_ONLY_FPS:-30}"
HAND_ONLY_STATUS_ENDPOINT="${WUJI_HAND_ONLY_STATUS_ENDPOINT:-${ROBOT_HAND_STATUS_ENDPOINT:-}}"

ARGS=(
    --bind "$HAND_ONLY_BIND"
    --left-address "${WUJI_GLOVE_LEFT_ADDRESS:-}"
    --right-address "${WUJI_GLOVE_RIGHT_ADDRESS:-}"
    --stale-s "${WUJI_GLOVE_STALE_S:-0.2}"
    --fps "$HAND_ONLY_FPS"
)
if [[ -n "$HAND_ONLY_STATUS_ENDPOINT" ]]; then
    ARGS+=(--status-endpoint "$HAND_ONLY_STATUS_ENDPOINT")
fi
if [[ -n "${WUJI_GLOVE_LEFT_RETARGETING_CONFIG:-}" ]]; then
    ARGS+=(--left-retargeting-config "$WUJI_GLOVE_LEFT_RETARGETING_CONFIG")
fi
if [[ -n "${WUJI_GLOVE_RIGHT_RETARGETING_CONFIG:-}" ]]; then
    ARGS+=(--right-retargeting-config "$WUJI_GLOVE_RIGHT_RETARGETING_CONFIG")
fi

cd "$REPO_ROOT"

echo "Launching Wuji glove hand-only test"
echo "  python:          $PYTHON_BIN"
echo "  publish_bind:    $HAND_ONLY_BIND"
echo "  left_glove:      ${WUJI_GLOVE_LEFT_ADDRESS:-}"
echo "  right_glove:     ${WUJI_GLOVE_RIGHT_ADDRESS:-}"
echo "  stale_s:         ${WUJI_GLOVE_STALE_S:-0.2}"
echo "  fps:             $HAND_ONLY_FPS"
echo "  status_endpoint: ${HAND_ONLY_STATUS_ENDPOINT:-<disabled>}"
echo "  config_file:     $LAUNCHER_CONFIG_FILE"
echo "  note:            do not run run_pico_manager.sh on the same PC/port"

exec "$PYTHON_BIN" "$SCRIPT_DIR/wuji_glove_hand_only_test.py" "${ARGS[@]}" "$@"
