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

prepend_path_var PYTHONPATH \
    "$REPO_ROOT/tracker/sonic" \
    "$REPO_ROOT/third_party/wuji_retargeting" \
    "$REPO_ROOT/third_party/MANUS_SDK/ManusSDK_v3.1.1/SDKClient_Linux"

MANUS_HAND_ONLY_BIND="${MANUS_HAND_ONLY_BIND:-tcp://*:5556}"
MANUS_HAND_ONLY_FPS="${MANUS_HAND_ONLY_FPS:-30}"
MANUS_HAND_ONLY_STALE_S="${MANUS_HAND_ONLY_STALE_S:-0.2}"
MANUS_HAND_ONLY_STATUS_ENDPOINT="${MANUS_HAND_ONLY_STATUS_ENDPOINT:-${ROBOT_HAND_STATUS_ENDPOINT:-}}"
MANUS_MCP_JOINT="${MANUS_MCP_JOINT:-proximal}"
MANUS_DEBUG_NODE_INFO="${MANUS_DEBUG_NODE_INFO:-0}"
MANUS_CALIBRATION_FILE="${MANUS_CALIBRATION_FILE:-$REPO_ROOT/third_party/MANUS_SDK/calibrations/Calibration.mcal}"
MANUS_LEFT_CALIBRATION_FILE="${MANUS_LEFT_CALIBRATION_FILE:-$REPO_ROOT/third_party/MANUS_SDK/calibrations/Calibration_left.mcal}"
MANUS_RIGHT_CALIBRATION_FILE="${MANUS_RIGHT_CALIBRATION_FILE:-$REPO_ROOT/third_party/MANUS_SDK/calibrations/Calibration_right.mcal}"
MANUS_LEFT_RETARGETING_CONFIG="${MANUS_LEFT_RETARGETING_CONFIG:-$REPO_ROOT/third_party/wuji_retargeting/example/config/retarget_manus_left.yaml}"
MANUS_RIGHT_RETARGETING_CONFIG="${MANUS_RIGHT_RETARGETING_CONFIG:-$REPO_ROOT/third_party/wuji_retargeting/example/config/retarget_manus_right.yaml}"

ARGS=(
    --bind "$MANUS_HAND_ONLY_BIND"
    --fps "$MANUS_HAND_ONLY_FPS"
    --stale-s "$MANUS_HAND_ONLY_STALE_S"
    --left-retargeting-config "$MANUS_LEFT_RETARGETING_CONFIG"
    --right-retargeting-config "$MANUS_RIGHT_RETARGETING_CONFIG"
    --calibration-file "$MANUS_CALIBRATION_FILE"
    --left-calibration-file "$MANUS_LEFT_CALIBRATION_FILE"
    --right-calibration-file "$MANUS_RIGHT_CALIBRATION_FILE"
    --manus-mcp-joint "$MANUS_MCP_JOINT"
)
if [[ "$MANUS_DEBUG_NODE_INFO" == "1" ]]; then
    ARGS+=(--debug-node-info)
fi
if [[ -n "$MANUS_HAND_ONLY_STATUS_ENDPOINT" ]]; then
    ARGS+=(--status-endpoint "$MANUS_HAND_ONLY_STATUS_ENDPOINT")
fi

cd "$REPO_ROOT"

echo "Launching MANUS hand-only test"
echo "  python:           $PYTHON_BIN"
echo "  publish_bind:     $MANUS_HAND_ONLY_BIND"
echo "  fps:              $MANUS_HAND_ONLY_FPS"
echo "  stale_s:          $MANUS_HAND_ONLY_STALE_S"
echo "  mcp_joint:        $MANUS_MCP_JOINT"
echo "  debug_node_info:  $MANUS_DEBUG_NODE_INFO"
echo "  calibration_file: $MANUS_CALIBRATION_FILE"
echo "  left_calibration: $MANUS_LEFT_CALIBRATION_FILE"
echo "  right_calibration:$MANUS_RIGHT_CALIBRATION_FILE"
echo "  status_endpoint:  ${MANUS_HAND_ONLY_STATUS_ENDPOINT:-<disabled>}"
echo "  left_config:      $MANUS_LEFT_RETARGETING_CONFIG"
echo "  right_config:     $MANUS_RIGHT_RETARGETING_CONFIG"
echo "  config_file:      $LAUNCHER_CONFIG_FILE"
echo "  note:             do not run run_pico_manager.sh on the same PC/port"

exec "$PYTHON_BIN" "$SCRIPT_DIR/manus_hand_only_test.py" "${ARGS[@]}" "$@"
