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
    "$REPO_ROOT/third_party/wuji_sdk" \
    "$REPO_ROOT/third_party/wuji_retargeting" \
    "$REPO_ROOT/third_party/MANUS_SDK/ManusSDK_v3.1.1/SDKClient_Linux" \
    "$REPO_ROOT/third_party/xrobotoolkit_sdk"
prepend_path_var LD_LIBRARY_PATH "$REPO_ROOT/third_party/xrobotoolkit_sdk/lib"

ROBOTICS_SERVICE_DIR="${ROBOTICS_SERVICE_DIR:-/opt/apps/roboticsservice}"
if [[ -d "$ROBOTICS_SERVICE_DIR" ]]; then
    prepend_path_var LD_LIBRARY_PATH \
        "$ROBOTICS_SERVICE_DIR/lib" \
        "$ROBOTICS_SERVICE_DIR" \
        "$ROBOTICS_SERVICE_DIR/SDK/x64"
    prepend_path_var QT_PLUGIN_PATH "$ROBOTICS_SERVICE_DIR/plugins"
    prepend_path_var QT_QML_PATH "$ROBOTICS_SERVICE_DIR/qml"
fi

PICO_MANAGER_PORT="5556"
PICO_TARGET_FPS="${PICO_TARGET_FPS:-50}"
PICO_FEEDBACK_HOST="127.0.0.1"
PICO_FEEDBACK_PORT="5557"
PICO_USE_CUDA="${PICO_USE_CUDA:-0}"
PICO_WAIST_TRACKING="${PICO_WAIST_TRACKING:-0}"
HAND_CONTROL_MODE="${HAND_CONTROL_MODE:-}"
PLANNER_CONTROL_SOURCE="${PLANNER_CONTROL_SOURCE:-keyboard}"
WUJI_GLOVE_LEFT_ADDRESS="${WUJI_GLOVE_LEFT_ADDRESS:-}"
WUJI_GLOVE_RIGHT_ADDRESS="${WUJI_GLOVE_RIGHT_ADDRESS:-}"
WUJI_GLOVE_STALE_S="${WUJI_GLOVE_STALE_S:-0.2}"
WUJI_GLOVE_LEFT_RETARGETING_CONFIG="${WUJI_GLOVE_LEFT_RETARGETING_CONFIG:-}"
WUJI_GLOVE_RIGHT_RETARGETING_CONFIG="${WUJI_GLOVE_RIGHT_RETARGETING_CONFIG:-}"
MANUS_STALE_S="${MANUS_STALE_S:-0.2}"
MANUS_INIT_TIMEOUT_S="${MANUS_INIT_TIMEOUT_S:-10.0}"
MANUS_MCP_JOINT="${MANUS_MCP_JOINT:-proximal}"
MANUS_DEBUG_NODE_INFO="${MANUS_DEBUG_NODE_INFO:-0}"
MANUS_LOAD_CALIBRATION="${MANUS_LOAD_CALIBRATION:-1}"
MANUS_CALIBRATION_FILE="${MANUS_CALIBRATION_FILE:-$REPO_ROOT/third_party/MANUS_SDK/calibrations/Calibration.mcal}"
MANUS_LEFT_CALIBRATION_FILE="${MANUS_LEFT_CALIBRATION_FILE:-$REPO_ROOT/third_party/MANUS_SDK/calibrations/Calibration_left.mcal}"
MANUS_RIGHT_CALIBRATION_FILE="${MANUS_RIGHT_CALIBRATION_FILE:-$REPO_ROOT/third_party/MANUS_SDK/calibrations/Calibration_right.mcal}"
MANUS_LEFT_RETARGETING_CONFIG="${MANUS_LEFT_RETARGETING_CONFIG:-$REPO_ROOT/third_party/wuji_retargeting/example/config/retarget_manus_left.yaml}"
MANUS_RIGHT_RETARGETING_CONFIG="${MANUS_RIGHT_RETARGETING_CONFIG:-$REPO_ROOT/third_party/wuji_retargeting/example/config/retarget_manus_right.yaml}"

if [[ "$HAND_CONTROL_MODE" != "gesture_wuji" && "$HAND_CONTROL_MODE" != "binary_trigger" && "$HAND_CONTROL_MODE" != "wuji_glove" && "$HAND_CONTROL_MODE" != "manus" ]]; then
    echo "Please set HAND_CONTROL_MODE to gesture_wuji, binary_trigger, wuji_glove, or manus in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ "$HAND_CONTROL_MODE" == "wuji_glove" ]]; then
    require_wuji_glove_addresses "$@"
fi
if [[ "$PLANNER_CONTROL_SOURCE" != "pico" && "$PLANNER_CONTROL_SOURCE" != "keyboard" && "$PLANNER_CONTROL_SOURCE" != "hybrid" ]]; then
    echo "Please set PLANNER_CONTROL_SOURCE to pico, keyboard, or hybrid in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi

EXTRA_ARGS=()
if [[ "$PICO_USE_CUDA" == "1" ]]; then
    EXTRA_ARGS+=("--cuda")
fi
if [[ "$PICO_WAIST_TRACKING" == "1" ]]; then
    EXTRA_ARGS+=("--waist_tracking")
fi
EXTRA_ARGS+=("--hand-control-mode" "$HAND_CONTROL_MODE")
if [[ "$HAND_CONTROL_MODE" == "wuji_glove" ]]; then
    EXTRA_ARGS+=("--wuji-glove-left-address" "$WUJI_GLOVE_LEFT_ADDRESS")
    EXTRA_ARGS+=("--wuji-glove-right-address" "$WUJI_GLOVE_RIGHT_ADDRESS")
    EXTRA_ARGS+=("--wuji-glove-stale-s" "$WUJI_GLOVE_STALE_S")
    if [[ -n "$WUJI_GLOVE_LEFT_RETARGETING_CONFIG" ]]; then
        EXTRA_ARGS+=("--wuji-glove-left-retargeting-config" "$WUJI_GLOVE_LEFT_RETARGETING_CONFIG")
    fi
    if [[ -n "$WUJI_GLOVE_RIGHT_RETARGETING_CONFIG" ]]; then
        EXTRA_ARGS+=("--wuji-glove-right-retargeting-config" "$WUJI_GLOVE_RIGHT_RETARGETING_CONFIG")
    fi
fi
if [[ "$HAND_CONTROL_MODE" == "manus" ]]; then
    EXTRA_ARGS+=("--manus-stale-s" "$MANUS_STALE_S")
    EXTRA_ARGS+=("--manus-init-timeout-s" "$MANUS_INIT_TIMEOUT_S")
    EXTRA_ARGS+=("--manus-left-retargeting-config" "$MANUS_LEFT_RETARGETING_CONFIG")
    EXTRA_ARGS+=("--manus-right-retargeting-config" "$MANUS_RIGHT_RETARGETING_CONFIG")
    EXTRA_ARGS+=("--manus-calibration-file" "$MANUS_CALIBRATION_FILE")
    EXTRA_ARGS+=("--manus-left-calibration-file" "$MANUS_LEFT_CALIBRATION_FILE")
    EXTRA_ARGS+=("--manus-right-calibration-file" "$MANUS_RIGHT_CALIBRATION_FILE")
    EXTRA_ARGS+=("--manus-mcp-joint" "$MANUS_MCP_JOINT")
    if [[ "$MANUS_DEBUG_NODE_INFO" == "1" ]]; then
        EXTRA_ARGS+=("--manus-debug-node-info")
    fi
    if [[ "$MANUS_LOAD_CALIBRATION" != "1" ]]; then
        EXTRA_ARGS+=("--no-manus-load-calibration")
    fi
fi
EXTRA_ARGS+=("--planner-control-source" "$PLANNER_CONTROL_SOURCE")

cd "$REPO_ROOT/tracker/sonic"

echo "Launching pico manager thread server"
echo "  python:         $PYTHON_BIN"
echo "  port:           $PICO_MANAGER_PORT"
echo "  target_fps:     $PICO_TARGET_FPS"
echo "  feedback_host:  $PICO_FEEDBACK_HOST"
echo "  feedback_port:  $PICO_FEEDBACK_PORT"
echo "  use_cuda:       $PICO_USE_CUDA"
echo "  waist_tracking: $PICO_WAIST_TRACKING"
echo "  hand_mode:      $HAND_CONTROL_MODE"
if [[ "$HAND_CONTROL_MODE" == "wuji_glove" ]]; then
    echo "  glove_left:     $WUJI_GLOVE_LEFT_ADDRESS"
    echo "  glove_right:    $WUJI_GLOVE_RIGHT_ADDRESS"
    echo "  glove_stale_s:  $WUJI_GLOVE_STALE_S"
fi
if [[ "$HAND_CONTROL_MODE" == "manus" ]]; then
    echo "  manus_stale_s:  $MANUS_STALE_S"
    echo "  manus_mcp:      $MANUS_MCP_JOINT"
    echo "  manus_left_cal: $MANUS_LEFT_CALIBRATION_FILE"
    echo "  manus_right_cal:$MANUS_RIGHT_CALIBRATION_FILE"
fi
echo "  planner_source: $PLANNER_CONTROL_SOURCE"
echo "  config_file:    $LAUNCHER_CONFIG_FILE"
echo "  controls:       X+Y start/stop policy, left-stick toggles pose, right-stick/A/B controls collector"

exec "$PYTHON_BIN" "$REPO_ROOT/tracker/sonic/gear_sonic/scripts/pico_manager_thread_server.py" \
    --manager \
    --port "$PICO_MANAGER_PORT" \
    --target_fps "$PICO_TARGET_FPS" \
    --zmq_feedback_host "$PICO_FEEDBACK_HOST" \
    --zmq_feedback_port "$PICO_FEEDBACK_PORT" \
    "${EXTRA_ARGS[@]}" \
    "$@"
