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

TASK_NAME="${TASK_NAME:-}"
if [[ -z "$TASK_NAME" ]]; then
    if [[ -t 0 ]]; then
        read -r -p "Enter task name: " TASK_NAME
    fi
fi
if [[ -z "$TASK_NAME" ]]; then
    echo "Please set TASK_NAME in $LAUNCHER_CONFIG_FILE or enter it at startup" >&2
    exit 1
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="$REPO_ROOT/datasets/sonic"
fi
CAMERA_HOST="${CAMERA_HOST:-}"
CAMERA_PORT="${CAMERA_PORT:-5560}"
HAND_CONTROL_MODE="${HAND_CONTROL_MODE:-}"
DEPLOY_HAND_BACKEND="${DEPLOY_HAND_BACKEND:-remote_wuji_proxy}"
ROBOT_HAND_HOST="${ROBOT_HAND_HOST:-${CAMERA_HOST}}"
ROBOT_HAND_STATUS_PORT="${ROBOT_HAND_STATUS_PORT:-5559}"
ROBOT_HAND_STATUS_ENDPOINT="${ROBOT_HAND_STATUS_ENDPOINT:-tcp://${ROBOT_HAND_HOST}:${ROBOT_HAND_STATUS_PORT}}"
XR_LISTEN="${XR_LISTEN:-0.0.0.0:13579}"
XR_VIDEO_HOST="${XR_VIDEO_HOST:-}"
XR_LISTEN_BACKEND="${XR_LISTEN_BACKEND:-gst}"
RECORD_DEPTH="${RECORD_DEPTH:-1}"
DEFER_DEPTH_COMPRESSION="${DEFER_DEPTH_COMPRESSION:-1}"
CAPTURE_FPS="${CAPTURE_FPS:-20}"
CAPTURE_MODE="${CAPTURE_MODE:-collector_timer}"
CAMERA_FPS="${CAMERA_FPS:-}"
LOG_LEVEL="${COLLECTOR_LOG_LEVEL:-INFO}"

if [[ -z "$CAMERA_FPS" ]]; then
    if [[ "$CAPTURE_FPS" == "50" ]]; then
        CAMERA_FPS="60"
    else
        CAMERA_FPS="$CAPTURE_FPS"
    fi
fi

if [[ -z "$CAMERA_HOST" || "$CAMERA_HOST" == *"<"*">"* ]]; then
    echo "Please set CAMERA_HOST in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ -z "$CAMERA_PORT" || "$CAMERA_PORT" == *"<"*">"* ]]; then
    echo "Please set CAMERA_PORT in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ -z "$ROBOT_HAND_HOST" || "$ROBOT_HAND_HOST" == *"<"*">"* ]]; then
    echo "Please set ROBOT_HAND_HOST in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ -z "$ROBOT_HAND_STATUS_PORT" || "$ROBOT_HAND_STATUS_PORT" == *"<"*">"* ]]; then
    echo "Please set ROBOT_HAND_STATUS_PORT in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ "$HAND_CONTROL_MODE" != "gesture_wuji" && "$HAND_CONTROL_MODE" != "binary_trigger" && "$HAND_CONTROL_MODE" != "wuji_glove" && "$HAND_CONTROL_MODE" != "manus" ]]; then
    echo "Please set HAND_CONTROL_MODE to gesture_wuji, binary_trigger, wuji_glove, or manus in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ "$DEPLOY_HAND_BACKEND" != "local_wuji" && "$DEPLOY_HAND_BACKEND" != "remote_wuji_proxy" ]]; then
    echo "Please set DEPLOY_HAND_BACKEND to local_wuji or remote_wuji_proxy in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ "$CAPTURE_FPS" != "20" && "$CAPTURE_FPS" != "30" && "$CAPTURE_FPS" != "50" ]]; then
    echo "Please set CAPTURE_FPS to 20, 30, or 50 in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ "$CAPTURE_MODE" != "collector_timer" && "$CAPTURE_MODE" != "official_latest" ]]; then
    echo "Please set CAPTURE_MODE to collector_timer or official_latest in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ "$CAPTURE_FPS" == "50" && "$CAPTURE_MODE" != "official_latest" ]] || \
   [[ "$CAPTURE_FPS" != "50" && "$CAPTURE_MODE" != "collector_timer" ]]; then
    echo "CAPTURE_MODE=$CAPTURE_MODE is incompatible with CAPTURE_FPS=$CAPTURE_FPS; use collector_timer at 20/30Hz or official_latest at 50Hz" >&2
    exit 1
fi
if [[ "$CAMERA_FPS" != "20" && "$CAMERA_FPS" != "30" && "$CAMERA_FPS" != "60" ]]; then
    echo "Please set CAMERA_FPS to 20, 30, or 60 in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi

CAMERA_ENDPOINT="tcp://${CAMERA_HOST}:${CAMERA_PORT}"
POSE_ENDPOINT="tcp://127.0.0.1:5556"
STATE_ACTION_ENDPOINT="tcp://127.0.0.1:5558"

ARGS=(
    --task-name "$TASK_NAME"
    --output-root "$OUTPUT_ROOT"
    --robot-name "g1"
    --pose-endpoint "$POSE_ENDPOINT"
    --state-action-endpoint "$STATE_ACTION_ENDPOINT"
    --camera-backend "remote_realsense"
    --camera-endpoint "$CAMERA_ENDPOINT"
    --hand-control-mode "$HAND_CONTROL_MODE"
    --hand-backend "$DEPLOY_HAND_BACKEND"
    --listen "$XR_LISTEN"
    --listen-backend "$XR_LISTEN_BACKEND"
    --capture-fps "$CAPTURE_FPS"
    --capture-mode "$CAPTURE_MODE"
    --camera-fps "$CAMERA_FPS"
    --log-level "$LOG_LEVEL"
)
if [[ -n "$XR_VIDEO_HOST" ]]; then
    ARGS+=(--xr-video-host "$XR_VIDEO_HOST")
fi
if [[ -n "$ROBOT_HAND_STATUS_ENDPOINT" ]]; then
    ARGS+=(--hand-status-endpoint "$ROBOT_HAND_STATUS_ENDPOINT")
fi

if [[ "$RECORD_DEPTH" == "1" ]]; then
    ARGS+=(--record-depth)
else
    ARGS+=(--no-record-depth)
fi
if [[ "$DEFER_DEPTH_COMPRESSION" == "1" ]]; then
    ARGS+=(--defer-depth-compression)
else
    ARGS+=(--no-defer-depth-compression)
fi

if [[ $# -gt 0 ]]; then
    ARGS+=("$@")
fi

cd "$REPO_ROOT"

echo "Launching collector on PC"
echo "  python:                $PYTHON_BIN"
echo "  task_name:             $TASK_NAME"
echo "  output_root:           $OUTPUT_ROOT"
echo "  pose_endpoint:         $POSE_ENDPOINT"
echo "  state_action_endpoint: $STATE_ACTION_ENDPOINT"
echo "  camera_endpoint:       $CAMERA_ENDPOINT"
echo "  camera_host:           $CAMERA_HOST"
echo "  camera_port:           $CAMERA_PORT"
echo "  hand_mode:             $HAND_CONTROL_MODE"
echo "  hand_backend:          $DEPLOY_HAND_BACKEND"
echo "  robot_hand_host:       $ROBOT_HAND_HOST"
echo "  hand_status_port:      $ROBOT_HAND_STATUS_PORT"
echo "  hand_status_endpoint:  ${ROBOT_HAND_STATUS_ENDPOINT:-<disabled>}"
echo "  xr_listen:             $XR_LISTEN"
echo "  xr_video_host:         ${XR_VIDEO_HOST:-<request-ip>}"
echo "  xr_listen_backend:     $XR_LISTEN_BACKEND"
echo "  capture_fps:           $CAPTURE_FPS"
echo "  capture_mode:          $CAPTURE_MODE"
echo "  camera_fps:            $CAMERA_FPS"
echo "  record_depth:          $RECORD_DEPTH"
echo "  defer_depth_compress:  $DEFER_DEPTH_COMPRESSION"
echo "  config_file:           $LAUNCHER_CONFIG_FILE"

exec "$PYTHON_BIN" "$COLLECTOR_DIR/run_collector.py" "${ARGS[@]}"
