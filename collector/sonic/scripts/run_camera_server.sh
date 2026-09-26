#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$COLLECTOR_DIR/../.." && pwd)"

DEFAULT_LAUNCHER_CONFIG_FILE="$SCRIPT_DIR/camera_server.env"
source "$SCRIPT_DIR/load_launcher_config.sh"

PYTHON_BIN="${COLLECTOR_PYTHON:-$REPO_ROOT/.venv_teleop/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${COLLECTOR_PYTHON_FALLBACK:-python3}"
fi

CAMERA_BIND_PORT="${CAMERA_BIND_PORT:-5560}"
CAMERA_BIND="${CAMERA_BIND:-tcp://0.0.0.0:${CAMERA_BIND_PORT}}"
CAMERA_WIDTH="${WIDTH:-640}"
CAMERA_HEIGHT="${HEIGHT:-480}"
CAPTURE_FPS="${CAPTURE_FPS:-}"
CAMERA_FPS="${CAMERA_FPS:-${FPS:-}}"
if [[ -z "$CAMERA_FPS" ]]; then
    if [[ "$CAPTURE_FPS" == "20" || "$CAPTURE_FPS" == "30" ]]; then
        CAMERA_FPS="$CAPTURE_FPS"
    else
        CAMERA_FPS="60"
    fi
fi
CAMERA_JPEG_QUALITY="${JPEG_QUALITY:-80}"
CAMERA_RECORD_DEPTH="${CAMERA_RECORD_DEPTH:-1}"
CAMERA_LOG_LEVEL="${CAMERA_LOG_LEVEL:-INFO}"

if [[ -n "$CAPTURE_FPS" && "$CAPTURE_FPS" != "20" && "$CAPTURE_FPS" != "30" && "$CAPTURE_FPS" != "50" ]]; then
    echo "Please set CAPTURE_FPS to 20, 30, or 50 in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ "$CAMERA_FPS" != "20" && "$CAMERA_FPS" != "30" && "$CAMERA_FPS" != "60" ]]; then
    echo "Please set CAMERA_FPS to 20, 30, or 60 in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi

DEPTH_FLAG="--record-depth"
if [[ "$CAMERA_RECORD_DEPTH" == "0" ]]; then
    DEPTH_FLAG="--no-record-depth"
fi

# The real head module needs its servo hold service in deployment AND collection.
# Diagnostic camera-only operation must be explicit and is not deployment readiness.
HEAD_SERVO_ENABLED="${HEAD_SERVO_ENABLED:-1}"
HEAD_APPROVED=0
if [[ "${1:-}" == --head-motion-approved ]]; then HEAD_APPROVED=1; shift; fi
[[ $# -eq 0 ]] || { echo 'Usage: run_camera_server.sh [--head-motion-approved]' >&2; exit 2; }
[[ "$HEAD_SERVO_ENABLED" == 0 || "$HEAD_SERVO_ENABLED" == 1 ]] || { echo 'HEAD_SERVO_ENABLED must be 0 or 1' >&2; exit 2; }
if [[ "$HEAD_SERVO_ENABLED" == 1 && "$HEAD_APPROVED" != 1 ]]; then
    echo 'Head servo hold is required. Verify the configured head encoder pose (default 3027/1849), then pass --head-motion-approved.' >&2
    echo 'For camera-only diagnostics, explicitly set HEAD_SERVO_ENABLED=0.' >&2
    exit 2
fi

cd "$COLLECTOR_DIR"

echo "Launching collector camera server"
echo "  python:        $PYTHON_BIN"
echo "  bind:          $CAMERA_BIND"
echo "  bind_port:     $CAMERA_BIND_PORT"
echo "  profile:       ${CAMERA_WIDTH}x${CAMERA_HEIGHT}@${CAMERA_FPS}"
echo "  jpeg_quality:  $CAMERA_JPEG_QUALITY"
echo "  record_depth:  $CAMERA_RECORD_DEPTH"
echo "  log_level:     $CAMERA_LOG_LEVEL"
echo "  config_file:   $LAUNCHER_CONFIG_FILE"

CAMERA_COMMAND=("$PYTHON_BIN" "$COLLECTOR_DIR/camera_server.py" \
    --bind "$CAMERA_BIND" \
    --width "$CAMERA_WIDTH" \
    --height "$CAMERA_HEIGHT" \
    --fps "$CAMERA_FPS" \
    --jpeg-quality "$CAMERA_JPEG_QUALITY" \
    "$DEPTH_FLAG" \
    --log-level "$CAMERA_LOG_LEVEL")
if [[ "$HEAD_SERVO_ENABLED" == 1 ]]; then
    exec python3 "$COLLECTOR_DIR/head/supervise.py" \
        --head-launcher "$SCRIPT_DIR/run_head_servo.sh" \
        --head-config "${HEAD_SERVO_CONFIG_FILE:-$SCRIPT_DIR/head_servo.env}" \
        -- "${CAMERA_COMMAND[@]}"
fi
echo 'Camera-only diagnostic mode: head servo readiness is NOT verified.'
exec "${CAMERA_COMMAND[@]}"
