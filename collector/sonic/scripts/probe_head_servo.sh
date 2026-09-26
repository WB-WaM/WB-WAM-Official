#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LAUNCHER_CONFIG_FILE="$SCRIPT_DIR/head_servo.env"
source "$SCRIPT_DIR/load_launcher_config.sh"
HEAD_DIR="$SCRIPT_DIR/../head"
BIN="${HEAD_SERVO_BINARY:-$HEAD_DIR/build/head_servo}"
[[ -x "$BIN" ]] || { echo 'Build the head service with head/setup_robot.sh first.' >&2; exit 1; }
ARGS=()
[[ -z "${HEAD_SERIAL_DEVICE:-}" ]] || ARGS+=(--device "$HEAD_SERIAL_DEVICE")
DEVICE="$(python3 "$HEAD_DIR/discover_serial.py" "${ARGS[@]}")"
exec "$BIN" --device "$DEVICE" --probe --duration 2
