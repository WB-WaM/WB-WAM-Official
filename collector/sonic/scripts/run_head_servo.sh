#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LAUNCHER_CONFIG_FILE="$SCRIPT_DIR/head_servo.env"
source "$SCRIPT_DIR/load_launcher_config.sh"
HEAD_DIR="$SCRIPT_DIR/../head"
BIN="${HEAD_SERVO_BINARY:-$HEAD_DIR/build/head_servo}"
[[ -x "$BIN" ]] || { echo 'Build the head service with head/setup_robot.sh first.' >&2; exit 1; }
CONTROL_ARGS=()
case "${HEAD_SERVO_MODE:-raw}" in
    raw)
        CONTROL_ARGS=(--hold-raw --joint0-encoder "${HEAD_JOINT0_ENCODER:-3027}" --joint1-encoder "${HEAD_JOINT1_ENCODER:-1849}")
        ;;
    teach) CONTROL_ARGS=(--teach) ;;
    target)
        [[ -n "${HEAD_SERVO0_CALIBRATION:-}" && -n "${HEAD_SERVO1_CALIBRATION:-}" ]] || {
            echo 'Target pose requires verified HEAD_SERVO0_CALIBRATION and HEAD_SERVO1_CALIBRATION in head_servo.env (Unitree lower-limit encoder counts).' >&2
            exit 1
        }
        CONTROL_ARGS=(--hold-target --joint0-deg "${HEAD_JOINT0_DEG:--50}" --joint1-deg "${HEAD_JOINT1_DEG:-10}"
                      --servo0-calibration "$HEAD_SERVO0_CALIBRATION" --servo1-calibration "$HEAD_SERVO1_CALIBRATION")
        ;;
    current) CONTROL_ARGS=(--hold-current) ;;
    *) echo 'HEAD_SERVO_MODE must be raw, teach, target or current' >&2; exit 2 ;;
esac
ARGS=()
[[ -z "${HEAD_SERIAL_DEVICE:-}" ]] || ARGS+=(--device "$HEAD_SERIAL_DEVICE")
DEVICE="$(python3 "$HEAD_DIR/discover_serial.py" "${ARGS[@]}")"
exec "$BIN" --device "$DEVICE" "${CONTROL_ARGS[@]}" "$@"
