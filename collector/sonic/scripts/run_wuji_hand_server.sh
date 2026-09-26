#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$COLLECTOR_DIR/../.." && pwd)"

DEFAULT_LAUNCHER_CONFIG_FILE="$SCRIPT_DIR/wuji_hand_server.env"
source "$SCRIPT_DIR/load_launcher_config.sh"

PYTHON_BIN="${COLLECTOR_PYTHON:-$REPO_ROOT/.venv_teleop/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${COLLECTOR_PYTHON_FALLBACK:-python3}"
fi

PC_ZMQ_HOST="${PC_ZMQ_HOST:-}"
WUJI_HAND_STATUS_PORT="${WUJI_HAND_STATUS_PORT:-5559}"
WUJI_HAND_STATUS_BIND="${WUJI_HAND_STATUS_BIND:-tcp://0.0.0.0:${WUJI_HAND_STATUS_PORT}}"
LEFT_WUJI_SERIAL="${LEFT_WUJI_SERIAL:-}"
RIGHT_WUJI_SERIAL="${RIGHT_WUJI_SERIAL:-}"
WUJI_HAND_DRY_RUN="${WUJI_HAND_DRY_RUN:-0}"
WUJI_HAND_LOG_LEVEL="${WUJI_HAND_LOG_LEVEL:-INFO}"

if [[ -z "$PC_ZMQ_HOST" || "$PC_ZMQ_HOST" == *"<"*">"* ]]; then
    echo "Please set PC_ZMQ_HOST in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ -z "$LEFT_WUJI_SERIAL" && "$WUJI_HAND_DRY_RUN" != "1" ]]; then
    echo "Please set LEFT_WUJI_SERIAL in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi
if [[ -z "$RIGHT_WUJI_SERIAL" && "$WUJI_HAND_DRY_RUN" != "1" ]]; then
    echo "Please set RIGHT_WUJI_SERIAL in $LAUNCHER_CONFIG_FILE" >&2
    exit 1
fi

POSE_ENDPOINT="tcp://${PC_ZMQ_HOST}:5556"

ARGS=(
    --pose-endpoint "$POSE_ENDPOINT"
    --status-bind "$WUJI_HAND_STATUS_BIND"
    --left-wuji-serial "$LEFT_WUJI_SERIAL"
    --right-wuji-serial "$RIGHT_WUJI_SERIAL"
    --log-level "$WUJI_HAND_LOG_LEVEL"
)
if [[ "$WUJI_HAND_DRY_RUN" == "1" ]]; then
    ARGS+=(--dry-run)
fi

cd "$COLLECTOR_DIR"

echo "Launching robot-side Wuji hand server"
echo "  python:        $PYTHON_BIN"
echo "  pose_endpoint: $POSE_ENDPOINT"
echo "  status_bind:   $WUJI_HAND_STATUS_BIND"
echo "  status_port:   $WUJI_HAND_STATUS_PORT"
echo "  left_serial:   $LEFT_WUJI_SERIAL"
echo "  right_serial:  $RIGHT_WUJI_SERIAL"
echo "  dry_run:       $WUJI_HAND_DRY_RUN"
echo "  log_level:     $WUJI_HAND_LOG_LEVEL"
echo "  config_file:   $LAUNCHER_CONFIG_FILE"

exec "$PYTHON_BIN" "$COLLECTOR_DIR/wuji_hand_server.py" "${ARGS[@]}"
