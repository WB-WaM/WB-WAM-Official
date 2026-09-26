#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$COLLECTOR_DIR/../.." && pwd)"

PYTHON_BIN="${COLLECTOR_PYTHON:-$REPO_ROOT/.venv_teleop/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${COLLECTOR_PYTHON_FALLBACK:-python3}"
fi

CURRENT_IP="${WUJI_GLOVE_CURRENT_IP:-192.168.1.100}"
CURRENT_PORT="${WUJI_GLOVE_CURRENT_PORT:-50000}"
TARGET_IP="${WUJI_GLOVE_TARGET_IP:-}"
TARGET_PORT="${WUJI_GLOVE_TARGET_PORT:-}"
DEVICE_NAME="${WUJI_GLOVE_DEVICE_NAME:-glove}"
IFACE="${WUJI_GLOVE_INTERFACE:-eno1}"
PC_TEMP_IP="${WUJI_GLOVE_PC_TEMP_IP:-192.168.1.10/24}"
ADD_TEMP_ALIAS="${WUJI_GLOVE_ADD_TEMP_ALIAS:-1}"
CONFIRM="${WUJI_GLOVE_CONFIRM:-0}"
DRY_RUN=0
ADDED_TEMP_ALIAS=0

usage() {
    cat <<'EOF'
Usage:
  collector/sonic/scripts/run_wuji_glove_set_ip.sh --target-ip 192.168.123.100
  collector/sonic/scripts/run_wuji_glove_set_ip.sh --target-ip 192.168.123.101

Defaults:
  --current-ip       192.168.1.100
  --current-port     50000
  --interface        eno1
  --pc-temp-ip       192.168.1.10/24

Common flow:
  1. Connect exactly one glove to the 192.168.123 switch.
  2. Run this script for 192.168.123.100.
  3. Power-cycle/reconnect that glove and verify ping.
  4. Repeat with the other glove for 192.168.123.101.
EOF
}

while (($#)); do
    case "$1" in
        --current-ip)
            CURRENT_IP="$2"
            shift 2
            ;;
        --current-port)
            CURRENT_PORT="$2"
            shift 2
            ;;
        --target-ip)
            TARGET_IP="$2"
            shift 2
            ;;
        --target-port)
            TARGET_PORT="$2"
            shift 2
            ;;
        --device-name)
            DEVICE_NAME="$2"
            shift 2
            ;;
        --interface)
            IFACE="$2"
            shift 2
            ;;
        --pc-temp-ip)
            PC_TEMP_IP="$2"
            shift 2
            ;;
        --no-temp-alias)
            ADD_TEMP_ALIAS=0
            shift
            ;;
        --yes|-y)
            CONFIRM=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$TARGET_IP" ]]; then
    echo "Please pass --target-ip, for example 192.168.123.100" >&2
    usage >&2
    exit 2
fi

cleanup() {
    if [[ "$ADDED_TEMP_ALIAS" == "1" ]]; then
        echo "[INFO] Removing temporary $PC_TEMP_IP from $IFACE"
        sudo ip addr del "$PC_TEMP_IP" dev "$IFACE" || true
    fi
}
trap cleanup EXIT

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
    "$REPO_ROOT/third_party/wuji_sdk" \
    "$REPO_ROOT/third_party/wuji_sdk/src"

echo "Wuji glove IP rewrite"
echo "  python:       $PYTHON_BIN"
echo "  sdk_path:     $REPO_ROOT/third_party/wuji_sdk"
echo "  interface:    $IFACE"
echo "  pc_temp_ip:   ${PC_TEMP_IP:-<disabled>}"
echo "  current:      $CURRENT_IP:$CURRENT_PORT"
echo "  target_ip:    $TARGET_IP"
echo "  target_port:  ${TARGET_PORT:-<unchanged>}"
echo "  device_name:  $DEVICE_NAME"
echo ""
echo "Connect exactly one glove with current IP $CURRENT_IP before continuing."

if [[ "$CONFIRM" != "1" ]]; then
    read -r -p "Type the target IP to continue: " reply
    if [[ "$reply" != "$TARGET_IP" ]]; then
        echo "Aborted: confirmation did not match target IP" >&2
        exit 1
    fi
fi

if [[ "$ADD_TEMP_ALIAS" == "1" && -n "$PC_TEMP_IP" ]]; then
    if ip -br addr show dev "$IFACE" | grep -Fq "$PC_TEMP_IP"; then
        echo "[INFO] $IFACE already has $PC_TEMP_IP"
    else
        echo "[INFO] Adding temporary $PC_TEMP_IP to $IFACE"
        sudo ip addr add "$PC_TEMP_IP" dev "$IFACE"
        ADDED_TEMP_ALIAS=1
    fi
fi

ARGS=(
    --current-ip "$CURRENT_IP"
    --current-port "$CURRENT_PORT"
    --target-ip "$TARGET_IP"
    --device-name "$DEVICE_NAME"
)

if [[ -n "$TARGET_PORT" ]]; then
    ARGS+=(--target-port "$TARGET_PORT")
fi
if [[ "$DRY_RUN" == "1" ]]; then
    ARGS+=(--dry-run)
fi

"$PYTHON_BIN" "$SCRIPT_DIR/wuji_glove_set_ip.py" "${ARGS[@]}"
