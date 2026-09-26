#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SONIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BRIDGE_ROOT="$(cd "$SONIC_ROOT/.." && pwd)"
REPO_ROOT="$(cd "$BRIDGE_ROOT/.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/tracker/sonic/gear_sonic_deploy"

DEFAULT_LAUNCHER_CONFIG_FILE="$REPO_ROOT/collector/sonic/scripts/collector_pc.env"
source "$REPO_ROOT/collector/sonic/scripts/load_launcher_config.sh"

ROBOT_INTERFACE="${ROBOT_INTERFACE:-real}"
CHECKPOINT_PREFIX="${G1_CHECKPOINT:-$REPO_ROOT/checkpoints/tracker/sonic/policy/release/model}"
DECODER_MODEL="${G1_DECODER_MODEL:-${CHECKPOINT_PREFIX}_decoder.onnx}"
OBS_CONFIG="${G1_OBS_CONFIG:-$REPO_ROOT/checkpoints/tracker/sonic/policy/release/observation_config.yaml}"
PLANNER="${G1_PLANNER:-$REPO_ROOT/checkpoints/tracker/sonic/planner/target_vel/V2/planner_sonic.onnx}"
MOTION_DATA="${G1_MOTION_DATA:-reference/example/}"
DEPLOY_HAND_BACKEND="${DEPLOY_HAND_BACKEND:-remote_wuji_proxy}"
DISABLE_TELEOP_SAFETY_GATE="${G1_DISABLE_TELEOP_SAFETY_GATE:-1}"
DISABLE_REPLAY_SAFETY="${G1_DISABLE_REPLAY_SAFETY:-1}"
BUILD_BEFORE_RUN="${G1_DEPLOY_BUILD:-0}"
DEPLOY_EXECUTABLE="${G1_DEPLOY_EXECUTABLE:-g1_deploy_onnx_ref_vla}"
ZMQ_PORT=5556
REPLAY_TARGET=""

show_usage() {
    echo "Usage: $0 [--zmq-port PORT] [--replay-target sim|real]"
    echo ""
    echo "Options:"
    echo "  --zmq-port PORT            Native pose subscriber port (default: 5556)"
    echo "  --replay-target sim|real   Require matching replay payload target"
    echo "  -h, --help                 Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zmq-port)
            if [[ $# -lt 2 || -z "${2:-}" ]]; then
                echo "Error: --zmq-port requires a value" >&2
                exit 2
            fi
            ZMQ_PORT="$2"
            shift 2
            ;;
        --replay-target)
            if [[ $# -lt 2 || -z "${2:-}" ]]; then
                echo "Error: --replay-target requires a value" >&2
                exit 2
            fi
            REPLAY_TARGET="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            show_usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$ZMQ_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || (( 10#$ZMQ_PORT > 65535 )); then
    echo "Error: --zmq-port must be an integer in 1..65535" >&2
    exit 2
fi
if [[ -n "$REPLAY_TARGET" && "$REPLAY_TARGET" != "sim" && "$REPLAY_TARGET" != "real" ]]; then
    echo "Error: --replay-target must be sim or real" >&2
    exit 2
fi

if [[ "$DEPLOY_HAND_BACKEND" != "remote_wuji_proxy" && "$DEPLOY_HAND_BACKEND" != "local_wuji" ]]; then
    echo "Invalid DEPLOY_HAND_BACKEND=$DEPLOY_HAND_BACKEND" >&2
    exit 1
fi

list_ipv4_interfaces() {
    ip -o -4 addr show | awk '{split($4, addr, "/"); print $2 ":" addr[1]}'
}

find_interface_by_ip_prefix() {
    local prefix="$1"
    list_ipv4_interfaces | while IFS=: read -r iface ip_addr; do
        if [[ "$ip_addr" == "$prefix"* ]]; then
            echo "$iface"
            return 0
        fi
    done
}

find_interface_by_ip() {
    local target_ip="$1"
    list_ipv4_interfaces | while IFS=: read -r iface ip_addr; do
        if [[ "$ip_addr" == "$target_ip" ]]; then
            echo "$iface"
            return 0
        fi
    done
}

resolve_deploy_interface() {
    local requested="$1"

    if [[ "$requested" == "sim" ]]; then
        RESOLVED_ROBOT_INTERFACE="lo"
        DEPLOY_ENV_TYPE="sim"
        return 0
    fi

    if [[ "$requested" == "real" ]]; then
        RESOLVED_ROBOT_INTERFACE="$(find_interface_by_ip_prefix "192.168.123." | head -n 1)"
        DEPLOY_ENV_TYPE="real"
        if [[ -z "$RESOLVED_ROBOT_INTERFACE" ]]; then
            echo "Could not auto-detect Unitree wired interface with IP 192.168.123.x." >&2
            echo "Available IPv4 interfaces:" >&2
            list_ipv4_interfaces >&2
            echo "Set ROBOT_INTERFACE explicitly in collector/sonic/scripts/collector_pc.env, for example ROBOT_INTERFACE=eno1." >&2
            exit 1
        fi
        return 0
    fi

    if [[ "$requested" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        RESOLVED_ROBOT_INTERFACE="$(find_interface_by_ip "$requested" | head -n 1)"
        if [[ -z "$RESOLVED_ROBOT_INTERFACE" ]]; then
            echo "Could not find a local interface with IP $requested." >&2
            echo "Available IPv4 interfaces:" >&2
            list_ipv4_interfaces >&2
            exit 1
        fi
    else
        RESOLVED_ROBOT_INTERFACE="$requested"
    fi

    if [[ "$RESOLVED_ROBOT_INTERFACE" == "lo" || "$RESOLVED_ROBOT_INTERFACE" == "lo0" ]]; then
        DEPLOY_ENV_TYPE="sim"
    else
        DEPLOY_ENV_TYPE="real"
    fi
}

resolve_deploy_interface "$ROBOT_INTERFACE"

cd "$DEPLOY_DIR"
if [[ "$BUILD_BEFORE_RUN" == "1" ]]; then
    just build
fi

if [[ ! -x "target/release/$DEPLOY_EXECUTABLE" ]]; then
    echo "Missing external-token deploy executable: target/release/$DEPLOY_EXECUTABLE" >&2
    echo "Run with G1_DEPLOY_BUILD=1, or run 'cd $DEPLOY_DIR && just build' first." >&2
    exit 1
fi

set +e
set +u
source "$REPO_ROOT/collector/sonic/scripts/load_launcher_config.sh"
set -u
set -e

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

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64 | amd64)
        UNITREE_LIB_ARCH="x86_64"
        ;;
    aarch64 | arm64)
        UNITREE_LIB_ARCH="aarch64"
        ;;
    *)
        UNITREE_LIB_ARCH="$HOST_ARCH"
        ;;
esac

UNITREE_SDK_ROOT="${UNITREE_SDK_ROOT:-$REPO_ROOT/third_party/unitree_sdk2}"
if [[ ! -d "$UNITREE_SDK_ROOT" ]]; then
    UNITREE_SDK_ROOT="$DEPLOY_DIR/thirdparty/unitree_sdk2"
fi

prepend_path_var LD_LIBRARY_PATH \
    "$UNITREE_SDK_ROOT/lib/$UNITREE_LIB_ARCH" \
    "$UNITREE_SDK_ROOT/lib" \
    "$UNITREE_SDK_ROOT/thirdparty/lib/$UNITREE_LIB_ARCH" \
    "$UNITREE_SDK_ROOT/thirdparty/lib" \
    "$DEPLOY_DIR/target/release"

EXTRA_ARGS=()

if [[ "$DEPLOY_ENV_TYPE" == "sim" ]]; then
    EXTRA_ARGS+=(--disable-crc-check)
fi
if [[ "$DISABLE_TELEOP_SAFETY_GATE" == "1" ]]; then
    EXTRA_ARGS+=(--disable-teleop-safety-gate)
fi
if [[ "$DISABLE_REPLAY_SAFETY" == "1" ]]; then
    EXTRA_ARGS+=(--disable-replay-safety)
fi
EXTRA_ARGS+=(--zmq-port "$ZMQ_PORT")
if [[ -n "$REPLAY_TARGET" ]]; then
    EXTRA_ARGS+=(--replay-target "$REPLAY_TARGET")
fi

echo "Launching deploy for external-policy+SONIC v4 external token mode"
echo "  executable:    $DEPLOY_EXECUTABLE"
echo "  decoder:       $DECODER_MODEL"
echo "  encoder:       <disabled>"
echo "  obs_config:    $OBS_CONFIG"
echo "  planner:       $PLANNER"
echo "  motion_data:   $MOTION_DATA"
echo "  interface:     $RESOLVED_ROBOT_INTERFACE (requested: $ROBOT_INTERFACE, env: $DEPLOY_ENV_TYPE)"
echo "  hand_backend:  $DEPLOY_HAND_BACKEND"
echo "  zmq_host:      127.0.0.1"
echo "  zmq_port:      $ZMQ_PORT"
echo "  replay_target: ${REPLAY_TARGET:-<legacy-unset>}"
echo "  teleop_safety: $([[ "$DISABLE_TELEOP_SAFETY_GATE" == "1" ]] && echo disabled || echo enabled)"
echo "  replay_safety: $([[ "$DISABLE_REPLAY_SAFETY" == "1" ]] && echo disabled || echo enabled)"
echo "  config_file:   $LAUNCHER_CONFIG_FILE"

exec just run "$DEPLOY_EXECUTABLE" "$RESOLVED_ROBOT_INTERFACE" "$DECODER_MODEL" "$MOTION_DATA" \
    --obs-config "$OBS_CONFIG" \
    --planner-file "$PLANNER" \
    --input-type zmq_manager \
    --output-type all \
    --zmq-host 127.0.0.1 \
    --hand-backend "$DEPLOY_HAND_BACKEND" \
    --left-wuji-serial "${LEFT_WUJI_SERIAL:-}" \
    --right-wuji-serial "${RIGHT_WUJI_SERIAL:-}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
