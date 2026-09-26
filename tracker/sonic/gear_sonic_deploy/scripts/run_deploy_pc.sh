#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DEPLOY_DIR/../../.." && pwd)"

DEFAULT_LAUNCHER_CONFIG_FILE="$REPO_ROOT/collector/sonic/scripts/collector_pc.env"
source "$REPO_ROOT/collector/sonic/scripts/load_launcher_config.sh"

DEPLOY_LAUNCHER="${G1_DEPLOY_LAUNCHER:-$DEPLOY_DIR/deploy.sh}"
ROBOT_INTERFACE="${ROBOT_INTERFACE:-real}"
BUILD_BEFORE_RUN="${G1_DEPLOY_BUILD:-0}"
DEPLOY_HAND_BACKEND="${DEPLOY_HAND_BACKEND:-remote_wuji_proxy}"
DISABLE_TELEOP_SAFETY_GATE="${G1_DISABLE_TELEOP_SAFETY_GATE:=1}"

if [[ "$DEPLOY_HAND_BACKEND" != "local_wuji" && "$DEPLOY_HAND_BACKEND" != "remote_wuji_proxy" ]]; then
    echo "Invalid DEPLOY_HAND_BACKEND=$DEPLOY_HAND_BACKEND (expected local_wuji or remote_wuji_proxy)" >&2
    exit 1
fi

ARGS=()
if [[ "$BUILD_BEFORE_RUN" == "1" ]]; then
    ARGS+=(--build)
fi
if [[ "$DISABLE_TELEOP_SAFETY_GATE" == "1" ]]; then
    ARGS+=(--disable-teleop-safety-gate)
fi
if [[ -n "${G1_CHECKPOINT:-}" ]]; then
    ARGS+=(--checkpoint "$G1_CHECKPOINT")
fi
if [[ -n "${G1_OBS_CONFIG:-}" ]]; then
    ARGS+=(--obs-config "$G1_OBS_CONFIG")
fi
if [[ -n "${G1_PLANNER:-}" ]]; then
    ARGS+=(--planner "$G1_PLANNER")
fi
if [[ -n "${G1_MOTION_DATA:-}" ]]; then
    ARGS+=(--motion-data "$G1_MOTION_DATA")
fi

ARGS+=(
    --input-type "zmq_manager"
    --output-type "all"
    --zmq-host "127.0.0.1"
    --hand-backend "$DEPLOY_HAND_BACKEND"
    "$ROBOT_INTERFACE"
)
if [[ -n "${LEFT_WUJI_SERIAL:-}" ]]; then
    ARGS+=(--left-wuji-serial "$LEFT_WUJI_SERIAL")
fi
if [[ -n "${RIGHT_WUJI_SERIAL:-}" ]]; then
    ARGS+=(--right-wuji-serial "$RIGHT_WUJI_SERIAL")
fi

if [[ $# -gt 0 ]]; then
    ARGS+=("$@")
fi

cd "$DEPLOY_DIR"

echo "Launching g1 deploy on PC"
echo "  launcher:       $DEPLOY_LAUNCHER"
echo "  robot_interface: $ROBOT_INTERFACE"
echo "  zmq_host:       127.0.0.1"
echo "  input_type:     zmq_manager"
echo "  output_type:    all"
echo "  hand_backend:   $DEPLOY_HAND_BACKEND"
echo "  teleop_safety:  $([[ "$DISABLE_TELEOP_SAFETY_GATE" == "1" ]] && echo disabled || echo enabled)"
echo "  left_wuji:      ${LEFT_WUJI_SERIAL:-<not passed>}"
echo "  right_wuji:     ${RIGHT_WUJI_SERIAL:-<not passed>}"
echo "  build:          $BUILD_BEFORE_RUN"
echo "  config_file:    $LAUNCHER_CONFIG_FILE"

exec bash "$DEPLOY_LAUNCHER" "${ARGS[@]}"
