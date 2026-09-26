#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$COLLECTOR_DIR/../.." && pwd)"

CAPTURE_FPS_VALUE="${CAPTURE_FPS:-20}"
CAPTURE_MODE_VALUE="${CAPTURE_MODE:-collector_timer}"
CAMERA_FPS_VALUE="${CAMERA_FPS:-}"
ACTION="print"
HEAD_ARGS=()

usage() {
    cat <<USAGE
Usage:
  collector/sonic/scripts/run_data_collection_flow.sh [--fps 20|30|50]
      [--capture-mode collector_timer|official_latest] [--camera-fps 20|30|60]
      [--head-motion-approved] <action>

Actions:
  print          Print the full multi-terminal startup flow. Default action.
  camera         Run camera + head servo on the robot (--head-motion-approved required).
  head-probe     Read both head servos without enabling torque.
  hand           Run robot-side Wuji hand server.
  deploy         Run SONIC deploy on the PC.
  pico           Run pico_manager_thread_server on the PC.
  collector      Run the PC collector.

Examples:
  collector/sonic/scripts/run_data_collection_flow.sh print
  collector/sonic/scripts/run_data_collection_flow.sh --fps 30 print
  collector/sonic/scripts/run_data_collection_flow.sh --fps 20 print
  collector/sonic/scripts/run_data_collection_flow.sh --fps 20 --head-motion-approved camera
  collector/sonic/scripts/run_data_collection_flow.sh --fps 20 collector
  collector/sonic/scripts/run_data_collection_flow.sh --fps 50 --capture-mode official_latest collector

Notes:
  - Default: collector_timer at 20Hz; collector_timer also supports 30Hz.
  - 50Hz requires --capture-mode official_latest, with a default 60Hz camera source.
  - PICO_TARGET_FPS is a separate teleop input-stream setting and is not changed here.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fps)
            if [[ $# -lt 2 ]]; then
                echo "--fps requires 20, 30, or 50" >&2
                exit 1
            fi
            CAPTURE_FPS_VALUE="$2"
            shift 2
            ;;
        --fps=*)
            CAPTURE_FPS_VALUE="${1#--fps=}"
            shift
            ;;
        --capture-mode)
            if [[ $# -lt 2 ]]; then
                echo "--capture-mode requires collector_timer or official_latest" >&2
                exit 1
            fi
            CAPTURE_MODE_VALUE="$2"
            shift 2
            ;;
        --capture-mode=*)
            CAPTURE_MODE_VALUE="${1#--capture-mode=}"
            shift
            ;;
        --camera-fps)
            if [[ $# -lt 2 ]]; then
                echo "--camera-fps requires 20, 30, or 60" >&2
                exit 1
            fi
            CAMERA_FPS_VALUE="$2"
            shift 2
            ;;
        --camera-fps=*)
            CAMERA_FPS_VALUE="${1#--camera-fps=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        print|plan)
            ACTION="print"
            shift
            ;;
        --head-motion-approved)
            HEAD_ARGS+=(--head-motion-approved)
            shift
            ;;
        head-probe)
            ACTION="head-probe"
            shift
            ;;
        camera|camera-server)
            ACTION="camera"
            shift
            ;;
        hand|hand-server|wuji)
            ACTION="hand"
            shift
            ;;
        deploy|sonic)
            ACTION="deploy"
            shift
            ;;
        pico|pico-manager)
            ACTION="pico"
            shift
            ;;
        collector|pc-collector)
            ACTION="collector"
            shift
            ;;
        *)
            echo "Unknown argument/action: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$CAPTURE_FPS_VALUE" != "20" && "$CAPTURE_FPS_VALUE" != "30" && "$CAPTURE_FPS_VALUE" != "50" ]]; then
    echo "CAPTURE_FPS must be 20, 30, or 50, got: $CAPTURE_FPS_VALUE" >&2
    exit 1
fi
case "$CAPTURE_MODE_VALUE" in
    collector_timer|official_latest) ;;
    *)
        echo "Unsupported CAPTURE_MODE: $CAPTURE_MODE_VALUE" >&2
        exit 1
        ;;
esac
if [[ "$CAPTURE_FPS_VALUE" == "50" ]]; then
    if [[ "$CAPTURE_MODE_VALUE" == "collector_timer" ]]; then
        echo "CAPTURE_MODE=collector_timer is incompatible with CAPTURE_FPS=50" >&2
        exit 1
    fi
elif [[ "$CAPTURE_MODE_VALUE" == "official_latest" ]]; then
    echo "CAPTURE_MODE=$CAPTURE_MODE_VALUE requires CAPTURE_FPS=50" >&2
    exit 1
fi
if [[ -z "$CAMERA_FPS_VALUE" ]]; then
    if [[ "$CAPTURE_FPS_VALUE" == "50" ]]; then
        CAMERA_FPS_VALUE="60"
    else
        CAMERA_FPS_VALUE="$CAPTURE_FPS_VALUE"
    fi
fi
if [[ "$CAMERA_FPS_VALUE" != "20" && "$CAMERA_FPS_VALUE" != "30" && "$CAMERA_FPS_VALUE" != "60" ]]; then
    echo "CAMERA_FPS must be 20, 30, or 60, got: $CAMERA_FPS_VALUE" >&2
    exit 1
fi

export CAPTURE_FPS="$CAPTURE_FPS_VALUE"
export CAPTURE_MODE="$CAPTURE_MODE_VALUE"
export CAMERA_FPS="$CAMERA_FPS_VALUE"

print_flow() {
    cat <<FLOW
# WB-WAM / SONIC data collection startup flow
# Selected capture FPS: ${CAPTURE_FPS}
# Selected capture mode: ${CAPTURE_MODE}
# Selected camera FPS:  ${CAMERA_FPS}

# 0. Edit configs once before running:
#    PC:          collector/sonic/scripts/collector_pc.env
#    Robot camera: collector/sonic/scripts/camera_server.env
#    Robot hand:  collector/sonic/scripts/wuji_hand_server.env
#    Robot head:  collector/sonic/scripts/head_servo.env
#
# Endpoint mapping:
# - Camera frames: robot CAMERA_BIND/CAMERA_BIND_PORT -> PC CAMERA_HOST:CAMERA_PORT
# - Hand feedback: robot WUJI_HAND_STATUS_BIND/WUJI_HAND_STATUS_PORT -> PC ROBOT_HAND_HOST:ROBOT_HAND_STATUS_PORT
# - Hand commands: robot subscribes to PC_ZMQ_HOST:5556 from pico_manager
#
# Deploy launcher split:
# - Data collection uses tracker/sonic/gear_sonic_deploy/scripts/run_deploy_pc.sh
# - VLA/WAM bridge inference uses bridge/sonic/scripts/run_deploy_v4_pc.sh
#   Do not swap these two launchers.

# 1. Robot terminal 1: head servo hold + RealSense camera (verify the saved head encoder pose)
cd ~/WB-WAM # Use the repository path on the robot if different.
CAPTURE_FPS=${CAPTURE_FPS} CAMERA_FPS=${CAMERA_FPS} collector/sonic/scripts/run_camera_server.sh --head-motion-approved

# 2. Robot terminal 2: Wuji hand feedback/control service
cd ~/WB-WAM
collector/sonic/scripts/run_wuji_hand_server.sh

# 3. PC terminal 1: SONIC deploy
cd "$REPO_ROOT"
tracker/sonic/gear_sonic_deploy/scripts/run_deploy_pc.sh

# 4. PC terminal 2: PICO manager / teleop stream
cd "$REPO_ROOT"
collector/sonic/scripts/run_pico_manager.sh

# 5. PC terminal 3: dataset collector
cd "$REPO_ROOT"
CAPTURE_FPS=${CAPTURE_FPS} CAPTURE_MODE=${CAPTURE_MODE} CAMERA_FPS=${CAMERA_FPS} collector/sonic/scripts/run_collector_pc.sh

# Runtime controls:
# - PICO left controller: X+Y starts/stops policy; left-stick click toggles PLANNER <-> POSE.
# - PICO right controller: right-stick click starts an episode, A saves, B discards.
# - PC keyboard remains available: pico_manager 4/1/2/3/5, collector s/q/d/exit.
FLOW
}

case "$ACTION" in
    print)
        print_flow
        ;;
    camera)
        echo "Starting camera server with CAMERA_FPS=${CAMERA_FPS}"
        exec "$SCRIPT_DIR/run_camera_server.sh" "${HEAD_ARGS[@]}"
        ;;
    head-probe)
        exec "$SCRIPT_DIR/probe_head_servo.sh"
        ;;
    hand)
        echo "Starting Wuji hand server"
        exec "$SCRIPT_DIR/run_wuji_hand_server.sh"
        ;;
    deploy)
        echo "Starting SONIC deploy"
        exec "$REPO_ROOT/tracker/sonic/gear_sonic_deploy/scripts/run_deploy_pc.sh"
        ;;
    pico)
        echo "Starting PICO manager"
        exec "$SCRIPT_DIR/run_pico_manager.sh"
        ;;
    collector)
        echo "Starting collector with CAPTURE_FPS=${CAPTURE_FPS}, CAPTURE_MODE=${CAPTURE_MODE}, CAMERA_FPS=${CAMERA_FPS}"
        exec "$SCRIPT_DIR/run_collector_pc.sh"
        ;;
esac
