#!/usr/bin/env bash
# PC entrypoint: copy only the head bundle, build/install remotely, reconnect, probe.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_SSH="${1:?Usage: setup_head_servo.sh USER@ROBOT_IP [robot-relative-repo-path]}"
ROBOT_REPO="${2:-WB-WAM}"
# Keep remote shell quoting unambiguous. Use SSH config for custom ports/options.
[[ "$ROBOT_SSH" =~ ^[A-Za-z0-9_.@:-]+$ && "$ROBOT_SSH" != -* ]] || { echo 'Invalid SSH target' >&2; exit 2; }
[[ "$ROBOT_REPO" =~ ^[A-Za-z0-9_./-]+$ && "$ROBOT_REPO" != /* && "$ROBOT_REPO" != -* && "/$ROBOT_REPO/" != *'/../'* ]] || { echo 'Use a simple repo path relative to the robot home' >&2; exit 2; }
command -v rsync >/dev/null || { echo 'Install rsync on the PC and robot first' >&2; exit 1; }
ssh "$ROBOT_SSH" "mkdir -p '$ROBOT_REPO/collector/sonic/head' '$ROBOT_REPO/collector/sonic/scripts'"
rsync -a --exclude=build/ --exclude=__pycache__/ "$SCRIPT_DIR/../head/" "$ROBOT_SSH:$ROBOT_REPO/collector/sonic/head/"
rsync -a "$SCRIPT_DIR/"{head_servo.env.example,load_launcher_config.sh,probe_head_servo.sh,run_head_servo.sh,run_camera_server.sh} "$ROBOT_SSH:$ROBOT_REPO/collector/sonic/scripts/"
ssh -tt "$ROBOT_SSH" "cd '$ROBOT_REPO' && cp -n collector/sonic/scripts/head_servo.env.example collector/sonic/scripts/head_servo.env && bash collector/sonic/head/setup_robot.sh --install"
# A fresh login picks up dialout membership; do not reuse a multiplexed connection.
ssh -o ControlPath=none "$ROBOT_SSH" "cd '$ROBOT_REPO' && bash collector/sonic/scripts/probe_head_servo.sh"
