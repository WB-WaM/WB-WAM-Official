#!/usr/bin/env bash
# Launch the pinned HumanoidArena worker from the configured Isaac Sim 5.1 environment.
set -euo pipefail

: "${HA_SIMULATION_PYTHON:?}"
: "${HA_ARENA_ROOT:?}"
: "${HA_DEPENDENCIES_ROOT:?}"
: "${HA_ISAAC_HOME:?}"
: "${HA_ISAAC_CACHE:?}"

arena="$HA_ARENA_ROOT/isaaclab_twist2_g1"
isaaclab_root="$HA_DEPENDENCIES_ROOT/IsaacLab"
[[ -d "$arena/assets/objects" && -d "$arena/assets/robots" ]] || {
  echo "Restore HumanoidArena assets under $arena/assets before native evaluation" >&2
  exit 2
}

mkdir -p "$HA_ISAAC_HOME" "$HA_ISAAC_CACHE"
export HOME="$HA_ISAAC_HOME"
export XDG_CACHE_HOME="$HA_ISAAC_CACHE"
export PATH="$(dirname "$HA_SIMULATION_PYTHON"):$PATH"
export PYTHONPATH="$arena:$isaaclab_root/source/isaaclab:$isaaclab_root/source/isaaclab_assets:$isaaclab_root/source/isaaclab_tasks:$HA_DEPENDENCIES_ROOT/unitree_sdk2_python:$HA_DEPENDENCIES_ROOT/GMR:${PYTHONPATH:-}"
export PROJECT_ROOT="$arena" GMR_ROOT="$HA_DEPENDENCIES_ROOT/GMR"
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=NO
export PYNPUT_BACKEND=dummy SONIC_VLA_ACTION_FORMAT=semantic_v3 LEROBOT_VLA_RECORD_OUTPUTS=0
export ROBOT_USD_OVERRIDE="$arena/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"

exec "$HA_SIMULATION_PYTHON" "$@"
