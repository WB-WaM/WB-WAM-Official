#!/usr/bin/env bash
set -euo pipefail

: "${HA_RUNTIME_ROOT:?}"
: "${HA_ARENA_ROOT:?}"
: "${HA_ARENA_ASSETS:?}"
: "${HA_DEPENDENCIES_ROOT:?}"
: "${HA_ISAAC_IMAGE:?}"
: "${HA_ISAAC_CONTAINER_MODE:?}"
: "${HA_APPTAINER:?}"
: "${HA_ISAAC_SITE:?}"
: "${HA_ISAAC_HOME:?}"
: "${HA_ISAAC_CACHE:?}"
: "${HA_SONIC_RELEASE:?}"

isaaclab_root="$HA_DEPENDENCIES_ROOT/IsaacLab"
python_paths="$HA_ARENA_ROOT/isaaclab_twist2_g1:$isaaclab_root/source/isaaclab:$isaaclab_root/source/isaaclab_assets:$isaaclab_root/source/isaaclab_tasks:$HA_ISAAC_SITE:$HA_ISAAC_SITE/rerun_sdk:$HA_DEPENDENCIES_ROOT/unitree_sdk2_python:$HA_DEPENDENCIES_ROOT/GMR"

extra_binds=()
extra_binds+=(--bind "$HA_ARENA_ROOT:$HA_ARENA_ROOT:ro")
extra_binds+=(--bind "$HA_ARENA_ASSETS:$HA_ARENA_ROOT/isaaclab_twist2_g1/assets:ro")
library_paths=()
runtime_is_bound=0
if [[ -n "${HA_CONTAINER_BIND_ROOTS:-}" ]]; then
  IFS=: read -r -a container_bind_roots <<< "$HA_CONTAINER_BIND_ROOTS"
  for root in "${container_bind_roots[@]}"; do
    if [[ -n "$root" ]]; then
      extra_binds+=(--bind "$root:$root")
      if [[ "$HA_RUNTIME_ROOT" == "$root" || "$HA_RUNTIME_ROOT" == "$root/"* ]]; then
        runtime_is_bound=1
      fi
    fi
  done
fi
if ((runtime_is_bound == 0)); then
  extra_binds+=(--bind "$HA_RUNTIME_ROOT:$HA_RUNTIME_ROOT")
fi
if [[ -n "${HA_PALIGEMMA_ROOT:-}" ]]; then
  extra_binds+=(--bind "$HA_PALIGEMMA_ROOT:/ai/Yichi/taowen/ckpts/checkpoints/paligemma-3b-pt-224")
fi
if [[ -n "${HA_CUDA_ROOT:-}" ]]; then
  extra_binds+=(--bind "$HA_CUDA_ROOT:$HA_CUDA_ROOT")
  library_paths+=("$HA_CUDA_ROOT/lib64")
fi
if [[ -n "${HA_CUDNN_ROOT:-}" ]]; then
  extra_binds+=(--bind "$HA_CUDNN_ROOT:$HA_CUDNN_ROOT")
  library_paths+=("$HA_CUDNN_ROOT/lib")
fi
ld_library_path=/.singularity.d/libs
if ((${#library_paths[@]})); then
  ld_library_path="$(IFS=:; echo "${library_paths[*]}"):$ld_library_path"
fi
if [[ -n "${HA_CONTAINER_LD_LIBRARY_PATH:-}" ]]; then
  ld_library_path="$HA_CONTAINER_LD_LIBRARY_PATH:$ld_library_path"
fi

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE
container_flags=(--nv)
case "$HA_ISAAC_CONTAINER_MODE" in
  sif) container_flags+=(--fakeroot --writable-tmpfs) ;;
  sandbox) container_flags+=(--cleanenv --userns --writable) ;;
  *) echo "HA_ISAAC_CONTAINER_MODE must be sif or sandbox" >&2; exit 2 ;;
esac
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HA_RUNTIME_ROOT/isaacsim_runtime/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$HA_RUNTIME_ROOT/isaacsim_runtime/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HA_ISAAC_HOME" "$HA_ISAAC_CACHE"

exec "$HA_APPTAINER" exec "${container_flags[@]}" \
  "${extra_binds[@]}" \
  --home "$HA_ISAAC_HOME" \
  --bind "$HA_ISAAC_CACHE:/isaac-sim/kit/cache" \
  --env "PYTHONPATH=$python_paths,PROJECT_ROOT=$HA_ARENA_ROOT/isaaclab_twist2_g1,GMR_ROOT=$HA_DEPENDENCIES_ROOT/GMR,PYTHONNOUSERSITE=1,OMNI_KIT_ACCEPT_EULA=YES,ACCEPT_EULA=Y,PRIVACY_CONSENT=NO,PYNPUT_BACKEND=dummy,SONIC_VLA_ACTION_FORMAT=semantic_v3,ROBOT_USD_OVERRIDE=$HA_ARENA_ROOT/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd,LEROBOT_VLA_RECORD_OUTPUTS=0,LD_LIBRARY_PATH=$ld_library_path" \
  "$HA_ISAAC_IMAGE" /isaac-sim/python.sh "$@"
