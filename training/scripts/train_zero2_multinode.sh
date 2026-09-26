#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/setup_video_runtime.sh
source "${SCRIPT_DIR}/setup_video_runtime.sh"
cd -- "${WBWAM_TRAINING_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_zero2_multinode.sh [hydra_overrides...]

Env mapping (same style as torchrun example):
  MLP_WORKER_GPU      -> nproc_per_node (used by default; no positional arg needed)
  MLP_WORKER_NUM      -> nnodes
  MLP_ROLE_INDEX      -> node_rank
  MLP_WORKER_0_HOST   -> master_addr
  MLP_WORKER_0_PORT   -> master_port

Alternative cluster env mapping:
  NPROC_PER_NODE      -> nproc_per_node
  WORLD_SIZE          -> nnodes
  RANK                -> node_rank
  MASTER_ADDR         -> master_addr
  MASTER_PORT         -> master_port

Optional env:
  RUN_ID              -> force same run id across all nodes
  RUN_ID_SYNC_PORT    -> TCPStore port used only for RUN_ID sync (default: MASTER_PORT + 11)
  RUN_ID_SYNC_TIMEOUT -> RUN_ID sync timeout seconds (default: 180)

Examples:
  # Cluster env already provides MLP_* vars:
  bash scripts/train_zero2_multinode.sh wb_task=midtrain

  # Hydra overrides only:
  RUN_ID=exp123 bash scripts/train_zero2_multinode.sh wb_task=midtrain
EOF
}

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

EXTRA_ARGS=("$@")
NPROC_PER_NODE="${MLP_WORKER_GPU:-${NPROC_PER_NODE:-}}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  echo "Error: missing nproc_per_node. Please set MLP_WORKER_GPU (or NPROC_PER_NODE)." >&2
  usage
  exit 1
fi

NUM_MACHINES="${MLP_WORKER_NUM:-${NNODES:-${WORLD_SIZE:-1}}}"
MACHINE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-${RANK:-0}}}"
MAIN_PROCESS_IP="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
MAIN_PROCESS_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29500}}"

if ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == wb_task/* ]]; then
    local name="${cfg##*/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    wb_task=*)
      cfg="${arg#wb_task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg##*/}"
      ;;
  esac
done

TOTAL_PROCESSES=$((NPROC_PER_NODE * NUM_MACHINES))
DEEPSPEED_MULTINODE_ARGS=()
if (( NUM_MACHINES > 1 )); then
  DEEPSPEED_MULTINODE_ARGS=(--deepspeed_multinode_launcher standard)
fi

export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${NPROC_PER_NODE}}"
export TORCH_DIST_TIMEOUT_SECONDS="${TORCH_DIST_TIMEOUT_SECONDS:-7200}"
export RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-3600}"
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-7200}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID=$(
      python - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} total_processes=${TOTAL_PROCESSES}"
echo "[launch] main_process_ip=${MAIN_PROCESS_IP} main_process_port=${MAIN_PROCESS_PORT} run_id=${RUN_ID}"
echo "[launch] timeout_env TORCH_DIST_TIMEOUT_SECONDS=${TORCH_DIST_TIMEOUT_SECONDS} RUN_ID_SYNC_TIMEOUT=${RUN_ID_SYNC_TIMEOUT} DEEPSPEED_TIMEOUT=${DEEPSPEED_TIMEOUT} TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING} TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}"

WBWAM_RUNS_ROOT="${WBWAM_RUNS_ROOT:-./runs}"

set +e
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  "${DEEPSPEED_MULTINODE_ARGS[@]}" \
  --num_processes "${TOTAL_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  scripts/train.py \
  "output_dir=${WBWAM_RUNS_ROOT}/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
LAUNCH_STATUS=$?
set -e

exit "${LAUNCH_STATUS}"
