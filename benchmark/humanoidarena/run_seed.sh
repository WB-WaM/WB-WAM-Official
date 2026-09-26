#!/usr/bin/env bash
# Internal worker: run all missing repeats for one task/seed in one simulator.
set -euo pipefail

required=(
  HA_BENCHMARK_ROOT HA_RUNTIME_ROOT HA_INFERENCE_PYTHON HA_ARENA_ROOT HA_ARENA_ASSETS
  HA_DEPENDENCIES_ROOT HA_WBWAM_ROOT
  HA_BASE_MODELS_ROOT HA_ISAAC_HOME HA_ISAAC_CACHE HA_SIMULATION_BACKEND
  HA_SONIC_RELEASE
  HA_INFERENCE_GPU HA_SIMULATION_GPU
  HA_MAX_PREEXISTING_GPU_MIB HA_TASK_NAME HA_GYM_TASK
  HA_ENV_CONFIG HA_INSTRUCTION_KEY HA_EXPECTED_PROMPT HA_CHECKPOINT HA_WEIGHT_FILE HA_OUTPUT HA_SEED
  HA_REPEATS HA_REPEAT_LIST HA_BATCH_TIMEOUT_SECONDS HA_MAX_STEPS HA_DENOISE_STEPS HA_REPLAN
)
case "${HA_SIMULATION_BACKEND:-}" in
  container) required+=(HA_ISAAC_IMAGE HA_ISAAC_SITE HA_ISAAC_CONTAINER_MODE HA_APPTAINER) ;;
  native) required+=(HA_SIMULATION_PYTHON) ;;
  *) echo "HA_SIMULATION_BACKEND must be container or native" >&2; exit 2 ;;
esac
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required environment variable: $name" >&2; exit 2; }
done
[[ "$HA_BATCH_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "HA_BATCH_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
}

benchmark_root=$HA_BENCHMARK_ROOT
if [[ "$HA_SIMULATION_BACKEND" == native ]]; then
  simulation_launcher="$benchmark_root/isaac_native_python.sh"
else
  simulation_launcher="$benchmark_root/isaac_python.sh"
fi
arena=$HA_ARENA_ROOT/isaaclab_twist2_g1
python=$HA_INFERENCE_PYTHON
output=$HA_OUTPUT
mkdir -p "$output"
[[ "$HA_MAX_PREEXISTING_GPU_MIB" =~ ^[0-9]+$ ]] || {
  echo "HA_MAX_PREEXISTING_GPU_MIB must be a non-negative integer" >&2
  exit 2
}
gpu_wait_seconds=${HA_GPU_WAIT_SECONDS:-180}
[[ "$gpu_wait_seconds" =~ ^[0-9]+$ ]] || {
  echo "HA_GPU_WAIT_SECONDS must be a non-negative integer" >&2
  exit 2
}

gpu_used_mib() {
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits \
    | awk -F, -v wanted="$1" '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)} $1 == wanted || $2 == wanted {print $3; exit}'
}
gpu_physical_index() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F, -v wanted="$1" '{gsub(/ /, "", $1); gsub(/ /, "", $2)} $1 == wanted || $2 == wanted {print $1; exit}'
}
for gpu in "$HA_INFERENCE_GPU" "$HA_SIMULATION_GPU"; do
  deadline=$((SECONDS + gpu_wait_seconds))
  while true; do
    used=$(gpu_used_mib "$gpu")
    [[ "$used" =~ ^[0-9]+$ ]] || { echo "cannot resolve GPU: $gpu" >&2; exit 2; }
    (( used < HA_MAX_PREEXISTING_GPU_MIB )) && break
    if (( SECONDS >= deadline )); then
      echo "GPU $gpu is still using ${used} MiB after waiting ${gpu_wait_seconds}s (limit ${HA_MAX_PREEXISTING_GPU_MIB} MiB)" >&2
      nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2 || true
      exit 2
    fi
    echo "Waiting for GPU $gpu memory: ${used} MiB used (limit ${HA_MAX_PREEXISTING_GPU_MIB} MiB)" >&2
    remaining=$((deadline - SECONDS))
    (( remaining > 5 )) && remaining=5
    sleep "$remaining"
  done
done
simulation_physical_index=$(gpu_physical_index "$HA_SIMULATION_GPU")
[[ "$simulation_physical_index" =~ ^[0-9]+$ ]] || {
  echo "cannot resolve physical index for simulation GPU: $HA_SIMULATION_GPU" >&2
  exit 2
}

export DIFFSYNTH_MODEL_BASE_PATH=$HA_BASE_MODELS_ROOT
export DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export SONIC_VLA_ACTION_FORMAT=semantic_v3
export ROBOT_USD_OVERRIDE="$arena/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
export LEROBOT_VLA_RECORD_OUTPUTS=0
export PYTHONPATH="$HA_WBWAM_ROOT/src"

label="${HA_TASK_NAME}_native50_semantic40_IDM_d${HA_DENOISE_STEPS}_r${HA_REPLAN}"
server_pid=
sim_pid=
cleanup() {
  status=$?
  trap - EXIT
  if [[ -n "$sim_pid" ]]; then
    kill -TERM -- "-$sim_pid" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 -- "-$sim_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$sim_pid" 2>/dev/null || true
    wait "$sim_pid" 2>/dev/null || true
  fi
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  "$python" "$benchmark_root/artifacts.py" summarize --root "$output" --repeats "$HA_REPEATS" || true
  exit "$status"
}
trap cleanup EXIT

"$python" "$benchmark_root/artifacts.py" prepare \
  --root "$output" --seed "$HA_SEED" --repeats "$HA_REPEATS" \
  --repeat-list "$HA_REPEAT_LIST" --task "$HA_GYM_TASK" --label "$label" \
  --max-steps "$HA_MAX_STEPS"

repo_root=$(cd "$benchmark_root/../.." && pwd)
if ! git -C "$repo_root" rev-parse HEAD >"$output/code_commit.txt" 2>/dev/null; then
  printf 'unknown (source archive or exported test tree)\n' >"$output/code_commit.txt"
fi
"$python" -m pip freeze >"$output/inference_packages.txt"
cat >"$output/run_config.json" <<JSON
{
  "task": "$HA_TASK_NAME",
  "seed": $HA_SEED,
  "persistent": true,
  "checkpoint": "$HA_CHECKPOINT",
  "max_steps": $HA_MAX_STEPS,
  "denoise_steps": $HA_DENOISE_STEPS,
  "replan": $HA_REPLAN,
  "inference_mode": "idm",
  "weight_file": "$HA_WEIGHT_FILE"
}
JSON

CUDA_VISIBLE_DEVICES="$HA_INFERENCE_GPU" "$python" -u "$benchmark_root/server.py" \
  --checkpoint-root "$HA_CHECKPOINT" --weight-file "$HA_WEIGHT_FILE" \
  --instruction-key "$HA_INSTRUCTION_KEY" --expected-prompt "$HA_EXPECTED_PROMPT" \
  --wbwam-root "$HA_WBWAM_ROOT" \
  --output "$output" --seed "$HA_SEED" --port 0 \
  --replan "$HA_REPLAN" --denoise-steps "$HA_DENOISE_STEPS" --inference-mode idm \
  >"$output/server.log" 2>&1 &
server_pid=$!

ready=0
for ((attempt=0; attempt<1200; attempt++)); do
  kill -0 "$server_pid" 2>/dev/null || { tail -n 80 "$output/server.log"; exit 1; }
  if grep -q "Native50 ready port=" "$output/server.log"; then
    port=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' "$output/server_health.json")
    curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >"$output/live_health.json"
    "$python" - "$output/live_health.json" "$HA_REPLAN" "$HA_DENOISE_STEPS" <<'PY'
import json
import sys

health = json.load(open(sys.argv[1]))
expected = {
    "execute_chunk_steps": int(sys.argv[2]),
    "num_inference_steps": int(sys.argv[3]),
    "inference_mode": "idm",
    "state_dim": 64,
    "action_dim": 40,
}
for key, value in expected.items():
    if health.get(key) != value:
        raise SystemExit(f"health mismatch {key}: {health.get(key)!r} != {value!r}")
PY
    ready=1
    break
  fi
  sleep 1
done
[[ "$ready" == 1 ]] || { echo "model server did not become ready" >&2; exit 1; }

cd "$arena"
export PROJECT_ROOT="$arena" PYTHONPATH="$arena"
# Keep Isaac Lab tensors on CPU, matching the validated Native50 protocol.
# Rendering and PhysX still use HA_SIMULATION_GPU.
setsid \
timeout --kill-after=60 "$HA_BATCH_TIMEOUT_SECONDS" \
  bash "$simulation_launcher" -u \
  script/eval_scripts/sonic_pi05/sim_eval_vla.py \
  --task "$HA_GYM_TASK" --env_config_yaml "$HA_ENV_CONFIG" \
  --seed "$HA_SEED" --episode_batch_json "$output/episode_batch.json" --max_steps "$HA_MAX_STEPS" \
  --sonic_encoder_path "$HA_SONIC_RELEASE/model_encoder.onnx" \
  --sonic_decoder_path "$HA_SONIC_RELEASE/model_decoder.onnx" \
  --model_path "$HA_SONIC_RELEASE/model_encoder.onnx" \
  --sonic_vla_root_rot6d_layout row --sonic_vla_root_max_delta_deg 26.0 \
  --lerobot_server_url "http://127.0.0.1:$port" --lerobot_server_timeout 600 \
  --robot_type unitree_g1_refpose_v3_1 \
  --recording_save_dir "$output/recordings" \
  --record_video_every_n 1 --step_log_every_n 100 \
  --device cpu --headless \
  --kit_args "--/renderer/activeGpu=$simulation_physical_index --/physics/cudaDevice=$simulation_physical_index --/renderer/multiGpu/enabled=false" \
  >"$output/sim.log" 2>&1 &
sim_pid=$!
set +e
wait "$sim_pid"
sim_status=$?
set -e
sim_pid=
(( sim_status == 0 )) || exit "$sim_status"

"$python" "$benchmark_root/artifacts.py" summarize \
  --root "$output" --repeats "$HA_REPEATS" --require-complete
