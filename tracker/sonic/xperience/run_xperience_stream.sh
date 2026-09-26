#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-${SCRIPT_DIR}/.env}"
DOWNLOADER="${SCRIPT_DIR}/xperience_hf_download.py"
PROCESSOR="${SCRIPT_DIR}/xperience_process_episode.py"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERROR] env file not found: ${ENV_FILE}" >&2
  exit 2
fi

set -a
source "${ENV_FILE}"
set +a

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_nonempty() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" ]]; then
    echo "[ERROR] ${name} must be set in ${ENV_FILE}" >&2
    exit 2
  fi
}

: "${RUN_SAMPLE:=false}"
: "${HF_REVISION:=main}"
: "${MIN_FREE_GB:=500}"
: "${MAX_EPISODES:=}"
: "${PROCESS_WORKERS:=1}"
: "${PREFETCH_EPISODES:=1}"
: "${QUEUE_WAIT_S:=5}"
: "${SLEEP_S:=0}"
: "${DELETE_RAW_AFTER_PARSE:=true}"
: "${OVERWRITE:=false}"
: "${RESIZE_VIDEOS:=true}"
: "${KEEP_INTERMEDIATE:=false}"
: "${PIPELINE_PYTHON:=python}"
: "${PIPELINE_PYTHONPATH:=}"
: "${REQUIRE_GMR:=true}"
: "${GMR_ROOT:=}"
: "${GMR_PYTHON:=}"
: "${SMPLX_FOLDER:=}"
: "${GMR_ROBOT:=unitree_g1}"
: "${REQUIRE_WUJI:=true}"
: "${WUJI_ROOT:=}"
: "${WUJI_PYTHON:=}"
: "${WUJI_LEFT_CONFIG:=}"
: "${WUJI_RIGHT_CONFIG:=}"
: "${ENCODER_MODEL:=}"

require_nonempty RAW_ROOT
require_nonempty PROCESSED_ROOT
: "${QUEUE_DIR:=${PROCESSED_ROOT}/../queue}"

mkdir -p "${RAW_ROOT}" "${PROCESSED_ROOT}" "${QUEUE_DIR}"
: "${LOG_DIR:=${PROCESSED_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

if [[ -n "${PIPELINE_PYTHONPATH}" ]]; then
  export PYTHONPATH="${PIPELINE_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/xperience_stream_${timestamp}.log"
exec > >(tee -a "${log_file}") 2>&1

process_workers_int=$((PROCESS_WORKERS))
prefetch_int=$((PREFETCH_EPISODES))
if (( process_workers_int < 1 )); then
  process_workers_int=1
fi
if (( prefetch_int < 0 )); then
  prefetch_int=0
fi
raw_capacity=$((process_workers_int + prefetch_int))
if (( raw_capacity < 1 )); then
  raw_capacity=1
fi

STOP_FILE="${QUEUE_DIR}/.xperience_stream_stop"
DOWNLOADER_STATUS="${QUEUE_DIR}/.xperience_downloader_${timestamp}.status"
rm -f "${STOP_FILE}" "${DOWNLOADER_STATUS}" "${QUEUE_DIR}"/*.json.processing.status
for lock in "${QUEUE_DIR}"/*.json.processing; do
  target="${lock%.processing}"
  echo "[INFO] recovering unfinished queue lock: ${lock} -> ${target}"
  mv "${lock}" "${target}"
done

downloader_args=(
  "${DOWNLOADER}"
  stream
  --raw-root "${RAW_ROOT}"
  --queue-dir "${QUEUE_DIR}"
  --processed-root "${PROCESSED_ROOT}"
  --revision "${HF_REVISION}"
  --min-free-gb "${MIN_FREE_GB}"
  --max-queued-episodes "${raw_capacity}"
  --queue-wait-s "${QUEUE_WAIT_S}"
  --sleep-s "${SLEEP_S}"
  --stop-file "${STOP_FILE}"
)

if is_true "${RUN_SAMPLE}"; then
  downloader_args+=(--sample)
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[WARN] HF_TOKEN is empty; this only works if the selected HF dataset is public for this machine."
fi
if [[ -n "${MAX_EPISODES}" ]]; then
  downloader_args+=(--max-episodes "${MAX_EPISODES}")
fi

process_common_args=(
  "${PROCESSOR}"
  --output-root "${PROCESSED_ROOT}"
  --min-free-gb "${MIN_FREE_GB}"
  --gmr-robot "${GMR_ROBOT}"
)

if is_true "${DELETE_RAW_AFTER_PARSE}"; then
  process_common_args+=(--delete-raw-after-parse)
fi
if is_true "${OVERWRITE}"; then
  process_common_args+=(--overwrite)
fi
if is_true "${RESIZE_VIDEOS}"; then
  process_common_args+=(--resize-videos)
else
  process_common_args+=(--no-resize-videos)
fi
if is_true "${KEEP_INTERMEDIATE}"; then
  process_common_args+=(--keep-intermediate)
fi
if is_true "${REQUIRE_GMR}"; then
  process_common_args+=(--require-gmr)
fi
if [[ -n "${GMR_ROOT}" ]]; then
  process_common_args+=(--gmr-root "${GMR_ROOT}")
fi
if [[ -n "${GMR_PYTHON}" ]]; then
  process_common_args+=(--gmr-python "${GMR_PYTHON}")
fi
if [[ -n "${SMPLX_FOLDER}" ]]; then
  process_common_args+=(--smplx-folder "${SMPLX_FOLDER}")
fi
if is_true "${REQUIRE_WUJI}"; then
  process_common_args+=(--require-wuji)
fi
if [[ -n "${WUJI_ROOT}" ]]; then
  process_common_args+=(--wuji-root "${WUJI_ROOT}")
fi
if [[ -n "${WUJI_PYTHON}" ]]; then
  process_common_args+=(--wuji-python "${WUJI_PYTHON}")
fi
if [[ -n "${WUJI_LEFT_CONFIG}" ]]; then
  process_common_args+=(--wuji-left-config "${WUJI_LEFT_CONFIG}")
fi
if [[ -n "${WUJI_RIGHT_CONFIG}" ]]; then
  process_common_args+=(--wuji-right-config "${WUJI_RIGHT_CONFIG}")
fi
if [[ -n "${ENCODER_MODEL}" ]]; then
  process_common_args+=(--encoder-model "${ENCODER_MODEL}")
fi

echo "[INFO] env: ${ENV_FILE}"
echo "[INFO] raw root: ${RAW_ROOT}"
echo "[INFO] processed root: ${PROCESSED_ROOT}"
echo "[INFO] queue dir: ${QUEUE_DIR}"
echo "[INFO] log: ${log_file}"
echo "[INFO] process workers: ${process_workers_int}"
echo "[INFO] prefetch episodes: ${prefetch_int}"
echo "[INFO] max raw/queue capacity: ${raw_capacity}"
echo "[INFO] downloader: ${PIPELINE_PYTHON} ${downloader_args[*]}"
echo "[INFO] processor: ${PIPELINE_PYTHON} ${process_common_args[*]} --manifest <queue_manifest>"

(
  set +e
  "${PIPELINE_PYTHON}" "${downloader_args[@]}"
  status=$?
  echo "${status}" > "${DOWNLOADER_STATUS}"
  exit "${status}"
) &
downloader_pid=$!

processor_pids=()
processor_status_files=()
manager_failed=0
downloader_status_read=0
downloader_status=0

read_downloader_status_if_done() {
  if (( downloader_status_read == 1 )); then
    return
  fi
  if [[ -f "${DOWNLOADER_STATUS}" ]]; then
    wait "${downloader_pid}" || true
    downloader_status="$(cat "${DOWNLOADER_STATUS}")"
    downloader_status_read=1
    if [[ "${downloader_status}" != "0" ]]; then
      manager_failed=1
      touch "${STOP_FILE}"
      echo "[ERROR] downloader exited with status ${downloader_status}"
    else
      echo "[INFO] downloader finished"
    fi
  fi
}

launch_processor() {
  local manifest="$1"
  local lock="${manifest}.processing"
  local status_file="${lock}.status"
  if [[ ! -f "${manifest}" ]]; then
    return
  fi
  if ! mv "${manifest}" "${lock}"; then
    return
  fi
  echo "[INFO] processing queue manifest: ${lock}"
  (
    set +e
    "${PIPELINE_PYTHON}" "${process_common_args[@]}" --manifest "${lock}"
    status=$?
    if [[ "${status}" == "0" ]]; then
      rm -f "${lock}"
    else
      failed="${manifest}.failed.${timestamp}"
      mv "${lock}" "${failed}" 2>/dev/null || true
      touch "${STOP_FILE}"
      echo "[ERROR] processor failed for ${lock}; moved manifest to ${failed}"
    fi
    echo "${status}" > "${status_file}"
    exit "${status}"
  ) &
  processor_pids+=("$!")
  processor_status_files+=("${status_file}")
}

prune_processors() {
  local new_pids=()
  local new_status_files=()
  local i
  for i in "${!processor_pids[@]}"; do
    local pid="${processor_pids[$i]}"
    local status_file="${processor_status_files[$i]}"
    if [[ -f "${status_file}" ]]; then
      wait "${pid}" || true
      local status
      status="$(cat "${status_file}")"
      rm -f "${status_file}"
      if [[ "${status}" != "0" ]]; then
        manager_failed=1
        touch "${STOP_FILE}"
      fi
    else
      new_pids+=("${pid}")
      new_status_files+=("${status_file}")
    fi
  done
  processor_pids=("${new_pids[@]}")
  processor_status_files=("${new_status_files[@]}")
}

while true; do
  read_downloader_status_if_done
  prune_processors

  if [[ ! -f "${STOP_FILE}" ]]; then
    slots=$((process_workers_int - ${#processor_pids[@]}))
    if (( slots > 0 )); then
      for manifest in "${QUEUE_DIR}"/*.json; do
        launch_processor "${manifest}"
        slots=$((slots - 1))
        if (( slots <= 0 )); then
          break
        fi
      done
    fi
  fi

  pending_manifests=("${QUEUE_DIR}"/*.json)
  if [[ -f "${STOP_FILE}" ]]; then
    if (( downloader_status_read == 1 && ${#processor_pids[@]} == 0 )); then
      break
    fi
  elif (( downloader_status_read == 1 && ${#processor_pids[@]} == 0 && ${#pending_manifests[@]} == 0 )); then
    break
  fi

  sleep 1
done

read_downloader_status_if_done
prune_processors

if (( manager_failed != 0 )); then
  echo "[STOP] stream manager failed; queue/raw data were left for resume."
  exit 1
fi

echo "[INFO] stream manager complete"
