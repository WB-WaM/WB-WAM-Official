#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
COLLECTION_ENV="${REPO_ROOT}/collector/humanoid_gpt/scripts/hgpt_collection.env"
GMR_ROOT="${PROJECT_ROOT}/.deps/GMR"
XRT_ROOT="${PROJECT_ROOT}/.deps/xrobotoolkit_sdk"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-${PROJECT_ROOT}/.deps/cyclonedds/install}"

# shellcheck disable=SC1090
source "${COLLECTION_ENV}"

NET_IFACE="${HGPT_NET}"
SERVICE_MODE="${HGPT_PICO_SERVICE_MODE}"
POLICY_PROVIDER="${HGPT_POLICY_PROVIDER}"
PUBLISH_LOWCMD=0
POSE_ENDPOINT="${HGPT_POSE_ENDPOINT}"
STATE_ACTION_BIND="${HGPT_STATE_ACTION_ENDPOINT}"
TRACK_DIR="storage/test"
NO_MOCAP=0

missing_value() {
  printf 'Missing value for %s.\n' "$1" >&2
  usage >&2
  exit 2
}

usage() {
  printf '%s\n' \
    "Usage: scripts/run_pico_real.sh [wrapper options]" \
    "" \
    "The default is --debug: subscribe to LowState and run inference without" \
    "publishing LowCmd. Motor command publishing requires --publish-lowcmd." \
    "" \
    "Wrapper options:" \
    "  --net NIC                     G1-facing Ethernet interface (default: eno1)" \
    "  --pico-service-mode MODE      auto or external (default: external)" \
    "  --policy-provider PROVIDER    cpu or tensorrt (default: tensorrt)" \
    "  --pose-endpoint ENDPOINT      Unified pose manager endpoint (default: tcp://127.0.0.1:5556)" \
    "  --state-action-bind ENDPOINT  Collector telemetry bind (default: tcp://127.0.0.1:5558)" \
    "  --track-dir PATH              Offline replay NPZ file/directory" \
    "  --no-mocap                    Disable Pico input for offline-only replay" \
    "  --publish-lowcmd              CONFIRM real LowCmd publication" \
    "  -h, --help                    Show this help" \
    "" \
    "All deploy.play_track options are locked by this safety wrapper."
}

while [[ $# -gt 0 ]]; do
  RAW_ARG="$1"
  OPTION_NAME="${RAW_ARG%%=*}"
  # Tyro accepts snake_case aliases for kebab-case options. Normalize before
  # deciding whether an option belongs to this safety wrapper.
  OPTION_NAME="${OPTION_NAME//_/-}"
  case "${OPTION_NAME}" in
    --net)
      if [[ "${RAW_ARG}" == *=* ]]; then
        NET_IFACE="${RAW_ARG#*=}"
        [[ -n "${NET_IFACE}" ]] || missing_value "${OPTION_NAME}"
        shift
      else
        [[ $# -ge 2 && -n "${2:-}" && "${2}" != -* ]] || \
          missing_value "${OPTION_NAME}"
        NET_IFACE="$2"
        shift 2
      fi
      ;;
    --pico-service-mode)
      if [[ "${RAW_ARG}" == *=* ]]; then
        SERVICE_MODE="${RAW_ARG#*=}"
        [[ -n "${SERVICE_MODE}" ]] || missing_value "${OPTION_NAME}"
        shift
      else
        [[ $# -ge 2 && -n "${2:-}" && "${2}" != -* ]] || \
          missing_value "${OPTION_NAME}"
        SERVICE_MODE="$2"
        shift 2
      fi
      ;;
    --policy-provider)
      if [[ "${RAW_ARG}" == *=* ]]; then
        POLICY_PROVIDER="${RAW_ARG#*=}"
        [[ -n "${POLICY_PROVIDER}" ]] || missing_value "${OPTION_NAME}"
        shift
      else
        [[ $# -ge 2 && -n "${2:-}" && "${2}" != -* ]] || \
          missing_value "${OPTION_NAME}"
        POLICY_PROVIDER="$2"
        shift 2
      fi
      ;;
    --pose-endpoint)
      if [[ "${RAW_ARG}" == *=* ]]; then
        POSE_ENDPOINT="${RAW_ARG#*=}"
        [[ -n "${POSE_ENDPOINT}" ]] || missing_value "${OPTION_NAME}"
        shift
      else
        [[ $# -ge 2 && -n "${2:-}" && "${2}" != -* ]] || missing_value "${OPTION_NAME}"
        POSE_ENDPOINT="$2"
        shift 2
      fi
      ;;
    --state-action-bind)
      if [[ "${RAW_ARG}" == *=* ]]; then
        STATE_ACTION_BIND="${RAW_ARG#*=}"
        [[ -n "${STATE_ACTION_BIND}" ]] || missing_value "${OPTION_NAME}"
        shift
      else
        [[ $# -ge 2 && -n "${2:-}" && "${2}" != -* ]] || missing_value "${OPTION_NAME}"
        STATE_ACTION_BIND="$2"
        shift 2
      fi
      ;;
    --track-dir)
      if [[ "${RAW_ARG}" == *=* ]]; then
        TRACK_DIR="${RAW_ARG#*=}"
        [[ -n "${TRACK_DIR}" ]] || missing_value "${OPTION_NAME}"
        shift
      else
        [[ $# -ge 2 && -n "${2:-}" && "${2}" != -* ]] || missing_value "${OPTION_NAME}"
        TRACK_DIR="$2"
        shift 2
      fi
      ;;
    --no-mocap)
      if [[ "${RAW_ARG}" == *=* ]]; then
        printf '%s does not accept a value.\n' "${OPTION_NAME}" >&2
        exit 2
      fi
      NO_MOCAP=1
      shift
      ;;
    --publish-lowcmd)
      if [[ "${RAW_ARG}" == *=* ]]; then
        printf '%s does not accept a value.\n' "${OPTION_NAME}" >&2
        exit 2
      fi
      PUBLISH_LOWCMD=1
      shift
      ;;
    -h|--help)
      if [[ "${RAW_ARG}" == *=* ]]; then
        printf '%s does not accept a value.\n' "${OPTION_NAME}" >&2
        exit 2
      fi
      usage
      exit 0
      ;;
    *)
      printf 'Option %s is locked or unsupported by this safety wrapper.\n' \
        "${RAW_ARG}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${NET_IFACE}" ]]; then
  printf 'Missing --net <robot_nic>; HGPT_NET is empty.\n' >&2
  usage >&2
  exit 2
fi
if [[ "${SERVICE_MODE}" != "auto" && "${SERVICE_MODE}" != "external" ]]; then
  printf 'Invalid --pico-service-mode: %s\n' "${SERVICE_MODE}" >&2
  exit 2
fi
if [[ "${POLICY_PROVIDER}" != "cpu" && "${POLICY_PROVIDER}" != "tensorrt" ]]; then
  printf 'Invalid --policy-provider: %s\n' "${POLICY_PROVIDER}" >&2
  exit 2
fi
if [[ "$(uname -m)" == "aarch64" ]]; then
  XRT_LIB_DIR="${XRT_ROOT}/lib/aarch64"
else
  XRT_LIB_DIR="${XRT_ROOT}/lib"
fi

export CYCLONEDDS_HOME
export CMAKE_PREFIX_PATH="${CYCLONEDDS_HOME}:${CMAKE_PREFIX_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${GMR_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${XRT_LIB_DIR}:${CYCLONEDDS_HOME}/lib:/opt/apps/roboticsservice/lib:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="/opt/apps/roboticsservice/plugins:${QT_PLUGIN_PATH:-}"
export QML2_IMPORT_PATH="/opt/apps/roboticsservice/qml:${QML2_IMPORT_PATH:-}"
export PYTHONUNBUFFERED=1

# XRobo uses localhost gRPC. Proxying it both breaks discovery and adds latency.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

if [[ -n "${HUMANOID_GPT_PYTHON:-}" ]]; then
  PYTHON_CMD=("${HUMANOID_GPT_PYTHON}")
elif [[ "${CONDA_DEFAULT_ENV:-}" == "h-gpt" ]]; then
  PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n h-gpt python)
else
  PYTHON_CMD=(python3)
fi

# Make pip-installed NVIDIA/TensorRT shared libraries visible to ORT.
while IFS= read -r runtime_lib; do
  if [[ -n "${runtime_lib}" ]]; then
    export LD_LIBRARY_PATH="${runtime_lib}:${LD_LIBRARY_PATH}"
  fi
done < <("${PYTHON_CMD[@]}" -c '
import site
from pathlib import Path
for base in map(Path, site.getsitepackages()):
    for pattern in ("tensorrt_libs", "nvidia/*/lib"):
        for path in sorted(base.glob(pattern)):
            if path.is_dir():
                print(path)
')

# Humanoid-GPT storage paths are relative to its project root.
# Enter it before both preflight and deployment when invoked from WB-WAM.
cd "${PROJECT_ROOT}"

CHECK_ARGS=(--net "${NET_IFACE}" --require-link --policy-provider "${POLICY_PROVIDER}")
if [[ "${PUBLISH_LOWCMD}" -eq 1 ]]; then
  CHECK_ARGS+=(--benchmark)
fi
"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/check_pico_real_env.py" \
  "${CHECK_ARGS[@]}"


printf '%s\n' \
  '[Safety] Suspend the G1 for initial tests and keep the remote in hand.' \
  '[Safety] L2+R2: robot debug state; publish mode then releases the MotionSwitcher owner.' \
  '[Safety] Start: default pose; A: control loop.' \
  '[Safety] Select: operator-requested emergency damping.'
SAFETY_ARGS=(--debug)
if [[ "${PUBLISH_LOWCMD}" -eq 1 ]]; then
  SAFETY_ARGS=()
  SAFETY_ARGS+=(--allow-real-publish)
  printf '%s\n' \
    '*** REAL LOWCMD PUBLICATION ENABLED ***' \
    'Suspend the G1, keep the remote in hand, and use SELECT for emergency stop.'
else
  printf '%s\n' \
    '[Real] DEBUG ONLY: LowState is consumed, but LowCmd is not published.' \
    '[Real] Re-run with --publish-lowcmd only after the debug preflight passes.'
fi

POSE_ARGS=()
if [[ -n "${POSE_ENDPOINT}" ]]; then
  POSE_ARGS+=(--pose-endpoint "${POSE_ENDPOINT}")
fi

MOCAP_ARGS=(--mocap-type pico --pico-service-mode "${SERVICE_MODE}")
if [[ "${NO_MOCAP}" -eq 1 ]]; then
  MOCAP_ARGS=(--no-mocap)
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_CMD[@]}" -m deploy.play_track \
  --real \
  --net "${NET_IFACE}" \
  --freq 50 \
  --onnx-walk storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx \
  --onnx-track storage/ckpts/pns_wo_priv216.onnx \
  --policy-type mlp \
  --track-dir "${TRACK_DIR}" \
  "${MOCAP_ARGS[@]}" \
  "${POSE_ARGS[@]}" \
  --buffer-ms 0 \
  --real-policy-provider "${POLICY_PROVIDER}" \
  --state-action-bind "${STATE_ACTION_BIND}" \
  --no-visualize-retarget \
  --real-low-state-startup-timeout-s 5 \
  --real-control-join-timeout-s 1 \
  --real-damping-duration-s 0.5 \
  "${SAFETY_ARGS[@]}"
