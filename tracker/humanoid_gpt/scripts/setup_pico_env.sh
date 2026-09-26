#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PYTHON_BIN="${HUMANOID_GPT_PYTHON:-python3}"
GMR_COMMIT="bb1bbe40774794fceb2a7c579a3464a28e68c844"
GMR_ROOT="${PROJECT_ROOT}/.deps/GMR"
XRT_SOURCE="${REPO_ROOT}/third_party/xrobotoolkit_sdk"
XRT_ROOT="${PROJECT_ROOT}/.deps/xrobotoolkit_sdk"
WUJI_SOURCE="${REPO_ROOT}/third_party/wuji_retargeting"
WUJI_ROOT="${PROJECT_ROOT}/.deps/wuji_retargeting"
MANUS_SOURCE="${REPO_ROOT}/third_party/MANUS_SDK/ManusSDK_v3.1.1/SDKClient_Linux"
MANUS_ROOT="${PROJECT_ROOT}/.deps/manus_sdk"
WHEEL_ROOT="${PROJECT_ROOT}/.deps/wheels"
CHECKPOINT_SOURCE="${HUMANOID_GPT_CHECKPOINT_SOURCE:-${PROJECT_ROOT}/storage/ckpts}"
SKIP_CHECKPOINTS=0
WITH_MANUS=0

usage() {
  printf '%s\n' \
    "Usage: scripts/setup_pico_env.sh [options]" \
    "" \
    "Options:" \
    "  --python PATH              Python 3.11/3.12 interpreter (default: python3)" \
    "  --checkpoint-source PATH   Existing HumanoidGPT checkpoint directory" \
    "  --skip-checkpoints         Do not copy/check ONNX checkpoints" \
    "  --with-manus              Build MANUS Python binding inside .deps" \
    "  -h, --help                 Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      if [[ $# -lt 2 || -z "${2:-}" || "${2}" == -* ]]; then
        printf 'Missing value for --python.\n' >&2
        usage >&2
        exit 2
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    --checkpoint-source)
      if [[ $# -lt 2 || -z "${2:-}" || "${2}" == -* ]]; then
        printf 'Missing value for --checkpoint-source.\n' >&2
        usage >&2
        exit 2
      fi
      CHECKPOINT_SOURCE="$2"
      shift 2
      ;;
    --with-manus)
      WITH_MANUS=1
      shift
      ;;
    --skip-checkpoints)
      SKIP_CHECKPOINTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

"${PYTHON_BIN}" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), sys.version'

mkdir -p "${PROJECT_ROOT}/.deps" "${WHEEL_ROOT}" "${PROJECT_ROOT}/storage/ckpts"

printf '[setup] Installing this Humanoid-GPT checkout (editable)\n'
"${PYTHON_BIN}" -m pip uninstall -y onnxruntime-gpu
"${PYTHON_BIN}" -m pip install --no-build-isolation -e "${PROJECT_ROOT}[pico,dev,cpu]"

if [[ ! -e "${GMR_ROOT}" ]]; then
  printf '[setup] Cloning GMR into %s\n' "${GMR_ROOT}"
  git clone https://github.com/YanjieZe/GMR.git "${GMR_ROOT}"
elif [[ ! -d "${GMR_ROOT}/.git" ]]; then
  printf 'GMR path exists but is not a git checkout: %s\n' "${GMR_ROOT}" >&2
  exit 1
fi

if ! git -C "${GMR_ROOT}" cat-file -e "${GMR_COMMIT}^{commit}" 2>/dev/null; then
  printf '[setup] Fetching pinned GMR commit %s\n' "${GMR_COMMIT}"
  git -C "${GMR_ROOT}" fetch origin "${GMR_COMMIT}"
fi
git -C "${GMR_ROOT}" checkout --detach "${GMR_COMMIT}"
"${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps -e "${GMR_ROOT}"

if [[ ! -f "${XRT_SOURCE}/setup.py" ]]; then
  printf 'XRoboToolkit SDK source is missing: %s\n' "${XRT_SOURCE}" >&2
  exit 1
fi
mkdir -p "${XRT_ROOT}"
cp -a "${XRT_SOURCE}/." "${XRT_ROOT}/"

ARCH="$(uname -m)"
if [[ "${ARCH}" == "aarch64" ]]; then
  XRT_LIB_DIR="${XRT_ROOT}/lib/aarch64"
else
  XRT_LIB_DIR="${XRT_ROOT}/lib"
fi
if [[ ! -f "${XRT_LIB_DIR}/libPXREARobotSDK.so" ]]; then
  printf 'XRoboToolkit native library is missing: %s\n' "${XRT_LIB_DIR}" >&2
  exit 1
fi

PYBIND11_CMAKE_DIR="$("${PYTHON_BIN}" -m pybind11 --cmakedir)"
export CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${XRT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
PYTHON_TAG="$("${PYTHON_BIN}" -c \
  'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
# Do not let a wheel from another Python ABI win the old lexical tail selection.
find "${WHEEL_ROOT}" -maxdepth 1 -type f -name 'xrobotoolkit_sdk-*.whl' -delete
printf '[setup] Building the Python %s XRoboToolkit wheel\n' "$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
"${PYTHON_BIN}" -m pip wheel \
  --no-deps \
  --no-build-isolation \
  --wheel-dir "${WHEEL_ROOT}" \
  "${XRT_ROOT}"

XRT_WHEEL="$(find "${WHEEL_ROOT}" -maxdepth 1 -type f \
  -name "xrobotoolkit_sdk-*-${PYTHON_TAG}-${PYTHON_TAG}-*.whl" \
  -print -quit)"
if [[ -z "${XRT_WHEEL}" ]]; then
  printf 'XRoboToolkit %s wheel was not produced in %s\n' \
    "${PYTHON_TAG}" "${WHEEL_ROOT}" >&2
  exit 1
fi
"${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "${XRT_WHEEL}"

if [[ ! -f "${WUJI_SOURCE}/pyproject.toml" ]]; then
  printf "Wuji retargeting source is missing: %s\n" "${WUJI_SOURCE}" >&2
  exit 1
fi
mkdir -p "${WUJI_ROOT}"
cp -a "${WUJI_SOURCE}/." "${WUJI_ROOT}/"

if [[ "${WITH_MANUS}" -eq 1 ]]; then
  if [[ ! -f "${MANUS_SOURCE}/Makefile" ]]; then
    printf "MANUS SDK source is missing: %s\n" "${MANUS_SOURCE}" >&2
    exit 1
  fi
  mkdir -p "${MANUS_ROOT}"
  cp -a "${MANUS_SOURCE}/." "${MANUS_ROOT}/"
  find "${MANUS_ROOT}/objects" -type f -delete 2>/dev/null || true
  find "${MANUS_ROOT}" -maxdepth 1 -type f -name "ManusServer*.so" -delete
  make -C "${MANUS_ROOT}" PYTHON="${PYTHON_BIN}"
fi

if [[ "${SKIP_CHECKPOINTS}" -eq 0 ]]; then
  TRACK_SOURCE="${CHECKPOINT_SOURCE}/pns_wo_priv216.onnx"
  WALK_SOURCE="${CHECKPOINT_SOURCE}/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx"
  if [[ ! -f "${TRACK_SOURCE}" || ! -f "${WALK_SOURCE}" ]]; then
    printf 'Checkpoints not found under %s; use --checkpoint-source or --skip-checkpoints\n' "${CHECKPOINT_SOURCE}" >&2
    exit 1
  fi
  mkdir -p "${PROJECT_ROOT}/storage/ckpts/G1-Walk"
  TRACK_TARGET="${PROJECT_ROOT}/storage/ckpts/pns_wo_priv216.onnx"
  WALK_TARGET="${PROJECT_ROOT}/storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx"
  if [[ ! "${TRACK_SOURCE}" -ef "${TRACK_TARGET}" ]]; then
    cp -p "${TRACK_SOURCE}" "${TRACK_TARGET}"
  fi
  if [[ ! "${WALK_SOURCE}" -ef "${WALK_TARGET}" ]]; then
    cp -p "${WALK_SOURCE}" "${WALK_TARGET}"
  fi
fi

"${PYTHON_BIN}" -m pip check

CHECK_ARGS=(--service-mode auto)
if [[ "${WITH_MANUS}" -eq 1 ]]; then
  CHECK_ARGS+=(--require-manus)
fi
if [[ "${SKIP_CHECKPOINTS}" -eq 1 ]]; then
  CHECK_ARGS+=(--skip-checkpoints)
fi
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/check_pico_env.py" "${CHECK_ARGS[@]}"

printf '[setup] Pico/GMR HumanoidGPT environment is ready.\n'
