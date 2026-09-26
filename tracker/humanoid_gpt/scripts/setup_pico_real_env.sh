#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PYTHON_BIN="${HUMANOID_GPT_PYTHON:-python3}"
CYCLONEDDS_REF="releases/0.10.x"
CYCLONEDDS_COMMIT="5041f3560c088c99e5088b2b8520b69169621196"
CYCLONEDDS_ROOT="${PROJECT_ROOT}/.deps/cyclonedds"
CYCLONEDDS_INSTALL="${CYCLONEDDS_ROOT}/install"
UNITREE_SOURCE="${REPO_ROOT}/tracker/sonic/external_dependencies/unitree_sdk2_python"
UNITREE_ROOT="${PROJECT_ROOT}/.deps/unitree_sdk2_python"
RUN_PICO_SETUP=1
INSTALL_TENSORRT=0
CHECKPOINT_SOURCE=""

usage() {
  printf '%s\n' \
    "Usage: scripts/setup_pico_real_env.sh [options]" \
    "" \
    "Install PC-side Pico + GMR + Unitree DDS + TensorRT dependencies." \
    "All dependency source/build trees are kept under ignored .deps/." \
    "" \
    "Options:" \
    "  --python PATH              Python 3.11/3.12 interpreter (default: python3)" \
    "  --unitree-source PATH      Existing unitree_sdk2_python source checkout" \
    "  --checkpoint-source PATH   Forward checkpoint source to setup_pico_env.sh" \
    "  --skip-pico-setup          Keep the existing Pico/GMR/base environment" \
    "  --with-tensorrt            Replace CPU ORT with ORT-GPU/TensorRT" \
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
    --unitree-source)
      if [[ $# -lt 2 || -z "${2:-}" || "${2}" == -* ]]; then
        printf 'Missing value for --unitree-source.\n' >&2
        usage >&2
        exit 2
      fi
      UNITREE_SOURCE="$2"
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
    --skip-pico-setup)
      RUN_PICO_SETUP=0
      shift
      ;;
    --with-tensorrt)
      INSTALL_TENSORRT=1
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

for command in git cmake; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    printf 'Required build command is missing: %s\n' "${command}" >&2
    exit 1
  fi
done

mkdir -p "${PROJECT_ROOT}/.deps"

if [[ "${RUN_PICO_SETUP}" -eq 1 ]]; then
  PICO_ARGS=(--python "${PYTHON_BIN}")
  if [[ -n "${CHECKPOINT_SOURCE}" ]]; then
    PICO_ARGS+=(--checkpoint-source "${CHECKPOINT_SOURCE}")
  fi
  "${PROJECT_ROOT}/scripts/setup_pico_env.sh" "${PICO_ARGS[@]}"
fi

if [[ ! -e "${CYCLONEDDS_ROOT}" ]]; then
  printf '[setup-real] Cloning CycloneDDS %s into %s\n' \
    "${CYCLONEDDS_REF}" "${CYCLONEDDS_ROOT}"
  git clone --branch "${CYCLONEDDS_REF}" --depth 1 \
    https://github.com/eclipse-cyclonedds/cyclonedds.git \
    "${CYCLONEDDS_ROOT}"
elif [[ ! -d "${CYCLONEDDS_ROOT}/.git" ]]; then
  printf 'CycloneDDS path exists but is not a git checkout: %s\n' \
    "${CYCLONEDDS_ROOT}" >&2
  exit 1
else
  printf '[setup-real] Reusing CycloneDDS checkout: %s\n' "${CYCLONEDDS_ROOT}"
fi

if ! git -C "${CYCLONEDDS_ROOT}" cat-file -e \
  "${CYCLONEDDS_COMMIT}^{commit}" 2>/dev/null; then
  printf '[setup-real] Fetching pinned CycloneDDS commit %s\n' \
    "${CYCLONEDDS_COMMIT}"
  git -C "${CYCLONEDDS_ROOT}" fetch --depth 1 origin "${CYCLONEDDS_COMMIT}"
fi
git -C "${CYCLONEDDS_ROOT}" checkout --detach "${CYCLONEDDS_COMMIT}"
ACTUAL_CYCLONEDDS_COMMIT="$(git -C "${CYCLONEDDS_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_CYCLONEDDS_COMMIT}" != "${CYCLONEDDS_COMMIT}" ]]; then
  printf 'CycloneDDS commit mismatch: expected %s, got %s\n' \
    "${CYCLONEDDS_COMMIT}" "${ACTUAL_CYCLONEDDS_COMMIT}" >&2
  exit 1
fi

printf '[setup-real] Building CycloneDDS into %s\n' "${CYCLONEDDS_INSTALL}"
cmake -S "${CYCLONEDDS_ROOT}" -B "${CYCLONEDDS_ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${CYCLONEDDS_INSTALL}" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF
cmake --build "${CYCLONEDDS_ROOT}/build" --target install --parallel

if [[ ! -f "${UNITREE_SOURCE}/setup.py" ]]; then
  printf 'Unitree SDK Python source is missing: %s\n' "${UNITREE_SOURCE}" >&2
  exit 1
fi
mkdir -p "${UNITREE_ROOT}"
cp -a "${UNITREE_SOURCE}/." "${UNITREE_ROOT}/"

export CYCLONEDDS_HOME="${CYCLONEDDS_INSTALL}"
export CMAKE_PREFIX_PATH="${CYCLONEDDS_INSTALL}:${CMAKE_PREFIX_PATH:-}"
if [[ "$(uname -m)" == "aarch64" ]]; then
  XRT_LIB_DIR="${PROJECT_ROOT}/.deps/xrobotoolkit_sdk/lib/aarch64"
else
  XRT_LIB_DIR="${PROJECT_ROOT}/.deps/xrobotoolkit_sdk/lib"
fi
export LD_LIBRARY_PATH="${XRT_LIB_DIR}:${CYCLONEDDS_INSTALL}/lib:/opt/apps/roboticsservice/lib:${LD_LIBRARY_PATH:-}"

printf '[setup-real] Installing CycloneDDS Python and local Unitree SDK\n'
"${PYTHON_BIN}" -m pip install "cyclonedds==0.10.2"
"${PYTHON_BIN}" -m pip install --no-deps -e "${UNITREE_ROOT}"

if [[ "${INSTALL_TENSORRT}" -eq 1 ]]; then
  printf '[setup-real] Replacing CPU ONNX Runtime with GPU/TensorRT runtime\n'
  "${PYTHON_BIN}" -m pip uninstall -y onnxruntime onnxruntime-gpu
  "${PYTHON_BIN}" -m pip install "onnxruntime-gpu<1.24" "tensorrt-cu12==10.9.0.34" "nvidia-cuda-runtime-cu12==12.8.90"
fi

# Refresh editable metadata after selecting the mutually exclusive CPU/GPU ORT.
# --no-deps preserves the selected runtime package.
"${PYTHON_BIN}" -m pip install --no-deps -e "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pip check

CHECK_ARGS=(--policy-provider cpu)
if [[ "${INSTALL_TENSORRT}" -eq 1 ]]; then
  CHECK_ARGS=(--policy-provider tensorrt --benchmark)
fi
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/check_pico_real_env.py" \
  "${CHECK_ARGS[@]}"

printf '%s\n' \
  '[setup-real] PC-side real-robot environment is ready.' \
  '[setup-real] Next run the debug-only preflight:' \
  '  scripts/run_pico_real.sh --net <robot_nic> --pico-service-mode external'
