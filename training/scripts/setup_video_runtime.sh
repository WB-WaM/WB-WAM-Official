#!/usr/bin/env bash

# Prepare runtime libraries required by CUDA-enabled TorchCodec video decode.
# This keeps data loading on torchcodec instead of falling back to torchvision/pyav.

# shellcheck source=load_env.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/load_env.sh" || return 1

prepend_ld_library_path() {
  local path="$1"
  [[ -d "${path}" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${path}:"*) ;;
    *) export LD_LIBRARY_PATH="${path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

find_cuda_home() {
  local nvcc_path
  local candidate
  local -a candidates

  # Keep an explicitly configured CUDA toolkit only when it is usable. Some
  # cluster login environments export paths that are not mounted in containers.
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    printf '%s\n' "${CUDA_HOME}"
    return 0
  fi

  if nvcc_path="$(command -v nvcc 2>/dev/null)" && [[ -x "${nvcc_path}" ]]; then
    candidate="$(cd -- "$(dirname -- "${nvcc_path}")/.." && pwd -P)"
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  IFS=: read -r -a candidates <<< "${WBWAM_CUDA_CANDIDATES}"
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

if [[ "${WBWAM_SKIP_VIDEO_RUNTIME_SETUP:-0}" != "1" ]]; then
  # Source Lmod init when available; non-interactive Slurm shells may not define `module`.
  if ! command -v module >/dev/null 2>&1; then
    IFS=: read -r -a module_init_paths <<< "${WBWAM_MODULE_INIT_PATHS}"
    for init_script in "${module_init_paths[@]}"; do
      if [[ -r "${init_script}" ]]; then
        # shellcheck source=/dev/null
        source "${init_script}"
        break
      fi
    done
  fi

  if command -v module >/dev/null 2>&1; then
    runtime_shell_options="$-"
    set +u
    module load GCCcore/13.3.0 FFmpeg/7.0.2 >/dev/null 2>&1 || true
    if [[ "${runtime_shell_options}" == *u* ]]; then set -u; fi
  fi

  if resolved_cuda_home="$(find_cuda_home)"; then
    export CUDA_HOME="${resolved_cuda_home}"
  else
    unset CUDA_HOME
    echo "Warning: CUDA toolkit with nvcc was not found; DeepSpeed CUDA-op checks may fail." >&2
  fi
  export FFMPEG_ROOT="${FFMPEG_ROOT:-${WBWAM_FFMPEG_ROOT_DEFAULT}}"
  if [[ -n "${CUDA_HOME:-}" ]]; then
    prepend_ld_library_path "${CUDA_HOME}/lib64"
  fi
  if [[ -n "${FFMPEG_ROOT:-}" ]]; then
    prepend_ld_library_path "${FFMPEG_ROOT}/lib"
  fi
fi
