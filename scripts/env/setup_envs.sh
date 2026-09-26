#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SONIC_ROOT="$REPO_ROOT/tracker/sonic"

# shellcheck source=env_versions.sh
source "$SCRIPT_DIR/env_versions.sh"

FORCE=0
TARGETS=()

usage() {
    cat <<'EOF'
Usage:
  scripts/env/setup_envs.sh [core|all|teleop|sim|wam|check] [--force]

Targets:
  core      Install PC-side runtime envs: teleop, wam. Default.
  all       Install teleop, sim, wam.
  teleop    Create .venv_teleop for SONIC collector / Pico manager.
  sim       Create .venv_sim for SONIC MuJoCo simulation.
  wam       Create bridge/.venv-wam for WBWAM bridge.
  check     Print import/version checks for existing envs.

Options:
  --force   Remove the selected venv(s) before recreating them.

Examples:
  scripts/env/setup_envs.sh
  scripts/env/setup_envs.sh all --force
  scripts/env/setup_envs.sh wam
EOF
}

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --force)
            FORCE=1
            ;;
        core|all|teleop|sim|wam|check)
            TARGETS+=("$1")
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if ((${#TARGETS[@]} == 0)); then
    TARGETS=(core)
fi

expand_targets() {
    local expanded=()
    local target
    for target in "${TARGETS[@]}"; do
        case "$target" in
            core)
                expanded+=(teleop wam)
                ;;
            all)
                expanded+=(teleop sim wam)
                ;;
            *)
                expanded+=("$target")
                ;;
        esac
    done
    printf "%s\n" "${expanded[@]}" | awk '!seen[$0]++'
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        echo "[OK] $(uv --version)"
        return
    fi
    echo "[INFO] uv not found; installing with the official installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [[ -f "$HOME/.local/bin/env" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || {
        echo "[ERROR] uv installation succeeded, but uv is not on PATH" >&2
        exit 1
    }
    echo "[OK] $(uv --version)"
}

managed_python() {
    uv python install "$WBWAM_PYTHON_VERSION"
    uv python find --no-project "$WBWAM_PYTHON_VERSION"
}

create_venv() {
    local venv_dir="$1"
    local prompt="$2"
    local python_bin="$3"
    if [[ "$FORCE" == "1" && -d "$venv_dir" ]]; then
        echo "[INFO] Removing $venv_dir"
        rm -rf "$venv_dir"
    fi
    if [[ ! -d "$venv_dir" ]]; then
        echo "[INFO] Creating $venv_dir"
        uv venv "$venv_dir" --python "$python_bin" --prompt "$prompt"
    else
        echo "[INFO] Reusing existing $venv_dir"
    fi
}

setup_teleop() {
    local py manus_source manus_build
    py="$(managed_python)"
    create_venv "$REPO_ROOT/.venv_teleop" gear_sonic_teleop "$py"
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv_teleop/bin/activate"
    uv pip install \
        -e "$SONIC_ROOT/gear_sonic[teleop]" \
        "numpy==$WBWAM_TELEOP_NUMPY_VERSION" \
        "pin==3.8.0" \
        "cmeel-urdfdom==4.0.1" \
        "opencv-python==$WBWAM_OPENCV_VERSION" \
        "nlopt<2.10" \
        -e "$REPO_ROOT/third_party/wuji_retargeting"
    uv pip install cmake pybind11 setuptools
    export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
    uv pip install --no-build-isolation -e "$REPO_ROOT/third_party/xrobotoolkit_sdk"
    manus_source="$REPO_ROOT/third_party/MANUS_SDK/ManusSDK_v3.1.1/SDKClient_Linux"
    manus_build="$SONIC_ROOT/.deps/manus_sdk"
    mkdir -p "$manus_build"
    make -B -C "$manus_source" \
        PYTHON="$REPO_ROOT/.venv_teleop/bin/python" BUILD="$manus_build"
    PYTHONPATH="$manus_build${PYTHONPATH:+:$PYTHONPATH}" \
        "$REPO_ROOT/.venv_teleop/bin/python" -c 'import ManusServer'
    deactivate
}

setup_sim() {
    local py
    py="$(managed_python)"
    create_venv "$REPO_ROOT/.venv_sim" gear_sonic_sim "$py"
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv_sim/bin/activate"
    uv pip install \
        -e "$SONIC_ROOT/gear_sonic[sim]" \
        "numpy==$WBWAM_NUMPY_VERSION" \
        "opencv-python==$WBWAM_OPENCV_VERSION" \
        -e "$SONIC_ROOT/external_dependencies/unitree_sdk2_python"
    deactivate
}

setup_wam() {
    if [[ "$FORCE" == "1" ]]; then
        rm -rf "$REPO_ROOT/bridge/.venv-wam"
    fi
    "$REPO_ROOT/bridge/scripts/setup_env.sh"
}

ensure_uv
EXPANDED_TARGETS=()
while IFS= read -r target; do
    EXPANDED_TARGETS+=("$target")
done < <(expand_targets)
TARGETS=("${EXPANDED_TARGETS[@]}")

for target in "${TARGETS[@]}"; do
    echo ""
    echo "==> setup $target"
    case "$target" in
        teleop) setup_teleop ;;
        sim) setup_sim ;;
        wam) setup_wam ;;
        check) "$SCRIPT_DIR/check_envs.sh" ;;
    esac
done

echo ""
echo "[OK] Environment setup complete"
