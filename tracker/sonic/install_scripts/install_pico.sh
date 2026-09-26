#!/usr/bin/env bash
# install_pico.sh
# Sets up the gear_sonic_teleop venv for PICO VR teleop on any x86_64 or arm64
# machine (desktop, laptop, or G1 onboard).
# Usage:  bash install_scripts/install_pico.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
XRT_DIR="$REPO_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
XRT_X86_LIB="$XRT_DIR/lib/libPXREARobotSDK.so"
XRT_AARCH64_LIB="$XRT_DIR/lib/aarch64/libPXREARobotSDK.so"

is_git_lfs_pointer() {
    local file="$1"
    local first_line=""
    [ -f "$file" ] || return 1
    IFS= read -r first_line < "$file" || return 1
    [ "$first_line" = "version https://git-lfs.github.com/spec/v1" ]
}

print_lfs_fix() {
    local lib_path="$1"
    local rel_path="${lib_path#"$REPO_ROOT"/}"
    echo "[ERROR] $rel_path is a Git LFS pointer, not the real SDK shared library."
    echo "        Install Git LFS and fetch the binary assets, then re-run this script:"
    echo ""
    echo "          sudo apt install git-lfs"
    echo "          git lfs install"
    echo "          git lfs pull --include=\"$rel_path\""
    echo ""
    echo "        Alternative: build the SDK from source with:"
    echo "          bash external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/setup_ubuntu.sh"
}

# ── 0. Print detected architecture ───────────────────────────────────────────
ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

if [ "$ARCH" != "aarch64" ] && { [ ! -f "$XRT_X86_LIB" ] || is_git_lfs_pointer "$XRT_X86_LIB"; }; then
    print_lfs_fix "$XRT_X86_LIB"
    exit 1
fi

# ── 1. Ensure uv is installed and available ──────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found – installing via official installer …"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the uv env so it's available in this session
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Verify uv is now reachable
    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary not found on PATH."
        echo "        Please add ~/.local/bin (or ~/.cargo/bin) to your PATH and re-run."
        exit 1
    fi
fi
echo "[OK] uv $(uv --version)"

# ── 2. Install a uv-managed Python 3.10 (includes dev headers / Python.h) ────
echo "[INFO] Installing uv-managed Python 3.10 (includes development headers) …"
uv python install 3.10
MANAGED_PY="$(uv python find --no-project 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ──────────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_teleop (if present) …"
rm -rf .venv_teleop

# ── 4. Create venv & install teleop extra ─────────────────────────────────────
echo "[INFO] Creating .venv_teleop with uv-managed Python 3.10 …"
uv venv .venv_teleop --python "$MANAGED_PY" --prompt gear_sonic_teleop
# shellcheck disable=SC1091
source .venv_teleop/bin/activate
echo "[INFO] Installing gear_sonic[teleop] …"
uv pip install -e "gear_sonic[teleop]"

WUJI_RETARGETING_DIR="${WUJI_RETARGETING_ROOT:-}"
if [ -z "$WUJI_RETARGETING_DIR" ]; then
    if [ -d "$REPO_ROOT/../wuji-retargeting/wuji_retargeting" ]; then
        WUJI_RETARGETING_DIR="$REPO_ROOT/../wuji-retargeting"
    elif [ -d "$REPO_ROOT/../wuji-hand/wuji_retargeting" ]; then
        WUJI_RETARGETING_DIR="$REPO_ROOT/../wuji-hand"
    fi
fi

if [ -n "$WUJI_RETARGETING_DIR" ] && [ -d "$WUJI_RETARGETING_DIR/wuji_retargeting" ]; then
    echo "[INFO] Installing Wuji retargeting runtime dependency …"
    # nlopt 2.10 pulls numpy 2.x, but gear_sonic and cmeel-boost require numpy 1.26.x.
    uv pip install "numpy==1.26.4" "nlopt<2.10"

    if [ ! -f "$WUJI_RETARGETING_DIR/wuji_retargeting/wuji-description/hand/body/urdf/right.urdf" ]; then
        echo "[WARN] Wuji description assets not found. gesture_wuji will need:"
        echo "       git clone https://github.com/wuji-technology/wuji-description.git"
        echo "         $WUJI_RETARGETING_DIR/wuji_retargeting/wuji-description"
    fi
fi

# ── 5. Install xrobotoolkit_sdk (CMake-based, not a pip package) ──────────────
echo "[INFO] Installing XRoboToolkit SDK …"
# Install cmake + pybind11 into the venv so the CMake-based build can find them.
# Build with --no-build-isolation so CMake inherits the venv's pybind11.
uv pip install cmake pybind11 setuptools
echo "[OK] cmake $(cmake --version | head -1)"
# Point CMake at pybind11's cmake config so find_package(pybind11) succeeds
export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
echo "[OK] pybind11 cmake dir: $CMAKE_PREFIX_PATH"

# On aarch64 (Jetson Orin), build the PXREARobotSDK native lib from source
# because pre-built aarch64 binaries are not shipped in the repo.
if [ "$ARCH" = "aarch64" ] && { [ ! -f "$XRT_AARCH64_LIB" ] || is_git_lfs_pointer "$XRT_AARCH64_LIB"; }; then
    echo "[INFO] Building PXREARobotSDK for aarch64 (Jetson Orin) …"
    XRT_TMP="$XRT_DIR/tmp"
    mkdir -p "$XRT_TMP"
    if [ ! -d "$XRT_TMP/XRoboToolkit-PC-Service" ]; then
        git clone -b orin https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git "$XRT_TMP/XRoboToolkit-PC-Service"
    fi
    pushd "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK" > /dev/null
    bash build.sh
    popd > /dev/null
    mkdir -p "$XRT_DIR/lib/aarch64" "$XRT_DIR/include/aarch64"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" \
       "$XRT_DIR/include/aarch64/"
    cp -r "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann" \
       "$XRT_DIR/include/aarch64/nlohmann/"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so" \
       "$XRT_DIR/lib/aarch64/"
    rm -rf "$XRT_TMP"
    echo "[OK] PXREARobotSDK aarch64 native library built and installed"
fi

if [ "$ARCH" = "aarch64" ] && is_git_lfs_pointer "$XRT_AARCH64_LIB"; then
    echo "[ERROR] PXREARobotSDK aarch64 build did not replace the Git LFS pointer:"
    echo "        ${XRT_AARCH64_LIB#"$REPO_ROOT"/}"
    exit 1
fi

uv pip install --no-build-isolation -e external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/

# ── 6 & 7: sim extra + unitree_sdk2_python ────────────────────────────────────
# On the onboard Jetson Orin (user==unitree + aarch64) these are not needed:
#   • sim extra depends on mujoco which may lack aarch64 wheels
#   • unitree_sdk2_python requires CycloneDDS C lib (already on the robot)
# They are only installed on desktop / x86 dev machines.
if [ "$ARCH" = "aarch64" ] && [ "$(whoami)" = "unitree" ]; then
    echo "[SKIP] Skipping sim extra & unitree_sdk2_python (onboard Jetson Orin)"
else
    # ── 6. Install sim extra (for run_sim_loop.py / sim2sim testing)
    echo "[INFO] Installing sim extra …"
    uv pip install -e "gear_sonic[sim]"

    # ── 7. Install unitree_sdk2_python (needed by the sim2sim bridge)
    echo "[INFO] Installing unitree_sdk2_python …"
    uv pip install -e external_dependencies/unitree_sdk2_python
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the venv with:"
echo ""
echo "    source .venv_teleop/bin/activate"
echo ""
echo "  You should see (gear_sonic_teleop) in your prompt."
echo "══════════════════════════════════════════════════════════════"
