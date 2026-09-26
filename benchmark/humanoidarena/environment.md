# HumanoidArena Environment Setup

[中文说明](environment_zh.md)

This guide installs the complete environment required by the WB-WAM
HumanoidArena benchmark on a new Ubuntu machine. Use this guide instead of
combining commands from different HumanoidArena or Isaac Lab releases.

The benchmark has two independent Python processes:

```text
HumanoidArena simulator (Python 3.11, Isaac Sim, Isaac Lab, SONIC)
                              ⇅ HTTP
WB-WAM policy server (the repository's WB-WAM training/inference environment)
```

The two environments must remain separate. Installing Isaac Sim into the
WB-WAM policy environment is not supported.

## 1. Requirements

- Ubuntu 22.04 or newer.
- An NVIDIA GPU and driver compatible with CUDA 12.8 and Isaac Sim 5.1.0.
- Miniconda or Anaconda.
- Git and Git LFS.
- `ffmpeg`, `cmake`, and a C/C++ build toolchain.
- A large writable disk for the repository, environments, Isaac caches,
  checkpoints, assets, and results.
- Network access to GitHub, Hugging Face, NVIDIA PyPI, PyTorch wheels, and the
  selected WB-WAM base-model provider.

Install the system packages:

```bash
sudo apt update
sudo apt install -y \
  git git-lfs curl unzip ffmpeg cmake build-essential
git lfs install
```

Select a large-disk installation root. Every path in the rest of this guide is
derived from this variable:

```bash
export WBWAM_INSTALL_ROOT=/data/wbwam-humanoidarena
mkdir -p "$WBWAM_INSTALL_ROOT"/{conda-envs,dependencies,downloads,cache,results}
df -h "$WBWAM_INSTALL_ROOT"
findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS "$WBWAM_INSTALL_ROOT"
test -w "$WBWAM_INSTALL_ROOT"
nvidia-smi
```

Long installs should run in `tmux`, `screen`, or the machine's job scheduler.
If an SSH connection is interrupted, do not immediately rerun the installer:
first check that no old conda/pip process is still writing the same environment.
Never run two package installers against one environment directory.

Do not use a small system partition merely because it is the current working
directory. Isaac Sim, PyTorch, checkpoints, assets, and caches require
substantial space.
Do not select a path from `df` output alone: a large mount may be read-only in
the current container. Network filesystems are valid when writable, but conda
and Isaac Sim installation may be slower because they create many small files.

## 2. Clone WB-WAM and its pinned third-party repositories

HumanoidArena and GMR are Git submodules of WB-WAM. Clone them together so the
versions match the benchmark adapter:

```bash
cd "$WBWAM_INSTALL_ROOT"
git clone --recurse-submodules \
  https://github.com/WB-WaM/WB-WAM-Official.git WB-WAM
cd WB-WAM
git submodule update --init --recursive
git submodule status --recursive
```

Expected layout:

```text
$WBWAM_INSTALL_ROOT/WB-WAM/
├── benchmark/humanoidarena/
├── third_party/HumanoidArena/
└── third_party/GMR/
```

The leading character printed by `git submodule status` must not be `-` or
`+`: `-` means the submodule has not been initialized, while `+` means its
checkout does not match the commit pinned by WB-WAM. Do not update either
submodule to an arbitrary branch tip.

The current release pins these commits:

```text
HumanoidArena  eca00c5ff9bb5bdc76a4d12b18af71a96d4f5891
GMR            bb1bbe40774794fceb2a7c579a3464a28e68c844
```

These hashes are diagnostic information, not a replacement for the gitlinks in
the checked-out WB-WAM release. A future WB-WAM release may intentionally pin
different hashes.

## 3. Create the simulation environment

Create a dedicated Python 3.11 environment on the large disk:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p "$WBWAM_INSTALL_ROOT/conda-envs/unitree_sim_env" \
  python=3.11 pip
conda activate "$WBWAM_INSTALL_ROOT/conda-envs/unitree_sim_env"

python -m pip install --upgrade \
  "pip<26" "setuptools<81" "wheel<0.46"
python -m pip install \
  numpy==1.26.0 Pillow==11.3.0 psutil==5.9.8 packaging==23.0
```

Install the CUDA 12.8 PyTorch stack:

```bash
python -m pip install \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install Isaac Sim 5.1.0:

```bash
python -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
```

Before the first import or launch, enable non-interactive EULA acceptance in
your shell:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

The benchmark launcher exports this variable automatically. The command above
is needed only for manual Isaac Sim import or launch checks in the current
shell; it does not belong in `runtime.local.yaml`.

Verify the versions and GPU before continuing:

```bash
python - <<'PY'
from importlib.metadata import version
import isaacsim
import torch

print("Isaac Sim:", version("isaacsim"))
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Expected key values are Isaac Sim `5.1.0.0`, PyTorch `2.7.0+cu128`, and a
visible NVIDIA GPU.

## 4. Clone the pinned Isaac Lab source

WB-WAM's Isaac Sim 5.1 results used Isaac Lab 2.2.0 at:

```text
46dff135f44683f031edf346e544fcfd8456b2bb
```

The Isaac Lab repository's root `VERSION` is `2.2.0`. The `0.44.9` value in
`source/isaaclab/config/extension.toml` is an internal extension version, not
the Isaac Lab release number.

Clone and verify the exact revision:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git \
  "$WBWAM_INSTALL_ROOT/dependencies/IsaacLab"
git -C "$WBWAM_INSTALL_ROOT/dependencies/IsaacLab" \
  checkout 46dff135f44683f031edf346e544fcfd8456b2bb
test "$(cat "$WBWAM_INSTALL_ROOT/dependencies/IsaacLab/VERSION")" = "2.2.0"
```

The benchmark launcher adds the required Isaac Lab source directories to
`PYTHONPATH`. Therefore the source checkout is required, but installing the
Isaac Lab packages themselves is not expected to be necessary.

Do **not** run the following command for an evaluation-only environment:

```bash
# Do not run this for WB-WAM HumanoidArena evaluation:
./isaaclab.sh --install
```

Without an explicit framework argument, that script installs every supported
RL framework and may replace the pinned PyTorch/CUDA 12.8 stack with CUDA 13
packages. Installing Isaac Lab with
`pip install -e` also introduces a package-metadata conflict: Isaac Lab 2.2.0
declares `Pillow==11.2.1`, while Isaac Sim 5.1.0 declares `Pillow==11.3.0`.

The benchmark keeps Isaac Lab at the pinned commit, uses its source through the
launcher's `PYTHONPATH`, and installs only the runtime dependencies in the next
section. This does not modify Isaac Lab or HumanoidArena source code.

## 5. Install HumanoidArena runtime dependencies

HumanoidArena imports both `isaaclab` and `isaaclab_tasks`; installing Isaac
Sim alone is insufficient. Keep the source packages on `PYTHONPATH` through the
benchmark launcher, and install only their third-party runtime dependencies:

```bash
python -m pip install \
  -r third_party/HumanoidArena/isaaclab_twist2_g1/requirements.txt

python -m pip install \
  prettytable==3.3.0 hidapi==0.14.0.post2 gymnasium==1.2.0 \
  pyglet==1.5.31 transformers==4.57.6 einops==0.8.2 \
  warp-lang==1.17.0 starlette==0.45.3 tensorboard==2.21.0 \
  scikit-learn==1.9.1 numba==0.59.1 qpsolvers==4.13.0 \
  loop-rate-limiters==1.2.0 imageio-ffmpeg==0.6.0 h5py==3.14.0

# flatdict's legacy build imports pkg_resources. Disable its isolated build so
# it uses the already pinned setuptools<81 from section 3.
python -m pip install --no-build-isolation flatdict==4.0.1
```

Clone the pinned Unitree SDK beside Isaac Lab. GMR is already a pinned WB-WAM
submodule, so expose it in the same dependency directory with a symlink:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git \
  "$WBWAM_INSTALL_ROOT/dependencies/unitree_sdk2_python"
git -C "$WBWAM_INSTALL_ROOT/dependencies/unitree_sdk2_python" \
  checkout 65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
ln -s "$WBWAM_INSTALL_ROOT/WB-WAM/third_party/GMR" \
  "$WBWAM_INSTALL_ROOT/dependencies/GMR"
```

Install CycloneDDS C library 0.10.4 in a separate prefix, then build the matching
Python binding. Use this same procedure on Ubuntu 22.04 and 24.04 to avoid
differences in system package versions without replacing system libraries.
Disable the optional Iceoryx shared-memory transport for this build:

```bash
DDS_SRC="$WBWAM_INSTALL_ROOT/dependencies/cyclonedds-0.10.4-src"
DDS_BUILD="$WBWAM_INSTALL_ROOT/dependencies/cyclonedds-0.10.4-build"
export CYCLONEDDS_HOME="$WBWAM_INSTALL_ROOT/dependencies/cyclonedds-0.10.4-install"

git clone --depth 1 --branch 0.10.4 \
  https://github.com/eclipse-cyclonedds/cyclonedds.git \
  "$DDS_SRC"

cmake -S "$DDS_SRC" -B "$DDS_BUILD" \
  -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF \
  -DENABLE_SHM=OFF
cmake --build "$DDS_BUILD" --parallel 8
cmake --install "$DDS_BUILD"

python -m pip install --no-binary cyclonedds cyclonedds==0.10.4
```

Keep `CYCLONEDDS_HOME` set when running evaluation. In a new terminal, set it
again to the same absolute installation path; the simulator inherits this
variable from the evaluation launcher.

The benchmark launcher adds `unitree_sdk2_python` itself to `PYTHONPATH`, so an
editable Unitree package install is not required. Verify the dependency layer:

```bash
PYTHONPATH="$WBWAM_INSTALL_ROOT/dependencies/unitree_sdk2_python" \
python - <<'PY'
import cyclonedds
import onnxruntime
import unitree_sdk2py
print("HumanoidArena runtime dependencies import correctly")
PY
python -m pip check
```

These commands preserve Isaac Sim's required `Pillow==11.3.0` and do not modify
the Isaac Lab or HumanoidArena source trees.

## 6. Restore HumanoidArena simulation assets

Download the HumanoidArena asset archive from Google Drive:

[Download the HumanoidArena simulation assets from Google Drive](https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view?usp=sharing)

Use a browser or a Google Drive client. Save the downloaded zip on the large
disk rather than on a small system partition. The commands below use
`humanoidarena_assets.zip` as a local name; rename the downloaded file or
replace that name with its actual filename:

```bash
mkdir -p "$WBWAM_INSTALL_ROOT/downloads/humanoidarena-assets"
mv /path/to/downloaded-assets.zip \
  "$WBWAM_INSTALL_ROOT/downloads/humanoidarena-assets/humanoidarena_assets.zip"

cd "$WBWAM_INSTALL_ROOT/WB-WAM/third_party/HumanoidArena/isaaclab_twist2_g1"
mkdir -p assets
unzip "$WBWAM_INSTALL_ROOT/downloads/humanoidarena-assets/humanoidarena_assets.zip" \
  -d assets
```

The required final layout is exactly:

```text
$WBWAM_INSTALL_ROOT/WB-WAM/third_party/HumanoidArena/
└── isaaclab_twist2_g1/
    └── assets/
        ├── objects/
        └── robots/
```

Verify both directories before running simulation:

```bash
cd "$WBWAM_INSTALL_ROOT/WB-WAM"
test -d third_party/HumanoidArena/isaaclab_twist2_g1/assets/objects
test -d third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots
find third_party/HumanoidArena/isaaclab_twist2_g1/assets \
  -maxdepth 1 -mindepth 1 -type d -print
```

Both `test` commands must exit successfully, and the `find` output must include
directories ending in `/assets/objects` and `/assets/robots`. If extraction
instead creates `assets/assets/objects` and `assets/assets/robots`, move the
contents of the inner `assets/` directory up one level. Do not continue while
an extra archive-directory layer remains in the paths.

## 7. Download GEAR-SONIC artifacts

Install or reuse the Hugging Face CLI without modifying the pinned simulation
packages, then download the files used by HumanoidArena:

```bash
hf download nvidia/GEAR-SONIC \
  model_encoder.onnx model_decoder.onnx observation_config.yaml \
  --local-dir "$WBWAM_INSTALL_ROOT/downloads/sonic-release"
```

Verify the three files are nonempty:

```bash
test -s "$WBWAM_INSTALL_ROOT/downloads/sonic-release/model_encoder.onnx"
test -s "$WBWAM_INSTALL_ROOT/downloads/sonic-release/model_decoder.onnx"
test -s "$WBWAM_INSTALL_ROOT/downloads/sonic-release/observation_config.yaml"
```

Expected public artifact checksums:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model_encoder.onnx` | 50,100,513 | `013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3` |
| `model_decoder.onnx` | 40,900,688 | `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed` |
| `observation_config.yaml` | 2,336 | `466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b` |

HumanoidArena's public interface runs SONIC ONNX with `--device cpu`. PhysX and
rendering still use the selected NVIDIA GPU.

## 8. Install the WB-WAM policy environment

Create the WB-WAM training/inference environment by following the repository
root installation guide. It is separate from `unitree_sim_env`.

Record the resulting interpreter path because it is used as
`inference_python` in the local benchmark configuration:

```bash
conda activate wbwam
python -c 'import sys; print(sys.executable)'
```

Before simulation, verify the policy environment with the import and model
checks documented by the root README.

## 9. Download WB-WAM base models and HumanoidArena checkpoints

Run these commands from the WB-WAM repository root using the WB-WAM policy
environment, not the Isaac simulation environment.

Verify the downloader first:

```bash
command -v hf || command -v huggingface-cli
python -c 'import sys; print(sys.executable)'
curl -I --connect-timeout 15 \
  https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena
```

The download helper also searches beside `sys.executable`, so invoking it with
an absolute environment Python path works even when that environment has not
been activated.

If the official Hugging Face endpoint is unavailable from the machine, stop
before starting a large download. For a user who explicitly selects the China
mirror already documented by WB-WAM:

```bash
export HF_ENDPOINT=https://hf-mirror.com
curl -I --connect-timeout 15 \
  "$HF_ENDPOINT/WB-WAM/WB-WAM-HumanoidArena"
```

Do not silently replace the public release with an internal MinIO bundle or an
unverified mirror.

Download the pinned base-model files:

```bash
python benchmark/humanoidarena/download_base_models.py \
  --base "$WBWAM_INSTALL_ROOT/downloads/base-models" \
  --provider modelscope
```

Download all seven public HumanoidArena task checkpoints from Hugging Face:

```bash
python scripts/download_models.py arena \
  --output "$WBWAM_INSTALL_ROOT/downloads/checkpoints"
```

Each task checkpoint is large (the current Hammer checkpoint is approximately
12 GB). Check free space before downloading all seven; budget at least 100 GB
for the seven task bundles and temporary/cache overhead.

To download only one task for an initial smoke test:

```bash
python scripts/download_models.py arena --task hammer \
  --output "$WBWAM_INSTALL_ROOT/downloads/checkpoints"
```

The downloader inherits `HF_ENDPOINT` if the user explicitly configured a
Hugging Face mirror.

Each task directory must contain all three artifact types:

```text
$WBWAM_INSTALL_ROOT/downloads/checkpoints/humanoid_arena_native50/<task>/
├── config.yaml
├── dataset_stats.json
└── step_XXXXXX.pt
```

These checkpoints are model files from Hugging Face. They are not Git
submodules and must not be expected inside `git clone`.

## 10. Create the local runtime configuration

```bash
cd "$WBWAM_INSTALL_ROOT/WB-WAM"
cp benchmark/humanoidarena/configs/runtime.example.yaml \
  benchmark/humanoidarena/runtime.local.yaml
mkdir -p "$WBWAM_INSTALL_ROOT/cache/isaac-home"
mkdir -p "$WBWAM_INSTALL_ROOT/cache/isaac-cache"
```

Edit every path in `runtime.local.yaml`. At minimum, it must point to:

- the WB-WAM policy Python interpreter;
- the separate simulation Python interpreter;
- `third_party/HumanoidArena` and its asset directory;
- the dependency root containing the pinned Isaac Lab and GMR checkouts;
- the downloaded GEAR-SONIC directory;
- the base-model directory;
- writable Isaac home/cache directories.

`runtime.local.yaml` is machine-specific and ignored by Git. The
`runtime_root` entry is a cache/temp root, not another software installation.

## 11. Run a smoke test

After verifying the paths, package versions, Git commits, checkpoint files,
and available disk/GPU memory above, activate the WB-WAM policy environment
and run exactly one Hammer episode from the repository root:

```bash
python benchmark/humanoidarena/run_eval.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --task hammer \
  --checkpoint \
    "$WBWAM_INSTALL_ROOT/downloads/checkpoints/humanoid_arena_native50/hammer" \
  --output "$WBWAM_INSTALL_ROOT/results/smoke" \
  --seeds 0 \
  --repeats 1
```

A passing installation smoke must produce:

- one valid trial JSON;
- one playable MP4;
- `server_health.json`;
- a summary without runtime errors;
- no remaining policy/simulator process after exit;
- released GPU memory after exit.

A fall or timeout is a valid episode result. It is not an installation failure.

After this smoke passes, use [README.md](README.md) for single-task and full
seven-task evaluation commands and metric definitions.
