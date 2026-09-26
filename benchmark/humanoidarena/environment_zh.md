# HumanoidArena 环境安装

[English](environment.md)

本文说明如何在一台新的 Ubuntu 机器上，从头安装 WB-WAM HumanoidArena
评测所需的完整环境。请按顺序执行，不要混用其他 HumanoidArena 或 Isaac Lab
版本的安装命令。

评测包含两个相互独立的 Python 进程：

```text
HumanoidArena 仿真器（Python 3.11、Isaac Sim、Isaac Lab、SONIC）
                              ⇅ HTTP
WB-WAM 策略服务（仓库的训练/推理环境）
```

两个环境必须分开。不要把 Isaac Sim 安装进 WB-WAM 策略环境。

## 1. 系统要求

- Ubuntu 22.04 或更高版本。
- 与 CUDA 12.8 和 Isaac Sim 5.1.0 兼容的 NVIDIA GPU 与驱动。
- Miniconda 或 Anaconda。
- Git 和 Git LFS。
- 一块容量足够且可写的磁盘，用于仓库、环境、Isaac 缓存、checkpoint、
  仿真资产和结果。
- 能够访问 GitHub、Hugging Face、NVIDIA PyPI、PyTorch wheel 源以及所选
  WB-WAM 基础模型下载源。

安装系统依赖：

```bash
sudo apt update
sudo apt install -y \
  git git-lfs curl unzip ffmpeg cmake build-essential
git lfs install
```

选择大容量磁盘上的安装根目录。后续路径都从该变量派生：

```bash
export WBWAM_INSTALL_ROOT=/data/wbwam-humanoidarena
mkdir -p "$WBWAM_INSTALL_ROOT"/{conda-envs,dependencies,downloads,cache,results}
df -h "$WBWAM_INSTALL_ROOT"
findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS "$WBWAM_INSTALL_ROOT"
test -w "$WBWAM_INSTALL_ROOT"
nvidia-smi
```

建议在 `tmux`、`screen` 或任务调度器中执行耗时安装。SSH 中断后，先检查
是否仍有 conda/pip 进程在写同一个环境；不要同时对同一环境运行两个安装器。
不要因为当前目录方便就使用容量较小的系统盘，也不要只看 `df` 的容量：容器
中的大盘可能是只读挂载。

## 2. 克隆 WB-WAM 及固定版本的第三方仓库

HumanoidArena 和 GMR 都是 WB-WAM 的 Git 子模块：

```bash
cd "$WBWAM_INSTALL_ROOT"
git clone --recurse-submodules \
  https://github.com/WB-WaM/WB-WAM-Official.git WB-WAM
cd WB-WAM
git submodule update --init --recursive
git submodule status --recursive
```

目录应为：

```text
$WBWAM_INSTALL_ROOT/WB-WAM/
├── benchmark/humanoidarena/
├── third_party/HumanoidArena/
└── third_party/GMR/
```

`git submodule status` 输出的行首不能是 `-` 或 `+`。`-` 表示子模块尚未
初始化，`+` 表示当前 checkout 与 WB-WAM 固定的提交不一致。不要自行把
子模块切换到其他分支。

当前版本固定为：

```text
HumanoidArena  eca00c5ff9bb5bdc76a4d12b18af71a96d4f5891
GMR            bb1bbe40774794fceb2a7c579a3464a28e68c844
```

这些哈希仅用于排查问题；实际版本以当前 WB-WAM checkout 中记录的 gitlink
为准。

## 3. 创建仿真环境

在大容量磁盘上创建独立的 Python 3.11 环境：

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

安装 CUDA 12.8 对应的 PyTorch：

```bash
python -m pip install \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装 Isaac Sim 5.1.0：

```bash
python -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
```

首次 import 或启动前，在当前 shell 中启用非交互式 EULA 接受：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

评测启动器会自动导出该变量。上面的命令只用于在当前 shell 中手动执行
Isaac Sim import 或启动检查，不需要写入 `runtime.local.yaml`。

检查版本和 GPU：

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

关键输出应包含 Isaac Sim `5.1.0.0`、PyTorch `2.7.0+cu128`，并能看到
NVIDIA GPU。

## 4. 克隆固定版本的 Isaac Lab 源码

本评测使用 Isaac Lab 2.2.0，对应提交：

```text
46dff135f44683f031edf346e544fcfd8456b2bb
```

仓库根目录 `VERSION` 中的 `2.2.0` 才是 Isaac Lab 发布版本；
`source/isaaclab/config/extension.toml` 中的 `0.44.9` 是内部 extension
版本，不是 Isaac Lab 发布版本。

```bash
git clone https://github.com/isaac-sim/IsaacLab.git \
  "$WBWAM_INSTALL_ROOT/dependencies/IsaacLab"
git -C "$WBWAM_INSTALL_ROOT/dependencies/IsaacLab" \
  checkout 46dff135f44683f031edf346e544fcfd8456b2bb
test "$(cat "$WBWAM_INSTALL_ROOT/dependencies/IsaacLab/VERSION")" = "2.2.0"
```

评测启动器会把所需 Isaac Lab 源码目录加入 `PYTHONPATH`，因此需要源码
checkout，但不需要安装 Isaac Lab package。

评测环境中不要执行：

```bash
# WB-WAM HumanoidArena 评测不要执行：
./isaaclab.sh --install
```

不指定 framework 时，该脚本会安装所有支持的强化学习框架，并可能替换前面
固定的 PyTorch/CUDA 依赖。`pip install -e` 也会引入 package metadata
冲突：Isaac Lab 2.2.0 声明 `Pillow==11.2.1`，而 Isaac Sim 5.1.0 声明
`Pillow==11.3.0`。本评测只通过 `PYTHONPATH` 使用固定提交的源码，不修改
Isaac Lab 或 HumanoidArena。

## 5. 安装 HumanoidArena 运行依赖

在仓库根目录、仿真环境中执行：

```bash
python -m pip install \
  -r third_party/HumanoidArena/isaaclab_twist2_g1/requirements.txt

python -m pip install \
  prettytable==3.3.0 hidapi==0.14.0.post2 gymnasium==1.2.0 \
  pyglet==1.5.31 transformers==4.57.6 einops==0.8.2 \
  warp-lang==1.17.0 starlette==0.45.3 tensorboard==2.21.0 \
  scikit-learn==1.9.1 numba==0.59.1 qpsolvers==4.13.0 \
  loop-rate-limiters==1.2.0 imageio-ffmpeg==0.6.0 h5py==3.14.0

# flatdict 的旧构建脚本会导入 pkg_resources，因此只对该包关闭构建隔离。
python -m pip install --no-build-isolation flatdict==4.0.1
```

克隆固定版本的 Unitree SDK。GMR 已经是 WB-WAM 子模块，在依赖目录中建立
符号链接即可：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git \
  "$WBWAM_INSTALL_ROOT/dependencies/unitree_sdk2_python"
git -C "$WBWAM_INSTALL_ROOT/dependencies/unitree_sdk2_python" \
  checkout 65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
ln -s "$WBWAM_INSTALL_ROOT/WB-WAM/third_party/GMR" \
  "$WBWAM_INSTALL_ROOT/dependencies/GMR"
```

将 CycloneDDS C library 0.10.4 安装到独立目录，再构建相同版本的 Python
binding。Ubuntu 22.04 和 24.04 统一使用以下流程，避免系统包版本差异，
同时不替换系统自带库。本次构建关闭可选的 Iceoryx 共享内存传输：

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

运行评测时也须保持 `CYCLONEDDS_HOME` 有效。更换终端后，请将它重新设置为
相同的安装目录绝对路径；仿真子进程会从评测启动器继承该变量。

不要对 Unitree 执行 `pip install -e`，否则其 package metadata 会拉取不同
版本的 CycloneDDS。评测启动器会通过 `PYTHONPATH` 直接导入 Unitree 源码。

检查依赖：

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

## 6. 恢复 HumanoidArena 仿真资产

从 Google Drive 下载 HumanoidArena 仿真资产：

[下载 HumanoidArena 仿真资产](https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view?usp=sharing)

把 zip 保存到大容量磁盘。下面假设下载文件名为
`humanoidarena_assets.zip`；如文件名不同，请替换实际路径：

```bash
mkdir -p "$WBWAM_INSTALL_ROOT/downloads/humanoidarena-assets"
mv /path/to/downloaded-assets.zip \
  "$WBWAM_INSTALL_ROOT/downloads/humanoidarena-assets/humanoidarena_assets.zip"

cd "$WBWAM_INSTALL_ROOT/WB-WAM/third_party/HumanoidArena/isaaclab_twist2_g1"
mkdir -p assets
unzip "$WBWAM_INSTALL_ROOT/downloads/humanoidarena-assets/humanoidarena_assets.zip" \
  -d assets
```

最终目录必须是：

```text
$WBWAM_INSTALL_ROOT/WB-WAM/third_party/HumanoidArena/
└── isaaclab_twist2_g1/
    └── assets/
        ├── objects/
        └── robots/
```

检查目录：

```bash
cd "$WBWAM_INSTALL_ROOT/WB-WAM"
test -d third_party/HumanoidArena/isaaclab_twist2_g1/assets/objects
test -d third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots
find third_party/HumanoidArena/isaaclab_twist2_g1/assets \
  -maxdepth 1 -mindepth 1 -type d -print
```

如果解压后得到 `assets/assets/objects`，请把内层 `assets/` 的内容上移一层，
确保运行时路径没有多余目录层级。

## 7. 下载 GEAR-SONIC 文件

在仿真环境中执行：

```bash
hf download nvidia/GEAR-SONIC \
  model_encoder.onnx model_decoder.onnx observation_config.yaml \
  --local-dir "$WBWAM_INSTALL_ROOT/downloads/sonic-release"
```

检查文件：

```bash
test -s "$WBWAM_INSTALL_ROOT/downloads/sonic-release/model_encoder.onnx"
test -s "$WBWAM_INSTALL_ROOT/downloads/sonic-release/model_decoder.onnx"
test -s "$WBWAM_INSTALL_ROOT/downloads/sonic-release/observation_config.yaml"
```

公开文件的预期 checksum：

| 文件 | Bytes | SHA-256 |
| --- | ---: | --- |
| `model_encoder.onnx` | 50,100,513 | `013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3` |
| `model_decoder.onnx` | 40,900,688 | `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed` |
| `observation_config.yaml` | 2,336 | `466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b` |

HumanoidArena 的公开接口使用 `--device cpu` 运行 SONIC ONNX；PhysX 和渲染
仍使用所选 NVIDIA GPU。

## 8. 安装 WB-WAM 策略环境

按照仓库根目录的安装指南创建 WB-WAM 训练/推理环境。该环境与
`unitree_sim_env` 分开。

记录解释器的绝对路径，后续把它写入 `runtime.local.yaml` 的
`inference_python`：

```bash
conda activate wbwam
python -c 'import sys; print(sys.executable)'
```

运行仿真前，完成仓库根 README 中的 import 和模型检查。

## 9. 下载基础模型和 HumanoidArena checkpoint

以下命令应在 WB-WAM 仓库根目录、WB-WAM 策略环境中执行，而不是在 Isaac
仿真环境中执行。

先检查 Hugging Face CLI：

```bash
command -v hf || command -v huggingface-cli
python -c 'import sys; print(sys.executable)'
curl -I --connect-timeout 15 \
  https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena
```

下载脚本也会在 `sys.executable` 所在目录查找 CLI，因此即使没有 activate
环境，也可以用绝对 Python 路径调用。若用户明确选择仓库已说明的中国镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
curl -I --connect-timeout 15 \
  "$HF_ENDPOINT/WB-WAM/WB-WAM-HumanoidArena"
```

不要静默换用内部 MinIO 或未经确认的镜像。

下载固定版本的基础模型文件：

```bash
python benchmark/humanoidarena/download_base_models.py \
  --base "$WBWAM_INSTALL_ROOT/downloads/base-models" \
  --provider modelscope
```

从 Hugging Face 下载全部七个任务的 checkpoint：

```bash
python scripts/download_models.py arena \
  --output "$WBWAM_INSTALL_ROOT/downloads/checkpoints"
```

每个任务 checkpoint 较大；当前 Hammer checkpoint 约 12 GB。下载全部七个
任务前检查剩余空间，建议为 checkpoint 和下载缓存至少预留 100 GB。

只下载 Hammer 做首次 smoke test：

```bash
python scripts/download_models.py arena --task hammer \
  --output "$WBWAM_INSTALL_ROOT/downloads/checkpoints"
```

下载器会继承用户显式设置的 `HF_ENDPOINT`。每个任务目录必须包含：

```text
$WBWAM_INSTALL_ROOT/downloads/checkpoints/humanoid_arena_native50/<task>/
├── config.yaml
├── dataset_stats.json
└── step_XXXXXX.pt
```

这些 checkpoint 是从 Hugging Face 下载的模型文件，不是 Git 子模块。

## 10. 创建本机运行配置

```bash
cd "$WBWAM_INSTALL_ROOT/WB-WAM"
cp benchmark/humanoidarena/configs/runtime.example.yaml \
  benchmark/humanoidarena/runtime.local.yaml
mkdir -p "$WBWAM_INSTALL_ROOT/cache/isaac-home"
mkdir -p "$WBWAM_INSTALL_ROOT/cache/isaac-cache"
```

编辑 `runtime.local.yaml` 中的全部路径，至少包括：

- WB-WAM 策略环境的 Python 解释器；
- 独立仿真环境的 Python 解释器；
- `third_party/HumanoidArena` 及其资产目录；
- 包含固定版本 Isaac Lab、Unitree SDK 和 GMR 的依赖根目录；
- 下载后的 GEAR-SONIC 目录；
- 基础模型目录；
- 可写的 Isaac home/cache 目录。

`runtime.local.yaml` 是本机配置，已被 Git 忽略。`runtime_root` 只是缓存和
临时文件目录，不是另一套软件环境。

## 11. 运行 smoke test

完成以上路径、版本、Git commit、checkpoint、磁盘与 GPU 检查后，激活
WB-WAM 策略环境，并从仓库根目录运行一个 Hammer episode：

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

安装成功的 smoke test 应生成：

- 一个有效的 trial JSON；
- 一个可播放的 MP4；
- `server_health.json`；
- 不含 runtime error 的 summary；
- 退出后不残留策略或仿真进程；
- 退出后 GPU 显存被释放。

episode 的结果为 fall 或 timeout 不代表安装失败。

smoke test 通过后，按照[评测 README](README_zh.md)运行单任务或七任务完整
评测，并查看指标定义。
