<h1 align="center"><img src="assets/wbwam_logo_white.png" alt="WB-WAM" width="280"></h1>

<p align="center"><strong>Heterogeneous Body–Hand Pre-training for Humanoid Loco-Manipulation</strong></p>

<p align="center">
  <a href="https://wb-wam.github.io/"><img src="https://img.shields.io/badge/Project%20Page-WB--WAM-blue?style=flat&amp;logo=github" alt="Project Page"></a>
  <a href=""><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat&amp;logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/WB-WAM"><img src="https://img.shields.io/badge/Hugging%20Face-Models%20%26%20Datasets-orange?style=flat&amp;logo=huggingface" alt="Hugging Face Models &amp; Datasets"></a>
</p>

<p align="center">English · <a href="README_zh.md">中文</a></p>

<p align="center">
  <a href="https://wb-wam.github.io/"><img src="assets/wbwam_teaser.png" alt="WB-WAM paper teaser: three-stage training and real-robot tasks" width="100%"></a>
</p>

## Overview

WB-WAM is a world-action model for whole-body humanoid loco-manipulation. Through pre-training on heterogeneous data, intermediate training with PICO motion transfer, and robot-task post-training, it brings body, root, and dexterous-hand supervision into a unified action space and jointly learns visual dynamics and whole-body actions. During real-robot deployment, SONIC executes the body and root references, while hand targets control the dexterous hands directly. This repository provides training, data collection, Unitree G1 deployment, and HumanoidArena evaluation.

**Real-robot hardware support:** The released deployment supports only a Unitree G1 equipped with [Wuji Hands](https://www.wuji.tech/en/hand) and the [G1 2-DOF camera head module (RealSense D455)](https://www.hifivebot.shop/products/g1-head-dual-degree-of-freedom-module); other hand or camera configurations are not supported.

Wuji–G1 mounting adapter: download the [STL model](assets/hardware/wuji_g1_adapter.stl) for the hand-to-robot connector.

Models: [pre-train and mid-train](https://huggingface.co/WB-WAM/WB-WAM-Pretrain-Midtrain) · [HumanoidArena post-train](https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena). Download selected checkpoints with [`scripts/download_models.py`](scripts/download_models.py).

Datasets: [Pico](https://huggingface.co/datasets/WB-WAM/Pico) · [Self-Collected](https://huggingface.co/datasets/WB-WAM/Self-Collected).

PICO egocentric data collection and processing: [Pico-Ego-Collector](https://github.com/WB-WaM/Pico-Ego-Collector) provides recording import, episode annotation, G1 retargeting, and LeRobot v3 export.

<p align="center"><img src="assets/wbwam_pipeline.png" alt="WB-WAM pipeline: heterogeneous pre-training, PICO intermediate training, robot post-training, video and action prediction, and real-robot control" width="100%"></p>

| Component | Description |
| --- | --- |
| [Training](training/README.md) | Intermediate training on PICO data and post-training on self-collected and HumanoidArena data. |
| [Real-robot deployment](bridge/README.md) | Load a WB-WAM checkpoint and control the robot through SONIC. |
| Data collection | [SONIC/PICO](collector/sonic/README.md) or [Humanoid-GPT](collector/humanoid_gpt/README.md); convert collected episodes into training data. |
| [HumanoidArena evaluation](benchmark/humanoidarena/README.md) | Evaluate post-trained checkpoints in simulation. |

## Installation

### Environment setup with an AI agent

Want an AI agent to set up environments, download selected assets, or prepare deployment? Give it the [agent setup guide](docs/agent_setup.md). It will follow your selected workflow, ask only the missing questions in short steps, and confirm downloads when needed.

The agent can configure and verify both the PC and robot environments, including installation, downloads, configuration files, Wuji left/right identification, camera and head-servo setup, and dry-run checks. After the intended live operation is authorized and the on-site operator is ready, it can also start, monitor and stop hand/body controllers, policies and data collection. You handle physical connections, wearable devices and the two independent hand-side observations.

Copy the prompt for the environment you need into your AI agent:

```text
Please follow docs/agent_setup.md and help me set up the WB-WAM training environment.
```

```text
Please follow docs/agent_setup.md and help me set up the WB-WAM real-robot deployment environment.
```

```text
Please follow docs/agent_setup.md and help me set up the WB-WAM data collection environment.
```

```text
Please follow docs/agent_setup.md and help me set up the HumanoidArena evaluation environment.
```

### Shared WB-WAM environment for training and evaluation

Run the following commands from the repository root. Training and the **WB-WAM policy server** for HumanoidArena evaluation share one Python 3.10 conda environment. The Isaac simulator still requires a separate environment, as described in the [evaluation guide](benchmark/humanoidarena/README.md).

```bash
conda create -n wbwam python=3.10
conda activate wbwam
python -m pip install --upgrade pip
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./training
```

Training also needs the system FFmpeg shared libraries for video decoding. On Ubuntu, install them if needed:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Set `inference_python` in `benchmark/humanoidarena/runtime.local.yaml` to this environment's `bin/python`. For datasets, pre-trained weights, and training commands, see the [training guide](training/README.md). For simulator setup, assets, and evaluation commands, see the [HumanoidArena guide](benchmark/humanoidarena/README.md).

### Environments for real-robot deployment and collection

The real-robot WB-WAM policy and SONIC collector **do not share the conda environment above or each other's environment**:

| Process | Environment | Setup |
| --- | --- | --- |
| Real-robot WB-WAM policy | `bridge/.venv-wam` | `bridge/scripts/setup_env.sh` |
| SONIC/PICO collection and teleoperation | `.venv_teleop` | `scripts/env/setup_envs.sh teleop` |
| Robot-side camera and hand services | Separate robot-side environment | See [deployment](bridge/README.md) or [SONIC collection](collector/sonic/README.md) |

Humanoid-GPT collection also needs a [separate controller environment](tracker/humanoid_gpt/README.md).

## Safety disclaimer

Anyone deploying or running this project’s code must have adequate safety awareness and be fully capable of controlling the robot and stopping it promptly in an emergency. The authors and maintainers of this project accept no responsibility for any personal injury or property damage resulting from deployment or use of this code.
