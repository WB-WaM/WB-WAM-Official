<h1 align="center"><img src="assets/wbwam_logo_white.png" alt="WB-WAM" width="280"></h1>

<p align="center"><strong>Heterogeneous Body–Hand Pre-training for Humanoid Loco-Manipulation</strong></p>

<p align="center">
  <a href="https://wb-wam.github.io/"><img src="https://img.shields.io/badge/Project%20Page-WB--WAM-blue?style=flat&amp;logo=github" alt="Project Page"></a>
  <a href=""><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat&amp;logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/WB-WAM"><img src="https://img.shields.io/badge/Hugging%20Face-Models%20%26%20Datasets-orange?style=flat&amp;logo=huggingface" alt="Hugging Face Models &amp; Datasets"></a>
</p>

<p align="center"><a href="README.md">English</a> · 中文</p>

<p align="center">
  <a href="https://wb-wam.github.io/"><img src="assets/wbwam_teaser.png" alt="WB-WAM 论文 teaser：三阶段训练与真机任务展示" width="100%"></a>
</p>

## 项目简介

WB-WAM 是面向人形机器人全身移动操作的世界—动作模型。它通过异构数据预训练、基于 PICO 动作迁移的中间训练，以及面向机器人任务的后训练，将身体、根部与灵巧手监督纳入统一动作空间，联合学习视觉动态和全身动作。真机部署时，身体与根部参考由 SONIC 执行，手部目标直接控制灵巧手。本仓库提供训练、数据采集、Unitree G1 真机部署和 HumanoidArena 仿真评测的入口。

**真机硬件适配范围：** 当前开源部署仅支持配备[舞肌灵巧手](https://www.wuji.tech/zh/hand)和[宇树 G1 双自由度相机头部模组（RealSense D455）](https://www.unitree.com/cn/robocup)的 Unitree G1；其他手部或相机配置暂不支持。

Wuji–G1 连接件：下载 [STL 模型](assets/hardware/wuji_g1_adapter.stl)，用于灵巧手与机器人之间的机械连接。

模型：[预训练与中间训练](https://huggingface.co/WB-WAM/WB-WAM-Pretrain-Midtrain) · [HumanoidArena 后训练](https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena)。使用 [`scripts/download_models.py`](scripts/download_models.py) 按需下载权重。

数据集：[Pico](https://huggingface.co/datasets/WB-WAM/Pico) · [自收数据](https://huggingface.co/datasets/WB-WAM/Self-Collected)。

PICO 第一视角数据采集与处理：见 [Pico-Ego-Collector](https://github.com/WB-WaM/Pico-Ego-Collector)，支持录制数据导入、episode 标注、G1 动作重定向和 LeRobot v3 导出。

<p align="center"><img src="assets/wbwam_pipeline.png" alt="WB-WAM 流程：异构预训练、PICO 中间训练、机器人后训练，以及视频与动作预测和真机控制" width="100%"></p>

| 功能 | 说明 |
| --- | --- |
| [训练](training/README_zh.md) | PICO 数据的中间训练，以及自收和 HumanoidArena 数据的后训练。 |
| [真机部署](bridge/README_zh.md) | 加载 WB-WAM checkpoint，通过 SONIC 控制机器人。 |
| 数据采集 | [SONIC/PICO](collector/sonic/README_zh.md) 或 [Humanoid-GPT](collector/humanoid_gpt/README_zh.md)；采集后转换为训练数据。 |
| [HumanoidArena 评测](benchmark/humanoidarena/README_zh.md) | 在仿真中评测后训练 checkpoint。 |

## 安装

### AI Agent 环境配置

若要让 AI Agent 协助配置环境、下载所选资源或准备部署，可将英文 [配置指南](docs/agent_setup.md) 交给它；Agent 会按所选任务分步配置，每轮只询问关键缺失信息，并在需要下载时确认具体资源。

Agent 可直接配置并验证 PC 和机器人端环境，包括安装、下载、配置文件、Wuji 左右手识别、相机与头部舵机配置，以及 dry-run 检查。实际运行范围已获授权且现场人员就绪后，也可由 Agent 启动、监控和停止手部／身体控制、策略执行及数据采集。用户配合完成接线、穿戴设备和两次独立的左右手观察。

根据需要，将以下对应的一句提示词复制给 AI Agent：

```text
请参考 docs/agent_setup.md，协助我配置 WB-WAM 训练环境。
```

```text
请参考 docs/agent_setup.md，协助我配置 WB-WAM 真机部署环境。
```

```text
请参考 docs/agent_setup.md，协助我配置 WB-WAM 数据收集环境。
```

```text
请参考 docs/agent_setup.md，协助我配置 HumanoidArena 评测环境。
```

### 训练与评测共用的 WB-WAM 环境

以下命令从仓库根目录运行。训练与 HumanoidArena 的 **WB-WAM 策略推理**共用同一个 Python 3.10 conda 环境；HumanoidArena 的 Isaac 仿真器仍需按[评测说明](benchmark/humanoidarena/README_zh.md)安装独立环境。

```bash
conda create -n wbwam python=3.10
conda activate wbwam
python -m pip install --upgrade pip
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./training
```

训练的视频解码还需要系统 FFmpeg 共享库；Ubuntu 上若尚未安装：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

将 `benchmark/humanoidarena/runtime.local.yaml` 的 `inference_python` 指向这个环境的 `bin/python`。训练所需数据、预训练权重与任务命令见[训练说明](training/README_zh.md)；仿真器安装、资产与评测命令见[HumanoidArena 说明](benchmark/humanoidarena/README_zh.md)。

### 真机部署与采集环境

真机 WB-WAM policy 和 SONIC collector **不共用上述 conda 环境，也不共用彼此的环境**：

| 进程 | 环境 | 安装入口 |
| --- | --- | --- |
| WB-WAM 真机 policy | `bridge/.venv-wam` | `bridge/scripts/setup_env.sh` |
| SONIC/PICO 采集与遥操作 | `.venv_teleop` | `scripts/env/setup_envs.sh teleop` |
| 机器人端相机与手部服务 | 机器人端独立环境 | 见[真机部署](bridge/README_zh.md)或[SONIC 采集](collector/sonic/README_zh.md)说明 |

Humanoid-GPT 采集还需要独立的控制器环境，见[HGPT 采集说明](collector/humanoid_gpt/README_zh.md)。

## 安全免责声明

所有部署或运行本项目代码的人员均应具备充分的安全意识，并具备完全操控机器人及在紧急情况下及时停止机器人的能力。因部署或使用本项目代码造成的任何人身伤害或财产损失，本项目作者及维护者概不负责。
