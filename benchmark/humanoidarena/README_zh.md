# HumanoidArena 评测

[English](README.md)

评测由两个进程组成：WB-WAM 策略服务和原版 HumanoidArena 仿真器。两者通过 HTTP 通信，可以使用不同的 Python 环境。HumanoidArena 作为锁定版本的 Git 子模块引入；本评测不修改其源码。

```text
HumanoidArena 仿真器（环境、SONIC） ⇄ HTTP ⇄ WB-WAM 策略服务
```

## 1. 安装 HumanoidArena

检查 `third_party/HumanoidArena` 和 `third_party/GMR` 是否已克隆；如有缺失，在仓库根目录执行：

```bash
git submodule update --init --recursive
```

从头按照 WB-WAM 的 [HumanoidArena 环境安装指南](environment_zh.md)安装。该指南固定了仿真依赖版本，并包含本评测需要的公开模型和资产下载步骤。不要混用其他 HumanoidArena 或 Isaac Lab 版本的安装命令。WB-WAM 发布评测及下文结果使用 Python 3.11、Isaac Sim 5.1.0 和 Isaac Lab 2.2.0 生成。指南会将发布的 `objects/` 和 `robots/` 资产恢复到：

```text
third_party/HumanoidArena/isaaclab_twist2_g1/assets/
├── objects/
└── robots/
```

HumanoidArena 的公开依赖会安装 CPU 版 ONNX Runtime。本评测沿用上游 `--device cpu` 接口，让 Isaac Lab 张量和 SONIC ONNX 推理留在 CPU；PhysX 与渲染仍使用选定的 NVIDIA GPU，因此无须修改 HumanoidArena 源码。

GMR 由仓库子模块锁定；HumanoidArena 同时由子模块提交和 `benchmark/humanoidarena/upstream.lock.json` 锁定。

## 2. 准备 WB-WAM 权重

WB-WAM 策略服务复用训练的 conda 环境；按[仓库总览](../../README_zh.md)安装一次即可。HumanoidArena 的 Isaac 仿真环境仍独立。

下载固定版本的 WB-WAM 基础模型文件：

```bash
python benchmark/humanoidarena/download_base_models.py \
  --base /absolute/path/to/base-models \
  --provider modelscope
```

从仓库根目录下载指定的 [HumanoidArena 权重](https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena)（将 `hammer` 换成目标任务）：

```bash
python scripts/download_models.py arena --task hammer \
  --output /absolute/path/to/checkpoints
```

每个任务的权重目录应为：

```text
/absolute/path/to/checkpoints/humanoid_arena_native50/<task>/
├── config.yaml
├── dataset_stats.json
└── step_XXXXXX.pt
```

## 3. 配置本机路径

```bash
cp benchmark/humanoidarena/configs/runtime.example.yaml benchmark/humanoidarena/runtime.local.yaml
mkdir -p /absolute/path/to/writable-wbwam-benchmark-cache/isaac-home
mkdir -p /absolute/path/to/writable-wbwam-benchmark-cache/isaac-cache
```

编辑配置中的每个绝对路径。`runtime.local.yaml` 已被 Git 忽略。`runtime_root` 只是缓存和临时文件的可写目录，不要求把 Isaac Sim、Isaac Lab、资产、模型、权重或评测结果搬到这里。

单卡运行时，将两个 GPU 字段设为相同编号，并保持 `allow_shared_gpu: true`。策略服务、PhysX 和渲染使用这张 GPU；上游 HumanoidArena 环境和 SONIC ONNX 张量仍在 CPU。

## 4. 冒烟测试

正式评测前先跑 1 个 episode：

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_eval.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --task hammer \
  --checkpoint /absolute/path/to/checkpoints/humanoid_arena_native50/hammer \
  --output /absolute/path/to/results/smoke \
  --seeds 0 \
  --repeats 1
```

正常的冒烟测试会生成一个 trial JSON、一段 MP4、`server_health.json` 和汇总结果，且没有运行错误。机器人跌倒或 episode 超时仍属于有效的评测结果。

## 5. 评测单个任务

正式协议使用随机种子 `0,1,2`，每个种子运行 20 个 episode：

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_eval.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --task hammer \
  --checkpoint /absolute/path/to/checkpoints/humanoid_arena_native50/hammer \
  --output /absolute/path/to/results/native50 \
  --seeds 0,1,2 \
  --repeats 20
```

支持的任务名为 `kick_football`、`sit_sofa`、`obstacle_navigation`、`hammer`、`box_shelf`、`punch_markers`、`open_door`。

## 6. 评测全部七个任务

`run_all.py` 接受 1–7 个 GPU 编号。单卡时顺序执行；多卡时分配给独立 worker，每张 GPU 同时最多执行一个任务。

单卡：

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_all.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --checkpoint-root /absolute/path/to/checkpoints/humanoid_arena_native50 \
  --output /absolute/path/to/results/native50 \
  --gpus 0 \
  --seeds 0,1,2 \
  --repeats 20
```

七卡：

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_all.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --checkpoint-root /absolute/path/to/checkpoints/humanoid_arena_native50 \
  --output /absolute/path/to/results/native50 \
  --gpus 0,1,2,3,4,5,6 \
  --seeds 0,1,2 \
  --repeats 20
```

也可指定中间数量的 GPU，例如 `--gpus 0,2,5`。中断后重跑会保留有效的 episode JSON，只重新提交缺失的重复序号。

## 7. 指标与评测协议

每个任务按以下方式计算：

1. 每个种子的成功率 = 成功 episode 数 / 20。
2. 任务得分 = 三个种子成功率的平均值。
3. 任务标准差 = 三个种子成功率的总体标准差。
4. 总体均值和总体标准差根据全部 21 个「任务 × 种子」成功率计算。

每个 episode 的最大步数固定在 `benchmark/humanoidarena/configs/tasks.yaml`：

| 官方任务 | CLI 任务名 | 最大步数 |
| --- | --- | ---: |
| Football | `kick_football` | 2000 |
| SitSofa | `sit_sofa` | 2000 |
| Vision Navigation | `obstacle_navigation` | 1800 |
| DoubleDesk | `hammer` | 2000 |
| P&PBox | `box_shelf` | 1450 |
| Boxing | `punch_markers` | 900 |
| OpenDoor | `open_door` | 1800 |

同一任务、同一种子下，仿真器会在所有重复实验之间保持运行（`persistent=true`）。任务表还固定了自然语言指令；内部 Isaac 任务 ID 不会作为指令发送给 WB-WAM。

不启动仿真、只汇总现有结果：

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_eval.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --task all \
  --output /absolute/path/to/results/native50 \
  --seeds 0,1,2 \
  --repeats 20 \
  --summarize-only
```

## 8. 结果

下表的结果使用 Isaac Sim 5.1.0 和 NVIDIA RTX 5090 显卡获得，每个任务使用 3 个种子、每个种子运行 20 个 episode。表中为各任务跨种子成功率的均值 ± 总体标准差：

| 任务 | 成功率 |
| --- | ---: |
| Football | 70.0 ± 8.2% |
| SitSofa | 95.0 ± 4.1% |
| Vision Navigation | 76.7 ± 4.7% |
| DoubleDesk | 65.0 ± 4.1% |
| P&PBox | 86.7 ± 2.4% |
| Boxing | 81.7 ± 2.4% |
| OpenDoor | 98.3 ± 2.4% |
| **总体（21 个任务 × 种子成功率）** | **81.9 ± 12.3%** |

## 排错

- 权重或基础模型报错：查看 `server.log`。
- Isaac Sim、资产、CycloneDDS 或 SONIC 报错：查看 `sim.log`。
- 选中的 GPU 已被占用：每个 seed 启动前会最多等待 180 秒，让显存占用降到 `max_preexisting_gpu_mib` 以下；可用 `HA_GPU_WAIT_SECONDS` 调整等待时间。超时后检查 `nvidia-smi` 列出的进程，换一张空闲卡；只有确定原有占用是有意安排时才调高门槛。
- 仅在继续运行相同权重、相同配置时复用输出目录。
