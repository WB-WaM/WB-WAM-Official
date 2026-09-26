# WB-WAM 训练

[English](./README.md)

先按[仓库总览](../README_zh.md)安装训练与 HumanoidArena 评测共用的 `wbwam` conda 环境。以下其余命令均从 `training/` 目录运行。

## 环境

```bash
cp -n .env.example .env
```

安装后用一段实际训练视频检查解码（替换视频路径）：

```bash
VIDEO=/path/to/episode.mp4 python -c 'import os; from torchcodec.decoders import VideoDecoder; print(VideoDecoder(os.environ["VIDEO"])[0].shape)'
```

按机器修改 `.env`，然后执行 `source scripts/load_env.sh`。训练还需要 Wan2.2 的模型资产和 ActionDiT 初始化权重；按 `.env.example` 配置路径。若尚无 ActionDiT 初始化文件，可运行：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/wbwam_optional_idm.yaml \
  --output "$WBWAM_ACTION_DIT_PATH" --device cuda --dtype bfloat16
```

## 模型下载

从 [WB-WAM 预训练与中间训练模型仓库](https://huggingface.co/WB-WAM/WB-WAM-Pretrain-Midtrain)下载全量预训练和最新中间训练权重，以及配套配置、统计量：

下载脚本使用当前的 `HF_ENDPOINT`；如需镜像，可先设置 `export HF_ENDPOINT=https://hf-mirror.com`。

```bash
python ../scripts/download_models.py pretrain --output ./checkpoints/wbwam
python ../scripts/download_models.py midtrain --output ./checkpoints/wbwam
```

如需单独下载 [HumanoidArena 权重](https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena)，可运行 `python ../scripts/download_models.py arena --task hammer --output ./checkpoints/wbwam`（将 `hammer` 换成目标任务）。脚本保留训练时的目录结构，并检查权重、配置和统计量是否齐全。

在 `.env` 中设置：

```dotenv
WBWAM_PRETRAIN_CHECKPOINT=./checkpoints/wbwam/pretrain/step_037617.pt
WBWAM_MIDTRAIN_CHECKPOINT=./checkpoints/wbwam/midtrain/step_020560.pt
```

## 数据

### Pico 与自收数据

从 [Pico](https://huggingface.co/datasets/WB-WAM/Pico) 和[自收数据](https://huggingface.co/datasets/WB-WAM/Self-Collected)直接下载解压后的 LeRobot recording：

```bash
hf download WB-WAM/Pico --repo-type dataset --local-dir ./data/pico_archive
hf download WB-WAM/Self-Collected --repo-type dataset --local-dir ./data/real_archive
```

保持 `.env` 中的 `WB_WAM_DATA_ROOT=./data`；Pico 用于中间训练，`real_archive` 用于八个自收任务的后训练。无需解压。

### HumanoidArena

只下载[官方 SONIC v3.1 数据](https://huggingface.co/datasets/WilliamWang16/HumanoidArena_dataset_v3_1)中的 `sonic_8_refpose_v3_1`：

```bash
huggingface-cli download WilliamWang16/HumanoidArena_dataset_v3_1 \
  --repo-type dataset --revision a079beddd6b1521f762c991be8f36993f17ebeca \
  --include 'HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1/**' \
  --local-dir ./data/humanoid_arena
```

将 `.env` 的 `WBWAM_ARENA_NATIVE50_ROOT` 设为 `./data/humanoid_arena/HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1`。数据保持官方 50 Hz 和原始 reference action，不做离线重采样或 action 后移。

## 训练

先执行 `source scripts/load_env.sh`。midtrain 复用已下载的 pretrain norm stats；每个后训练任务独立计算 stats。各任务还需预计算语言 embedding。

### Pico 中间训练与自收后训练

Pico midtrain 从预训练权重启动：

```bash
python scripts/precompute_text_embeds.py wb_task=midtrain
bash scripts/train_zero2.sh 8 wb_task=midtrain \
  "data.pretrained_norm_stats=./checkpoints/wbwam/pretrain/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT"
```

自收后训练以 `pillow` 为例，从 pretrain 权重启动：

```bash
TASK=pillow
python scripts/compute_wb_norm_stats.py --output-dir "$WBWAM_STATS_ROOT/$TASK" "wb_task=posttrain/$TASK"
python scripts/precompute_text_embeds.py "wb_task=posttrain/$TASK"
bash scripts/train_zero2.sh 8 "wb_task=posttrain/$TASK" \
  "data.pretrained_norm_stats=$WBWAM_STATS_ROOT/$TASK/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT"
```

其余任务：`make_bed`、`close_curtain`、`wipe_table`、`checkout`、`push_cart`、`cloth_basket`、`fruit_basket`。`fruit_basket` 联合训练三条水果指令。

### HumanoidArena 后训练

七个任务均从全量预训练权重启动；以 `hammer` 为例：

```bash
TASK=hammer
python scripts/compute_wb_norm_stats.py \
  --output-dir "$WBWAM_STATS_ROOT/humanoid_arena_native50/$TASK" \
  "wb_task=posttrain/humanoid_arena_native50/$TASK"
python scripts/precompute_text_embeds.py "wb_task=posttrain/humanoid_arena_native50/$TASK"
bash scripts/train_zero2.sh 8 "wb_task=posttrain/humanoid_arena_native50/$TASK" \
  "data.pretrained_norm_stats=$WBWAM_STATS_ROOT/humanoid_arena_native50/$TASK/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT"
```

其余任务：`kick_football`、`box_shelf`、`punch_markers`、`open_door`、`sit_sofa`、`obstacle_navigation`。
