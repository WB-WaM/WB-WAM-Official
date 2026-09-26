# WB-WAM Training

[中文说明](./README_zh.md)

First install the shared `wbwam` conda environment for training and HumanoidArena policy inference using the [repository overview](../README.md). Run all remaining commands from the `training/` directory.

## Environment

```bash
cp -n .env.example .env
```

After installation, check decoding with an actual training video (replace the path):

```bash
VIDEO=/path/to/episode.mp4 python -c 'import os; from torchcodec.decoders import VideoDecoder; print(VideoDecoder(os.environ["VIDEO"])[0].shape)'
```

Edit `.env` for your machine, then run `source scripts/load_env.sh`. Training also requires Wan2.2 model assets and ActionDiT initialization weights; configure their paths as shown in `.env.example`. If the ActionDiT initialization file is not available, generate it with:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/wbwam_optional_idm.yaml \
  --output "$WBWAM_ACTION_DIT_PATH" --device cuda --dtype bfloat16
```

## Model Downloads

Download the full-pretraining and latest midtraining weights, together with their configs and statistics, from the [WB-WAM pre-train and mid-train repository](https://huggingface.co/WB-WAM/WB-WAM-Pretrain-Midtrain):

The downloader uses the current `HF_ENDPOINT`; set it to your preferred mirror if needed (for example, `export HF_ENDPOINT=https://hf-mirror.com`).

```bash
python ../scripts/download_models.py pretrain --output ./checkpoints/wbwam
python ../scripts/download_models.py midtrain --output ./checkpoints/wbwam
```

For a single [HumanoidArena checkpoint](https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena), use `python ../scripts/download_models.py arena --task hammer --output ./checkpoints/wbwam` (replace `hammer` with the task). The script keeps the training-time directory layout and checks that the checkpoint, config, and statistics are present.

Set these paths in `.env`:

```dotenv
WBWAM_PRETRAIN_CHECKPOINT=./checkpoints/wbwam/pretrain/step_037617.pt
WBWAM_MIDTRAIN_CHECKPOINT=./checkpoints/wbwam/midtrain/step_020560.pt
```

## Data

### Pico and Self-Collected Data

Download the unpacked LeRobot recordings from [Pico](https://huggingface.co/datasets/WB-WAM/Pico) and [Self-Collected](https://huggingface.co/datasets/WB-WAM/Self-Collected):

```bash
hf download WB-WAM/Pico --repo-type dataset --local-dir ./data/pico_archive
hf download WB-WAM/Self-Collected --repo-type dataset --local-dir ./data/real_archive
```

Keep `WB_WAM_DATA_ROOT=./data` in `.env`. Pico is used for midtraining; `real_archive` is used for posttraining on the eight self-collected tasks. No extraction step is needed.

### HumanoidArena

Download only `sonic_8_refpose_v3_1` from the [official SONIC v3.1 dataset](https://huggingface.co/datasets/WilliamWang16/HumanoidArena_dataset_v3_1):

```bash
huggingface-cli download WilliamWang16/HumanoidArena_dataset_v3_1 \
  --repo-type dataset --revision a079beddd6b1521f762c991be8f36993f17ebeca \
  --include 'HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1/**' \
  --local-dir ./data/humanoid_arena
```

Set `WBWAM_ARENA_NATIVE50_ROOT` in `.env` to `./data/humanoid_arena/HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1`. Keep the official 50 Hz frequency and original reference actions; do not resample offline or shift the actions in time.

## Training

First run `source scripts/load_env.sh`. Midtraining reuses the downloaded pretraining normalization statistics; compute separate statistics for each posttraining task. Precompute text embeddings for each task.

### Pico Midtraining and Self-Collected Posttraining

Start Pico midtraining from the pretraining weights:

```bash
python scripts/precompute_text_embeds.py wb_task=midtrain
bash scripts/train_zero2.sh 8 wb_task=midtrain \
  "data.pretrained_norm_stats=./checkpoints/wbwam/pretrain/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT"
```

For self-collected posttraining, use `pillow` as an example and start from the pretraining weights:

```bash
TASK=pillow
python scripts/compute_wb_norm_stats.py --output-dir "$WBWAM_STATS_ROOT/$TASK" "wb_task=posttrain/$TASK"
python scripts/precompute_text_embeds.py "wb_task=posttrain/$TASK"
bash scripts/train_zero2.sh 8 "wb_task=posttrain/$TASK" \
  "data.pretrained_norm_stats=$WBWAM_STATS_ROOT/$TASK/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT"
```

The other tasks are `make_bed`, `close_curtain`, `wipe_table`, `checkout`, `push_cart`, `cloth_basket`, and `fruit_basket`. The `fruit_basket` model is trained jointly on three fruit instructions.

### HumanoidArena Posttraining

All seven tasks start from the full-pretraining weights. For example, `hammer`:

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

The other tasks are `kick_football`, `box_shelf`, `punch_markers`, `open_door`, `sit_sofa`, and `obstacle_navigation`.
