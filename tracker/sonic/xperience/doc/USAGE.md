# Xperience 数据处理使用说明

本目录是 Xperience-10M 在本仓库中的最小化处理流程说明，面向 `WB-WAM/tracker/sonic/xperience`。

## 目录结构

- `minimal_xperience_pipeline.py`：核心转换脚本（HuggingFace 下载/转换/inspect/自检）
- `xperience_hf_download.py`：HF 下载脚本（按 episode 下载）
- `xperience_process_episode.py`：单集处理入口（给 stream 脚本调度）
- `run_xperience_stream.sh`：统一管理脚本（下载 + 队列 + 并发处理）
- `.env`：统一运行参数（原始目录、并发数、阈值、模型路径等）
- `MINIMAL_FORMAT.md`：数据格式说明

## 一键运行（推荐）

1. 在项目目录准备 `.env`（与当前代码共用）
2. 挂载/创建根目录（例如 `/gpfs/$USER/xperience_data`）
3. 运行：

```bash
cd <repo-root>/tracker/sonic/xperience
bash run_xperience_stream.sh
```

默认行为会：

- 从 `raw_root` 下载 `annotation.hdf5` / `stereo_left.mp4` / `stereo_right.mp4`
- 下载完成后立即调度处理
- 最终产物保留：
  - `state_action_107_104.parquet`
  - `stereo_left.mp4`
  - `stereo_right.mp4`
  - `caption.json`
  - `episode_metadata.json`
  - `lerobot_episode_manifest.json`
  - `processing_report.json`
  - `annotation_minimal.hdf5`
- 其他中间产物默认清理（仅保留 `annotation_minimal.hdf5`）

## 关键参数说明（`.env`）

- `XPERIENCE_DATA_ROOT`：所有数据根目录
- `RAW_ROOT`：下载后的原始目录
- `PROCESSED_ROOT`：处理后目录
- `QUEUE_DIR`：任务队列目录（JSON manifest）
- `LOG_DIR`：日志目录
- `PROCESS_WORKERS`：并发处理 worker 数
- `PREFETCH_EPISODES`：下载预读队列长度
- `MIN_FREE_GB`：保底剩余空间，低于则停更
- `DELETE_RAW_AFTER_PARSE`：处理成功后是否删原始 `annotation`+`mp4`
- `REQUIRE_GMR` / `REQUIRE_WUJI`：是否要求对应流程必须成功
- `KEEP_INTERMEDIATE`：是否保留更多调试中间文件
- `ENCODER_MODEL`：SONIC encoder ONNX 模型路径
- `WUJI_ROOT`、`WUJI_LEFT_CONFIG`、`WUJI_RIGHT_CONFIG`：WUJI 配置
- `GMR_PYTHON`、`WUJI_PYTHON`：对应环境 python

## 常用子命令

### inspect hdf5

检查一个 `annotation.hdf5` 是否正常（含手部非有限值报告）：

```bash
python minimal_xperience_pipeline.py inspect-hdf5 /path/to/annotation.hdf5 --output /tmp/inspect.json
```

### 单集转换

```bash
python xperience_process_episode.py \
  --queue-manifest /tmp/some_episode.json
```

队列清单由下载脚本按 episode 自动生成。

### 自检（不联网）

```bash
python minimal_xperience_pipeline.py self-test --tmp-dir /tmp/xperience_minimal_pipeline_test
```

## 输出对齐说明

- 默认帧率 `20 FPS`
- 状态：`107` 维（base 重力、角速度、加速度 + G1 body qpos/dq + WUJI40）
- 动作：`104` 维（SONIC token 64 + WUJI target 40）
- 视频会按 224×224 重采样（ffmpeg 可用时）
- 深度数据保留在 `annotation_minimal.hdf5` 的 `depth` 组中（用于后续扩展到 LeRobot）

## 故障排查

- 看日志：`${LOG_DIR}/xperience_stream_*.log`
- 看单集处理报告：`processing_report.json`
- 看 queue：`${QUEUE_DIR}`
- 处理完成标记：`processing_complete.json`

## 说明

`.env` 与脚本均为默认配置入口，服务器上改参数时建议先只改 `.env`，避免重复改脚本参数。
