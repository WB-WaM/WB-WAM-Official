# SONIC 采集数据 → LeRobot v3

[English](README.md)

此离线工具将 SONIC G1/舞肌原始 episode 导出为开源 `real_archive` 使用的原生 LeRobot v3 布局。它不依赖 `datasets/`、`training/`、bridge、机器人 SDK 或模型权重。整个 `processing/` 目录也可复制到其他位置，在 Linux 的 Python 3.10–3.12 环境中运行。在独立虚拟环境中安装 [requirements.txt](requirements.txt)；PyAV 提供 CPU 视频编解码。命令见 [Collector 快速入门](../README_zh.md#5-转换为训练数据)。

## 输入与输出

```text
原始任务目录（或多个任务的父目录）          新归档
my_task/                             my_task/
  metadata.json                        record_0001/
  episode_000000/                         data/chunk-000/file-000.parquet
    data.json                            videos/observation.images.primary/
    color_d455/*.jpg                        chunk-000/file-000.mp4
    depth_d455/...（不导出）                meta/info.json, stats.json
  episode_000001/...                      meta/tasks.parquet
                                         meta/episodes/chunk-000/file-000.parquet
```

`metadata.json` 必须明确声明 `capture_fps: 20` 和主相机。每个 `data.json` 可以是帧列表，或包含 `frames` 列表的对象；帧索引必须从零连续递增。身体观测、非零 wxyz 四元数、双手有效的实际反馈、下一帧 SONIC token 和双手目标都必须存在且为有限数值。缺失或无效样本会报错，不会删除样本或压缩时间轴。缺失图像、指向 episode 目录外的路径也会被拒绝；复用的相机帧会保留。

每个源任务目录对应一条 record。标准化后任务文本相同的多个录制目录会归入同一个任务文件夹，record 按源目录排序编号。任务文本默认来自 `metadata.task_name`，会去掉末尾日期并将下划线替换为空格。工具没有内置任务名或历史 episode 排除项，也不自动划分训练/验证集；WB-WAM 可在训练时按 episode 划分。

## 数据约定

转换器保留采集器原有的图像/状态配对，不重新对齐时钟、不插值、不重采样。输出时间戳为 `frame_index/20`；更详细的时钟和相机诊断仍保留在原始数据中。`N` 帧源数据产生 `N-1` 条训练样本，因为最后一帧只提供动作标签。

| 字段 | 内容 |
| --- | --- |
| `observation.state[110]` | 当前重力方向 3、角速度 3、加速度 3、身体关节 29、关节速度 29、双手实际反馈 40、root 3 |
| `action[136]` | 下一帧 `obs.token_state` 64、由人操作得到的双手目标 40、身体实际关节 29、root 3 |
| `state_mask_110`, `action_mask_136` | `True` 表示无效；导出的值全部有效（`False`） |
| `observation.images.primary` | 当前 RGB；20 Hz H.264，默认 360×270 |

root 3 是 roll、pitch 和**机身坐标系下的 z 轴角速度**，不是 root XYZ。关节位置为绝对值，保留采集器的关节顺序。手部标签是遥操作目标，不是下一帧的实际手部位置。准确的切片与标签含义写在 `meta/info.json`。不导出深度或原始 PICO/SMPL 数据流。统计量是物理量汇总，不是预归一化的训练张量。

## 参数与安全约束

- `--input`、`--output`：必填，必须是不重叠的不同目录。拒绝已有且非本工具创建的输出；没有递归 `--overwrite`。
- `--task "Pick up the object."`、`--task-id pick_object`：单任务输入时可覆盖提示词和目录名，不改变动作标签。
- `--image-size WIDTH HEIGHT`：正偶数尺寸，默认 `360 270`；长宽比必须与源图一致，不拉伸或静默裁剪。
- `--limit-episodes N`：每条 record 取未排除的前 N 条 episode，适合冒烟检查。
- `--exclude-episode task_dir/episode_000058`：可重复指定已审查的排除项；单任务输入也可只写 episode 名。未知选项会报错。
- `--video-max-frames N`：默认 3600，在 episode 之间分割；不会拆开 episode，单条 episode 可以超过目标帧数。每条 record 只有一个数值 Parquet，以适配 WB-WAM 归档读取器。
- `--dry-run`：不写入，仅检查发现结果、路径和指纹；完整数值校验与视频编解码在正式转换时执行。
- `--resume`：校验源 metadata、帧 JSON、引用的 RGB 文件、选项及已完成输出的哈希。复用完整 record；中断的 record 会在临时区重建，不从单帧恢复。输入或输出发生变化时须使用**新的输出目录**。转换过程中不要继续向输入目录采集。

每条 record 在原子发布前都会完整解码并校验。归档包含 `.conversion.json` 和完成状态元数据，以便安全地在本地续跑；不会复制源数据的绝对路径、机器 IP 或硬件序列号。导出不等于语义质检：反馈卡住、示教质量差及任务是否成功仍需人工审核。保留原始数据，以便今后修改标签。

## 加载与独立验证

`validate_lerobot.py --root ...` 可验证一条 record 或整个归档，检查数值行、episode/task 索引、下一帧身体/root 标签、视频帧数和 PTS 以及统计量。它不会修复数据；失败时返回非零状态。可用 `--report /path/outside/the/archive/report.json` 在归档外输出 JSON 报告。

官方 LeRobot 环境的加载示例（已按 **0.4.4** 检查兼容性）：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/sonic", root="/path/to/archive/my_task/record_0001", video_backend="pyav")
sample = dataset[0]
```

WB-WAM 训练时，设置 `WB_WAM_REAL_ROOT=/path/to/archive`，并在后训练配置中选择输出的任务文件夹；相机键为 `observation.images.primary`。训练环境仍需能正常解码视频，并准备匹配的文本 embedding 和归一化统计；转换不会准备模型资产。见[训练说明](../../../training/README_zh.md)。

此工具取代旧版按 episode 导出，作为开源数据处理流程；不会迁移或覆盖旧数据。格式参考：[LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/v0.4.4/lerobot-dataset-v3)。
