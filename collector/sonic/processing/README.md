# SONIC collector → LeRobot v3

[中文](README_zh.md)

This offline tool exports raw SONIC G1/Wuji episodes directly to the native v3
layout used by the released `real_archive`. It does not depend on `datasets/`,
`training/`, the bridge, hardware SDKs or model weights. The complete `processing/`
directory can also be copied elsewhere and run using Python 3.10–3.12 on Linux.
Install [requirements.txt](requirements.txt) in a separate virtual environment;
PyAV supplies the CPU video encoder/decoder. The commands are in the
[Collector quick start](../README.md#5-export-collected-episodes-for-training).

## Input and output

```text
Raw task (or a parent of several tasks)     New archive
my_task/                                  my_task/
  metadata.json                             record_0001/
  episode_000000/                              data/chunk-000/file-000.parquet
    data.json                                 videos/observation.images.primary/
    color_d455/*.jpg                             chunk-000/file-000.mp4
    depth_d455/...  (not exported)             meta/info.json, stats.json
  episode_000001/...                           meta/tasks.parquet
                                              meta/episodes/chunk-000/file-000.parquet
```

Requirements: `metadata.json` explicitly declares `capture_fps: 20` and a primary
camera. Each `data.json` is a list of frames, or an object with a `frames` list.
Frame indices must be contiguous from zero. Body observations, a nonzero wxyz
quaternion, valid measured feedback for both hands, next-frame SONIC tokens and
hand targets must be present and finite. Missing/invalid samples cause an error;
they are not deleted and the timeline is not compressed. Missing image files and
paths escaping an episode are rejected. Reused camera frames are retained.

One source task directory becomes one record. Equal normalized task texts group
multiple recordings under the same task folder; records are numbered in sorted
source-directory order. Task text defaults to `metadata.task_name`, with a trailing
date removed and underscores replaced by spaces. There are no built-in task names
or historical episode exclusions. No train/validation split is invented; WB-WAM
can split episodes at training time.

## Data contract

The converter preserves the collector's existing image/state pairing. It does
not realign clocks, interpolate, or resample. Output timestamps are `frame_index/20`;
the detailed clock and camera diagnostics remain in the raw data. `N` source
frames produce `N-1` training samples, since the last frame supplies only a label.

| Field | Contents |
| --- | --- |
| `observation.state[110]` | Current gravity3, angular velocity3, acceleration3, body joints29, joint velocities29, measured hands40, root3 |
| `action[136]` | Next `obs.token_state`64, `human_derived` hand targets40, measured body joints29, root3 |
| `state_mask_110`, `action_mask_136` | `True` means invalid; exported values are all valid (`False`) |
| `observation.images.primary` | Current RGB; 20 Hz H.264, default 360×270 |

Root3 is roll, pitch and **body-frame z angular velocity**, not root XYZ. Joint
positions are absolute; the collector's joint ordering is preserved. Hand labels
are teleoperation targets, not next measured hand positions. Exact slices and label
semantics are saved in `meta/info.json`. No depth or raw PICO/SMPL streams are exported.
Statistics are physical-value summaries, not pre-normalized training tensors.

## Options and safeguards

- `--input`, `--output`: required, separate non-overlapping directories. Existing
  unowned output is rejected. There is deliberately no recursive `--overwrite`.
- `--task "Pick up the object."`, `--task-id pick_object`: optional prompt/folder
  overrides for a single-task input. They do not change action labels.
- `--image-size WIDTH HEIGHT`: positive even dimensions; default `360 270`.
  Aspect ratio must match the source; images are not stretched or silently cropped.
- `--limit-episodes N`: first N non-excluded episodes per record, useful for a smoke test.
- `--exclude-episode task_dir/episode_000058`: repeat for audited exclusions. A bare
  episode name is accepted for single-task input. Unknown selections are errors.
- `--video-max-frames N`: default 3600; split between episodes. An episode is never
  cut across video files and may exceed this target. There is one numerical Parquet
  per record, as required by the WB-WAM archive reader.
- `--dry-run`: check discovery, paths and fingerprints without writing. Full
  numerical validation and video encoding/decoding run during conversion.
- `--resume`: hash-check source metadata, frame JSON, referenced RGB files, options
  and completed outputs. Completed records are reused; an interrupted record is
  rebuilt in staging, not resumed at individual frames. Changed inputs or outputs
  require a **new output directory**. Do not collect into the input while converting.

Each record is fully decoded and validated before atomic publication. The archive
contains `.conversion.json` and completion metadata for safe local resume; no
absolute source paths, machine IPs or hardware serial numbers are copied into them.
Export is not a semantic quality audit: frozen-but-valid feedback, poor demonstrations
and task success still need review. Keep raw data for future label changes.

## Loading and independent validation

`validate_lerobot.py --root ...` accepts one record or an archive and checks all
numerical rows, episode/task indices, next-state body/root labels, video frame
counts/PTS, and statistics. It never repairs data; failures return nonzero.
Use `--report /path/outside/the/archive/report.json` for an optional JSON report.

With the official LeRobot environment (compatibility checked against **0.4.4**):

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/sonic", root="/path/to/archive/my_task/record_0001", video_backend="pyav")
sample = dataset[0]
```

For WB-WAM, set `WB_WAM_REAL_ROOT=/path/to/archive` and select the output task folder
in the post-training recipe. Its camera key is `observation.images.primary`.
The training environment still needs a working video decoder, matching text
embeddings and normalization setup; conversion does not prepare model assets.
See the [training guide](../../../training/README.md).

This replaces the old episode-based export as the public workflow. Existing data
is not migrated or overwritten. Format reference: [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/v0.4.4/lerobot-dataset-v3).
