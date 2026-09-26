# HumanoidArena Evaluation

[中文说明](README_zh.md)

The benchmark uses two processes: a WB-WAM policy server and the upstream
HumanoidArena simulator. They communicate over HTTP and may use separate Python
environments. HumanoidArena is included as a pinned Git submodule and is not
modified by this benchmark.

## 1. Install HumanoidArena

Check that `third_party/HumanoidArena` and `third_party/GMR` have been cloned. If either is missing, run this from the repository root:

```bash
git submodule update --init --recursive
```

Follow the WB-WAM [HumanoidArena environment setup guide](environment.md) from
start to finish. It pins the simulator stack and documents every public model
and asset download required by this benchmark. Do not combine its commands
with installation commands from a different HumanoidArena or Isaac Lab
release. The results reported below were produced with Python 3.11, Isaac Sim
5.1.0, and Isaac Lab 2.2.0. The environment guide restores the released
`objects/` and `robots/` assets under:

```text
third_party/HumanoidArena/isaaclab_twist2_g1/assets/
├── objects/
└── robots/
```

HumanoidArena's public requirements install CPU ONNX Runtime. This benchmark
keeps Isaac Lab tensors and SONIC ONNX inference on CPU, matching the public
upstream `--device cpu` interface; PhysX and rendering still run on the selected
NVIDIA GPU. This avoids changing HumanoidArena source code.

The repository pins GMR through its submodule and HumanoidArena through both
its submodule gitlink and `benchmark/humanoidarena/upstream.lock.json`.

## 2. Install WB-WAM and prepare the checkpoint

The WB-WAM policy server reuses the training conda environment; install it once
using the [repository overview](../../README.md). The Isaac simulator still
uses its separate environment.

Download the pinned WB-WAM base-model files:

```bash
python benchmark/humanoidarena/download_base_models.py \
  --base /absolute/path/to/base-models \
  --provider modelscope
```

Download the selected [HumanoidArena checkpoint](https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena) from the repository root (replace `hammer` with the task):

```bash
python scripts/download_models.py arena --task hammer \
  --output /absolute/path/to/checkpoints
```

A task checkpoint has the following on-disk format:

```text
/absolute/path/to/checkpoints/humanoid_arena_native50/<task>/
├── config.yaml
├── dataset_stats.json
└── step_XXXXXX.pt
```

## 3. Configure local paths

```bash
cp benchmark/humanoidarena/configs/runtime.example.yaml benchmark/humanoidarena/runtime.local.yaml
mkdir -p /absolute/path/to/writable-wbwam-benchmark-cache/isaac-home
mkdir -p /absolute/path/to/writable-wbwam-benchmark-cache/isaac-cache
```

Edit every absolute path. `runtime.local.yaml` is ignored by Git. The
`runtime_root` field is only a writable location for cache and temporary files;
it is not another software installation. Existing Isaac Sim, Isaac Lab, assets,
models, checkpoints, and results may remain wherever the user installed them.

For a one-GPU run, set both GPU fields to the same index and keep
`allow_shared_gpu: true`. The policy server, PhysX, and renderer use that GPU;
the upstream HumanoidArena environment and SONIC ONNX tensors remain on CPU.

## 4. Smoke test

Run one episode before a full evaluation:

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_eval.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --task hammer \
  --checkpoint /absolute/path/to/checkpoints/humanoid_arena_native50/hammer \
  --output /absolute/path/to/results/smoke \
  --seeds 0 \
  --repeats 1
```

A valid smoke run produces one trial JSON, one MP4, `server_health.json`, and a
summary without runtime errors. A fall or timeout is a valid episode outcome.

## 5. Run one task

The official protocol uses seeds `0,1,2` and 20 episodes per seed:

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_eval.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --task hammer \
  --checkpoint /absolute/path/to/checkpoints/humanoid_arena_native50/hammer \
  --output /absolute/path/to/results/native50 \
  --seeds 0,1,2 \
  --repeats 20
```

Valid task names are `kick_football`, `sit_sofa`, `obstacle_navigation`,
`hammer`, `box_shelf`, `punch_markers`, and `open_door`.

## 6. Run all seven tasks

`run_all.py` accepts one to seven GPU IDs. With one GPU the tasks run
sequentially; with multiple GPUs they are distributed across independent
workers, and each GPU still runs at most one task at a time.

One GPU:

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_all.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --checkpoint-root /absolute/path/to/checkpoints/humanoid_arena_native50 \
  --output /absolute/path/to/results/native50 \
  --gpus 0 \
  --seeds 0,1,2 \
  --repeats 20
```

Seven GPUs:

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_all.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --checkpoint-root /absolute/path/to/checkpoints/humanoid_arena_native50 \
  --output /absolute/path/to/results/native50 \
  --gpus 0,1,2,3,4,5,6 \
  --seeds 0,1,2 \
  --repeats 20
```

The same command supports any intermediate GPU list, for example
`--gpus 0,2,5`. Interrupted runs are resumable: valid episode JSON files are
kept and only missing repeat indices are submitted again.

## 7. Metrics and protocol

For each task:

1. Each seed success rate is `successful episodes / 20`.
2. The task score is the mean of the three seed success rates.
3. The reported task standard deviation is the population standard deviation
   of those three rates.
4. Overall mean and population standard deviation are computed over all 21
   task-seed success rates.

The maximum episode lengths are fixed in `benchmark/humanoidarena/configs/tasks.yaml`:

| Official task | CLI task | Max steps |
|---|---|---:|
| Football | `kick_football` | 2000 |
| SitSofa | `sit_sofa` | 2000 |
| Vision Navigation | `obstacle_navigation` | 1800 |
| DoubleDesk | `hammer` | 2000 |
| P&PBox | `box_shelf` | 1450 |
| Boxing | `punch_markers` | 900 |
| OpenDoor | `open_door` | 1800 |

The simulator remains alive for all repeats of one task/seed
(`persistent=true`). The task registry also fixes the natural-language prompt;
internal Isaac task IDs are never sent to WB-WAM as instructions.

Summarize existing outputs without starting simulation:

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_eval.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --task all \
  --output /absolute/path/to/results/native50 \
  --seeds 0,1,2 \
  --repeats 20 \
  --summarize-only
```

## 8. Results

These results were obtained with Isaac Sim 5.1.0 on an NVIDIA RTX 5090 GPU,
using three seeds and 20 episodes per seed. Values are mean ± population
standard deviation across seed-level success rates.

| Task | Success rate |
|---|---:|
| Football | 70.0 ± 8.2% |
| SitSofa | 95.0 ± 4.1% |
| Vision Navigation | 76.7 ± 4.7% |
| DoubleDesk | 65.0 ± 4.1% |
| P&PBox | 86.7 ± 2.4% |
| Boxing | 81.7 ± 2.4% |
| OpenDoor | 98.3 ± 2.4% |
| **Overall (21 task-seed rates)** | **81.9 ± 12.3%** |

## Troubleshooting

- Inspect `server.log` for checkpoint or base-model errors.
- Inspect `sim.log` for Isaac Sim, asset, CycloneDDS, or SONIC errors.
- If a selected GPU is already occupied, choose another ID or raise
  `max_preexisting_gpu_mib` only when the existing allocation is intentional.
  Before each seed, the launcher waits up to 180 seconds for GPU memory to fall
  below this limit; set `HA_GPU_WAIT_SECONDS` to change the wait time. If it
  still fails, inspect the processes reported by `nvidia-smi`.
- Reuse an output directory only to resume the same checkpoint and settings.
