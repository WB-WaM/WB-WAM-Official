<div align="center">

# 🤖 Humanoid-GPT

### [CVPR 2026] Humanoid Generative Pre-Training for Zero-Shot Motion Tracking

<p align="center">
  <a href="https://cvpr.thecvf.com/Conferences/2026"><img src="https://img.shields.io/badge/CVPR-2026-4b44ce.svg?style=flat-square" alt="CVPR 2026"></a>
  <a href="https://arxiv.org/abs/2606.03985"><img src="https://img.shields.io/badge/arXiv-2606.03985-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://qizekun.github.io/Humanoid-GPT/"><img src="https://img.shields.io/badge/Project-Page-blue.svg?style=flat-square" alt="Project Page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <img src="storage/assets/teaser.png" width="100%" alt="Humanoid-GPT Teaser">
</p>

</div>

---

## 📖 Overview

**Humanoid-GPT** is the first **GPT-style humanoid motion Transformer** trained with causal attention on a billion-scale motion corpus for whole-body control. Unlike prior shallow MLP trackers constrained by scarce data and an agility–generalization trade-off, Humanoid-GPT is pre-trained on a **2B-frame retargeted corpus** that unifies all major mocap datasets with large-scale in-house recordings.

<details>
<summary><b>🔬 Key Contributions</b></summary>

- **Billion-Scale Pre-Training**: First to scale humanoid motion learning to 2B frames
- **GPT-Style Architecture**: Causal Transformer with Rotary Position Embeddings (RoPE)
- **Zero-Shot Generalization**: Track arbitrary unseen motions without fine-tuning

</details>

### ✨ Highlights

| Feature             | Description                                                               |
|---------------------|---------------------------------------------------------------------------|
| 🧠 **Architecture** | Causal Transformer with RoPE, supporting variable-length motion sequences |
| 📊 **Scale**        | Pre-trained on 2B motion frames from unified mocap datasets               |
| 🎯 **Zero-Shot**    | Unprecedented generalization to unseen motions and tasks                  |
| 🤖 **Platform**     | Optimized for Unitree G1 humanoid robot (29 DOF whole-body)               |
| ⚡  **Speed**        | GPU-accelerated simulation with MuJoCo-MJX                                |

---

## 📦 Installation

### Prerequisites

- NVIDIA GPU with CUDA 12.x
- **MacOS** is also supported for testing if you skip **jax[cuda12]** and use **mjpython** (e.g. `mjpython -m scripts.app`).
- Conda / Miniconda

### Quick Start

```bash
git clone https://github.com/qizekun/Humanoid-GPT.git
cd Humanoid-GPT

conda create -n h-gpt python=3.12 -y
conda activate h-gpt

pip install -e ".[cuda]"     # use ".[cpu]" for CPU inference or MacOS
```

On MacOS, use `mjpython` instead of `python` for the MuJoCo viewer (e.g. `mjpython -m scripts.app`).

### 🔧 G1 Hardware Version

We support multiple Unitree G1 hardware versions via the `G1_VERSION` env var (default `5010`). The asset folder `storage/assets/unitree_g1_${G1_VERSION}/` is selected automatically:

```bash
G1_VERSION=5010 python -m scripts.inference ...                   # default: 5010
```

`G1_VERSION` selects the MuJoCo/FK model, policy geometry, and reviewed safety
limits; it is not hardware auto-detection and it does not force the DDS
`LowCmd.mode_machine` header. Real deployment copies the live
`LowState.mode_machine` value into `LowCmd`, as in Unitree's low-level
examples. The `5010` asset in this checkout remains responsible for the
29-DoF model, policy geometry, gains, and safety limits only.

---

## 🚀 Inference & Evaluation

A pre-trained tracking policy (`.onnx`) and a sample trajectory under
`storage/test/` are all you need to get started.

```bash
# Interactive Gradio demo
python -m scripts.app

# Track a single motion / a folder of motions
python -m scripts.inference --load_path storage/ckpts/pns_wo_priv216.onnx --mocap_path storage/test

# Parallel evaluation over a folder of trajectories
python -m scripts.eval_parallel --load_path storage/ckpts/pns_wo_priv216.onnx \
    --mocap_path storage/test --workers 32 --privileged

# Visualize a reference trajectory
python -m scripts.vis --mocap_path storage/test
```

The expected motion format is a `.npz` containing either `qpos` directly, or
`root_pos` / `root_rot` / `dof_pos` arrays. To convert retargeted mocap into
the keypoint representation the policy consumes:

```bash
python tracking/convert_qpos2kpt.py --mocap_npz <mocap_path.npz> --debug   # single file (debug viz)
python tracking/convert_parallel.py --src_dir <in_dir> --save_dir <out_dir> --num_workers 32
```

---

## Pico teleoperation in MuJoCo and on G1

This checkout adds a Pico control path for both MuJoCo and a PC-hosted G1
deployment:

```text
Pico / XRoboToolkit global body poses (24 x 7, xyz + xyzw)
  -> Unity-to-GMR coordinates and first-frame XY/yaw/foot-ground calibration
  -> GMR (xrobot -> unitree_g1)
  -> G1 qpos36 (root xyz + root wxyz + 29 joints)
  -> LiveRefConverter
  -> HumanoidGPT ONNX tracking policy
  -> 29 motor position targets
  -> MuJoCo, or Unitree DDS over Ethernet -> G1
```

The direct Pico path remains available for body-only fallback. For synchronized
body and WujiHand control, the independent HGPT Pose Manager owns Pico/GMR,
Pico or MANUS hand retargeting, keyboard modes, and offline replay, then publishes
one SONIC-compatible ZMQ `pose` stream. HGPT still sends body `LowCmd` over
Unitree DDS, while the unchanged robot-side SONIC Wuji server consumes only the
20+20 hand fields and drives USB.

### WB-WAM teleoperation and collection quickstart

Run PC tracker commands from `tracker/humanoid_gpt/` in the `h-gpt` environment.
Use a separate terminal for each long-running process. Complete the
[environment setup](#environment) and [real-robot preflight](#pc-hosted-g1-deployment)
before enabling hardware output.

The launchers share a local `hgpt_collection.env` copied from
[hgpt_collection.env.example](../../collector/humanoid_gpt/scripts/hgpt_collection.env.example):
MANUS hands, external Pico Service, TensorRT, pose port `5556`, and telemetry
port `5558`. Set the NIC and robot endpoints for your setup;
exported `HGPT_*` variables override defaults, and explicit CLI options win.

```text
Pico body + MANUS hands -> Pose Manager :5556 -> HGPT controller -> G1 body
                                            -> Wuji server -> hands
HGPT telemetry :5558 + Wuji feedback :5559 + Camera :5560 -> Collector
```

Install MANUS support with `scripts/setup_pico_env.sh --with-manus`; its binding
lives in `.deps/manus_sdk`. Stop other Pico/MANUS readers first, including the
SONIC Pico manager, `SDKClient_Linux.out`, and hand-only tests. For external
Pico Service, start `/opt/apps/roboticsservice/runService.sh` separately, or use
`--pico-service-mode auto` to let the manager start it.

For simulation or inspection, explicitly disable hand output: the wrapper
**enables hand publication by default**, even when the flag is omitted.

```bash
scripts/run_hgpt_pose_manager.sh --hand-source pico --no-publish-wuji-hand
# Separate PC terminal, also in tracker/humanoid_gpt/:
scripts/run_pico_sim.sh --pose-endpoint tcp://127.0.0.1:5556 --headless
```

For real teleoperation, first start the Wuji server on the robot **from the
repository root**. Configure the PC address and hand serials in
[wuji_hand_server.env.example](../../collector/sonic/scripts/wuji_hand_server.env.example)
copied to the local `wuji_hand_server.env`.

```bash
collector/sonic/scripts/run_wuji_hand_server.sh
```

Then start the pose manager and controller on the PC:

```bash
# PC terminal 1, in tracker/humanoid_gpt/:
scripts/run_hgpt_pose_manager.sh \
  --hand-source manus --pico-service-mode external --publish-wuji-hand

# PC terminal 2, in tracker/humanoid_gpt/:
scripts/run_pico_real.sh \
  --net eno1 --pose-endpoint tcp://127.0.0.1:5556 \
  --state-action-bind tcp://127.0.0.1:5558 \
  --policy-provider tensorrt --publish-lowcmd
```

`--publish-lowcmd` enables real body commands. Omitting it disables **body**
commands only; disable hand publication separately for a no-actuation check.
Suspend the robot for initial tests and keep the G1 remote in hand.

```text
Start services -> G1 L2+R2 (debug) -> Start (default pose) -> A (control loop)
              -> align operator/robot poses -> Pose Manager 1 (live tracking)
```

| Device | Controls |
| --- | --- |
| G1 remote | `Select`: emergency damping |
| Pose Manager window | `0`: walk; `1`: live tracking; `2+`: offline motion; `R`: recalibrate |
| Pico left stick click | Toggle walk/live; re-entering live recalibrates root XY/yaw/ground |
| Pico right controller | Stick click: start episode; `A`: save; `B`: discard |
| Collector terminal | `s`: start; `q`: save; `d`: discard; `exit`: quit |

For optional recording, complete the [head driver and servo setup](../../collector/sonic/head/README.md), then start the camera and head hold on the camera host, followed by the
collector on the PC, both **from the repository root**:

```bash
# Camera host (separate terminal); HGPT's default profile expects 60 Hz images:
CAMERA_FPS=60 collector/sonic/scripts/run_camera_server.sh --head-motion-approved

# PC terminal 3; replace the example robot IP with your camera host:
HGPT_CAMERA_ENDPOINT=tcp://192.168.123.164:5560 \
collector/humanoid_gpt/scripts/run_collector_pc.sh --task-name example_task
```

The collector's default camera endpoint is local (`tcp://127.0.0.1:5560`);
override it for a remote camera server. See the
[HGPT Collector README](../../collector/humanoid_gpt/README.md) for the 20 Hz
recording profile, saved fields, and startup checks.

### Offline references and startup behavior

Offline NPZ requires
`qpos(T,36)` and optionally accepts `left_wuji_qpos(T,20)`,
`right_wuji_qpos(T,20)` and per-side valid arrays. Source rates such as 20 Hz
are resampled to 50 Hz with quaternion SLERP and no interpolation across invalid
hand gaps. Mode 0 and completed sequences always publish invalid hands.

The Pose Manager waits indefinitely for the first Pico body frame and prints a
status line once per second. Pass `--no-pico-wait-forever` to restore the
15-second fail-fast startup timeout.

### Environment

Activate the existing Python 3.12 environment, then run the isolated setup.
GMR is pinned under `.deps/GMR`, XRoboToolkit is copied and built under
`.deps/xrobotoolkit_sdk`, and model weights are copied into ignored
`storage/ckpts/`.

```bash
conda activate h-gpt
scripts/setup_pico_env.sh

# Use another existing checkpoint directory when needed:
scripts/setup_pico_env.sh --checkpoint-source /path/to/old/Humanoid-GPT/storage/ckpts

python scripts/check_pico_env.py --service-mode auto
```

The setup requires network access only when it needs to install Python
packages or clone the pinned GMR revision. It never builds inside the
repository-level `third_party/xrobotoolkit_sdk`.

### Preview Pico -> GMR

Preview retargeting before loading either HumanoidGPT policy:

```bash
export LD_LIBRARY_PATH="$(pwd)/.deps/xrobotoolkit_sdk/lib:${LD_LIBRARY_PATH:-}"
python -m deploy.pico_preview

# External Robotics Service, no viewer, record the raw and retargeted stream:
python -m deploy.pico_preview \
  --service-mode external \
  --headless \
  --duration-s 60 \
  --record-path storage/pico_preview.npz
```

The NPZ recording contains `body_poses` with shape `(T, 24, 7)`,
`timestamps_ns`, and `qpos` with shape `(T, 36)`. Console statistics report
input/GMR rates, processing latency, frame age, invalid frames, and stale
events.

### Run HumanoidGPT

```bash
scripts/run_pico_sim.sh

# MuJoCo without its viewer; useful for a bounded smoke run:
scripts/run_pico_sim.sh --headless --max-steps 500 --no-visualize-retarget

# Connect to a Robotics Service that was started separately:
scripts/run_pico_sim.sh --pico-service-mode external
```

Focus the pygame controller window before using the keyboard:

- `0`: Walk policy.
- `1`: Pico online pose tracking.
- `2` and above: offline motions from `--track-dir`.
- `R`: reset MuJoCo in simulation and, in online mode, request a new Pico
  XY/yaw/foot-ground calibration; on the real robot it only recalibrates mode 1.
- Backtick: exit the simulation or real control loop.

The first fresh frame after entering mode 1 or pressing `R` defines pelvis
XY/yaw and maps the lower foot to ground `z=0`. Keep the intended starting
pose for that frame. Later vertical motion is preserved, so sitting and
standing remain absolute GMR poses rather than being normalized away.

Mode 1 reads and converts the latest available reference every control cycle,
matching upstream HGPT. On the first cycle the new reference is used for both
`ref_curr` and `ref_next`; later cycles use the previous reference as
`ref_curr` and the latest reference as `ref_next`. Entering mode 1 or pressing
`R` resets policy/reference history and requests recalibration. There is no
reference-age gate, stale latch, static measured-pose hold, or `0` then `1`
re-arm requirement. Pico defaults to a 0 ms jitter buffer, while other mocap
sources retain their 30 ms default.

### PC-hosted G1 deployment

Install the real-robot dependencies after the Pico simulation setup. The
script copies the repository's existing Unitree Python SDK into ignored
`.deps/unitree_sdk2_python`, builds CycloneDDS under `.deps/cyclonedds`, and
keeps CPU ONNX Runtime by default:

```bash
conda activate h-gpt
scripts/setup_pico_real_env.sh --skip-pico-setup

# Optional NVIDIA/TensorRT provider instead of the default CPU provider:
scripts/setup_pico_real_env.sh --skip-pico-setup --with-tensorrt
```

Connect the PC and G1 by Ethernet, configure the robot-facing NIC in the G1
subnet, and suspend the robot for the initial tests. Start with the wrapper's
default debug mode: it subscribes to `LowState`, receives Pico, follows the
`Start`/`A` startup sequence, and runs policy inference, but never writes
`LowCmd`.

```bash
scripts/run_pico_real.sh \
  --net <robot_nic> \
  --pico-service-mode external
```

Real `--debug` also rejects `--enable-hand`: `HandCmd` has no debug publication
gate, so a read-only debug run must not initialize hand command output.

The wrapper checks the NIC, DDS/Unitree imports, CRC library, checkpoints, and
selected ONNX provider before launch. When the debug run is clean, explicitly
enable motor command publication:

```bash
scripts/run_pico_real.sh \
  --net <robot_nic> \
  --pico-service-mode external \
  --publish-lowcmd
```

`LowCmd.mode_machine` is initialized automatically from the first valid
`LowState.mode_machine` frame. No override is required. Before every DDS
`Write`, the runtime verifies that the live LowState value, the LowCmd object,
and the serialized CRC byte mirror still agree; a change or divergence aborts
before publication.

`--publish-lowcmd` also runs a 200-sample tracking-policy benchmark and refuses
to launch when p95 inference latency exceeds 15 ms. The shared launcher profile
defaults to TensorRT; install and check it first, or explicitly select
`--policy-provider cpu` for a CPU debug check. The wrapper owns `--real`,
`--debug`, and the internal publication gate so they cannot be bypassed through
forwarded arguments.

Before a non-debug controller is initialized, the runtime follows Unitree's
low-level startup sequence: it queries MotionSwitcher, releases any active
high-level owner such as `ai`, and confirms that `CheckMode` returns an empty
name. Release is bounded to ten attempts at 0.5-second intervals. Any RPC
failure or mode that remains active aborts fail-closed before LowCmd
publication. `L2 + R2` is still required to put the robot in debug/low-level
state, but its remote indication alone is not treated as proof that the
MotionSwitcher owner was released.

Keep the G1 remote in hand throughout the hardware test:

1. Suspend the robot and enter debug mode with `L2 + R2`.
2. Press `Start` to leave damping and move toward the default pose.
3. Press `A` to start the locomotion/tracking loop.
4. Use keyboard `0` for walk and `1` for Pico online pose.
5. Press `Select` for emergency stop/damping.

Only explicit operator stops (`Select`, keyboard kill, SIGINT/Ctrl-C, or
SIGTERM) enter the serialized damping path. If the control worker raises, the
loop stops and propagates the error; it does not enter a static measured-pose
hold or automatically publish damping.

The added PC-side reference-age, measured-state, `LowState` age, scheduler-gap,
and inference-deadline watchdogs have been removed. Wrong-shape/NaN/Inf
targets and inconsistent DDS/CRC/machine headers remain rejected as
wire-integrity errors.

Finite 29D policy motor targets are sent unchanged: the real path applies
neither a physical-position clamp nor a target slew limiter. Mode 1 continuously
uses the latest Pico/pose-manager reference without a stale-age latch.

Run the focused verification suite from this directory:

```bash
pytest tests
ruff check deploy/pico.py deploy/pico_preview.py deploy/retarget.py \
  deploy/play_track.py scripts/check_pico_env.py \
  scripts/check_pico_real_env.py tests
```

---

## 🤖 Real-Robot Deployment

Deployment on Unitree G1 is split into sub-modules under `deploy/` — start with
**[`deploy/DEPLOY.md`](deploy/DEPLOY.md)** for install / SDK setup, then:

```bash
# Simulation
python -m deploy.play_track --track-dir storage/test

# Real robot (debug/no LowCmd by default)
scripts/run_pico_real.sh --net <nic_name> --pico-service-mode external
```

- 🖥️ [`onboard_deploy/`](deploy/onboard_deploy/DEPLOY_ONBOARD.md) — on-board (Jetson Orin) deploy.
- 🖥️ `onboard_deploy_wo_GMR/` — on-board variant that streams retargeting from a host.
- ✋ [`brainco/`](deploy/brainco/BRAINCO.md) — BrainCo dexterous-hand tracking variant.

---

## 📁 Project Structure

```
Humanoid-GPT/
├── 📂 tracking/   # Inference core: constants, infer_utils, ONNX policy wrapper (policy.py),
│                  # keypoint conversion (convert_qpos2kpt.py) and tracking metrics
├── 📂 scripts/    # inference.py · eval_parallel.py · vis.py · app.py (gradio demo)
├── 📂 deploy/     # Real-robot deployment — see deploy/DEPLOY.md
│   ├── onboard_deploy/        # On-board (Jetson) SSH deployment
│   ├── onboard_deploy_wo_GMR/ # On-board variant with host-side retargeting
│   └── brainco/               # BrainCo dexterous-hand tracking variant
├── 📂 projects/   # Optional side modules
│   ├── hme/                  # Harmonic Motion Encoder (Periodic Autoencoder)
│   ├── gqs/                  # General Quality Selection (physics + diversity scoring)
│   └── tracking_transformer/ # Transformer tracking policy (inference / deploy)
├── 📂 utils/      # MuJoCo / MJX simulation, transforms, video rendering
└── 📂 storage/    # Assets, configs, sample trajectory, released checkpoints
```

---

## 📚 Citation

```bibtex
@article{humanoid-gpt26,
    title     = {Humanoid-GPT: Humanoid Generative Pre-Training for Zero-Shot Motion Tracking},
    author    = {Qi, Zekun and Chen, Xuchuan and others},
    journal   = {arXiv preprint arXiv:2606.03985},
    year      = {2026}
}
```

---

## 📄 License · Acknowledgments

Licensed under **Apache 2.0**. Built on top of [MuJoCo](https://mujoco.org/), [Brax](https://github.com/google/brax) and the [Unitree](https://www.unitree.com/) G1 platform.
