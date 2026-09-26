# Deployment Guide for Unitree G1

This directory contains the deployment pipeline of **Humanoid-GPT** for Unitree G1.
The same tracking inference stack is used in simulation and on hardware.
The normal hardware layout runs this process on a Linux control PC and uses
Unitree DDS over Ethernet; installing HumanoidGPT on the robot computer is not
required.

Main entry point:

```bash
python -m deploy.play_track
```

## Overview

The deployment stack supports:

- **Simulation mode**: walk control, online retargeting, and offline trajectory tracking in MuJoCo.
- **Real-robot mode**: low-level DDS control on Unitree G1 with shared observation/action computation.

The WB-WAM Pico path is PC-hosted end to end:

```text
Pico/XRoboToolkit 24 x 7 global body poses
  -> GMR (xrobot -> unitree_g1) -> G1 qpos36
  -> HumanoidGPT ONNX -> 29 motor targets
  -> Unitree DDS over the G1-facing Ethernet interface
```

Pico v1 is **body-only**: it does not synthesize hand commands, and
`--enable-hand` is rejected with `--mocap-type pico`. XRoboToolkit, GMR, and
policy inference all run in one PC-side process tree (the retarget worker uses
shared memory), so this path does not need ZMQ. A ZMQ bridge remains useful in
architectures such as SONIC where producers and consumers are intentionally
split across independent processes, environments, or hosts.

Core files:


| File              | Description                                                              |
| ----------------- | ------------------------------------------------------------------------ |
| `play_track.py`   | Unified runtime entry for simulation and real robot                      |
| `walk_policy.py`  | ONNX walk policy wrapper                                                 |
| `retarget.py`     | Online mocap retarget subprocess (Pico / PNLink / OptiTrack / Xsens)     |
| `real_robot.py`   | Low-level robot interface (IMU/joints readout and PD command publishing) |
| `hand_control.py` | Dex3-1 hand controller                                                   |
| `keyboard_cmd.py` | Keyboard UI for mode/velocity control                                    |
| `constants.py`    | Deploy constants (PD gains, motor IDs, DDS topics)                       |


## Installation

All commands below are executed from repository root.
For the WB-WAM Pico path, the guarded scripts automate the sections below and
keep dependency sources/builds in ignored `.deps/`:

```bash
scripts/setup_pico_env.sh
scripts/setup_pico_real_env.sh --skip-pico-setup
python scripts/check_pico_real_env.py --policy-provider cpu
```

`setup_pico_real_env.sh` reuses the Unitree Python SDK already vendored by
SONIC, but copies it into HumanoidGPT `.deps/` before editable installation.
Its first run needs network access for CycloneDDS 0.10.x and any missing Python
packages.


### 1. Base environment for Humanoid-GPT

```bash
conda create -n h-gpt python=3.12 -y
conda activate h-gpt
pip install -e .
```

### 2. Download third-party libraries

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1ArtgwKxVHXTO4KXsKXPLdhy1yAtKKnz9 -O thirdparty.zip
unzip thirdparty.zip
rm thirdparty.zip
```

Alternatively, download `[thirdparty.zip](https://drive.google.com/file/d/1bfgFhrv6tfuDOkt11AOJAO2IHTRXlYey/view?usp=sharing)` manually and extract it to the repository root so that a `thirdparty/` folder appears at the top level.

After extraction, the directory should look like:

```
thirdparty/
├── GMR-galbot/          # Online retargeting (Section 3)
├── noitom/              # PNLink mocap backend (Section 3)
├── cyclonedds/          # DDS middleware for real-robot communication (Section 4)
└── unitree_sdk2_python/ # Unitree G1 SDK Python bindings (Section 4)
```

### 3. Online retargeting dependencies

```bash
pip install -e thirdparty/GMR-galbot
pip install -e thirdparty/noitom
```

`noitom` is required for the default `pnlink` mocap backend.
If only OptiTrack is used, run with `--mocap-type optitrack`.

### 4. Real-robot dependencies

Build CycloneDDS:

```bash
cd thirdparty/cyclonedds
mkdir -p build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
cd ../../..
```

Install Unitree SDK Python:

```bash
export CYCLONEDDS_HOME="$PWD/thirdparty/cyclonedds/install"
pip install -e thirdparty/unitree_sdk2_python
```

### 5. TensorRT acceleration (optional)

PC deployment defaults to CPU ONNX Runtime and validates inference latency
before LowCmd publication. TensorRT is optional for hosts that have a working
NVIDIA/CUDA stack:

```bash
pip uninstall onnxruntime -y
pip install onnxruntime-gpu tensorrt-cu12
```

You may need to add this into bashrc:

```bash
# Expose TensorRT / NVIDIA runtime libs from the h-gpt env to the dynamic linker
for _d in "$HOME/miniconda3/envs/h-gpt/lib"/python*/site-packages/{tensorrt_libs,nvidia/*/lib}; do
  [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
unset _d
```

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

`TensorrtExecutionProvider` must appear in the provider list.

The WB-WAM setup equivalent is:

```bash
scripts/setup_pico_real_env.sh --skip-pico-setup --with-tensorrt
```

## Robot Bring-Up (Real Mode)

For initial tests, suspend the robot for safety.

1. Power on the battery (short press, then long press for ~2 s).
2. After head indicator stabilization, enter debug mode via `L2 + R2`.
3. Optionally verify mode switching with `L2 + A` (position) and `L2 + B` (damping).

Network setup:

1. Connect host and robot via Ethernet.
2. Configure host IP in the same subnet as the robot.
3. Verify connectivity: `ping <robot_ip>`.
4. Find network interface name:

```bash
ifconfig
# or
ip addr
```

Pass the interface name to `--net`.

## Running

### Simulation

```bash
python -m deploy.play_track
python -m deploy.play_track --no-mocap
python -m deploy.play_track --track-dir storage/test
python -m deploy.play_track --track-dir storage/test/human_walking_50Hz_29dof.npz
```

### Real robot

The guarded Pico wrapper defaults to `--debug`, so it never publishes LowCmd:

```bash
scripts/run_pico_real.sh \
  --net <nic_name> \
  --pico-service-mode external
```

Debug mode still subscribes to `LowState`, receives Pico frames, performs GMR
and policy inference, and follows the `Start`/`A` startup sequence. The final
DDS `LowCmd` write is disabled. Use this run to verify the complete PC-side
chain with the robot suspended. Real `--debug` rejects `--enable-hand` because
`HandCmd` has no debug publication gate.

`G1_VERSION` selects the robot asset and reviewed limits; it does not force the
DDS command header. At startup, `LowCmd.mode_machine` is copied directly from
the first valid `LowState.mode_machine` frame.

Only after the suspended/debug test succeeds, explicitly enable publication:

```bash
scripts/run_pico_real.sh \
  --net <nic_name> \
  --pico-service-mode external \
  --publish-lowcmd
```

No mode-machine override is required. Before every DDS `Write`, the runtime
checks that the live LowState value, the LowCmd object, and the serialized CRC
byte mirror still match. A live value change or cached-header divergence stops
publication before `Write`.

The publication path runs an environment/network check and a 200-sample ONNX
latency benchmark before launch. `--publish-lowcmd` maps to the internal
`--allow-real-publish` gate; direct non-debug real mode is rejected without
that gate. Optional TensorRT use is selected with
`--policy-provider tensorrt`. CPU is the default and is a supported real
deployment provider; its preflight requires tracking-policy p95 latency no
greater than 15 ms before a 50 Hz publication run is allowed.

Before initializing a non-debug LowCmd controller, the runtime follows the
Unitree low-level example: it calls MotionSwitcher `CheckMode`, releases any
active high-level owner such as `ai`, and confirms that the returned name is
empty. Release is bounded to ten attempts at 0.5-second intervals. A failed
RPC or a mode that remains active aborts fail-closed. Enter G1 debug/low-level
state with `L2 + R2` before using `--publish-lowcmd`; the remote indication is
not used as a substitute for the MotionSwitcher RPC check.

Other upstream mocap examples can still use the entry point directly in
read-only debug mode:

```bash
python -m deploy.play_track --real --debug --net <nic_name>
python -m deploy.play_track --real --debug --net <nic_name> \
  --mocap-type optitrack \
  --server-ip <server_ip> --client-ip <client_ip>
```

## Control Interface

### Keyboard control (GUI)


| Key     | Function                                           |
| ------- | -------------------------------------------------- |
| `0`     | Walk mode                                          |
| `1`     | Online retarget mode                               |
| `2`-`9` | Offline trajectory modes (sorted from `track_dir`) |
| `W/S`   | Linear velocity x (+/-)                            |
| `A/D`   | Linear velocity y (+/-)                            |
| `Q/E`   | Yaw rate (+/-)                                     |
| `R`     | Reset simulation; in mode 1 request Pico recalibration |
| Backtick | Exit the simulation or real control loop          |


Mode keys are single-character digits; in practice, keep offline trajectories within modes `2..9`.

In mode 1, entry or `R` resets policy/reference history and establishes Pico
pelvis XY/yaw and foot-ground calibration. Each control cycle reads the latest
reference; the first reference is used as both current and next, then later
cycles pair the previous and latest references. There is no reference-age latch
or explicit re-arm sequence.

### Remote controller sequence (real robot)

1. `Start`: damping to default posture.
2. `A`: enter locomotion/tracking loop.
3. `Select`: emergency stop and return to damping.

For the first LowCmd test, suspend the robot, keep the remote in hand, and
enter robot debug mode with `L2 + R2` before starting the wrapper. The program
waits for `Start`, moves to the default pose over two seconds, then waits for
`A`. Only explicit operator stops (`Select`, keyboard kill, SIGINT/Ctrl-C, or
SIGTERM) publish damping. If the control worker raises, publication stops and
the error is propagated; no static measured-pose hold or automatic damping is
started.

## Real-Robot Runtime Guards

The added PC-side reference-age, measured-state, `LowState` age, scheduler-gap,
and inference-deadline watchdogs have been removed. Mode 1 reads the latest
Pico/pose-manager reference every control cycle and uses upstream HGPT
previous/latest reference pairing, without a stale latch or re-arm gate.

Wrong-shape/NaN/Inf LowCmd targets and inconsistent DDS/CRC/machine headers
remain rejected because they are transport-integrity requirements.

## Main CLI Arguments


| Argument               | Default                            | Meaning                                        |
| ---------------------- | ---------------------------------- | ---------------------------------------------- |
| `--real`               | `False`                            | Enable real-robot mode                         |
| `--net`                | `enx6c1ff76e8ef5`                  | DDS network interface; pass the actual G1 NIC  |
| `--freq`               | `50`                               | Control frequency (Hz)                         |
| `--onnx-walk`          | `storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx` | Walk policy path |
| `--onnx-track`         | `storage/ckpts/pns_wo_priv216.onnx` | Tracking policy path                          |
| `--policy-type`        | `mlp`                              | Policy architecture (`mlp`)                    |
| `--track-dir`          | `storage/test`                     | Offline trajectory folder or single `.npz`     |
| `--no-mocap`           | `False`                            | Disable online mocap in simulation             |
| `--mocap-type`         | `pnlink`                           | `pico`, `pnlink`, `optitrack`, or `xsens`      |
| `--server-ip`          | `169.254.117.205`                  | Mocap server IP                                |
| `--client-ip`          | `169.254.117.206`                  | Mocap client IP                                |
| `--human-height`       | `1.7`                              | Retargeting height prior                       |
| `--visualize-retarget` | `True`                             | Enable retarget visualization process          |
| `--enable-hand`        | `False`                            | Dex3-1 control; rejected for Pico and real debug |
| `--debug`              | `False`                            | Real mode without LowCmd/HandCmd publication   |
| `--allow-real-publish` | `False`                            | Explicit non-debug LowCmd publication gate     |
| `--real-policy-provider` | `cpu`                            | Real policy provider: `cpu` or `tensorrt`      |
| `--pico-service-mode`  | `auto`                             | Pico Robotics Service: `auto` or `external`    |
