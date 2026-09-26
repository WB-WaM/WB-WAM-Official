# WB-WAM Real-Robot Deployment

Before deploying or operating a real robot, read the [safety disclaimer](../README.md#safety-disclaimer).

**Head servos are required:** complete [head driver installation, robot-side build and tests](../collector/sonic/head/README.md) before deployment/collection. From the PC repository root, run `collector/sonic/scripts/setup_head_servo.sh USER@ROBOT_IP`. During both inference and collection, use `run_camera_server.sh --head-motion-approved` on the robot to start head-pose holding together with the camera. The default is `HEAD_SERVO_MODE=raw`, holding `HEAD_JOINT0_ENCODER=3027` and `HEAD_JOINT1_ENCODER=1849` (encoder counts, not degrees). These values are provided in the environment example; no head calibration or initial teaching is required. Camera-only operation cannot establish head readiness after a servo failure.

[中文](./README_zh.md)

**Hardware compatibility:** This deployment supports only a Unitree G1 with [Wuji Hands](https://www.wuji.tech/en/hand) and the [G1 2-DOF camera head module (RealSense D455)](https://www.hifivebot.shop/products/g1-head-dual-degree-of-freedom-module); other hand or camera configurations are not supported. This page does not cover data collection. Unless marked as robot-side commands, run everything from the PC repository root. The PC requires Linux, a CUDA GPU, and `uv`.

## Hardware wiring and configuration

| Device / connection | Real-robot deployment |
| --- | --- |
| Unitree G1 ↔ PC | Ethernet; select the PC's robot-facing interface |
| Head camera → robot | USB; camera service runs on the robot |
| Left/right Wuji hands → robot | USB; hand service runs on the robot |

Prepare the robot SSH connection and PC robot-facing interface, then verify the head camera key, left/right Wuji serials and PC service endpoints. Policy deployment does not require PICO or MANUS; see the [collection guide](../collector/sonic/README.md) for additional devices and input checks.

**Default wiring:** Connect the robot to the PC by Ethernet. Both Wuji hands and the head camera connect directly to the **robot via USB**; their services also run on the robot. Copy the standalone [Wuji USB serial discovery program](../collector/sonic/scripts/discover_wuji_hands.py) to the robot and run it there with `python3`; the PC cannot enumerate robot-side USB devices over Ethernet. Left/right confirmation is required for deployment setup: keep both hands connected, reset both hands, then let the first thumb repeat slowly over a small range until its side is confirmed; only then start and independently confirm the second thumb. Write the confirmed USB serials into robot-local `collector/sonic/scripts/wuji_hand_server.env`. See [robot-side camera and Wuji services](#3-robot-side-camera-and-wuji-services) below for copying, identification, and configuration.

```text
Robot camera, body and hand state → PC WB-WAM → SONIC Encoder → PC SONIC → Robot
```

## 1. PC environment and hardware-free check

Check that `third_party/GMR` and `third_party/HumanoidArena` have been cloned. If either is missing, run this from the repository root:

```bash
git submodule update --init --recursive
```

```bash
bridge/scripts/setup_env.sh
cp -n bridge/.env.example bridge/.env
bridge/scripts/run_bridge.sh --config bridge/configs/deploy.yaml \
  --dry-run --mock-policy --fake-camera --fake-state
```

The last command needs neither model weights nor a robot.

Follow the [native SONIC deployment instructions](../tracker/sonic/docs/source/getting_started/installation_deploy.md) to configure this repository's `tracker/sonic/gear_sonic_deploy/` and build `g1_deploy_onnx_ref_vla`. Real deployment also requires the SONIC encoder, decoder, observation config, and planner assets.

## 2. Deployment configuration

Edit `bridge/.env` to set the WB-WAM checkpoint, SONIC encoder, text cache, robot addresses, and camera ID. See the [training guide](../training/README.md) for downloading the open-source weights, then set `WBWAM_BRIDGE_CHECKPOINT_PATH` to the actual checkpoint path. Each checkpoint must have its matching `config.yaml` and `dataset_stats.json`, which are found next to the weights by default. The SONIC encoder must be prepared separately.

```bash
cp -n collector/sonic/scripts/collector_pc.env.example \
  collector/sonic/scripts/collector_pc.env
```

The SONIC launcher still reads the robot network interface and native decoder/planner paths from `collector_pc.env`. In [deploy.yaml](configs/deploy.yaml), update `task.prompt` and check the image size and control settings. Relative paths in `bridge/.env` are resolved from the repository root, and exported environment variables take precedence.

After configuration, check the checkpoint's training config and statistics without loading model weights:

```bash
bridge/scripts/run_bridge.sh --config bridge/configs/deploy.yaml --check-config
```

## 3. Robot-side camera and Wuji services

**Default wiring:** PC ↔ Ethernet ↔ robot; both Wuji hands and the head camera connect to the robot by USB. Run USB serial discovery and camera/hand services on the robot. Ethernet does not expose these USB devices to the PC; use `remote_wuji_proxy` for this topology.

These services reuse code under `collector/sonic/`. For first-time setup, copy the complete service directory from the PC. Copying only the two launch scripts is insufficient:

```bash
ROBOT_SSH=USER@ROBOT_IP
ssh "$ROBOT_SSH" 'mkdir -p "$HOME/WB-WAM/collector/sonic"'
rsync -av --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='datasets/' --exclude='scripts/*.env' \
  collector/sonic/ "$ROBOT_SSH:WB-WAM/collector/sonic/"
ssh "$ROBOT_SSH" 'cd WB-WAM && \
  cp -n collector/sonic/scripts/camera_server.env.example collector/sonic/scripts/camera_server.env && \
  cp -n collector/sonic/scripts/wuji_hand_server.env.example collector/sonic/scripts/wuji_hand_server.env'
```

The copy above includes the standalone serial discovery program. **On the robot**, run it with system Python before installing the SDK:

```bash
cd ~/WB-WAM
python3 collector/sonic/scripts/discover_wuji_hands.py
```

Without arguments it reads Linux USB descriptors without opening the hand SDK or enabling motors; the explicit `--identify` mode below actuates the hands. It lists USB ports and **USB serials**, which are the identifiers required by `wujihandpy.Hand`; product serial numbers on labels may differ (see the [Wuji SDK serial-number guide](https://docs.wuji.tech/docs/en/wujihandpy/latest/tutorial/)). It recognizes USB IDs `0483:2000` and legacy `0483:7530`. If no devices appear, check that the program is running on the robot, hand power, USB cabling, and `lsusb` output.

Install a separate environment on the robot from `~/WB-WAM`. Do not copy the PC virtual environment or model weights:

```bash
cd ~/WB-WAM
sudo apt update
sudo apt install -y python3-venv libusb-1.0-0 usbutils curl
python3 -m venv .venv_robot
.venv_robot/bin/python -m pip install --upgrade pip
.venv_robot/bin/python -m pip install \
  'numpy==1.26.4' 'opencv-python-headless==4.11.0.86' \
  pyzmq pyrealsense2 wujihandpy
```

Set `COLLECTOR_PYTHON="$REPO_ROOT/.venv_robot/bin/python"` in both `camera_server.env` and `wuji_hand_server.env` on the robot. Check the camera port; set `PC_ZMQ_HOST` and the left/right hand serial numbers in the hand configuration. On first use, confirm that the RealSense and Wuji SDKs work and that USB permissions are correct. Do not expose the robot network to the Internet.

### Required: identify left/right by thumb motion with both hands connected

After installing the robot environment above, keep both hands connected by USB. Stop competing hand controllers, clear the hands' surroundings, and observe on site. Run the following on the robot; `--motion-approved` confirms readiness for the reset and thumb movements:

```bash
.venv_robot/bin/python collector/sonic/scripts/discover_wuji_hands.py \
  --identify --motion-approved --result ~/wuji-identification-01.json
```

The program slowly resets both hands to a neutral open pose within joint limits and prints `RESET DONE`. Thumb joint 1 (the second joint counted from the palm) on the first hand then **repeats slowly until confirmed before the second hand starts**. Travel is 30 degrees (about 0.524 rad), with 1.8 seconds each way and a 0.3-second hold at each end: about 4.2 seconds per cycle. Other joints retain their open targets. Reset means returning to this pose, not recalibration or clearing faults. **Ignore reset movements and observe the currently announced identification stage.**

Keep the identification terminal open and confirm each hand separately through the same process's standard input. Use the robot's own left/right perspective; when facing the robot, its left is on the observer's right:

1. After `HAND 1/2 AWAITING CONFIRMATION`, observe the repeating thumb and enter `confirm 1 left` or `confirm 1 right`, followed by Enter. If unsure, keep watching; no restart is needed.
2. The program finishes the current cycle, returns to neutral and disables the first hand before starting the second hand's loop. After `HAND 2/2 AWAITING CONFIRMATION`, observe again and enter `confirm 2 left` or `confirm 2 right`. **Observe and confirm both hands independently; do not pre-send the second answer or infer it from the first.** The confirmed sides must differ.

The result is saved only after the second confirmation, completion of the current cycle, and successful motor disable. Typing `stop`, pressing Ctrl+C, losing control input, or exceeding the default 300-second timeout for either stage aborts identification and attempts to disable motors, without saving a completed result or advancing automatically. `--stage-timeout` adjusts the limit up to 600 seconds; use a new result filename for a retry. Over SSH, keep an interactive terminal open, for example `ssh -tt <robot-SSH-address>`; do not use `ssh -n`, background execution or pre-piped confirmations.

After both confirmations and successful process exit, validate the result and save the robot-local configuration (this command sends no motion):

```bash
python3 collector/sonic/scripts/discover_wuji_hands.py \
  --resolve ~/wuji-identification-01.json \
  --write-env collector/sonic/scripts/wuji_hand_server.env
```

The script checks the connected pair against both independent confirmations, updates only `LEFT_WUJI_SERIAL` and `RIGHT_WUJI_SERIAL`, preserves other settings, and verifies shell syntax and file readback without printing serials. If the env file is missing, it is created from the adjacent `wuji_hand_server.env.example`; configure the remaining host settings as described in this section. Omit `--write-env` to print assignments without editing a file. Deployment setup remains incomplete until both confirmations and configuration checks pass. Old results that only recorded motion order are rejected and require identification again. Do not infer sides from old results or USB enumeration order; motion failures, incomplete confirmations or changed devices must not produce a mapping or update configuration.

For a separate observation/tuning session after identification has exited, move both thumbs together without changing the confirmed mapping:

```bash
.venv_robot/bin/python collector/sonic/scripts/discover_wuji_hands.py \
  --tune-thumbs --motion-approved --thumb-joint 1 --amplitude-deg 30 --period 4.2
```

Each cycle reports measured angular travel for both hands and elapsed cycle time. Enter `stop` to disable motors, then restart with new `--amplitude-deg` and `--period` values. Amplitude is the total travel of one thumb joint from open to flexed, not that angle in each direction around neutral. Both hands' joint limits and the 0.3 rad/s speed limit still apply; excessive-speed combinations are rejected and require a longer period. This simultaneous test does not replace independent side identification.

Open two separate terminals on the robot:

```bash
collector/sonic/scripts/run_camera_server.sh --head-motion-approved
```

```bash
collector/sonic/scripts/run_wuji_hand_server.sh
```

The hand service enables the motors. Keep people and obstacles clear of the hands before starting it, and verify that both hands initialize successfully.

## 4. PC startup and operation

Terminal 1, start SONIC:

```bash
bridge/sonic/scripts/run_deploy_v4_pc.sh
```

Terminal 2, start the WB-WAM policy:

```bash
bridge/scripts/run_bridge.sh --config bridge/configs/deploy.yaml
```

In the policy terminal, use `4 → 1 → 1 → 4 → 1`:

- `4`, then `1`: warm up inference, then execute a newly generated action chunk.
- `1` while running: pause policy actions; Planner controls the body as both hands gradually open.
- `4` while paused: return to the initial pose and warm up again.
- `1` after the return completes: continue with a new action chunk. Do not resume execution directly from the paused state.

**Caution: the robot moves quickly when transitioning from stopping the policy (`1`) to returning to the initial pose (`4`). Before returning, ensure the entire travel path of both hands is clear of people and objects so neither hand collides with anything.**

`e` sends a software stop and exits; `q` returns to Planner idle and exits. Before executing, confirm that body and hand feedback keep updating, the workspace is clear, and a hardware E-stop is available. A keyboard stop is **not** a hardware E-stop.
