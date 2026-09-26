# HumanoidGPT Data Collection

Before deploying or operating a real robot, read the [safety disclaimer](../../README.md#safety-disclaimer).

**Head servos are required:** complete [head driver installation, robot-side build and tests](../sonic/head/README.md) before deployment/collection. From the PC repository root, run `collector/sonic/scripts/setup_head_servo.sh USER@ROBOT_IP`. During both inference and collection, use `run_camera_server.sh --head-motion-approved` on the robot to start head-pose holding together with the camera. The default is `HEAD_SERVO_MODE=raw`, holding `HEAD_JOINT0_ENCODER=3027` and `HEAD_JOINT1_ENCODER=1849` (encoder counts, not degrees). These values are provided in the environment example; no head calibration or initial teaching is required. Camera-only operation cannot establish head readiness after a servo failure.

[中文](./README_zh.md)

This page covers collecting and converting HGPT teleoperation data, not model deployment. Before starting a real robot, complete the [HGPT environment and safety checks](../../tracker/humanoid_gpt/README.md#wb-wam-teleoperation-and-collection-quickstart). Unless noted otherwise, run commands from the repository root; use a separate terminal for each long-running service.

```text
PICO / MANUS → Pose Manager ──reference action──→ HGPT controller → G1
                    │                              │
                    └──Wuji target──→ Wuji service → hands
                                                   ↓
Camera RGB/depth ─────────────────────────────→ Collector ← robot state, actions and hand feedback
                                                   ↓
                                           20 Hz raw episodes
                                                   ↓
                                         LeRobot v3 conversion
```

## Configuration

Copy the local configuration and set the robot interface, robot-side hand feedback address, and camera address:

```bash
cp -n collector/humanoid_gpt/scripts/hgpt_collection.env.example \
  collector/humanoid_gpt/scripts/hgpt_collection.env
python collector/humanoid_gpt/scripts/check_env.py
```

The three PC launchers share `hgpt_collection.env`. Defaults are MANUS hand input, an external PICO Service, TensorRT, and local pose `:5556` and state/action `:5558` endpoints. The default camera endpoint is `tcp://127.0.0.1:5560`. If the camera service runs on the robot, use `tcp://ROBOT_IP:5560` or forward the robot port to the PC. Exported `HGPT_*` variables override template defaults; explicit CLI options take precedence. Do not commit a local configuration containing real addresses.

## Start collection

Start the Wuji feedback and camera services on the robot first. Then start the Pose Manager, HGPT controller, and Collector on the PC. Run each PC process in its own terminal:

```bash
# Robot side, from the repository root; the default profile needs a 60 FPS camera source.
collector/sonic/scripts/run_wuji_hand_server.sh
CAMERA_FPS=60 collector/sonic/scripts/run_camera_server.sh --head-motion-approved

# PC terminal 1, from the repository root.
cd tracker/humanoid_gpt
scripts/run_hgpt_pose_manager.sh \
  --hand-source manus --pico-service-mode external --publish-wuji-hand

# PC terminal 2, enter tracker/humanoid_gpt/ from the repository root.
cd tracker/humanoid_gpt
scripts/run_pico_real.sh \
  --net ROBOT_INTERFACE --pose-endpoint tcp://127.0.0.1:5556 \
  --state-action-bind tcp://127.0.0.1:5558 \
  --policy-provider tensorrt --publish-lowcmd

# PC terminal 3, from the repository root.
collector/humanoid_gpt/scripts/run_collector_pc.sh --task-name example_task
```

`--publish-lowcmd` enables real-robot body commands. Omitting it does **not** automatically disable Wuji hand commands; disable hand publication separately for a no-actuation check. Put the robot in a safe debug state first, keep the remote in the operator's hand, and have an E-stop ready. Follow the HGPT safety guide above for the exact enable sequence.

On the PICO right controller, press the stick to start an episode, `A` to save, and `B` to discard. In the Collector terminal, use `s` to start, `q` to save, `d` to discard, and `exit` to quit. In the Pose Manager window, `0` selects walking, `1` selects online tracking, and `2+` selects offline motions. Save/discard commands trigger only on rising edges; pressing `A+B` together is ignored.

## Capture rate and raw data

Defaults are `--capture-fps 20 --camera-fps 60 --downsample-method auto`. At 20 Hz, `auto` selects interpolation: it writes one row every 50 ms and waits up to 40 ms for bracketing 50 Hz telemetry. Continuous values are linearly interpolated, while quaternions use SLERP. Camera images are not interpolated; the nearest real frame by PC receive time is selected. Joint velocities are recomputed from aligned 20 Hz body joint positions. The Collector adds no EMA; Pose Manager smoothing of the PICO/GMR reference still affects the collected reference.

To use the original 50 Hz latest-sample mode explicitly:

```bash
collector/humanoid_gpt/scripts/run_collector_pc.sh \
  --task-name example_task --capture-fps 50
```

At 50 Hz, `auto` selects `causal_latest`, **not** the 20 Hz interpolation mode. The LeRobot converter below currently accepts only raw data with `capture_fps=20`; it cannot convert 50 Hz episodes directly.

Raw episodes are written by default under `datasets/humanoid_gpt/<task_name>/episode_*/`. Each contains `data.json`, `motion.npz`, RGB images, and optional depth. Robot state, measured hand feedback, hand targets, HGPT references, and controller actions are retained. The root position/orientation in `motion.npz` is the PICO/GMR reference, not a measurement of the robot's world-frame root pose.

## Convert to LeRobot training data

The converter uses local helpers in `collector/humanoid_gpt/processing/` and does not require the legacy `datasets/preprocess/` code. The `datasets/` paths below are local data input/output locations.

Before conversion, replace placeholder task text such as `example_task` with a real task description. For one 20 Hz task, run a source check before converting:

```bash
python collector/humanoid_gpt/scripts/convert_hgpt_to_lerobot_v3.py \
  --source-root datasets/humanoid_gpt/YOUR_TASK \
  --task 'Describe the task here.' --dry-run

python collector/humanoid_gpt/scripts/convert_hgpt_to_lerobot_v3.py \
  --source-root datasets/humanoid_gpt/YOUR_TASK \
  --task 'Describe the task here.' \
  --output-root datasets/humanoid_gpt_lerobot_v3_20hz
```

The result uses LeRobot v3 file organization. Each row's image and state are from time `t`, while its action is from the next frame `t+1`; do not shift actions again in the training loader. **This is not the same layout as the released Pico/Real datasets.** The HGPT converter writes 146D `states` and 136D `action`, retains an additional HGPT reference `qpos36`, and provides both HGPT-reference and physical-robot action views. Select the corresponding data configuration; do not reuse the Pico/Real configuration for 110D `observation.state`. See the [converter](scripts/convert_hgpt_to_lerobot_v3.py) for arguments and field definitions.

## Startup checks and troubleshooting

At startup, the Collector checks that the output directory supports writes, sync, rename, and read-back. The first start request in a process also checks body/hand training data, reference validity, and source freshness. If this check fails, it prints the reasons and stays idle; fix the inputs and trigger start again. Missing camera frames also prevent frame writes. A failed save enters `save_failed` and preserves the current episode for retry or manual recovery.

```bash
python collector/humanoid_gpt/scripts/check_env.py
pytest tracker/humanoid_gpt/tests
ruff check collector/humanoid_gpt tracker/humanoid_gpt
```
