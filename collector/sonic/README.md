# SONIC/PICO Data Collection

Before deploying or operating a real robot, read the [safety disclaimer](../../README.md#safety-disclaimer).

**Head servos are required:** complete [head driver installation, robot-side build and tests](head/README.md) before deployment/collection. From the PC repository root, run `collector/sonic/scripts/setup_head_servo.sh USER@ROBOT_IP`. During both inference and collection, use `run_camera_server.sh --head-motion-approved` on the robot to start head-pose holding together with the camera. The default is `HEAD_SERVO_MODE=raw`, holding `HEAD_JOINT0_ENCODER=3027` and `HEAD_JOINT1_ENCODER=1849` (encoder counts, not degrees). These values are provided in the environment example; no head calibration or initial teaching is required. Camera-only operation cannot establish head readiness after a servo failure.

This page covers collecting synchronized PICO teleoperation, camera, and Wuji hand data. It does not start WB-WAM model inference. Unless marked as robot-side commands, run everything from the PC repository root. Each long-running service needs its own terminal.

## Hardware wiring and configuration

| Device / connection | SONIC/PICO collection |
| --- | --- |
| Unitree G1 ↔ PC | Ethernet; select the PC's robot-facing interface |
| Head camera → robot | USB; camera service runs on the robot |
| Left/right Wuji hands → robot | USB; hand service runs on the robot |
| PICO ↔ PC | **Ethernet by default**, through a compatible headset Ethernet adapter and PC port / wired LAN |
| MANUS gloves → wireless receiver → PC | **USB wireless receiver plugged into the PC**; gloves paired to the receiver |

Collection reuses the deployment hardware checks for the robot, camera and Wuji hands, and adds PICO and MANUS. It does not require the WB-WAM policy environment or model weights. Record the **PICO wired IP** and the **PC IP reachable from PICO**. There is no universal PICO IP: read the actual headset network details and verify the wired route. Sections 1A and 1B cover IP configuration, headset setup, PICO/MANUS probe commands and glove calibration.

**Default wiring:** Connect the robot to the PC by Ethernet. Both Wuji hands and the head camera connect directly to the **robot via USB**; their services also run on the robot. Copy the standalone [Wuji USB serial discovery program](scripts/discover_wuji_hands.py) to the robot and run it there with `python3`; the PC cannot enumerate robot-side USB devices over Ethernet. Left/right confirmation is required for collection setup: keep both hands connected, reset both hands, then let the first thumb repeat slowly over a small range until its side is confirmed; only then start and independently confirm the second thumb. Write the confirmed USB serials into robot-local `collector/sonic/scripts/wuji_hand_server.env`. See the [deployment guide](../../bridge/README.md) for copying, identification, and configuration. Collection additionally connects PICO to the PC by Ethernet and plugs the MANUS USB wireless receiver into the PC; sections 1A and 1B cover configuration and required data checks.

```text
PICO / hand input → PC Pose Manager → PC SONIC low-level control → Robot
                             └────────────────→ Robot Wuji service
Camera, body and hand feedback ─────────────────→ PC Collector → episode
```

## 1. Configure the PC

Install the collection environment and create the local configuration from its template:

```bash
scripts/env/setup_envs.sh teleop
cp -n collector/sonic/scripts/collector_pc.env.example \
  collector/sonic/scripts/collector_pc.env
```

The teleop setup builds the MANUS Python binding for its Python interpreter; a C++ compiler, `make`, and ncurses development files must be available. The example configuration selects `HAND_CONTROL_MODE=manus` by default.

In `collector_pc.env`, set the task name `TASK_NAME`, output directory `OUTPUT_ROOT`, robot interface `ROBOT_INTERFACE`, camera `CAMERA_HOST/CAMERA_PORT`, hand feedback address, `PC_ZMQ_HOST`, and `HAND_CONTROL_MODE`. PICO uses Ethernet by default and must reach `XR_LISTEN`; set `XR_VIDEO_HOST` to its verified wired IP. The PC-local pose/body connections use ports 5556/5558. Match the camera and hand service addresses to the robot-side configuration. Do not commit a machine-specific `.env`.

## 1A. Wired PICO setup

Connect PICO to the PC using a compatible headset Ethernet adapter and cable, or through the same wired LAN. Plug the MANUS wireless receiver into the PC by USB. The robot uses its wired PC link; the camera and both Wuji hands remain connected to the robot by USB. Collection reuses the [deployment guide's](../../bridge/README.md) network, camera and required Wuji side-identification checks, without requiring the WB-WAM policy environment or model weights.

1. **Collect actual addresses.** Read PICO's Ethernet IP from the headset network details / XRoboToolkit Network panel. On the PC run `ip -br -4 addr` and `ip route get <PICO_IP>`; verify the PICO-facing wired interface and use the route's `src` as the PC address for that link. If Wi-Fi is also active, verify the Ethernet address rather than trusting a displayed Wi-Fi address. An existing ADB connection can inspect `adb shell ip -4 addr show`. `ping -c 3 <PICO_IP>` is a connectivity check, not proof of valid tracking.
2. **Configure a direct link if needed.** PC and PICO need different addresses in the same subnet. Use existing DHCP on a wired LAN. A direct cable alone does not provide DHCP. On a NetworkManager PC with an otherwise unconfigured dedicated PICO interface, inspect existing routes first, then create a separate shared connection (replace the interface placeholder):

   ```bash
   nmcli connection add type ethernet ifname '<PICO_INTERFACE>' \
     con-name wbwam-pico ipv4.method shared ipv6.method disabled
   nmcli connection up wbwam-pico
   nmcli -g IP4.ADDRESS device show '<PICO_INTERFACE>'
   ```

   Set the headset to automatic addressing / DHCP and read its assigned IP. Reuse working profiles rather than duplicating them; do not change the robot-facing interface. If static addressing is needed, choose nonconflicting addresses/masks based on the actual network and use the headset's supported Ethernet settings. Screenshot addresses are examples only. Separate robot/PICO links can use separate subnets; verify each route.
3. **Install and connect XRoboToolkit.** Follow the [XRoboToolkit setup](../../tracker/sonic/docs/source/getting_started/vr_teleop_setup.md) for the matching PC OS/architecture service and repository-used PICO app. Enable headset developer mode and install the APK. Internet/Wi-Fi may be used temporarily for download; collection defaults to Ethernet. Start the installed PC service (standard entry point `/opt/apps/roboticsservice/runService.sh`, after checking it exists). In the headset app set `PC Service` to the **PC's PICO-facing wired IP**, select `Enter` / `Reconnect`, and confirm `WORKING`.
4. **Set tracking.** Pair both controllers and ankle trackers and complete the headset's body calibration. Enable `Head`, `Controller`, and `Send`; select `Full body` for `Pico Motion Tracker`. MANUS provides finger data by default. With the headset worn, run `.venv_teleop/bin/python collector/sonic/scripts/probe_pico.py`: require valid headset/controller/24-joint body poses and advancing timestamps. The probe does not start SONIC or publish actions.
5. **Configure the PC and check video.** Edit local `collector/sonic/scripts/collector_pc.env`:

   | Field / location | Value |
   | --- | --- |
   | PICO app `PC Service` | PC IP on the PICO link, not PICO's own IP or the robot IP |
   | `XR_VIDEO_HOST` | PICO's verified Ethernet IP for video return |
   | `XR_LISTEN` | `0.0.0.0:13579` by default: PC camera-request listener; use the PC's actual IP as the remote destination |
   | `PC_ZMQ_HOST` | PC IP reachable from the robot, which can differ from the PICO-facing IP |
   | `CAMERA_HOST` / `ROBOT_HAND_HOST` | Robot-side camera / hand service host addresses |

   `XR_LISTEN` belongs to the collector's camera-request service; it does not replace XRoboToolkit PC Service. Once the collector is running, enable Remote Vision / Camera Listen in the headset, inspect `OPEN_CAMERA` / `stream_ip` logs for the PICO wired address, and confirm a visible headset image. Controls depend on the installed client. Update `XR_VIDEO_HOST` after network changes.

## 1B. Automatic MANUS USB receiver and glove checks

**Required for collection:** before collecting data with MANUS, run the full left/right glove-data check below, both during initial setup and for subsequent collection sessions. A full check already passed in the current session with unchanged hardware can be reused; repeat after receiver/glove reconnection, power cycling, re-pairing, or data faults. USB detection, a successful SDK import, or a previous session's pass is insufficient.

Plug the wireless receiver into a **PC USB port**, power both MANUS gloves, and pair them to that receiver. First run:

```bash
python3 collector/sonic/scripts/probe_manus.py --usb-only
```

This lists candidates with the repository's MANUS vendor ID `3325`, their USB paths and available identifiers. USB order does not identify sides. If absent, check the receiver is on the PC and inspect `lsusb`. For permission failures, inspect the repository's `tracker/sonic/decoupled_wbc/docker/70-manus-hid.rules`; install it under `/etc/udev/rules.d/` and reload udev when needed as part of setup.

After `scripts/env/setup_envs.sh teleop`, stop competing MANUS consumers, wear both gloves and gently move both hands. Run:

```bash
.venv_teleop/bin/python collector/sonic/scripts/probe_manus.py --duration-s 12
```

Require exit code 0, `changing skeleton data received` for both sides with distinct glove IDs, and `PASS: both glove streams detected`. Only then are pairing/connection and live data verified; operator calibration remains a separate check. On failure, do not start collection: check power, pairing and permissions, then move both gloves and retry. Do not skip the check.

It uses the repository's `ManusServer` Integrated SDK to discover left/right glove IDs and requires changing valid skeleton data from both sides, then shuts down the SDK. It sends no robot commands or vibration and does not load/change calibration. A USB receiver alone, one glove, or repeated cached data cannot pass. See [MANUS setup](https://docs.manus-meta.com/3.1.0/Plugins/SDK/getting%20started/) for pairing and calibration; address actual license or pairing errors separately from USB detection.

Then set `HAND_CONTROL_MODE=manus`. Sides come from SDK glove IDs; the USB wireless receiver needs no configured IP, and its identifier must not go into Wuji serial fields. Set `MANUS_LEFT_CALIBRATION_FILE` and `MANUS_RIGHT_CALIBRATION_FILE` to the current operator's matching `.mcal` files and keep `MANUS_LOAD_CALIBRATION=1`. If missing, calibrate/save each side in the MANUS SDK Client first. Automatic discovery is not calibration. `save_manus_calibration.sh left/right <source.mcal>` copies files but overwrites the corresponding project files, so inspect existing files first. Do not use the command-publishing `run_manus_hand_only_test.sh` for receiver discovery.

## 2. Configure robot-side camera and Wuji services

Default wiring is PC ↔ Ethernet ↔ robot, with both Wuji hands and the head camera connected directly to the robot by USB. Copy and run `scripts/discover_wuji_hands.py` on the robot to identify the hand USB serials; the PC cannot enumerate them over Ethernet. After the copy below, run `python3 collector/sonic/scripts/discover_wuji_hands.py` from the robot's `~/WB-WAM`. Follow the [deployment guide](../../bridge/README.md#3-robot-side-camera-and-wuji-services) to identify left/right with both hands connected: after reset, observe the first thumb repeating slowly and confirm its side before the second thumb starts; independently confirm the second hand, resolve the serial mapping, and fill the robot-local `wuji_hand_server.env`.

For first-time setup, copy the complete service directory from the PC to the robot; copying only the launch scripts is insufficient. The robot runs only camera and Wuji services, so it does not need the PC virtual environment or model weights. This example uses `~/WB-WAM` on the robot:

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

Install a separate Python environment on the robot; do not copy the PC virtual environment:

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

Set `COLLECTOR_PYTHON="$REPO_ROOT/.venv_robot/bin/python"` in both `camera_server.env` and `wuji_hand_server.env` on the robot. Check the camera port, and set `PC_ZMQ_HOST` and the left/right hand serial numbers. The camera and Wuji SDKs and USB permissions must work; on first setup, configure device access as required by each SDK and verify the devices. Expose ports 5560 (camera), 5559 (hand feedback), and 5556 (PC hand commands) only on a trusted robot network. The camera service defaults to a physical capture rate of **60 FPS**, independent of the Collector recording rate below.

## 3. Start collection

First print the multi-terminal startup sequence for the current settings:

```bash
collector/sonic/scripts/run_data_collection_flow.sh print
```

Start the services in the printed order, each in its own terminal. Robot terminal 1:

```bash
collector/sonic/scripts/run_camera_server.sh --head-motion-approved
```

Robot terminal 2:

```bash
collector/sonic/scripts/run_wuji_hand_server.sh
```

PC terminal 1, start SONIC low-level control for teleoperation:

```bash
tracker/sonic/gear_sonic_deploy/scripts/run_deploy_pc.sh
```

PC terminal 2, start the PICO manager:

```bash
collector/sonic/scripts/run_pico_manager.sh
```

PC terminal 3, start recording:

```bash
collector/sonic/scripts/run_collector_pc.sh
```

`run_deploy_pc.sh` is the SONIC low-level control launcher used for collection, **not** the WB-WAM model deployment launcher. The Wuji service enables its motors. Before recording, verify camera, body, and both hand feedback streams, clear the workspace, and have a hardware E-stop ready.

On the PICO right controller, press the stick to start an episode, `A` to save, or `B` to discard. The PC Collector terminal also accepts `s` to start, `q` to save, `d` to discard, and `exit` to quit.

## 4. Sampling and input options

The default recording rate is **20 Hz**. Available capture modes are:

| Collector settings | Recording behavior |
| --- | --- |
| `CAPTURE_MODE=collector_timer`, `CAPTURE_FPS=20` (default; also supports 30) | Sample on the Collector clock, then merge offline using nearest/held values. |
| `CAPTURE_MODE=official_latest`, `CAPTURE_FPS=50` (optional) | Latch the latest received samples every 20 ms; select camera frames causally and record image age, reuse, and stale status. |

Start the default 20 Hz mode:

```bash
# Default: 20 Hz collector_timer.
collector/sonic/scripts/run_collector_pc.sh
```

When `CAMERA_FPS` is empty, the PC requests 20/30 FPS for 20/30 Hz recording and 60 FPS for 50 Hz recording. **This does not change the robot camera service's own 60 FPS default.** With `DEFER_DEPTH_COMPRESSION=1`, saving an episode first writes raw depth frames; pending frames are compressed on exit or Ctrl+C. Set it to `0` for per-episode compression. The old `auto` and `encoder_clock_exact` modes are no longer accepted.

PICO Camera Listen uses GStreamer by default. If it is not installed on the PC, install the system plugins and Python bindings:

```bash
sudo apt update
sudo apt install -y pkg-config libcairo2-dev libgirepository1.0-dev \
  gobject-introspection gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
uv pip install --python .venv_teleop/bin/python pycairo PyGObject
```

Select the hand input with `HAND_CONTROL_MODE`: `manus`, `wuji_glove`, `gesture_wuji`, or `binary_trigger`. Each produces a 20D Wuji target for each hand. Use `collector/sonic/scripts/run_wuji_glove_probe.sh` to check the Wuji glove connection.

## 5. Export collected episodes for training

The CPU-only converter accepts **20 Hz** SONIC collector data and writes native
LeRobot v3 records compatible with the WB-WAM `real_archive` layout. Run offline
on the PC with Python 3.10–3.12; no robot SDK, GPU or model weights are required:

```bash
python3 -m venv .venv-process
.venv-process/bin/python -m pip install -r collector/sonic/processing/requirements.txt
.venv-process/bin/python collector/sonic/processing/convert_to_lerobot.py \
  --input /path/to/collected_task --output /path/to/new_archive
.venv-process/bin/python collector/sonic/processing/validate_lerobot.py \
  --root /path/to/new_archive
```

`--input` can also be a parent containing multiple tasks. The output is
`<archive>/<task>/record_XXXX/`; load an individual record, or use the archive
root in WB-WAM's post-training configuration. Original data is never overwritten.
Use `--dry-run` to preview selection, or `--resume` to verify and reuse completed
records with unchanged inputs/options. 30/50 Hz input is rejected, not resampled.

Each sample pairs the current RGB/state with next-frame action labels: measured
body joints/root, observed SONIC token and **teleoperation hand targets**, not
measured hand actions. Output contains 110D state, 136D action, masks and RGB
(default 360×270); depth is not exported. See [format and options](processing/README.md)
for input requirements, explicit exclusions and loader examples.
