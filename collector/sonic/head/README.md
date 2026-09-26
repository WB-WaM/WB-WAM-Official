# Unitree G1 head camera and servos

## Default pose: no head calibration

Deployment and collection use `HEAD_SERVO_MODE=raw` with the following defaults in robot-local `collector/sonic/scripts/head_servo.env`:

```bash
HEAD_SERVO_MODE=raw
HEAD_JOINT0_ENCODER=3027
HEAD_JOINT1_ENCODER=1849
```

These are raw encoder counts, not degrees. The environment example, launcher and controller provide the same pose. Setup creates the configuration and fills missing values while preserving explicit overrides. **No head calibration or initial teaching is required.** At startup, the controller moves to this pose and holds it before camera acquisition starts.

This shared robot-side bundle is required by WB-WAM deployment and SONIC/PICO or HGPT collection. It separates three components: RealSense D455 image acquisition, the CH340 USB-UART kernel driver, and the two head servos. A visible camera image does not prove the servos are available.

## Sources and build

- `ch341/ch341.c` is unmodified GPL-2.0 source from NVIDIA Jetson Linux **36.4.3**, `kernel/kernel-jammy-src/drivers/usb/serial/ch341.c`. The license is alongside it. [BSP sources](https://developer.download.nvidia.com/embedded/L4T/r36_Release_v4.3/sources/public_sources.tbz2), SHA256 `2c177804679e3ed650dabec6fa958388579896f170570c6171a1b6c386669216`; source SHA256 is in `ch341/SHA256SUMS`.
- `vendor/DynamixelSDK` contains only the C++ source/headers and Apache-2.0 license from the [Unitree head-servo package](https://oss-global-cdn.unitree.com/static/c473dfd3aba74c0cb71ba2ffdebd84c4.zip), SHA256 `31b5ebc9e5ea86188ad7be047438f38c94d99a25bc618bad63fc4481e2f832ff`. `vendor/unitree_reference` retains the relevant vendor control/calibration examples as reference, not executable setup steps. The Unitree example files have no separate license notice in the supplied archive; their provenance is preserved here. Register compatibility for the robot-tested XC330-M288 is documented in the [ROBOTIS manual](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/). No precompiled SDK libraries or vendor desktop installers are included.
- `head_servo.cpp` uses that SDK and the vendor's Protocol 2.0 / 1 Mbps / servo IDs 0 and 1 interface. It builds independently of Unitree DDS. `--probe` only pings and reads; unlike the vendor read examples, it never enables torque or changes gains. `--hold-current` requires both supported XL430-W250 (1060) or XC330-M288 (1240) servos in existing position mode 3, torque off and no device errors. It sets each present encoder position as the goal before enabling, monitors both axes, and disables torque on bounded completion, stop or fault. The default launcher uses `--hold-raw` with encoder counts 3027/1849. It ramps at 10 deg/s, validates joint/encoder limits before enabling either axis and waits for position feedback before reporting ready. Normal pose holding preserves calibration, operating mode and gains; optional powered teaching temporarily adjusts and restores gains.

The driver module is compiled **on the robot** using `/lib/modules/$(uname -r)/build`, including the running kernel's configuration and `Module.symvers`. No kernel image is replaced. The bundled fallback supports `5.15.148-tegra` / L4T 36.4.3; an existing system `ch341` module is reused. Other kernels without a driver require matching source/headers, not a forced module load. See [NVIDIA kernel guidance](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Kernel/KernelCustomization.html).

## Setup (both deployment and collection)

Use the robot SSH target and authenticate interactively without putting a password in commands.

On the **PC**, from the repository root:

```bash
collector/sonic/scripts/setup_head_servo.sh USER@ROBOT_IP
# Optional second argument: repo path relative to the robot home, default WB-WAM.
```

The helper copies only this bundle and its launcher/probe scripts (including the supervised camera launcher; the camera Python environment/service must already be installed), creates/preserves `head_servo.env`, fills missing default pose values and an unset serial path from unique device discovery, compiles the SDK and `head_servo`, builds/installs `ch341` when missing, fixes only the CH340 match in brltty's udev rules, adds the SSH user to `dialout`, then uses a new SSH login to probe both servos. It never uninstalls brltty or disables unrelated braille devices. Build prerequisites are `build-essential`, `cmake`, `python3`, `kmod`, `rsync`, and the **matching** `nvidia-l4t-kernel-headers` package; install missing packages on the robot as part of head setup, without upgrading its kernel. `sudo` is required for the kernel/udev/group changes.

On the **robot**, from `~/WB-WAM`, the same steps can be run separately:

```bash
bash collector/sonic/head/setup_robot.sh --build-only
bash collector/sonic/head/setup_robot.sh --install
# Reconnect SSH to apply dialout membership, then:
collector/sonic/scripts/probe_head_servo.sh
```

Require `PASS: both head servos replied; no control registers written`, not merely a `/dev/ttyUSB*` node. The probe selects the sole `1a86:7523` adapter automatically, preferring a stable `/dev/serial/by-id/` or `by-path/` alias. If several adapters exist, set `HEAD_SERIAL_DEVICE` in robot-local `collector/sonic/scripts/head_servo.env` to the actual head adapter. Do not guess `/dev/ttyUSB0`. A missing adapter, wrong model/mode, hardware error, permission error or missing reply is a failed/pending head check.

## Verify the configured pose

With the operator ready to support the head when torque is released and the surroundings clear, run on the robot:

```bash
collector/sonic/scripts/run_head_servo.sh --motion-approved --duration 3
collector/sonic/scripts/probe_head_servo.sh
```

Require successful position feedback and both torque registers back at zero after completion. `--duration 3` holds for three seconds after reaching the configured pose; ramp time is additional. Support the head before stopping because torque-off releases it. This fixed pose does not follow PICO. For current-pose diagnosis only, explicitly set `HEAD_SERVO_MODE=current`; that check does not verify reaching the configured target.

## Optional: change the saved pose

Only to choose a different pose, run from the robot repository root:

```bash
HEAD_SERVO_MODE=teach collector/sonic/scripts/run_head_servo.sh --motion-approved
```

At `TEACH READY`, both axes have torque enabled with P=0, D=100 damping. Support and position the camera manually. Once steady, enter `hold`: the program checks stability, captures and locks the pose without torque-off, restores the original holding gains, and prints `CAPTURED`. Save those counts as `HEAD_JOINT0_ENCODER` / `HEAD_JOINT1_ENCODER` in robot-local `head_servo.env`, with `HEAD_SERVO_MODE=raw`. **Do not stop before capture: releasing torque can change the pose.** This adjustment is optional and is not part of initial environment setup.

Teaching times out after 300 seconds. Timeout, closed input, `stop` during teaching or a fault disables torque and restores gains. After capture, holding continues until stopped; support the head before stopping with Ctrl+C.

## Deployment and collection startup

On the **robot**, in one terminal, from `~/WB-WAM`:

```bash
collector/sonic/scripts/run_camera_server.sh --head-motion-approved
```

The default `HEAD_SERVO_ENABLED=1` starts `head_servo` first and requires its target-reached readiness message before starting RealSense. Keep this terminal running during both inference and collection. The supervisor stops the camera if head feedback fails and stops the head when the camera exits. Ctrl+C stops both and requests torque-disable cleanup. A lost USB link or forced process kill may prevent a disable command reaching the servos; report cleanup failures and have the operator stop/support the hardware.

For diagnosis only, `HEAD_SERVO_ENABLED=0 collector/sonic/scripts/run_camera_server.sh` runs camera acquisition alone. This does not satisfy head readiness for deployment/collection. Fake-camera dry-runs never start this robot-side service.

If no TTY appears, check USB enumeration, `modinfo ch341`, `lsmod`, matching headers and `journalctl -k` for brltty conflicts. If brltty already detached the port before setup, reconnect only the head USB adapter and rerun the probe. Do not unplug Wuji hands. Rebuild after any kernel ABI upgrade. Do not unload a serial driver while a controller holds the port.
