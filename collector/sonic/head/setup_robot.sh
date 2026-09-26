#!/usr/bin/env bash
# Build on the robot, against its RUNNING kernel; never copy a PC-built .ko.
set -euo pipefail
HEAD_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:---build-only}"
case "$ACTION" in --build-only|--install) ;; *) echo 'Usage: setup_robot.sh [--build-only|--install]' >&2; exit 2;; esac
HEAD_BUILD_DIR="${HEAD_BUILD_DIR:-$HEAD_SOURCE_DIR/build}"
KERNEL_RELEASE="$(uname -r)"
KERNEL_BUILD="/lib/modules/$KERNEL_RELEASE/build"
for tool in make cmake c++ python3 modinfo; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; install build-essential cmake python3 kmod" >&2; exit 1; }
done
cmake -S "$HEAD_SOURCE_DIR" -B "$HEAD_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$HEAD_BUILD_DIR" -j "${HEAD_BUILD_JOBS:-2}"
ctest --test-dir "$HEAD_BUILD_DIR" --output-on-failure

# Existing distro modules take precedence; otherwise compile the bundled exact-release source.
MODULE_TO_INSTALL=""
if modinfo ch341 >/dev/null 2>&1; then
  echo "Using existing kernel ch341 driver: $(modinfo -F filename ch341)"
else
  [[ "$KERNEL_RELEASE" == '5.15.148-tegra' ]] || { echo "Bundled driver is for L4T 36.4.3 / 5.15.148-tegra. Supply a matching driver for $KERNEL_RELEASE." >&2; exit 1; }
  grep -q 'R36 (release), REVISION: 4.3,' /etc/nv_tegra_release || { echo 'L4T release mismatch' >&2; exit 1; }
  for f in .config Module.symvers include/generated/autoconf.h; do
    [[ -s "$KERNEL_BUILD/$f" ]] || { echo "Missing running-kernel build file $KERNEL_BUILD/$f; install matching nvidia-l4t-kernel-headers (do not upgrade the kernel)." >&2; exit 1; }
  done
  [[ "$(make -s -C "$KERNEL_BUILD" kernelrelease)" == "$KERNEL_RELEASE" ]] || { echo 'Kernel header release mismatch' >&2; exit 1; }
  grep -q '^CONFIG_MODVERSIONS=y' "$KERNEL_BUILD/.config" || { echo 'Unexpected MODVERSIONS configuration' >&2; exit 1; }
  mkdir -p "$HEAD_BUILD_DIR/ch341"
  cp "$HEAD_SOURCE_DIR/ch341/ch341.c" "$HEAD_SOURCE_DIR/ch341/Makefile" "$HEAD_BUILD_DIR/ch341/"
  (cd "$HEAD_SOURCE_DIR/ch341" && sha256sum -c SHA256SUMS)
  make -C "$KERNEL_BUILD" M="$HEAD_BUILD_DIR/ch341" modules -j "${HEAD_BUILD_JOBS:-2}"
  MODULE_TO_INSTALL="$HEAD_BUILD_DIR/ch341/ch341.ko"
  [[ "$(modinfo -F vermagic "$MODULE_TO_INSTALL")" == "$KERNEL_RELEASE "* ]] || { echo 'Built module vermagic mismatch' >&2; exit 1; }
fi
[[ "$ACTION" == --install ]] || { echo 'Build passed; run --install on the robot to install/load the module.'; exit 0; }
sudo -v
if [[ -n "$MODULE_TO_INSTALL" ]]; then
  sudo install -D -m 644 "$MODULE_TO_INSTALL" "/lib/modules/$KERNEL_RELEASE/updates/wbwam/ch341.ko"
  sudo depmod -a "$KERNEL_RELEASE"
fi
# Exclude ONLY the CH340 match from brltty auto-detection, preserving other braille devices.
sudo python3 "$HEAD_SOURCE_DIR/configure_usb.py"
sudo udevadm control --reload-rules
sudo usermod -a -G dialout "${SUDO_USER:-$USER}"
sudo modprobe ch341
# Recover a CH340 interface detached by the old brltty process without cycling other USB devices.
sudo python3 - <<'PYUSB'
from pathlib import Path
import subprocess
for device in Path('/sys/bus/usb/devices').glob('*'):
    try:
        if (device / 'idVendor').read_text().strip() != '1a86' or (device / 'idProduct').read_text().strip() != '7523':
            continue
    except FileNotFoundError:
        continue
    subprocess.run(['udevadm', 'trigger', '--action=change', str(device)], check=True)
    for interface in device.glob(device.name + ':*'):
        if not (interface / 'driver').exists():
            try:
                Path('/sys/bus/usb/drivers/ch341/bind').write_text(interface.name)
            except OSError as e:
                raise SystemExit(f'Cannot rebind head CH340: {e}; reconnect the head USB adapter')
PYUSB
sudo udevadm trigger --subsystem-match=tty --action=change
sudo udevadm settle
REPO_ROOT="$(cd "$HEAD_SOURCE_DIR/../../.." && pwd)"
HEAD_CONFIG="$HEAD_SOURCE_DIR/../scripts/head_servo.env"
[[ -f "$HEAD_CONFIG" ]] || cp "$HEAD_SOURCE_DIR/../scripts/head_servo.env.example" "$HEAD_CONFIG"
# This is the same trusted robot-local shell config used by the launchers.
source "$HEAD_CONFIG"
# Fill missing pose settings in older configs without replacing explicit overrides.
for HEAD_SETTING in HEAD_SERVO_MODE HEAD_JOINT0_ENCODER HEAD_JOINT1_ENCODER; do
  if [[ -z "${!HEAD_SETTING:-}" ]]; then
    case "$HEAD_SETTING" in
      HEAD_SERVO_MODE) HEAD_DEFAULT=raw ;;
      HEAD_JOINT0_ENCODER) HEAD_DEFAULT=3027 ;;
      HEAD_JOINT1_ENCODER) HEAD_DEFAULT=1849 ;;
    esac
    printf '\n%s=%q\n' "$HEAD_SETTING" "$HEAD_DEFAULT" >> "$HEAD_CONFIG"
  fi
done
SELECT_ARGS=()
[[ -z "${HEAD_SERIAL_DEVICE:-}" ]] || SELECT_ARGS+=(--device "$HEAD_SERIAL_DEVICE")
HEAD_DETECTED_DEVICE="$(python3 "$HEAD_SOURCE_DIR/discover_serial.py" --wait 5 "${SELECT_ARGS[@]}")"
if [[ -z "${HEAD_SERIAL_DEVICE:-}" ]]; then
  printf '\n# Verified robot-local head adapter\nHEAD_SERIAL_DEVICE=%q\n' "$HEAD_DETECTED_DEVICE" >> "$HEAD_CONFIG"
fi
printf 'Head serial device: %s\n' "$HEAD_DETECTED_DEVICE"
printf '%s\n' 'Driver installed. Open a new SSH login for dialout membership, then run probe_head_servo.sh.'
