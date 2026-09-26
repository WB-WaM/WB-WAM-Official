#!/usr/bin/env python3
"""Check PC-side dependencies for Pico -> HumanoidGPT -> Unitree G1."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import importlib
import json
import os
import platform
import socket
import struct
import sys
import time
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DDS_HOME = PROJECT_ROOT / ".deps/cyclonedds/install"
DEFAULT_UNITREE_ROOT = PROJECT_ROOT / ".deps/unitree_sdk2_python"
EXPECTED_HG_LOW_CMD_CRC = 0xFE172F9F


class Checker:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[OK]   {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")

    def import_module(self, name: str):
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            self.fail(f"import {name}: {exc}")
            return None
        self.ok(f"import {name}")
        return module

    def optional_import(self, name: str, allowed_missing: bool):
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            if allowed_missing:
                self.warn(f"import {name}: {exc}")
            else:
                self.fail(f"import {name}: {exc}")
            return None
        self.ok(f"import {name}")
        return module


def _editable_root(distribution_name: str) -> Path | None:
    target_name = distribution_name.casefold().replace("_", "-")
    for distribution in metadata.distributions():
        installed_name = (distribution.metadata.get("Name") or "").casefold()
        if installed_name.replace("_", "-") != target_name:
            continue
        direct_url_text = distribution.read_text("direct_url.json")
        if not direct_url_text:
            continue
        try:
            direct_url = json.loads(direct_url_text)
        except (TypeError, json.JSONDecodeError):
            continue
        parsed = urlparse(direct_url.get("url", ""))
        editable = direct_url.get("dir_info", {}).get("editable", False)
        if parsed.scheme == "file" and editable:
            return Path(unquote(parsed.path)).resolve()
    return None


def _check_native_dds(checker: Checker, dds_home: Path) -> None:
    candidates = (
        dds_home / "lib/libddsc.so",
        dds_home / "lib64/libddsc.so",
    )
    library = next((path for path in candidates if path.is_file()), None)
    if library is None:
        checker.fail(f"CycloneDDS native library missing under {dds_home}")
        return
    try:
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        checker.fail(f"load {library}: {exc}")
        return
    checker.ok(f"CycloneDDS native library: {library}")


def _interface_ipv4(name: str) -> str | None:
    request = struct.pack("256s", name.encode("utf-8")[:15])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            response = fcntl.ioctl(sock.fileno(), 0x8915, request)
        except OSError:
            return None
    return socket.inet_ntoa(response[20:24])


def _check_network(checker: Checker, name: str, require_link: bool) -> None:
    interface = Path("/sys/class/net") / name
    if not interface.is_dir():
        checker.fail(f"network interface does not exist: {name}")
        return
    operstate_path = interface / "operstate"
    state = (
        operstate_path.read_text().strip() if operstate_path.is_file() else "unknown"
    )
    ipv4 = _interface_ipv4(name)
    message = f"network interface {name}: state={state}, ipv4={ipv4 or 'none'}"
    if require_link and state != "up":
        checker.fail(message)
    elif state != "up":
        checker.warn(message)
    else:
        checker.ok(message)
    if ipv4 is None:
        message = f"{name} has no IPv4 address; configure the G1 subnet before launch"
        if require_link:
            checker.fail(message)
        else:
            checker.warn(message)


def _check_crc(checker: Checker) -> None:
    try:
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        from deploy.real_robot import FastHGLowCmdPacker

        command = unitree_hg_msg_dds__LowCmd_()
        crc = CRC()
        sdk_value = crc.Crc(command)
        if not isinstance(sdk_value, int):
            raise TypeError(f"unexpected CRC type: {type(sdk_value)!r}")
        fast_value = FastHGLowCmdPacker(command, crc.crc_lib).crc()
        if sdk_value != EXPECTED_HG_LOW_CMD_CRC:
            raise RuntimeError(
                f"expected 0x{EXPECTED_HG_LOW_CMD_CRC:08x}, got 0x{sdk_value:08x}"
            )
        if fast_value != sdk_value:
            raise RuntimeError(
                f"SDK CRC 0x{sdk_value:08x} != fast packer 0x{fast_value:08x}"
            )
    except Exception as exc:
        checker.fail(f"Unitree HG LowCmd CRC smoke: {exc}")
        return
    checker.ok(f"Unitree HG LowCmd CRC: SDK=fast=expected=0x{sdk_value:08x}")


def _check_checkpoints(checker: Checker) -> None:
    paths = (
        PROJECT_ROOT / "storage/ckpts/pns_wo_priv216.onnx",
        PROJECT_ROOT / "storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx",
    )
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            checker.ok(f"checkpoint: {path}")
        else:
            checker.fail(f"checkpoint missing or empty: {path}")


def _check_g1_asset_selection(checker: Checker) -> None:
    try:
        from tracking.constants import G1_VERSION, ROOT_PATH, TRACK_XML
    except Exception as exc:
        checker.fail(f"G1 asset selection import: {exc}")
        return

    version = str(G1_VERSION).strip()
    if not ROOT_PATH.is_dir() or not TRACK_XML.is_file():
        checker.fail(
            f"G1_VERSION={version} asset is incomplete: root={ROOT_PATH}, "
            f"track_xml={TRACK_XML}"
        )
        return
    checker.ok(f"G1 asset selection: G1_VERSION={version}, root={ROOT_PATH}")


def _benchmark_onnx(
    checker: Checker,
    ort,
    provider: str,
    steps: int,
    max_inference_ms: float,
) -> None:
    path = PROJECT_ROOT / "storage/ckpts/pns_wo_priv216.onnx"
    if not path.is_file():
        return
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if provider == "tensorrt":
        cache = PROJECT_ROOT / "storage/logs/trt_cache" / path.stem
        cache.mkdir(parents=True, exist_ok=True)
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache),
                },
            )
        ]
    else:
        providers = ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        expected = (
            "TensorrtExecutionProvider"
            if provider == "tensorrt"
            else "CPUExecutionProvider"
        )
        actual = session.get_providers()
        if not actual or actual[0] != expected:
            raise RuntimeError(f"expected {expected} first, got {actual}")
        feeds = {}
        for value in session.get_inputs():
            shape = [
                dim if isinstance(dim, int) and dim > 0 else 1 for dim in value.shape
            ]
            dtype = np.int64 if value.type == "tensor(int64)" else np.float32
            feeds[value.name] = np.zeros(shape, dtype=dtype)
        for _ in range(30):
            session.run(None, feeds)
        timings_ms = []
        for _ in range(steps):
            started = time.perf_counter_ns()
            session.run(None, feeds)
            timings_ms.append((time.perf_counter_ns() - started) / 1e6)
    except Exception as exc:
        checker.fail(f"{provider} tracking ONNX benchmark: {exc}")
        return
    p50, p95 = np.percentile(timings_ms, [50, 95])
    message = (
        f"{provider} tracking ONNX latency: p50={p50:.3f} ms, "
        f"p95={p95:.3f} ms, max={max(timings_ms):.3f} ms, n={steps}"
    )
    if p95 > max_inference_ms:
        checker.warn(f"{message}; diagnostic limit={max_inference_ms:.3f} ms exceeded")
    else:
        checker.ok(message)


def _benchmark_walk(
    checker: Checker,
    ort,
    policy_provider: str,
    steps: int,
    max_inference_ms: float,
) -> None:
    path = PROJECT_ROOT / "storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx"
    if not path.is_file():
        return
    try:
        from deploy.walk_policy import WalkPolicy

        walk_provider = "cpu" if policy_provider == "cpu" else "cuda"
        policy = WalkPolicy(str(path), provider=walk_provider)
        expected = (
            "CPUExecutionProvider"
            if walk_provider == "cpu"
            else "CUDAExecutionProvider"
        )
        actual = policy.infer_fn.get_providers()
        if not actual or actual[0] != expected:
            raise RuntimeError(f"expected {expected} first, got {actual}")
        root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        root_gyro = np.zeros(3, dtype=np.float32)
        joint_qpos = policy.default_qpos
        joint_qvel = np.zeros_like(joint_qpos)
        command = np.zeros(3, dtype=np.float32)
        for _ in range(30):
            policy.infer(root_quat, root_gyro, joint_qpos, joint_qvel, command)
        timings_ms = []
        for _ in range(steps):
            started = time.perf_counter_ns()
            policy.infer(root_quat, root_gyro, joint_qpos, joint_qvel, command)
            timings_ms.append((time.perf_counter_ns() - started) / 1e6)
    except Exception as exc:
        checker.fail(f"walking ONNX benchmark: {exc}")
        return
    p50, p95 = np.percentile(timings_ms, [50, 95])
    message = (
        f"walk ONNX ({expected}) latency: p50={p50:.3f} ms, "
        f"p95={p95:.3f} ms, max={max(timings_ms):.3f} ms, n={steps}"
    )
    if p95 > max_inference_ms:
        checker.warn(f"{message}; diagnostic limit={max_inference_ms:.3f} ms exceeded")
    else:
        checker.ok(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", help="G1-facing network interface")
    parser.add_argument(
        "--require-link",
        action="store_true",
        help="fail when --net is not up (the run wrapper enables this)",
    )
    parser.add_argument(
        "--dds-home",
        type=Path,
        default=Path(os.environ.get("CYCLONEDDS_HOME", DEFAULT_DDS_HOME)),
    )
    parser.add_argument("--policy-provider", choices=("cpu", "tensorrt"), default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=200)
    parser.add_argument("--max-inference-ms", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checker = Checker()

    if (3, 11) <= sys.version_info[:2] < (3, 13):
        checker.ok(f"Python {platform.python_version()}")
    else:
        checker.fail(
            f"Python {platform.python_version()} is unsupported; use 3.11 or 3.12"
        )

    project_root = _editable_root("Humanoid-GPT")
    if project_root == PROJECT_ROOT:
        checker.ok(f"Humanoid-GPT editable install: {project_root}")
    else:
        checker.fail(
            "Humanoid-GPT editable install does not point to this checkout "
            f"(got {project_root})"
        )

    unitree_root = _editable_root("unitree-sdk2py")
    if unitree_root == DEFAULT_UNITREE_ROOT:
        checker.ok(f"Unitree SDK editable install: {unitree_root}")
    else:
        checker.fail(
            "Unitree SDK editable install does not point inside HumanoidGPT .deps "
            f"(got {unitree_root})"
        )

    _check_native_dds(checker, args.dds_home.resolve())

    for module_name in (
        "cyclonedds",
        "unitree_sdk2py.core.channel",
        "unitree_sdk2py.idl.default",
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "deploy.real_robot",
        "general_motion_retargeting",
        "xrobotoolkit_sdk",
    ):
        checker.import_module(module_name)
    _check_g1_asset_selection(checker)
    _check_crc(checker)

    try:
        cyclone_version = metadata.version("cyclonedds")
    except metadata.PackageNotFoundError:
        checker.fail("CycloneDDS Python distribution is not installed")
    else:
        if cyclone_version == "0.10.2":
            checker.ok(f"CycloneDDS Python version: {cyclone_version}")
        else:
            checker.fail(f"CycloneDDS Python must be 0.10.2, got {cyclone_version}")

    ort = checker.import_module("onnxruntime")
    if ort is not None:
        providers = ort.get_available_providers()
        expected_provider = (
            "TensorrtExecutionProvider"
            if args.policy_provider == "tensorrt"
            else "CPUExecutionProvider"
        )
        if expected_provider in providers:
            checker.ok(f"ONNX Runtime providers: {providers}")
        else:
            checker.fail(f"{expected_provider} unavailable: {providers}")

    tensorrt = checker.optional_import("tensorrt", args.policy_provider != "tensorrt")
    if tensorrt is not None:
        checker.ok(f"TensorRT Python version: {getattr(tensorrt, '__version__', '?')}")

    _check_checkpoints(checker)

    if args.benchmark and ort is not None:
        if args.benchmark_steps < 1:
            checker.fail("--benchmark-steps must be positive")
        elif not np.isfinite(args.max_inference_ms) or args.max_inference_ms <= 0:
            checker.fail("--max-inference-ms must be finite and positive")
        else:
            _benchmark_onnx(
                checker,
                ort,
                args.policy_provider,
                args.benchmark_steps,
                args.max_inference_ms,
            )
            _benchmark_walk(
                checker,
                ort,
                args.policy_provider,
                args.benchmark_steps,
                args.max_inference_ms,
            )

    if args.net:
        _check_network(checker, args.net, args.require_link)
    else:
        checker.warn("network check skipped; pass --net <robot_nic> before launch")

    print(
        f"Real environment check complete: failures={checker.failures}, "
        f"warnings={checker.warnings}"
    )
    return 1 if checker.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
