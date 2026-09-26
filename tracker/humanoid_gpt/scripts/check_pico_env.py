#!/usr/bin/env python3
"""Check the HumanoidGPT Pico/GMR simulation environment."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _installed_editable_root() -> Path | None:
    target_name = "humanoid-gpt"
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


def _resolve_gmr_asset(value, gmr_module) -> Path | None:
    if not isinstance(value, (str, os.PathLike)):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        Path(gmr_module.__file__).resolve().parent / path,
        PROJECT_ROOT / ".deps/GMR" / path,
    ]
    return next(
        (candidate for candidate in candidates if candidate.exists()), candidates[0]
    )


def _preload_xrt_library(checker: Checker, xrt_root: Path) -> None:
    if platform.machine() == "aarch64":
        library = xrt_root / "lib/aarch64/libPXREARobotSDK.so"
    else:
        library = xrt_root / "lib/libPXREARobotSDK.so"
    if not library.is_file():
        checker.fail(f"XRobo native library missing: {library}")
        return
    try:
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        checker.fail(f"load {library}: {exc}")
        return
    checker.ok(f"XRobo native library: {library}")


def _check_gmr_contract(checker: Checker, gmr_module) -> None:
    params = checker.import_module("general_motion_retargeting.params")
    if params is None:
        return
    ik_configs = getattr(params, "IK_CONFIG_DICT", {})
    robot_xmls = getattr(params, "ROBOT_XML_DICT", {})
    try:
        ik_value = ik_configs["xrobot"]["unitree_g1"]
    except (KeyError, TypeError):
        checker.fail("GMR lacks required xrobot -> unitree_g1 IK config")
    else:
        path = _resolve_gmr_asset(ik_value, gmr_module)
        if path is not None and path.is_file():
            checker.ok(f"GMR xrobot -> unitree_g1 config: {path}")
        else:
            checker.fail(f"GMR IK config asset missing: {path or ik_value}")

    try:
        xml_value = robot_xmls["unitree_g1"]
    except (KeyError, TypeError):
        checker.fail("GMR lacks unitree_g1 robot XML")
    else:
        path = _resolve_gmr_asset(xml_value, gmr_module)
        if path is not None and path.is_file():
            checker.ok(f"GMR unitree_g1 XML: {path}")
        else:
            checker.fail(f"GMR robot XML asset missing: {path or xml_value}")

    try:
        gmr_module.GeneralMotionRetargeting(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=1.7,
        )
    except Exception as exc:
        checker.fail(f"construct GMR(xrobot -> unitree_g1): {exc}")
    else:
        checker.ok("construct GMR(xrobot -> unitree_g1)")


def _check_onnx(checker: Checker, ort, path: Path, label: str) -> None:
    if not path.is_file():
        checker.fail(f"{label} ONNX missing: {path}")
        return
    try:
        session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError("model has no inputs or outputs")
    except Exception as exc:
        checker.fail(f"{label} ONNX smoke: {exc}")
        return
    checker.ok(
        f"{label} ONNX: {path} ({len(inputs)} input(s), {len(outputs)} output(s))"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service-mode",
        choices=("auto", "external"),
        default="auto",
    )
    parser.add_argument(
        "--xrt-root",
        type=Path,
        default=PROJECT_ROOT / ".deps/xrobotoolkit_sdk",
    )
    parser.add_argument(
        "--track-onnx",
        type=Path,
        default=PROJECT_ROOT / "storage/ckpts/pns_wo_priv216.onnx",
    )
    parser.add_argument(
        "--walk-onnx",
        type=Path,
        default=PROJECT_ROOT
        / "storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx",
    )
    parser.add_argument("--require-manus", action="store_true")
    parser.add_argument("--skip-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checker = Checker()
    wuji_root = PROJECT_ROOT / ".deps/wuji_retargeting"
    if wuji_root.is_dir():
        sys.path.insert(0, str(wuji_root))

    if (3, 11) <= sys.version_info[:2] < (3, 13):
        checker.ok(f"Python {platform.python_version()}")
    else:
        checker.fail(
            f"Python {platform.python_version()} is unsupported; use 3.11 or 3.12"
        )

    editable_root = _installed_editable_root()
    if editable_root == PROJECT_ROOT:
        checker.ok(f"Humanoid-GPT editable install: {editable_root}")
    else:
        checker.fail(
            "Humanoid-GPT editable install does not point to this checkout "
            f"(got {editable_root})"
        )

    for module_name in (
        "numpy",
        "scipy",
        "mujoco",
        "onnxruntime",
        "torch",
        "jax",
        "mink",
        "qpsolvers",
        "rich",
        "zmq",
        "wuji_retargeting",
        "deploy.pose_zmq",
        "deploy.pose_manager",
        "deploy.play_track",
    ):
        checker.import_module(module_name)

    if args.require_manus:
        manus_root = PROJECT_ROOT / ".deps/manus_sdk"
        binding = next(manus_root.glob("ManusServer*.so"), None)
        if binding is None:
            checker.fail(f"MANUS Python binding missing under {manus_root}")
        else:
            checker.ok(f"MANUS Python binding: {binding}")
            sys.path.insert(0, str(manus_root))
            checker.import_module("ManusServer")
    service = Path("/opt/apps/roboticsservice/runService.sh")
    if service.is_file():
        checker.ok(f"Robotics Service: {service}")
    elif args.service_mode == "auto":
        checker.fail(f"Robotics Service missing for auto mode: {service}")
    else:
        checker.warn(
            f"local Robotics Service missing; external mode selected: {service}"
        )

    _preload_xrt_library(checker, args.xrt_root)
    checker.import_module("xrobotoolkit_sdk")

    gmr = checker.import_module("general_motion_retargeting")
    if gmr is not None:
        _check_gmr_contract(checker, gmr)

    g1_version = os.environ.get("G1_VERSION", "5010")
    track_xml = (
        PROJECT_ROOT / f"storage/assets/unitree_g1_{g1_version}/scene_mjx_track.xml"
    )
    if track_xml.is_file():
        checker.ok(f"MuJoCo tracking XML: {track_xml}")
    else:
        checker.fail(f"MuJoCo tracking XML missing: {track_xml}")

    ort = (
        importlib.import_module("onnxruntime") if "onnxruntime" in sys.modules else None
    )
    if ort is not None:
        providers = ort.get_available_providers()
        if "CPUExecutionProvider" in providers:
            checker.ok(f"ONNX providers: {providers}")
        else:
            checker.fail(f"CPUExecutionProvider unavailable: {providers}")
        if "TensorrtExecutionProvider" not in providers:
            checker.warn("TensorRT provider unavailable (allowed for MuJoCo v1)")
        if not args.skip_checkpoints:
            _check_onnx(checker, ort, args.track_onnx, "tracking")
            _check_onnx(checker, ort, args.walk_onnx, "walking")

    print(
        f"Environment check complete: failures={checker.failures}, "
        f"warnings={checker.warnings}"
    )
    return 1 if checker.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
