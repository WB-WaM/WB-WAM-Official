from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = PROJECT_ROOT / "scripts/run_pico_real.sh"
POSE_WRAPPER = PROJECT_ROOT / "scripts/run_hgpt_pose_manager.sh"
COLLECTION_ENV_KEYS = (
    "HGPT_HAND_SOURCE",
    "HGPT_PICO_SERVICE_MODE",
    "HGPT_PUBLISH_WUJI_HAND",
    "HGPT_NET",
    "HGPT_POLICY_PROVIDER",
    "HGPT_POSE_ENDPOINT",
    "HGPT_STATE_ACTION_ENDPOINT",
    "HGPT_HAND_STATUS_ENDPOINT",
    "HGPT_CAMERA_ENDPOINT",
    "HUMANOID_GPT_PICO_SERVICE_MODE",
)


def _run_script(
    tmp_path: Path,
    wrapper: Path,
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess, str]:
    fake_python = tmp_path / "fake-python"
    capture = tmp_path / "calls.txt"
    capture.unlink(missing_ok=True)
    fake_python.write_text(
        """#!/bin/sh
if [ "$1" = "-c" ]; then
  exit 0
fi
if [ "$1" = "-m" ]; then
  label=RUN
else
  label=CHECK
fi
{
  printf '%s\\n' "$label"
  printf '%s\\n' "$@"
  printf '%s\\n' END
} >> "$WRAPPER_CAPTURE"
"""
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    for key in COLLECTION_ENV_KEYS:
        env.pop(key, None)
    if env_overrides:
        env.update(env_overrides)
    env["HUMANOID_GPT_PYTHON"] = str(fake_python)
    env["WRAPPER_CAPTURE"] = str(capture)
    result = subprocess.run(
        [str(wrapper), *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    text = capture.read_text() if capture.exists() else ""
    return result, text


def _run_wrapper(
    tmp_path: Path,
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess, str]:
    return _run_script(
        tmp_path,
        WRAPPER,
        *args,
        env_overrides=env_overrides,
    )


def _call(capture: str, label: str) -> list[str]:
    lines = capture.splitlines()
    start = lines.index(label) + 1
    end = lines.index("END", start)
    return lines[start:end]


def _value(args: list[str], option: str) -> str:
    index = args.index(option)
    return args[index + 1]


def test_debug_wrapper_has_fixed_safe_arguments(tmp_path: Path) -> None:
    result, capture = _run_wrapper(tmp_path)
    assert result.returncode == 0, result.stderr
    check = _call(capture, "CHECK")
    run = _call(capture, "RUN")
    assert "--benchmark" not in check
    assert _value(check, "--net") == "eno1"
    assert _value(check, "--policy-provider") == "tensorrt"
    assert run[-1] == "--debug"
    assert "--allow-real-publish" not in run
    assert _value(run, "--net") == "eno1"
    assert _value(run, "--freq") == "50"
    assert _value(run, "--mocap-type") == "pico"
    assert _value(run, "--buffer-ms") == "0"
    assert _value(run, "--pico-service-mode") == "external"
    assert _value(run, "--real-policy-provider") == "tensorrt"
    assert _value(run, "--pose-endpoint") == "tcp://127.0.0.1:5556"
    assert _value(run, "--state-action-bind") == "tcp://127.0.0.1:5558"
    assert "--real-target-clip-abort-rad" not in run
    assert "--real-max-consecutive-target-clips" not in run
    assert _value(run, "--real-control-join-timeout-s") == "1"


def test_publish_alias_runs_benchmark_and_enables_only_final_gate(
    tmp_path: Path,
) -> None:
    result, capture = _run_wrapper(
        tmp_path,
        "--publish_lowcmd",
    )
    assert result.returncode == 0, result.stderr
    check = _call(capture, "CHECK")
    run = _call(capture, "RUN")
    assert "--benchmark" in check
    assert run[-1] == "--allow-real-publish"
    assert "--debug" not in run
    assert "REAL LOWCMD PUBLICATION ENABLED" in result.stdout


def test_environment_defaults_and_cli_have_final_precedence(tmp_path: Path) -> None:
    result, capture = _run_wrapper(
        tmp_path,
        "--net=cli0",
        "--pico-service-mode=auto",
        "--policy-provider=cpu",
        "--pose-endpoint=tcp://127.0.0.1:6001",
        "--state-action-bind=tcp://127.0.0.1:6002",
        env_overrides={
            "HGPT_NET": "env0",
            "HGPT_PICO_SERVICE_MODE": "external",
            "HGPT_POLICY_PROVIDER": "tensorrt",
            "HGPT_POSE_ENDPOINT": "tcp://127.0.0.1:7001",
            "HGPT_STATE_ACTION_ENDPOINT": "tcp://127.0.0.1:7002",
        },
    )
    assert result.returncode == 0, result.stderr
    check = _call(capture, "CHECK")
    run = _call(capture, "RUN")
    assert _value(check, "--net") == "cli0"
    assert _value(check, "--policy-provider") == "cpu"
    assert _value(run, "--pico-service-mode") == "auto"
    assert _value(run, "--pose-endpoint") == "tcp://127.0.0.1:6001"
    assert _value(run, "--state-action-bind") == "tcp://127.0.0.1:6002"


def test_pose_wrapper_defaults_to_manus_and_wuji_publish(tmp_path: Path) -> None:
    result, capture = _run_script(tmp_path, POSE_WRAPPER)
    assert result.returncode == 0, result.stderr
    run = _call(capture, "RUN")
    assert run[:2] == ["-m", "deploy.pose_manager"]
    assert _value(run, "--hand-source") == "manus"
    assert _value(run, "--pico-service-mode") == "external"
    assert _value(run, "--hand-status-endpoint") == "tcp://192.168.123.164:5559"
    assert "--publish-wuji-hand" in run


def test_empty_environment_values_fall_back_to_shared_defaults(tmp_path: Path) -> None:
    empty_environment = dict.fromkeys(COLLECTION_ENV_KEYS, "")

    result, capture = _run_wrapper(tmp_path, env_overrides=empty_environment)
    assert result.returncode == 0, result.stderr
    check = _call(capture, "CHECK")
    run = _call(capture, "RUN")
    assert _value(check, "--net") == "eno1"
    assert _value(check, "--policy-provider") == "tensorrt"
    assert _value(run, "--pico-service-mode") == "external"
    assert _value(run, "--pose-endpoint") == "tcp://127.0.0.1:5556"
    assert _value(run, "--state-action-bind") == "tcp://127.0.0.1:5558"

    result, capture = _run_script(
        tmp_path,
        POSE_WRAPPER,
        env_overrides=empty_environment,
    )
    assert result.returncode == 0, result.stderr
    run = _call(capture, "RUN")
    assert _value(run, "--hand-source") == "manus"
    assert _value(run, "--pico-service-mode") == "external"
    assert _value(run, "--hand-status-endpoint") == "tcp://192.168.123.164:5559"
    assert "--publish-wuji-hand" in run


def test_pose_wrapper_environment_overrides_shared_defaults(tmp_path: Path) -> None:
    result, capture = _run_script(
        tmp_path,
        POSE_WRAPPER,
        env_overrides={
            "HGPT_HAND_SOURCE": "pico",
            "HGPT_PICO_SERVICE_MODE": "auto",
            "HGPT_HAND_STATUS_ENDPOINT": "tcp://127.0.0.1:7003",
            "HGPT_PUBLISH_WUJI_HAND": "false",
        },
    )
    assert result.returncode == 0, result.stderr
    run = _call(capture, "RUN")
    assert _value(run, "--hand-source") == "pico"
    assert _value(run, "--pico-service-mode") == "auto"
    assert _value(run, "--hand-status-endpoint") == "tcp://127.0.0.1:7003"
    assert "--no-publish-wuji-hand" in run
    assert "--publish-wuji-hand" not in run


def test_pose_wrapper_cli_overrides_environment(tmp_path: Path) -> None:
    result, capture = _run_script(
        tmp_path,
        POSE_WRAPPER,
        "--hand-source",
        "manus",
        "--pico-service-mode",
        "external",
        "--hand-status-endpoint",
        "tcp://127.0.0.1:6003",
        "--publish-wuji-hand",
        env_overrides={
            "HGPT_HAND_SOURCE": "pico",
            "HGPT_PICO_SERVICE_MODE": "auto",
            "HGPT_HAND_STATUS_ENDPOINT": "tcp://127.0.0.1:7003",
            "HGPT_PUBLISH_WUJI_HAND": "false",
        },
    )
    assert result.returncode == 0, result.stderr
    run = _call(capture, "RUN")
    assert _value(run, "--hand-source") == "manus"
    assert _value(run, "--pico-service-mode") == "external"
    assert _value(run, "--hand-status-endpoint") == "tcp://127.0.0.1:6003"
    assert "--publish-wuji-hand" in run
    assert "--no-publish-wuji-hand" not in run


@pytest.mark.parametrize(
    "locked",
    [
        ("--no_debug",),
        ("--allow_real_publish",),
        ("--real_policy_provider=tensorrt",),
        ("--mocap_type=pnlink",),
        ("--buffer_ms=50",),
        ("--freq=100",),
        ("--onnx_track=other.onnx",),
        ("--onnx-walk", "other.onnx"),
        ("--policy_type=transformer",),
        ("--convert_xml_path=other.xml",),
        ("--enable_hand",),
        ("--", "--no-debug"),
        ("--headless",),
        ("--mode-machine-override", "5"),
    ],
)
def test_locked_or_unknown_play_track_options_fail_before_preflight(
    tmp_path: Path,
    locked: tuple[str, ...],
) -> None:
    result, capture = _run_wrapper(tmp_path, "--net", "eno1", *locked)
    assert result.returncode == 2
    assert not capture
    assert "locked or unsupported" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("--net",),
        ("--net", "--publish-lowcmd"),
        ("--pico-service-mode",),
        ("--policy-provider",),
    ],
)
def test_missing_values_have_friendly_usage(
    tmp_path: Path,
    args: tuple[str, ...],
) -> None:
    result, capture = _run_wrapper(tmp_path, *args)
    assert result.returncode == 2
    assert not capture
    assert "Missing value" in result.stderr
    assert "unbound variable" not in result.stderr


def test_boolean_wrapper_options_reject_values(tmp_path: Path) -> None:
    result, capture = _run_wrapper(
        tmp_path,
        "--net=eno1",
        "--publish-lowcmd=true",
    )
    assert result.returncode == 2
    assert not capture
    assert "does not accept a value" in result.stderr
