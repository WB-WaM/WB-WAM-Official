#!/usr/bin/env python3
"""Run a WB-WAM checkpoint through the SONIC execution backend."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

BRIDGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BRIDGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridge.common.live_runtime import (  # noqa: E402
    ObservationSource,
    _dry_run,
    _read_yaml,
    _resolve_prompt,
    _run_live,
)
from bridge.common.policy import MockPolicyAdapter, PolicyAdapter  # noqa: E402
from bridge.common.rtc import RTC_MODE_OFF, normalize_rtc_mode  # noqa: E402
from bridge.sonic.action_schema import ACTION_DIM  # noqa: E402
from bridge.wbwam.actions.representation import ActionRepresentation  # noqa: E402
from bridge.wbwam.adapter import WBWAMAdapter  # noqa: E402
from bridge.wbwam.sonic import (  # noqa: E402
    PhysicalWAMToSonicAdapter,
    normalize_relative_base_source,
)
from bridge.wbwam.text_cache import ensure_prompt_embedding  # noqa: E402


def _validate_last_action_schedule(action_config: dict, runtime_config: dict) -> None:
    if normalize_relative_base_source(action_config.get("relative_base_source")) != "last_action":
        return
    chunk_schedule = str(runtime_config.get("chunk_schedule") or "periodic").strip().lower()
    if chunk_schedule not in {"sequential", "serial"}:
        raise ValueError(
            "sonic_action.relative_base_source=last_action requires "
            "runtime.chunk_schedule=sequential so executed actions are committed"
        )


def _make_wam_adapter(
    config: dict,
    input_config: dict,
    *,
    mock_policy: bool,
) -> PolicyAdapter:
    policy_config = dict(config.get("policy") or {})
    action_config = dict(config.get("sonic_action") or {})
    adapter_name = str(policy_config.get("adapter", "wbwam")).lower()
    if adapter_name not in {"wbwam", "mock"}:
        raise ValueError(f"unsupported adapter {adapter_name!r}; expected wbwam or mock")
    if mock_policy or adapter_name == "mock":
        # Exercise the runtime without model weights or a SONIC ONNX encoder.
        return MockPolicyAdapter(
            action_horizon=int(policy_config.get("action_horizon", 32)),
            action_dim=ACTION_DIM,
            state_dim=int(input_config.get("expected_state_dim", 107)),
            latency_s=float(policy_config.get("mock_latency_s", 0.0)),
        )
    representation = ActionRepresentation.parse(policy_config.get("output_representation", "absolute"))
    if representation == ActionRepresentation.SONIC_TOKEN:
        raise ValueError("WB-WAM deployment requires physical 72D actions")
    model_input_config = dict(input_config)
    if bool(action_config.get("project_model_state", True)):
        model_input_config.update(base_state_dim=72, expected_state_dim=72)
    adapter = WBWAMAdapter(policy_config=policy_config, input_config=model_input_config)
    expected_ranges = representation.relative_joint_ranges(
        relative_hands=bool(action_config.get("relative_hands", True))
    )
    if adapter.relative_joint_ranges != expected_ranges:
        raise ValueError(
            f"policy.output_representation={representation.value!r} maps to "
            f"relative_joint_ranges={expected_ranges}, but checkpoint uses {adapter.relative_joint_ranges}"
        )
    action_config["relative_joint_ranges"] = [list(item) for item in adapter.relative_joint_ranges]
    if (
        "relative_action_reference" in action_config
        and action_config["relative_action_reference"] != adapter.relative_action_reference
    ):
        raise ValueError("sonic_action.relative_action_reference conflicts with checkpoint")
    action_config["relative_action_reference"] = adapter.relative_action_reference
    return PhysicalWAMToSonicAdapter(
        adapter,
        representation=representation,
        action_config=action_config,
        state_layout=input_config.get("state_layout"),
    )


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_config: Path | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config or BRIDGE_ROOT / "configs" / "deploy.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-policy", action="store_true", help="Use a local mock WAM instead of loading WBWAM")
    parser.add_argument("--fake-camera", action="store_true")
    parser.add_argument("--fake-state", action="store_true")
    parser.add_argument(
        "--observation-time-alignment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable camera/body/hand alignment on the PC receive clock (enabled by default)",
    )
    parser.add_argument("--episode", type=Path, default=None, help="Use a recorded collector episode as input")
    parser.add_argument(
        "--execute-chunk-steps",
        type=int,
        default=None,
        help="Override the number of actions executed before re-inference",
    )
    parser.add_argument("--pose-port", type=int, default=None, help="Override the execution pose publish port")
    parser.add_argument(
        "--snap-token-grid", action="store_true", help="Override config and snap token to 1/16 FSQ grid"
    )
    parser.add_argument("--prompt", default=None, help="Override language instruction")
    parser.add_argument(
        "--checkpoint", default=None, help="Override policy.checkpoint_path (file or run directory)"
    )
    parser.add_argument(
        "--check-config", action="store_true", help="Validate checkpoint metadata without loading model weights"
    )
    parser.add_argument(
        "--inference-mode",
        choices=("idm", "wbwam", "wbwam-style"),
        default=None,
        help="Override policy.inference_mode: two-stage video-to-action or direct first-frame action",
    )
    parser.add_argument(
        "--relative-base",
        "--relative-base-source",
        dest="relative_base_source",
        choices=("curr_obs", "last_action"),
        default=None,
        help=(
            "Override sonic_action.relative_base_source for relative action restoration. "
            "curr_obs anchors to the live measured state; last_action anchors to the last "
            "published predicted action and falls back to curr_obs for the first chunk. "
            "Model proprio always remains the live measured state."
        ),
    )
    parser.add_argument(
        "--root-orientation-mode",
        choices=("actual_imu", "predicted_relative"),
        default=None,
        help=(
            "Override sonic_action.root_orientation_mode. actual_imu rebuilds each "
            "SONIC token from fresh robot base_quat; predicted_relative keeps the "
            "legacy inference-time reference-relative encoding."
        ),
    )
    parser.add_argument("--mock-policy-latency-s", type=float, default=None, help="Sleep in mock policy inference")
    parser.add_argument(
        "--smoke-warmup-iters",
        type=int,
        default=3,
        help="Additional unmeasured WBWAM warmup iterations after the reported cold call",
    )
    parser.add_argument(
        "--smoke-iters",
        type=int,
        default=20,
        help="Measured inference iterations used for dry-run latency statistics",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    default_config: Path | None = None,
) -> int:
    args = parse_args(
        argv,
        default_config=default_config,
    )
    config = _read_yaml(args.config)
    policy_config = config.setdefault("policy", {})
    if args.checkpoint is not None:
        policy_config["checkpoint_path"] = args.checkpoint
    if args.check_config:
        from bridge.wbwam.checkpoint import validate_deployment_config

        artifacts, _cfg, codec = validate_deployment_config(policy_config)
        print(f"[OK] config={artifacts.config} stats={artifacts.stats} weights={artifacts.weights}")
        print(f"[OK] semantic state/action={codec.state_dim}/{codec.action_dim}; model={codec.action_target_dim}")
        return 0
    if args.inference_mode is not None:
        config.setdefault("policy", {})["inference_mode"] = args.inference_mode
    if args.relative_base_source is not None:
        config.setdefault("sonic_action", {})["relative_base_source"] = args.relative_base_source
    if args.root_orientation_mode is not None:
        config.setdefault("sonic_action", {})["root_orientation_mode"] = args.root_orientation_mode
    runtime_config = dict(config.get("runtime") or {})
    if runtime_config.get("visualization_endpoint"):
        raise ValueError("visualization_endpoint is not supported by the deployment-only bridge")
    if args.observation_time_alignment is not None:
        runtime_config["observation_time_alignment"] = args.observation_time_alignment
    _validate_last_action_schedule(config.get("sonic_action") or {}, runtime_config)
    if args.execute_chunk_steps is not None:
        if args.execute_chunk_steps <= 0:
            raise ValueError("--execute-chunk-steps must be > 0")
        runtime_config["execute_chunk_steps"] = args.execute_chunk_steps
    pose_port = args.pose_port
    if pose_port is not None:
        if not 1 <= pose_port <= 65535:
            raise ValueError("--pose-port must be in 1..65535")
        runtime_config["pose_port"] = pose_port
    rtc_mode = normalize_rtc_mode(runtime_config.get("rtc_mode", RTC_MODE_OFF))
    if rtc_mode != RTC_MODE_OFF:
        raise ValueError("WAM bridge supports only runtime.rtc_mode=off; WBWAM does not implement RTC guidance.")
    runtime_config["rtc_mode_resolved"] = RTC_MODE_OFF
    runtime_config["rtc_enabled_resolved"] = False
    input_config = dict(config.get("wam_input") or {})
    if args.mock_policy_latency_s is not None:
        config.setdefault("policy", {})["mock_latency_s"] = args.mock_policy_latency_s
    if args.snap_token_grid:
        runtime_config["snap_token_grid"] = True
    prompt = _resolve_prompt(
        args_prompt=args.prompt,
        config_prompt=str((config.get("task") or {}).get("prompt", "")),
        runtime_config=runtime_config,
    )
    policy_config = config.setdefault("policy", {})
    if not args.mock_policy and str(policy_config.get("adapter", "wbwam")).lower() == "wbwam":
        import torch

        from bridge.wbwam.checkpoint import validate_deployment_config

        validate_deployment_config(policy_config)
        if str(policy_config.get("device", "cuda")).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --mock-policy for a CPU-only smoke test")
        policy_config["prompt_embedding_path"] = str(
            ensure_prompt_embedding(args.config, prompt, policy_config=policy_config)
        )
    source = ObservationSource(args, runtime_config, input_config)
    try:
        adapter = _make_wam_adapter(
            config,
            input_config,
            mock_policy=args.mock_policy,
        )
        expected_action_dim = ACTION_DIM
        if adapter.info.action_dim != expected_action_dim:
            raise ValueError(f"policy action_dim={adapter.info.action_dim}, expected {expected_action_dim}")
        if args.dry_run:
            from bridge.common.benchmark import (
                prepare_cuda_benchmark,
                report_cuda_peak_memory,
            )

            benchmark_adapter, cuda_handle = prepare_cuda_benchmark(adapter)
            result = _dry_run(
                adapter=benchmark_adapter,
                source=source,
                runtime_config=runtime_config,
                prompt=prompt,
                warmup_iterations=args.smoke_warmup_iters,
                timed_iterations=args.smoke_iters,
            )
            report_cuda_peak_memory(cuda_handle)
            return result
        return _run_live(adapter=adapter, source=source, runtime_config=runtime_config, prompt=prompt)
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
