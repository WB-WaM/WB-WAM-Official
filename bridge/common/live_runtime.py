#!/usr/bin/env python3
"""Run a robot policy and stream its SONIC token actions to deploy via ZMQ v4."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable
from enum import Enum
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np

from bridge.common.camera_client import RemoteCameraClient  # noqa: E402
from bridge.common.camera_selection import resolve_camera_key  # noqa: E402
from bridge.common.config import read_deployment_config  # noqa: E402
from bridge.common.observation_builder import (  # noqa: E402
    ACTION_DIM as OBS_LAST_ACTION_DIM,
    STATE_DIM,
    append_last_action_to_state,
    build_fake_state,
    build_observation,
    build_rgbd_observation,
    build_state_from_frame,
    load_episode_depth_image,
    load_episode_frames,
    load_episode_image,
)
from bridge.common.policy import PolicyAdapter  # noqa: E402
from bridge.common.rtc import (  # noqa: E402
    RTC_MODE_INFERENCE_GUIDANCE,
    RTC_MODE_OFF,
    RTC_MODE_TRAINING_PREFIX,
    build_rtc_prefix_info,
    estimate_rtc_prefix_steps,
    normalize_rtc_mode,
    rtc_mode_uses_overlap,
    summarize_action_delta,
)
from bridge.common.state_layouts import STATE_LAYOUT_QUAT, normalize_state_layout  # noqa: E402
from bridge.common.state_subscriber import (  # noqa: E402
    AlignedRobotStateSubscriber,
    BaseQuatSnapshot,
    BaseQuatSubscriber,
    RobotStateSubscriber,
    build_state_from_latest,
    hand_feedback_has_valid_actual,
)
from bridge.sonic.action_schema import (  # noqa: E402
    TOKEN_DIM,
    WUJI_QPOS_DIM,
    SonicAction,
    apply_wuji_qpos_limits,
    normalize_action_chunk,
    split_action_chunk,
)
from bridge.sonic.initial_poses import (  # noqa: E402
    OPEN_WUJI_QPOS,
    build_initial_sonic_action,
    summarize_initial_sonic_action,
)
from bridge.sonic.protocol_v4 import SonicV4Payload, SonicV4Publisher, build_pose_v4_message  # noqa: E402


class BridgeMode(str, Enum):
    PLANNER_IDLE = "PLANNER_IDLE"
    POLICY_READY = "POLICY_READY"
    POLICY_ARMING = "POLICY_ARMING"
    POLICY_RUNNING = "POLICY_RUNNING"
    POLICY_PAUSED = "POLICY_PAUSED"
    RETURNING_INITIAL = "RETURNING_INITIAL"


CHUNK_SCHEDULE_PERIODIC = "periodic"
CHUNK_SCHEDULE_SEQUENTIAL = "sequential"


def _normalize_chunk_schedule(value: object) -> str:
    normalized = str(value or CHUNK_SCHEDULE_PERIODIC).strip().lower()
    if normalized in {"periodic", "continuous"}:
        return CHUNK_SCHEDULE_PERIODIC
    if normalized in {"sequential", "serial"}:
        return CHUNK_SCHEDULE_SEQUENTIAL
    raise ValueError(f"runtime.chunk_schedule must be periodic or sequential, got {value!r}")


def _chunk_schedule_allows_inference(
    *,
    chunk_schedule: str,
    mode: BridgeMode,
    has_cached_chunk: bool,
    chunk_index: int,
    chunk_end_index: int,
) -> bool:
    if chunk_schedule == CHUNK_SCHEDULE_PERIODIC:
        return mode in {
            BridgeMode.POLICY_READY,
            BridgeMode.POLICY_ARMING,
            BridgeMode.POLICY_RUNNING,
        }
    if mode in {BridgeMode.POLICY_READY, BridgeMode.POLICY_ARMING}:
        return True
    return mode == BridgeMode.POLICY_RUNNING and has_cached_chunk and chunk_index >= chunk_end_index


def _rtc_mode_max_delay(*, mode: str, adapter: PolicyAdapter) -> int:
    if mode == RTC_MODE_TRAINING_PREFIX:
        return int(adapter.info.rtc_max_delay)
    if mode == RTC_MODE_INFERENCE_GUIDANCE:
        return int(adapter.info.action_horizon)
    return 0


def _read_yaml(path: Path) -> dict[str, Any]:
    return read_deployment_config(path)


def _fake_image(width: int = 640, height: int = 480) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 1] = 64
    image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    return image


def _resolve_prompt(*, args_prompt: str | None, config_prompt: str, runtime_config: dict[str, Any]) -> str:
    if args_prompt is not None:
        prompt = args_prompt.strip()
        if not prompt:
            raise ValueError("--prompt cannot be empty")
        return prompt

    default_prompt = str(config_prompt or "").strip()
    if bool(runtime_config.get("prompt_interactive", True)) and sys.stdin.isatty():
        suffix = f" [{default_prompt}]" if default_prompt else ""
        entered = input(f"Task prompt{suffix}: ").strip()
        prompt = entered or default_prompt
    else:
        prompt = default_prompt

    if not prompt:
        raise ValueError("task prompt is empty; set task.prompt in config or pass --prompt")
    print(f"[INFO] Policy task prompt: {prompt}")
    return prompt


class ObservationSource:
    def __init__(self, args: argparse.Namespace, runtime_config: dict[str, Any], input_config: dict[str, Any]):
        self.args = args
        self.runtime_config = runtime_config
        self.input_config = input_config
        self.episode_frames: list[dict[str, Any]] | None = None
        self.episode_dir: Path | None = None
        self.episode_index = 0
        self.camera_client: RemoteCameraClient | None = None
        self.state_subscriber: RobotStateSubscriber | AlignedRobotStateSubscriber | None = None
        self.last_left_hand: np.ndarray | None = None
        self.last_right_hand: np.ndarray | None = None
        self.last_action: np.ndarray | None = None
        self.observation_time_alignment = bool(runtime_config.get("observation_time_alignment", True))
        self.observation_alignment_max_skew_ms = float(
            runtime_config.get("observation_alignment_max_skew_ms", 25.0)
        )
        self.observation_alignment_timeout_s = float(runtime_config.get("observation_alignment_timeout_s", 5.0))
        self.observation_alignment_log_interval_s = float(
            runtime_config.get("observation_alignment_log_interval_s", 5.0)
        )
        self._last_alignment_log_time = -1.0e9
        self.include_last_action = bool(input_config.get("include_last_action", False))
        self.base_state_dim = int(input_config.get("base_state_dim", STATE_DIM))
        self.last_action_dim = int(input_config.get("last_action_dim", OBS_LAST_ACTION_DIM))
        self.state_layout = normalize_state_layout(input_config.get("state_layout", STATE_LAYOUT_QUAT))
        image_mode_value = input_config.get("image_mode")
        self.image_mode = "rgb" if image_mode_value is None else str(image_mode_value).strip().lower()
        if self.image_mode not in {"rgb", "rgbd"}:
            raise ValueError(f"vla_input.image_mode must be 'rgb' or 'rgbd', got {self.image_mode!r}")
        default_expected = self.base_state_dim + (self.last_action_dim if self.include_last_action else 0)
        self.expected_state_dim = int(input_config.get("expected_state_dim", default_expected))
        for name, value in (
            ("observation_alignment_max_skew_ms", self.observation_alignment_max_skew_ms),
            ("observation_alignment_timeout_s", self.observation_alignment_timeout_s),
            ("observation_alignment_log_interval_s", self.observation_alignment_log_interval_s),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"runtime.{name} must be finite and > 0")

        if args.episode is not None:
            self.episode_dir = args.episode.expanduser().resolve()
            self.episode_frames = load_episode_frames(self.episode_dir)
            if not self.episode_frames:
                raise ValueError(f"episode has no frames: {self.episode_dir}")
        self.observation_alignment_active = bool(
            self.observation_time_alignment
            and self.episode_frames is None
            and not args.fake_camera
            and not args.fake_state
        )

        if not args.fake_camera and self.episode_frames is None:
            endpoint = str(runtime_config["camera_endpoint"])
            timeout_ms = int(runtime_config.get("camera_timeout_ms", 1000))
            camera_key = str(runtime_config.get("camera_key", "d435"))
            if bool(runtime_config.get("select_camera_on_start", True)) and sys.stdin.isatty():
                camera_key = resolve_camera_key(
                    endpoint=endpoint,
                    default_key=camera_key,
                    timeout_ms=timeout_ms,
                    interactive=True,
                )
            else:
                camera_key = resolve_camera_key(
                    endpoint=endpoint,
                    default_key=camera_key,
                    timeout_ms=timeout_ms,
                    interactive=False,
                )
            self.camera_client = RemoteCameraClient(
                endpoint=endpoint,
                camera_key=camera_key,
                timeout_ms=timeout_ms,
            )

        if not args.fake_state and self.episode_frames is None:
            subscriber_type = (
                AlignedRobotStateSubscriber if self.observation_alignment_active else RobotStateSubscriber
            )
            self.state_subscriber = subscriber_type(
                state_action_endpoint=str(runtime_config["state_action_endpoint"]),
                hand_status_endpoint=str(runtime_config.get("hand_status_endpoint", "")),
            )

        if self.observation_time_alignment:
            if self.observation_alignment_active:
                print(
                    "[INFO] observation time alignment enabled: "
                    "clock=pc_receive_monotonic "
                    f"max_skew={self.observation_alignment_max_skew_ms:g}ms"
                )
            else:
                print("[INFO] observation time alignment bypassed for fake or recorded inputs")

    def next(self, *, prompt: str) -> dict[str, Any]:
        if self.episode_frames is not None:
            assert self.episode_dir is not None
            frame = self.episode_frames[min(self.episode_index, len(self.episode_frames) - 1)]
            self.episode_index += 1
            image = load_episode_image(
                self.episode_dir,
                frame,
                camera=str(self.runtime_config.get("episode_camera", "primary")),
            )
            state = build_state_from_frame(frame, state_layout=self.state_layout)
            state = self._compose_state(state)
            if self.image_mode == "rgbd":
                depth = load_episode_depth_image(
                    self.episode_dir,
                    frame,
                    camera=str(self.runtime_config.get("episode_camera", "primary")),
                )
                return build_rgbd_observation(
                    image=image,
                    depth=depth,
                    state=state,
                    prompt=prompt,
                    expected_state_dim=self.expected_state_dim,
                    depth_max_mm=float(self.runtime_config.get("depth_max_mm", 5000.0)),
                )
            return build_observation(
                image=image,
                state=state,
                prompt=prompt,
                expected_state_dim=self.expected_state_dim,
            )

        if self.observation_alignment_active:
            return self._next_aligned_live(prompt=prompt)

        state = build_fake_state(state_layout=self.state_layout) if self.args.fake_state else self._live_state()
        state = self._compose_state(state)
        if self.image_mode == "rgbd":
            if self.args.fake_camera:
                image = _fake_image()
                depth = np.zeros(image.shape[:2], dtype=np.uint16)
            else:
                image, depth = self._live_rgbd()
            return build_rgbd_observation(
                image=image,
                depth=depth,
                state=state,
                prompt=prompt,
                expected_state_dim=self.expected_state_dim,
                depth_max_mm=float(self.runtime_config.get("depth_max_mm", 5000.0)),
            )

        image = _fake_image() if self.args.fake_camera else self._live_image()
        return build_observation(
            image=image,
            state=state,
            prompt=prompt,
            expected_state_dim=self.expected_state_dim,
        )

    def _next_aligned_live(self, *, prompt: str) -> dict[str, Any]:
        if self.camera_client is None:
            raise RuntimeError("camera client is not initialized")
        if not isinstance(self.state_subscriber, AlignedRobotStateSubscriber):
            raise RuntimeError("aligned robot-state subscriber is not initialized")

        require_hand_actual = bool(self.runtime_config.get("require_hand_actual", False))
        deadline = time.monotonic() + self.observation_alignment_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            camera_sample = (
                self.camera_client.get_rgbd_sample()
                if self.image_mode == "rgbd"
                else self.camera_client.get_rgb_sample()
            )
            remaining_s = max(deadline - time.monotonic(), 1.0e-6)
            try:
                latest = self.state_subscriber.wait_for_aligned(
                    target_monotonic_ns=camera_sample.receive_monotonic_ns,
                    max_skew_ms=self.observation_alignment_max_skew_ms,
                    timeout_s=remaining_s,
                    require_hand_actual=require_hand_actual,
                )
            except TimeoutError as exc:
                last_error = exc
                continue

            if require_hand_actual and not hand_feedback_has_valid_actual(latest.hand_feedback):
                last_error = RuntimeError("hand actual position is required but not valid")
                continue
            state = build_state_from_latest(
                latest,
                fallback_left_hand=self.last_left_hand,
                fallback_right_hand=self.last_right_hand,
                state_layout=self.state_layout,
            )
            state = self._compose_state(state)
            self._log_observation_alignment(
                camera_receive_monotonic_ns=camera_sample.receive_monotonic_ns,
                latest=latest,
            )
            if self.image_mode == "rgbd":
                if camera_sample.depth is None:
                    raise RuntimeError("aligned RGB-D camera sample is missing depth")
                return build_rgbd_observation(
                    image=camera_sample.rgb,
                    depth=camera_sample.depth,
                    state=state,
                    prompt=prompt,
                    expected_state_dim=self.expected_state_dim,
                    depth_max_mm=float(self.runtime_config.get("depth_max_mm", 5000.0)),
                )
            return build_observation(
                image=camera_sample.rgb,
                state=state,
                prompt=prompt,
                expected_state_dim=self.expected_state_dim,
            )

        detail = "" if last_error is None else f": {last_error}"
        raise TimeoutError(
            f"timed out after {self.observation_alignment_timeout_s:g}s acquiring aligned observation{detail}"
        )

    def _log_observation_alignment(
        self,
        *,
        camera_receive_monotonic_ns: int,
        latest: Any,
    ) -> None:
        now = time.monotonic()
        if now - self._last_alignment_log_time < self.observation_alignment_log_interval_s:
            return
        self._last_alignment_log_time = now
        state_skew_ms = (latest.state_receive_monotonic_ns - camera_receive_monotonic_ns) / 1.0e6
        hand_summary = "n/a"
        if latest.hand_receive_monotonic_ns >= 0:
            hand_summary = f"{(latest.hand_receive_monotonic_ns - camera_receive_monotonic_ns) / 1.0e6:+.1f}ms"
        print(f"[ALIGN] observation state-camera={state_skew_ms:+.1f}ms hand-camera={hand_summary}")

    def _compose_state(self, state: np.ndarray) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.size != self.base_state_dim:
            raise ValueError(f"base state dim {state_arr.size}, expected {self.base_state_dim}")
        if not self.include_last_action:
            return state_arr
        return append_last_action_to_state(
            state_arr,
            last_action=self.last_action,
            action_dim=self.last_action_dim,
        )

    def _live_image(self) -> np.ndarray:
        if self.camera_client is None:
            raise RuntimeError("camera client is not initialized")
        return self.camera_client.get_rgb()

    def _live_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        if self.camera_client is None:
            raise RuntimeError("camera client is not initialized")
        return self.camera_client.get_rgbd()

    def _live_state(self) -> np.ndarray:
        if self.state_subscriber is None:
            raise RuntimeError("state subscriber is not initialized")
        require_hand_actual = bool(self.runtime_config.get("require_hand_actual", False))
        timeout_key = "hand_status_timeout_s" if require_hand_actual else "state_timeout_s"
        timeout_s = float(self.runtime_config.get(timeout_key, self.runtime_config.get("state_timeout_s", 5.0)))
        latest = self.state_subscriber.wait_for_first(
            timeout_s=timeout_s,
            require_hand_actual=require_hand_actual,
        )
        if require_hand_actual:
            feedback = latest.hand_feedback
            if not hand_feedback_has_valid_actual(feedback):
                raise RuntimeError("hand actual position is required but not valid")
        return build_state_from_latest(
            latest,
            fallback_left_hand=self.last_left_hand,
            fallback_right_hand=self.last_right_hand,
            state_layout=self.state_layout,
        )

    def remember_hands(self, left: np.ndarray, right: np.ndarray) -> None:
        self.last_left_hand = left.astype(np.float32, copy=True)
        self.last_right_hand = right.astype(np.float32, copy=True)

    def remember_action(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()

    def close(self) -> None:
        if self.camera_client is not None:
            self.camera_client.close()
        if self.state_subscriber is not None:
            self.state_subscriber.close()


def _summarize_action_chunk(actions: np.ndarray) -> str:
    return (
        f"shape={tuple(actions.shape)} "
        f"min={float(np.min(actions)):.4f} "
        f"max={float(np.max(actions)):.4f} "
        f"mean={float(np.mean(actions)):.4f}"
    )


def _split_checked_action_chunk(
    actions: np.ndarray,
    runtime_config: dict[str, Any],
) -> tuple[np.ndarray, list[SonicAction]]:
    raw_chunk = normalize_action_chunk(actions)
    sonic_actions = split_action_chunk(
        raw_chunk,
        snap_token_grid=bool(runtime_config.get("snap_token_grid", False)),
        token_min=runtime_config.get("token_min"),
        token_max=runtime_config.get("token_max"),
    )

    max_abs_token = runtime_config.get("max_abs_token")
    if max_abs_token is not None:
        max_abs = max(float(np.max(np.abs(step.token_state))) for step in sonic_actions)
        if max_abs > float(max_abs_token):
            raise ValueError(f"token abs max {max_abs:.4f} exceeds limit {float(max_abs_token):.4f}")

    return raw_chunk, sonic_actions


def _split_checked_action_step(
    action: np.ndarray,
    runtime_config: dict[str, Any],
) -> tuple[np.ndarray, SonicAction]:
    values = np.asarray(action, dtype=np.float32).reshape(1, -1)
    raw_chunk, sonic_actions = _split_checked_action_chunk(values, runtime_config)
    return raw_chunk[0], sonic_actions[0]


def _raw_action_from_sonic(action: SonicAction) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(action.token_state, dtype=np.float32).reshape(-1),
            np.asarray(action.left_wuji_qpos, dtype=np.float32).reshape(-1),
            np.asarray(action.right_wuji_qpos, dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32, copy=False)


def _limit_hand_qpos_delta(
    action: SonicAction,
    *,
    previous_action: SonicAction | None,
    max_delta: float | None,
) -> SonicAction:
    if previous_action is None or max_delta is None or float(max_delta) <= 0.0:
        return action
    limit = float(max_delta)
    left_prev = np.asarray(previous_action.left_wuji_qpos, dtype=np.float32).reshape(-1)
    right_prev = np.asarray(previous_action.right_wuji_qpos, dtype=np.float32).reshape(-1)
    left = np.asarray(action.left_wuji_qpos, dtype=np.float32).reshape(-1)
    right = np.asarray(action.right_wuji_qpos, dtype=np.float32).reshape(-1)
    limited_left = left_prev + np.clip(left - left_prev, -limit, limit)
    limited_right = right_prev + np.clip(right - right_prev, -limit, limit)
    return SonicAction(
        token_state=np.asarray(action.token_state, dtype=np.float32).copy(),
        left_wuji_qpos=limited_left.astype(np.float32, copy=False),
        right_wuji_qpos=limited_right.astype(np.float32, copy=False),
    )


def _build_return_initial_action(
    *,
    previous_action: SonicAction | None,
    hand_max_delta: float | None,
) -> SonicAction:
    """Build one initial-token return frame with rate-limited hands."""

    return _limit_hand_qpos_delta(
        build_initial_sonic_action(),
        previous_action=previous_action,
        max_delta=hand_max_delta,
    )


def _build_paused_open_hands_action(
    *,
    previous_action: SonicAction | None,
    hand_max_delta: float | None,
) -> SonicAction:
    """Keep the last body token while rate-limiting both hands toward open."""

    base_action = previous_action or build_initial_sonic_action()
    target = SonicAction(
        token_state=np.asarray(base_action.token_state, dtype=np.float32).copy(),
        left_wuji_qpos=OPEN_WUJI_QPOS.copy(),
        right_wuji_qpos=OPEN_WUJI_QPOS.copy(),
    )
    return _limit_hand_qpos_delta(
        target,
        previous_action=previous_action,
        max_delta=hand_max_delta,
    )


def _hands_are_open(action: SonicAction, *, atol: float = 1.0e-6) -> bool:
    return bool(
        np.allclose(action.left_wuji_qpos, OPEN_WUJI_QPOS, rtol=0.0, atol=atol)
        and np.allclose(action.right_wuji_qpos, OPEN_WUJI_QPOS, rtol=0.0, atol=atol)
    )


def _chunk_start_index(
    *,
    runtime_config: dict[str, Any],
    inference_delay_s: float,
    publish_rate_hz: float,
    horizon: int,
    execute_chunk_steps: int,
) -> int:
    strategy = str(runtime_config.get("chunk_start_strategy", "drop_tail")).strip().lower()
    if strategy in {"drop_tail", "drop-tail", "first", "zero"}:
        return 0
    if strategy in {"rtc_drop_prefix", "rtc-drop-prefix", "drop_prefix", "drop-prefix"}:
        rtc_mode = runtime_config.get("rtc_mode_resolved")
        rtc_enabled = (
            rtc_mode_uses_overlap(str(rtc_mode))
            if rtc_mode is not None
            else bool(runtime_config.get("rtc_enabled_resolved", runtime_config.get("rtc_enabled", False)))
        )
        if not rtc_enabled:
            return 0
        prefix_steps = estimate_rtc_prefix_steps(
            runtime_config,
            inference_duration_s=None,
            publish_rate_hz=publish_rate_hz,
            max_delay_steps=int(runtime_config.get("rtc_max_delay", horizon)),
            action_horizon=horizon,
        )
        return min(max(0, prefix_steps), max(0, horizon - 1))
    if strategy not in {"skip_stale", "skip-stale", "latency_compensated", "latency-compensated"}:
        raise ValueError(
            "runtime.chunk_start_strategy must be 'drop_tail', 'skip_stale'/"
            "'latency_compensated', or 'rtc_drop_prefix', "
            f"got {strategy!r}"
        )
    usable_steps = max(1, min(horizon, execute_chunk_steps))
    fixed_pipeline_delay_s = float(runtime_config.get("fixed_pipeline_delay_s", 0.0))
    if fixed_pipeline_delay_s < 0.0:
        raise ValueError("runtime.fixed_pipeline_delay_s must be >= 0")
    compensated_delay_s = max(0.0, inference_delay_s) + fixed_pipeline_delay_s
    stale_steps = int(round(compensated_delay_s * publish_rate_hz))
    return min(stale_steps, usable_steps - 1)


def _chunk_end_index(*, start_index: int, horizon: int, execute_chunk_steps: int) -> int:
    start_index = max(0, min(int(start_index), max(0, int(horizon) - 1)))
    execute_steps = max(1, int(execute_chunk_steps))
    return max(start_index + 1, min(int(horizon), start_index + execute_steps))


def _inference_is_fresh_for_arm(*, inference_start: float, execute_arm_time: float | None) -> bool:
    """Return whether a result was started after the execute key armed control."""

    return execute_arm_time is None or inference_start >= execute_arm_time


def _put_latest(result_queue: queue.Queue, item: Any) -> None:
    try:
        result_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        result_queue.get_nowait()
    except queue.Empty:
        pass
    result_queue.put_nowait(item)


class RuntimeMetrics:
    def __init__(self, *, window_size: int = 20):
        self.inference_durations: deque[float] = deque(maxlen=window_size)
        self.chunk_arrival_times: deque[float] = deque(maxlen=window_size)
        self.publish_times: deque[float] = deque(maxlen=window_size)
        self.last_chunk_arrival_time: float | None = None
        self.last_inference_duration_s: float | None = None
        self.chunk_underruns = 0
        self.chunk_hold_steps = 0
        self.last_warning_time = -1.0e9
        self.chunk_count = 0

    def record_chunk(self, *, arrival_time: float, inference_duration_s: float) -> None:
        self.inference_durations.append(inference_duration_s)
        self.chunk_arrival_times.append(arrival_time)
        self.last_chunk_arrival_time = arrival_time
        self.last_inference_duration_s = inference_duration_s
        self.chunk_hold_steps = 0
        self.chunk_count += 1

    def record_publish(self, *, publish_time: float) -> None:
        self.publish_times.append(publish_time)

    def record_hold_step(self, *, expected: bool = False) -> None:
        if not expected:
            self.chunk_underruns += 1
        self.chunk_hold_steps += 1

    @staticmethod
    def _rate_from_times(times: deque[float]) -> float | None:
        if len(times) < 2:
            return None
        elapsed = times[-1] - times[0]
        if elapsed <= 0:
            return None
        return (len(times) - 1) / elapsed

    @property
    def actual_inference_hz(self) -> float | None:
        return self._rate_from_times(self.chunk_arrival_times)

    @property
    def actual_publish_hz(self) -> float | None:
        return self._rate_from_times(self.publish_times)

    @property
    def mean_inference_duration_s(self) -> float | None:
        if not self.inference_durations:
            return None
        return float(sum(self.inference_durations) / len(self.inference_durations))


def _fmt_metric(value: float | None, *, suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def _wait_for_minimum_inference_latency(
    *,
    inference_start: float,
    minimum_latency_s: float,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> float:
    """Delay a synthetic result until the configured end-to-end latency.

    Real adapter/source work counts toward the minimum. This lets synthetic
    policies exercise the same sequential hold interval as WBWAM
    without adding the configured latency on top of encoder/runtime overhead.
    """

    minimum = float(minimum_latency_s)
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError("simulated_inference_latency_s must be finite and >= 0")
    now = monotonic_fn()
    remaining = minimum - max(0.0, now - float(inference_start))
    if remaining > 0.0:
        sleep_fn(remaining)
        now = monotonic_fn()
    return now


def _inference_worker_loop(
    *,
    adapter: PolicyAdapter,
    source: ObservationSource,
    runtime_config: dict[str, Any],
    prompt: str,
    request_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            request = request_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        busy_event.set()
        inference_start = time.monotonic()
        try:
            observation = source.next(prompt=prompt)
            request = request if isinstance(request, dict) else {}
            actions = adapter.infer(
                observation,
                rtc_prefix_actions=request.get("rtc_prefix_actions"),
                rtc_prefix_mask=request.get("rtc_prefix_mask"),
                rtc_guidance_target=request.get("rtc_guidance_target"),
                rtc_guidance_mask=request.get("rtc_guidance_mask"),
                rtc_guidance_scale=request.get("rtc_guidance_scale"),
            )
            inference_done_time = _wait_for_minimum_inference_latency(
                inference_start=inference_start,
                minimum_latency_s=float(runtime_config.get("simulated_inference_latency_s", 0.0)),
            )
            inference_duration_s = inference_done_time - inference_start
            raw_chunk, sonic_actions = _split_checked_action_chunk(actions, runtime_config)
            inference_context = adapter.take_inference_context()
            _put_latest(
                result_queue,
                (
                    raw_chunk,
                    sonic_actions,
                    inference_start,
                    inference_done_time,
                    inference_duration_s,
                    inference_context,
                ),
            )
        except Exception as exc:
            print(f"[WARNING] Policy inference chunk dropped: {exc}")
        finally:
            busy_event.clear()


def _start_keyboard_control_listener(
    *,
    enabled: bool,
    runtime_config: dict[str, Any],
    control_queue: queue.Queue,
    stop_event: threading.Event,
) -> threading.Thread | None:
    if not enabled:
        return None
    if not sys.stdin.isatty():
        print("[INFO] keyboard bridge controls disabled because stdin is not interactive")
        return None

    prepare_key = str(runtime_config.get("prepare_key", "4")).lower()
    execute_key = str(runtime_config.get("execute_key", "1")).lower()
    estop_key = str(runtime_config.get("estop_key", "e")).lower()
    quit_key = str(runtime_config.get("quit_key", "q")).lower()
    command_aliases = {
        prepare_key: "prepare",
        "prepare": "prepare",
        "ready": "prepare",
        execute_key: "execute",
        "execute": "execute",
        "start": "execute",
        "toggle": "execute",
        estop_key: "estop",
        "estop": "estop",
        "stop": "estop",
        quit_key: "quit",
        "quit": "quit",
        "exit": "quit",
    }

    def _loop() -> None:
        print(
            "[INFO] bridge controls: "
            f"'{prepare_key}'=POLICY_READY, '{execute_key}'=toggle Policy, "
            f"'{estop_key}'=estop, '{quit_key}'=planner+quit"
        )
        print("[INFO] single-key controls enabled; no Enter key required")
        try:
            import select
            import termios
            import tty

            stdin_fd = sys.stdin.fileno()
            old_attrs = termios.tcgetattr(stdin_fd)
            tty.setcbreak(stdin_fd)
            try:
                buffer = ""
                while not stop_event.is_set():
                    readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if not readable:
                        continue
                    ch = os.read(stdin_fd, 1).decode(errors="ignore")
                    if not ch:
                        return
                    if ch in {"\n", "\r"}:
                        token = buffer.strip().lower()
                        buffer = ""
                        if not token:
                            continue
                    elif ch in {"\x03", "\x04"}:
                        token = estop_key
                    elif ch.isalnum():
                        token = ch.lower()
                        # Allow long commands such as "estop" if someone pastes/types them.
                        if token not in command_aliases:
                            buffer += token
                            if len(buffer) > 16:
                                buffer = buffer[-16:]
                            continue
                    else:
                        continue

                    command = command_aliases.get(token)
                    if command is None:
                        print(
                            "[INFO] unknown command; use "
                            f"'{prepare_key}', '{execute_key}', '{estop_key}', or '{quit_key}'"
                        )
                        continue
                    control_queue.put(command)
                    if command in {"estop", "quit"}:
                        return
            finally:
                termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_attrs)
        except Exception as exc:
            print(f"[WARNING] single-key controls unavailable; falling back to Enter mode: {exc}")
            while not stop_event.is_set():
                line = sys.stdin.readline()
                if not line:
                    return
                command = command_aliases.get(line.strip().lower())
                if command is None:
                    print(
                        "[INFO] unknown command; use "
                        f"'{prepare_key}', '{execute_key}', '{estop_key}', or '{quit_key}'"
                    )
                    continue
                control_queue.put(command)
                if command in {"estop", "quit"}:
                    return

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def _publish_initial_pose(
    *,
    publisher: SonicV4Publisher,
    runtime_config: dict[str, Any],
    frame_index: int,
    repeat_key: str = "initial_pose_repeat",
    rate_key: str = "initial_pose_rate_hz",
    repeat_default: int = 20,
    rate_default: float = 10.0,
    label: str = "initial pose",
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    repeat = int(runtime_config.get(repeat_key, repeat_default))
    rate_hz = float(runtime_config.get(rate_key, rate_default))
    if repeat <= 0:
        return frame_index
    if rate_hz <= 0:
        raise ValueError(f"{rate_key} must be > 0")

    action = build_initial_sonic_action()
    period = 1.0 / rate_hz
    print(f"[INFO] publishing {label} token: repeat={repeat} rate={rate_hz:g}Hz")
    print(f"[INFO] initial pose summary: {summarize_initial_sonic_action()}")
    for _ in range(repeat):
        publisher.publish(
            SonicV4Payload(
                frame_index=frame_index,
                timestamp_monotonic_ns=time.monotonic_ns(),
                action=action,
            )
        )
        frame_index += 1
        sleep_fn(period)
    return frame_index


def _publish_open_hands_once(
    *,
    publisher: SonicV4Publisher,
    frame_index: int,
    last_action: SonicAction | None,
) -> int:
    base_action = last_action or build_initial_sonic_action()
    open_action = SonicAction(
        token_state=base_action.token_state.astype(np.float32, copy=True),
        left_wuji_qpos=OPEN_WUJI_QPOS.copy(),
        right_wuji_qpos=OPEN_WUJI_QPOS.copy(),
    )
    publisher.publish(
        SonicV4Payload(
            frame_index=frame_index,
            timestamp_monotonic_ns=time.monotonic_ns(),
            action=open_action,
        )
    )
    return frame_index + 1


def _dry_run(
    *,
    adapter: PolicyAdapter,
    source: ObservationSource,
    runtime_config: dict[str, Any],
    prompt: str,
    warmup_iterations: int = 0,
    timed_iterations: int = 1,
) -> int:
    if warmup_iterations < 0:
        raise ValueError(f"warmup_iterations must be >= 0, got {warmup_iterations}")
    if timed_iterations <= 0:
        raise ValueError(f"timed_iterations must be > 0, got {timed_iterations}")
    observation = source.next(prompt=prompt)

    cold_start_ns = time.perf_counter_ns()
    actions = adapter.infer(observation)
    cold_start_ms = (time.perf_counter_ns() - cold_start_ns) / 1e6
    raw_chunk, sonic_actions = _split_checked_action_chunk(actions, runtime_config)

    for _ in range(warmup_iterations):
        warmup_actions = adapter.infer(observation)
        _split_checked_action_chunk(warmup_actions, runtime_config)

    latency_ms = []
    for _ in range(timed_iterations):
        started_ns = time.perf_counter_ns()
        actions = adapter.infer(observation)
        latency_ms.append((time.perf_counter_ns() - started_ns) / 1e6)
        raw_chunk, sonic_actions = _split_checked_action_chunk(actions, runtime_config)

    latency_arr = np.asarray(latency_ms, dtype=np.float64)
    latency_p50 = float(np.percentile(latency_arr, 50))
    latency_p95 = float(np.percentile(latency_arr, 95))
    payload = SonicV4Payload(
        frame_index=0,
        timestamp_monotonic_ns=time.monotonic_ns(),
        action=sonic_actions[0],
    )
    message = build_pose_v4_message(payload)
    initial_message = build_pose_v4_message(
        SonicV4Payload(
            frame_index=0,
            timestamp_monotonic_ns=time.monotonic_ns(),
            action=build_initial_sonic_action(),
        )
    )
    print("Policy-to-SONIC bridge dry run")
    print(f"  policy:       {adapter.info.name}")
    print(f"  runtime:      {adapter.info.runtime_version}")
    print(f"  state_dim:    {adapter.info.state_dim}")
    print(f"  rtc:          {runtime_config.get('rtc_mode_resolved', RTC_MODE_OFF)}")
    print(f"  policy_rtc:   {adapter.info.rtc_enabled}")
    print(f"  policy_dim:   {adapter.info.action_dim}")
    print(f"  policy_h:     {adapter.info.action_horizon}")
    obs_depth = observation.get("observation/depth_image")
    depth_summary = "" if obs_depth is None else f" depth={obs_depth.shape}"
    print(
        f"  observation:  image={observation['observation/image'].shape}"
        f"{depth_summary} state={observation['states'].shape}"
    )
    print(f"  actions:      {_summarize_action_chunk(raw_chunk)}")
    print(f"  sonic_steps:  {len(sonic_actions)}")
    print(f"  token_dim:    {sonic_actions[0].token_state.size}")
    print(f"  left_hand:    {sonic_actions[0].left_wuji_qpos.size}")
    print(f"  right_hand:   {sonic_actions[0].right_wuji_qpos.size}")
    print(f"  init_summary: {summarize_initial_sonic_action()}")
    print(f"  initial_pose: {len(initial_message)} bytes")
    print(f"  zmq_payload:  {len(message)} bytes")
    print("  inference latency (adapter.infer end-to-end):")
    print(f"    cold:       {cold_start_ms:.3f} ms ({1000.0 / cold_start_ms:.3f} Hz)")
    print(f"    warmup:     {warmup_iterations} iteration(s), excluded")
    print(
        "    steady:     "
        f"n={timed_iterations} p50={latency_p50:.3f} ms "
        f"p95={latency_p95:.3f} ms mean={latency_arr.mean():.3f} ms "
        f"min={latency_arr.min():.3f} ms max={latency_arr.max():.3f} ms "
        f"({1000.0 / latency_arr.mean():.3f} Hz mean)"
    )
    print("    samples_ms: " + ", ".join(f"{value:.3f}" for value in latency_ms))
    print("[INFO] dry run complete; no ZMQ messages sent")
    return 0


def _run_live(
    *,
    adapter: PolicyAdapter,
    source: ObservationSource,
    runtime_config: dict[str, Any],
    prompt: str,
) -> int:
    publish_rate_hz = float(runtime_config.get("action_publish_rate_hz", runtime_config.get("rate_hz", 10.0)))
    inference_rate_hz = float(runtime_config.get("inference_rate_hz", 2.0))
    execute_chunk_steps = int(runtime_config.get("execute_chunk_steps", runtime_config.get("execute_steps", 1)))
    chunk_schedule = _normalize_chunk_schedule(runtime_config.get("chunk_schedule"))
    chunk_start_strategy = str(runtime_config.get("chunk_start_strategy", "drop_tail")).strip().lower()
    if chunk_schedule == CHUNK_SCHEDULE_SEQUENTIAL and chunk_start_strategy not in {
        "drop_tail",
        "drop-tail",
        "first",
        "zero",
    }:
        raise ValueError("sequential chunk_schedule requires chunk_start_strategy=zero")
    planner_idle_rate_hz = float(runtime_config.get("planner_idle_rate_hz", 2.0))
    metrics_log_interval_s = float(runtime_config.get("metrics_log_interval_s", 1.0))
    inference_warn_ratio = float(runtime_config.get("inference_warn_ratio", 0.8))
    chunk_stale_warn_s = float(runtime_config.get("chunk_stale_warn_s", 1.0))
    max_chunk_hold_steps = int(runtime_config.get("max_chunk_hold_steps", 10))
    simulated_inference_latency_s = float(runtime_config.get("simulated_inference_latency_s", 0.0))
    return_initial_repeat = int(runtime_config.get("return_initial_pose_repeat", 45))
    return_initial_rate_hz = float(runtime_config.get("return_initial_pose_rate_hz", 30.0))
    needs_live_base_quat = bool(adapter.needs_live_base_quat)
    base_quat_stale_ms = float(runtime_config.get("dynamic_base_quat_stale_ms", 100.0))
    full_action_horizon = int(adapter.info.action_horizon)
    valid_horizon_value = adapter.info.valid_action_horizon
    valid_action_horizon = full_action_horizon if valid_horizon_value is None else int(valid_horizon_value)
    if not 1 <= valid_action_horizon <= full_action_horizon:
        raise ValueError(f"valid action horizon {valid_action_horizon} must be in 1..{full_action_horizon}")
    if publish_rate_hz <= 0:
        raise ValueError("action_publish_rate_hz must be > 0")
    if inference_rate_hz <= 0:
        raise ValueError("inference_rate_hz must be > 0")
    if execute_chunk_steps <= 0:
        raise ValueError("execute_chunk_steps must be > 0")
    if planner_idle_rate_hz <= 0:
        raise ValueError("planner_idle_rate_hz must be > 0")
    if metrics_log_interval_s <= 0:
        raise ValueError("metrics_log_interval_s must be > 0")
    if inference_warn_ratio <= 0:
        raise ValueError("inference_warn_ratio must be > 0")
    if chunk_stale_warn_s <= 0:
        raise ValueError("chunk_stale_warn_s must be > 0")
    if max_chunk_hold_steps < 0:
        raise ValueError("max_chunk_hold_steps must be >= 0")
    if not np.isfinite(simulated_inference_latency_s) or simulated_inference_latency_s < 0.0:
        raise ValueError("runtime.simulated_inference_latency_s must be finite and >= 0")
    if return_initial_repeat <= 0:
        raise ValueError("return_initial_pose_repeat must be > 0")
    if not np.isfinite(return_initial_rate_hz) or return_initial_rate_hz <= 0.0:
        raise ValueError("return_initial_pose_rate_hz must be finite and > 0")
    if needs_live_base_quat and base_quat_stale_ms <= 0:
        raise ValueError("dynamic_base_quat_stale_ms must be > 0")
    if (
        needs_live_base_quat
        and normalize_rtc_mode(runtime_config.get("rtc_mode_resolved", RTC_MODE_OFF)) != RTC_MODE_OFF
    ):
        raise ValueError("live base-quaternion action materialization requires rtc_mode=off")

    publisher = SonicV4Publisher(port=int(runtime_config.get("pose_port", 5556)))
    print(f"[INFO] ZMQ pose v4 publisher bound at {publisher.endpoint}")
    publisher_warmup_s = float(runtime_config.get("publisher_warmup_s", 0.5))
    if publisher_warmup_s > 0:
        print(f"[INFO] waiting {publisher_warmup_s:.2f}s for deploy ZMQ subscriber to connect")
        time.sleep(publisher_warmup_s)
    print(
        "[INFO] live runtime: "
        f"schedule={chunk_schedule} inference={inference_rate_hz:g}Hz "
        f"publish={publish_rate_hz:g}Hz "
        f"simulated_min_latency={simulated_inference_latency_s:g}s "
        f"valid_action_horizon={valid_action_horizon}/{full_action_horizon} "
        f"valid_action_idx=0..{valid_action_horizon - 1} "
        f"execute_chunk_steps={execute_chunk_steps} planner_idle={planner_idle_rate_hz:g}Hz"
    )
    print(
        "[INFO] pause/return cycle: "
        "execute=planner pause, prepare=return initial + Policy warmup "
        f"frames={return_initial_repeat} rate={return_initial_rate_hz:g}Hz"
    )
    base_quat_subscriber: BaseQuatSubscriber | None = None
    if needs_live_base_quat:
        state_action_endpoint = str(runtime_config.get("state_action_endpoint", "")).strip()
        if not state_action_endpoint:
            raise ValueError("dynamic root orientation requires runtime.state_action_endpoint")
        base_quat_subscriber = BaseQuatSubscriber(
            state_action_endpoint=state_action_endpoint,
        )
        print(
            "[INFO] dynamic SONIC root orientation enabled: "
            f"base_quat={state_action_endpoint} stale={base_quat_stale_ms:g}ms"
        )

    request_queue: queue.Queue = queue.Queue(maxsize=1)
    result_queue: queue.Queue = queue.Queue(maxsize=1)
    control_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    busy_event = threading.Event()
    _start_keyboard_control_listener(
        enabled=bool(runtime_config.get("enable_keyboard_estop", True)),
        runtime_config=runtime_config,
        control_queue=control_queue,
        stop_event=stop_event,
    )
    worker = threading.Thread(
        target=_inference_worker_loop,
        kwargs={
            "adapter": adapter,
            "source": source,
            "runtime_config": runtime_config,
            "prompt": prompt,
            "request_queue": request_queue,
            "result_queue": result_queue,
            "stop_event": stop_event,
            "busy_event": busy_event,
        },
        daemon=True,
    )
    worker.start()

    frame_index = 0
    cached_raw_chunk: np.ndarray | None = None
    cached_sonic_actions: list[SonicAction] | None = None
    cached_inference_context: Any = None
    chunk_reference_committed = False
    execute_arm_time: float | None = None
    chunk_index = 0
    chunk_end_index = 0
    last_request_time = -1.0e9
    last_wait_log = 0.0
    last_planner_idle_time = -1.0e9
    last_metrics_log_time = -1.0e9
    last_published_action: SonicAction | None = None
    last_raw_action: np.ndarray | None = None
    last_stationary_action: SonicAction | None = None
    last_stationary_raw_action: np.ndarray | None = None
    latest_base_quat: BaseQuatSnapshot | None = None
    last_base_quat_warning_time = -1.0e9
    return_initial_sent = 0
    return_initial_next_publish_time = -1.0e9
    paused_hands_open = False
    paused_hands_next_publish_time = -1.0e9
    publish_period = 1.0 / publish_rate_hz
    inference_interval = 1.0 / inference_rate_hz
    planner_idle_period = 1.0 / planner_idle_rate_hz
    mode = BridgeMode.PLANNER_IDLE
    estop_requested = False
    normal_quit_requested = False
    metrics = RuntimeMetrics()

    def warn_dynamic_orientation(message: str) -> None:
        nonlocal last_base_quat_warning_time
        now = time.monotonic()
        if now - last_base_quat_warning_time >= 1.0:
            print(f"[WARNING] dynamic SONIC orientation: {message}")
            last_base_quat_warning_time = now

    def poll_base_quat() -> None:
        nonlocal latest_base_quat
        if base_quat_subscriber is None:
            return
        try:
            snapshot = base_quat_subscriber.poll(timeout_ms=0)
        except Exception as exc:
            warn_dynamic_orientation(f"invalid robot_state_action sample ignored: {exc}")
            return
        if snapshot is not None:
            latest_base_quat = snapshot

    def fresh_base_quat() -> BaseQuatSnapshot | None:
        if latest_base_quat is None:
            return None
        age_ms = (time.monotonic_ns() - latest_base_quat.receive_monotonic_ns) / 1.0e6
        if age_ms > base_quat_stale_ms:
            return None
        return latest_base_quat

    def finalize_action(
        candidate: SonicAction,
        *,
        hold_published_hands: bool = False,
    ) -> tuple[SonicAction, np.ndarray]:
        if hold_published_hands and last_published_action is not None:
            action = SonicAction(
                token_state=np.asarray(candidate.token_state, dtype=np.float32),
                left_wuji_qpos=np.asarray(
                    last_published_action.left_wuji_qpos,
                    dtype=np.float32,
                ).copy(),
                right_wuji_qpos=np.asarray(
                    last_published_action.right_wuji_qpos,
                    dtype=np.float32,
                ).copy(),
            )
        else:
            action = _limit_hand_qpos_delta(
                candidate,
                previous_action=last_published_action,
                max_delta=runtime_config.get("hand_qpos_max_delta_per_frame"),
            )
        # Physical-policy hands have already been restored to absolute qpos.
        # Clamp again here so the exact command published is always valid.
        action = SonicAction(
            token_state=np.asarray(action.token_state, dtype=np.float32),
            left_wuji_qpos=apply_wuji_qpos_limits(action.left_wuji_qpos),
            right_wuji_qpos=apply_wuji_qpos_limits(action.right_wuji_qpos),
        )
        return action, _raw_action_from_sonic(action)

    def materialize_dynamic_action(
        *,
        context: Any,
        action_index: int,
        snapshot: BaseQuatSnapshot,
        stationary: bool,
        hold_published_hands: bool = False,
    ) -> tuple[SonicAction, np.ndarray]:
        values = adapter.materialize_action(
            context,
            action_index=action_index,
            base_quat_wxyz=snapshot.base_quat_wxyz,
            stationary=stationary,
        )
        if values is None:
            raise RuntimeError("adapter returned no dynamically materialized action")
        _, checked = _split_checked_action_step(values, runtime_config)
        return finalize_action(
            checked,
            hold_published_hands=hold_published_hands,
        )

    def prepare_dynamic_pair(
        *,
        context: Any,
        action_index: int,
        snapshot: BaseQuatSnapshot,
    ) -> tuple[SonicAction, np.ndarray, SonicAction, np.ndarray]:
        try:
            action, raw_action = materialize_dynamic_action(
                context=context,
                action_index=action_index,
                snapshot=snapshot,
                stationary=False,
            )
            stationary_action, stationary_raw_action = materialize_dynamic_action(
                context=context,
                action_index=action_index,
                snapshot=snapshot,
                stationary=True,
            )
        except Exception:
            adapter.rollback_action_materialization(context)
            raise
        adapter.commit_action_materialization(context)
        return action, raw_action, stationary_action, stationary_raw_action

    def request_inference_if_possible(*, force: bool = False) -> None:
        nonlocal frame_index, last_request_time
        if not _chunk_schedule_allows_inference(
            chunk_schedule=chunk_schedule,
            mode=mode,
            has_cached_chunk=cached_sonic_actions is not None,
            chunk_index=chunk_index,
            chunk_end_index=chunk_end_index,
        ):
            return
        now = time.monotonic()
        if not force and now - last_request_time < inference_interval:
            return
        if busy_event.is_set():
            return
        request: dict[str, Any] = {}
        rtc_mode = str(runtime_config.get("rtc_mode_resolved", RTC_MODE_OFF))
        if rtc_mode_uses_overlap(rtc_mode) and mode in {
            BridgeMode.POLICY_READY,
            BridgeMode.POLICY_ARMING,
            BridgeMode.POLICY_RUNNING,
        }:
            rtc_max_delay = _rtc_mode_max_delay(mode=rtc_mode, adapter=adapter)
            prefix_steps = estimate_rtc_prefix_steps(
                runtime_config,
                inference_duration_s=metrics.mean_inference_duration_s,
                publish_rate_hz=publish_rate_hz,
                max_delay_steps=rtc_max_delay,
                action_horizon=adapter.info.action_horizon,
            )
            rtc_prefix = build_rtc_prefix_info(
                cached_raw_chunk,
                chunk_index=chunk_index,
                action_horizon=adapter.info.action_horizon,
                action_dim=adapter.info.action_dim,
                prefix_steps=prefix_steps,
                last_raw_action=last_raw_action,
            )
            if rtc_prefix is not None:
                if bool(runtime_config.get("rtc_debug", False)):
                    print(
                        "[RTC] request "
                        f"mode={rtc_mode} prefix_steps={int(rtc_prefix.mask.sum())} "
                        f"chunk_index={chunk_index} sources={list(rtc_prefix.source_indices)}"
                    )
                if rtc_mode == RTC_MODE_TRAINING_PREFIX:
                    request["rtc_prefix_actions"] = rtc_prefix.actions
                    request["rtc_prefix_mask"] = rtc_prefix.mask
                elif rtc_mode == RTC_MODE_INFERENCE_GUIDANCE:
                    guidance_mask = np.zeros_like(rtc_prefix.actions, dtype=np.float32)
                    guidance_mask[rtc_prefix.mask] = 1.0
                    request["rtc_guidance_target"] = rtc_prefix.actions
                    request["rtc_guidance_mask"] = guidance_mask
                    request["rtc_guidance_scale"] = float(runtime_config.get("rtc_guidance_scale", 1.0))
        try:
            if (
                bool(getattr(source, "observation_alignment_active", False))
                and mode in {BridgeMode.POLICY_READY, BridgeMode.POLICY_ARMING}
                and last_published_action is not None
            ):
                # Planner poses intentionally omit hand commands, so the Wuji
                # server otherwise has no fresh actual-position sample to join.
                # Re-publish the already-held pose once; Planner still owns body.
                publisher.publish(
                    SonicV4Payload(
                        frame_index=frame_index,
                        timestamp_monotonic_ns=time.monotonic_ns(),
                        action=last_published_action,
                    )
                )
                frame_index += 1
            request_queue.put_nowait(request)
            last_request_time = now
        except queue.Full:
            return

    def clear_cached_chunk() -> None:
        nonlocal cached_raw_chunk, cached_sonic_actions, cached_inference_context
        nonlocal chunk_index, chunk_end_index, chunk_reference_committed
        nonlocal last_stationary_action, last_stationary_raw_action
        cached_raw_chunk = None
        cached_sonic_actions = None
        cached_inference_context = None
        last_stationary_action = None
        last_stationary_raw_action = None
        chunk_reference_committed = False
        chunk_index = 0
        chunk_end_index = 0

    def send_planner_idle_now(*, repeats: int = 1) -> None:
        nonlocal last_planner_idle_time
        publisher.send_start(planner=True, repeats=1)
        publisher.send_planner_idle(repeats=repeats)
        last_planner_idle_time = time.monotonic()

    def enter_planner_idle(*, reason: str) -> None:
        nonlocal mode, execute_arm_time
        mode = BridgeMode.PLANNER_IDLE
        execute_arm_time = None
        clear_cached_chunk()
        adapter.reset_execution_context()
        publisher.send_start(planner=True)
        send_planner_idle_now(repeats=3)
        print(f"[INFO] mode -> {mode.value}: {reason}")

    def pause_vla() -> None:
        nonlocal mode, execute_arm_time
        nonlocal paused_hands_open, paused_hands_next_publish_time
        mode = BridgeMode.POLICY_PAUSED
        execute_arm_time = None
        clear_cached_chunk()
        adapter.reset_execution_context()
        send_planner_idle_now(repeats=3)
        paused_hands_open = False
        paused_hands_next_publish_time = time.monotonic()
        print(f"[INFO] mode -> {mode.value}: Policy paused; Planner owns the body")
        print("[INFO] opening both hands with the configured per-frame slew limit")
        print("[INFO] press '4' to return to the initial pose and warm up Policy")

    def publish_paused_open_hands_if_due() -> None:
        nonlocal frame_index, last_published_action, last_raw_action
        nonlocal paused_hands_open, paused_hands_next_publish_time
        if mode != BridgeMode.POLICY_PAUSED or paused_hands_open:
            return
        now = time.monotonic()
        if now < paused_hands_next_publish_time:
            return
        action = _build_paused_open_hands_action(
            previous_action=last_published_action,
            hand_max_delta=runtime_config.get("hand_qpos_max_delta_per_frame", 0.06),
        )
        raw_action = _raw_action_from_sonic(action)
        publisher.publish(
            SonicV4Payload(
                frame_index=frame_index,
                timestamp_monotonic_ns=time.monotonic_ns(),
                action=action,
            )
        )
        frame_index += 1
        paused_hands_next_publish_time = now + publish_period
        last_published_action = action
        last_raw_action = raw_action.copy()
        source.remember_hands(action.left_wuji_qpos, action.right_wuji_qpos)
        source.remember_action(raw_action)
        if _hands_are_open(action):
            paused_hands_open = True
            print("[INFO] both hands are commanded fully open")

    def begin_return_initial() -> None:
        nonlocal mode, execute_arm_time
        nonlocal return_initial_sent, return_initial_next_publish_time
        clear_cached_chunk()
        adapter.reset_execution_context()
        execute_arm_time = None
        return_initial_sent = 0
        return_initial_next_publish_time = time.monotonic()
        mode = BridgeMode.RETURNING_INITIAL
        publisher.send_start(planner=False)
        print(
            f"[INFO] mode -> {mode.value}: publishing initial token "
            f"for {return_initial_repeat} frames at {return_initial_rate_hz:g}Hz"
        )
        print("[INFO] Policy inference remains stopped; 'e' and 'q' stay responsive")

    def finish_return_initial() -> None:
        nonlocal mode, execute_arm_time
        clear_cached_chunk()
        adapter.reset_execution_context()
        execute_arm_time = None
        send_planner_idle_now(repeats=3)
        mode = BridgeMode.POLICY_READY
        print("[INFO] initial-pose return complete; Planner restored")
        print(f"[INFO] mode -> {mode.value}: warming a fresh Policy chunk")
        request_inference_if_possible(force=True)

    def publish_return_initial_if_due() -> None:
        nonlocal frame_index, last_published_action, last_raw_action
        nonlocal return_initial_sent, return_initial_next_publish_time
        if mode != BridgeMode.RETURNING_INITIAL:
            return
        now = time.monotonic()
        if now < return_initial_next_publish_time:
            return
        action = _build_return_initial_action(
            previous_action=last_published_action,
            hand_max_delta=runtime_config.get("hand_qpos_max_delta_per_frame"),
        )
        raw_action = _raw_action_from_sonic(action)
        publisher.publish(
            SonicV4Payload(
                frame_index=frame_index,
                timestamp_monotonic_ns=time.monotonic_ns(),
                action=action,
            )
        )
        frame_index += 1
        return_initial_sent += 1
        return_initial_next_publish_time = now + 1.0 / return_initial_rate_hz
        last_published_action = action
        last_raw_action = raw_action.copy()
        source.remember_hands(action.left_wuji_qpos, action.right_wuji_qpos)
        source.remember_action(raw_action)
        log_every = max(1, int(round(return_initial_rate_hz)))
        if return_initial_sent == 1 or return_initial_sent % log_every == 0:
            print(f"[INFO] returning initial pose: frame={return_initial_sent}/{return_initial_repeat}")
        if return_initial_sent >= return_initial_repeat:
            finish_return_initial()

    def prepare_vla() -> None:
        nonlocal mode, frame_index, last_published_action, last_raw_action, execute_arm_time
        if mode == BridgeMode.POLICY_RUNNING:
            print("[INFO] Policy is already running; press execute key to pause in Planner")
            return
        if mode == BridgeMode.POLICY_ARMING:
            print("[INFO] Policy is already arming; waiting for a post-keypress action chunk")
            return
        if mode == BridgeMode.RETURNING_INITIAL:
            print("[INFO] robot is already returning to the initial pose")
            return
        if mode == BridgeMode.POLICY_PAUSED:
            begin_return_initial()
            return
        execute_arm_time = None
        if mode != BridgeMode.POLICY_READY:
            mode = BridgeMode.POLICY_READY
            print(f"[INFO] mode -> {mode.value}: preparing Policy chunk, robot stays in planner")
            print("[INFO] Policy warmup started: waiting for the first action chunk before pressing execute")
            if bool(runtime_config.get("send_initial_pose_on_start", True)):
                initial_action = build_initial_sonic_action()
                frame_index = _publish_initial_pose(
                    publisher=publisher,
                    runtime_config=runtime_config,
                    frame_index=frame_index,
                )
                last_published_action = initial_action
                last_raw_action = _raw_action_from_sonic(initial_action)
        else:
            print("[INFO] Policy is already ready; refreshing cached action chunk")
        request_inference_if_possible(force=True)

    def start_vla() -> None:
        nonlocal mode, execute_arm_time
        if mode == BridgeMode.POLICY_ARMING:
            print("[INFO] Policy is already arming; robot remains in planner")
            return
        if mode == BridgeMode.RETURNING_INITIAL:
            print("[INFO] execute ignored while returning to the initial pose")
            return
        if mode == BridgeMode.POLICY_PAUSED:
            print("[INFO] Policy is paused; press '4' to return and warm up before executing")
            return
        if mode == BridgeMode.PLANNER_IDLE:
            prepare_vla()
        clear_cached_chunk()
        execute_arm_time = time.monotonic()
        mode = BridgeMode.POLICY_ARMING
        request_inference_if_possible(force=True)
        print(f"[INFO] mode -> {mode.value}: waiting for a post-keypress Policy chunk")
        print("[INFO] robot remains in planner until that fresh chunk is ready")

    def handle_control_command(command: str) -> bool:
        nonlocal estop_requested, normal_quit_requested
        if command == "prepare":
            prepare_vla()
            return False
        if command == "execute":
            if mode == BridgeMode.POLICY_RUNNING:
                pause_vla()
            else:
                start_vla()
            return False
        if command == "quit":
            normal_quit_requested = True
            enter_planner_idle(reason="normal quit requested")
            return True
        if command == "estop":
            estop_requested = True
            print("[ESTOP] software stop requested")
            return True
        return False

    def try_arm_dynamic_policy() -> None:
        nonlocal mode, execute_arm_time
        nonlocal last_stationary_action, last_stationary_raw_action
        if not needs_live_base_quat or mode != BridgeMode.POLICY_ARMING:
            return
        if cached_sonic_actions is None or cached_inference_context is None:
            return
        snapshot = fresh_base_quat()
        if snapshot is None:
            warn_dynamic_orientation("waiting for a fresh base_quat before disabling planner")
            return
        try:
            _, _, stationary_action, stationary_raw_action = prepare_dynamic_pair(
                context=cached_inference_context,
                action_index=chunk_index,
                snapshot=snapshot,
            )
        except Exception as exc:
            warn_dynamic_orientation(f"arming token validation failed; planner remains active: {exc}")
            return
        # Install a validated true-stationary fallback before relinquishing
        # planner control. Any later IMU/encoder failure can now safely hold.
        last_stationary_action = stationary_action
        last_stationary_raw_action = stationary_raw_action
        publisher.send_start(planner=False)
        mode = BridgeMode.POLICY_RUNNING
        execute_arm_time = None
        print("[INFO] fresh post-keypress Policy chunk and base_quat ready; planner disabled")
        print(f"[INFO] mode -> {mode.value}: publishing dynamic v4 token actions")

    def log_runtime_metrics(*, force: bool = False) -> None:
        nonlocal last_metrics_log_time
        now = time.monotonic()
        if not force and now - last_metrics_log_time < metrics_log_interval_s:
            return
        last_metrics_log_time = now
        actual_inf = metrics.actual_inference_hz
        actual_pub = metrics.actual_publish_hz
        mean_infer = metrics.mean_inference_duration_s
        chunk_age = None if metrics.last_chunk_arrival_time is None else now - metrics.last_chunk_arrival_time
        chunk_end = 0 if cached_sonic_actions is None else chunk_end_index
        current_idx = 0 if chunk_end == 0 else min(chunk_index, chunk_end - 1)
        if chunk_schedule == CHUNK_SCHEDULE_SEQUENTIAL:
            inference_summary = f"chunk_rate={_fmt_metric(actual_inf, suffix='Hz')} "
        else:
            inference_summary = (
                f"target_inf={inference_rate_hz:.2f}Hz actual_inf={_fmt_metric(actual_inf, suffix='Hz')} "
            )
        print(
            "[METRIC] "
            f"mode={mode.value} "
            f"schedule={chunk_schedule} "
            f"{inference_summary}"
            f"infer={_fmt_metric(mean_infer, suffix='s')} "
            f"chunk_age={_fmt_metric(chunk_age, suffix='s')} "
            f"publish={_fmt_metric(actual_pub, suffix='Hz')} "
            f"idx={current_idx}/{max(chunk_end - 1, 0)} "
            f"underrun={metrics.chunk_underruns} "
            f"hold={metrics.chunk_hold_steps}"
        )

        warning_interval_s = max(1.0, metrics_log_interval_s)
        can_warn = now - metrics.last_warning_time >= warning_interval_s
        if (
            chunk_schedule == CHUNK_SCHEDULE_PERIODIC
            and actual_inf is not None
            and actual_inf < inference_rate_hz * inference_warn_ratio
            and can_warn
        ):
            print(
                "[WARNING] Policy actual inference rate is below target: "
                f"actual={actual_inf:.2f}Hz target={inference_rate_hz:.2f}Hz"
            )
            metrics.last_warning_time = now
        if chunk_age is not None and chunk_age > chunk_stale_warn_s and can_warn:
            print(f"[WARNING] Policy chunk is stale: age={chunk_age:.2f}s threshold={chunk_stale_warn_s:.2f}s")
            metrics.last_warning_time = now
        if max_chunk_hold_steps > 0 and metrics.chunk_hold_steps >= max_chunk_hold_steps and can_warn:
            print(
                f"[WARNING] Policy chunk underrun: holding last step for {metrics.chunk_hold_steps} publish cycles"
            )
            metrics.last_warning_time = now

    try:
        if bool(runtime_config.get("start_in_planner", True)):
            enter_planner_idle(reason="startup")
        else:
            prepare_vla()

        while True:
            loop_start = time.monotonic()
            poll_base_quat()
            should_exit = False
            try:
                while True:
                    should_exit = handle_control_command(control_queue.get_nowait()) or should_exit
            except queue.Empty:
                pass
            if should_exit:
                break

            try:
                while True:
                    (
                        raw_chunk,
                        sonic_actions,
                        inference_start,
                        inference_done_time,
                        inference_duration_s,
                        inference_context,
                    ) = result_queue.get_nowait()
                    if mode in {
                        BridgeMode.PLANNER_IDLE,
                        BridgeMode.POLICY_PAUSED,
                        BridgeMode.RETURNING_INITIAL,
                    }:
                        continue
                    if mode == BridgeMode.POLICY_ARMING and not _inference_is_fresh_for_arm(
                        inference_start=inference_start,
                        execute_arm_time=execute_arm_time,
                    ):
                        print("[INFO] discarded pre-keypress Policy chunk while arming")
                        request_inference_if_possible(force=True)
                        continue
                    if not _chunk_schedule_allows_inference(
                        chunk_schedule=chunk_schedule,
                        mode=mode,
                        has_cached_chunk=cached_sonic_actions is not None,
                        chunk_index=chunk_index,
                        chunk_end_index=chunk_end_index,
                    ):
                        print("[INFO] discarded action chunk that arrived during sequential execution")
                        continue

                    if len(sonic_actions) < valid_action_horizon:
                        raise RuntimeError(
                            "Policy returned fewer actions than its valid action horizon: "
                            f"got={len(sonic_actions)} valid={valid_action_horizon}"
                        )

                    chunk_arrival_time = time.monotonic()
                    inference_delay = chunk_arrival_time - inference_start
                    cached_raw_chunk = raw_chunk
                    cached_sonic_actions = sonic_actions
                    cached_inference_context = inference_context
                    chunk_reference_committed = False
                    metrics.record_chunk(
                        arrival_time=chunk_arrival_time, inference_duration_s=inference_duration_s
                    )
                    chunk_index = (
                        0
                        if chunk_schedule == CHUNK_SCHEDULE_SEQUENTIAL
                        else _chunk_start_index(
                            runtime_config=runtime_config,
                            inference_delay_s=inference_delay,
                            publish_rate_hz=publish_rate_hz,
                            horizon=valid_action_horizon,
                            execute_chunk_steps=execute_chunk_steps,
                        )
                    )
                    chunk_end_index = _chunk_end_index(
                        start_index=chunk_index,
                        horizon=valid_action_horizon,
                        execute_chunk_steps=execute_chunk_steps,
                    )
                    print(
                        "[INFO] new action chunk: "
                        f"mode={mode.value} h={len(sonic_actions)} valid_h={valid_action_horizon} "
                        f"publish_idx={chunk_index}..{chunk_end_index - 1} "
                        f"infer={inference_duration_s:.3f}s latency={inference_delay:.3f}s "
                        f"done_to_arrival={chunk_arrival_time - inference_done_time:.3f}s"
                    )
                    if bool(runtime_config.get("rtc_debug", False)):
                        first_raw = raw_chunk[min(chunk_index, len(raw_chunk) - 1)] if len(raw_chunk) else None
                        print(
                            "[RTC] boundary "
                            f"mode={runtime_config.get('rtc_mode_resolved', RTC_MODE_OFF)} "
                            f"start_idx={chunk_index} "
                            + summarize_action_delta(
                                last_raw_action,
                                first_raw,
                                token_dim=TOKEN_DIM,
                                hand_dim=WUJI_QPOS_DIM,
                            )
                        )
                    if mode == BridgeMode.POLICY_ARMING and not needs_live_base_quat:
                        publisher.send_start(planner=False)
                        mode = BridgeMode.POLICY_RUNNING
                        execute_arm_time = None
                        print("[INFO] fresh post-keypress Policy chunk ready; planner disabled")
                        print(f"[INFO] mode -> {mode.value}: publishing v4 token actions")

                    if mode == BridgeMode.POLICY_READY:
                        if metrics.chunk_count == 1:
                            print(
                                "[INFO] first Policy chunk ready. This may include model/CUDA warmup; "
                                "wait for a fresh low-latency chunk before pressing '1'."
                            )
                        elif inference_duration_s <= 1.0:
                            print("[INFO] POLICY_READY has a fresh chunk; press '1' to start Policy control.")
            except queue.Empty:
                pass

            try_arm_dynamic_policy()

            if mode == BridgeMode.RETURNING_INITIAL:
                publish_return_initial_if_due()
                if mode == BridgeMode.RETURNING_INITIAL:
                    wait_s = max(return_initial_next_publish_time - time.monotonic(), 0.0)
                    if wait_s > 0.0:
                        time.sleep(min(wait_s, 0.02))
                continue

            if mode in {
                BridgeMode.POLICY_READY,
                BridgeMode.POLICY_ARMING,
                BridgeMode.POLICY_RUNNING,
            }:
                request_inference_if_possible()
            if mode == BridgeMode.POLICY_PAUSED:
                publish_paused_open_hands_if_due()
            if mode in {
                BridgeMode.PLANNER_IDLE,
                BridgeMode.POLICY_READY,
                BridgeMode.POLICY_ARMING,
                BridgeMode.POLICY_PAUSED,
            }:
                now = time.monotonic()
                if now - last_planner_idle_time >= planner_idle_period:
                    send_planner_idle_now()

            if mode != BridgeMode.POLICY_RUNNING:
                if mode in {BridgeMode.POLICY_READY, BridgeMode.POLICY_ARMING}:
                    log_runtime_metrics()
                time.sleep(min(publish_period, 0.05))
                continue

            if cached_sonic_actions is None or cached_raw_chunk is None:
                now = time.monotonic()
                if now - last_wait_log > 1.0:
                    print("[INFO] waiting for first Policy action chunk...")
                    last_wait_log = now
                time.sleep(min(publish_period, 0.05))
                continue

            chunk_end = max(1, chunk_end_index)
            current_idx = min(chunk_index, chunk_end - 1)
            scheduled_hold = chunk_index >= chunk_end
            advance_chunk = True
            if needs_live_base_quat:
                snapshot = fresh_base_quat()
                if snapshot is None:
                    warn_dynamic_orientation("base_quat is missing/stale; publishing the last stationary token")
                    action = last_stationary_action
                    raw_action = last_stationary_raw_action
                    advance_chunk = False
                else:
                    try:
                        if scheduled_hold:
                            action, raw_action = materialize_dynamic_action(
                                context=cached_inference_context,
                                action_index=current_idx,
                                snapshot=snapshot,
                                stationary=True,
                                hold_published_hands=True,
                            )
                            stationary_action = action
                            stationary_raw_action = raw_action
                            advance_chunk = False
                        else:
                            (
                                action,
                                raw_action,
                                stationary_action,
                                stationary_raw_action,
                            ) = prepare_dynamic_pair(
                                context=cached_inference_context,
                                action_index=current_idx,
                                snapshot=snapshot,
                            )
                        last_stationary_action = stationary_action
                        last_stationary_raw_action = stationary_raw_action
                    except Exception as exc:
                        warn_dynamic_orientation(
                            f"token materialization failed; holding last stationary token: {exc}"
                        )
                        action = last_stationary_action
                        raw_action = last_stationary_raw_action
                        advance_chunk = False
                if action is None or raw_action is None:
                    # Arming installs this fallback before planner release, so
                    # this branch should be unreachable. Never substitute the
                    # inference-time static token if the invariant is broken.
                    warn_dynamic_orientation("no validated stationary fallback; refusing to publish or advance")
                    time.sleep(min(publish_period, 0.05))
                    continue
                if not advance_chunk:
                    metrics.record_hold_step(
                        expected=(scheduled_hold and chunk_schedule == CHUNK_SCHEDULE_SEQUENTIAL)
                    )
            else:
                if scheduled_hold:
                    metrics.record_hold_step(expected=chunk_schedule == CHUNK_SCHEDULE_SEQUENTIAL)
                action, raw_action = finalize_action(
                    cached_sonic_actions[current_idx],
                    hold_published_hands=scheduled_hold,
                )
            publisher.publish(
                SonicV4Payload(
                    frame_index=frame_index,
                    timestamp_monotonic_ns=time.monotonic_ns(),
                    action=action,
                )
            )
            metrics.record_publish(publish_time=time.monotonic())
            last_published_action = action
            last_raw_action = raw_action.astype(np.float32, copy=True)
            source.remember_hands(action.left_wuji_qpos, action.right_wuji_qpos)
            source.remember_action(raw_action)
            if (
                chunk_schedule == CHUNK_SCHEDULE_SEQUENTIAL
                and advance_chunk
                and not chunk_reference_committed
                and current_idx == chunk_end - 1
            ):
                adapter.commit_executed_action(
                    cached_inference_context,
                    action_index=current_idx,
                    published_action=raw_action,
                )
                chunk_reference_committed = True
            if frame_index % max(1, int(publish_rate_hz)) == 0:
                print(f"[INFO] sent frame={frame_index} chunk_idx={current_idx}/{chunk_end - 1}")
            log_runtime_metrics()
            frame_index += 1
            if advance_chunk:
                chunk_index += 1

            elapsed = time.monotonic() - loop_start
            remaining = publish_period - elapsed
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("[ESTOP] interrupted; sending software stop")
        estop_requested = True
    finally:
        stop_event.set()
        if estop_requested and bool(runtime_config.get("open_hands_on_estop", False)):
            try:
                frame_index = _publish_open_hands_once(
                    publisher=publisher,
                    frame_index=frame_index,
                    last_action=last_published_action,
                )
                print("[ESTOP] published one open-hand command before stop")
            except Exception as exc:
                print(f"[WARNING] failed to publish open-hand estop command: {exc}")
        if estop_requested:
            print("[ESTOP] sending stop command; use hardware e-stop if the robot is unsafe")
            publisher.send_stop()
        elif normal_quit_requested:
            print("[INFO] normal bridge quit complete; deploy left in planner idle")
        else:
            print("[INFO] bridge exiting; deploy left in current planner/vla state")
        worker.join(timeout=1.0)
        try:
            if base_quat_subscriber is not None:
                base_quat_subscriber.close()
        finally:
            publisher.close()
    return 0
