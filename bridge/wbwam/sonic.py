from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from bridge.common.policy import PolicyAdapter, PolicyInfo
from bridge.sonic.action_schema import ACTION_DIM, TOKEN_DIM, WUJI_QPOS_DIM
from bridge.sonic.encoder import SonicEncoder
from bridge.sonic.encoder_input import (
    build_g1_encoder_input_row,
    build_g1_encoder_inputs,
    full_future_action_horizon,
    future_padding_counts,
)
from bridge.sonic.joint_order import (
    DEFAULT_ANGLES_ISAACLAB,
    ENCODER_VARIANT_SONIC_CANONICAL,
    LEGACY_BRIDGE_DEFAULT_ANGLES_ISAACLAB,
    normalize_encoder_variant,
)
from bridge.sonic.motion_schema import G1MotionChunk
from bridge.wbwam.actions.representation import ActionRepresentation
from bridge.wbwam.actions.schema import PhysicalActionLayout
from bridge.wbwam.actions.trajectory import (
    current_g1_state,
    current_wb_semantic_state,
    physical_actions_to_motion,
    project_live_state_for_physical_wam,
    restore_physical_actions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RELATIVE_BASE_CURR_OBS = "curr_obs"
RELATIVE_BASE_LAST_ACTION = "last_action"
ALLOW_RELATIVE_REFERENCE_MISMATCH_KEY = "allow_relative_reference_mismatch"
RELATIVE_BASE_SOURCES = (RELATIVE_BASE_CURR_OBS, RELATIVE_BASE_LAST_ACTION)
ROOT_ORIENTATION_PREDICTED_RELATIVE = "predicted_relative"
ROOT_ORIENTATION_ACTUAL_IMU = "actual_imu"
ROOT_ORIENTATION_MODES = (
    ROOT_ORIENTATION_PREDICTED_RELATIVE,
    ROOT_ORIENTATION_ACTUAL_IMU,
)


@dataclass
class PhysicalWAMInferenceContext:
    """Physical trajectory retained until its SONIC actions are executed."""

    absolute_actions: np.ndarray
    motion: G1MotionChunk
    policy_context: Any = None
    heading_alignment_rotation: np.ndarray | None = None
    heading_reference_index: int | None = None
    pending_heading_alignment_rotation: np.ndarray | None = None
    pending_heading_reference_index: int | None = None


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _motion_for_encoder_variant(motion: G1MotionChunk, variant: str) -> G1MotionChunk:
    variant = normalize_encoder_variant(variant)
    if variant == ENCODER_VARIANT_SONIC_CANONICAL:
        return motion
    q_offset = LEGACY_BRIDGE_DEFAULT_ANGLES_ISAACLAB - DEFAULT_ANGLES_ISAACLAB
    return G1MotionChunk(
        q=motion.q + q_offset[None, :],
        dq=motion.dq,
        root_rotation=motion.root_rotation,
        timestep_s=motion.timestep_s,
        left_hand_qpos=motion.left_hand_qpos,
        right_hand_qpos=motion.right_hand_qpos,
    )


def _variant_key(variant: str, field: str) -> str:
    return f"{normalize_encoder_variant(variant)}__{field}"


def normalize_relative_base_source(value: Any) -> str:
    key = str(value or RELATIVE_BASE_CURR_OBS).strip().lower().replace("-", "_")
    aliases = {
        "current_obs": RELATIVE_BASE_CURR_OBS,
        "current_observation": RELATIVE_BASE_CURR_OBS,
        "previous_action": RELATIVE_BASE_LAST_ACTION,
    }
    key = aliases.get(key, key)
    if key not in RELATIVE_BASE_SOURCES:
        raise ValueError(f"unsupported relative base source {value!r}; expected one of {RELATIVE_BASE_SOURCES}")
    return key


def validate_relative_reference_base_contract(
    *,
    relative_action_reference: Any,
    relative_base_source: Any,
    allow_relative_reference_mismatch: Any = False,
) -> bool:
    reference = str(relative_action_reference).strip().lower()
    if reference not in {"observation", "first_action"}:
        raise ValueError(f"unsupported relative_action_reference={reference!r}")
    base_source = normalize_relative_base_source(relative_base_source)
    if not isinstance(allow_relative_reference_mismatch, bool):
        raise ValueError(f"sonic_action.{ALLOW_RELATIVE_REFERENCE_MISMATCH_KEY} must be true or false")
    mismatch = reference == "observation" and base_source == RELATIVE_BASE_LAST_ACTION
    if mismatch and not allow_relative_reference_mismatch:
        raise ValueError(
            "relative_action_reference=observation with relative_base_source=last_action "
            f"requires sonic_action.{ALLOW_RELATIVE_REFERENCE_MISMATCH_KEY}=true"
        )
    return mismatch


def normalize_root_orientation_mode(value: Any) -> str:
    key = str(value or ROOT_ORIENTATION_PREDICTED_RELATIVE).strip().lower().replace("-", "_")
    aliases = {
        "static": ROOT_ORIENTATION_PREDICTED_RELATIVE,
        "reference_relative": ROOT_ORIENTATION_PREDICTED_RELATIVE,
        "dynamic": ROOT_ORIENTATION_ACTUAL_IMU,
        "measured": ROOT_ORIENTATION_ACTUAL_IMU,
        "imu": ROOT_ORIENTATION_ACTUAL_IMU,
    }
    key = aliases.get(key, key)
    if key not in ROOT_ORIENTATION_MODES:
        raise ValueError(f"unsupported root orientation mode {value!r}; expected one of {ROOT_ORIENTATION_MODES}")
    return key


class PhysicalWAMToSonicAdapter(PolicyAdapter):
    """Turn a physical WAM chunk into SONIC token + Wuji actions."""

    def __init__(
        self,
        policy: PolicyAdapter,
        *,
        representation: ActionRepresentation,
        action_config: dict[str, Any],
        state_layout: str | None,
    ) -> None:
        if representation == ActionRepresentation.SONIC_TOKEN:
            raise ValueError("PhysicalWAMToSonicAdapter requires a physical representation")
        self._policy = policy
        self._representation = representation
        self._config = dict(action_config)
        self._state_layout = state_layout
        self._layout = PhysicalActionLayout.from_config(action_config)
        self._timestep_s = float(action_config.get("timestep_s", 0.05))
        self._future_dt_s = float(action_config.get("future_dt_s", 0.1))
        self._relative_hands = bool(action_config.get("relative_hands", True))
        self._relative_joint_ranges = action_config.get("relative_joint_ranges")
        self._relative_action_reference = (
            str(action_config.get("relative_action_reference", "observation")).strip().lower()
        )
        self._relative_base_source = normalize_relative_base_source(action_config.get("relative_base_source"))
        self._intentional_reference_mismatch = validate_relative_reference_base_contract(
            relative_action_reference=self._relative_action_reference,
            relative_base_source=self._relative_base_source,
            allow_relative_reference_mismatch=action_config.get(ALLOW_RELATIVE_REFERENCE_MISMATCH_KEY, False),
        )
        self._root_orientation_mode = normalize_root_orientation_mode(action_config.get("root_orientation_mode"))
        self._project_model_state = bool(action_config.get("project_model_state", True))
        has_relative_actions = bool(self._relative_joint_ranges)
        if self._relative_action_reference == "first_action" and not has_relative_actions:
            raise ValueError("relative_action_reference=first_action requires relative_joint_ranges")
        if self._relative_base_source == RELATIVE_BASE_LAST_ACTION:
            if not has_relative_actions:
                raise ValueError(
                    "sonic_action.relative_base_source=last_action requires a relative action representation"
                )
        self._encoder_variant = normalize_encoder_variant(action_config.get("encoder_variant"))
        shadow_variants = action_config.get("encoder_shadow_variants") or ()
        self._encoder_shadow_variants = tuple(
            dict.fromkeys(normalize_encoder_variant(item) for item in shadow_variants)
        )
        capture_root = action_config.get("encoder_ab_capture_dir")
        self._encoder_ab_capture_dir: Path | None = None
        self._encoder_ab_capture_index = 0
        image_capture_interval_s = float(action_config.get("encoder_ab_image_capture_interval_s", 0.0))
        if not np.isfinite(image_capture_interval_s) or image_capture_interval_s < 0.0:
            raise ValueError("encoder_ab_image_capture_interval_s must be finite and >= 0")
        self._encoder_ab_image_capture_interval_ns = int(image_capture_interval_s * 1e9)
        self._encoder_ab_last_image_capture_ns: int | None = None
        self._encoder_ab_image_dir: Path | None = None
        if capture_root is not None:
            root = _resolve_repo_path(capture_root)
            session = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
            self._encoder_ab_capture_dir = root / session
            self._encoder_ab_capture_dir.mkdir(parents=True, exist_ok=False)
            if self._encoder_ab_image_capture_interval_ns > 0:
                self._encoder_ab_image_dir = self._encoder_ab_capture_dir / "images"
                self._encoder_ab_image_dir.mkdir()
            print(
                "[EncoderAB] shadow capture enabled: "
                f"selected={self._encoder_variant} shadows={self._encoder_shadow_variants} "
                f"dir={self._encoder_ab_capture_dir}"
            )
        elif self._encoder_shadow_variants:
            raise ValueError("sonic_action.encoder_shadow_variants requires sonic_action.encoder_ab_capture_dir")
        encoder_model = action_config.get(
            "encoder_model",
            "checkpoints/tracker/sonic/policy/release/model_encoder.onnx",
        )
        providers = action_config.get("encoder_providers")
        self._encoder = SonicEncoder(
            _resolve_repo_path(encoder_model),
            providers=None if providers is None else list(providers),
            snap_fsq_grid=bool(action_config.get("snap_fsq_grid", True)),
        )
        self._dynamic_encoder: SonicEncoder | None = None
        if self._root_orientation_mode == ROOT_ORIENTATION_ACTUAL_IMU:
            dynamic_providers = action_config.get("dynamic_encoder_providers", ["CPUExecutionProvider"])
            self._dynamic_encoder = SonicEncoder(
                _resolve_repo_path(encoder_model),
                providers=list(dynamic_providers),
                snap_fsq_grid=bool(action_config.get("snap_fsq_grid", True)),
            )
        self._execution_context_lock = threading.Lock()
        self._latest_inference_context: PhysicalWAMInferenceContext | None = None
        self._committed_action_base: np.ndarray | None = None
        self._info = PolicyInfo(
            name=f"{policy.info.name}+sonic_encoder",
            action_dim=ACTION_DIM,
            action_horizon=policy.info.action_horizon,
            state_dim=policy.info.state_dim,
            runtime_version=(
                f"{policy.info.runtime_version}+sonic_encoder"
                f"+relative_base:{self._relative_base_source}"
                "+model_proprio:curr_obs"
                f"+root_orientation:{self._root_orientation_mode}"
            ),
            valid_action_horizon=policy.info.action_horizon,
        )
        self._complete_future_action_horizon = full_future_action_horizon(
            horizon=policy.info.action_horizon,
            timestep_s=self._timestep_s,
            future_dt_s=self._future_dt_s,
        )
        if has_relative_actions:
            fallback = (
                "; first chunk falls back to curr_obs"
                if self._relative_base_source == RELATIVE_BASE_LAST_ACTION
                else ""
            )
            print(f"[WAM] Relative action restore base: {self._relative_base_source} {fallback}")
            print("[WAM] Model proprio source: curr_obs (live measured state)")
            if self._intentional_reference_mismatch:
                print(
                    "[WAM] Intentional reference mismatch: training=curr_obs, deployment=previous published target"
                )
        print(f"[WAM] SONIC root orientation: {self._root_orientation_mode}")

    def _encode_variants(
        self,
        motion: G1MotionChunk,
    ) -> dict[str, dict[str, np.ndarray | G1MotionChunk]]:
        variants = tuple(dict.fromkeys((self._encoder_variant, *self._encoder_shadow_variants)))
        encoded: dict[str, dict[str, np.ndarray | G1MotionChunk]] = {}
        for variant in variants:
            variant_motion = _motion_for_encoder_variant(motion, variant)
            encoder_inputs = build_g1_encoder_inputs(
                variant_motion,
                future_dt_s=self._future_dt_s,
            )
            tokens = self._encoder.encode(encoder_inputs)
            assert variant_motion.left_hand_qpos is not None
            assert variant_motion.right_hand_qpos is not None
            actions = np.concatenate(
                [
                    tokens,
                    variant_motion.left_hand_qpos,
                    variant_motion.right_hand_qpos,
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            encoded[variant] = {
                "motion": variant_motion,
                "encoder_inputs": encoder_inputs,
                "tokens": tokens,
                "actions": actions,
            }
        return encoded

    def _capture_encoder_ab(
        self,
        *,
        live_state: np.ndarray,
        model_state: np.ndarray,
        raw_actions: np.ndarray,
        relative_base: np.ndarray,
        relative_base_mode: str,
        relative_base_source: str,
        absolute_actions: np.ndarray,
        encoded: dict[str, dict[str, np.ndarray | G1MotionChunk]],
    ) -> Path | None:
        if self._encoder_ab_capture_dir is None:
            return None
        self._encoder_ab_capture_index += 1
        path = self._encoder_ab_capture_dir / f"chunk_{self._encoder_ab_capture_index:06d}.npz"
        payload: dict[str, np.ndarray] = {
            "format_version": np.asarray([1], dtype=np.int64),
            "captured_monotonic_ns": np.asarray([time.monotonic_ns()], dtype=np.int64),
            "timestep_s": np.asarray([self._timestep_s], dtype=np.float64),
            "future_dt_s": np.asarray([self._future_dt_s], dtype=np.float64),
            "selected_variant": np.asarray([self._encoder_variant]),
            "root_orientation_mode": np.asarray([self._root_orientation_mode]),
            "valid_action_horizon": np.asarray([self._info.valid_action_horizon], dtype=np.int64),
            "complete_future_action_horizon": np.asarray([self._complete_future_action_horizon], dtype=np.int64),
            "future_padding_counts": future_padding_counts(
                horizon=self._info.action_horizon,
                timestep_s=self._timestep_s,
                future_dt_s=self._future_dt_s,
            ),
            "live_state": np.asarray(live_state, dtype=np.float32),
            "model_state": np.asarray(model_state, dtype=np.float32),
            "wam_actions": np.asarray(raw_actions, dtype=np.float32),
            "relative_base": np.asarray(relative_base, dtype=np.float32),
            "relative_base_mode": np.asarray([relative_base_mode]),
            "relative_base_source": np.asarray([relative_base_source]),
            "model_state_source": np.asarray([RELATIVE_BASE_CURR_OBS]),
            "absolute_actions": np.asarray(absolute_actions, dtype=np.float32),
        }
        for variant, item in encoded.items():
            motion = item["motion"]
            assert isinstance(motion, G1MotionChunk)
            payload[_variant_key(variant, "q")] = motion.q
            payload[_variant_key(variant, "dq")] = motion.dq
            payload[_variant_key(variant, "root_rotation")] = motion.root_rotation
            payload[_variant_key(variant, "encoder_inputs")] = np.asarray(item["encoder_inputs"])
            payload[_variant_key(variant, "tokens")] = np.asarray(item["tokens"])
            payload[_variant_key(variant, "actions")] = np.asarray(item["actions"])
        np.savez_compressed(path, **payload)
        if self._encoder_ab_capture_index == 1 or self._encoder_ab_capture_index % 10 == 0:
            print(f"[EncoderAB] captured chunk {self._encoder_ab_capture_index}: {path}")
        return path

    def _capture_source_image(self, source_image: Any) -> Path | None:
        if self._encoder_ab_image_dir is None or source_image is None:
            return None
        captured_monotonic_ns = time.monotonic_ns()
        if (
            self._encoder_ab_last_image_capture_ns is not None
            and captured_monotonic_ns - self._encoder_ab_last_image_capture_ns
            < self._encoder_ab_image_capture_interval_ns
        ):
            return None

        from PIL import Image

        image = np.asarray(source_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"capture source image must be HxWx3 RGB, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)
        chunk_index = self._encoder_ab_capture_index + 1
        path = self._encoder_ab_image_dir / (f"chunk_{chunk_index:06d}_{captured_monotonic_ns}.jpg")
        Image.fromarray(image, mode="RGB").save(path, format="JPEG", quality=95)
        self._encoder_ab_last_image_capture_ns = captured_monotonic_ns
        print(f"[EncoderAB] captured input image: {path}")
        return path

    def infer(
        self,
        observation: dict[str, Any],
        *,
        rtc_prefix_actions: np.ndarray | None = None,
        rtc_prefix_mask: np.ndarray | None = None,
        rtc_guidance_target: np.ndarray | None = None,
        rtc_guidance_mask: np.ndarray | None = None,
        rtc_guidance_scale: float | None = None,
    ) -> np.ndarray:
        live_state = np.asarray(observation["states"], dtype=np.float32).reshape(-1)
        live_current = current_g1_state(live_state, state_layout=self._state_layout)
        with self._execution_context_lock:
            committed_action_base = (
                None if self._committed_action_base is None else self._committed_action_base.copy()
            )
        use_last_action = (
            self._relative_base_source == RELATIVE_BASE_LAST_ACTION and committed_action_base is not None
        )
        restore_base = (
            current_g1_state(committed_action_base, state_layout=self._state_layout)
            if use_last_action
            else live_current
        )
        resolved_base_source = RELATIVE_BASE_LAST_ACTION if use_last_action else RELATIVE_BASE_CURR_OBS
        model_observation = observation
        if self._project_model_state:
            model_observation = dict(observation)
            # Relative action targets and model proprio have different bases.
            # The 0724_norm training runs only changed the action target; their
            # proprio continues to come from the measured ``states[...]``
            # slices. Never replace it with the previous commanded action.
            model_observation["states"] = project_live_state_for_physical_wam(
                live_state,
                state_layout=self._state_layout,
            )
        raw_actions = self._policy.infer(
            model_observation,
            rtc_prefix_actions=rtc_prefix_actions,
            rtc_prefix_mask=rtc_prefix_mask,
            rtc_guidance_target=rtc_guidance_target,
            rtc_guidance_mask=rtc_guidance_mask,
            rtc_guidance_scale=rtc_guidance_scale,
        )
        take_policy_context = getattr(self._policy, "take_inference_context", None)
        policy_context = take_policy_context() if callable(take_policy_context) else None
        if self._relative_action_reference == "first_action":
            raw_actions = np.asarray(raw_actions)
            if raw_actions.ndim != 2 or raw_actions.shape[0] == 0:
                raise ValueError(
                    f"first_action clamp requires physical actions with shape [H,D], got {raw_actions.shape}"
                )
            raw_actions = raw_actions.copy()
            for start, end in self._relative_joint_ranges:
                raw_actions[0, start:end] = 0.0
                if np.any(raw_actions[0, start:end] != 0.0):
                    raise RuntimeError(f"failed to zero first_action relative_joint_range [{start}, {end})")
        absolute_actions = restore_physical_actions(
            raw_actions,
            restore_base,
            representation=self._representation,
            relative_hands=self._relative_hands,
            relative_joint_ranges=self._relative_joint_ranges,
        )
        padding_mask_value = getattr(policy_context, "padding_mask", None)
        if padding_mask_value is not None:
            padding_mask = np.asarray(padding_mask_value, dtype=bool)
            expected_shape = (absolute_actions.shape[0],)
            if padding_mask.shape != expected_shape:
                raise ValueError(
                    f"policy context padding_mask must have shape {expected_shape}, got {padding_mask.shape}"
                )
            if np.any(padding_mask):
                absolute_actions = absolute_actions.copy()
                absolute_actions[padding_mask, self._layout.root3.stop - 1] = 0.0
        canonical_motion = physical_actions_to_motion(
            absolute_actions,
            restore_base,
            representation=ActionRepresentation.ABSOLUTE,
            timestep_s=self._timestep_s,
            layout=self._layout,
            previous_body_q_delta=getattr(
                policy_context,
                "previous_body_q_delta",
                None,
            ),
        )
        encoded = self._encode_variants(canonical_motion)
        model_state = np.asarray(model_observation["states"], dtype=np.float32).reshape(-1)
        self._capture_source_image(observation.get("observation/image"))
        self._capture_encoder_ab(
            live_state=live_state,
            model_state=model_state,
            raw_actions=np.asarray(raw_actions, dtype=np.float32),
            relative_base=current_wb_semantic_state(restore_base),
            relative_base_mode=self._relative_base_source,
            relative_base_source=resolved_base_source,
            absolute_actions=absolute_actions,
            encoded=encoded,
        )
        selected_motion = encoded[self._encoder_variant]["motion"]
        assert isinstance(selected_motion, G1MotionChunk)
        with self._execution_context_lock:
            self._latest_inference_context = PhysicalWAMInferenceContext(
                absolute_actions=absolute_actions.copy(),
                motion=selected_motion,
                policy_context=policy_context,
            )
        selected = encoded[self._encoder_variant]["actions"]
        return np.asarray(selected, dtype=np.float32)

    def take_inference_context(self) -> PhysicalWAMInferenceContext | None:
        with self._execution_context_lock:
            context = self._latest_inference_context
            self._latest_inference_context = None
        return context

    @property
    def needs_live_base_quat(self) -> bool:
        return self._root_orientation_mode == ROOT_ORIENTATION_ACTUAL_IMU

    def materialize_action(
        self,
        context: Any,
        *,
        action_index: int,
        base_quat_wxyz: np.ndarray,
        stationary: bool = False,
    ) -> np.ndarray | None:
        if not self.needs_live_base_quat:
            return None
        if not isinstance(context, PhysicalWAMInferenceContext):
            raise TypeError(
                f"dynamic SONIC encoding requires PhysicalWAMInferenceContext, got {type(context).__name__}"
            )
        index = int(action_index)
        if not 0 <= index < context.motion.horizon:
            raise IndexError(f"action index {index} is outside physical motion horizon {context.motion.horizon}")
        # The alignment is fixed at the first action that this chunk actually
        # publishes. This is index 0 for sequential scheduling, while retained
        # periodic modes may deliberately skip a stale prefix.
        if context.heading_alignment_rotation is None and context.pending_heading_alignment_rotation is None:
            context.pending_heading_alignment_rotation = context.motion.heading_alignment_rotation(
                base_quat_wxyz,
                reference_index=index,
            )
            context.pending_heading_reference_index = index
        heading_alignment = (
            context.heading_alignment_rotation
            if context.heading_alignment_rotation is not None
            else context.pending_heading_alignment_rotation
        )
        assert heading_alignment is not None
        assert self._dynamic_encoder is not None
        encoder_input = build_g1_encoder_input_row(
            context.motion,
            action_index=index,
            actual_base_quat_wxyz=base_quat_wxyz,
            heading_alignment_rotation=heading_alignment,
            future_dt_s=self._future_dt_s,
            stationary=stationary,
        )
        token = self._dynamic_encoder.encode(encoder_input)[0]
        assert context.motion.left_hand_qpos is not None
        assert context.motion.right_hand_qpos is not None
        action = np.concatenate(
            [
                token,
                context.motion.left_hand_qpos[index],
                context.motion.right_hand_qpos[index],
            ]
        ).astype(np.float32, copy=False)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(f"dynamic SONIC action must be finite {(ACTION_DIM,)}, got {action.shape}")
        return action

    def commit_action_materialization(self, context: Any) -> None:
        if not isinstance(context, PhysicalWAMInferenceContext):
            return
        if context.heading_alignment_rotation is None and context.pending_heading_alignment_rotation is not None:
            context.heading_alignment_rotation = context.pending_heading_alignment_rotation
            context.heading_reference_index = context.pending_heading_reference_index
        context.pending_heading_alignment_rotation = None
        context.pending_heading_reference_index = None

    def rollback_action_materialization(self, context: Any) -> None:
        if not isinstance(context, PhysicalWAMInferenceContext):
            return
        context.pending_heading_alignment_rotation = None
        context.pending_heading_reference_index = None

    def commit_executed_action(
        self,
        context: Any,
        *,
        action_index: int,
        published_action: np.ndarray | None = None,
    ) -> None:
        policy_context = context.policy_context if isinstance(context, PhysicalWAMInferenceContext) else None
        if context is not None and self._relative_base_source == RELATIVE_BASE_LAST_ACTION:
            if isinstance(context, PhysicalWAMInferenceContext):
                actions = np.asarray(context.absolute_actions, dtype=np.float32)
            else:
                # Compatibility for captures/tests created before the
                # structured inference context was introduced.
                actions = np.asarray(context, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] < self._layout.minimum_dim:
                raise ValueError(f"invalid WBWAM execution context shape {actions.shape}")
            index = int(action_index)
            if not 0 <= index < actions.shape[0]:
                raise IndexError(f"executed action index {index} is outside horizon {actions.shape[0]}")
            committed = actions[index].copy()
            if published_action is not None:
                published = np.asarray(published_action, dtype=np.float32).reshape(-1)
                if published.shape != (ACTION_DIM,) or not np.isfinite(published).all():
                    raise ValueError(
                        f"published SONIC action must be finite {(ACTION_DIM,)}, got {published.shape}"
                    )
                left_start = TOKEN_DIM
                right_start = TOKEN_DIM + WUJI_QPOS_DIM
                committed[self._layout.left_hand] = published[left_start : left_start + WUJI_QPOS_DIM]
                committed[self._layout.right_hand] = published[right_start : right_start + WUJI_QPOS_DIM]
            with self._execution_context_lock:
                self._committed_action_base = committed

        # The inner adapter owns model-space context. Do not forward the
        # outer 104D SONIC action as ``published_action`` to a 72D policy.
        commit_policy_action = getattr(self._policy, "commit_executed_action", None)
        if callable(commit_policy_action):
            commit_policy_action(
                policy_context,
                action_index=action_index,
                published_action=None,
            )

    def reset_execution_context(self) -> None:
        with self._execution_context_lock:
            self._latest_inference_context = None
            self._committed_action_base = None
        reset_policy_context = getattr(self._policy, "reset_execution_context", None)
        if callable(reset_policy_context):
            reset_policy_context()

    @property
    def info(self) -> PolicyInfo:
        return self._info
