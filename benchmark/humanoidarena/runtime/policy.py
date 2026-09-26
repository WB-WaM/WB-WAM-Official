from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicyInfo:
    name: str
    action_dim: int
    action_horizon: int
    state_dim: int | None = None
    runtime_version: str = "mock"
    rtc_enabled: bool = False
    rtc_max_delay: int = 0
    valid_action_horizon: int | None = None


class PolicyAdapter(ABC):
    @abstractmethod
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
        """Return an action chunk with shape [H, action_dim]."""

    @property
    @abstractmethod
    def info(self) -> PolicyInfo:
        """Static information about the loaded policy."""

    def take_inference_context(self) -> Any:
        """Return adapter-private context associated with the latest inference."""

        return None

    @property
    def needs_live_base_quat(self) -> bool:
        """Whether actions must be materialized against fresh robot IMU data."""

        return False

    def materialize_action(
        self,
        context: Any,
        *,
        action_index: int,
        base_quat_wxyz: np.ndarray,
        stationary: bool = False,
    ) -> np.ndarray | None:
        """Build one publishable action against the latest robot state.

        Adapters that return ``needs_live_base_quat=True`` must override this
        and return a finite action. For adapters that do not request this
        capability, the runtime ignores this method and uses the cached chunk.
        """

        del context, action_index, base_quat_wxyz, stationary
        return None

    def commit_action_materialization(self, context: Any) -> None:
        """Commit adapter-private state prepared while validating an action."""

        del context

    def rollback_action_materialization(self, context: Any) -> None:
        """Discard adapter-private state from a failed action validation."""

        del context

    def commit_executed_action(
        self,
        context: Any,
        *,
        action_index: int,
        published_action: np.ndarray | None = None,
    ) -> None:
        """Commit the last action actually executed from an accepted chunk.

        ``published_action`` is the final adapter output after runtime safety
        transforms. Physical adapters may use it to reconcile directly
        commanded fields, such as dexterous-hand qpos, with their model-space
        execution context.
        """

        del context, action_index, published_action

    def reset_execution_context(self) -> None:
        """Forget action history when control returns to the planner."""


class MockPolicyAdapter(PolicyAdapter):
    def __init__(
        self,
        *,
        action_horizon: int = 4,
        action_dim: int = 104,
        state_dim: int | None = None,
        latency_s: float = 0.0,
    ):
        self._info = PolicyInfo(
            name="mock",
            action_dim=action_dim,
            action_horizon=action_horizon,
            state_dim=state_dim,
        )
        self._latency_s = float(latency_s)

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
        del observation, rtc_prefix_actions, rtc_prefix_mask
        del rtc_guidance_target, rtc_guidance_mask, rtc_guidance_scale
        if self._latency_s > 0:
            import time

            time.sleep(self._latency_s)
        actions = np.zeros((self._info.action_horizon, self._info.action_dim), dtype=np.float32)
        # Keep the first token value non-zero so dry-run payloads are easy to inspect.
        actions[:, 0] = 0.0625
        return actions

    @property
    def info(self) -> PolicyInfo:
        return self._info
