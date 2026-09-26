"""Pico/MANUS keypoint adapters and WujiHand retarget validation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
WUJI_ROOT = PROJECT_ROOT / ".deps" / "wuji_retargeting"
if not (WUJI_ROOT / "wuji_retargeting").is_dir():
    WUJI_ROOT = REPO_ROOT / "third_party" / "wuji_retargeting"

PICO_TO_MEDIAPIPE = np.array(
    [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25],
    dtype=np.int64,
)
WUJI_QPOS_SIZE = 20
WUJI_OPEN_QPOS = np.zeros(WUJI_QPOS_SIZE, dtype=np.float32)
WUJI_LOWER_LIMITS = np.array(
    [
        0.0,
        -0.1387,
        0.0,
        0.0,
        0.0,
        -0.3700,
        0.0,
        0.0,
        0.0,
        -0.3700,
        0.0,
        0.0,
        0.0,
        -0.3700,
        0.0,
        0.0,
        0.0,
        -0.3700,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
WUJI_UPPER_LIMITS = np.array(
    [
        1.6033,
        0.9324,
        1.5623,
        1.5568,
        1.5604,
        0.3700,
        1.5485,
        1.5753,
        1.5516,
        0.3700,
        1.5512,
        1.5745,
        1.5585,
        0.3700,
        1.5487,
        1.5634,
        1.5585,
        0.3700,
        1.5490,
        1.5735,
    ],
    dtype=np.float32,
)


def validate_keypoints(keypoints: np.ndarray, *, name: str = "hand") -> np.ndarray:
    result = np.asarray(keypoints, dtype=np.float32)
    if result.shape != (21, 3):
        raise ValueError(f"{name} keypoints must be (21, 3), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} keypoints contain NaN/Inf")
    extent = float(np.linalg.norm(np.ptp(result, axis=0)))
    if not 0.04 <= extent <= 0.50:
        raise ValueError(f"{name} scale is implausible: extent={extent:.3f} m")
    return np.ascontiguousarray(result)


def pico_hand_to_mediapipe(hand_state: np.ndarray) -> np.ndarray:
    """Map OpenXR's 26 joints to MediaPipe's 21 landmark order."""
    state = np.asarray(hand_state, dtype=np.float32)
    if state.ndim == 2 and state.shape[0] == 26 and state.shape[1] >= 3:
        positions = state[:, :3]
    elif state.shape == (26, 4, 4):
        positions = state[:, :3, 3]
    else:
        raise ValueError(
            f"Pico hand state must be (26,N>=3) or (26,4,4), got {state.shape}"
        )
    return validate_keypoints(positions[PICO_TO_MEDIAPIPE], name="Pico hand")


@dataclass(slots=True)
class HandPair:
    left: np.ndarray
    left_valid: bool
    right: np.ndarray
    right_valid: bool


class PicoHandSource:
    def __init__(self, xrt_module):
        self._xrt = xrt_module
        self._last_left = np.zeros((21, 3), dtype=np.float32)
        self._last_right = np.zeros((21, 3), dtype=np.float32)

    def _side(self, side: str) -> tuple[np.ndarray, bool]:
        active = getattr(self._xrt, f"get_{side}_hand_is_active")
        getter = getattr(self._xrt, f"get_{side}_hand_tracking_state")
        previous = self._last_left if side == "left" else self._last_right
        try:
            if not bool(active()):
                return previous.copy(), False
            keypoints = pico_hand_to_mediapipe(getter())
        except Exception:
            return previous.copy(), False
        if side == "left":
            self._last_left = keypoints.copy()
        else:
            self._last_right = keypoints.copy()
        return keypoints, True

    def read(self) -> HandPair:
        left, left_valid = self._side("left")
        right, right_valid = self._side("right")
        return HandPair(left, left_valid, right, right_valid)

    def close(self) -> None:
        pass


def validate_wuji_qpos(qpos: np.ndarray, *, name: str) -> tuple[np.ndarray, int]:
    result = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if result.shape != (WUJI_QPOS_SIZE,):
        raise ValueError(f"{name} Wuji qpos must be (20,), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} Wuji qpos contains NaN/Inf")
    clipped = np.clip(result, WUJI_LOWER_LIMITS, WUJI_UPPER_LIMITS)
    count = int(np.count_nonzero(np.abs(clipped - result) > 1e-6))
    return np.ascontiguousarray(clipped, dtype=np.float32), count


class WujiRetargetPair:
    """Own the two stateful Wuji retargeters and their validation counters."""

    def __init__(self, left_config: str | Path, right_config: str | Path):
        root = str(WUJI_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from wuji_retargeting import Retargeter
        except ImportError as exc:
            raise RuntimeError(
                "wuji_retargeting is unavailable; run scripts/setup_pico_env.sh"
            ) from exc
        self.left = Retargeter.from_yaml(str(Path(left_config).resolve()), "left")
        self.right = Retargeter.from_yaml(str(Path(right_config).resolve()), "right")
        self.clipped = 0
        self.invalid = 0

    def retarget(self, pair: HandPair) -> tuple[np.ndarray, bool, np.ndarray, bool]:
        outputs: list[np.ndarray] = []
        validities: list[bool] = []
        for side, keypoints, tracked in (
            ("left", pair.left, pair.left_valid),
            ("right", pair.right, pair.right_valid),
        ):
            if not tracked:
                outputs.append(WUJI_OPEN_QPOS.copy())
                validities.append(False)
                continue
            try:
                qpos = getattr(self, side).retarget(
                    validate_keypoints(keypoints, name=side)
                )
                qpos, clipped = validate_wuji_qpos(qpos, name=side)
                self.clipped += clipped
                outputs.append(qpos)
                validities.append(True)
            except Exception:
                self.invalid += 1
                outputs.append(WUJI_OPEN_QPOS.copy())
                validities.append(False)
        return outputs[0], validities[0], outputs[1], validities[1]


def default_retarget_configs(hand_source: str) -> tuple[Path, Path]:
    config_root = WUJI_ROOT / "example" / "config"
    if hand_source == "manus":
        return (
            config_root / "retarget_manus_left.yaml",
            config_root / "retarget_manus_right.yaml",
        )
    config = config_root / "customer.yaml"
    return config, config
