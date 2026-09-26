"""Synchronized body and WujiHand offline pose replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deploy.reference_alignment import align_qpos_first_frame, quat_yaw_wxyz
from deploy.retarget import validate_g1_qpos
from deploy.wuji_hand import WUJI_OPEN_QPOS, validate_wuji_qpos


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.asarray(alpha, dtype=np.float64).reshape(-1, 1)
    near = sin_theta < 1e-6
    result = np.where(
        near,
        (1.0 - a) * q0 + a * q1,
        np.sin((1.0 - a) * theta) / np.where(near, 1.0, sin_theta) * q0
        + np.sin(a * theta) / np.where(near, 1.0, sin_theta) * q1,
    )
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _interp_rows(
    values: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray
) -> np.ndarray:
    return np.stack(
        [np.interp(dst_t, src_t, values[:, i]) for i in range(values.shape[1])], axis=1
    )


def _resample_qpos(
    qpos: np.ndarray, source_hz: float, target_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_t = np.arange(len(qpos), dtype=np.float64) / source_hz
    count = max(1, int(round(source_t[-1] * target_hz)) + 1)
    target_t = np.arange(count, dtype=np.float64) / target_hz
    target_t = np.minimum(target_t, source_t[-1])
    hi = np.searchsorted(source_t, target_t, side="right")
    hi = np.clip(hi, 1, len(source_t) - 1)
    lo = hi - 1
    denom = source_t[hi] - source_t[lo]
    alpha = np.divide(
        target_t - source_t[lo], denom, out=np.zeros_like(target_t), where=denom > 0
    )
    output = _interp_rows(qpos, source_t, target_t)
    output[:, 3:7] = _slerp(qpos[lo, 3:7], qpos[hi, 3:7], alpha)
    return output.astype(np.float32), lo, hi


@dataclass(slots=True)
class OfflinePose:
    qpos: np.ndarray
    left: np.ndarray
    right: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    filename: str

    def __len__(self) -> int:
        return len(self.qpos)


def load_offline_pose(
    path: str | Path,
    *,
    default_qpos: np.ndarray,
    target_hz: float = 50.0,
    transition_s: float = 0.5,
    end_hold_s: float = 0.5,
) -> OfflinePose:
    path = Path(path)
    raw = dict(np.load(path, allow_pickle=False))
    if "qpos" not in raw:
        raise ValueError(f"{path.name}: required qpos(T,36) is missing")
    qpos = np.asarray(raw["qpos"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2:
        raise ValueError(
            f"{path.name}: qpos must be (T,36) with T>=2, got {qpos.shape}"
        )
    qpos = np.stack([validate_g1_qpos(row) for row in qpos])
    qpos = align_qpos_first_frame(
        qpos,
        target_xy=default_qpos[:2],
        target_yaw=quat_yaw_wxyz(default_qpos[3:7]),
    )
    source_hz = float(np.asarray(raw.get("frequency", 50.0)).reshape(-1)[0])
    if not np.isfinite(source_hz) or source_hz <= 0:
        raise ValueError(f"{path.name}: frequency must be positive")
    body, lo, hi = _resample_qpos(qpos, source_hz, target_hz)

    hand_outputs: dict[str, np.ndarray] = {}
    hand_valid: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        key = f"{side}_wuji_qpos"
        valid_key = f"{side}_wuji_qpos_valid"
        if key not in raw:
            hand_outputs[side] = np.tile(WUJI_OPEN_QPOS, (len(body), 1))
            hand_valid[side] = np.zeros(len(body), dtype=bool)
            continue
        values = np.asarray(raw[key], dtype=np.float32)
        if values.shape != (len(qpos), 20):
            raise ValueError(
                f"{path.name}: {key} must be {(len(qpos), 20)}, got {values.shape}"
            )
        values = np.stack([validate_wuji_qpos(row, name=side)[0] for row in values])
        source_t = np.arange(len(values), dtype=np.float64) / source_hz
        target_t = np.minimum(np.arange(len(body)) / target_hz, source_t[-1])
        hand_outputs[side] = _interp_rows(values, source_t, target_t).astype(np.float32)
        source_valid = np.asarray(
            raw.get(valid_key, np.ones(len(values), dtype=bool)), dtype=bool
        ).reshape(-1)
        if source_valid.shape != (len(values),):
            raise ValueError(f"{path.name}: {valid_key} must be ({len(values)},)")
        hand_valid[side] = source_valid[lo] & source_valid[hi]

    transition = max(0, int(round(transition_s * target_hz)))
    hold = max(0, int(round(end_hold_s * target_hz)))
    default = validate_g1_qpos(default_qpos)
    if transition:
        alpha = np.linspace(0.0, 1.0, transition, endpoint=False, dtype=np.float32)
        start_body = (
            default[None] * (1.0 - alpha[:, None]) + body[0][None] * alpha[:, None]
        )
        start_body[:, 3:7] = _slerp(
            np.tile(default[3:7], (transition, 1)),
            np.tile(body[0, 3:7], (transition, 1)),
            alpha,
        )
        end_alpha = np.linspace(0.0, 1.0, transition, endpoint=False, dtype=np.float32)
        end_body = (
            body[-1][None] * (1.0 - end_alpha[:, None])
            + default[None] * end_alpha[:, None]
        )
        end_body[:, 3:7] = _slerp(
            np.tile(body[-1, 3:7], (transition, 1)),
            np.tile(default[3:7], (transition, 1)),
            end_alpha,
        )
        body = np.concatenate([start_body, body, end_body], axis=0).astype(np.float32)
        for side in ("left", "right"):
            values = hand_outputs[side]
            start = (
                WUJI_OPEN_QPOS[None] * (1.0 - alpha[:, None])
                + values[0][None] * alpha[:, None]
            )
            end = (
                values[-1][None] * (1.0 - end_alpha[:, None])
                + WUJI_OPEN_QPOS[None] * end_alpha[:, None]
            )
            hand_outputs[side] = np.concatenate([start, values, end], axis=0).astype(
                np.float32
            )
            start_valid = bool(hand_valid[side][0])
            end_valid = bool(hand_valid[side][-1])
            hand_valid[side] = np.concatenate(
                [
                    np.full(transition, start_valid, dtype=bool),
                    hand_valid[side],
                    np.full(transition, end_valid, dtype=bool),
                ]
            )
    if hold:
        body = np.concatenate([body, np.tile(default, (hold, 1))])
        for side in ("left", "right"):
            hand_outputs[side] = np.concatenate(
                [hand_outputs[side], np.tile(WUJI_OPEN_QPOS, (hold, 1))]
            )
            hand_valid[side] = np.concatenate(
                [hand_valid[side], np.zeros(hold, dtype=bool)]
            )
    return OfflinePose(
        body,
        hand_outputs["left"],
        hand_outputs["right"],
        hand_valid["left"],
        hand_valid["right"],
        path.name,
    )


def load_offline_directory(
    path: str | Path, *, default_qpos: np.ndarray, target_hz: float = 50.0
) -> list[OfflinePose]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.npz"))
    motions: list[OfflinePose] = []
    for file in files:
        try:
            motions.append(
                load_offline_pose(file, default_qpos=default_qpos, target_hz=target_hz)
            )
        except ValueError as exc:
            print(f"[PoseManager] Skipping offline motion: {exc}")
    return motions
