#!/usr/bin/env python3
"""Shared MANUS Integrated SDK skeleton provider."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
MANUS_SOURCE_DIR = (
    REPO_ROOT
    / "third_party"
    / "MANUS_SDK"
    / "ManusSDK_v3.1.1"
    / "SDKClient_Linux"
)
MANUS_BINDING_DIR = REPO_ROOT / "tracker" / "sonic" / ".deps" / "manus_sdk"
if not MANUS_BINDING_DIR.is_dir():
    MANUS_BINDING_DIR = MANUS_SOURCE_DIR
if MANUS_BINDING_DIR.is_dir():
    _binding_dir = str(MANUS_BINDING_DIR)
    if _binding_dir not in sys.path:
        sys.path.insert(0, _binding_dir)

try:
    import ManusServer  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "ManusServer pybind module is missing or built for a different Python. "
        "Run scripts/env/setup_envs.sh teleop to build it for this interpreter."
    ) from exc

DEFAULT_MANUS_CALIBRATION_DIR = REPO_ROOT / "third_party" / "MANUS_SDK" / "calibrations"
DEFAULT_MANUS_CALIBRATION_FILE = DEFAULT_MANUS_CALIBRATION_DIR / "Calibration.mcal"
DEFAULT_MANUS_LEFT_CALIBRATION_FILE = DEFAULT_MANUS_CALIBRATION_DIR / "Calibration_left.mcal"
DEFAULT_MANUS_RIGHT_CALIBRATION_FILE = DEFAULT_MANUS_CALIBRATION_DIR / "Calibration_right.mcal"
DEFAULT_MANUS_STALE_S = 0.2
DEFAULT_MANUS_INIT_TIMEOUT_S = 10.0
DEFAULT_MANUS_MCP_JOINT = "proximal"
VALID_MANUS_MCP_JOINTS = ("proximal", "metacarpal")

# Fallback only. Prefer SDK NodeInfo below because raw node order may differ by SDK/device.
MANUS_RAW_TO_MEDIAPIPE = np.array(
    [0, 21, 22, 23, 24, 2, 3, 4, 5, 7, 8, 9, 10, 17, 18, 19, 20, 12, 13, 14, 15],
    dtype=np.int64,
)

CHAIN_TYPE_FINGER_THUMB = 5
CHAIN_TYPE_FINGER_INDEX = 6
CHAIN_TYPE_FINGER_MIDDLE = 7
CHAIN_TYPE_FINGER_RING = 8
CHAIN_TYPE_FINGER_PINKY = 9
CHAIN_TYPE_HAND = 13

FINGER_JOINT_INVALID = 0
FINGER_JOINT_METACARPAL = 1
FINGER_JOINT_PROXIMAL = 2
FINGER_JOINT_INTERMEDIATE = 3
FINGER_JOINT_DISTAL = 4
FINGER_JOINT_TIP = 5

CHAIN_LABELS = {
    0: "invalid",
    CHAIN_TYPE_FINGER_THUMB: "thumb",
    CHAIN_TYPE_FINGER_INDEX: "index",
    CHAIN_TYPE_FINGER_MIDDLE: "middle",
    CHAIN_TYPE_FINGER_RING: "ring",
    CHAIN_TYPE_FINGER_PINKY: "pinky",
    CHAIN_TYPE_HAND: "hand",
}
JOINT_LABELS = {
    FINGER_JOINT_INVALID: "invalid",
    FINGER_JOINT_METACARPAL: "metacarpal",
    FINGER_JOINT_PROXIMAL: "proximal",
    FINGER_JOINT_INTERMEDIATE: "intermediate",
    FINGER_JOINT_DISTAL: "distal",
    FINGER_JOINT_TIP: "tip",
}
FINGER_TO_MEDIAPIPE = {
    CHAIN_TYPE_FINGER_THUMB: (1, 2, 3, 4),
    CHAIN_TYPE_FINGER_INDEX: (5, 6, 7, 8),
    CHAIN_TYPE_FINGER_MIDDLE: (9, 10, 11, 12),
    CHAIN_TYPE_FINGER_RING: (13, 14, 15, 16),
    CHAIN_TYPE_FINGER_PINKY: (17, 18, 19, 20),
}
THUMB_JOINT_SEQUENCE = (
    FINGER_JOINT_METACARPAL,
    FINGER_JOINT_PROXIMAL,
    FINGER_JOINT_INTERMEDIATE,
    FINGER_JOINT_TIP,
)
NON_THUMB_PROXIMAL_MCP_SEQUENCE = (
    FINGER_JOINT_PROXIMAL,
    FINGER_JOINT_INTERMEDIATE,
    FINGER_JOINT_DISTAL,
    FINGER_JOINT_TIP,
)
NON_THUMB_METACARPAL_MCP_SEQUENCE = (
    FINGER_JOINT_METACARPAL,
    FINGER_JOINT_PROXIMAL,
    FINGER_JOINT_INTERMEDIATE,
    FINGER_JOINT_TIP,
)


def _state_id(state: dict, key: str) -> int | None:
    values = state.get(key)
    if not values:
        return None
    glove_id = int(values[0])
    return glove_id if glove_id > 0 else None


def _as_int_array(values: object, *, expected_len: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.shape[0] < expected_len:
        return None
    return arr[:expected_len]


def _finger_indices_by_joint(
    *,
    chain_type: np.ndarray,
    joint_type: np.ndarray,
    chain_id: int,
    desired_joints: tuple[int, int, int, int],
) -> list[int] | None:
    selected: list[int] = []
    for joint_id in desired_joints:
        matches = np.flatnonzero((chain_type == chain_id) & (joint_type == joint_id))
        if matches.size == 0:
            return None
        selected.append(int(matches[0]))
    return selected


def _fallback_finger_indices(
    *,
    chain_type: np.ndarray,
    joint_type: np.ndarray,
    chain_id: int,
    prefer_tail: bool,
) -> list[int] | None:
    matches = np.flatnonzero(chain_type == chain_id)
    if matches.size < 4:
        return None
    ordered = sorted((int(idx) for idx in matches), key=lambda idx: (int(joint_type[idx]), idx))
    return ordered[-4:] if prefer_tail else ordered[:4]


def _raw_position_to_keypoints(
    raw_position: object,
    *,
    chain_type: object | None = None,
    joint_type: object | None = None,
    mcp_joint: str = "proximal",
    return_mapping: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[int]]:
    raw = np.asarray(raw_position, dtype=np.float32).reshape(-1, 3)
    mapping: list[int]

    chain = _as_int_array(chain_type, expected_len=raw.shape[0])
    joint = _as_int_array(joint_type, expected_len=raw.shape[0])
    if chain is not None and joint is not None and any(chain == CHAIN_TYPE_FINGER_THUMB):
        keypoints = np.full((21, 3), np.nan, dtype=np.float32)
        mapping = [-1] * 21
        hand_matches = np.flatnonzero(chain == CHAIN_TYPE_HAND)
        wrist_idx = int(hand_matches[0]) if hand_matches.size else 0
        keypoints[0] = raw[wrist_idx]
        mapping[0] = wrist_idx

        non_thumb_sequence = (
            NON_THUMB_METACARPAL_MCP_SEQUENCE
            if mcp_joint == "metacarpal"
            else NON_THUMB_PROXIMAL_MCP_SEQUENCE
        )
        for finger_chain, mp_indices in FINGER_TO_MEDIAPIPE.items():
            if finger_chain == CHAIN_TYPE_FINGER_THUMB:
                desired = THUMB_JOINT_SEQUENCE
                prefer_tail = False
            else:
                desired = non_thumb_sequence
                prefer_tail = mcp_joint == "proximal"
            raw_indices = _finger_indices_by_joint(
                chain_type=chain,
                joint_type=joint,
                chain_id=finger_chain,
                desired_joints=desired,
            )
            if raw_indices is None:
                raw_indices = _fallback_finger_indices(
                    chain_type=chain,
                    joint_type=joint,
                    chain_id=finger_chain,
                    prefer_tail=prefer_tail,
                )
            if raw_indices is None:
                raise ValueError(f"Missing MANUS nodes for {CHAIN_LABELS.get(finger_chain, finger_chain)}")
            for mp_idx, raw_idx in zip(mp_indices, raw_indices):
                keypoints[mp_idx] = raw[raw_idx]
                mapping[mp_idx] = raw_idx
    elif raw.shape[0] == 21:
        keypoints = raw
        mapping = list(range(21))
    elif raw.shape[0] >= 25:
        keypoints = raw[MANUS_RAW_TO_MEDIAPIPE]
        mapping = [int(idx) for idx in MANUS_RAW_TO_MEDIAPIPE]
    else:
        raise ValueError(f"Expected at least 21 MANUS raw nodes, got {raw.shape[0]}")

    if keypoints.shape != (21, 3):
        raise ValueError(f"Expected keypoint shape (21, 3), got {keypoints.shape}")
    if not np.all(np.isfinite(keypoints)):
        raise ValueError("MANUS keypoints contain non-finite values")
    keypoints = keypoints.astype(np.float32, copy=False)
    if return_mapping:
        return keypoints, mapping
    return keypoints


def _print_node_info(
    *,
    side: str,
    glove_id: int,
    position: object,
    chain_type: object | None,
    joint_type: object | None,
    node_id: object | None,
    parent_id: object | None,
    mapping: list[int],
) -> None:
    raw = np.asarray(position, dtype=np.float32).reshape(-1, 3)
    chain = _as_int_array(chain_type, expected_len=raw.shape[0])
    joint = _as_int_array(joint_type, expected_len=raw.shape[0])
    nodes = _as_int_array(node_id, expected_len=raw.shape[0])
    parents = _as_int_array(parent_id, expected_len=raw.shape[0])
    print(f"[MANUS:NODE] {side} glove=0x{glove_id:08X} raw_nodes={raw.shape[0]} mp_mapping={mapping}")
    if chain is None or joint is None:
        print("[MANUS:NODE] chain/joint metadata unavailable; using fallback index mapping")
        return
    for idx in range(raw.shape[0]):
        node = int(nodes[idx]) if nodes is not None else idx
        parent = int(parents[idx]) if parents is not None else -1
        chain_label = CHAIN_LABELS.get(int(chain[idx]), str(int(chain[idx])))
        joint_label = JOINT_LABELS.get(int(joint[idx]), str(int(joint[idx])))
        print(
            f"[MANUS:NODE] idx={idx:02d} node={node:03d} parent={parent:03d} "
            f"chain={chain_label:<6} joint={joint_label:<12} "
            f"pos=({raw[idx,0]:+.4f},{raw[idx,1]:+.4f},{raw[idx,2]:+.4f})"
        )


class ManusIntegratedProvider:
    def __init__(
        self,
        *,
        stale_s: float,
        init_timeout_s: float,
        calibration_file: str,
        left_calibration_file: str,
        right_calibration_file: str,
        load_calibration: bool,
        debug_node_info: bool,
        mcp_joint: str,
    ):
        if stale_s <= 0.0:
            raise ValueError("--stale-s must be > 0")
        self.stale_s = float(stale_s)
        self._latest: dict[str, dict[str, object] | None] = {"left": None, "right": None}
        self._last_error_log_at: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._debug_node_info = bool(debug_node_info)
        self._debug_printed: set[str] = set()
        self._mcp_joint = str(mcp_joint)
        self._load_calibration_enabled = bool(load_calibration)
        self._calibration_fallback_path = Path(calibration_file).expanduser()
        self._calibration_side_paths = {
            "left": Path(left_calibration_file).expanduser(),
            "right": Path(right_calibration_file).expanduser(),
        }
        self._calibration_attempted: set[tuple[str, int]] = set()
        self._calibrated_glove_ids: dict[str, int] = {}
        self._started = False

        init_result = int(ManusServer.init(float(init_timeout_s)))
        if init_result != 0:
            raise RuntimeError(f"ManusServer.init failed or timed out: code={init_result}")
        self._started = True

        if self._load_calibration_enabled:
            self._load_calibrations()

    def close(self) -> None:
        if self._started:
            ManusServer.shutdown()
            self._started = False

    def latest_keypoints(
        self,
        last_left_keypoints: np.ndarray | None = None,
        last_right_keypoints: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, np.ndarray, bool]:
        self.poll()
        now = time.monotonic()
        left_keypoints, left_valid = self._side_keypoints("left", last_left_keypoints, now)
        right_keypoints, right_valid = self._side_keypoints("right", last_right_keypoints, now)
        return left_keypoints, left_valid, right_keypoints, right_valid

    def poll(self) -> None:
        state = ManusServer.get_latest_state()
        for side, id_key in (("left", "left_glove_id"), ("right", "right_glove_id")):
            glove_id = _state_id(state, id_key)
            if glove_id is None:
                continue
            if self._load_calibration_enabled:
                self._load_calibration_for_side(side, glove_id)
            position = state.get(f"{glove_id}_position")
            if not position:
                continue
            chain_type = state.get(f"{glove_id}_chain_type")
            joint_type = state.get(f"{glove_id}_joint_type")
            node_id = state.get(f"{glove_id}_node_id")
            parent_id = state.get(f"{glove_id}_parent_id")
            should_print_nodes = self._debug_node_info and side not in self._debug_printed
            try:
                converted = _raw_position_to_keypoints(
                    position,
                    chain_type=chain_type,
                    joint_type=joint_type,
                    mcp_joint=self._mcp_joint,
                    return_mapping=should_print_nodes,
                )
            except Exception as exc:  # noqa: BLE001
                self._log_error(side, exc)
                continue
            if should_print_nodes:
                keypoints, mapping = converted
                _print_node_info(
                    side=side,
                    glove_id=glove_id,
                    position=position,
                    chain_type=chain_type,
                    joint_type=joint_type,
                    node_id=node_id,
                    parent_id=parent_id,
                    mapping=mapping,
                )
                self._debug_printed.add(side)
            else:
                keypoints = converted
            self._latest[side] = {
                "keypoints": keypoints,
                "received_at": time.monotonic(),
                "glove_id": glove_id,
            }

    def _side_keypoints(
        self,
        side: str,
        fallback: np.ndarray | None,
        now: float,
    ) -> tuple[np.ndarray, bool]:
        record = self._latest.get(side)
        if record is not None:
            age_s = now - float(record["received_at"])
            if age_s <= self.stale_s:
                return np.asarray(record["keypoints"], dtype=np.float32).copy(), True
        if fallback is not None:
            return np.asarray(fallback, dtype=np.float32).copy(), False
        return np.zeros((21, 3), dtype=np.float32), False

    def _wait_for_glove_ids(self, timeout_s: float = 3.0) -> dict[str, int]:
        deadline = time.monotonic() + timeout_s
        best_ids: dict[str, int] = {}
        while time.monotonic() < deadline:
            state = ManusServer.get_latest_state()
            ids = {
                side: glove_id
                for side, glove_id in (
                    ("left", _state_id(state, "left_glove_id")),
                    ("right", _state_id(state, "right_glove_id")),
                )
                if glove_id is not None
            }
            if len(ids) > len(best_ids):
                best_ids = ids
            if len(ids) >= 2:
                return ids
            time.sleep(0.05)
        return best_ids

    def _load_calibrations(self) -> None:
        glove_ids = self._wait_for_glove_ids()
        if not glove_ids:
            print("[WARN] No MANUS glove ids available yet; calibration will be retried when gloves appear")
            return

        for side, glove_id in glove_ids.items():
            self._load_calibration_for_side(side, glove_id)

    def _load_calibration_for_side(self, side: str, glove_id: int) -> None:
        key = (side, int(glove_id))
        if key in self._calibration_attempted:
            return
        self._calibration_attempted.add(key)

        path = self._calibration_side_paths.get(side, self._calibration_fallback_path)
        source = "side"
        if not path.exists():
            if self._calibration_fallback_path.exists():
                print(
                    f"[WARN] {side} calibration file not found: {path}; "
                    f"using fallback {self._calibration_fallback_path}"
                )
                path = self._calibration_fallback_path
                source = "fallback"
            else:
                print(f"[WARN] {side} calibration file not found: {path}; skipped")
                return

        result = ManusServer.load_calibration(str(path), int(glove_id))
        ok = bool(result.get("ok", False))
        if ok:
            self._calibrated_glove_ids[side] = int(glove_id)
        print(
            f"[CAL] {side} glove=0x{glove_id:08X} ok={ok} "
            f"source={source} file={path} "
            f"sdk_return={result.get('sdk_return')} set_return={result.get('set_return')} "
            f"bytes={result.get('bytes')}"
        )

    def _log_error(self, side: str, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_error_log_at.get(side, 0.0) >= 2.0:
            print(f"[MANUS] Failed to read {side} raw skeleton: {exc}")
            self._last_error_log_at[side] = now


__all__ = [
    "DEFAULT_MANUS_CALIBRATION_DIR",
    "DEFAULT_MANUS_CALIBRATION_FILE",
    "DEFAULT_MANUS_INIT_TIMEOUT_S",
    "DEFAULT_MANUS_LEFT_CALIBRATION_FILE",
    "DEFAULT_MANUS_MCP_JOINT",
    "DEFAULT_MANUS_RIGHT_CALIBRATION_FILE",
    "DEFAULT_MANUS_STALE_S",
    "MANUS_BINDING_DIR",
    "ManusIntegratedProvider",
    "VALID_MANUS_MCP_JOINTS",
]
