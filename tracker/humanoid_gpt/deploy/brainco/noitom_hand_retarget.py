"""Noitom-format finger retargeting helpers for BrainCo / Dex3 hands.

Vendored from the original ``deploy.noitom_bvh_tracking.bvh_noitom_streamer``
so that ``play_track_brainco`` stays self-contained in this release. The
functions map a Noitom mocap ``frame`` dict (joint -> (pos, quat_wxyz)) to a
target hand qpos vector.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


_DEX3_CLOSE_POSES = {
    "left": np.array([0.0, 1.0, 1.74, -1.57, -1.74, -1.57, -1.74], dtype=np.float32),
    "right": np.array([0.0, -1.0, -1.74, 1.57, 1.74, 1.57, 1.74], dtype=np.float32),
}

_FINGER_BEND_MAX = {
    "thumb_base": 0.85,
    "thumb_tip": 1.15,
    "base": 1.25,
    "tip": 1.35,
}

_BRAINCO_THUMB_RANGE = (1.5184, 1.0472, 1.0472)
_BRAINCO_THUMB_TIP_RANGE = 0.001
_BRAINCO_FINGER_RANGE = (1.4661, 1.693)
_BRAINCO4_FINGERS = ("Index", "Middle", "Ring", "Pinky")


def _safe_joint_pos(frame: dict, joint: str) -> np.ndarray:
    return np.asarray(frame[joint][0], dtype=np.float64)


def _safe_joint_quat(frame: dict, joint: str) -> np.ndarray:
    return np.asarray(frame[joint][1], dtype=np.float64)


def _quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> R:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def _angle_between(v0: np.ndarray, v1: np.ndarray) -> float:
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-8 or n1 < 1e-8:
        return 0.0
    cos = float(np.dot(v0 / n0, v1 / n1))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return v / norm


def _joint_bend(frame: dict, a: str, b: str, c: str) -> float:
    """Return unsigned bend angle at joint b from three global positions."""
    pa = _safe_joint_pos(frame, a)
    pb = _safe_joint_pos(frame, b)
    pc = _safe_joint_pos(frame, c)
    return _angle_between(pb - pa, pc - pb)


def _scaled_bend(angle: float, max_angle: float, out_max: float) -> float:
    if max_angle <= 0.0:
        return 0.0
    return float(np.clip(angle / max_angle, 0.0, 1.0) * out_max)


def _hand_scale(frame: dict, side: str) -> float:
    p_index = _safe_joint_pos(frame, f"{side}InHandIndex")
    p_pinky = _safe_joint_pos(frame, f"{side}InHandPinky")
    p_hand = _safe_joint_pos(frame, f"{side}Hand")
    p_middle = _safe_joint_pos(frame, f"{side}InHandMiddle")
    palm_width = np.linalg.norm(p_index - p_pinky)
    palm_len = np.linalg.norm(p_middle - p_hand)
    scale = max(palm_width, palm_len, 1e-3)
    return float(scale)


def _closure_from_tip_distance(frame: dict, side: str, finger: str) -> float:
    """Estimate closure from fingertip distance to the thumb tip.

    The thresholds are normalized by palm size so the same BVH retarget works
    across actors with different hand scales.
    """
    scale = _hand_scale(frame, side)
    thumb_tip = _safe_joint_pos(frame, f"{side}HandThumb3")
    finger_tip = _safe_joint_pos(frame, f"{side}Hand{finger}3")
    d_norm = float(np.linalg.norm(finger_tip - thumb_tip) / scale)
    return float(np.clip((1.75 - d_norm) / (1.75 - 0.55), 0.0, 1.0))


def _finger_chain_close(frame: dict, side: str, finger: str) -> float:
    """Curl estimate from fingertip-vector shortening.

    This mirrors the BrainCo vector-retargeting idea more closely than a pure
    joint-angle heuristic: the hand is represented by vectors from a finger
    root to the fingertip, and curl is inferred from endpoint contraction.
    """
    root = _safe_joint_pos(frame, f"{side}InHand{finger}")
    p1 = _safe_joint_pos(frame, f"{side}Hand{finger}1")
    p2 = _safe_joint_pos(frame, f"{side}Hand{finger}2")
    tip = _safe_joint_pos(frame, f"{side}Hand{finger}3")
    chain_len = (
        np.linalg.norm(p1 - root)
        + np.linalg.norm(p2 - p1)
        + np.linalg.norm(tip - p2)
    )
    if chain_len < 1e-8:
        return 0.0
    straightness = np.linalg.norm(tip - root) / chain_len
    return float(np.clip((1.0 - straightness) / 0.42, 0.0, 1.0))


def _thumb_chain_close(frame: dict, side: str) -> float:
    p1 = _safe_joint_pos(frame, f"{side}HandThumb1")
    p2 = _safe_joint_pos(frame, f"{side}HandThumb2")
    tip = _safe_joint_pos(frame, f"{side}HandThumb3")
    chain_len = np.linalg.norm(p2 - p1) + np.linalg.norm(tip - p2)
    if chain_len < 1e-8:
        return 0.0
    straightness = np.linalg.norm(tip - p1) / chain_len
    return float(np.clip((1.0 - straightness) / 0.35, 0.0, 1.0))


def _brainco4_tip_offsets(side: str) -> dict[str, np.ndarray]:
    sign = 1.0 if side == "Left" else -1.0
    return {
        "Thumb": np.array([0.0, 0.023 * sign, 0.0], dtype=np.float64),
        "Index": np.array([0.0, 0.020 * sign, 0.0], dtype=np.float64),
        "Middle": np.array([0.0, 0.025 * sign, 0.0], dtype=np.float64),
        "Ring": np.array([0.0, 0.021 * sign, 0.0], dtype=np.float64),
        "Pinky": np.array([0.0, 0.017 * sign, 0.0], dtype=np.float64),
    }


def _brainco4_remap_rotation(side: str) -> R:
    if side == "Left":
        mat = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
    else:
        mat = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float64)
    return R.from_matrix(mat)


def _localize_noitom_hand_brainco4(frame: dict, side: str) -> dict[str, np.ndarray]:
    """Local wrist-frame hand points, following GMR_withHand's Noitom path."""
    root_name = f"{side}Hand"
    root_pos = _safe_joint_pos(frame, root_name)
    root_rot_inv = _quat_wxyz_to_rotation(_safe_joint_quat(frame, root_name)).inv()
    remap = _brainco4_remap_rotation(side)
    tip_offsets = _brainco4_tip_offsets(side)
    local: dict[str, np.ndarray] = {root_name: np.zeros(3, dtype=np.float64)}

    names = [root_name]
    names += [f"{side}HandThumb{i}" for i in (1, 2, 3)]
    for finger in _BRAINCO4_FINGERS:
        names += [f"{side}InHand{finger}", f"{side}Hand{finger}1", f"{side}Hand{finger}2", f"{side}Hand{finger}3"]

    for name in names:
        pos = _safe_joint_pos(frame, name)
        rot = _quat_wxyz_to_rotation(_safe_joint_quat(frame, name))
        local_rot = remap * root_rot_inv * rot
        local[name] = (remap * root_rot_inv).apply(pos - root_pos)
        if name.endswith("3"):
            for finger, offset in tip_offsets.items():
                if finger in name:
                    local[f"{side}Hand{finger}Tip"] = local[name] + local_rot.apply(offset)
                    break

    return local


def _local_hand_scale(local: dict[str, np.ndarray], side: str) -> float:
    p_index = local[f"{side}InHandIndex"]
    p_pinky = local[f"{side}InHandPinky"]
    p_middle = local[f"{side}InHandMiddle"]
    palm_width = np.linalg.norm(p_index - p_pinky)
    palm_len = np.linalg.norm(p_middle - local[f"{side}Hand"])
    return float(max(palm_width, palm_len, 1e-3))


def _local_chain_close(points: tuple[np.ndarray, ...], max_contraction: float) -> float:
    chain_len = sum(np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1))
    if chain_len < 1e-8:
        return 0.0
    straightness = np.linalg.norm(points[-1] - points[0]) / chain_len
    return float(np.clip((1.0 - straightness) / max_contraction, 0.0, 1.0))


def _finger_pair_qpos(frame: dict, side: str, finger: str, out0: float, out1: float) -> tuple[float, float]:
    base = _joint_bend(
        frame,
        f"{side}InHand{finger}",
        f"{side}Hand{finger}1",
        f"{side}Hand{finger}2",
    )
    tip = _joint_bend(
        frame,
        f"{side}Hand{finger}1",
        f"{side}Hand{finger}2",
        f"{side}Hand{finger}3",
    )
    return (
        _scaled_bend(base, _FINGER_BEND_MAX["base"], out0),
        _scaled_bend(tip, _FINGER_BEND_MAX["tip"], out1),
    )


def _finger_pair_qpos2(frame: dict, side: str, finger: str, out0: float, out1: float) -> tuple[float, float]:
    base, tip = _finger_pair_qpos(frame, side, finger, out0, out1)
    close = _closure_from_tip_distance(frame, side, finger)
    base = max(base, out0 * 0.22 * close)
    tip = max(tip, out1 * 0.18 * close)
    return base, tip


def _finger_pair_qpos3(frame: dict, side: str, finger: str, out0: float, out1: float) -> tuple[float, float]:
    endpoint_close = _finger_chain_close(frame, side, finger)
    base_angle, tip_angle = _finger_pair_qpos(frame, side, finger, out0, out1)
    base_angle_close = base_angle / out0 if abs(out0) > 1e-8 else 0.0
    tip_angle_close = tip_angle / out1 if abs(out1) > 1e-8 else 0.0
    base_close = 0.70 * endpoint_close + 0.30 * base_angle_close
    tip_close = 0.55 * endpoint_close + 0.45 * tip_angle_close
    return (
        float(np.clip(base_close, 0.0, 1.0) * out0),
        float(np.clip(tip_close, 0.0, 1.0) * out1),
    )


def _finger_pair_qpos4(
    frame: dict,
    local: dict[str, np.ndarray],
    side: str,
    finger: str,
    out0: float,
    out1: float,
) -> tuple[float, float]:
    root = local[f"{side}InHand{finger}"]
    p1 = local[f"{side}Hand{finger}1"]
    p2 = local[f"{side}Hand{finger}2"]
    tip = local[f"{side}Hand{finger}Tip"]
    endpoint_close = _local_chain_close((root, p1, p2, tip), max_contraction=0.42)

    base_angle, tip_angle = _finger_pair_qpos(frame, side, finger, out0, out1)
    base_angle_close = base_angle / out0 if abs(out0) > 1e-8 else 0.0
    tip_angle_close = tip_angle / out1 if abs(out1) > 1e-8 else 0.0

    scale = _local_hand_scale(local, side)
    thumb_tip = local[f"{side}HandThumbTip"]
    pinch_dist = np.linalg.norm(tip - thumb_tip) / scale
    pinch_close = float(np.clip((1.55 - pinch_dist) / (1.55 - 0.42), 0.0, 1.0))
    pinch_weight = 0.38 if finger == "Index" else 0.18

    base_close = 0.68 * endpoint_close + 0.24 * base_angle_close + pinch_weight * pinch_close
    tip_close = 0.52 * endpoint_close + 0.36 * tip_angle_close + 0.70 * pinch_weight * pinch_close

    # Keep fingers from over-closing through the thumb/palm during contact.
    contact_damp = float(np.clip((0.38 - pinch_dist) / 0.20, 0.0, 1.0))
    base_cap = 0.92 - 0.18 * contact_damp
    tip_cap = 0.88 - 0.22 * contact_damp

    return (
        float(np.clip(base_close, 0.0, base_cap) * out0),
        float(np.clip(tip_close, 0.0, tip_cap) * out1),
    )


def _thumb_triplet_qpos(frame: dict, side: str, out0: float, out1: float, out2: float) -> tuple[float, float, float]:
    base = _joint_bend(frame, f"{side}Hand", f"{side}HandThumb1", f"{side}HandThumb2")
    tip = _joint_bend(frame, f"{side}HandThumb1", f"{side}HandThumb2", f"{side}HandThumb3")
    # Noitom's three thumb bones give a stable flexion signal, while thumb
    # abduction is model-specific; keep the abduction channel neutral.
    return (
        float(out0) * 0.0,
        _scaled_bend(base, _FINGER_BEND_MAX["thumb_base"], out1),
        _scaled_bend(tip, _FINGER_BEND_MAX["thumb_tip"], out2),
    )


def _thumb_quad_qpos2(
    frame: dict,
    side: str,
    out0: float,
    out1: float,
    out2: float,
    out3: float,
) -> tuple[float, float, float, float]:
    """Thumb retarget with opposition plus flexion for all four XML joints.

    BrainCo's first thumb joint is an opposition/abduction-like DOF. Noitom
    gives reliable global finger positions but its thumb coordinate axes are
    not the same as the robot hand, so this estimates thumb0 geometrically:
    thumb direction inside the palm plane + pinch distance to index/middle.
    The fourth XML thumb joint has a tiny symmetric range, so it is coupled
    to distal thumb bend and pinch as a signed fingertip curl.
    """
    p_hand = _safe_joint_pos(frame, f"{side}Hand")
    p_index = _safe_joint_pos(frame, f"{side}InHandIndex")
    p_middle = _safe_joint_pos(frame, f"{side}InHandMiddle")
    p_pinky = _safe_joint_pos(frame, f"{side}InHandPinky")
    p_thumb1 = _safe_joint_pos(frame, f"{side}HandThumb1")
    p_thumb2 = _safe_joint_pos(frame, f"{side}HandThumb2")
    p_thumb3 = _safe_joint_pos(frame, f"{side}HandThumb3")
    p_index_tip = _safe_joint_pos(frame, f"{side}HandIndex3")
    p_middle_tip = _safe_joint_pos(frame, f"{side}HandMiddle3")

    forward = _safe_normalize(p_middle - p_hand)
    across = _safe_normalize(p_index - p_pinky)
    palm_normal = _safe_normalize(np.cross(across, forward))
    if np.linalg.norm(palm_normal) < 1e-8:
        palm_normal = _safe_normalize(np.cross(p_index - p_hand, p_pinky - p_hand))

    thumb_dir = _safe_normalize(p_thumb2 - p_thumb1)
    thumb_in_palm = thumb_dir - palm_normal * np.dot(thumb_dir, palm_normal)
    thumb_in_palm = _safe_normalize(thumb_in_palm)

    toward_fingers = _safe_normalize((p_index_tip + p_middle_tip) * 0.5 - p_thumb1)
    toward_fingers = toward_fingers - palm_normal * np.dot(toward_fingers, palm_normal)
    toward_fingers = _safe_normalize(toward_fingers)

    side_open = across if side == "Left" else -across
    open_angle = _angle_between(thumb_in_palm, side_open)
    toward_angle = _angle_between(thumb_in_palm, toward_fingers)
    directional_close = np.clip(open_angle / 1.35, 0.0, 1.0)
    directional_close *= np.clip((1.35 - toward_angle) / 1.35, 0.0, 1.0)

    scale = _hand_scale(frame, side)
    pinch_index = np.linalg.norm(p_thumb3 - p_index_tip) / scale
    pinch_middle = np.linalg.norm(p_thumb3 - p_middle_tip) / scale
    pinch_close = np.clip((1.65 - min(pinch_index, pinch_middle)) / (1.65 - 0.45), 0.0, 1.0)

    base_bend = _joint_bend(frame, f"{side}Hand", f"{side}HandThumb1", f"{side}HandThumb2")
    tip_bend = _joint_bend(frame, f"{side}HandThumb1", f"{side}HandThumb2", f"{side}HandThumb3")
    base_close = np.clip(base_bend / _FINGER_BEND_MAX["thumb_base"], 0.0, 1.0)
    tip_close = np.clip(tip_bend / _FINGER_BEND_MAX["thumb_tip"], 0.0, 1.0)

    thumb0_close = max(float(directional_close), float(pinch_close) * 0.85)
    thumb1_close = max(float(base_close), float(pinch_close) * 0.45)
    thumb2_close = max(float(tip_close), float(pinch_close) * 0.35)
    thumb3_close = max(float(tip_close) * 0.65, float(pinch_close) * 0.55)
    thumb3_sign = -1.0 if side == "Left" else 1.0

    return (
        float(np.clip(thumb0_close, 0.0, 1.0) * out0),
        float(np.clip(thumb1_close, 0.0, 1.0) * out1),
        float(np.clip(thumb2_close, 0.0, 1.0) * out2),
        float(thumb3_sign * np.clip(thumb3_close, 0.0, 1.0) * out3),
    )


def _thumb_quad_qpos3(
    frame: dict,
    side: str,
    out0: float,
    out1: float,
    out2: float,
    out3: float,
) -> tuple[float, float, float, float]:
    """BrainCo vector-style thumb mapping.

    The reference pipeline builds five wrist-to-fingertip vectors and lets
    dex-retargeting solve the BrainCo hand. This online approximation uses
    the same inputs: thumb/index/middle fingertip vectors relative to the
    wrist, with endpoint contraction driving flexion.
    """
    p_hand = _safe_joint_pos(frame, f"{side}Hand")
    p_index = _safe_joint_pos(frame, f"{side}InHandIndex")
    p_middle = _safe_joint_pos(frame, f"{side}InHandMiddle")
    p_pinky = _safe_joint_pos(frame, f"{side}InHandPinky")
    p_thumb1 = _safe_joint_pos(frame, f"{side}HandThumb1")
    p_thumb2 = _safe_joint_pos(frame, f"{side}HandThumb2")
    p_thumb3 = _safe_joint_pos(frame, f"{side}HandThumb3")
    p_index_tip = _safe_joint_pos(frame, f"{side}HandIndex3")
    p_middle_tip = _safe_joint_pos(frame, f"{side}HandMiddle3")

    forward = _safe_normalize(p_middle - p_hand)
    across = _safe_normalize(p_index - p_pinky)
    palm_normal = _safe_normalize(np.cross(across, forward))
    if np.linalg.norm(palm_normal) < 1e-8:
        palm_normal = _safe_normalize(np.cross(p_index - p_hand, p_pinky - p_hand))

    wrist_to_thumb = p_thumb3 - p_hand
    thumb_in_palm = wrist_to_thumb - palm_normal * np.dot(wrist_to_thumb, palm_normal)
    thumb_in_palm = _safe_normalize(thumb_in_palm)
    toward_fingers = _safe_normalize((p_index_tip + p_middle_tip) * 0.5 - p_hand)
    toward_fingers = toward_fingers - palm_normal * np.dot(toward_fingers, palm_normal)
    toward_fingers = _safe_normalize(toward_fingers)

    side_open = across if side == "Left" else -across
    open_angle = _angle_between(thumb_in_palm, side_open)
    toward_angle = _angle_between(thumb_in_palm, toward_fingers)
    opposition_close = np.clip(open_angle / 1.45, 0.0, 1.0)
    opposition_close *= np.clip((1.45 - toward_angle) / 1.45, 0.0, 1.0)

    scale = _hand_scale(frame, side)
    pinch_index = np.linalg.norm(p_thumb3 - p_index_tip) / scale
    pinch_middle = np.linalg.norm(p_thumb3 - p_middle_tip) / scale
    pinch_close = np.clip((1.70 - min(pinch_index, pinch_middle)) / (1.70 - 0.50), 0.0, 1.0)

    endpoint_close = _thumb_chain_close(frame, side)
    base_bend = _joint_bend(frame, f"{side}Hand", f"{side}HandThumb1", f"{side}HandThumb2")
    tip_bend = _joint_bend(frame, f"{side}HandThumb1", f"{side}HandThumb2", f"{side}HandThumb3")
    base_angle_close = np.clip(base_bend / _FINGER_BEND_MAX["thumb_base"], 0.0, 1.0)
    tip_angle_close = np.clip(tip_bend / _FINGER_BEND_MAX["thumb_tip"], 0.0, 1.0)

    thumb0_close = max(float(opposition_close), float(pinch_close) * 0.80)
    thumb1_close = 0.55 * float(endpoint_close) + 0.30 * float(base_angle_close) + 0.15 * float(pinch_close)
    thumb2_close = 0.50 * float(endpoint_close) + 0.35 * float(tip_angle_close) + 0.15 * float(pinch_close)
    thumb3_close = 0.60 * float(tip_angle_close) + 0.40 * float(pinch_close)
    thumb3_sign = -1.0 if side == "Left" else 1.0

    return (
        float(np.clip(thumb0_close, 0.0, 1.0) * out0),
        float(np.clip(thumb1_close, 0.0, 1.0) * out1),
        float(np.clip(thumb2_close, 0.0, 1.0) * out2),
        float(thumb3_sign * np.clip(thumb3_close, 0.0, 1.0) * out3),
    )


def _thumb_quad_qpos4(
    frame: dict,
    local: dict[str, np.ndarray],
    side: str,
    out0: float,
    out1: float,
    out2: float,
    out3: float,
) -> tuple[float, float, float, float]:
    p_hand = local[f"{side}Hand"]
    p_index = local[f"{side}InHandIndex"]
    p_middle = local[f"{side}InHandMiddle"]
    p_pinky = local[f"{side}InHandPinky"]
    p_thumb1 = local[f"{side}HandThumb1"]
    p_thumb2 = local[f"{side}HandThumb2"]
    p_thumb_tip = local[f"{side}HandThumbTip"]
    p_index_tip = local[f"{side}HandIndexTip"]
    p_middle_tip = local[f"{side}HandMiddleTip"]

    forward = _safe_normalize(p_middle - p_hand)
    across = _safe_normalize(p_index - p_pinky)
    palm_normal = _safe_normalize(np.cross(across, forward))
    if np.linalg.norm(palm_normal) < 1e-8:
        palm_normal = _safe_normalize(np.cross(p_index - p_hand, p_pinky - p_hand))

    thumb_vec = p_thumb_tip - p_hand
    thumb_in_palm = thumb_vec - palm_normal * np.dot(thumb_vec, palm_normal)
    thumb_in_palm = _safe_normalize(thumb_in_palm)
    target_vec = (0.72 * p_index_tip + 0.28 * p_middle_tip) - p_hand
    target_in_palm = target_vec - palm_normal * np.dot(target_vec, palm_normal)
    target_in_palm = _safe_normalize(target_in_palm)

    side_open = across if side == "Left" else -across
    open_angle = _angle_between(thumb_in_palm, side_open)
    target_angle = _angle_between(thumb_in_palm, target_in_palm)
    opposition_close = np.clip(open_angle / 1.42, 0.0, 1.0)
    opposition_close *= np.clip((1.42 - target_angle) / 1.42, 0.0, 1.0)

    scale = _local_hand_scale(local, side)
    pinch_index = np.linalg.norm(p_thumb_tip - p_index_tip) / scale
    pinch_middle = np.linalg.norm(p_thumb_tip - p_middle_tip) / scale
    pinch_close = float(np.clip((1.48 - min(pinch_index, pinch_middle)) / (1.48 - 0.40), 0.0, 1.0))

    endpoint_close = _local_chain_close((p_thumb1, p_thumb2, p_thumb_tip), max_contraction=0.34)
    base_bend = _joint_bend(frame, f"{side}Hand", f"{side}HandThumb1", f"{side}HandThumb2")
    tip_bend = _joint_bend(frame, f"{side}HandThumb1", f"{side}HandThumb2", f"{side}HandThumb3")
    base_angle_close = np.clip(base_bend / _FINGER_BEND_MAX["thumb_base"], 0.0, 1.0)
    tip_angle_close = np.clip(tip_bend / _FINGER_BEND_MAX["thumb_tip"], 0.0, 1.0)

    contact_damp = float(np.clip((0.36 - pinch_index) / 0.18, 0.0, 1.0))
    thumb0_close = max(float(opposition_close), 0.78 * pinch_close)
    thumb1_close = 0.50 * endpoint_close + 0.30 * float(base_angle_close) + 0.20 * pinch_close
    thumb2_close = 0.46 * endpoint_close + 0.36 * float(tip_angle_close) + 0.18 * pinch_close
    thumb3_close = 0.60 * float(tip_angle_close) + 0.40 * pinch_close

    thumb0_close = min(thumb0_close, 0.92 - 0.12 * contact_damp)
    thumb1_close = min(thumb1_close, 0.86 - 0.12 * contact_damp)
    thumb2_close = min(thumb2_close, 0.84 - 0.10 * contact_damp)
    thumb3_sign = -1.0 if side == "Left" else 1.0

    return (
        float(np.clip(thumb0_close, 0.0, 1.0) * out0),
        float(np.clip(thumb1_close, 0.0, 1.0) * out1),
        float(np.clip(thumb2_close, 0.0, 1.0) * out2),
        float(thumb3_sign * np.clip(thumb3_close, 0.0, 1.0) * out3),
    )


def _retarget_noitom_fingers_to_dex3(frame: dict) -> np.ndarray:
    """Map Noitom finger global positions to Dex3 left/right 7-DoF qpos.

    Order matches deploy.hand_control: thumb0/1/2, middle0/1, index0/1.
    """
    out = np.zeros(14, dtype=np.float32)
    for hand_i, side in enumerate(("Left", "Right")):
        close = _DEX3_CLOSE_POSES[side.lower()]
        thumb = _thumb_triplet_qpos(frame, side, close[0], close[1], close[2])
        middle = _finger_pair_qpos(frame, side, "Middle", close[3], close[4])
        index = _finger_pair_qpos(frame, side, "Index", close[5], close[6])
        out[hand_i * 7:(hand_i + 1) * 7] = np.asarray([*thumb, *middle, *index], dtype=np.float32)
    return out


def _retarget_noitom_fingers_to_brainco(frame: dict) -> np.ndarray:
    """Map Noitom fingers to BrainCo hand qpos.

    Output order is compatible with retarget_human_bvh_to_unitree-g1:
    left 12 then right 12, each hand as
    thumb0/1/2, thumb placeholder, index0/1, middle0/1, ring0/1, pinky0/1.
    """
    out = np.zeros(24, dtype=np.float32)
    for hand_i, side in enumerate(("Left", "Right")):
        thumb = _thumb_triplet_qpos(frame, side, *_BRAINCO_THUMB_RANGE)
        index = _finger_pair_qpos(frame, side, "Index", *_BRAINCO_FINGER_RANGE)
        middle = _finger_pair_qpos(frame, side, "Middle", *_BRAINCO_FINGER_RANGE)
        ring = _finger_pair_qpos(frame, side, "Ring", *_BRAINCO_FINGER_RANGE)
        pinky = _finger_pair_qpos(frame, side, "Pinky", *_BRAINCO_FINGER_RANGE)
        out[hand_i * 12:(hand_i + 1) * 12] = np.asarray(
            [*thumb, 0.0, *index, *middle, *ring, *pinky],
            dtype=np.float32,
        )
    return out


def _retarget_noitom_fingers_to_brainco2(frame: dict) -> np.ndarray:
    """Map Noitom fingers to BrainCo hand qpos with geometric thumb opposition.

    Output order is the same 24-D BrainCo order as
    ``_retarget_noitom_fingers_to_brainco``. Compared with v1, all four thumb
    channels are filled: thumb0 from palm-plane opposition, thumb1/2 from
    thumb flexion, and thumb3 from a small signed distal fingertip curl.
    """
    out = np.zeros(24, dtype=np.float32)
    for hand_i, side in enumerate(("Left", "Right")):
        thumb = _thumb_quad_qpos2(
            frame,
            side,
            *_BRAINCO_THUMB_RANGE,
            _BRAINCO_THUMB_TIP_RANGE,
        )
        index = _finger_pair_qpos2(frame, side, "Index", *_BRAINCO_FINGER_RANGE)
        middle = _finger_pair_qpos2(frame, side, "Middle", *_BRAINCO_FINGER_RANGE)
        ring = _finger_pair_qpos2(frame, side, "Ring", *_BRAINCO_FINGER_RANGE)
        pinky = _finger_pair_qpos2(frame, side, "Pinky", *_BRAINCO_FINGER_RANGE)
        out[hand_i * 12:(hand_i + 1) * 12] = np.asarray(
            [*thumb, *index, *middle, *ring, *pinky],
            dtype=np.float32,
        )
    return out


def _retarget_noitom_fingers_to_brainco3(frame: dict) -> np.ndarray:
    """BrainCo hand qpos using a vector-retargeting style approximation.

    The standalone retarget_human_bvh_to_unitree-g1 pipeline feeds five
    wrist-to-fingertip vectors into dex-retargeting, optimizes BrainCo joints,
    then reorders internal qpos as
    ``[thumb, index, middle, ring, pinky]``. This version keeps that output
    order but uses a lightweight online geometric solve, so it can run inside
    the Noitom worker without the external optimizer dependency.
    """
    out = np.zeros(24, dtype=np.float32)
    for hand_i, side in enumerate(("Left", "Right")):
        thumb = _thumb_quad_qpos3(
            frame,
            side,
            *_BRAINCO_THUMB_RANGE,
            _BRAINCO_THUMB_TIP_RANGE,
        )
        index = _finger_pair_qpos3(frame, side, "Index", *_BRAINCO_FINGER_RANGE)
        middle = _finger_pair_qpos3(frame, side, "Middle", *_BRAINCO_FINGER_RANGE)
        ring = _finger_pair_qpos3(frame, side, "Ring", *_BRAINCO_FINGER_RANGE)
        pinky = _finger_pair_qpos3(frame, side, "Pinky", *_BRAINCO_FINGER_RANGE)
        out[hand_i * 12:(hand_i + 1) * 12] = np.asarray(
            [*thumb, *index, *middle, *ring, *pinky],
            dtype=np.float32,
        )
    return out


def _retarget_noitom_hand_qpos(frame: dict, target: str) -> np.ndarray:
    if target == "dex3":
        return _retarget_noitom_fingers_to_dex3(frame)
    if target == "brainco":
        return _retarget_noitom_fingers_to_brainco(frame)
    if target == "brainco2":
        return _retarget_noitom_fingers_to_brainco2(frame)
    if target == "brainco3":
        return _retarget_noitom_fingers_to_brainco3(frame)
    raise ValueError(f"Unknown hand target: {target!r}. Expected 'brainco', 'brainco2', 'brainco3', or 'dex3'.")


def _hand_qpos_dof(target: str) -> int:
    if target == "dex3":
        return 14
    if target in ("brainco", "brainco2", "brainco3"):
        return 24
    raise ValueError(f"Unknown hand target: {target!r}. Expected 'brainco', 'brainco2', 'brainco3', or 'dex3'.")
