"""Training-only, differentiable G1 link geometry in the pelvis frame."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import nn

MUJOCO_JOINT_NAMES = tuple(
    [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
MUJOCO_TO_ISAACLAB = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)
JOINT_NAMES = tuple(MUJOCO_JOINT_NAMES[i] for i in MUJOCO_TO_ISAACLAB)
_DEFAULT_MUJOCO = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)
DEFAULT_ANGLES = tuple(_DEFAULT_MUJOCO[i] for i in MUJOCO_TO_ISAACLAB)
LINK_GROUPS = {
    "upper": tuple(
        f"{side}_{link}_link" for side in ("left", "right") for link in ("shoulder_roll", "elbow", "wrist_yaw")
    ),
    "lower": tuple(
        f"{side}_{link}_link" for side in ("left", "right") for link in ("hip_roll", "knee", "ankle_roll")
    ),
    "torso": ("torso_link",),
}
LINK_NAMES = tuple(link for links in LINK_GROUPS.values() for link in links)
DEFAULT_URDF = Path(__file__).resolve().parents[5] / "tracker/sonic/gear_sonic/data/robots/g1/g1_29dof.urdf"


def joint_offset(joint_space: str) -> torch.Tensor:
    """`absolute` action representation alone does not specify the URDF zero."""
    if joint_space == "isaaclab_delta":
        return torch.tensor(DEFAULT_ANGLES, dtype=torch.float32)
    if joint_space == "isaaclab_qpos":
        return torch.zeros(29, dtype=torch.float32)
    raise ValueError(f"FK requires explicit fk_joint_space=isaaclab_delta|isaaclab_qpos, got {joint_space!r}")


class _GeometryModule(nn.Module):
    def _apply(self, fn, recurse=True):
        # DeepSpeed calls model.bfloat16(): do not round geometry or norm stats.
        def preserve_precision(tensor):
            converted = fn(tensor)
            if tensor.is_floating_point() and converted.dtype in (torch.float16, torch.bfloat16):
                return tensor.to(device=converted.device)
            return converted

        return super()._apply(preserve_precision, recurse=recurse)


def _vector(element, attribute: str, default: str) -> tuple[float, ...]:
    values = tuple(float(x) for x in (element.get(attribute, default) if element is not None else default).split())
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError(f"Invalid URDF {attribute}: {values}")
    return values


def _rpy_matrix(rpy) -> list[list[float]]:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rotation_angle_squared(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared SO(3) geodesic angle in radians; finite at identity and pi.

    Unlike acos(trace), atan2 has no singular derivative at identity. The norm
    uses PyTorch's zero subgradient there; the exact pi cut locus is inherently
    nondifferentiable, but requires no artificial clipping/dead zone.
    """
    relative = prediction.transpose(-1, -2) @ target
    skew = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
    return torch.atan2(sine, cosine).square()


class G1ForwardKinematics(_GeometryModule):
    """Physical IsaacLab q29 -> pelvis-local origins and optional rotations."""

    def __init__(self, urdf_path=DEFAULT_URDF):
        super().__init__()
        self.urdf_path = str(Path(urdf_path).resolve())
        payload = Path(self.urdf_path).read_bytes()
        self.urdf_sha256 = hashlib.sha256(payload).hexdigest()
        robot = ET.fromstring(payload)
        links = [link.attrib["name"] for link in robot.findall("link")]
        if len(set(links)) != len(links) or not {"pelvis", *LINK_NAMES}.issubset(links):
            raise ValueError("G1 URDF must contain unique links, pelvis and all 13 tracked links")
        by_child = {}
        for joint in robot.findall("joint"):
            child = joint.find("child").attrib["link"]
            if child in by_child:
                raise ValueError(f"Duplicate URDF parent for {child}")
            by_child[child] = joint
        self.parents, self.joint_indices, selected_nodes = [], [], []
        node_indices = {"pelvis": 0}
        dependencies = [set()]
        rotations, translations, axes = [], [], []
        visiting = set()

        def visit(link):
            if link in node_indices:
                return node_indices[link]
            if link in visiting or link not in by_child:
                raise ValueError(f"URDF link {link} is not in an acyclic pelvis subtree")
            visiting.add(link)
            joint = by_child[link]
            parent = visit(joint.find("parent").attrib["link"])
            kind, name = joint.attrib["type"], joint.attrib["name"]
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError(f"Unsupported FK joint {name}: {kind}")
            if kind != "fixed" and name not in JOINT_NAMES:
                raise ValueError(f"Unknown G1 joint: {name}")
            index = -1 if kind == "fixed" else JOINT_NAMES.index(name)
            origin = joint.find("origin")
            rotations.append(_rpy_matrix(_vector(origin, "rpy", "0 0 0")))
            translations.append(_vector(origin, "xyz", "0 0 0"))
            axis = _vector(joint.find("axis"), "xyz", "1 0 0")
            norm = math.sqrt(sum(x * x for x in axis))
            if norm == 0:
                raise ValueError(f"Zero URDF joint axis: {name}")
            axes.append([x / norm for x in axis])
            self.parents.append(parent)
            self.joint_indices.append(index)
            node_indices[link] = len(self.parents)
            dependencies.append(dependencies[parent] | ({index} if index >= 0 else set()))
            visiting.remove(link)
            return node_indices[link]

        for link in LINK_NAMES:
            selected_nodes.append(visit(link))
        indices = [i for i in self.joint_indices if i >= 0]
        if sorted(indices) != list(range(29)):
            raise ValueError("Tracked G1 subtree must contain each of the 29 body joints exactly once")
        self.register_buffer("origin_rotation", torch.tensor(rotations, dtype=torch.float64), persistent=False)
        self.register_buffer(
            "origin_translation", torch.tensor(translations, dtype=torch.float64), persistent=False
        )
        axis = torch.tensor(axes, dtype=torch.float64)
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(-1, 3, 3)
        self.register_buffer("axis_skew", skew, persistent=False)
        self.register_buffer("axis_outer", axis[..., :, None] * axis[..., None, :], persistent=False)
        self.register_buffer("selected_nodes", torch.tensor(selected_nodes), persistent=False)
        mask = [[i in dependencies[node] for i in range(29)] for node in selected_nodes]
        self.register_buffer("chain_mask", torch.tensor(mask, dtype=torch.bool), persistent=False)

    def forward(
        self, q: torch.Tensor, *, return_rotations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if q.shape[-1] != 29 or q.dtype not in (torch.float32, torch.float64):
            raise ValueError("G1 FK requires physical q[...,29] in float32/float64")
        with torch.autocast(device_type=q.device.type, enabled=False):
            origin_r = self.origin_rotation.to(q)
            origin_p = self.origin_translation.to(q)
            skew, outer = self.axis_skew.to(q), self.axis_outer.to(q)
            eye = torch.eye(3, device=q.device, dtype=q.dtype)
            rotations = [eye.expand(*q.shape[:-1], 3, 3)]
            positions = [q.new_zeros(*q.shape[:-1], 3)]
            for i, (parent, joint) in enumerate(zip(self.parents, self.joint_indices)):
                rotation = rotations[parent] @ origin_r[i]
                positions.append(positions[parent] + (rotations[parent] @ origin_p[i, :, None]).squeeze(-1))
                if joint >= 0:
                    angle = q[..., joint, None, None]
                    c, s = angle.cos(), angle.sin()
                    rotation = rotation @ (c * eye + s * skew[i] + (1 - c) * outer[i])
                rotations.append(rotation)
            selected_positions = torch.stack(positions, dim=-2).index_select(-2, self.selected_nodes)
            if return_rotations:
                selected_rotations = torch.stack(rotations, dim=-3).index_select(-3, self.selected_nodes)
                return selected_positions, selected_rotations
            return selected_positions


class G1LinkPositionLoss(_GeometryModule):
    """Position plus optional orientation loss; name retained for compatibility."""

    def __init__(
        self,
        *,
        scale,
        offset,
        model_indices,
        weight=0.05,
        position_scale=0.3,
        orientation_weight=0.0,
        orientation_scale=0.4,
        group_weights=None,
        urdf_path=DEFAULT_URDF,
    ):
        super().__init__()
        self.weight, self.position_scale = float(weight), float(position_scale)
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("FK weight must be finite and non-negative")
        if not math.isfinite(self.position_scale) or self.position_scale <= 0:
            raise ValueError("FK position_scale must be finite and positive")
        self.orientation_weight, self.orientation_scale = float(orientation_weight), float(orientation_scale)
        if not math.isfinite(self.orientation_weight) or self.orientation_weight < 0:
            raise ValueError("FK orientation_weight must be finite and non-negative")
        if not math.isfinite(self.orientation_scale) or self.orientation_scale <= 0:
            raise ValueError("FK orientation_scale must be finite and positive")
        weights = {name: 1.0 for name in LINK_GROUPS}
        if group_weights is not None:
            if set(group_weights) - set(weights):
                raise ValueError("Unknown FK group_weights; expected upper/lower/torso")
            weights.update({k: float(v) for k, v in group_weights.items()})
        if not all(math.isfinite(w) and w >= 0 for w in weights.values()):
            raise ValueError("FK group weights must be finite and non-negative")
        self.group_weights = weights
        scale, offset = torch.as_tensor(scale).detach().float(), torch.as_tensor(offset).detach().float()
        if scale.shape != offset.shape or scale.ndim not in (1, 2) or scale.shape[-1] != 29:
            raise ValueError("FK norm scale/offset must match [29] or [T,29]")
        if not torch.isfinite(scale).all() or not torch.isfinite(offset).all() or not (scale > 0).all():
            raise ValueError("FK normalization requires finite affine parameters and positive scale")
        indices = torch.as_tensor(model_indices, dtype=torch.long)
        if indices.shape != (29,) or len(set(indices.tolist())) != 29 or (indices < 0).any():
            raise ValueError("FK model_indices must select 29 distinct action dimensions")
        self.register_buffer("scale", scale.clone(), persistent=False)
        self.register_buffer("offset", offset.clone(), persistent=False)
        self.register_buffer("model_indices", indices, persistent=False)
        self.fk = G1ForwardKinematics(urdf_path)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        normalizer = getattr(dataset, "normalizer", None)
        normalizer = getattr(normalizer, "action_normalizer", None)
        if normalizer is None or normalizer.mode != "q01/q99":
            raise ValueError("FK currently requires the actual WB q01/q99 action normalizer")
        children = getattr(dataset, "datasets", [dataset])
        mappings = []
        for child in children:
            if not getattr(child, "return_fk_targets", False) or child.action_representation != "absolute":
                raise ValueError("FK requires absolute WB data with return_fk_targets enabled")
            indices = (
                list(range(child.action_dim)) if child.model_dim_indices is None else list(child.model_dim_indices)
            )
            mappings.append(indices[:29])
        if not mappings or any(mapping != mappings[0] for mapping in mappings):
            raise ValueError("FK requires a consistent semantic-to-model mapping across datasets")
        scale, offset = normalizer._match_horizon(torch.empty(dataset.num_frames - 1, dataset.action_dim))
        return cls(scale=scale[..., :29], offset=offset[..., :29], model_indices=mappings[0], **kwargs)

    def forward(
        self,
        *,
        pred_flow,
        noisy_action,
        sigma,
        target,
        joint_zero,
        action_is_pad,
        action_dim_is_pad,
        timestep_weight,
    ):
        if pred_flow.shape != noisy_action.shape or pred_flow.ndim != 3:
            raise ValueError("FK flow/noisy action must match [B,T,D]")
        batch, horizon, _ = pred_flow.shape
        if target.shape != (batch, horizon, 29) or joint_zero.shape != (batch, 29):
            raise ValueError("FK target/offset must be [B,T,29]/[B,29]")
        if action_is_pad is None or action_dim_is_pad is None:
            raise ValueError("FK requires temporal and dimension action masks")
        if action_is_pad.shape != (batch, horizon) or action_dim_is_pad.shape != pred_flow.shape:
            raise ValueError("FK action mask shape mismatch")
        with torch.autocast(device_type=pred_flow.device.type, enabled=False):
            target = target.detach().to(device=pred_flow.device, dtype=torch.float32)
            joint_zero = joint_zero.detach().to(device=pred_flow.device, dtype=torch.float32)[:, None]
            scale, offset = self.scale.float(), self.offset.float()
            if scale.ndim == 2:
                if scale.shape[0] < horizon:
                    raise ValueError("FK norm stats horizon is shorter than action horizon")
                scale, offset = scale[:horizon], offset[:horizon]
            invalid = action_dim_is_pad.index_select(-1, self.model_indices) | action_is_pad[..., None]
            if not torch.isfinite(target.masked_fill(invalid, 0)).all() or not torch.isfinite(joint_zero).all():
                raise ValueError("Non-finite valid FK target or joint zero")
            target = target.masked_fill(invalid, 0)
            clipped = ((target * scale + offset).abs() > 5) & ~invalid
            valid_joint = ~(invalid | clipped)
            valid_link = ~(~valid_joint[..., None, :] & self.fk.chain_mask).any(-1)
            noisy = noisy_action.index_select(-1, self.model_indices).float()
            flow = pred_flow.index_select(-1, self.model_indices).float()
            clean = noisy - sigma.float().reshape(batch, 1, 1) * flow
            q_pred = ((clean - offset) / scale + joint_zero).masked_fill(~valid_joint, 0)
            q_target = (target + joint_zero).masked_fill(~valid_joint, 0)
            use_orientation = self.orientation_weight > 0
            if use_orientation:
                pred_pos, pred_rot = self.fk(q_pred, return_rotations=True)
                with torch.no_grad():
                    target_pos, target_rot = self.fk(q_target, return_rotations=True)
                orientation_error = _rotation_angle_squared(pred_rot, target_rot).masked_fill(~valid_link, 0)
                orientation_per_sample = pred_pos.new_zeros(batch)
            else:
                pred_pos = self.fk(q_pred)
                with torch.no_grad():
                    target_pos = self.fk(q_target)
            error = (pred_pos - target_pos).square().sum(-1).masked_fill(~valid_link, 0)
            per_sample = error.new_zeros(batch)
            metrics = {}
            start = 0
            for name, links in LINK_GROUPS.items():
                stop = start + len(links)
                mask = valid_link[..., start:stop]
                count = mask.sum((1, 2))
                mean_square = error[..., start:stop].sum((1, 2)) / count.clamp_min(1)
                per_sample = per_sample + self.group_weights[name] * mean_square / self.position_scale**2
                # Mean per-sample RMSE, including zero for unsupported samples.
                metrics[f"fk_{name}_rmse_m"] = mean_square.sqrt().mean().detach()
                metrics[f"fk_{name}_valid_ratio"] = mask.float().mean().detach()
                if use_orientation:
                    orientation_mean_square = orientation_error[..., start:stop].sum((1, 2)) / count.clamp_min(1)
                    orientation_per_sample = (
                        orientation_per_sample
                        + self.group_weights[name] * orientation_mean_square / self.orientation_scale**2
                    )
                    metrics[f"fk_{name}_orientation_rmse_deg"] = orientation_mean_square.sqrt().mean().detach() * (
                        180 / math.pi
                    )
                start = stop
            loss = (per_sample * timestep_weight.float().reshape(batch)).mean()
            if use_orientation:
                orientation_loss = (orientation_per_sample * timestep_weight.float().reshape(batch)).mean()
                metrics["loss_fk_position"] = loss.detach()
                metrics["loss_fk_orientation"] = orientation_loss.detach()
                loss = loss + self.orientation_weight * orientation_loss
            metrics["loss_fk"] = loss.detach()
            metrics["loss_fk_weighted"] = self.weight * loss.detach()
            metrics["fk_clipped_joint_ratio"] = (
                (clipped.sum((1, 2)) / (~invalid).sum((1, 2)).clamp_min(1)).mean().detach()
            )
            return self.weight * loss, {name: value.item() for name, value in metrics.items()}
