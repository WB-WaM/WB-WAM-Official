# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable, Mapping
from typing import Any, Optional

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as transforms_F

RELATIVE_ACTION_REFERENCES = ("observation", "first_action")


def normalize_relative_action_reference(value: str) -> str:
    """Return a validated fixed reference for relative action targets."""
    reference = str(value).strip().lower()
    if reference not in RELATIVE_ACTION_REFERENCES:
        raise ValueError(
            f"Unsupported relative_action_reference: {value!r}. "
            f"Expected one of: {list(RELATIVE_ACTION_REFERENCES)}."
        )
    return reference


def list_column_to_numpy(column, dtype, width: Optional[int] = None) -> np.ndarray:
    """Convert a fixed-width Arrow-style list column to a NumPy array."""
    arr = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    length = len(arr)
    expected = None if width is None else int(width)
    try:
        values = arr.values.to_numpy(zero_copy_only=False)
        if expected is None and hasattr(arr, "offsets"):
            offsets = arr.offsets.to_numpy(zero_copy_only=False)
            diffs = np.diff(offsets)
            if diffs.size:
                first = int(diffs[0])
                if np.all(diffs == first):
                    expected = first
        if expected is not None and expected >= 0 and values.size == length * expected:
            return values.reshape(length, expected).astype(dtype, copy=False)
    except Exception:
        pass
    return np.asarray(column.to_pylist(), dtype=dtype)


def concat_feature_slices(
    array: np.ndarray,
    slices: list[tuple[int, int, str]],
) -> np.ndarray:
    """Select and concatenate configured feature ranges."""
    if len(slices) == 1 and slices[0][0] == 0 and slices[0][1] == array.shape[1]:
        return array
    return np.concatenate([array[:, start:end] for start, end, _label in slices], axis=1)


def right_pad_tensor_and_mask(
    values: torch.Tensor,
    dim_is_pad: torch.Tensor,
    target_dim: int,
    *,
    feature_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad values with zeros and mark the added dimensions invalid."""
    if values.shape != dim_is_pad.shape:
        raise ValueError(
            f"{feature_name} values/mask shape mismatch: {tuple(values.shape)} vs {tuple(dim_is_pad.shape)}"
        )
    if dim_is_pad.dtype != torch.bool:
        raise TypeError(f"{feature_name} dim mask must be bool, got {dim_is_pad.dtype}")

    source_dim = int(values.shape[-1])
    target_dim = int(target_dim)
    if target_dim < source_dim:
        raise ValueError(f"{feature_name} target_dim={target_dim} is smaller than selected dim={source_dim}.")
    if target_dim == source_dim:
        return values, dim_is_pad

    pad_shape = (*values.shape[:-1], target_dim - source_dim)
    values_pad = values.new_zeros(pad_shape)
    mask_pad = torch.ones(pad_shape, dtype=torch.bool, device=dim_is_pad.device)
    return torch.cat((values, values_pad), dim=-1), torch.cat((dim_is_pad, mask_pad), dim=-1)


def scatter_tensor_and_mask(
    values: torch.Tensor,
    dim_is_pad: torch.Tensor,
    target_dim: int,
    target_indices: Iterable[int],
    *,
    feature_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter source dimensions into fixed model slots without changing values."""
    if values.shape != dim_is_pad.shape:
        raise ValueError(
            f"{feature_name} values/mask shape mismatch: {tuple(values.shape)} vs {tuple(dim_is_pad.shape)}"
        )
    if dim_is_pad.dtype != torch.bool:
        raise TypeError(f"{feature_name} dim mask must be bool, got {dim_is_pad.dtype}")

    indices = tuple(int(index) for index in target_indices)
    source_dim = int(values.shape[-1])
    target_dim = int(target_dim)
    if len(indices) != source_dim:
        raise ValueError(f"{feature_name} target_indices must have {source_dim} entries, got {len(indices)}.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{feature_name} target_indices must be unique.")
    if any(index < 0 or index >= target_dim for index in indices):
        raise ValueError(f"{feature_name} target_indices must be in [0, {target_dim}), got {indices}.")

    values_target = values.new_zeros((*values.shape[:-1], target_dim))
    mask_target = torch.ones(
        (*dim_is_pad.shape[:-1], target_dim),
        dtype=torch.bool,
        device=dim_is_pad.device,
    )
    values_target.index_copy_(
        -1,
        torch.tensor(indices, dtype=torch.long, device=values.device),
        values,
    )
    mask_target.index_copy_(
        -1,
        torch.tensor(indices, dtype=torch.long, device=dim_is_pad.device),
        dim_is_pad,
    )
    return values_target, mask_target


def apply_relative_joint_action(
    action: np.ndarray,
    action_dim_is_pad: np.ndarray,
    reference: np.ndarray,
    reference_dim_is_pad: np.ndarray,
    relative_joint_ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert configured action ranges to fixed-base relative targets."""
    if action.shape != action_dim_is_pad.shape:
        raise ValueError(f"Action/mask shape mismatch: {action.shape} vs {action_dim_is_pad.shape}")
    if reference.shape != reference_dim_is_pad.shape:
        raise ValueError(f"Reference/mask shape mismatch: {reference.shape} vs {reference_dim_is_pad.shape}")
    if action.shape[-1] != reference.shape[-1]:
        raise ValueError(
            f"Relative joint action requires matching action/reference dims, got "
            f"{action.shape[-1]} and {reference.shape[-1]}."
        )

    relative = action.copy()
    relative_mask = action_dim_is_pad.copy()
    for start, end in relative_joint_ranges:
        relative[..., start:end] -= reference[..., start:end]
        relative_mask[..., start:end] |= reference_dim_is_pad[..., start:end]
    relative[relative_mask] = 0.0
    return relative, relative_mask


def restore_absolute_joint_action(
    relative_action: np.ndarray | torch.Tensor,
    current_state: np.ndarray | torch.Tensor,
    relative_joint_ranges: Iterable[tuple[int, int]],
) -> np.ndarray | torch.Tensor:
    """Restore fixed-base relative joint targets to absolute joint positions."""
    absolute = relative_action.clone() if isinstance(relative_action, torch.Tensor) else relative_action.copy()
    base = current_state
    if base.ndim == absolute.ndim - 1:
        base = base.unsqueeze(-2) if isinstance(base, torch.Tensor) else np.expand_dims(base, axis=-2)
    if base.shape[-1] != absolute.shape[-1]:
        raise ValueError(
            f"Relative joint restore requires matching action/state dims, got "
            f"{absolute.shape[-1]} and {base.shape[-1]}."
        )
    for start, end in relative_joint_ranges:
        absolute[..., start:end] += base[..., start:end]
    return absolute


def _config_section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = config.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise TypeError(f"image_augmentation.{key} must be a mapping, got {type(section).__name__}.")
    return section


def _validate_probability(value: Any, name: str) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return probability


def _validate_range(
    value: Any,
    name: str,
    *,
    minimum: Optional[float] = None,
    strictly_positive: bool = False,
) -> tuple[float, float]:
    try:
        low, high = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}.") from exc
    low, high = float(low), float(high)
    if low > high:
        raise ValueError(f"{name} must be ordered low-to-high, got {value!r}.")
    if minimum is not None and low < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    if strictly_positive and low <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return low, high


class VideoAugmentor:
    """Apply temporally coherent augmentation to a `[T,C,H,W]` clip."""

    _SECTIONS = {"enabled", "random_resized_crop", "color_jitter", "low_light_noise"}

    def __init__(
        self,
        config: Optional[Mapping[str, Any]],
        image_size: Iterable[int],
    ) -> None:
        config = {} if config is None else config
        if not isinstance(config, Mapping):
            raise TypeError(f"image_augmentation must be a mapping, got {type(config).__name__}.")
        unknown = sorted(set(config) - self._SECTIONS)
        if unknown:
            raise ValueError(f"Unknown image_augmentation keys: {unknown}.")

        size = tuple(int(x) for x in image_size)
        if len(size) != 2 or any(x <= 0 for x in size):
            raise ValueError(f"image_size must contain two positive values, got {size}.")
        self.enabled = bool(config.get("enabled", False))

        crop = _config_section(config, "random_resized_crop")
        self.crop_probability = _validate_probability(
            crop.get("probability", 1.0),
            "image_augmentation.random_resized_crop.probability",
        )
        crop_scale = _validate_range(
            crop.get("scale", (0.9, 1.0)),
            "image_augmentation.random_resized_crop.scale",
            strictly_positive=True,
        )
        crop_ratio = _validate_range(
            crop.get("ratio", (0.95, 1.05)),
            "image_augmentation.random_resized_crop.ratio",
            strictly_positive=True,
        )
        self.random_resized_crop = transforms.RandomResizedCrop(
            size=size,
            scale=crop_scale,
            ratio=crop_ratio,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )

        jitter = _config_section(config, "color_jitter")
        self.jitter_probability = _validate_probability(
            jitter.get("probability", 0.8),
            "image_augmentation.color_jitter.probability",
        )
        brightness = float(jitter.get("brightness", 0.1))
        contrast = float(jitter.get("contrast", 0.1))
        saturation = float(jitter.get("saturation", 0.1))
        hue = float(jitter.get("hue", 0.02))
        if min(brightness, contrast, saturation) < 0.0:
            raise ValueError("Color jitter brightness, contrast, and saturation must be non-negative.")
        if not 0.0 <= hue <= 0.5:
            raise ValueError(f"Color jitter hue must be in [0, 0.5], got {hue}.")
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

        low_light = _config_section(config, "low_light_noise")
        self.low_light_probability = _validate_probability(
            low_light.get("probability", 0.25),
            "image_augmentation.low_light_noise.probability",
        )
        self.exposure = _validate_range(
            low_light.get("exposure", (0.55, 0.9)),
            "image_augmentation.low_light_noise.exposure",
            minimum=0.0,
        )
        self.gamma = _validate_range(
            low_light.get("gamma", (1.0, 1.5)),
            "image_augmentation.low_light_noise.gamma",
            strictly_positive=True,
        )
        self.shot_noise_peak = _validate_range(
            low_light.get("shot_noise_peak", (64.0, 192.0)),
            "image_augmentation.low_light_noise.shot_noise_peak",
            strictly_positive=True,
        )
        self.read_noise_std = _validate_range(
            low_light.get("read_noise_std", (0.0, 0.02)),
            "image_augmentation.low_light_noise.read_noise_std",
            minimum=0.0,
        )

    @staticmethod
    def _sample(values: tuple[float, float], device: torch.device) -> float:
        low, high = values
        if low == high:
            return low
        return float(torch.empty((), device=device).uniform_(low, high).item())

    @staticmethod
    def _draw(probability: float, device: torch.device) -> bool:
        return probability > 0.0 and bool(torch.rand((), device=device).item() < probability)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return frames
        if frames.ndim != 4:
            raise ValueError(f"Video augmentation expects [T,C,H,W], got {tuple(frames.shape)}.")
        if not torch.is_floating_point(frames):
            raise TypeError(f"Video augmentation expects floating-point pixels, got {frames.dtype}.")

        if self._draw(self.crop_probability, frames.device):
            frames = self.random_resized_crop(frames)
        if self._draw(self.jitter_probability, frames.device):
            frames = self.color_jitter(frames)
        if self._draw(self.low_light_probability, frames.device):
            exposure = self._sample(self.exposure, frames.device)
            gamma = self._sample(self.gamma, frames.device)
            shot_noise_peak = self._sample(self.shot_noise_peak, frames.device)
            read_noise_std = self._sample(self.read_noise_std, frames.device)
            frames = frames.mul(exposure).clamp_(0.0, 1.0).pow_(gamma)
            noise_std = torch.sqrt(frames / shot_noise_peak + read_noise_std**2)
            frames = frames + torch.randn_like(frames) * noise_std
        return frames.clamp_(0.0, 1.0)


def obtain_image_size(data_dict: dict, input_keys: list) -> tuple[int, int]:
    r"""Function for obtaining the image size from the data dict.

    Args:
        data_dict (dict): Input data dict
        input_keys (list): List of input keys
    Returns:
        width (int): Width of the input image
        height (int): Height of the input image
    """

    data1 = data_dict[input_keys[0]]
    if isinstance(data1, Image.Image):
        width, height = data1.size
    elif isinstance(data1, torch.Tensor):
        height, width = data1.size()[-2:]
    else:
        raise ValueError("data to random crop should be PIL Image or tensor")

    return width, height


class Augmentor:
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        r"""Base augmentor class

        Args:
            input_keys (list): List of input keys
            output_keys (list): List of output keys
            args (dict): Arguments associated with the augmentation
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = args

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise ValueError("Augmentor not implemented")


class ResizeSmallestSideAspectPreserving(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs aspect-ratio preserving resizing.
        Image is resized to the dimension which has the smaller ratio of (size / target_size).
        First we compute (w_img / w_target) and (h_img / h_target) and resize the image
        to the dimension that has the smaller of these ratios.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        img_w, img_h = self.args["img_w"], self.args["img_h"]

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert target_size[0] >= img_h and target_size[1] >= img_w, (
            f"Resize error. orig {(orig_w, orig_h)} desire {(img_w, img_h)} compute {target_size}"
        )

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,  # type: ignore
                interpolation=self.args.get("interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class CenterCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs center crop.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
            We also save the cropping parameters in the aug_params dict
            so that it will be used by other transforms.
        """
        assert (self.args is not None) and ("img_w" in self.args) and ("img_h" in self.args), (
            "Please specify size in args"
        )

        img_w, img_h = self.args["img_w"], self.args["img_h"]

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        for key in self.input_keys:
            data_dict[key] = transforms_F.center_crop(data_dict[key], [img_h, img_w])

        # We also add the aug params we use. This will be useful for other transforms
        crop_x0 = (orig_w - img_w) // 2
        crop_y0 = (orig_h - img_h) // 2
        cropping_params = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": img_w,
            "crop_h": img_h,
        }

        if "aug_params" not in data_dict:
            data_dict["aug_params"] = dict()

        data_dict["aug_params"]["cropping"] = cropping_params
        data_dict["padding_mask"] = torch.zeros((1, cropping_params["crop_h"], cropping_params["crop_w"]))
        return data_dict


class Normalize(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs data normalization.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """
        assert self.args is not None, "Please specify args"

        mean = self.args["mean"]
        std = self.args["std"]

        for key in self.input_keys:
            if isinstance(data_dict[key], torch.Tensor):
                data = data_dict[key].to(dtype=torch.float32)
                if data_dict[key].dtype == torch.uint8:
                    data = data / 255.0
                data_dict[key] = data.to(dtype=torch.get_default_dtype())
            else:
                data_dict[key] = transforms_F.to_tensor(
                    data_dict[key]
                )  # division by 255 is applied in to_tensor()

            data_dict[key] = transforms_F.normalize(tensor=data_dict[key], mean=mean, std=std)
        return data_dict
