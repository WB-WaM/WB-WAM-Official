from __future__ import annotations

from enum import Enum
import inspect
from typing import Any, Mapping

import numpy as np

from bridge.common.policy import PolicyAdapter, PolicyInfo
from bridge.wbwam.checkpoint import (
    _resolve_path,
    format_deployment_prompt,
    validate_deployment_config,
)
from bridge.wbwam.codec import WBFlatCodec
from bridge.wbwam.text_cache import load_prompt_embedding


class WAMInferenceMode(str, Enum):
    """Deployment-time conditioning path for the trained Action Expert."""

    IDM = "idm"
    WBWAM = "wbwam"

    @classmethod
    def parse(cls, value: Any) -> "WAMInferenceMode":
        key = str(value or "").strip().lower().replace("-style", "")
        aliases = {"first_frame": cls.WBWAM, "first-frame": cls.WBWAM}
        if key in aliases:
            return aliases[key]
        try:
            return cls(key)
        except ValueError as exc:
            raise ValueError(f"unsupported policy.inference_mode={value!r}; expected 'idm' or 'wbwam'") from exc


_MODEL_TARGET_BY_INFERENCE_MODE = {
    WAMInferenceMode.IDM: "wbwam.runtime.create_wbwam_idm",
    WAMInferenceMode.WBWAM: "wbwam.runtime.create_wbwam",
}


def _configure_model_for_inference(model_cfg: Any, mode: WAMInferenceMode) -> str:
    """Keep optional-IDM checkpoints intact; adapt older single-mode configs."""

    from omegaconf import OmegaConf

    configured_target = str(model_cfg.get("_target_", ""))
    if configured_target.endswith("create_wbwam_optional_idm"):
        return configured_target

    target = _MODEL_TARGET_BY_INFERENCE_MODE[mode]
    OmegaConf.update(model_cfg, "_target_", target, merge=False)
    if "action_idm_prob" in model_cfg:
        del model_cfg["action_idm_prob"]
    return target


def _validate_model_for_inference(model: Any, mode: WAMInferenceMode) -> None:
    expected = {"WBWAMIDM", "WBWAMOptionalIDM"} if mode == WAMInferenceMode.IDM else {"WBWAM", "WBWAMOptionalIDM"}
    actual = model.__class__.__name__
    if actual not in expected:
        raise ValueError(f"policy.inference_mode={mode.value!r} requires one of {sorted(expected)}, got {actual}")
    if not callable(getattr(model, "infer_action", None)):
        raise ValueError(f"loaded {actual} model does not expose infer_action()")


def _run_action_inference(
    model: Any,
    *,
    mode: WAMInferenceMode,
    kwargs: dict[str, Any],
    num_video_frames: int,
) -> dict[str, Any]:
    """Dispatch to either two-stage IDM or direct first-frame action inference."""

    call_kwargs = dict(kwargs)
    parameters = inspect.signature(model.infer_action).parameters
    if "action_infer_mode" in parameters:
        call_kwargs["action_infer_mode"] = "idm" if mode == WAMInferenceMode.IDM else "first_frame"
    if mode == WAMInferenceMode.IDM:
        call_kwargs["num_video_frames"] = int(num_video_frames)
    return model.infer_action(**call_kwargs)


def _validate_required_prompt_embeddings(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not value:
        raise ValueError("policy.required_prompt_embeddings must be a non-empty list")
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each required prompt embedding must be a mapping")
        path = _resolve_path(item.get("path"))
        expected_prompt = str(item.get("prompt", "")).strip()
        if path is None or not path.is_file() or not expected_prompt:
            raise FileNotFoundError(f"invalid required WBWAM prompt embedding: {item!r}")
        _, _, cached_prompt = load_prompt_embedding(path)
        if cached_prompt != expected_prompt:
            raise ValueError(f"WBWAM cached prompt mismatch for {path}")


def _resolve_inference_seed(policy_config: dict[str, Any], training_config: Any) -> int | None:
    value = policy_config["seed"] if "seed" in policy_config else training_config.get("seed")
    return None if value is None else int(value)


def _resize_rgb(image: np.ndarray, *, size_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    image_arr = np.asarray(image)
    if image_arr.ndim != 3 or image_arr.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got {image_arr.shape}")
    if image_arr.dtype != np.uint8:
        image_arr = np.clip(image_arr, 0, 255).astype(np.uint8)
    height, width = size_hw
    resized = Image.fromarray(image_arr, mode="RGB").resize((width, height), resample=Image.BILINEAR)
    return np.array(resized, dtype=np.uint8, copy=True)


def _size_hw(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"size must be [H, W], got {value!r}")
    return int(value[0]), int(value[1])


def compose_wbwam_image(observation: dict[str, Any], input_config: dict[str, Any]) -> np.ndarray:
    """Return a uint8 HxWx3 WBWAM input image from a bridge observation."""

    mode = str(input_config.get("image_mode", "rgb")).strip().lower()
    concat = str(input_config.get("concat_multi_camera", "vertical")).strip().lower()
    if mode == "rgb":
        rgb_size = _size_hw(input_config.get("rgb_size"), default=(256, 320))
        return _resize_rgb(observation["observation/image"], size_hw=rgb_size)
    if mode != "rgbd":
        raise ValueError(f"wam_input.image_mode must be 'rgb' or 'rgbd', got {mode!r}")

    rgb_size = _size_hw(input_config.get("rgb_size"), default=(240, 320))
    depth_size = _size_hw(input_config.get("depth_size"), default=rgb_size)
    rgb = _resize_rgb(observation["observation/image"], size_hw=rgb_size)
    depth = _resize_rgb(observation["observation/depth_image"], size_hw=depth_size)
    if concat == "vertical":
        if rgb.shape[1] != depth.shape[1]:
            raise ValueError(f"vertical RGBD concat requires same width, got {rgb.shape} and {depth.shape}")
        return np.concatenate([rgb, depth], axis=0)
    if concat == "horizontal":
        if rgb.shape[0] != depth.shape[0]:
            raise ValueError(f"horizontal RGBD concat requires same height, got {rgb.shape} and {depth.shape}")
        return np.concatenate([rgb, depth], axis=1)
    raise ValueError(f"unsupported concat_multi_camera={concat!r}")


def extract_wbwam_state(observation: dict[str, Any], input_config: dict[str, Any]) -> np.ndarray:
    state = np.asarray(observation["states"], dtype=np.float32).reshape(-1)
    expected = input_config.get("expected_state_dim") or input_config.get("base_state_dim")
    if expected is not None and state.size != int(expected):
        raise ValueError(f"state dim {state.size}, expected {int(expected)}")
    if not np.all(np.isfinite(state)):
        raise ValueError("state contains NaN/Inf")
    return state


def _mixed_precision_to_dtype(mixed_precision: str):
    import torch

    key = str(mixed_precision).strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported mixed precision {mixed_precision!r}")


def _wbwam_uses_video_kv_cache(model: Any) -> bool:
    mot = getattr(model, "mot", None)
    return (
        model.__class__.__name__ == "WBWAM"
        and callable(getattr(mot, "prefill_video_cache", None))
        and callable(getattr(mot, "forward_action_with_video_cache", None))
    )


def _configure_inference_action_layout(model: Any, codec: WBFlatCodec) -> None:
    model_dim_indices = codec.model_dim_indices
    if model_dim_indices is None:
        model.set_inference_action_valid_dim(codec.action_dim)
        return
    setter = getattr(model, "set_inference_action_valid_indices", None)
    if not callable(setter):
        raise TypeError("mapped WB action layout requires model.set_inference_action_valid_indices()")
    setter(model_dim_indices)


class WBWAMAdapter(PolicyAdapter):
    def __init__(self, *, policy_config: dict[str, Any], input_config: dict[str, Any]):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        import torch

        artifacts, cfg, codec = validate_deployment_config(policy_config)
        image_size = tuple(cfg.data.get("video_size", (224, 224)))
        configured_size = tuple(input_config.get("rgb_size", image_size))
        if str(input_config.get("image_mode", "rgb")) != "rgb" or configured_size != image_size:
            raise ValueError(f"wam_input must use RGB with rgb_size={list(image_size)} to match training")
        input_config = dict(input_config, rgb_size=list(image_size))
        checkpoint_path = artifacts.weights
        dataset_stats_path = artifacts.stats
        self._inference_mode = WAMInferenceMode.parse(policy_config.get("inference_mode"))
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        configured_model_target = str(model_cfg.get("_target_", ""))
        deployment_model_target = _configure_model_for_inference(model_cfg, self._inference_mode)
        prompt_embedding_path = _resolve_path(policy_config.get("prompt_embedding_path"))
        model_cfg.load_text_encoder = prompt_embedding_path is None
        # The trained checkpoint supplies both experts; only frozen Wan assets
        # (VAE and optionally UMT5) still need to be available.
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
        model_cfg.compile_training_denoise = False
        device = str(policy_config.get("device", cfg.get("device", "cuda")))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --mock-policy for a CPU-only smoke test")
        model_dtype = _mixed_precision_to_dtype(
            str(policy_config.get("mixed_precision", cfg.get("mixed_precision", "bf16")))
        )

        _validate_required_prompt_embeddings(policy_config.get("required_prompt_embeddings"))
        self._model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        payload = self._model.load_checkpoint(
            str(checkpoint_path),
            strict=True,
        )
        if "mot" not in payload or "proprio_encoder" not in payload:
            raise ValueError("WB-WAM deployment requires complete mot and proprio_encoder weights")
        del payload
        self._model = self._model.to(device).eval()
        _validate_model_for_inference(self._model, self._inference_mode)
        self._uses_video_kv_cache = _wbwam_uses_video_kv_cache(self._model)
        self._prompt_context = None
        self._prompt_context_mask = None
        self._cached_prompt = None
        if prompt_embedding_path is not None:
            if not prompt_embedding_path.exists():
                raise FileNotFoundError(f"WBWAM prompt embedding cache not found: {prompt_embedding_path}")
            (
                self._prompt_context,
                self._prompt_context_mask,
                self._cached_prompt,
            ) = load_prompt_embedding(prompt_embedding_path)
            expected_context_len = int(cfg.data.get("context_len", model_cfg.get("tokenizer_max_len", 192)))
            expected_text_dim = int(model_cfg.video_dit_config["text_dim"])
            if tuple(self._prompt_context.shape) != (expected_context_len, expected_text_dim):
                raise ValueError(
                    "WBWAM cached context shape does not match training config: "
                    f"{tuple(self._prompt_context.shape)} != "
                    f"({expected_context_len}, {expected_text_dim})"
                )
            print(
                "[TextCache] WBWAM loaded cached context; UMT5 text encoder is not resident in the policy process."
            )

        data_cfg = cfg.data
        self._flat_codec = codec
        inferred_action_dim = codec.action_dim
        inferred_state_dim = codec.state_dim
        data_num_frames = int(data_cfg.num_frames)
        action_video_freq_ratio = int(data_cfg.get("action_video_freq_ratio", 4))
        self._action_representation = codec.action_representation
        self._relative_action_reference = codec.relative_action_reference
        self._relative_joint_ranges = codec.relative_joint_ranges
        _configure_inference_action_layout(self._model, codec)
        expected_action_horizon = data_num_frames - 1
        action_dim = int(policy_config.get("action_dim", inferred_action_dim))
        state_dim = int(policy_config.get("state_dim", inferred_state_dim))
        if action_dim != inferred_action_dim:
            raise ValueError(
                f"policy.action_dim={action_dim} does not match training semantic action dim {inferred_action_dim}"
            )
        if state_dim != inferred_state_dim:
            raise ValueError(
                f"policy.state_dim={state_dim} does not match training semantic state dim {inferred_state_dim}"
            )
        action_horizon = int(policy_config.get("action_horizon", expected_action_horizon))
        if action_horizon != expected_action_horizon:
            raise ValueError(
                f"policy.action_horizon={action_horizon} does not match training horizon={expected_action_horizon}"
            )
        self._info = PolicyInfo(
            name=str(policy_config.get("name", "wbwam")),
            action_dim=action_dim,
            action_horizon=action_horizon,
            state_dim=state_dim,
            runtime_version=f"wbwam:{self._inference_mode.value}",
        )
        self._input_config = dict(input_config)
        self._action_horizon = action_horizon
        self._num_inference_steps = int(
            policy_config.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10))
        )
        self._sigma_shift = policy_config.get("sigma_shift")
        self._seed = _resolve_inference_seed(policy_config, cfg)
        self._text_cfg_scale = float(policy_config.get("text_cfg_scale", 1.0))
        self._negative_prompt = str(policy_config.get("negative_prompt", ""))
        self._rand_device = str(policy_config.get("rand_device", "cpu"))
        self._tiled = bool(policy_config.get("tiled", False))
        self._training_config = cfg
        self._policy_config = dict(policy_config)
        self._compile_action_infer = bool(policy_config.get("compile_action_infer", False))
        expected_num_video_frames = expected_action_horizon // action_video_freq_ratio + 1
        self._num_video_frames = int(policy_config.get("num_video_frames", expected_num_video_frames))
        if self._num_video_frames != expected_num_video_frames:
            raise ValueError(
                "policy.num_video_frames="
                f"{self._num_video_frames} does not match training video horizon={expected_num_video_frames}"
            )

        print(
            "[INFO] WBWAM policy loaded: "
            f"name={self._info.name} state_dim={state_dim} horizon={action_horizon} "
            f"action_dim={action_dim} model_class={self._model.__class__.__name__} "
            f"inference_mode={self._inference_mode.value} "
            f"seed={self._seed if self._seed is not None else 'random'} "
            f"video_kv_cache={self._uses_video_kv_cache} ckpt={checkpoint_path} stats={dataset_stats_path}"
        )
        if configured_model_target != deployment_model_target:
            print(
                "[INFO] WAM deployment model target selected by inference mode: "
                f"{configured_model_target} -> {deployment_model_target}"
            )

    def _normalize_state(self, state: np.ndarray):
        return self._flat_codec.normalize_state(state)

    def _denormalize_action(self, action):
        return self._flat_codec.denormalize_action(action)

    def _format_prompt(self, task: str) -> str:
        return format_deployment_prompt(self._training_config, self._policy_config, task)

    def _prompt_conditioning(self, prompt: str) -> dict[str, Any]:
        if self._prompt_context is None:
            return {"prompt": prompt}
        if prompt != self._cached_prompt:
            raise ValueError(
                "WBWAM task prompt changed after the cached model was loaded. "
                "Restart the bridge with the new --prompt so its embedding can be generated "
                "before loading WAM."
            )
        return {
            "prompt": None,
            "context": self._prompt_context,
            "context_mask": self._prompt_context_mask,
        }

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
        del rtc_prefix_actions, rtc_prefix_mask, rtc_guidance_target, rtc_guidance_mask, rtc_guidance_scale
        import torch

        image = compose_wbwam_image(observation, self._input_config)
        image_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=self._model.device,
                dtype=self._model.torch_dtype,
            )
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        state = extract_wbwam_state(observation, self._input_config)
        proprio = self._normalize_state(state)
        task_prompt = str(observation.get("prompt", ""))
        prompt = self._format_prompt(task_prompt)

        kwargs = {
            "input_image": image_tensor,
            "action_horizon": self._action_horizon,
            "proprio": proprio,
            "negative_prompt": self._negative_prompt,
            "text_cfg_scale": self._text_cfg_scale,
            "num_inference_steps": self._num_inference_steps,
            "sigma_shift": None if self._sigma_shift is None else float(self._sigma_shift),
            "seed": None if self._seed is None else int(self._seed),
            "rand_device": self._rand_device,
            "tiled": self._tiled,
            "compile_action_infer": self._compile_action_infer,
        }
        kwargs.update(self._prompt_conditioning(prompt))
        with torch.no_grad():
            pred = _run_action_inference(
                self._model,
                mode=self._inference_mode,
                kwargs=kwargs,
                num_video_frames=self._num_video_frames,
            )
        if "action" not in pred:
            raise ValueError("WBWAM result missing 'action'")
        return self._denormalize_action(pred["action"])

    @property
    def action_representation(self) -> str | None:
        return self._action_representation

    @property
    def relative_action_reference(self) -> str | None:
        return self._relative_action_reference

    @property
    def relative_joint_ranges(self) -> tuple[tuple[int, int], ...] | None:
        return self._relative_joint_ranges

    @property
    def info(self) -> PolicyInfo:
        return self._info
