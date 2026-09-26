from __future__ import annotations

from enum import Enum
import inspect
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from .policy import PolicyAdapter, PolicyInfo
from .text_cache import load_prompt_embedding
from .wb_flat import WBFlatCodec

REPO_ROOT = Path(__file__).resolve().parents[3]

_HGPT_REFERENCE76_STATE_SLICES = ["states[110:146]", "states[67:107]"]
_HGPT_REFERENCE76_ACTION_SLICES = ["action[0:36]", "action[64:104]"]
_HGPT_REFERENCE76_STATE_SLICE_METADATA = [
    [
        {"start": 110, "end": 146, "label": "states[110:146]"},
        {"start": 67, "end": 107, "label": "states[67:107]"},
    ]
]
_HGPT_REFERENCE76_ACTION_SLICE_METADATA = [
    [
        {"start": 0, "end": 36, "label": "action[0:36]"},
        {"start": 64, "end": 104, "label": "action[64:104]"},
    ]
]
_HGPT_XYZ75_STATE_SLICES = [
    "states[9:38]",
    "states[107:110]",
    "states[67:87]",
    "states[87:107]",
    "states[110:113]",
]
_HGPT_XYZ75_ACTION_SLICES = [
    "action[104:133]",
    "action[133:136]",
    "action[64:84]",
    "action[84:104]",
    "action[0:3]",
]
_HGPT_XYZ75_STATE_SLICE_METADATA = [
    [
        {"start": 9, "end": 38, "label": "states[9:38]"},
        {"start": 107, "end": 110, "label": "states[107:110]"},
        {"start": 67, "end": 87, "label": "states[67:87]"},
        {"start": 87, "end": 107, "label": "states[87:107]"},
        {"start": 110, "end": 113, "label": "states[110:113]"},
    ]
]
_HGPT_XYZ75_ACTION_SLICE_METADATA = [
    [
        {"start": 104, "end": 133, "label": "action[104:133]"},
        {"start": 133, "end": 136, "label": "action[133:136]"},
        {"start": 64, "end": 84, "label": "action[64:84]"},
        {"start": 84, "end": 104, "label": "action[84:104]"},
        {"start": 0, "end": 3, "label": "action[0:3]"},
    ]
]
_HGPT_CONTRACTS = ("reference76", "xyz75")


class WAMInferenceMode(str, Enum):
    """Deployment-time conditioning path for the trained Action Expert."""

    IDM = "idm"
    WBWAM = "wbwam"

    @classmethod
    def parse(cls, value: Any) -> "WAMInferenceMode":
        key = str(value or "").strip().lower().replace("-style", "")
        aliases = {"first_frame": cls.WBWAM, "first-frame": cls.WBWAM, "ff": cls.WBWAM}
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


def _resolve_path(value: str | Path | None, *, base: Path = REPO_ROOT) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _checkpoint_run_root(path: Path) -> Path:
    parent = path.parent
    if parent.name == "weights":
        if parent.parent.name == "checkpoints":
            return parent.parent.parent
        return parent.parent
    return parent


def _metadata_run_root(path: Path) -> Path:
    parent = path.parent
    if parent.name == "metadata":
        return parent.parent
    return parent


def _resolve_checkpoint_path(value: str | Path | None) -> Path:
    checkpoint_path = _resolve_path(value)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"WB-WAM checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"WB-WAM checkpoint must be a file: {checkpoint_path}")
    if checkpoint_path.stat().st_size == 0:
        raise ValueError(
            f"WB-WAM checkpoint is empty; replace the placeholder with the real weights: {checkpoint_path}"
        )
    return checkpoint_path


def _validate_run_artifacts(
    config_path: Path | None,
    dataset_stats_path: Path,
    checkpoint_path: Path,
) -> None:
    if config_path is None:
        return
    roots = {
        "config": _metadata_run_root(config_path).resolve(),
        "dataset_stats": _metadata_run_root(dataset_stats_path).resolve(),
        "checkpoint": _checkpoint_run_root(checkpoint_path).resolve(),
    }
    if len(set(roots.values())) != 1:
        detail = ", ".join(f"{name}={path}" for name, path in roots.items())
        raise ValueError(f"WB-WAM artifacts must come from the same training run: {detail}")


def _artifact_config_path(policy_config: Mapping[str, Any], runtime_config_path: Path | None) -> Path | None:
    """Return the immutable training config used for same-run validation.

    A deployment may load a derived runtime YAML from its output directory.
    In that case, ``artifact_config_path`` preserves the source training-run
    identity while ``config_path`` remains the YAML that is actually loaded.
    """

    value = policy_config.get("artifact_config_path")
    artifact_config_path = _resolve_path(value) if value is not None else runtime_config_path
    if artifact_config_path is not None and not artifact_config_path.is_file():
        raise FileNotFoundError(f"WB-WAM artifact config not found: {artifact_config_path}")
    return artifact_config_path


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "ListConfig":
        return [_plain_value(item) for item in value]
    return value


def _stats_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    return np.asarray(value).shape


def _validate_hgpt_flat_checkpoint_contract(
    data_config: Any,
    dataset_stats: Mapping[str, Any],
    *,
    action_horizon: int,
    hgpt_contract: str = "reference76",
) -> None:
    """Validate one of the explicit flat HGPT deployment contracts."""

    contract = str(hgpt_contract).strip().lower()
    if contract not in _HGPT_CONTRACTS:
        raise ValueError(
            f"unsupported policy.hgpt_contract={hgpt_contract!r}; expected one of {list(_HGPT_CONTRACTS)}"
        )

    if contract == "xyz75":
        if int(action_horizon) != 32:
            raise ValueError(f"HGPT xyz75 WB-WAM checkpoint requires action_horizon=32, got {action_horizon}")
        semantic_dim = 75
        state_slices = _HGPT_XYZ75_STATE_SLICES
        action_slices = _HGPT_XYZ75_ACTION_SLICES
        state_slice_metadata = _HGPT_XYZ75_STATE_SLICE_METADATA
        action_slice_metadata = _HGPT_XYZ75_ACTION_SLICE_METADATA
    else:
        semantic_dim = 76
        state_slices = _HGPT_REFERENCE76_STATE_SLICES
        action_slices = _HGPT_REFERENCE76_ACTION_SLICES
        state_slice_metadata = _HGPT_REFERENCE76_STATE_SLICE_METADATA
        action_slice_metadata = _HGPT_REFERENCE76_ACTION_SLICE_METADATA

    expected_config = {
        "state_dim": semantic_dim,
        "action_dim": semantic_dim,
        "proprio_dim": 96,
        "action_target_dim": 96,
        "state_slices": state_slices,
        "action_slices": action_slices,
        "action_representation": "absolute",
        "relative_joint_ranges": [],
        "relative_action_reference": "observation",
    }
    if contract == "xyz75":
        expected_config.update(
            raw_state_dim=146,
            raw_action_dim=136,
            model_dim_indices=None,
        )
    for key, expected in expected_config.items():
        actual = _plain_value(data_config.get(key))
        if actual != expected:
            raise ValueError(f"HGPT {contract} WB-WAM checkpoint requires data.{key}={expected!r}, got {actual!r}")

    metadata = dataset_stats.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"HGPT {contract} WB-WAM dataset_stats requires a metadata mapping")
    expected_metadata = {
        "state_dim": semantic_dim,
        "action_dim": semantic_dim,
        "proprio_dim": 96,
        "action_target_dim": 96,
        "raw_state_dim": 146,
        "raw_action_dim": 136,
        "action_horizon": int(action_horizon),
        "action_representation": "absolute",
        "relative_joint_ranges": [],
        "state_slices": state_slice_metadata,
        "action_slices": action_slice_metadata,
    }
    if contract == "xyz75":
        expected_metadata["relative_action_reference"] = "observation"
    for key, expected in expected_metadata.items():
        actual = _plain_value(metadata.get(key))
        if actual != expected:
            raise ValueError(
                f"HGPT {contract} WB-WAM dataset_stats requires metadata.{key}={expected!r}, got {actual!r}"
            )
    stats_reference = _plain_value(metadata.get("relative_action_reference"))
    if contract == "reference76" and stats_reference not in {None, "observation"}:
        raise ValueError(
            "HGPT WB-WAM absolute dataset_stats requires absent or observation metadata.relative_action_reference"
        )
    if contract == "xyz75" and _plain_value(metadata.get("model_dim_indices")) is not None:
        raise ValueError("HGPT xyz75 WB-WAM dataset_stats requires absent or null metadata.model_dim_indices")

    state_stats = (dataset_stats.get("state") or {}).get("proprio")
    action_stats = (dataset_stats.get("action") or {}).get("action")
    if not isinstance(state_stats, Mapping) or not isinstance(action_stats, Mapping):
        raise ValueError(f"HGPT {contract} WB-WAM dataset_stats requires state.proprio and action.action")
    for key in ("global_q01", "global_q99"):
        if key not in state_stats or _stats_shape(state_stats[key]) != (semantic_dim,):
            raise ValueError(
                f"HGPT {contract} WB-WAM dataset_stats state.proprio.{key} must have shape [{semantic_dim}]"
            )
    expected_action_shape = (int(action_horizon), semantic_dim)
    for key in ("stepwise_q01", "stepwise_q99"):
        if key not in action_stats or _stats_shape(action_stats[key]) != expected_action_shape:
            raise ValueError(
                f"HGPT {contract} WB-WAM dataset_stats "
                f"action.action.{key} must have shape [{action_horizon},{semantic_dim}]"
            )


def _resolve_inference_seed(policy_config: dict[str, Any], training_config: Any) -> int | None:
    value = policy_config["seed"] if "seed" in policy_config else training_config.get("seed")
    return None if value is None else int(value)


def _add_wbwam_paths(wbwam_root: Path) -> None:
    src_root = wbwam_root / "src"
    for path in (wbwam_root, src_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _resize_rgb(image: np.ndarray, *, size_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    image_arr = np.asarray(image)
    if image_arr.ndim != 3 or image_arr.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got {image_arr.shape}")
    if image_arr.dtype != np.uint8:
        image_arr = np.clip(image_arr, 0, 255).astype(np.uint8)
    height, width = size_hw
    resized = Image.fromarray(image_arr, mode="RGB").resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _size_hw(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"size must be [H, W], got {value!r}")
    return int(value[0]), int(value[1])


def compose_wbwam_image(observation: dict[str, Any], input_config: dict[str, Any]) -> np.ndarray:
    """Return a uint8 HxWx3 WB-WAM input image from a simulator observation."""

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
    action_model_dim_indices = codec.action_model_dim_indices
    if action_model_dim_indices is None:
        model.set_inference_action_valid_dim(codec.action_dim)
        return
    setter = getattr(model, "set_inference_action_valid_indices", None)
    if not callable(setter):
        raise TypeError("mapped WB action layout requires model.set_inference_action_valid_indices()")
    setter(action_model_dim_indices)


class WBWAMAdapter(PolicyAdapter):
    def __init__(self, *, policy_config: dict[str, Any], input_config: dict[str, Any]):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        import torch

        wbwam_root = _resolve_path(policy_config.get("wbwam_root", "training"))
        assert wbwam_root is not None
        _add_wbwam_paths(wbwam_root)

        from wbwam.datasets.normalization import load_dataset_stats_from_json
        from wbwam.datasets.prompt_builder import DEFAULT_PROMPT

        config_path_value = policy_config.get("config_path")
        config_path = _resolve_path(config_path_value)
        if config_path_value is not None:
            if config_path is None or not config_path.exists():
                raise FileNotFoundError(f"WB-WAM training config not found: {config_path}")
            cfg = OmegaConf.load(config_path)
        else:
            config_name = str(policy_config.get("config_name", "train"))
            overrides = []
            wb_task_name = policy_config.get("wb_task")
            task_name = policy_config.get("task")
            if wb_task_name:
                overrides.append(f"wb_task={wb_task_name}")
            elif task_name:
                overrides.append(f"task={task_name}")
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_dir(version_base="1.3", config_dir=str(wbwam_root / "configs")):
                cfg = compose(config_name=config_name, overrides=overrides)

        if "skip_dit_load_from_pretrain" in policy_config:
            OmegaConf.update(
                cfg,
                "model.skip_dit_load_from_pretrain",
                bool(policy_config["skip_dit_load_from_pretrain"]),
                merge=True,
            )
        if "action_dit_pretrained_path" in policy_config:
            OmegaConf.update(
                cfg,
                "model.action_dit_pretrained_path",
                policy_config["action_dit_pretrained_path"],
                merge=True,
            )

        self._inference_mode = WAMInferenceMode.parse(policy_config.get("inference_mode"))
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        configured_model_target = str(model_cfg.get("_target_", ""))
        deployment_model_target = _configure_model_for_inference(model_cfg, self._inference_mode)
        prompt_embedding_path = _resolve_path(policy_config.get("prompt_embedding_path"))
        model_cfg.load_text_encoder = prompt_embedding_path is None
        device = str(policy_config.get("device", cfg.get("device", "cuda")))
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARNING] CUDA is unavailable; falling back to cpu for WB-WAM")
            device = "cpu"
        model_dtype = _mixed_precision_to_dtype(
            str(policy_config.get("mixed_precision", cfg.get("mixed_precision", "bf16")))
        )

        checkpoint_path = _resolve_checkpoint_path(policy_config.get("checkpoint_path"))
        self._model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        self._model.load_checkpoint(
            str(checkpoint_path),
            strict=bool(policy_config.get("strict_checkpoint_load", False)),
        )
        self._model = self._model.to(device).eval()
        _validate_model_for_inference(self._model, self._inference_mode)
        self._uses_video_kv_cache = _wbwam_uses_video_kv_cache(self._model)
        self._prompt_context = None
        self._prompt_context_mask = None
        self._cached_prompt = None
        if prompt_embedding_path is not None:
            if not prompt_embedding_path.exists():
                raise FileNotFoundError(f"WB-WAM prompt embedding cache not found: {prompt_embedding_path}")
            (
                self._prompt_context,
                self._prompt_context_mask,
                self._cached_prompt,
            ) = load_prompt_embedding(prompt_embedding_path)
            expected_context_len = int(cfg.data.get("context_len", model_cfg.get("tokenizer_max_len", 192)))
            expected_text_dim = int(model_cfg.video_dit_config["text_dim"])
            if tuple(self._prompt_context.shape) != (expected_context_len, expected_text_dim):
                raise ValueError(
                    "WB-WAM cached context shape does not match training config: "
                    f"{tuple(self._prompt_context.shape)} != "
                    f"({expected_context_len}, {expected_text_dim})"
                )
            print(
                "[TextCache] WB-WAM loaded cached context; "
                "UMT5 text encoder is not resident in the policy process."
            )

        data_cfg = cfg.data
        stats_value = policy_config.get("dataset_stats_path") or data_cfg.get("pretrained_norm_stats")
        dataset_stats_path = _resolve_path(stats_value)
        if dataset_stats_path is None or not dataset_stats_path.exists():
            raise FileNotFoundError(f"WB-WAM dataset_stats.json not found: {dataset_stats_path}")
        artifact_config_path = _artifact_config_path(policy_config, config_path)
        _validate_run_artifacts(artifact_config_path, dataset_stats_path, checkpoint_path)
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))

        data_target = str(data_cfg.get("_target_", ""))
        is_flat_wb = data_target.endswith("WBMixtureDataset") or (
            "state_dim" in data_cfg and "action_dim" in data_cfg and "train" not in data_cfg
        )
        self._processor = None
        self._flat_codec = None
        self._action_representation: str | None = None
        self._relative_action_reference: str | None = None
        self._relative_joint_ranges: tuple[tuple[int, int], ...] | None = None
        if is_flat_wb:
            self._flat_codec = WBFlatCodec(data_cfg, dataset_stats)
            inferred_action_dim = self._flat_codec.action_dim
            inferred_state_dim = self._flat_codec.state_dim
            data_num_frames = int(data_cfg.num_frames)
            action_video_freq_ratio = int(data_cfg.get("action_video_freq_ratio", 4))
            self._action_representation = self._flat_codec.action_representation
            self._relative_action_reference = self._flat_codec.relative_action_reference
            self._relative_joint_ranges = self._flat_codec.relative_joint_ranges
            model_proprio_dim = int(model_cfg.get("proprio_dim", self._flat_codec.proprio_dim))
            model_action_dim = int(
                model_cfg.action_dit_config.get("action_dim", self._flat_codec.action_target_dim)
            )
            if model_proprio_dim != self._flat_codec.proprio_dim:
                raise ValueError(
                    f"model.proprio_dim={model_proprio_dim} does not match "
                    f"data.proprio_dim={self._flat_codec.proprio_dim}"
                )
            if model_action_dim != self._flat_codec.action_target_dim:
                raise ValueError(
                    "model.action_dit_config.action_dim="
                    f"{model_action_dim} does not match "
                    f"data.action_target_dim={self._flat_codec.action_target_dim}"
                )
            _configure_inference_action_layout(self._model, self._flat_codec)
            if self._flat_codec.action_model_dim_indices is not None:
                invalid_dims = sorted(
                    set(range(model_action_dim)) - set(self._flat_codec.action_model_dim_indices)
                )
                print(
                    "[INFO] WB-WAM sparse action padding fixed at zero during inference: "
                    f"invalid_dims={invalid_dims} model_dim={model_action_dim}"
                )
            elif inferred_action_dim < model_action_dim:
                print(
                    "[INFO] WB-WAM action padding fixed at zero during inference: "
                    f"valid_dim={inferred_action_dim} model_dim={model_action_dim}"
                )
        else:
            self._processor = instantiate(data_cfg.train.processor).eval()
            self._processor.set_normalizer_from_stats(dataset_stats)
            inferred_action_dim = int(data_cfg.train.processor.action_output_dim)
            inferred_state_dim = int(data_cfg.train.processor.proprio_output_dim)
            data_num_frames = int(data_cfg.train.num_frames)
            action_video_freq_ratio = int(data_cfg.train.get("action_video_freq_ratio", 4))

        execution_backend = str(policy_config.get("execution_backend", "sonic")).strip().lower()
        if execution_backend not in {"sonic", "hgpt"}:
            raise ValueError(
                f"unsupported policy.execution_backend={execution_backend!r}; expected 'sonic' or 'hgpt'"
            )
        expected_action_horizon = data_num_frames - 1
        if execution_backend == "hgpt":
            if self._flat_codec is None:
                raise ValueError("HGPT execution requires a flat WB checkpoint contract")
            _validate_hgpt_flat_checkpoint_contract(
                data_cfg,
                dataset_stats,
                action_horizon=expected_action_horizon,
                hgpt_contract=policy_config.get("hgpt_contract", "reference76"),
            )

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
        default_prompt_template = data_cfg.get("instruction_template", DEFAULT_PROMPT)
        self._prompt_template = str(policy_config.get("prompt_template", default_prompt_template))
        self._prompt_fields = {
            "visual_description": str(policy_config.get("visual_description", "")),
            "control_description": str(policy_config.get("control_description", "")),
        }
        self._compile_action_infer = bool(policy_config.get("compile_action_infer", False))
        expected_num_video_frames = expected_action_horizon // action_video_freq_ratio + 1
        self._num_video_frames = int(policy_config.get("num_video_frames", expected_num_video_frames))
        if self._num_video_frames != expected_num_video_frames:
            raise ValueError(
                "policy.num_video_frames="
                f"{self._num_video_frames} does not match training video horizon={expected_num_video_frames}"
            )

        print(
            "[INFO] WB-WAM policy loaded: "
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
        import torch

        if self._flat_codec is not None:
            return self._flat_codec.normalize_state(state)
        if self._processor is None:
            raise RuntimeError("WB-WAM normalization backend is not initialized")
        state_meta = self._processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("WB-WAM adapter expects one merged state key")
        state_key = state_meta[0]["key"]
        batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self._processor.action_state_transform(batch)
        batch = self._processor.normalizer.forward(batch)
        return batch["state"][state_key]

    def _denormalize_action(self, action):
        import torch

        if self._flat_codec is not None:
            return self._flat_codec.denormalize_action(action)
        if self._processor is None:
            raise RuntimeError("WB-WAM normalization backend is not initialized")
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"expected WB-WAM action [B,T,D], got {tuple(action.shape)}")
        action_meta = self._processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("WB-WAM adapter expects one merged action key")
        action_key = action_meta[0]["key"]
        normalizer = self._processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()[0].astype(np.float32, copy=False)

    def _format_prompt(self, task: str) -> str:
        try:
            return self._prompt_template.format(task=task, **self._prompt_fields)
        except KeyError as exc:
            raise ValueError(f"unknown field in policy.prompt_template: {exc}") from exc

    def _prompt_conditioning(self, prompt: str) -> dict[str, Any]:
        if self._prompt_context is None:
            return {"prompt": prompt}
        if prompt != self._cached_prompt:
            raise ValueError(
                "WB-WAM task prompt changed after the cached model was loaded. "
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
            raise ValueError("WB-WAM result missing 'action'")
        return self._denormalize_action(pred["action"])

    def infer_video_preview(
        self,
        *,
        image: np.ndarray,
        prompt: str,
        state: np.ndarray | None = None,
        num_video_frames: int | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
    ) -> tuple[list[Any], np.ndarray | None]:
        import torch

        if not callable(getattr(self._model, "infer_joint", None)):
            raise ValueError(f"loaded WB-WAM model does not expose infer_joint(): {type(self._model).__name__}")
        observation: dict[str, Any] = {"observation/image": np.asarray(image)}
        image_arr = compose_wbwam_image(observation, self._input_config)
        image_tensor = (
            torch.from_numpy(image_arr)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=self._model.device,
                dtype=self._model.torch_dtype,
            )
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0

        if state is None:
            state = np.zeros((self._info.state_dim,), dtype=np.float32)
        state_arr = extract_wbwam_state({"states": state}, self._input_config)
        proprio = self._normalize_state(state_arr)
        model_prompt = self._format_prompt(str(prompt))

        kwargs = {
            "input_image": image_tensor,
            "num_video_frames": int(num_video_frames or self._num_video_frames),
            "action_horizon": self._action_horizon,
            "proprio": proprio,
            "negative_prompt": self._negative_prompt,
            "text_cfg_scale": self._text_cfg_scale,
            "num_inference_steps": int(num_inference_steps or self._num_inference_steps),
            "sigma_shift": None if self._sigma_shift is None else float(self._sigma_shift),
            "seed": seed if seed is not None else (None if self._seed is None else int(self._seed)),
            "rand_device": self._rand_device,
            "tiled": self._tiled,
            "test_action_with_infer_action": False,
        }
        kwargs.update(self._prompt_conditioning(model_prompt))
        with torch.no_grad():
            pred = self._model.infer_joint(**kwargs)
        frames = list(pred.get("video") or [])
        if not frames:
            raise ValueError("WB-WAM infer_joint() returned no video frames")
        action = self._denormalize_action(pred["action"]) if "action" in pred else None
        return frames, action

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
