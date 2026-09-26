import hashlib
import logging
import os
from pathlib import Path
from typing import Any
import uuid

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from tqdm import tqdm
from wbwam.datasets.wb_archive import resolve_wb_dataset_configs
from wbwam.datasets.wb_dataset import WBDataset
from wbwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from wbwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from wbwam.utils.config_resolvers import register_default_resolvers
from wbwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_BATCH_SIZE = 16


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _collect_dataset_settings(data_cfg: DictConfig):
    if "context_len" not in data_cfg or data_cfg.get("context_len") is None:
        raise KeyError("Top-level `cfg.data.context_len` is required.")
    context_len = int(data_cfg.context_len)

    wb_archives = data_cfg.get("archives")
    wb_datasets = data_cfg.get("datasets")
    if wb_archives is None and wb_datasets is None:
        raise ValueError("Expected WB schema `cfg.data.archives` or `cfg.data.datasets`.")

    archives_list = None if wb_archives is None else OmegaConf.to_container(wb_archives, resolve=True)
    explicit_datasets = None if wb_datasets is None else OmegaConf.to_container(wb_datasets, resolve=True)
    datasets_list = resolve_wb_dataset_configs(
        archives=archives_list,
        datasets=explicit_datasets,
    )

    cache_dir = data_cfg.get("text_embedding_cache_dir")
    if cache_dir is None or not str(cache_dir).strip():
        raise ValueError("WB configs require top-level `data.text_embedding_cache_dir`.")
    cache_dir_path = Path(str(cache_dir)).expanduser()
    prompt_source = {
        "mode": "wb",
        "datasets": datasets_list,
        "cache_dir": cache_dir_path,
        "override_instruction": data_cfg.get("override_instruction"),
        "num_frames": int(data_cfg.get("num_frames", 33)),
        "action_video_freq_ratio": int(data_cfg.get("action_video_freq_ratio", 8)),
        "video_size": list(data_cfg.get("video_size", [224, 224])),
        "camera_key": str(data_cfg.get("camera_key", "observation.images.stereo_left")),
        "state_dim": int(data_cfg.get("state_dim", 110)),
        "action_dim": int(data_cfg.get("action_dim", 136)),
        "raw_state_dim": data_cfg.get("raw_state_dim"),
        "raw_action_dim": data_cfg.get("raw_action_dim"),
        "dataset_format": data_cfg.get("dataset_format", "sealed"),
        "state_key": data_cfg.get("state_key", "observation.state"),
        "action_key": data_cfg.get("action_key", "action"),
        "state_mask_key": data_cfg.get("state_mask_key", "state_mask_110"),
        "action_mask_key": data_cfg.get("action_mask_key", "action_mask_136"),
        "action_representation": data_cfg.get("action_representation", "absolute"),
        "relative_joint_ranges": data_cfg.get("relative_joint_ranges"),
        "relative_action_reference": data_cfg.get("relative_action_reference", "observation"),
        "state_slices": (
            None
            if data_cfg.get("state_slices") is None
            else OmegaConf.to_container(data_cfg.get("state_slices"), resolve=True)
        ),
        "action_slices": (
            None
            if data_cfg.get("action_slices") is None
            else OmegaConf.to_container(data_cfg.get("action_slices"), resolve=True)
        ),
        "context_len": context_len,
        "instruction_template": data_cfg.get("instruction_template"),
        "include_robot_metadata_in_prompt": bool(data_cfg.get("include_robot_metadata_in_prompt", True)),
    }
    logger.info("Discovered WB config with %d dataset roots.", len(datasets_list))
    return [prompt_source], [cache_dir_path], context_len


def _read_unique_wb_prompts(source: dict[str, Any], override_instruction: Any = None) -> list[str]:
    prompts: list[str] = []
    seen = set()
    datasets = source["datasets"]
    logger.info("Reading WB tasks from %d dataset roots...", len(datasets))
    for cfg in datasets:
        cfg = dict(cfg)
        ds_override = (
            override_instruction
            if override_instruction is not None
            else cfg.get("override_instruction", source.get("override_instruction"))
        )
        ds = WBDataset(
            root=str(cfg["root"]),
            name=cfg.get("name"),
            num_frames=int(source["num_frames"]),
            action_video_freq_ratio=int(source["action_video_freq_ratio"]),
            video_size=source["video_size"],
            camera_key=str(cfg.get("camera_key", source["camera_key"])),
            state_dim=int(source["state_dim"]),
            action_dim=int(source["action_dim"]),
            raw_state_dim=cfg.get("raw_state_dim", source.get("raw_state_dim")),
            raw_action_dim=cfg.get("raw_action_dim", source.get("raw_action_dim")),
            dataset_format=cfg.get("dataset_format", source["dataset_format"]),
            state_key=cfg.get("state_key", source["state_key"]),
            action_key=cfg.get("action_key", source["action_key"]),
            state_mask_key=cfg.get("state_mask_key", source["state_mask_key"]),
            action_mask_key=cfg.get("action_mask_key", source["action_mask_key"]),
            action_representation=source["action_representation"],
            relative_joint_ranges=source["relative_joint_ranges"],
            relative_action_reference=source["relative_action_reference"],
            state_slices=cfg.get("state_slices", source.get("state_slices")),
            action_slices=cfg.get("action_slices", source.get("action_slices")),
            exclude_episode_ranges=cfg.get("exclude_episode_ranges"),
            exclude_episode_indices=cfg.get("exclude_episode_indices"),
            val_set_proportion=0.0,
            is_training_set=True,
            use_text_embed_cache=False,
            context_len=int(source["context_len"]),
            instruction_template=source.get("instruction_template"),
            instruction_config=cfg.get("instruction"),
            include_robot_metadata_in_prompt=bool(source["include_robot_metadata_in_prompt"]),
            override_instruction=ds_override,
            shard_cache_size=0,
            strict_video_decode=False,
        )
        for prompt in ds.unique_prompts():
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    logger.info("Loaded %d unique WB prompts.", len(prompts))
    return prompts


def _add_prompt_target(prompt_to_cache_dirs: dict[str, list[Path]], prompt: str, cache_dir: Path) -> None:
    if prompt not in prompt_to_cache_dirs:
        prompt_to_cache_dirs[prompt] = []
    if cache_dir not in prompt_to_cache_dirs[prompt]:
        prompt_to_cache_dirs[prompt].append(cache_dir)


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _cache_suffix_from_cfgs(data_cfg: DictConfig | dict, model_cfg: DictConfig | dict) -> str:
    value = None
    if data_cfg is not None:
        value = data_cfg.get("text_embedding_cache_suffix", None)
    if value is None and model_cfg is not None:
        value = model_cfg.get("text_embedding_cache_suffix", None)
    if value is None or str(value).strip() == "":
        return "wan22ti2v5b"
    return str(value).strip()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone "
            "--nproc_per_node=%d scripts/precompute_text_embeds.py",
            torch.cuda.device_count(),
        )

    overwrite = _to_bool(cfg.get("overwrite", True))
    model_cfg = cfg.model
    if model_cfg is None:
        raise ValueError("`cfg.model` is required.")
    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")

    prompt_sources, cache_dirs, context_len = _collect_dataset_settings(cfg.data)
    if not cache_dirs:
        raise ValueError("No `text_embedding_cache_dir` found under `cfg.data`.")

    prompt_to_cache_dirs: dict[str, list[Path]] = {}
    global_override_instruction = cfg.get("override_instruction")
    has_global_override = (
        global_override_instruction is not None and str(global_override_instruction).strip() != ""
    )
    if has_global_override:
        logger.info(
            "Using override_instruction; encoded prompts for %d prompt sources.",
            len(prompt_sources),
        )
    else:
        logger.info("Scanning WB dataset metadata for prompts...")

    for source in prompt_sources:
        override = global_override_instruction if has_global_override else None
        for prompt in _read_unique_wb_prompts(source, override_instruction=override):
            _add_prompt_target(prompt_to_cache_dirs, prompt, source["cache_dir"])
    prompts = list(prompt_to_cache_dirs.keys())
    if not prompts:
        logger.warning("No prompts found from WB dataset metadata; nothing to do.")
        return

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    cache_suffix = _cache_suffix_from_cfgs(cfg.data, model_cfg)

    logger.info(
        "Preparing text encoder with model_id=%s tokenizer_model_id=%s "
        "device=%s dtype=%s context_len=%d overwrite=%s",
        model_id,
        tokenizer_model_id,
        device,
        torch_dtype,
        context_len,
        overwrite,
    )

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    stats = {str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0} for cache_dir in cache_dirs}

    prompts = prompts[rank::world_size] if is_distributed else prompts

    if not overwrite:
        fully_cached_local = 0
        prompts_to_encode: list[str] = []
        for prompt in prompts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{cache_suffix}.pt"
            fully_cached = True
            for cache_dir in prompt_to_cache_dirs[prompt]:
                cache_path = cache_dir / filename
                if not cache_path.exists():
                    fully_cached = False
                    break
            if fully_cached:
                fully_cached_local += 1
                for cache_dir in prompt_to_cache_dirs[prompt]:
                    stats[str(cache_dir)]["skip"] += 1
            else:
                prompts_to_encode.append(prompt)

        prompts = prompts_to_encode

        fully_cached_global = fully_cached_local
        to_encode_global = len(prompts)
        if is_distributed:
            reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
            count_tensor = torch.tensor([fully_cached_local, len(prompts)], device=reduce_device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            fully_cached_global = int(count_tensor[0].item())
            to_encode_global = int(count_tensor[1].item())

        if (not is_distributed) or rank == 0:
            logger.info(
                "overwrite=false: fully cached prompts=%d, prompts to encode=%d",
                fully_cached_global,
                to_encode_global,
            )

    logger.info("Writing caches to %d directories.", len(cache_dirs))
    prompts_encoded_local = len(prompts)
    prompts_encoded_global = prompts_encoded_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([prompts_encoded_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor.item())

    over_length_prompts = 0
    with tqdm(
        total=len(prompts),
        desc=f"Encoding prompts (rank {rank}/{world_size})" if is_distributed else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                over_length_prompts += int(mask.all(dim=1).sum().item())
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    payload = {
                        "context": context_i,
                        "mask": mask_i,
                    }

                    for cache_dir in prompt_to_cache_dirs[prompt]:
                        cache_path = cache_dir / f"{hashed}.t5_len{context_len}.{cache_suffix}.pt"
                        key = str(cache_dir)
                        if cache_path.exists() and not overwrite:
                            stats[key]["skip"] += 1
                            continue

                        if cache_path.exists():
                            stats[key]["overwrite"] += 1
                        else:
                            stats[key]["new"] += 1

                        _atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    over_length_global = over_length_prompts
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_prompts], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length_global = int(over_tensor.item())

        counts_tensor = torch.tensor(
            [
                [stats[str(cache_dir)]["new"], stats[str(cache_dir)]["overwrite"], stats[str(cache_dir)]["skip"]]
                for cache_dir in cache_dirs
            ],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            for idx, cache_dir in enumerate(cache_dirs):
                key = str(cache_dir)
                stats[key]["new"] = int(counts_tensor[idx, 0].item())
                stats[key]["overwrite"] = int(counts_tensor[idx, 1].item())
                stats[key]["skip"] = int(counts_tensor[idx, 2].item())

    if (not is_distributed) or rank == 0:
        logger.info("Finished precomputing text embeddings.")
        logger.info(
            "Over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
            context_len,
            over_length_global,
            prompts_encoded_global,
        )
        for cache_dir in cache_dirs:
            key = str(cache_dir)
            logger.info(
                "Cache dir: %s | new=%d overwrite=%d skip=%d",
                key,
                stats[key]["new"],
                stats[key]["overwrite"],
                stats[key]["skip"],
            )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
