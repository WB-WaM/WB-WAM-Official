"""Deployment prompt embedding cache for low-memory WB-WAM inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class PromptEmbeddingSpec:
    """Everything required to reproduce one deployment text embedding."""

    prompt: str
    prompt_sha256: str
    output_path: Path
    wbwam_root: Path
    model_id: str
    tokenizer_model_id: str
    context_len: int
    cache_suffix: str
    redirect_common_files: bool


def deployment_prompt_embedding_spec(
    deployment_config_path: str | Path,
    task_prompt: str,
) -> PromptEmbeddingSpec:
    """Resolve the exact training-style prompt and its local deployment cache path."""

    config_path = _resolve_path(deployment_config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        deployment_cfg = yaml.safe_load(stream) or {}
    policy_cfg = dict(deployment_cfg.get("policy") or {})

    training_config_path = _resolve_path(policy_cfg["config_path"])
    with training_config_path.open("r", encoding="utf-8") as stream:
        training_cfg = yaml.safe_load(stream) or {}
    model_cfg = dict(training_cfg.get("model") or {})
    data_cfg = dict(training_cfg.get("data") or {})

    template = str(policy_cfg.get("prompt_template") or data_cfg.get("instruction_template") or "{task}")
    fields = {
        "task": str(task_prompt),
        "visual_description": str(policy_cfg.get("visual_description", "")),
        "control_description": str(policy_cfg.get("control_description", "")),
    }
    try:
        prompt = template.format(**fields)
    except KeyError as exc:
        raise ValueError(f"unknown field in policy prompt template: {exc}") from exc

    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    context_len = int(data_cfg.get("context_len", model_cfg.get("tokenizer_max_len", 192)))
    cache_suffix = str(data_cfg.get("text_embedding_cache_suffix") or "wan22ti2v5b")
    checkpoint_path = _resolve_path(policy_cfg["checkpoint_path"])
    cache_dir_value = policy_cfg.get("deployment_text_embedding_cache_dir")
    cache_dir = (
        _resolve_path(cache_dir_value)
        if cache_dir_value is not None
        else checkpoint_path.parent / "deployment_text_embeds"
    )
    output_path = cache_dir / f"{prompt_sha256}.t5_len{context_len}.{cache_suffix}.pt"

    return PromptEmbeddingSpec(
        prompt=prompt,
        prompt_sha256=prompt_sha256,
        output_path=output_path,
        wbwam_root=_resolve_path(policy_cfg.get("wbwam_root", "training")),
        model_id=str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
        tokenizer_model_id=str(model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")),
        context_len=context_len,
        cache_suffix=cache_suffix,
        redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
    )


def _load_cache_payload(path: Path, *, expected: PromptEmbeddingSpec | None = None) -> dict[str, Any]:
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "context" not in payload or "mask" not in payload:
        raise ValueError(f"invalid WB-WAM prompt embedding cache: {path}")
    context = payload["context"]
    mask = payload["mask"]
    if context.ndim != 2 or mask.ndim != 1 or context.shape[0] != mask.shape[0]:
        raise ValueError(f"invalid context/mask shapes in {path}: {tuple(context.shape)} and {tuple(mask.shape)}")
    if expected is not None:
        if context.shape[0] != expected.context_len:
            raise ValueError(
                f"cached context length {context.shape[0]} != expected {expected.context_len}: {path}"
            )
        if payload.get("prompt_sha256") != expected.prompt_sha256:
            raise ValueError(f"cached prompt hash does not match requested prompt: {path}")
        if payload.get("model_id") != expected.model_id:
            raise ValueError(f"cached text model does not match {expected.model_id}: {path}")
        if payload.get("tokenizer_model_id") != expected.tokenizer_model_id:
            raise ValueError(f"cached tokenizer does not match {expected.tokenizer_model_id}: {path}")
    return payload


def load_prompt_embedding(path: str | Path) -> tuple[Any, Any, str]:
    """Load a validated CPU context, normalized mask, and its exact prompt."""

    import torch

    cache_path = _resolve_path(path)
    payload = _load_cache_payload(cache_path)
    context = payload["context"].clone()
    original_mask = payload["mask"].to(dtype=torch.bool)
    context[~original_mask] = 0
    context_mask = torch.ones_like(original_mask)
    prompt = str(payload.get("prompt", ""))
    if not prompt:
        raise ValueError(f"cached prompt text is missing: {cache_path}")
    return context, context_mask, prompt


def generate_prompt_embedding(spec: PromptEmbeddingSpec) -> Path:
    """Load only UMT5, encode one prompt, and atomically save the CPU tensors."""

    import torch

    for path in (spec.wbwam_root, spec.wbwam_root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from wbwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from wbwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=spec.model_id,
        tokenizer_model_id=spec.tokenizer_model_id,
        redirect_common_files=spec.redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    print(f"[TextCache] Encoding deployment prompt with UMT5 only: device={device} context_len={spec.context_len}")
    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=spec.context_len,
        clean="whitespace",
    )
    with torch.no_grad():
        ids, mask = tokenizer([spec.prompt], return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device=device, dtype=torch.bool)
        context = text_encoder(ids, mask)[0].detach().to(device="cpu", dtype=dtype).contiguous()
        mask = mask[0].detach().to(device="cpu", dtype=torch.bool).contiguous()

    payload = {
        "context": context,
        "mask": mask,
        "prompt": spec.prompt,
        "prompt_sha256": spec.prompt_sha256,
        "model_id": spec.model_id,
        "tokenizer_model_id": spec.tokenizer_model_id,
        "context_len": spec.context_len,
        "cache_suffix": spec.cache_suffix,
    }
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = spec.output_path.parent / f".{spec.output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, spec.output_path)
    print(f"[TextCache] Saved: {spec.output_path}")
    return spec.output_path


def ensure_prompt_embedding(
    deployment_config_path: str | Path,
    task_prompt: str,
) -> Path:
    """Create a missing cache in an isolated process so UMT5 memory is released."""

    spec = deployment_prompt_embedding_spec(deployment_config_path, task_prompt)
    if spec.output_path.exists():
        _load_cache_payload(spec.output_path, expected=spec)
        print(f"[TextCache] Reusing cached prompt embedding: {spec.output_path}")
        return spec.output_path

    print("[TextCache] Cache miss; starting isolated UMT5 encoder process...")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(_resolve_path(deployment_config_path)),
            "--prompt",
            str(task_prompt),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    _load_cache_payload(spec.output_path, expected=spec)
    print("[TextCache] Encoder process exited; UMT5 GPU memory has been released.")
    return spec.output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = deployment_prompt_embedding_spec(args.config, args.prompt)
    if spec.output_path.exists():
        _load_cache_payload(spec.output_path, expected=spec)
        print(f"[TextCache] Already cached: {spec.output_path}")
        return 0
    generate_prompt_embedding(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
