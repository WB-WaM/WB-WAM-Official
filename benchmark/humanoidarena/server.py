"""Independent WB-WAM Native50 HTTP server; no bridge runtime dependency.

The benchmark-local adapter loads the repository's WB-WAM implementation.
The simulator owns all semantic40 -> SONIC conversion.
"""

from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import numpy as np
import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    # The 5090 checkpoint wrapper loads this file through importlib instead of
    # executing it as a script, so Python does not add this directory for us.
    sys.path.insert(0, str(BENCHMARK_ROOT))


def local_runtime_config(config_path, destination_dir=None):
    """Add an explicit model-loader setting without overwriting training config.

    Keep the sidecar in metadata so the frozen adapter's same-run checks remain
    enabled for config, statistics and weights.
    """
    config = yaml.safe_load(config_path.read_text())
    runtime = copy.deepcopy(config)
    runtime["model"]["redirect_common_files"] = False
    content = yaml.safe_dump(runtime, sort_keys=False)
    name = "native50_local_" + hashlib.sha256(content.encode()).hexdigest()[:16] + ".yaml"
    path = Path(destination_dir or config_path.parent) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_text() != content:
                    raise ValueError(f"Conflicting runtime config: {path}")
        finally:
            temporary.unlink()
    return path


def local_prompt_spec(deployment, prompt):
    from runtime.text_cache import deployment_prompt_embedding_spec

    # Keep the frozen model source immutable and make the deployment's
    # original-pth contract explicit here.
    return replace(deployment_prompt_embedding_spec(deployment, prompt), redirect_common_files=False)


def validate_base_files(spec):
    from wbwam.models.wan22.helpers.loader import _resolve_configs

    _, text, vae, tokenizer = _resolve_configs(spec.model_id, spec.tokenizer_model_id, redirect_common_files=False)
    resolved = {}
    for name, config in (("text_encoder", text), ("vae", vae), ("tokenizer", tokenizer)):
        config.skip_download = True
        config.download_if_necessary()
        paths = config.path if isinstance(config.path, list) else [config.path]
        if not paths or any(not p or not Path(p).exists() for p in paths):
            raise FileNotFoundError(f"Missing local {name}: {config.model_id}/{config.origin_file_pattern}")
        resolved[name] = paths
    print(json.dumps({"resolved_base_models": resolved}), flush=True)
    return resolved


def validate_contract(config, stats):
    data, meta = config["data"], stats["metadata"]
    for key, expected in {
        "state_dim": 64,
        "action_dim": 40,
        "proprio_dim": 96,
        "action_target_dim": 96,
        "action_representation": "native",
    }.items():
        if data.get(key) != expected or meta.get(key) != expected:
            raise ValueError(f"Native50 contract mismatch: {key}")
    if meta.get("dataset_fps") != [50]:
        raise ValueError("Expected task-local 50Hz statistics")
    for key, size in (("state_model_dim_indices", 64), ("action_model_dim_indices", 40)):
        indices = data[key]
        if indices != meta[key] or len(indices) != size or len(set(indices)) != size:
            raise ValueError(f"Invalid/mismatched {key}")
        if any(not isinstance(i, int) or not 0 <= i < 96 for i in indices):
            raise ValueError(f"Invalid slot in {key}")
    if data.get("relative_joint_ranges") or data.get("model_dim_indices") is not None:
        raise ValueError("Native50 must preserve native values and independent maps")
    if data.get("camera_key") != "observation.images.front":
        raise ValueError("Expected front camera")
    if data["num_frames"] != 33 or data["action_video_freq_ratio"] != 4:
        raise ValueError("This deployment expects 32 action rows and 9 video frames")


def resolve_prompt(config, instruction_key, expected_prompt):
    instruction = config["data"]["datasets"][0]["instruction"]
    try:
        prompt = instruction["task_overrides"][instruction_key]
    except KeyError as error:
        raise ValueError(f"Checkpoint has no prompt for instruction key {instruction_key}") from error
    if prompt != expected_prompt:
        raise ValueError(
            f"Checkpoint prompt mismatch for instruction key {instruction_key}: {prompt!r} != {expected_prompt!r}"
        )
    return prompt, instruction


def decode_image(observation):
    item = observation["images"]["front"]
    shape = tuple(item["shape"])
    if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0 or item["dtype"] != "uint8":
        raise ValueError("Expected uint8 HWC RGB front image")
    raw = base64.b64decode(item.get("data_b64", item.get("data")), validate=True)
    if len(raw) != int(np.prod(shape)):
        raise ValueError("Image byte count mismatch")
    return np.frombuffer(raw, np.uint8).reshape(shape).copy()


class Service:
    def __init__(self, policy, prompt, health, capture_dir, replan=14, record_captures=True):
        if not 1 <= replan <= 32:
            raise ValueError("replan must be within 1..32")
        self.policy, self.prompt, self.health = policy, prompt, health
        self.capture_dir, self.replan = Path(capture_dir), replan
        self.record_captures = record_captures
        if self.record_captures:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.count = 0
        self.episode_seed = None

    def reset(self, seed=None):
        with self.lock:
            self.episode_seed = seed
            self.policy._seed = seed
        return {"status": "ok", "seed": seed}

    def infer(self, payload):
        obs = payload["observation"]
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape != (64,) or not np.isfinite(state).all():
            raise ValueError("Expected finite official state[64], without legacy hand injection")
        image = decode_image(obs)
        with self.lock:
            action = np.asarray(
                self.policy.infer({"states": state, "observation/image": image, "prompt": self.prompt}),
                dtype=np.float32,
            )
            if action.shape != (32, 40) or not np.isfinite(action).all():
                raise ValueError(f"Expected finite native action[32,40], got {action.shape}")
            # Preserve raw hand values: thresholding belongs to official runtime.
            if self.record_captures:
                np.savez_compressed(
                    self.capture_dir / f"inference_{self.count:06d}.npz",
                    state=state,
                    action=action,
                    rgb=image,
                    normalized_model_action=getattr(self.policy, "last_normalized_action", np.empty(0)),
                    episode_seed=-1 if self.episode_seed is None else self.episode_seed,
                )
            self.count += 1
            print(
                json.dumps(
                    {
                        "inference": self.count,
                        "episode_seed": self.episode_seed,
                        "hand_min": action[:, 38:40].min(axis=0).tolist(),
                        "hand_max": action[:, 38:40].max(axis=0).tolist(),
                    }
                ),
                flush=True,
            )
            return {"action_chunk": action[: self.replan].tolist()}


def make_server(service, port):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(
                200 if self.path == "/health" else 404,
                {"status": "ok", **service.health} if self.path == "/health" else {},
            )

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 16 << 20:
                    raise ValueError("Invalid body size")
                data = json.loads(self.rfile.read(size))
                if self.path == "/reset":
                    result = service.reset(data.get("seed"))
                elif self.path == "/infer":
                    result = service.infer(data)
                else:
                    self.reply(404, {})
                    return
                self.reply(200, result)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                self.reply(500, {"error": str(exc)})

    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--weight-file", default="step_019150.pt")
    parser.add_argument("--instruction-key", default="0")
    parser.add_argument("--expected-prompt", required=True)
    parser.add_argument("--wbwam-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--replan", type=int, default=14)
    parser.add_argument("--denoise-steps", type=int, default=20)
    parser.add_argument("--inference-mode", choices=("idm", "wbwam", "ff"), default="idm")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--record-inference-captures", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.replan <= 32:
        raise ValueError("replan must be within 1..32")
    if not 1 <= args.denoise_steps <= 100:
        raise ValueError("denoise_steps must be within 1..100")
    inference_mode = "wbwam" if args.inference_mode == "ff" else args.inference_mode
    root = args.checkpoint_root.resolve()
    if (root / "config.yaml").is_file():
        config_path, stats_path, weights_dir = (
            root / "config.yaml",
            root / "dataset_stats.json",
            root,
        )
    else:
        config_path, stats_path, weights_dir = (
            root / "metadata/config.yaml",
            root / "metadata/dataset_stats.json",
            root / "weights",
        )
    config = yaml.safe_load(config_path.read_text())
    stats = json.loads(stats_path.read_text())
    validate_contract(config, stats)
    args.output.mkdir(parents=True, exist_ok=True)
    runtime_config_path = local_runtime_config(config_path, args.output)
    weights = weights_dir / args.weight_file
    if not weights.is_file() or weights.stat().st_size < 1_000_000:
        raise ValueError("Missing/empty native50 checkpoint")
    prompt, instruction = resolve_prompt(config, args.instruction_key, args.expected_prompt)
    wbwam_root = args.wbwam_root.resolve()
    from runtime import wbwam_adapter as adapter_module

    if not Path(adapter_module.__file__).resolve().is_relative_to(BENCHMARK_ROOT):
        raise RuntimeError("Wrong Native50 adapter import: expected benchmark-local runtime")
    policy_config = dict(
        config_path=str(runtime_config_path),
        artifact_config_path=str(config_path),
        checkpoint_path=str(weights),
        dataset_stats_path=str(stats_path),
        wbwam_root=str(wbwam_root),
        inference_mode=inference_mode,
        device="cuda:0",
        mixed_precision="bf16",
        skip_dit_load_from_pretrain=True,
        action_dit_pretrained_path=None,
        redirect_common_files=False,
        state_dim=64,
        action_dim=40,
        action_horizon=32,
        num_video_frames=9,
        num_inference_steps=args.denoise_steps,
        seed=args.seed,
        strict_checkpoint_load=True,
        visual_description=instruction["visual_description"],
        control_description=instruction["control_description"],
        deployment_text_embedding_cache_dir=str(args.output / "text_cache"),
    )
    deployment = args.output / "deployment.json"
    deployment.write_text(json.dumps({"policy": policy_config}, indent=2))
    (args.output / "runtime_config.yaml").write_text(runtime_config_path.read_text())
    # Resolve through the SAME loader before attempting any GPU weight load.
    sys.path.insert(0, str(wbwam_root / "src"))
    prompt_spec = local_prompt_spec(deployment, prompt)
    resolved = validate_base_files(prompt_spec)
    (args.output / "resolved_base_models.json").write_text(json.dumps(resolved, indent=2))
    if args.prepare_only:
        return
    from runtime.text_cache import generate_prompt_embedding, load_prompt_embedding

    if args.cache_only:
        generate_prompt_embedding(prompt_spec)
        return
    embedding = prompt_spec.output_path
    if not embedding.exists():
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--checkpoint-root",
                str(root),
                "--wbwam-root",
                str(wbwam_root),
                "--output",
                str(args.output),
                "--seed",
                str(args.seed),
                "--weight-file",
                args.weight_file,
                "--instruction-key",
                args.instruction_key,
                "--expected-prompt",
                args.expected_prompt,
                "--replan",
                str(args.replan),
                "--denoise-steps",
                str(args.denoise_steps),
                "--inference-mode",
                args.inference_mode,
                "--port",
                str(args.port),
                "--cache-only",
            ],
            check=True,
        )
    _, _, cached_prompt = load_prompt_embedding(embedding)
    if cached_prompt != prompt_spec.prompt:
        raise ValueError("Cached prompt does not match training instruction")
    policy_config["prompt_embedding_path"] = str(embedding)

    class CapturedAdapter(adapter_module.WBWAMAdapter):
        def _denormalize_action(self, action):
            self.last_normalized_action = action.detach().float().cpu().numpy().copy()
            return super()._denormalize_action(action)

    policy = CapturedAdapter(
        policy_config=policy_config,
        input_config={
            "state_layout": None,
            "image_mode": "rgb",
            "base_state_dim": 64,
            "expected_state_dim": 64,
            "rgb_size": [224, 224],
        },
    )
    health = dict(
        deployment_variant="native50_semantic40",
        action_format="semantic_v3",
        adapter_source="benchmark_local_native50_runtime",
        checkpoint=str(weights),
        config=str(config_path),
        stats=str(stats_path),
        state_dim=64,
        action_dim=40,
        runtime_config=str(runtime_config_path),
        action_source_hz=50,
        execute_chunk_steps=args.replan,
        num_inference_steps=args.denoise_steps,
        inference_mode=inference_mode,
        action_horizon=32,
        reference_window="official_latest_hold",
        instruction_key=args.instruction_key,
        instruction_prompt=prompt,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    )
    server = make_server(
        Service(
            policy,
            prompt,
            health,
            args.output / "capture",
            replan=args.replan,
            record_captures=args.record_inference_captures,
        ),
        args.port,
    )
    health["port"] = server.server_address[1]
    (args.output / "server_health.json").write_text(json.dumps(health, indent=2))
    print(f"Native50 ready port={health['port']}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
