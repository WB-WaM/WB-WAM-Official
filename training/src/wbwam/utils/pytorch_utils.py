from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Callable, Optional
import os
import random
import collections

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
import numpy as np


@dataclass(frozen=True)
class DistributedTopology:
    world_size: int
    rank: int
    local_world_size: int
    local_rank: int
    num_nodes: int
    node_rank: int


def resolve_distributed_topology() -> DistributedTopology:
    """Resolve global/node topology from torch.distributed state and launcher env."""
    if dist.is_available() and dist.is_initialized():
        world_size = int(dist.get_world_size())
        rank = int(dist.get_rank())
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))

    if world_size <= 1:
        local_world_size = 1
        local_rank = 0
    else:
        default_local_world_size = (
            torch.cuda.device_count()
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else world_size
        )
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(default_local_world_size)))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank % local_world_size)))

    if local_world_size <= 0:
        raise ValueError(f"`local_world_size` must be > 0, got {local_world_size}.")
    if world_size % local_world_size != 0:
        raise ValueError(
            f"`WORLD_SIZE` ({world_size}) must be divisible by `LOCAL_WORLD_SIZE` ({local_world_size})."
        )
    if local_rank < 0 or local_rank >= local_world_size:
        raise ValueError(
            f"`LOCAL_RANK` must be in [0, {local_world_size}), got {local_rank}."
        )

    return DistributedTopology(
        world_size=world_size,
        rank=rank,
        local_world_size=local_world_size,
        local_rank=local_rank,
        num_nodes=world_size // local_world_size,
        node_rank=rank // local_world_size,
    )


def ensure_distributed_initialized(topology: DistributedTopology) -> None:
    """Initialize torch.distributed before dataset construction when sharding needs broadcast."""
    if topology.world_size <= 1:
        return
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training with NCCL.")
    torch.cuda.set_device(topology.local_rank)
    backend = "nccl"

    timeout_seconds = int(os.environ.get("TORCH_DIST_TIMEOUT_SECONDS", "7200"))
    timeout = timedelta(seconds=timeout_seconds)
    # torch.distributed.new_group(timeout=None) does not inherit the default
    # process group's timeout. Override c10d defaults before DeepSpeed creates
    # any subgroups without an explicit timeout.
    dist_c10d.default_pg_timeout = timeout
    _ = dist_c10d.default_pg_nccl_timeout
    dist_c10d.default_pg_nccl_timeout = timeout
    if dist.is_available() and dist.is_initialized():
        print(
            "[dist_init] process group already initialized "
            f"world_size={dist.get_world_size()} rank={dist.get_rank()} "
            f"default_timeout_seconds={timeout_seconds}",
            flush=True,
        )
        return

    print(
        "[dist_init] initializing process group "
        f"backend={backend} world_size={topology.world_size} rank={topology.rank} "
        f"local_rank={topology.local_rank} timeout_seconds={timeout_seconds}",
        flush=True,
    )
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timeout,
    )


def _resolve_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0"))))


def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)

    # Process-specific seeding: offset by global rank so each process gets a different seed
    global_rank = _resolve_global_rank()
    process_seed = seed + global_rank

    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)

    return worker_init_function if get_worker_init_fn else None


def worker_init_function(worker_id: int) -> None:
    """
    Borrowed directly from PyTorch-Lightning; inspired by this issue comment in the PyTorch repo:
        > Ref: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    Intuition: You can think of the seed sequence spawn function as a "janky" torch.Generator() or jax.PRNGKey that
    you can run iterative splitting on to get new (predictable) randomness.

    :param worker_id: Identifier for the given worker [0, num_workers) for the Dataloader in question.
    """
    # Get current global `rank` (if running distributed) and `process_seed`
    process_seed = torch.initial_seed()
    global_rank = _resolve_global_rank()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch:
    #   > https://pytorch.org/docs/stable/data.html#data-loading-randomness
    base_seed = process_seed - worker_id

    # "Magic" code --> basically creates a seed sequence that mixes different "sources" and seeds every library...
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed --> generate_state(k) produces a `k` element array!
    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq, tf_seed_seq = seed_seq.spawn(3)

    # Torch Manual seed takes 64 bits (so just specify a dtype of uint64
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)


def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def dict_to_array(x):
    data = np.concatenate([item for _, item in x.items()], axis=-1)
    return data


def pad_remaining_dims(x, target):
    assert x.shape == target.shape[:len(x.shape)]
    return x.reshape(x.shape + (1,)*(len(target.shape) - len(x.shape)))


def dict_apply_split(
        x: Dict[str, torch.Tensor], 
        split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results


def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer

def is_rank0() -> bool:
    """
    Best-effort check for main process without any synchronization.
    """
    # Prefer torch.distributed state if initialized.
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    # Fallback to environment variables commonly set by launchers.
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ.get(key, "0") in ("0", "0\n", "")

    return True
