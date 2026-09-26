"""CUDA synchronization and peak-memory reporting for deployment dry runs."""

from __future__ import annotations

from typing import Any

from bridge.common.policy import PolicyAdapter, PolicyInfo


class _SynchronizedCudaAdapter(PolicyAdapter):
    def __init__(self, adapter: PolicyAdapter, torch_module: Any, device: int) -> None:
        self._adapter = adapter
        self._torch = torch_module
        self._device = device

    def infer(self, observation: dict[str, Any], **kwargs: Any):
        self._torch.cuda.synchronize(self._device)
        actions = self._adapter.infer(observation, **kwargs)
        self._torch.cuda.synchronize(self._device)
        return actions

    @property
    def info(self) -> PolicyInfo:
        return self._adapter.info


def prepare_cuda_benchmark(adapter: PolicyAdapter) -> tuple[PolicyAdapter, tuple[Any, int] | None]:
    """Synchronize timed calls and reset peak memory when CUDA is available."""

    try:
        import torch
    except ImportError:
        return adapter, None
    if not torch.cuda.is_available():
        return adapter, None
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return _SynchronizedCudaAdapter(adapter, torch, device), (torch, device)


def report_cuda_peak_memory(handle: tuple[Any, int] | None) -> None:
    if handle is None:
        return
    torch, device = handle
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    print(f"  cuda_peak:    allocated={allocated:.3f} GiB reserved={reserved:.3f} GiB")


__all__ = ["prepare_cuda_benchmark", "report_cuda_peak_memory"]
