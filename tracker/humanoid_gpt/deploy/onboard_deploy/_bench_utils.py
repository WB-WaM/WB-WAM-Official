"""Shared profiling helpers for the on-board benchmark scripts.

Used by:
- bench_onnx_infer.py  (only ONNX inference)
- bench_offline.py     (full pipeline with fake sensors, no DDS)
- bench_online.py      (full pipeline with real DDS sensor + optional publish)
"""

from __future__ import annotations

import importlib
import platform
import sys
import time
from collections import defaultdict

import numpy as np


class BenchProfiler:
    """Accumulate per-segment latencies and print percentile summaries.

    Usage::

        prof = BenchProfiler()
        for _ in range(N):
            with prof.time("01_sensor"):
                root_quat = ...
            with prof.time("02_infer"):
                action = policy.infer(obs)
        print(prof.summary("Headline"))
    """

    def __init__(self):
        self.timings: dict[str, list[float]] = defaultdict(list)

    def time(self, name: str) -> "_ScopedTimer":
        return _ScopedTimer(self, name)

    def record(self, name: str, ms: float) -> None:
        self.timings[name].append(ms)

    def reset(self) -> None:
        self.timings.clear()

    def summary(self, header: str = "", iters: int | None = None) -> str:
        if not self.timings:
            return "(no timings recorded)"
        name_w = max(max(len(k) for k in self.timings), 18)
        lines: list[str] = []
        if header:
            lines.append(f"\n{'=' * 88}")
            lines.append(f"  {header}")
            if iters is not None:
                lines.append(f"  iters: {iters}")
            lines.append(f"{'=' * 88}")
        lines.append(
            f"  {'segment':<{name_w}}  {'mean':>8}  {'p50':>8}  "
            f"{'p90':>8}  {'p99':>8}  {'max':>8}  {'std':>8}"
        )
        lines.append(
            f"  {'-' * name_w}  {'-' * 8}  {'-' * 8}  {'-' * 8}  "
            f"{'-' * 8}  {'-' * 8}  {'-' * 8}"
        )
        for name in sorted(self.timings.keys()):
            arr = np.asarray(self.timings[name])
            lines.append(
                f"  {name:<{name_w}}  "
                f"{arr.mean():>7.3f}  "
                f"{np.percentile(arr, 50):>7.3f}  "
                f"{np.percentile(arr, 90):>7.3f}  "
                f"{np.percentile(arr, 99):>7.3f}  "
                f"{arr.max():>7.3f}  "
                f"{arr.std():>7.3f}"
            )
        lines.append(f"{'=' * 88}")
        return "\n".join(lines)


class _ScopedTimer:
    __slots__ = ("profiler", "name", "_t0")

    def __init__(self, profiler: BenchProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self._t0 = 0.0

    def __enter__(self) -> "_ScopedTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.profiler.record(self.name, (time.perf_counter() - self._t0) * 1e3)


def print_environment_info(extra_packages: list[str] | None = None) -> None:
    """Print Python / numpy / scipy / mujoco / ORT / cyclonedds versions.

    Helps comparing benchmarks across different conda environments.
    """
    pkgs = ["numpy", "scipy", "mujoco", "onnxruntime", "cyclonedds"]
    if extra_packages:
        pkgs.extend(extra_packages)
    print(f"\n{'=' * 88}")
    print("  Environment")
    print(f"{'=' * 88}")
    print(f"  Python  : {sys.version.split()[0]} ({platform.machine()})")
    print(f"  Platform: {platform.system()} {platform.release()}")
    for pkg in pkgs:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg:<11}: {ver}")
        except ImportError:
            print(f"  {pkg:<11}: (not installed)")
    print(f"{'=' * 88}")


def aarch64_preload() -> None:
    """Preload aarch64 libraries that ORT/torch need on Jetson."""
    if platform.machine() != "aarch64":
        return
    import ctypes
    import site
    from pathlib import Path

    for _lib in ["/lib/aarch64-linux-gnu/libGLdispatch.so.0"]:
        try:
            ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    for _sp in site.getsitepackages():
        for _p in Path(_sp).glob("torch.libs/libgomp-*.so*"):
            try:
                ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break


def print_budget_summary(
    step_total_ms: np.ndarray,
    target_freq_hz: int = 50,
) -> None:
    """Pretty-print throughput / margin / deadline-miss statistics."""
    budget_ms = 1000.0 / target_freq_hz
    over = step_total_ms > budget_ms
    print(
        f"  Throughput (mean):           "
        f"{1000.0 / step_total_ms.mean():.0f} Hz"
    )
    print(
        f"  Margin to {target_freq_hz}Hz at mean:      "
        f"{budget_ms - step_total_ms.mean():+.2f} ms"
    )
    print(
        f"  Margin to {target_freq_hz}Hz at p99:       "
        f"{budget_ms - np.percentile(step_total_ms, 99):+.2f} ms"
    )
    if over.any():
        n = int(over.sum())
        print(
            f"  Deadline misses:             "
            f"{n}/{len(step_total_ms)} ({100 * n / len(step_total_ms):.1f}%) "
            f"steps exceeded {budget_ms:.1f} ms"
        )
    else:
        print(
            f"  Deadline misses:             0/{len(step_total_ms)}  "
            f"(all steps under {budget_ms:.1f} ms)"
        )
    print()
