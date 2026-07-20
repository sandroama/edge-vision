"""Latency measurement harness — the same numbers, regardless of backend.

The harness exists for one reason: every later phase's headline number
(speed-up, FPS, throughput, watts/frame) is built on top of these
percentiles, so they have to be measured the same way every time.

Two backends:

    1. **CPU / "auto"** — wraps ``time.perf_counter`` around the callable.
       Suitable for ONNX Runtime CPU, PyTorch CPU eager, anything that
       blocks the host on completion.

    2. **CUDA** — uses ``torch.cuda.Event(enable_timing=True)`` so the
       reported number is the *kernel* time, not the host-launch time.
       The harness inserts a synchronization barrier after each iteration
       so the events are well-ordered. Lazy-imports torch.

Both paths warm up before measuring (default 10 iterations) — first calls
incur graph compilation, kernel autotune, allocator warmup, etc. Reporting
those would lie about the steady-state.

Output: a ``LatencyResult`` dataclass with p50/p95/p99 + mean/std + FPS.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LatencyResult:
    """Standardised latency report."""

    n: int
    n_warmup: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    fps: float
    backend: str          # "cpu" | "cuda"
    device: str           # "cpu" | "cuda:0" | etc.

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "backend": self.backend,
            "device": self.device,
            "n": self.n,
            "n_warmup": self.n_warmup,
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "mean_ms": round(self.mean_ms, 4),
            "std_ms": round(self.std_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "fps": round(self.fps, 2),
        }

    def as_row(self) -> str:
        """Single-line summary for log scans."""
        return (
            f"[{self.backend}/{self.device}] "
            f"p50={self.p50_ms:.2f}ms  p95={self.p95_ms:.2f}ms  "
            f"p99={self.p99_ms:.2f}ms  fps={self.fps:.1f}  (n={self.n})"
        )


def _percentile(samples: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not samples:
        return 0.0
    return float(np.percentile(samples, q))


def _summarize(samples_ms: list[float], *, n_warmup: int, backend: str, device: str) -> LatencyResult:
    if not samples_ms:
        return LatencyResult(
            n=0,
            n_warmup=n_warmup,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            mean_ms=0.0,
            std_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            fps=0.0,
            backend=backend,
            device=device,
        )

    mean_ms = float(statistics.fmean(samples_ms))
    std_ms = float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0
    fps = 1000.0 / mean_ms if mean_ms > 0 else math.inf

    return LatencyResult(
        n=len(samples_ms),
        n_warmup=n_warmup,
        p50_ms=_percentile(samples_ms, 50),
        p95_ms=_percentile(samples_ms, 95),
        p99_ms=_percentile(samples_ms, 99),
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
        fps=fps,
        backend=backend,
        device=device,
    )


# --------------------------------------------------------------------------- CPU


def measure_latency_cpu(
    fn: Callable[[], Any],
    *,
    n_runs: int = 100,
    n_warmup: int = 10,
    device: str = "cpu",
) -> LatencyResult:
    """Time ``fn()`` with ``time.perf_counter``.

    The callable should produce a single inference / forward pass — pre/post
    processing inside it is fine, but anything async (e.g. CUDA kernels that
    return immediately) will *not* be timed correctly here. Use
    :func:`measure_latency_cuda` for that.
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0")
    if n_warmup < 0:
        raise ValueError("n_warmup must be >= 0")

    for _ in range(n_warmup):
        fn()

    samples_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1e3
        samples_ms.append(elapsed)

    return _summarize(samples_ms, n_warmup=n_warmup, backend="cpu", device=device)


# --------------------------------------------------------------------------- CUDA


def measure_latency_cuda(
    fn: Callable[[], Any],
    *,
    n_runs: int = 100,
    n_warmup: int = 10,
    device: str = "cuda:0",
) -> LatencyResult:
    """Time ``fn()`` with CUDA events for accurate kernel-time measurement.

    Lazy-imports torch. Raises ImportError if torch isn't installed. Each
    sample includes a ``torch.cuda.synchronize()`` barrier after the end
    event so that overlapped streams don't bleed into the next sample.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "measure_latency_cuda requires PyTorch with CUDA. "
            "Install with: pip install -e '.[dev,gpu]'"
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "measure_latency_cuda called but torch.cuda.is_available() is False. "
            "Use measure_latency_cpu instead."
        )

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(n_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))

    return _summarize(samples_ms, n_warmup=n_warmup, backend="cuda", device=device)


# --------------------------------------------------------------------------- public dispatcher


def measure_latency(
    fn: Callable[[], Any],
    *,
    n_runs: int = 100,
    n_warmup: int = 10,
    backend: str = "auto",
    device: str | None = None,
) -> LatencyResult:
    """Top-level entry — picks the right measurement backend.

    Args:
        fn: zero-arg callable that runs one forward pass.
        n_runs: timed iterations to sample (excluding warmup).
        n_warmup: warmup iterations (not measured).
        backend: ``"auto"`` (CUDA if available, else CPU) | ``"cpu"`` | ``"cuda"``.
        device: optional human-readable device label for the report.
    """
    if backend == "cpu":
        return measure_latency_cpu(
            fn, n_runs=n_runs, n_warmup=n_warmup, device=device or "cpu"
        )
    if backend == "cuda":
        return measure_latency_cuda(
            fn, n_runs=n_runs, n_warmup=n_warmup, device=device or "cuda:0"
        )
    if backend == "auto":
        # Try CUDA, fall back to CPU silently.
        try:
            import torch

            if torch.cuda.is_available():
                return measure_latency_cuda(
                    fn,
                    n_runs=n_runs,
                    n_warmup=n_warmup,
                    device=device or "cuda:0",
                )
        except ImportError:
            pass
        return measure_latency_cpu(
            fn, n_runs=n_runs, n_warmup=n_warmup, device=device or "cpu"
        )
    raise ValueError(f"Unknown backend: {backend!r} (use 'auto', 'cpu', or 'cuda')")
