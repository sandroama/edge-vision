"""CPU + RAM profiling for ONNX Runtime CPU inference paths.

The GPU power monitor covers the RTX-5080 backends. For the ONNX-CPU path
we need a parallel story: CPU utilisation and RSS growth over time. This
module uses psutil — available everywhere, no GPU required.

The result feeds directly into the Pareto aggregator as the "ONNX-CPU"
entry: the watts/frame column for CPU is estimated from the CPU socket TDP
scaled by utilisation.

Usage::

    with cpu_profiler(sample_ms=100) as profiler:
        run_inference_for_n_seconds(30)
    profile = profiler.summarise()
    print(profile.as_row())
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


@dataclass
class CpuProfile:
    """Summary of a CPU / RAM monitoring session."""

    duration_s: float
    sample_interval_ms: float
    n_samples: int
    mean_cpu_pct: float    # averaged across all logical cores, 0–100
    max_cpu_pct: float
    mean_rss_mb: float     # resident set size (process memory)
    max_rss_mb: float
    # Estimated CPU watt draw. cpu_tdp_w must be set externally via the
    # CpuProfiler constructor. Default 65W is a rough 9950X TDP @ load.
    estimated_power_w: float | None

    def as_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 2),
            "sample_interval_ms": self.sample_interval_ms,
            "n_samples": self.n_samples,
            "mean_cpu_pct": round(self.mean_cpu_pct, 1),
            "max_cpu_pct": round(self.max_cpu_pct, 1),
            "mean_rss_mb": round(self.mean_rss_mb, 1),
            "max_rss_mb": round(self.max_rss_mb, 1),
            "estimated_power_w": round(self.estimated_power_w, 2)
            if self.estimated_power_w is not None
            else None,
        }

    def as_row(self) -> str:
        pw = f"{self.estimated_power_w:.1f}W" if self.estimated_power_w is not None else "n/a"
        return (
            f"[CPU] "
            f"cpu_mean={self.mean_cpu_pct:.1f}%  max={self.max_cpu_pct:.1f}%  "
            f"rss_mean={self.mean_rss_mb:.0f}MB  est_power={pw}  n={self.n_samples}"
        )


class CpuProfiler:
    """Background-thread psutil sampler."""

    def __init__(
        self,
        sample_ms: float = 100.0,
        cpu_tdp_w: float = 65.0,
    ) -> None:
        self.sample_ms = sample_ms
        self.cpu_tdp_w = cpu_tdp_w
        self._cpu_pcts: list[float] = []
        self._rss_mbs: list[float] = []
        self._ts: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil  # noqa: F401 — validates presence
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "CpuProfiler requires psutil. Install with: pip install -e '.[dev]'"
            ) from e
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def summarise(self) -> CpuProfile:
        if not self._cpu_pcts:
            return CpuProfile(
                duration_s=0.0,
                sample_interval_ms=self.sample_ms,
                n_samples=0,
                mean_cpu_pct=0.0,
                max_cpu_pct=0.0,
                mean_rss_mb=0.0,
                max_rss_mb=0.0,
                estimated_power_w=None,
            )
        cpu = np.array(self._cpu_pcts)
        rss = np.array(self._rss_mbs)
        duration = self._ts[-1] - self._ts[0] if len(self._ts) > 1 else 0.0
        mean_util = float(cpu.mean())
        est_power = self.cpu_tdp_w * (mean_util / 100.0)

        return CpuProfile(
            duration_s=duration,
            sample_interval_ms=self.sample_ms,
            n_samples=len(cpu),
            mean_cpu_pct=mean_util,
            max_cpu_pct=float(cpu.max()),
            mean_rss_mb=float(rss.mean()),
            max_rss_mb=float(rss.max()),
            estimated_power_w=est_power,
        )

    def _loop(self) -> None:
        import psutil

        proc = psutil.Process()
        interval = self.sample_ms / 1000.0

        while not self._stop.is_set():
            t = time.perf_counter()
            cpu_pct = psutil.cpu_percent(interval=None)
            rss_mb = proc.memory_info().rss / (1 << 20)
            self._cpu_pcts.append(cpu_pct)
            self._rss_mbs.append(rss_mb)
            self._ts.append(t)
            elapsed = time.perf_counter() - t
            remaining = interval - elapsed
            if remaining > 0:
                self._stop.wait(timeout=remaining)


@contextmanager
def cpu_profiler(
    sample_ms: float = 100.0,
    cpu_tdp_w: float = 65.0,
) -> Generator[CpuProfiler, None, None]:
    """Context manager equivalent of the CPU profiler."""
    profiler = CpuProfiler(sample_ms=sample_ms, cpu_tdp_w=cpu_tdp_w)
    profiler.start()
    try:
        yield profiler
    finally:
        profiler.stop()
