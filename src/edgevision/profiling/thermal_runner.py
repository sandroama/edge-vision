"""Thermal workload runner — runs a target function for a fixed wall-clock
duration while the NVML power monitor samples in the background.

The canonical Phase-5 run is 15 minutes (900 s) of sustained inference.
This module orchestrates that: it counts iterations, respects the time
budget, and produces a ``ThermalRunResult`` that ties together the latency
harness and the power profile.

Design choice: this is *not* a context manager — it has a clear start/end
and returns a rich result object. Context managers are for resource cleanup;
this is for measurement.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from edgevision.inference.latency_harness import LatencyResult, _summarize
from edgevision.profiling.nvml_power import PowerProfile


@dataclass
class ThermalRunResult:
    """Bundled output of one sustained inference run."""

    latency: LatencyResult
    power: PowerProfile
    n_iterations: int
    duration_s: float
    fps: float
    config_label: str

    @property
    def watts_per_frame(self) -> float | None:
        """Mean watts / FPS = watt-seconds per frame."""
        if self.fps <= 0 or self.power.mean_power_w <= 0:
            return None
        return self.power.mean_power_w / self.fps

    def as_dict(self) -> dict:
        d: dict[str, Any] = {
            "config_label": self.config_label,
            "n_iterations": self.n_iterations,
            "duration_s": round(self.duration_s, 2),
            "fps": round(self.fps, 2),
            "watts_per_frame": round(self.watts_per_frame, 4) if self.watts_per_frame else None,
            "latency": self.latency.as_dict(),
            "power": self.power.as_dict(),
        }
        return d

    def as_row(self) -> str:
        wpf = f"{self.watts_per_frame:.4f}" if self.watts_per_frame else "n/a"
        return (
            f"[{self.config_label}] "
            f"fps={self.fps:.1f}  p95={self.latency.p95_ms:.2f}ms  "
            f"power_mean={self.power.mean_power_w:.1f}W  "
            f"watts/frame={wpf}  throttle={self.power.throttle_events}"
        )


def run_sustained(
    fn: Callable[[], Any],
    *,
    duration_s: float = 900.0,
    sample_ms: float = 100.0,
    device_index: int = 0,
    config_label: str = "unknown",
    warmup_iterations: int = 50,
    use_mock_power: bool = False,
    mock_power_kwargs: dict | None = None,
) -> ThermalRunResult:
    """Run ``fn()`` repeatedly for ``duration_s`` seconds, sampling NVML.

    Args:
        fn: zero-arg callable — one model forward pass.
        duration_s: how long to sustain the workload (seconds). Set to 10 for
            a quick sanity check; 900 for the real 15-minute Phase-5 sweep.
        sample_ms: NVML sampling interval.
        device_index: GPU index (0 on single-GPU systems).
        config_label: tag for the Pareto table row.
        warmup_iterations: iterations to skip before timing + power sampling.
        use_mock_power: use the ``MockPowerMonitor`` instead of real NVML.
            Set automatically in tests or on CPU-only machines.
        mock_power_kwargs: forwarded to ``MockPowerMonitor.__init__``.

    Returns:
        ``ThermalRunResult`` with combined latency + power profile.
    """
    # Warmup.
    for _ in range(warmup_iterations):
        fn()

    # Choose power monitor.
    if use_mock_power:
        from edgevision.profiling.nvml_power import MockPowerMonitor

        power_mon = MockPowerMonitor(**(mock_power_kwargs or {}))
    else:
        from edgevision.profiling.nvml_power import PowerMonitor

        power_mon = PowerMonitor(device_index=device_index, sample_ms=sample_ms)
        power_mon.start()

    # Timed loop.
    samples_ms: list[float] = []
    n_iters = 0
    t_start = time.perf_counter()
    deadline = t_start + duration_s

    try:
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - t0) * 1e3
            samples_ms.append(elapsed)
            n_iters += 1
    finally:
        if not use_mock_power:
            power_mon.stop()  # type: ignore[union-attr]

    actual_duration = time.perf_counter() - t_start
    fps = n_iters / actual_duration if actual_duration > 0 else 0.0

    latency = _summarize(
        samples_ms,
        n_warmup=warmup_iterations,
        backend="cpu" if use_mock_power else "cuda",
        device=config_label,
    )
    power_profile = power_mon.summarise()

    return ThermalRunResult(
        latency=latency,
        power=power_profile,
        n_iterations=n_iters,
        duration_s=actual_duration,
        fps=fps,
        config_label=config_label,
    )
