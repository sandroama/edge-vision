"""GPU power and thermal monitoring via NVML (pynvml).

Samples GPU power draw, temperature, and clock speeds at a fixed interval
while a target workload runs, then summarises into a ``PowerProfile``. This
is the data that feeds the (mAP, p95-latency, watts/frame) Pareto plot.

Why NVML rather than command-line tools:
    * ``nvidia-smi`` has ~100 ms OS-level overhead per query — fine for one-shot
      but meaningless for a 15-min profile that needs consistent 100 ms samples.
    * pynvml binds directly to the NVML library, giving <1 ms overhead per
      sample. All power / temp / clock data comes from the same source.

Usage::

    with power_monitor(device_index=0, sample_ms=100) as monitor:
        # run the model for N iterations here
        pass
    profile = monitor.summarise()
    print(profile.as_row())

Lazy-imports pynvml so the module is importable without it (GPU-gated tests
use importorskip). Falls back to a synthetic mock when requested.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


@dataclass
class PowerSample:
    """One raw NVML measurement."""

    timestamp_s: float
    power_mw: int        # milliwatts
    temp_c: int          # degrees Celsius
    gpu_clock_mhz: int
    mem_clock_mhz: int

    @property
    def power_w(self) -> float:
        return self.power_mw / 1000.0


@dataclass
class PowerProfile:
    """Summary of a sustained power-monitoring session."""

    device_index: int
    device_name: str
    duration_s: float
    sample_interval_ms: float
    n_samples: int

    mean_power_w: float
    p50_power_w: float
    p95_power_w: float
    max_power_w: float

    mean_temp_c: float
    max_temp_c: float

    mean_gpu_clock_mhz: float
    min_gpu_clock_mhz: float   # throttle detection: min << mean means throttling

    throttle_events: int       # samples where GPU clock < 90% of peak observed

    @property
    def watts_per_frame(self) -> float | None:
        """Placeholder — populated by the Pareto aggregator with FPS data."""
        return None

    def as_dict(self) -> dict:
        return {
            "device_index": self.device_index,
            "device_name": self.device_name,
            "duration_s": round(self.duration_s, 2),
            "sample_interval_ms": self.sample_interval_ms,
            "n_samples": self.n_samples,
            "mean_power_w": round(self.mean_power_w, 2),
            "p50_power_w": round(self.p50_power_w, 2),
            "p95_power_w": round(self.p95_power_w, 2),
            "max_power_w": round(self.max_power_w, 2),
            "mean_temp_c": round(self.mean_temp_c, 1),
            "max_temp_c": round(self.max_temp_c, 1),
            "mean_gpu_clock_mhz": round(self.mean_gpu_clock_mhz, 0),
            "min_gpu_clock_mhz": round(self.min_gpu_clock_mhz, 0),
            "throttle_events": self.throttle_events,
        }

    def as_row(self) -> str:
        return (
            f"[GPU:{self.device_index} {self.device_name}] "
            f"pwr mean={self.mean_power_w:.1f}W p95={self.p95_power_w:.1f}W  "
            f"temp mean={self.mean_temp_c:.0f}°C max={self.max_temp_c:.0f}°C  "
            f"throttle={self.throttle_events} events  n={self.n_samples}"
        )


def _summarise_samples(
    samples: list[PowerSample],
    *,
    device_index: int,
    device_name: str,
    sample_interval_ms: float,
) -> PowerProfile:
    if not samples:
        return PowerProfile(
            device_index=device_index,
            device_name=device_name,
            duration_s=0.0,
            sample_interval_ms=sample_interval_ms,
            n_samples=0,
            mean_power_w=0.0, p50_power_w=0.0, p95_power_w=0.0, max_power_w=0.0,
            mean_temp_c=0.0, max_temp_c=0.0,
            mean_gpu_clock_mhz=0.0, min_gpu_clock_mhz=0.0,
            throttle_events=0,
        )

    power_w = np.array([s.power_w for s in samples])
    temp_c = np.array([s.temp_c for s in samples], dtype=float)
    clocks = np.array([s.gpu_clock_mhz for s in samples], dtype=float)

    peak_clock = float(clocks.max())
    throttle_threshold = peak_clock * 0.90
    throttle_events = int((clocks < throttle_threshold).sum())

    duration_s = samples[-1].timestamp_s - samples[0].timestamp_s if len(samples) > 1 else 0.0

    return PowerProfile(
        device_index=device_index,
        device_name=device_name,
        duration_s=duration_s,
        sample_interval_ms=sample_interval_ms,
        n_samples=len(samples),
        mean_power_w=float(power_w.mean()),
        p50_power_w=float(np.percentile(power_w, 50)),
        p95_power_w=float(np.percentile(power_w, 95)),
        max_power_w=float(power_w.max()),
        mean_temp_c=float(temp_c.mean()),
        max_temp_c=float(temp_c.max()),
        mean_gpu_clock_mhz=float(clocks.mean()),
        min_gpu_clock_mhz=float(clocks.min()),
        throttle_events=throttle_events,
    )


# --------------------------------------------------------------------------- monitor context


class PowerMonitor:
    """Background-thread NVML sampler.

    Start sampling via the ``power_monitor`` context manager. The thread
    runs until ``stop()`` is called; ``summarise()`` returns the profile.
    """

    def __init__(self, device_index: int = 0, sample_ms: float = 100.0) -> None:
        self.device_index = device_index
        self.sample_ms = sample_ms
        self._samples: list[PowerSample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._device_name = "unknown"

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            name = pynvml.nvmlDeviceGetName(handle)
            self._device_name = name.decode() if isinstance(name, bytes) else str(name)
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PowerMonitor requires pynvml. Install with `pip install -e '.[gpu]'`."
            ) from e

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def summarise(self) -> PowerProfile:
        return _summarise_samples(
            self._samples,
            device_index=self.device_index,
            device_name=self._device_name,
            sample_interval_ms=self.sample_ms,
        )

    def _sample_loop(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        interval = self.sample_ms / 1000.0

        while not self._stop_event.is_set():
            t = time.perf_counter()
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                gpu_clk = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS
                )
                mem_clk = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                )
                self._samples.append(
                    PowerSample(
                        timestamp_s=t,
                        power_mw=int(power_mw),
                        temp_c=int(temp),
                        gpu_clock_mhz=int(gpu_clk),
                        mem_clock_mhz=int(mem_clk),
                    )
                )
            except Exception:  # pragma: no cover - defensive
                pass
            # Sleep for the remainder of the interval.
            elapsed = time.perf_counter() - t
            remaining = interval - elapsed
            if remaining > 0:
                self._stop_event.wait(timeout=remaining)


@contextmanager
def power_monitor(
    device_index: int = 0,
    sample_ms: float = 100.0,
) -> Generator[PowerMonitor, None, None]:
    """Context manager that runs the NVML sampler for the duration of its body.

    Usage::

        with power_monitor(device_index=0, sample_ms=100) as monitor:
            run_inference_for_n_seconds(n=60)
        profile = monitor.summarise()
    """
    monitor = PowerMonitor(device_index=device_index, sample_ms=sample_ms)
    monitor.start()
    try:
        yield monitor
    finally:
        monitor.stop()


# --------------------------------------------------------------------------- mock


class MockPowerMonitor:
    """Synthetic monitor for tests — generates plausible random power data.

    Produces a ``PowerProfile`` with configurable mean + noise parameters
    without requiring an actual GPU or pynvml.
    """

    def __init__(
        self,
        *,
        device_index: int = 0,
        mean_power_w: float = 200.0,
        noise_w: float = 5.0,
        mean_temp_c: float = 65.0,
        noise_temp: float = 2.0,
        base_clock_mhz: int = 2400,
        n_samples: int = 100,
        sample_ms: float = 100.0,
        seed: int = 0,
    ) -> None:
        self.device_index = device_index
        self._mean_p = mean_power_w
        self._noise_p = noise_w
        self._mean_t = mean_temp_c
        self._noise_t = noise_temp
        self._base_clk = base_clock_mhz
        self._n = n_samples
        self._sample_ms = sample_ms
        self._seed = seed

    def summarise(self) -> PowerProfile:
        rng = np.random.default_rng(self._seed)
        power = self._mean_p + rng.normal(0, self._noise_p, self._n)
        power = np.clip(power, 0, None)
        temps = self._mean_t + rng.normal(0, self._noise_t, self._n)
        # Simulate a mild throttle event in the last 10% of samples.
        clocks = np.full(self._n, self._base_clk, dtype=float)
        throttle_start = int(self._n * 0.90)
        clocks[throttle_start:] = self._base_clk * 0.85

        samples = [
            PowerSample(
                timestamp_s=float(i * self._sample_ms / 1000.0),
                power_mw=int(p * 1000),
                temp_c=int(t),
                gpu_clock_mhz=int(c),
                mem_clock_mhz=8000,
            )
            for i, (p, t, c) in enumerate(zip(power, temps, clocks, strict=True))
        ]
        return _summarise_samples(
            samples,
            device_index=self.device_index,
            device_name="MockGPU",
            sample_interval_ms=self._sample_ms,
        )
