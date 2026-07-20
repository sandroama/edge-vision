"""Hardware profiling — GPU power + thermal + CPU profile.

Phase 5 modules:
    nvml_power     — NVML 100ms power sampling; MockPowerMonitor for tests.
    thermal_runner — Sustained inference runner combining latency + power.
    cpu_profile    — psutil CPU% + RSS + estimated watt draw for ONNX-CPU.
"""

from edgevision.profiling.cpu_profile import (
    CpuProfile,
    CpuProfiler,
    cpu_profiler,
)
from edgevision.profiling.nvml_power import (
    MockPowerMonitor,
    PowerMonitor,
    PowerProfile,
    PowerSample,
    power_monitor,
)
from edgevision.profiling.thermal_runner import (
    ThermalRunResult,
    run_sustained,
)

__all__ = [
    "CpuProfile",
    "CpuProfiler",
    "MockPowerMonitor",
    "PowerMonitor",
    "PowerProfile",
    "PowerSample",
    "ThermalRunResult",
    "cpu_profiler",
    "power_monitor",
    "run_sustained",
]
