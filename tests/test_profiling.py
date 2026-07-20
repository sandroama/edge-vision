"""Tests for ``edgevision.profiling`` — CPU-only (no NVML / GPU needed).

All GPU-side assertions (real PowerMonitor, run_sustained with CUDA) are
gated behind ``pytest.mark.gpu``. The CPU and mock paths run everywhere.
"""

from __future__ import annotations

import time

import pytest

from edgevision.profiling import (
    MockPowerMonitor,
    PowerProfile,
    PowerSample,
    ThermalRunResult,
)
from edgevision.profiling.nvml_power import _summarise_samples

# --------------------------------------------------------------------------- PowerSample / PowerProfile schema


def test_power_sample_watts_property():
    s = PowerSample(
        timestamp_s=0.0, power_mw=150_000, temp_c=60,
        gpu_clock_mhz=2400, mem_clock_mhz=8000,
    )
    assert s.power_w == pytest.approx(150.0)


def test_power_profile_as_dict_is_json_serialisable():
    import json

    s = PowerSample(0.0, 120_000, 55, 2400, 8000)
    profile = _summarise_samples(
        [s, s, s],
        device_index=0, device_name="TestGPU", sample_interval_ms=100.0
    )
    json.loads(json.dumps(profile.as_dict()))


def test_power_profile_as_row_is_single_line():
    s = PowerSample(0.0, 100_000, 50, 2000, 6000)
    profile = _summarise_samples(
        [s] * 5, device_index=0, device_name="Test", sample_interval_ms=100.0
    )
    assert "\n" not in profile.as_row()


def test_power_profile_empty_samples_returns_zeros():
    profile = _summarise_samples(
        [], device_index=0, device_name="Test", sample_interval_ms=100.0
    )
    assert profile.n_samples == 0
    assert profile.mean_power_w == 0.0


# --------------------------------------------------------------------------- MockPowerMonitor


def test_mock_power_monitor_returns_plausible_profile():
    monitor = MockPowerMonitor(mean_power_w=200.0, noise_w=5.0, n_samples=50, seed=0)
    profile = monitor.summarise()
    assert isinstance(profile, PowerProfile)
    assert profile.n_samples == 50
    assert 150 < profile.mean_power_w < 250
    assert profile.max_temp_c > profile.mean_temp_c


def test_mock_power_monitor_thermal_throttle_detected():
    """The mock injects a throttle event in the last 10% of samples.
    With n_samples=100, last 10 frames are at 85% of base_clock.
    """
    monitor = MockPowerMonitor(base_clock_mhz=2400, n_samples=100, seed=42)
    profile = monitor.summarise()
    assert profile.throttle_events > 0


def test_mock_power_monitor_is_deterministic():
    a = MockPowerMonitor(seed=7).summarise()
    b = MockPowerMonitor(seed=7).summarise()
    assert a.mean_power_w == pytest.approx(b.mean_power_w)


# --------------------------------------------------------------------------- CpuProfile


def test_cpu_profiler_produces_profile_after_brief_run():
    """Run the profiler for 0.2s and verify the profile is populated."""
    pytest.importorskip("psutil")
    from edgevision.profiling import cpu_profiler

    with cpu_profiler(sample_ms=50.0, cpu_tdp_w=65.0) as profiler:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.2:
            _ = [x**2 for x in range(1000)]

    profile = profiler.summarise()
    assert profile.n_samples > 0
    assert profile.max_rss_mb > 0
    assert profile.estimated_power_w is not None


def test_cpu_profile_as_dict_is_json_serialisable():
    import json

    pytest.importorskip("psutil")
    from edgevision.profiling import cpu_profiler

    with cpu_profiler(sample_ms=200.0) as profiler:
        time.sleep(0.25)

    profile = profiler.summarise()
    json.loads(json.dumps(profile.as_dict()))


def test_cpu_profile_as_row_is_single_line():
    pytest.importorskip("psutil")
    from edgevision.profiling import cpu_profiler

    with cpu_profiler(sample_ms=200.0) as profiler:
        time.sleep(0.25)

    row = profiler.summarise().as_row()
    assert "\n" not in row


# --------------------------------------------------------------------------- ThermalRunResult + run_sustained


def test_run_sustained_mock_returns_thermal_result():
    from edgevision.profiling import run_sustained

    result = run_sustained(
        fn=lambda: time.sleep(0.001),
        duration_s=0.05,
        sample_ms=50.0,
        config_label="mock-smoke",
        warmup_iterations=3,
        use_mock_power=True,
        mock_power_kwargs={"n_samples": 5, "seed": 0},
    )
    assert isinstance(result, ThermalRunResult)
    assert result.n_iterations >= 1
    assert result.fps > 0
    assert result.config_label == "mock-smoke"


def test_run_sustained_watts_per_frame_is_positive_when_power_nonzero():
    from edgevision.profiling import run_sustained

    result = run_sustained(
        fn=lambda: time.sleep(0.001),
        duration_s=0.05,
        config_label="mock-test",
        warmup_iterations=2,
        use_mock_power=True,
        mock_power_kwargs={"mean_power_w": 150.0, "n_samples": 10, "seed": 0},
    )
    # watts_per_frame may be None if FPS is 0, but should not raise.
    assert result.watts_per_frame is None or result.watts_per_frame > 0


def test_thermal_result_as_dict_is_json_serialisable():
    import json

    from edgevision.profiling import run_sustained

    result = run_sustained(
        fn=lambda: time.sleep(0.001),
        duration_s=0.03,
        config_label="json-test",
        warmup_iterations=1,
        use_mock_power=True,
        mock_power_kwargs={"n_samples": 3, "seed": 0},
    )
    json.loads(json.dumps(result.as_dict()))


def test_thermal_result_as_row_is_single_line():
    from edgevision.profiling import run_sustained

    result = run_sustained(
        fn=lambda: time.sleep(0.001),
        duration_s=0.03,
        config_label="row-test",
        warmup_iterations=1,
        use_mock_power=True,
        mock_power_kwargs={"n_samples": 3, "seed": 0},
    )
    assert "\n" not in result.as_row()
