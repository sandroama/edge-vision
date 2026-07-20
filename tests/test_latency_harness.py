"""Tests for ``edgevision.inference.latency_harness``.

These tests are CPU-only and do not require torch — they exercise the
``measure_latency_cpu`` path with a known-time callable so we can assert
exact statistical properties.
"""

from __future__ import annotations

import time

import pytest

from edgevision.inference import (
    LatencyResult,
    measure_latency,
    measure_latency_cpu,
)


def _sleep_for(seconds: float):
    """Return a callable that sleeps for ``seconds`` each call."""

    def _fn():
        time.sleep(seconds)

    return _fn


def test_returns_latency_result_with_expected_shape():
    result = measure_latency_cpu(_sleep_for(0.001), n_runs=5, n_warmup=2)
    assert isinstance(result, LatencyResult)
    assert result.n == 5
    assert result.n_warmup == 2
    assert result.backend == "cpu"
    assert result.device == "cpu"
    # Sleep is bounded below by ~1ms; allow generous slack for test machine
    # jitter.
    assert result.p50_ms >= 0.5
    assert result.p95_ms >= 0.5
    assert result.fps > 0


def test_percentiles_are_ordered():
    result = measure_latency_cpu(_sleep_for(0.001), n_runs=20, n_warmup=2)
    assert result.min_ms <= result.p50_ms <= result.p95_ms <= result.p99_ms <= result.max_ms


def test_fps_is_inverse_of_mean():
    result = measure_latency_cpu(_sleep_for(0.001), n_runs=10, n_warmup=2)
    expected_fps = 1000.0 / result.mean_ms
    # Allow a small rounding window.
    assert abs(result.fps - expected_fps) < 1e-6


def test_warmup_iterations_are_executed_but_not_measured():
    counter = {"calls": 0}

    def _fn():
        counter["calls"] += 1

    result = measure_latency_cpu(_fn, n_runs=5, n_warmup=3)
    assert counter["calls"] == 5 + 3
    assert result.n == 5


def test_zero_warmup_is_allowed():
    result = measure_latency_cpu(_sleep_for(0.0005), n_runs=3, n_warmup=0)
    assert result.n == 3
    assert result.n_warmup == 0


def test_invalid_n_runs_rejected():
    with pytest.raises(ValueError, match="n_runs"):
        measure_latency_cpu(lambda: None, n_runs=0)


def test_invalid_n_warmup_rejected():
    with pytest.raises(ValueError, match="n_warmup"):
        measure_latency_cpu(lambda: None, n_runs=1, n_warmup=-1)


def test_dispatcher_picks_cpu_when_no_cuda(monkeypatch):
    """``measure_latency(..., backend='auto')`` should fall back to CPU when
    torch is missing OR when CUDA is unavailable."""
    result = measure_latency(_sleep_for(0.0005), n_runs=2, n_warmup=1, backend="auto")
    # Whatever backend was chosen, the returned report must be valid.
    assert isinstance(result, LatencyResult)
    assert result.n == 2
    assert result.backend in {"cpu", "cuda"}


def test_dispatcher_explicit_cpu():
    result = measure_latency(
        _sleep_for(0.0005), n_runs=2, n_warmup=1, backend="cpu", device="my-cpu"
    )
    assert result.backend == "cpu"
    assert result.device == "my-cpu"


def test_dispatcher_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        measure_latency(lambda: None, backend="opencl")  # type: ignore[arg-type]


def test_as_dict_is_serialisable_to_json():
    import json

    result = measure_latency_cpu(_sleep_for(0.0005), n_runs=2, n_warmup=1)
    payload = result.as_dict()
    # Round-trip through JSON to confirm everything's a primitive.
    json.loads(json.dumps(payload))


def test_as_row_is_a_single_line():
    result = measure_latency_cpu(_sleep_for(0.0005), n_runs=2, n_warmup=1)
    row = result.as_row()
    assert "\n" not in row
    assert "p50=" in row
    assert "fps=" in row
