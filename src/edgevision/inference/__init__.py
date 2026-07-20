"""Inference latency harness.

Phase 2: ``latency_harness`` — batch=1 p50/p95/p99 with CUDA events on GPU,
``time.perf_counter`` on CPU.
"""

from edgevision.inference.latency_harness import (
    LatencyResult,
    measure_latency,
    measure_latency_cpu,
    measure_latency_cuda,
)

__all__ = [
    "LatencyResult",
    "measure_latency",
    "measure_latency_cpu",
    "measure_latency_cuda",
]
