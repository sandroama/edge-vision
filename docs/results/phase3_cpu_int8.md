# Phase 3 — CPU INT8 lane (measured, no GPU)

> **Status:** MEASURED on CPU. This is the GPU-free slice of the Pareto frontier
> — real ONNX Runtime INT8 size + CPU latency, produced by
> [`scripts/bench_cpu_int8.py`](../../scripts/bench_cpu_int8.py). The
> TensorRT latency and NVML watts/frame rows remain **GPU-pending**
> (see [`NEXT_STEPS.md`](../../NEXT_STEPS.md)); nothing here implies those.

## What was measured

- **Model:** `edgevision.models.tiny_model.make_tiny_model (RT-DETR-shaped CI stand-in)` — an RT-DETR-shaped tiny detector
  (input `[1, 3, 640, 640]`, opset 17,
  8 initializers). Used so the CPU lane runs with
  **no 120 MB HF download**; it is a shape-faithful stand-in, *not* a real-mAP
  detector, so no accuracy number is claimed here.
- **Quantization:** `onnxruntime.quantization.quantize_dynamic (QInt8 weights, dynamic activations)` — a CPU-only operation (weights ->
  int8, activation scales computed at runtime; no calibration set needed).
- **Inference:** `CPUExecutionProvider` (explicit), intra-op threads =
  1, graph optimization = `all`,
  40 warmup + **200 timed** single-image forward
  passes per model.

## Model size: FP32 -> INT8 (dynamic)

| Model | Size |
|---|---|
| FP32 ONNX | 3271.05 KiB (3,349,556 bytes) |
| INT8 ONNX (dynamic) | 898.35 KiB (919,912 bytes) |
| **Reduction** | **72.54%** |

## CPU latency (single image, 95% bootstrap CIs in brackets)

Bracketed ranges are percentile-bootstrap 95% CIs (10,000 replicates)
computed from the 200 raw per-run samples this run generated (all
samples are checked into `phase3_cpu_int8.json`).

| Model | p50 ms [95% CI] | p95 ms [95% CI] | p99 ms [95% CI] | mean +/- std ms | FPS (mean) |
|---|---|---|---|---|---|
| FP32 | 1.2739 [1.254, 1.3032] | 1.5471 [1.4497, 1.7441] | 1.8662 [1.6713, 2.0518] | 1.3184 +/- 0.1323 | 758.49 |
| INT8 (dynamic) | 3.871 [3.8332, 3.9191] | 4.453 [4.3791, 4.6173] | 4.7369 [4.5441, 5.3703] | 3.9479 +/- 0.2784 | 253.3 |

**Median-latency speed-up (FP32 / INT8): 0.3291x**
(95% CI [0.3228, 0.3371]).

> Reading the speed-up honestly: dynamic INT8 on a small CPU graph trades int8
> weight storage (the real, portable 72.54% size win) against
> per-op quantize/dequantize overhead. On a graph this small the latency change
> can be modest or even negative; the **size reduction is the robust CPU-lane
> result**, and both numbers are reported exactly as measured. On larger,
> compute-bound graphs INT8 latency gains are typically larger — but that is not
> claimed here, only what was measured on this model.

## Environment

- Python 3.11.15 · macOS-26.4.1-arm64-arm-64bit
- processor: `arm`
- onnxruntime 1.26.0 · onnx 1.21.0 · numpy 1.26.4
- available providers: ['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
- generated (UTC): 2026-06-03T01:31:52.187025+00:00

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install onnx onnxruntime "numpy<2" sympy torch   # CPU-only; no CUDA needed
pip install -e . --no-deps
python scripts/bench_cpu_int8.py --n-runs 200 --n-warmup 40
```

## What this is NOT

- ❌ Not a TensorRT or GPU result. The `trt-*` rows and NVML watts/frame stay
  **GPU-pending** ([`NEXT_STEPS.md`](../../NEXT_STEPS.md)).
- ❌ Not an mAP claim. The tiny stand-in model has no meaningful accuracy; the
  real INT8-accuracy row (RQ-E1) needs the COCO calibration run on a GPU via the
  static-QDQ path (`edgevision.quantization.onnx_qdq.quantize_static`).
- ✅ Is a real, reproducible measurement of the **CPU INT8 size reduction and
  CPU inference latency** — the part of the Pareto that needs no GPU.
