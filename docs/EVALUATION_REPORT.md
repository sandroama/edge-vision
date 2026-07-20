# Evaluation Report — edge-vision

> **Status:** Filled phase-by-phase; final consolidation lands at the end of Phase 6.
> The Pareto frontier has **two lanes**, reported separately and never conflated:
>
> - **CPU lane — MEASURED today (no GPU).** ONNX Runtime dynamic INT8: real
>   model-size reduction + CPU p50/p95/p99 latency with bootstrap 95% CIs.
>   See [results/phase3_cpu_int8.md](results/phase3_cpu_int8.md).
> - **GPU lane — PENDING an RTX-class run.** TensorRT FP16/INT8 latency, real
>   COCO mAP, and NVML watts/frame (the headline Pareto plot). See
>   [../NEXT_STEPS.md](../NEXT_STEPS.md). None of these numbers are claimed yet.

## Headline finding

**The full accuracy ↔ latency ↔ power Pareto (RTX 5080 vs CPU) is GPU-pending.**
What is **measured today**, with no GPU, is the **CPU INT8 corner** of it:

| Metric (CPU lane, ONNX Runtime `quantize_dynamic`, `CPUExecutionProvider`) | FP32 | INT8 (dynamic) |
|---|---|---|
| Model size | 3271.05 KiB | **898.35 KiB (−72.54%)** |
| CPU p50 latency | 1.274 ms | 3.871 ms |
| CPU p95 latency | 1.547 ms | 4.453 ms |
| Median speed-up (FP32/INT8) | — | **0.33×** (95% CI [0.323, 0.337]) |

Measured on Python 3.11 / ONNX Runtime 1.26 / single-thread CPU, 200 timed runs
each, percentile-bootstrap CIs from the raw per-run samples (all checked into
[results/phase3_cpu_int8.json](results/phase3_cpu_int8.json)). The **size
reduction (−72.5%) is the robust CPU-lane result**; the **negative latency
speed-up is reported exactly as measured** — dynamic INT8 adds per-op
quantize/dequantize overhead that dominates on a small, memory-light graph.
INT8 *latency* wins are a GPU/TensorRT story (GPU lane, pending).

> The model used here is the project's RT-DETR-shaped CI stand-in
> (`edgevision.models.tiny_model`), chosen so the lane runs with no 120 MB HF
> download; it is **not** a real-mAP detector, so no accuracy number is claimed
> on it. Real INT8 *accuracy* (RQ-E1) needs the COCO calibration run on the
> static-QDQ path (GPU lane).

---

## RQ-E1 — Accuracy retention under PTQ
**CPU lane (size, measured):** ONNX dynamic INT8 → **−72.54% model size**, no GPU
([results/phase3_cpu_int8.md](results/phase3_cpu_int8.md)).
**Accuracy (mAP retention) — pending:** the FP32/FP16/INT8 mAP + per-class drop
table needs real COCO + the static-QDQ / TRT-INT8 calibration path on a GPU.
*See [results/phase3_quantization.md](results/phase3_quantization.md).*

## RQ-E2 — Distillation gain at fixed latency
*See [results/phase4_distillation.md](results/phase4_distillation.md). Pending Phase 4 (GPU, ~50-epoch run).*

## RQ-E3 — Compile-pipeline speed-up
**CPU lane (measured):** PyTorch-eager → ONNX export → ONNX-Runtime CPU is wired
and benched; INT8-vs-FP32 CPU latency on the CI graph is measured
([results/phase3_cpu_int8.md](results/phase3_cpu_int8.md)).
**GPU lane — pending:** TRT-FP16/INT8 per-stage latency + the PyTorch→TRT
speed-up. *See [results/phase2_compile.md](results/phase2_compile.md).*

## RQ-E4 — Power-latency frontier (HEADLINE)
*See [results/phase5_power.md](results/phase5_power.md). **GPU-pending** — NVML
watts/frame only exists on NVIDIA GPUs.*

## RQ-E5 — MobileSAM bolt-on cost
*See [results/phase6_segmentation.md](results/phase6_segmentation.md). Pending Phase 6.*

---

## Failure analysis (TBD — Phase 6)

*Per-class accuracy drops under INT8. Thermal throttle events. Distillation surprises (positive or null).*

## Reproducibility checklist (carry forward to release)

- [ ] PyTorch checkpoint hash logged
- [ ] ONNX model hash logged
- [ ] TRT engine builder version logged
- [ ] CUDA + driver version logged
- [ ] Calibration set image IDs logged
- [ ] NVML sample interval logged
- [ ] CPU governor + thermal state logged for ONNX-CPU runs
