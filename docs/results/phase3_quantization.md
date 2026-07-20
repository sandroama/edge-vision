# Phase 3 — Quantization (results)

> **Status:** Phase 3 modules wired. CPU-side (ORT QDQ, calibration pipeline,
> quant_eval) is tested and green. Real INT8 TRT + ONNX QDQ **accuracy** numbers
> require a GPU run on real COCO val2017 — the harnesses are in place.
>
> **Measured CPU-INT8 size + latency (no GPU) lives in a sibling doc:**
> [phase3_cpu_int8.md](phase3_cpu_int8.md) reports a real **−72.5% model-size
> reduction** and full CPU p50/p95/p99 latency (with bootstrap 95% CIs) from
> `onnxruntime.quantization.quantize_dynamic`. That is the *dynamic*, data-free
> CPU path; this doc covers the *static* (calibration-bearing) QDQ + TRT-INT8
> path that produces the **mAP-retention** table below.

## Module status

| Module | File | Tests | Notes |
|---|---|---|---|
| Calibration dataset | `src/edgevision/quantization/calib_dataset.py` | 16/16 ✅ | Uniform / stratified / first sampling; `BatchProvider` iterator; synthetic loader for CI |
| ONNX QDQ (CPU INT8) | `src/edgevision/quantization/onnx_qdq.py` | 3/3 ✅ (2 gated on ort.quantization) | `quantize_static` + `QDQQuantizationConfig`; per-channel QInt8 by default |
| TRT INT8 calibrator | `src/edgevision/quantization/trt_int8.py` | 1 always + 3 GPU-gated ✅ | `IInt8EntropyCalibrator2` wrapper; cache management; `build_int8_engine` convenience call |
| Quant eval | `src/edgevision/evaluation/quant_eval.py` | 14/14 ✅ | `QuantizationDelta`, per-class drop ranking, `compare_metrics`, `summary_table` |
| Quant smoke script | `scripts/run_quant_smoke.py` | 1/1 ✅ | `--candidate-recall / --candidate-fp-rate` for CI smoke; reads from real COCO with paths |

**Running totals: 94 tests passing, 10 skipping cleanly (torch / tensorrt / ort.quantization
absent from the local Python 3.15a environment). 0 failing.**

## CI smoke (synthetic dataset, simple eval backend)

```
[edge-vision] dataset: 4 images, 8 GT boxes, 3 classes

Reference (fp32 / mock-perfect):
  Precision : 1.000   Recall : 1.000   F1 : 1.000

Candidate (int8 / mock-noisy, recall=0.65, fp_rate=0.20):
  Precision : 1.000   Recall : 0.625   F1 : 0.769

RQ-E1 delta:
  Overall F1: 1.000 -> 0.769  (delta=-0.231)
  Worst 3 classes:
    - class_02   ref=1.000  cand=0.667  delta=-0.333  retained= 66.7%
    - class_00   ref=1.000  cand=0.800  delta=-0.200  retained= 80.0%
    - class_01   ref=1.000  cand=0.800  delta=-0.200  retained= 80.0%
```

This uses mock detectors to stand in for FP32 and INT8 models — the numbers
are illustrative, not real. The point is the harness correctly identifies
which classes degrade the most, producing the per-class drop table that is
the actual deliverable for RQ-E1.

## RQ-E1 — accuracy retention under PTQ (pending GPU run)

Run the GPU pipeline to fill in this table:

```bash
# 0. Ensure real COCO data + FP32 ONNX artifact from Phase 1+2.
# 1. Build calibration provider from COCO val2017.
python scripts/run_quant_smoke.py \
    --coco-annotations data/coco/annotations/instances_val2017.json \
    --coco-images data/coco/val2017 \
    --max-images 500 \
    --eval-backend pycocotools \
    --out-json docs/results/phase3_quantization.json
```

| Config | mAP@[0.5:0.95] | mAP@0.5 | FPS (RTX 5080) | Size (MB) | Retained % |
|---|---|---|---|---|---|
| FP32 (baseline) | TBD | TBD | TBD | ~120 | 100% |
| FP16 | TBD | TBD | TBD | ~60 | TBD |
| INT8 TRT (per-channel) | TBD | TBD | TBD | ~30 | TBD |
| INT8 ORT QDQ (per-channel) | TBD | TBD | TBD | ~30 | TBD |

Expected from literature: FP16 retains ≥99% of FP32 mAP; INT8 per-channel
retains ≥96% on COCO-scale detectors. Classes that break first are usually
small-object / rare (e.g. `toothbrush`, `hair drier`) — the per-class drop
table from `quant_eval` exposes that.

## Design notes

* **Calibrator reuse.** The `.calib.cache` file next to the engine is
  architecture-specific. Delete it when moving the checkpoint to a new
  GPU or after a major model change.
* **Per-channel vs per-tensor INT8.** Per-channel (one scale per output
  channel) costs a small build-time overhead but consistently yields better
  mAP than the per-tensor default. `QDQQuantizationConfig.per_channel=True`
  is the default for both the ONNX and TRT paths.
* **Output head exclusion.** If the detection head produces activations
  with a very different range than the backbone, exclude it from INT8 via
  `TrtBuildConfig` `strict_types=False` and
  `QDQQuantizationConfig.nodes_to_exclude`. This is the most common INT8
  tuning knob.
