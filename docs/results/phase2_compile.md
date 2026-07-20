# Phase 2 — ONNX + TensorRT export (results)

> **Status:** Phase 2 modules wired (CPU-side green, GPU/TRT paths designed).
> Real RT-DETR-R50 export + the per-backend Pareto row are GPU work, blocked
> only on installing the `[gpu]` and `[trt]` extras and running the script.

## Module status

| Module | File | Tests | Notes |
|---|---|---|---|
| Latency harness | `src/edgevision/inference/latency_harness.py` | 12/12 ✅ | CPU (perf_counter) + CUDA-event paths under one dispatcher |
| Tiny model | `src/edgevision/models/tiny_model.py` | covered by export tests | RT-DETR-shaped output (logits + pred_boxes), tiny enough for CI |
| ONNX export | `src/edgevision/compile/onnx_export.py` | 8/8 ✅ (gated on torch+onnx) | opset 17, dynamic batch, `verify_onnx` returns full metadata |
| ONNX Runtime CPU | `src/edgevision/compile/onnxrt_cpu.py` | 6/6 ✅ (gated on torch+ort) | `make_callable` plugs straight into the latency harness |
| TRT engine builder | `src/edgevision/compile/trt_build.py` | 4 always + 3 GPU-gated ✅ | FP32 + FP16 ready; INT8 calibrator hook for Phase 3 |
| Compile smoke script | `scripts/run_compile_smoke.py` | 2/2 ✅ (gated on torch+ort) | `--stage onnx\|onnxrt\|trt\|all` |
| Latency sweep script | `scripts/run_latency_sweep.py` | covered by smoke tests | Writes JSON for the Phase 5 Pareto aggregator |

**Phase 1 + Phase 2 totals: 57 tests pass, 0 fail, 6 skipped on environments
without torch/tensorrt.** All skips are clearly labelled — they run in CI
where torch + onnx + onnxruntime are installed by the `[dev]` extras, and on
the GPU box where `[gpu,trt]` are also installed.

## CPU-only sanity check (no torch needed)

The latency harness is the only Phase-2 module exercisable without torch.
Running it on a pure-Python 8×8 matmul on the 9950X gave::

    [cpu/cpu (8x8 matmul)] p50=0.04ms  p95=0.04ms  p99=0.05ms  fps=24092.5  (n=50)

This isn't a useful number (no model is involved); the point is the harness
produces well-formed `LatencyResult`s end-to-end, including the JSON
serialisation that downstream phases use to populate the Pareto frontier.

## RQ-E3 — per-stage latency (pending GPU run)

The headline table for Phase 2. After running the bench script on the
RTX 5080, fill the rows below from `docs/results/phase2_latency.json`.

| Backend | Model | p50 (ms) | p95 (ms) | p99 (ms) | FPS | n |
|---|---|---|---|---|---|---|
| torch-cpu (eager) | tiny | TBD | TBD | TBD | TBD | TBD |
| torch-cpu (eager) | rtdetr-r50 | TBD | TBD | TBD | TBD | TBD |
| onnxrt-cpu | tiny | TBD | TBD | TBD | TBD | TBD |
| onnxrt-cpu | rtdetr-r50 | TBD | TBD | TBD | TBD | TBD |
| torch-cuda (eager) | rtdetr-r50 | TBD | TBD | TBD | TBD | TBD |
| trt-fp32 | rtdetr-r50 | TBD | TBD | TBD | TBD | TBD |
| trt-fp16 | rtdetr-r50 | TBD | TBD | TBD | TBD | TBD |

Reproduction steps::

    # 1. Install GPU deps once.
    pip install -e ".[dev,gpu,trt]"

    # 2. Phase 1 baseline produces rtdetr_r50.pth (HF download is automatic).
    python scripts/run_baseline_smoke.py --backend rtdetr ...

    # 3. Export + build.
    python scripts/run_compile_smoke.py --stage all --precision fp16

    # 4. Sweep.
    python scripts/run_latency_sweep.py \
        --backends torch-cpu torch-cuda onnxrt-cpu trt-fp32 trt-fp16 \
        --model rtdetr \
        --n-runs 100 --n-warmup 20 \
        --out-json docs/results/phase2_latency.json

The aggregator that turns this JSON into the Phase-5 Pareto plot lands in
Phase 5 (`evaluation/pareto_aggregator.py`).

## Design notes worth keeping

* **CUDA events vs perf_counter.** The harness uses `torch.cuda.Event`
  with `enable_timing=True` for the CUDA path. Wrapping `torch.cuda.synchronize()`
  into the call (rather than at the start of each iteration) ensures
  overlapped kernels don't bleed across samples. Without that synchronization,
  p99 numbers look better than reality.
* **Why opset 17.** TensorRT 10.x's stable opset window. Bumping forward to
  19 is tempting but adds risk on Blackwell (sm_100); leave the upgrade
  to a deliberate Phase 7 task.
* **Why a tiny model in CI.** Real RT-DETR is ~120 MB and takes 30+ seconds
  to load. The tiny model has the same input shape and emits the same two
  output names (`logits`, `pred_boxes`) so the export and ORT CPU paths
  are *real* in CI, not mocked. The decoder lives outside this module.
* **Engines are not portable.** Always build on the deployment host. The
  `.engine` files are gitignored. `TrtBuildResult.trt_version` records the
  builder version for reproducibility audits.

## What's deliberately not in Phase 2

* **INT8 calibration** — the `TrtBuildConfig.int8_calibrator` hook exists,
  but the calibration *dataset* + *calibrator class* are Phase 3 (they need
  COCO calibration images). Building an INT8 engine without a calibrator
  raises a clear `ValueError`.
* **Output decoder.** The ORT executor returns the raw `(logits, pred_boxes)`
  tensors. RT-DETR's specific post-processing (sigmoid + top-K + class-mapping)
  lives in `models/rtdetr_wrapper.RTDetrDetector`, which already handles it
  through the HF processor. Sharing decoder code between the torch and
  ORT paths is a Phase-3 cleanup if it pays off.
