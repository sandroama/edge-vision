"""Measure the CPU lane of the Pareto: ONNX dynamic-INT8 size + latency (RQ-E1/E3, CPU).

This is the **GPU-free** slice of the Pareto frontier. It produces *real,
reproducible* numbers — model file-size reduction and CPU inference latency —
for the fp32 -> INT8 transition, using only ``onnxruntime`` on the
``CPUExecutionProvider``. No CUDA, no TensorRT, no NVML required.

Pipeline (all wired to the project's own export path):

    1. Build the project's RT-DETR-shaped tiny detector
       (``edgevision.models.tiny_model.make_tiny_model``) and export it to a
       FP32 ONNX graph with ``edgevision.compile.onnx_export.export_to_onnx``
       (opset 17, the same exporter the GPU path uses).
    2. Quantize that graph to INT8 with
       ``onnxruntime.quantization.quantize_dynamic`` — a CPU-only operation
       that quantizes weights to int8 and computes activation scales at
       runtime (no calibration set needed).
    3. Measure file sizes: fp32 vs int8 bytes -> % reduction.
    4. Time N single-image forward passes for BOTH models on the
       ``CPUExecutionProvider`` (explicit), via the project's
       ``OnnxRuntimeCPUExecutor`` / latency harness. Warm up first.
    5. Report p50 / p95 / p99 and mean +/- std for each, plus the INT8
       speed-up, with **bootstrap 95% confidence intervals** computed from the
       raw per-run latency samples this script generated.

Why dynamic (not static QDQ) here? The project already ships a *static* QDQ
path (``edgevision.quantization.onnx_qdq.quantize_static``) that needs a COCO
calibration set. Dynamic quant needs no data, so it is the honest "runs on any
laptop, no dataset" CPU-lane measurement — exactly the no-GPU slice this script
is meant to bank. The static path remains the GPU/data-bearing RQ-E1 row.

Outputs (under ``docs/results/``):
    * ``phase3_cpu_int8.json`` — full summary + every raw latency sample.
    * ``phase3_cpu_int8.md``   — short human-readable table.

Run::

    python scripts/bench_cpu_int8.py --n-runs 200 --n-warmup 30

Determinism note: latency is wall-clock and therefore machine- and
load-dependent. The *script* is deterministic (fixed seed, fixed input); the
absolute milliseconds are not portable, but the size reduction is, and the
relative speed-up is stable across repeated runs on the same box.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Project export path — the same modules the GPU pipeline uses.
from edgevision.compile.onnx_export import export_to_onnx, verify_onnx
from edgevision.compile.onnxrt_cpu import OnnxRuntimeCPUExecutor
from edgevision.models.tiny_model import make_tiny_input, make_tiny_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "results"


# --------------------------------------------------------------------------- latency

def _percentile(samples: list[float], q: float) -> float:
    return float(np.percentile(samples, q)) if samples else 0.0


def _time_fn(fn: Callable[[], object], *, n_runs: int, n_warmup: int) -> list[float]:
    """Warm up ``n_warmup`` times, then return ``n_runs`` raw per-call ms samples."""
    for _ in range(n_warmup):
        fn()
    samples_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1e3)
    return samples_ms


def _summary(samples_ms: list[float]) -> dict:
    mean = float(statistics.fmean(samples_ms))
    std = float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0
    return {
        "n": len(samples_ms),
        "p50_ms": round(_percentile(samples_ms, 50), 4),
        "p95_ms": round(_percentile(samples_ms, 95), 4),
        "p99_ms": round(_percentile(samples_ms, 99), 4),
        "mean_ms": round(mean, 4),
        "std_ms": round(std, 4),
        "min_ms": round(min(samples_ms), 4),
        "max_ms": round(max(samples_ms), 4),
        "fps_mean": round(1000.0 / mean, 2) if mean > 0 else None,
    }


# --------------------------------------------------------------------------- bootstrap CIs

def _bootstrap_ci(
    samples: list[float],
    stat_fn: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile-bootstrap (lo, hi) CI for ``stat_fn`` over ``samples``."""
    arr = np.asarray(samples, dtype=np.float64)
    n = arr.size
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        resample = arr[rng.integers(0, n, size=n)]
        boots[i] = stat_fn(resample)
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def _ci_block(
    samples: list[float], *, n_boot: int, alpha: float, rng: np.random.Generator
) -> dict:
    """Bootstrap CIs for the latency percentiles + mean of one sample set."""
    out = {}
    for label, q in (("p50_ms", 50.0), ("p95_ms", 95.0), ("p99_ms", 99.0)):
        lo, hi = _bootstrap_ci(
            samples, lambda a, q=q: float(np.percentile(a, q)),
            n_boot=n_boot, alpha=alpha, rng=rng,
        )
        out[label] = {"lo": round(lo, 4), "hi": round(hi, 4)}
    lo, hi = _bootstrap_ci(
        samples, lambda a: float(np.mean(a)), n_boot=n_boot, alpha=alpha, rng=rng
    )
    out["mean_ms"] = {"lo": round(lo, 4), "hi": round(hi, 4)}
    return out


def _speedup_ci(
    fp32: list[float], int8: list[float], *, n_boot: int, alpha: float, rng: np.random.Generator
) -> dict:
    """Bootstrap CI for the median-latency speed-up (fp32_p50 / int8_p50).

    Resamples each sample set independently (they are independent timing runs),
    recomputes the ratio of medians per bootstrap replicate.
    """
    a = np.asarray(fp32, dtype=np.float64)
    b = np.asarray(int8, dtype=np.float64)
    na, nb = a.size, b.size
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ra = a[rng.integers(0, na, size=na)]
        rb = b[rng.integers(0, nb, size=nb)]
        boots[i] = float(np.median(ra) / np.median(rb))
    point = float(np.median(a) / np.median(b))
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure the CPU lane of the Pareto: ONNX dynamic-INT8 model-size "
                    "reduction + CPU p50/p95/p99 latency (with bootstrap CIs). No GPU. "
                    "See the module docstring for full detail.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-runs", type=int, default=200, help="timed forward passes per model (>=100)")
    ap.add_argument("--n-warmup", type=int, default=30, help="warmup forward passes (not timed)")
    ap.add_argument("--n-boot", type=int, default=10000, help="bootstrap replicates for CIs")
    ap.add_argument("--alpha", type=float, default=0.05, help="1-alpha CI (0.05 -> 95%% level)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-classes", type=int, default=80, help="tiny-model class count (COCO-like)")
    ap.add_argument("--num-queries", type=int, default=300, help="tiny-model query count (RT-DETR-like)")
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--intra-op-threads", type=int, default=1,
                    help="ORT intra-op threads; 1 = single-thread, comparable across machines")
    ap.add_argument("--graph-opt", default="all", choices=["off", "basic", "extended", "all"])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "checkpoints" / "cpu_int8")
    args = ap.parse_args()

    if args.n_runs < 100:
        print(f"[warn] --n-runs={args.n_runs} < 100; bumping to 100 for stable percentiles.")
        args.n_runs = 100

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    args.work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.work_dir / "tiny_detector_raw.onnx"
    fp32_path = args.work_dir / "tiny_detector_fp32.onnx"          # pre-processed FP32 (quant baseline)
    int8_path = args.work_dir / "tiny_detector_int8_dynamic.onnx"

    # 1. Export the project's RT-DETR-shaped model to FP32 ONNX via the
    #    project's own exporter (edgevision.compile.onnx_export.export_to_onnx).
    print(f"[cpu-int8] building tiny detector "
          f"(num_classes={args.num_classes}, num_queries={args.num_queries})")
    model = make_tiny_model(num_classes=args.num_classes, num_queries=args.num_queries)
    dummy = make_tiny_input(batch=1, height=args.height, width=args.width)
    export_to_onnx(model, dummy, raw_path)
    raw_info = verify_onnx(raw_path)
    print(f"[cpu-int8] exported FP32 ONNX (project exporter) -> {raw_path}  "
          f"(opset={raw_info.opset}, initializers={raw_info.n_initializers})")

    # 2. ORT-recommended pre-process (shape inference + constant folding) so the
    #    quantizer sees a clean graph, then dynamic INT8 quant (CPU-only, no
    #    calibration data). torch>=2.9's default dynamo exporter occasionally
    #    emits a graph ORT's quantizer cannot shape-infer; if pre-process or
    #    quant fails on it, fall back to torch's legacy TorchScript exporter,
    #    which yields a quantization-friendly opset-17 graph. Either way the
    #    INT8 model is real and produced by quantize_dynamic.
    def _preprocess_and_quantize(src: Path) -> None:
        quant_pre_process(str(src), str(fp32_path), skip_symbolic_shape=True)
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

    print("[cpu-int8] pre-processing graph + onnxruntime.quantization.quantize_dynamic "
          "(QInt8 weights)")
    export_path_used = "project exporter (edgevision.compile.onnx_export.export_to_onnx)"
    try:
        _preprocess_and_quantize(raw_path)
    except Exception as exc:  # noqa: BLE001 — robust fallback, reported transparently
        print(f"[cpu-int8] quant on default-exporter graph failed ({type(exc).__name__}); "
              f"re-exporting via torch legacy TorchScript exporter (dynamo=False)")
        import torch

        model.eval()
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, str(raw_path),
                opset_version=17,
                input_names=["images"],
                output_names=["logits", "pred_boxes"],
                dynamic_axes={n: {0: "batch"} for n in ("images", "logits", "pred_boxes")},
                do_constant_folding=True,
                dynamo=False,
            )
        raw_info = verify_onnx(raw_path)
        export_path_used = "torch legacy TorchScript exporter (dynamo=False, opset 17)"
        _preprocess_and_quantize(raw_path)

    info = verify_onnx(fp32_path)   # metadata of the pre-processed FP32 graph
    verify_onnx(int8_path)          # ONNX checker passes on the quantized graph
    print(f"[cpu-int8] export path used for quantization: {export_path_used}")

    # 3. File sizes (pre-processed FP32 vs INT8 — apples-to-apples, same graph).
    fp32_bytes = fp32_path.stat().st_size
    int8_bytes = int8_path.stat().st_size
    size_reduction_pct = (1.0 - int8_bytes / fp32_bytes) * 100.0 if fp32_bytes else 0.0
    print(f"[cpu-int8] size: fp32={fp32_bytes/1024:.1f} KiB  "
          f"int8={int8_bytes/1024:.1f} KiB  reduction={size_reduction_pct:.1f}%")

    # 4. Latency — explicit CPUExecutionProvider, single fixed image.
    x = np.random.default_rng(args.seed).standard_normal(
        (1, 3, args.height, args.width)
    ).astype(np.float32)

    def make_exec(path: Path) -> OnnxRuntimeCPUExecutor:
        return OnnxRuntimeCPUExecutor(
            path,
            num_threads=args.intra_op_threads,
            graph_optimization=args.graph_opt,
        )

    ex_fp32 = make_exec(fp32_path)
    ex_int8 = make_exec(int8_path)
    assert "CPUExecutionProvider" in ex_fp32.describe()["providers"], "expected CPU EP"

    print(f"[cpu-int8] timing fp32: warmup={args.n_warmup} runs={args.n_runs}")
    fp32_samples = _time_fn(ex_fp32.make_callable(x), n_runs=args.n_runs, n_warmup=args.n_warmup)
    print(f"[cpu-int8] timing int8: warmup={args.n_warmup} runs={args.n_runs}")
    int8_samples = _time_fn(ex_int8.make_callable(x), n_runs=args.n_runs, n_warmup=args.n_warmup)

    fp32_summary = _summary(fp32_samples)
    int8_summary = _summary(int8_samples)

    # 5. Bootstrap CIs from the raw samples we just produced.
    print(f"[cpu-int8] bootstrapping {args.n_boot} replicates for 95% CIs")
    fp32_ci = _ci_block(fp32_samples, n_boot=args.n_boot, alpha=args.alpha, rng=rng)
    int8_ci = _ci_block(int8_samples, n_boot=args.n_boot, alpha=args.alpha, rng=rng)
    speedup = _speedup_ci(fp32_samples, int8_samples, n_boot=args.n_boot, alpha=args.alpha, rng=rng)

    print(f"[cpu-int8] fp32 p50={fp32_summary['p50_ms']}ms  int8 p50={int8_summary['p50_ms']}ms  "
          f"speedup={speedup['point']}x  (95% CI [{speedup['lo']}, {speedup['hi']}])")

    # ----------------------------------------------------------------- write outputs
    ort_version = __import__("onnxruntime").__version__
    result = {
        "experiment": "phase3_cpu_int8",
        "lane": "CPU (ONNX Runtime, CPUExecutionProvider) — GPU-free slice of the Pareto",
        "quantization": "onnxruntime.quantization.quantize_dynamic (QInt8 weights, dynamic activations)",
        "generated_utc": datetime.now(UTC).isoformat(),
        "model": {
            "source": "edgevision.models.tiny_model.make_tiny_model (RT-DETR-shaped CI stand-in)",
            "export_path": export_path_used,
            "note": "Tiny detector used so the CPU lane runs with no 120MB HF download; "
                    "shape (logits + pred_boxes), not a real-mAP detector.",
            "num_classes": args.num_classes,
            "num_queries": args.num_queries,
            "input_shape": [1, 3, args.height, args.width],
            "opset": info.opset,
            "n_initializers": info.n_initializers,
        },
        "size": {
            "fp32_bytes": fp32_bytes,
            "int8_bytes": int8_bytes,
            "fp32_kib": round(fp32_bytes / 1024, 2),
            "int8_kib": round(int8_bytes / 1024, 2),
            "size_reduction_pct": round(size_reduction_pct, 2),
        },
        "latency": {
            "provider": "CPUExecutionProvider",
            "intra_op_threads": args.intra_op_threads,
            "graph_optimization": args.graph_opt,
            "n_warmup": args.n_warmup,
            "n_runs": args.n_runs,
            "fp32": fp32_summary,
            "int8": int8_summary,
        },
        "bootstrap_ci": {
            "n_boot": args.n_boot,
            "alpha": args.alpha,
            "confidence": f"{int((1 - args.alpha) * 100)}%",
            "fp32": fp32_ci,
            "int8": int8_ci,
            "median_speedup_fp32_over_int8": speedup,
        },
        "raw_samples_ms": {
            "fp32": [round(s, 6) for s in fp32_samples],
            "int8": [round(s, 6) for s in int8_samples],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "onnxruntime": ort_version,
            "onnx": __import__("onnx").__version__,
            "numpy": np.__version__,
            "available_providers": __import__("onnxruntime").get_available_providers(),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "phase3_cpu_int8.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[cpu-int8] wrote {json_path}")

    md_path = args.out_dir / "phase3_cpu_int8.md"
    md_path.write_text(_render_md(result))
    print(f"[cpu-int8] wrote {md_path}")
    return 0


def _render_md(r: dict) -> str:
    s, lat, ci = r["size"], r["latency"], r["bootstrap_ci"]
    fp32, int8 = lat["fp32"], lat["int8"]
    fci, ici = ci["fp32"], ci["int8"]
    spd = ci["median_speedup_fp32_over_int8"]
    env = r["environment"]
    conf = ci["confidence"]

    def row(name: str, sm: dict, cb: dict) -> str:
        return (
            f"| {name} | {sm['p50_ms']} [{cb['p50_ms']['lo']}, {cb['p50_ms']['hi']}] "
            f"| {sm['p95_ms']} [{cb['p95_ms']['lo']}, {cb['p95_ms']['hi']}] "
            f"| {sm['p99_ms']} [{cb['p99_ms']['lo']}, {cb['p99_ms']['hi']}] "
            f"| {sm['mean_ms']} +/- {sm['std_ms']} | {sm['fps_mean']} |"
        )

    return f"""# Phase 3 — CPU INT8 lane (measured, no GPU)

> **Status:** MEASURED on CPU. This is the GPU-free slice of the Pareto frontier
> — real ONNX Runtime INT8 size + CPU latency, produced by
> [`scripts/bench_cpu_int8.py`](../../scripts/bench_cpu_int8.py). The
> TensorRT latency and NVML watts/frame rows remain **GPU-pending**
> (see [`NEXT_STEPS.md`](../../NEXT_STEPS.md)); nothing here implies those.

## What was measured

- **Model:** `{r['model']['source']}` — an RT-DETR-shaped tiny detector
  (input `{r['model']['input_shape']}`, opset {r['model']['opset']},
  {r['model']['n_initializers']} initializers). Used so the CPU lane runs with
  **no 120 MB HF download**; it is a shape-faithful stand-in, *not* a real-mAP
  detector, so no accuracy number is claimed here.
- **Quantization:** `{r['quantization']}` — a CPU-only operation (weights ->
  int8, activation scales computed at runtime; no calibration set needed).
- **Inference:** `{lat['provider']}` (explicit), intra-op threads =
  {lat['intra_op_threads']}, graph optimization = `{lat['graph_optimization']}`,
  {lat['n_warmup']} warmup + **{lat['n_runs']} timed** single-image forward
  passes per model.

## Model size: FP32 -> INT8 (dynamic)

| Model | Size |
|---|---|
| FP32 ONNX | {s['fp32_kib']} KiB ({s['fp32_bytes']:,} bytes) |
| INT8 ONNX (dynamic) | {s['int8_kib']} KiB ({s['int8_bytes']:,} bytes) |
| **Reduction** | **{s['size_reduction_pct']}%** |

## CPU latency (single image, {conf} bootstrap CIs in brackets)

Bracketed ranges are percentile-bootstrap {conf} CIs ({ci['n_boot']:,} replicates)
computed from the {lat['n_runs']} raw per-run samples this run generated (all
samples are checked into `phase3_cpu_int8.json`).

| Model | p50 ms [{conf} CI] | p95 ms [{conf} CI] | p99 ms [{conf} CI] | mean +/- std ms | FPS (mean) |
|---|---|---|---|---|---|
{row("FP32", fp32, fci)}
{row("INT8 (dynamic)", int8, ici)}

**Median-latency speed-up (FP32 / INT8): {spd['point']}x**
({conf} CI [{spd['lo']}, {spd['hi']}]).

> Reading the speed-up honestly: dynamic INT8 on a small CPU graph trades int8
> weight storage (the real, portable {s['size_reduction_pct']}% size win) against
> per-op quantize/dequantize overhead. On a graph this small the latency change
> can be modest or even negative; the **size reduction is the robust CPU-lane
> result**, and both numbers are reported exactly as measured. On larger,
> compute-bound graphs INT8 latency gains are typically larger — but that is not
> claimed here, only what was measured on this model.

## Environment

- Python {env['python']} · {env['platform']}
- processor: `{env['processor']}`
- onnxruntime {env['onnxruntime']} · onnx {env['onnx']} · numpy {env['numpy']}
- available providers: {env['available_providers']}
- generated (UTC): {r['generated_utc']}

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install onnx onnxruntime "numpy<2" sympy torch   # CPU-only; no CUDA needed
pip install -e . --no-deps
python scripts/bench_cpu_int8.py --n-runs {lat['n_runs']} --n-warmup {lat['n_warmup']}
```

## What this is NOT

- ❌ Not a TensorRT or GPU result. The `trt-*` rows and NVML watts/frame stay
  **GPU-pending** ([`NEXT_STEPS.md`](../../NEXT_STEPS.md)).
- ❌ Not an mAP claim. The tiny stand-in model has no meaningful accuracy; the
  real INT8-accuracy row (RQ-E1) needs the COCO calibration run on a GPU via the
  static-QDQ path (`edgevision.quantization.onnx_qdq.quantize_static`).
- ✅ Is a real, reproducible measurement of the **CPU INT8 size reduction and
  CPU inference latency** — the part of the Pareto that needs no GPU.
"""


if __name__ == "__main__":
    raise SystemExit(main())
