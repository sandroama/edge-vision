# API — edge-vision

> **Status (honest):** there is **no running HTTP service yet**.
> `src/edgevision/api/` contains only a placeholder `__init__.py`; the FastAPI
> app (`api/main.py`) is a **Phase 6** item that has not been implemented. This
> page therefore documents (A) the **programmatic + CLI surface you can use
> today**, and (B) the **planned HTTP contract**, clearly labelled as pending.
> No example request/response bodies are presented as if a server were running.

---

## A. Available today — CLI + library surface

### A.1 Console command

Installed by `pip install -e .` (declared under `[project.scripts]`):

```bash
edgevision-smoke [--max-images N] [--eval-backend {auto,pycocotools,simple}]
                 [--mock-recall R] [--mock-fp-rate R] [--seed S]
```

Runs the CPU-only mock baseline (synthetic COCO dataset → `MockRTDetrDetector`
→ COCO evaluator → summary table) from any working directory. Exits 0.
Implemented in `edgevision.cli.smoke:main`.

### A.2 Runnable scripts

All under `scripts/` and CPU-runnable (the GPU stages raise a clear
`NotImplementedError` / skip on CPU). Run any with `--help` for the full flag
list; the load-bearing flags:

| Script | Key flags | Purpose |
|---|---|---|
| `run_baseline_smoke.py` | `--backend {mock,rtdetr}`, `--coco-annotations`, `--coco-images`, `--max-images`, `--eval-backend {auto,pycocotools,simple}`, `--mock-recall`, `--mock-fp-rate` | Phase-1 detection baseline (mock on CPU; `rtdetr` needs GPU + COCO) |
| `bench_cpu_int8.py` | `--n-runs`, `--n-warmup`, `--n-boot`, `--alpha`, `--intra-op-threads`, `--graph-opt {off,basic,extended,all}`, `--height`, `--width`, `--out-dir` | **Measured** CPU INT8 size + latency + bootstrap CIs → `docs/results/phase3_cpu_int8.{json,md}` |
| `run_compile_smoke.py` | `--stage {onnx,onnxrt,trt,all}`, `--precision {fp32,fp16}`, `--opset`, `--no-dynamic-batch` | torch → ONNX → ONNX-Runtime CPU round-trip (`trt` is GPU-gated) |
| `run_quant_smoke.py` | `--coco-annotations`, `--coco-images`, `--max-images`, `--eval-backend`, `--candidate-recall`, `--candidate-fp-rate`, `--out-json` | Quantization-eval smoke → `docs/results/phase3_quantization.json` |
| `run_distill_smoke.py` | `--backend {tiny,rtdetr}`, `--epochs`, `--batches`, `--temperature`, `--alpha`, `--lr`, `--out-json` | KD smoke (`tiny` on CPU; `rtdetr` needs GPU) |
| `run_latency_sweep.py` | `--num-images`, backend selection | Multi-backend latency bench (GPU backends pending) |
| `run_power_sweep.py` | `--configs ...`, `--duration-sec`, `--sample-ms`, `--mock-power`, `--out-json`, `--mAP-fp32` | Power/thermal sweep (`mock-*` configs on CPU; real configs need NVML/GPU) |

### A.3 Library entry points

Importable from the installed `edgevision` package. Signatures below are the
real current ones.

```python
# Compile: torch -> ONNX, and read back graph metadata
from edgevision.compile import export_to_onnx, verify_onnx, OnnxModelInfo, DEFAULT_OPSET
export_to_onnx(model, dummy_input, output_path, *, opset=17, dynamic_batch=True,
               input_names=("images",), output_names=("logits", "pred_boxes"),
               do_constant_folding=True) -> Path
verify_onnx(onnx_path: str | Path) -> OnnxModelInfo

# Run ONNX on CPU (ONNX Runtime, CPUExecutionProvider)
from edgevision.compile import OnnxRuntimeCPUExecutor
OnnxRuntimeCPUExecutor(onnx_path, *, num_threads=None, input_name=None,
                       output_names=None, graph_optimization="basic")

# Static ONNX QDQ quantization (per-channel QInt8 by default)
from edgevision.quantization.onnx_qdq import quantize_static, QDQQuantizationConfig
quantize_static(onnx_in, onnx_out, *, provider, input_name="images", config=None)

# Detection wrapper factory (mock by default; "rtdetr" lazy-loads HF + needs GPU)
from edgevision.models.rtdetr_wrapper import make_detector
make_detector(backend="mock", *, gt_by_image_id=None,
              model_name="PekingU/rtdetr_r50vd_coco_o365", device="auto",
              confidence_threshold=0.3)

# Latency harness (auto-dispatch CPU perf_counter / CUDA events)
from edgevision.inference.latency_harness import measure_latency, LatencyResult

# Pareto aggregation: JSON rows in, frontier + report out
from edgevision.evaluation.pareto_aggregator import (
    pareto_frontier, dominates, write_report, load_configs_from_jsons,
)
write_report(configs: list[ParetoConfig], out_dir="docs/results") -> Path
```

See [USAGE.md](USAGE.md) for runnable command sequences and
[architecture.md](architecture.md) for the module map.

---

## B. Planned HTTP API (Phase 6 — NOT yet implemented)

The following contract is defined in [BUILD_PLAN.md](../BUILD_PLAN.md) (Phase 6)
and is **not runnable today**. `Makefile`'s `make api` target points at
`edgevision.api.main:app`, which does not exist yet; it will error until the
service is implemented.

Planned endpoints (FastAPI, contract only — no responses are claimed as real):

| Method | Path | Planned purpose |
|---|---|---|
| `GET`  | `/health` | Liveness/readiness probe |
| `POST` | `/v1/detect` | Run detection on an uploaded image → bounding boxes + scores |
| `POST` | `/v1/segment` | Detect, then prompt MobileSAM with the detector's boxes → masks |
| `POST` | `/v1/explain` | Per-request latency / backend / config metadata for the result |

This depends on two other Phase-6 items also pending:
`src/edgevision/models/mobilesam_wrapper.py` (the segmentation head) and a real
ONNX-CPU demo in `hf_space/app.py`. Tracking and acceptance criteria live in
[NEXT_STEPS.md](../NEXT_STEPS.md) (Option C) and [BUILD_PLAN.md](../BUILD_PLAN.md).

When the service lands, this section will be replaced with the real request and
response schemas (derived from the implemented Pydantic models), not before.
