# Usage — edge-vision

End-to-end "how do I run it" guide for the **CPU lane** (everything below runs on
a laptop, no GPU, no model/dataset download). The headline **GPU lane** —
TensorRT FP16/INT8 latency + NVML watts/frame — is documented separately in
[NEXT_STEPS.md](../NEXT_STEPS.md) and is the only part that needs an NVIDIA GPU.

> **Two lanes, reported separately.** The CPU lane below is *measured today*. GPU
> numbers stay pending an RTX-class run. See the "CPU lane vs GPU lane" note at
> the bottom.

---

## 1. Environment setup

```bash
git clone https://github.com/sandroama/edge-vision.git && cd edge-vision
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest/ruff/mypy; add ,ui for the dashboard
```

- `".[dev]"` installs the CPU runtime (torch, onnx, onnxruntime, numpy, …) plus
  the dev tools. No CUDA/TensorRT wheels are pulled.
- The dashboard needs the `ui` extra: `pip install -e ".[dev,ui]"` (Streamlit + Plotly).
- GPU/TensorRT extras (`gpu`, `trt`) are intentionally **not** installed here —
  see [NEXT_STEPS.md](../NEXT_STEPS.md) for the GPU box recipe.

Verify the install with the packaged console command:

```bash
edgevision-smoke                  # runs the mock baseline from any directory
```

---

## 2. The measured CPU INT8 benchmark (the real number)

This is the project's one fully-measured result: ONNX-Runtime dynamic INT8
model-size reduction + CPU p50/p95/p99 latency with bootstrap 95% CIs. No GPU,
no calibration data, no download (it exports the RT-DETR-shaped tiny CI model).

```bash
python scripts/bench_cpu_int8.py --n-runs 200 --n-warmup 40
```

Outputs (overwritten each run):

- `docs/results/phase3_cpu_int8.json` — full result incl. all `200×2` raw latency
  samples, percentile bootstrap CIs, and the environment fingerprint.
- `docs/results/phase3_cpu_int8.md` — the human-readable table.

Useful flags (all have defaults; see `--help`): `--n-boot` (bootstrap replicates),
`--alpha` (CI level, `0.05` → 95%), `--intra-op-threads` (`1` = single-thread,
comparable across machines), `--graph-opt {off,basic,extended,all}`, `--seed`.

> **Honest result:** on this small CI-stand-in graph, dynamic INT8 is ~3×
> *slower* on CPU (per-op quant/dequant overhead dominates a memory-light graph).
> The robust CPU-lane win is the **−72.54% file-size reduction**; INT8 *latency*
> gains are a GPU/TensorRT story. Reported exactly as measured.

---

## 3. CPU / mock smokes (no GPU)

Each pipeline stage has a CPU-runnable smoke. All of these run today and exit 0.

```bash
# Phase 1 — baseline: synthetic COCO dataset + MockRTDetrDetector + COCO eval
python scripts/run_baseline_smoke.py
#   real RT-DETR over COCO needs a GPU + data:
#   python scripts/run_baseline_smoke.py --backend rtdetr \
#       --coco-annotations data/coco/annotations/instances_val2017.json \
#       --coco-images data/coco/val2017

# Phase 2 — compile: torch -> ONNX (and ONNX Runtime CPU round-trip)
python scripts/run_compile_smoke.py --stage onnx
python scripts/run_compile_smoke.py --stage onnxrt
python scripts/run_compile_smoke.py --stage all      # onnx + onnxrt (trt is GPU-gated)

# Phase 3 — quantization eval smoke (writes docs/results/phase3_quantization.json)
python scripts/run_quant_smoke.py

# Phase 4 — distillation smoke (TinyDetector KD loop on synthetic batches)
python scripts/run_distill_smoke.py --backend tiny --epochs 3 --batches 4

# Phase 5 — power sweep in mock mode (MockPowerMonitor; no NVML/GPU needed)
python scripts/run_power_sweep.py --configs mock-a mock-b --duration-sec 1 --mock-power
```

Notes:

- `run_compile_smoke.py --stage trt` and any non-`mock-*` config in
  `run_power_sweep.py` require a CUDA GPU / TensorRT and raise a clear
  `NotImplementedError`/skip on CPU — that is expected.
- `run_distill_smoke.py --backend rtdetr` needs the real HF models (GPU).

---

## 4. Dashboard

The Streamlit Pareto scatter reads the JSON written by the power sweep /
aggregator. It renders a labelled placeholder when no results JSON is present.

```bash
pip install -e ".[dev,ui]"           # if not already
streamlit run dashboard/pareto_plot.py
```

Equivalent Make shortcut: `make ui`.

---

## 5. Make shortcuts

| Command | What it runs (CPU unless noted) |
|---|---|
| `make test` | `pytest tests/ -v` (full suite) |
| `make test-fast` | `pytest tests/ -v -m "not slow"` (fast subset) |
| `make smoke` | `scripts/run_baseline_smoke.py` (mock baseline) |
| `make export-onnx` | `scripts/run_compile_smoke.py --stage onnx` |
| `make build-trt` | `scripts/run_compile_smoke.py --stage trt` — **GPU** |
| `make bench` | `scripts/run_latency_sweep.py` — backends incl. GPU |
| `make distill` | `scripts/run_distill_smoke.py` (tiny CPU smoke) |
| `make power-sweep` | `scripts/run_power_sweep.py --duration-sec 900` — **GPU/NVML** |
| `make ui` | `streamlit run dashboard/pareto_plot.py` |
| `make lint` / `make format` | `ruff check` / `ruff format` over `src tests scripts dashboard` |

---

## CPU lane vs GPU lane

| Lane | Runs where | Status | Where to look |
|---|---|---|---|
| **CPU lane** | Any laptop, `".[dev]"` only | **Measured today** | this page + [phase3_cpu_int8.md](results/phase3_cpu_int8.md) |
| **GPU lane** | RTX-class GPU + CUDA + TensorRT + NVML | **Pending a GPU run** | [NEXT_STEPS.md](../NEXT_STEPS.md) — cold-start runbook |

The GPU lane fills RQ-E1 (PTQ accuracy retention), RQ-E3 (compile-pipeline
speed-up), RQ-E4 (power-latency frontier — the headline plot), and RQ-E2
(distillation). None of those numbers are claimed until that run lands; the
runbook to produce them is in [NEXT_STEPS.md](../NEXT_STEPS.md).

For the contributor-facing test/lint/marker workflow, see
[DEVELOPMENT.md](DEVELOPMENT.md). For the programmatic / CLI surface and the
planned HTTP API, see [API.md](API.md).
