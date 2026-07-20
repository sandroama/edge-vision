# Development — edge-vision

Contributor guide: project layout, how to run tests/lint with the project venv,
the pytest markers, and the CPU-lane-vs-GPU-lane testing philosophy. The
user-facing "how to run it" page is [USAGE.md](USAGE.md).

---

## Project layout

```
edge-vision/
├── src/edgevision/            # the installed package (src layout)
│   ├── cli/                   # console-script entry points (edgevision-smoke)
│   ├── data/                  # coco_loader, preprocessor
│   ├── models/                # rtdetr_wrapper, tiny_model
│   ├── distillation/          # loss, student_train
│   ├── pruning/               # structured_prune
│   ├── quantization/          # calib_dataset, onnx_qdq, trt_int8
│   ├── compile/               # onnx_export, onnxrt_cpu, trt_build, trt_runtime
│   ├── inference/             # latency_harness
│   ├── profiling/             # cpu_profile, nvml_power, thermal_runner
│   ├── evaluation/            # coco_eval, quant_eval, pareto_aggregator
│   ├── dashboard/             # pareto_plot helpers
│   ├── api/                   # FastAPI service (Phase 6 — not yet implemented)
│   └── schemas.py
├── scripts/                   # runnable CLI smokes + benches (NOT importable pkg)
├── tests/                     # pytest (23 files, conftest.py)
├── dashboard/                 # top-level Streamlit app (pareto_plot.py)
├── hf_space/                  # HF Space placeholder (ONNX-CPU demo target)
├── docs/                      # architecture, research_questions, results/, this file
├── configs/                   # YAML per experiment (Hydra-style)
└── .github/workflows/ci.yml   # ruff check + ruff format --check + pytest (3.11, 3.12)
```

Two important layout facts:

- **`scripts/` is not part of the installed package.** `[tool.setuptools.packages.find]`
  has `where = ["src"]`, so only `edgevision.*` is importable after
  `pip install -e .`. The packaged `edgevision-smoke` command therefore lives in
  `src/edgevision/cli/smoke.py` (reusing the library functions), not in
  `scripts/run_baseline_smoke.py` — so it runs from any working directory.
- **Two dashboard locations.** The runnable Streamlit app is the top-level
  `dashboard/pareto_plot.py`; `src/edgevision/dashboard/` is a thin package
  placeholder.

---

## Environment

Use the **project venv** for everything — the system `python3` on this machine
is a 3.15 alpha and is not supported.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # core + pytest, pytest-cov, ruff, mypy
# add ,ui for the Streamlit dashboard; ,gpu / ,trt only on a CUDA box
```

---

## Running tests

```bash
# Fast / CI-equivalent subset — what CI gates on:
python -m pytest -q -m "not gpu and not trt and not slow"
#   => 150 passed, 2 skipped, 30 deselected

# Just drop the slow tests (handy locally):
python -m pytest -q -m "not slow"

# Full suite (includes the slow torch/onnx round-trips):
python -m pytest -q
#   => 183 passed, 8 skipped, 0 failed  (torch 2.12 / onnx 1.21 / ort 1.26, CPU)
```

`make test` (`.venv/bin/python -m pytest tests/ -v`) and `make test-fast`
(`-m "not slow"`) wrap the
common cases.

### Markers

Declared in `pyproject.toml` under `[tool.pytest.ini_options]`:

| Marker | Gates | Notes |
|---|---|---|
| `slow` | torch/onnx-heavy tests (real export, KD loop, pruning) | deselected by CI; run them locally with the full `pytest -q` |
| `gpu` | tests needing a CUDA GPU | always skipped on CPU |
| `trt` | tests needing TensorRT | always skipped without the `trt` extra |

CI runs `-m "not gpu and not trt and not slow"`, which is why the slow tests do
not affect the CI count.

### Known torch≥2.12 toolchain drift (version-guarded)

Three `slow` tests are `@pytest.mark.skipif(_TORCH_GE_212, ...)` because torch
2.12 changed upstream behavior (not project-logic bugs); they pass on the
project's targeted `torch>=2.4` floor and **skip cleanly** on torch≥2.12:

- `tests/test_onnx_export.py::test_export_uses_pinned_default_opset` — torch
  2.12's dynamo exporter forces ONNX opset 18 over the pinned 17.
- `tests/test_distillation.py::test_apply_pruning_increases_sparsity` and
  `::test_apply_then_remove_pruning_clean` — torch 2.12's
  `nn.utils.prune.l1_unstructured` zeros 0 weights on the tiny CI `Conv2d`
  layers at this amount, so sparsity is 0.0 and the follow-up `prune.remove`
  errors.

The assertions are intentionally left intact (only skipped on the drift
versions) so the checks still hold once the toolchain settles.

---

## Lint & format

CI runs both of these over `src tests scripts dashboard`:

```bash
ruff check src tests scripts dashboard        # or: make lint
ruff format --check src tests scripts dashboard
ruff format src tests scripts dashboard       # apply formatting; or: make format
```

`ruff check` is clean. A few intentional naming choices carry inline
`# noqa: N8xx` with a reason — chiefly `mAP*` (the universal COCO metric
spelling, and a serialized field name in `docs/results/*.json`), `T` (the
KD temperature symbol), and `F` for `torch.nn.functional`. Prefer a justified
`# noqa` over renaming a public/serialized symbol.

> **Toolchain note.** `ruff>=0.6` is unpinned in `[dev]`; very recent ruff
> (0.15.x) reformats some line-wrapping/comment alignment differently from the
> version the repo was first formatted with, so `ruff format --check` may report
> pre-existing cosmetic drift in files untouched by your change. Run
> `ruff format` (or pin ruff) before relying on the format-check gate.

Type-check with `make typecheck` (`mypy src`); mypy is configured leniently
(`disallow_untyped_defs = false`).

---

## Testing philosophy: CPU smoke for everything

Every module has a CPU-runnable smoke that CI exercises with mock/synthetic
inputs — distillation, quantization, compilation, profiling, the Pareto
aggregator, the dashboard. The *headline* numbers (TensorRT latency, NVML
watts/frame, real COCO mAP) require a GPU and are deliberately kept **pending**,
with the cold-start runbook in [NEXT_STEPS.md](../NEXT_STEPS.md). GPU-only code
paths are gated behind `pytest -m gpu` / `-m trt` and a clear runtime
`NotImplementedError` rather than silently faked. The two `scripts/*.py` `TODO`
markers (`run_distill_full.py`, `run_power_sweep.py`) are intentional
GPU-wiring placeholders, cross-referenced in `BUILD_PLAN.md`.

---

## Adding a test

- Put real assertions on observable behavior; if a test needs torch/onnx, guard
  with `pytest.importorskip("torch")` (no need to bind the result if you do not
  use the module directly).
- Mark anything that loads a real model, exports ONNX, or runs a training loop
  as `@pytest.mark.slow`; GPU/TensorRT-only as `@pytest.mark.gpu` / `@pytest.mark.trt`.
- Keep results numbers in `docs/results/*` script-generated — never hand-typed.
