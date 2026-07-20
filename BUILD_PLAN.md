# Build Plan — edge-vision (Project Alpha-1)

A 7-phase build plan. Each phase ends with a deliverable: a working sub-system **and** at least one row in the final Pareto-frontier table.

Time estimates assume part-time graduate-student pace (~10–15 hr/week). Hardware: AMD Ryzen 9 9950X + RTX 5080 (16 GB Blackwell) + 64 GB RAM.

---

## Phase 0 — Scaffolding ✅ COMPLETE

- [x] Tier README at `Projects_Alpha/README.md`
- [x] Project skeleton (this directory)
- [x] `pyproject.toml` with detection/quant/profiling deps + optional `[gpu]` and `[trt]` extras
- [x] Makefile mirrors `sceneiq`: `make test / smoke / api / ui / export-onnx / build-trt / bench / power-sweep`
- [x] CI workflow (`.github/workflows/ci.yml`) — CPU-only, lint + format + tests on Python 3.11/3.12
- [x] `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, `DEPLOYMENT.md`
- [x] `docs/architecture.md`, `docs/research_questions.md`
- [x] Module skeletons (`src/edgevision/{data,models,distillation,pruning,quantization,compile,inference,profiling,evaluation,dashboard,api}/__init__.py`)
- [x] Empty test scaffold (`tests/test_imports.py`) green at scaffold

**Phase 0 deliverable:** ✅ scaffold tests green.

---

## Phase 1 — Baseline reproduction 🟢 WIRED (CPU smoke green; GPU run pending)

**Goal:** reproduce RT-DETR-R50 published mAP on COCO val2017 within 0.5 pp on the RTX 5080.

- [x] `src/edgevision/schemas.py` — `BoundingBox`, `Detection`, `GroundTruthBox`, `Image`, `ImageDetections`
- [x] `src/edgevision/data/coco_loader.py` — `CocoDataset.from_json` (real) + `CocoDataset.synthetic` (CI-friendly) + `to_coco_dict` round-trip
- [x] `src/edgevision/data/preprocessor.py` — numpy-only letterbox + ImageNet normalize + un-letterbox; `RTDetrImageProcessor` handles the real path inside the wrapper
- [x] `src/edgevision/models/rtdetr_wrapper.py` — `MockRTDetrDetector` (deterministic GT replay) + `RTDetrDetector` (lazy-imported HF transformers) behind a `Detector` `Protocol`
- [x] `src/edgevision/evaluation/coco_eval.py` — pycocotools `COCOeval` backend + dependency-free `evaluate_simple` fallback under one `evaluate(...)` dispatcher
- [x] `scripts/run_baseline_smoke.py` — `--backend mock|rtdetr`, `--eval-backend auto|pycocotools|simple`
- [x] `tests/test_schemas.py`, `tests/test_coco_loader.py`, `tests/test_preprocessor.py`, `tests/test_rtdetr_wrapper.py`, `tests/test_coco_eval.py`, `tests/test_baseline_smoke.py` — **41 tests, all passing on CPU**
- [x] `docs/results/phase1_baseline.md` — sanity-check table; real-RT-DETR row pending GPU run
- [ ] **GPU run with `--backend rtdetr` over real COCO val2017** — blocked on data download + ~10 min RTX-5080 inference

**Deliverable:** ✅ end-to-end harness wired and tested. Real RT-DETR-R50 reproduction on COCO val2017 is one CLI invocation away (see `docs/results/phase1_baseline.md`).

---

## Phase 2 — ONNX + TensorRT export 🟢 WIRED (CPU smoke green; GPU run pending) — RQ-E3

**Goal:** three runnable engines (ONNX-CPU, TRT-FP32, TRT-FP16) producing the same mAP within rounding, with measured per-stage latency.

- [x] `src/edgevision/inference/latency_harness.py` — `LatencyResult` + `measure_latency_cpu` (`time.perf_counter`) + `measure_latency_cuda` (`torch.cuda.Event`) + `measure_latency` dispatcher
- [x] `src/edgevision/models/tiny_model.py` — RT-DETR-shaped tiny detector (logits + pred_boxes) for CI-fast export tests
- [x] `src/edgevision/compile/onnx_export.py` — `torch.onnx.export` opset 17, dynamic batch + `verify_onnx` returning `OnnxModelInfo`
- [x] `src/edgevision/compile/trt_build.py` — `TrtBuildConfig` + `build_engine` (FP32 + FP16); INT8 calibrator hook exposed for Phase 3
- [x] `src/edgevision/compile/onnxrt_cpu.py` — `OnnxRuntimeCPUExecutor` with `make_callable` for the latency harness
- [x] `scripts/run_compile_smoke.py` — `--stage onnx|onnxrt|trt|all` end-to-end pipeline check
- [x] `scripts/run_latency_sweep.py` — multi-backend bench writing JSON for Phase 5's Pareto aggregator
- [x] `tests/test_latency_harness.py`, `tests/test_onnx_export.py`, `tests/test_onnxrt_cpu.py`, `tests/test_trt_build.py`, `tests/test_compile_smoke.py` — **57 tests passing, 6 skipping cleanly when torch/tensorrt absent**
- [x] `docs/results/phase2_compile.md` — module status + reproduction recipe; numbers TBD on GPU run
- [ ] **GPU run** — `pip install -e .[dev,gpu,trt]` then run the export + build + sweep against real RT-DETR-R50

**Deliverable:** ✅ End-to-end compile pipeline ready. The CI-runnable subset (export → ONNX Runtime CPU → latency harness) is green; the TRT FP16/FP32 engines are one CLI call away on the RTX 5080.

---

## Phase 3 — Quantization 🟢 WIRED (CPU smoke green; GPU run pending) — RQ-E1

**Goal:** INT8 engine ≥ 0.95× FP32 mAP at ≥ 1.5× FP16 throughput on RTX 5080. Or an honest writeup of where INT8 broke and why.

- [x] `src/edgevision/quantization/calib_dataset.py` — uniform / stratified / first sampling; `BatchProvider`; synthetic loader for CI
- [x] `src/edgevision/quantization/trt_int8.py` — `IInt8EntropyCalibrator2` wrapper + cache + `build_int8_engine` convenience call
- [x] `src/edgevision/quantization/onnx_qdq.py` — `quantize_static` (ORT QDQ, per-channel QInt8 default) + `QDQQuantizationConfig`
- [x] `src/edgevision/evaluation/quant_eval.py` — `compare_metrics`, `QuantizationDelta`, per-class drop ranking, `summary_table`
- [x] `scripts/run_quant_smoke.py` — CI smoke with mock detectors; real COCO + pycocotools path via `--coco-annotations/--coco-images`
- [x] `tests/test_calib_dataset.py` (16), `tests/test_quant_eval.py` (14), `tests/test_onnx_qdq.py` (3), `tests/test_trt_int8.py` (4), `tests/test_quant_smoke.py` (1)
- [x] `docs/results/phase3_quantization.md` — CI numbers + GPU reproduction recipe; TBD rows for real RTX-5080 numbers
- [ ] **GPU run** — `quantize_static` on real COCO val2017 + `build_int8_engine` + pycocotools mAP for the real RQ-E1 table

**Deliverable:** ✅ Full quantization pipeline end-to-end. Per-class drop analysis identifies which classes degrade first — the actual interview artifact for RQ-E1. GPU table pending.

---

## Phase 4 — Distillation + Pruning 🟢 WIRED (CPU smoke green; GPU run pending) — RQ-E2

**Goal:** distilled R18 student on the Pareto frontier at ≤ 8 ms p95 on RTX 5080. Or a documented null result.

- [x] `src/edgevision/distillation/loss.py` — `LogitKDLoss` (KL+T²), `FeatureKDLoss` (MSE), `CombinedDetectionKDLoss` (α·KL + β·MSE); `KDLossConfig`
- [x] `src/edgevision/distillation/student_train.py` — CPU tiny smoke + full GPU path (HF transformers); `DistillationResult` JSON output
- [x] `src/edgevision/pruning/structured_prune.py` — L1/random *mask* pruning via `torch.nn.utils.prune` (`remove_pruning` for ONNX-ready dense weights) **and** true channel-removal surgery (`channel_prune_conv_chain`, measured in phase4_cpu_pruning.md)
- [x] `scripts/run_distill_smoke.py` — CPU tiny smoke; graceful ImportError on missing torch
- [x] `scripts/run_distill_full.py` — GPU scaffold; DataLoader wiring is the remaining TODO
- [x] `tests/test_distillation.py` (9 config tests + 12 torch-gated), `tests/test_distill_smoke.py` (1 torch-gated)
- [x] `docs/results/phase4_distillation.md` — design rationale + GPU recipe; TBD rows for the real 50-epoch run
- [ ] **GPU run** — implement `_make_dataloader` in `run_distill_full.py`, train ~50 epochs on COCO train2017, export student

**Deliverable:** ✅ Full KD pipeline end-to-end. CPU smoke converges in <2 s. GPU run requires DataLoader wiring + RTX-5080 time.

---

## Phase 5 — Power + Thermal Profiling 🟢 WIRED (mock smoke green; GPU run pending) — RQ-E4

**Goal:** the headline plot — mAP × p95-latency × watts/frame across 7 configs.

- [x] `src/edgevision/profiling/nvml_power.py` — `PowerMonitor` (pynvml, 100 ms samples) + `MockPowerMonitor` (deterministic synthetic) + `power_monitor` context manager
- [x] `src/edgevision/profiling/thermal_runner.py` — `run_sustained` orchestrates N-second inference + NVML sampling; `ThermalRunResult` bundles latency + power profile
- [x] `src/edgevision/profiling/cpu_profile.py` — `CpuProfiler` (psutil CPU% + RSS + TDP-scaled watt estimate) + `cpu_profiler` context manager
- [x] `src/edgevision/evaluation/pareto_aggregator.py` — `dominates` / `is_dominated` / `pareto_frontier` / `summary_table` / `write_report` → `phase5_pareto.md` + `phase5_pareto.json`
- [x] `dashboard/pareto_plot.py` — Streamlit scatter (Plotly), metrics header, full config table; placeholder when no JSON present
- [x] `scripts/run_power_sweep.py` — real ONNX Runtime CPU and TensorRT runtime dispatch; explicit per-label `--artifact` + pycocotools `--metrics` bindings; mock/unmeasured rows excluded from Pareto; real NVML path for GPU
- [x] `src/edgevision/compile/trt_runtime.py` — TensorRT 10/PyCUDA execution context, shape/address binding, synchronized callable, and fail-fast host prerequisite errors
- [x] CPU-safe dispatch tests — nested metrics selectors, simple/mock-metric rejection, missing-artifact errors, ONNX/TRT routing, and no published-mAP reuse in mock mode
- [x] `tests/test_profiling.py` (11/14 run always; 3 psutil-gated), `tests/test_pareto_aggregator.py` (16/16 always)
- [x] `docs/results/phase5_power.md` — CI smoke numbers; GPU reproduction recipe; design notes on throttle detection + watts/frame
- [ ] **GPU run** — build engines, produce a pycocotools metrics JSON for each exact artifact, then run the 15-min bound sweep on RTX-5080

**Deliverable:** ✅ Full profiling + Pareto pipeline wired. Mock smoke confirms the pipeline end-to-end. GPU table pending real engine runs.

---

## Phase 6 — MobileSAM bolt-on + Demo + Final report (weeks 7–8, 10–14 hr) — RQ-E5

**Goal:** live detection + segmentation demo at ≥ 30 FPS on RTX 5080; HF Space deployed; consolidated EVALUATION_REPORT.

- [ ] `src/edgevision/models/mobilesam_wrapper.py` — MobileSAM with detector-bbox prompts; reuse `Segmenter` interface pattern from [sceneiq](https://github.com/sandroama/sceneiq) `perception/segmentation_sam2.py`
- [ ] `src/edgevision/compile/mobilesam_export.py` — ONNX + TRT export for image-encoder + decoder
- [ ] `src/edgevision/api/main.py` — FastAPI `/v1/detect`, `/v1/segment`, `/v1/explain`, `/health`
- [ ] `dashboard/live_demo.py` — Streamlit live-webcam detection + segmentation
- [ ] `hf_space/` — HF Space with ONNX-CPU fallback (HF Spaces have no GPU)
- [ ] 60-second screen recording in `docs/demo.mp4`
- [ ] `tests/test_api.py`, `tests/test_mobilesam_wrapper.py` (CPU mocks)
- [ ] `docs/results/phase6_segmentation.md` — RQ-E5 cost-of-segmentation row
- [ ] `docs/EVALUATION_REPORT.md` — consolidates RQ-E1..E5

**Phase 6 deliverable:** Live demo, HF Space link, final report.

---

## Phase 7 — Stretch (post weeks 8–10, optional)

Stub-only placeholders, no implementation pressure:
- [ ] `scripts/jetson_bench.py` — empty stub for when a Jetson Orin Nano is acquired
- [ ] `src/edgevision/compile/coreml_export.py` — optional CoreML path for "Apple AIML" interview
- [ ] `triton/` — optional Triton Inference Server config for "production serving" interview

---

## Continuous tasks (across all phases)

- [ ] Add tests in `tests/` as you implement each module — don't let them slide
- [ ] Track every benchmark with W&B (tag with the relevant RQ)
- [ ] Keep `docs/EVALUATION_REPORT.md` updated phase-by-phase — don't leave it for the end
- [ ] Push to GitHub at every phase boundary

---

## When you're stuck

- **Blackwell sm_100 + TensorRT Python API mismatch:** TensorRT 10.x supports sm_100; if a wheel's missing, build from `nvcr.io/nvidia/tensorrt` container. Fall back to the FP32 engine + ONNX-CPU path so demos still work.
- **INT8 mAP collapses:** that's a finding — report which classes broke and try per-channel calibration before declaring failure.
- **Distillation not converging:** loss-weighting between feature/logit KD is sensitive. Sweep three settings on a 1-epoch run before committing to the 50-epoch full run.
- **Live demo too slow:** lower the input resolution, drop to FP16 student, or batch frames offscreen. The 30 FPS bar is on RTX 5080; CPU fallback can show ~5 FPS and that's fine in the demo narrative.
- **Power samples noisy:** NVML power is sampled — average over ≥1000 frames for stable numbers; report std-dev too.

---

## What "done" looks like

A `EVALUATION_REPORT.md` with 5 ablation tables (one per RQ-E), the Pareto plot, an honest failure analysis, a 60-second demo video, and a working hosted demo (HF Spaces ONNX-CPU). That's the artifact recruiters click on. That's what gets the interview.
