# Research Questions — edge-vision

Five research questions, each producing one row in the final `EVALUATION_REPORT.md`. RQ-E4 is the project's headline result.

---

## RQ-E1 — Accuracy retention under post-training quantization

**Question.** How does mAP@COCO degrade FP32 → FP16 → INT8 (per-tensor vs per-channel) for RT-DETR-R50?

**Why it matters.** PTQ is the cheapest deployment lever — no retraining required. The interesting finding is *which classes break first* (small / rare classes are typically the casualties), not the overall mAP delta.

**Method.** TRT INT8 PTQ with entropy calibrator using a 100–500 image stratified COCO subset. ONNX QDQ static quantization for the CPU path. Both per-tensor and per-channel scales for INT8.

**Metrics.** mAP@[0.5:0.95] (overall + per-class), inference FPS, model size in MB.

**Deliverable.** `docs/results/phase3_quantization.md` table comparing FP32 / FP16 / INT8(PT) / INT8(PC) on (mAP, FPS, MB).

**Hypothesis.** FP16 retains mAP within 0.2 pp of FP32. INT8 per-channel retains within 1 pp; INT8 per-tensor degrades 2–4 pp on classes with few training examples.

---

## RQ-E2 — Distillation gain at fixed latency

**Question.** At a fixed p95 budget on RTX 5080 (target: 8 ms), can a distilled RT-DETR-R18 student beat a quantized RT-DETR-R50 teacher on mAP?

**Why it matters.** This is the right framing of the model-size-vs-quantization tradeoff. "Bigger model + INT8" vs "smaller model + FP16" — at the same latency budget, which wins?

**Method.** Train R50 (FP16 teacher) → R18 (FP16 student) with feature-map MSE + logit KL + Hungarian-matched query distillation on COCO train2017 for ~50 epochs. Compare against the R50-INT8 engine at matched p95 latency.

**Metrics.** mAP@[0.5:0.95], p95 latency on RTX 5080, model size in MB.

**Deliverable.** `docs/results/phase4_distillation.md` table.

**Hypothesis.** Distilled student R18-FP16 ties teacher R50-INT8 within ±1 pp mAP at half the latency. Pareto-frontier story holds *or* the null-result writeup is the value (which classes does the student lose?).

---

## RQ-E3 — Compile-pipeline speed-up

**Question.** PyTorch eager → ONNX → TensorRT INT8 — what's the per-stage latency, FPS, and memory footprint? How does ONNX-CPU compare?

**Why it matters.** The compile pipeline is what separates "research code" from "production". This RQ quantifies the value of each step instead of treating them as a black-box "make it fast."

**Method.** Same RT-DETR-R50 weights, four backends: PyTorch eager (FP32), ONNX-CPU (FP32 + INT8), TRT-FP32, TRT-FP16, TRT-INT8. 100-image batch=1 latency sweep with CUDA events on GPU and `time.perf_counter` on CPU.

**Metrics.** p50 / p95 / p99 latency, FPS, peak VRAM, peak RSS.

**Deliverable.** `docs/results/phase2_compile.md` (Phase 2) extended with INT8 row from Phase 3.

**Hypothesis.** TRT-FP16 ≥ 3× faster than PyTorch eager. TRT-INT8 ≥ 1.5× TRT-FP16. ONNX-CPU sits at ~5× slower than TRT-FP16 but is still serviceable for the HF Space demo.

---

## RQ-E4 — Power-latency frontier (HEADLINE)

**Question.** What is the (mAP × p95-latency × watts/frame) Pareto frontier across {FP32, FP16, INT8, distilled-FP16, distilled-INT8, ONNX-CPU-FP32, ONNX-CPU-INT8}? Where does sustained 15-min load trigger thermal throttling?

**Why it matters.** This is the artifact that separates a portfolio project from a real-engineering writeup. *"Watts per frame"* isn't on most ML candidates' radar; reporting it gets recruiter attention.

**Method.** 15-min sustained workload per config, NVML 100 ms power samples, GPU temperature + clock + throttle-reason flags logged. CPU configs use `psutil` for CPU/RAM/power-via-RAPL where available.

**Metrics.** Mean watts/frame, peak watts, p95 latency, mAP, throttle-event count.

**Deliverable.** `docs/results/phase5_power.md` — the Pareto plot. **The figure that goes in interviews.**

**Hypothesis.** Distilled-INT8 on RTX 5080 wins on watts/frame. The teacher FP32 hits a thermal-throttle cliff after ~8 minutes of sustained load.

---

## RQ-E5 — Cost of bolting MobileSAM onto the detector

**Question.** Adding MobileSAM after the detector (using bbox prompts) — what does it cost in latency and watts/frame? Does the joint pipeline still hit ≥30 FPS on RTX 5080?

**Why it matters.** "Detect + segment" is the standard demo combo. Reporting the joint cost honestly (rather than each stage in isolation) is what makes the Pareto frontier credible.

**Method.** Run RT-DETR-R50-INT8 → MobileSAM-FP16 with image-encoder cached per-frame and decoder run per-bbox. Compare to detector-only baseline.

**Metrics.** Joint p95 latency, joint FPS, watts/frame, mIoU on COCO segmentation labels (instance masks).

**Deliverable.** `docs/results/phase6_segmentation.md` — RQ-E5 cost-of-segmentation row.

**Hypothesis.** Joint pipeline at 25–35 FPS on RTX 5080. CPU fallback drops to ~3 FPS — that's fine, the GPU number is the headline.

---

## How RQs connect to phases

| Phase | RQs answered |
|---|---|
| Phase 1 — Baseline reproduction | (precondition for all) |
| Phase 2 — ONNX + TRT export | RQ-E3 (FP32/FP16 rows) |
| Phase 3 — Quantization | RQ-E1, completes RQ-E3 (INT8 row) |
| Phase 4 — Distillation + Pruning | RQ-E2 |
| Phase 5 — Power + Thermal Profiling | **RQ-E4** |
| Phase 6 — MobileSAM + Demo | RQ-E5, consolidates RQ-E1..E5 in `EVALUATION_REPORT.md` |
