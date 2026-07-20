# Next Steps — edge-vision

This project is **code-complete through Phases 0–5** (191 test functions; 183
pass / 8 skip on CPU). What's missing is a single block of **GPU wall-clock time** to turn the
wired pipelines into the measured headline tables. This file is the cold-start
runbook to do exactly that.

---

## What is already measured on CPU (no GPU needed)

- **INT8 axis:** `scripts/bench_cpu_int8.py` → [`docs/results/phase3_cpu_int8.md`](docs/results/phase3_cpu_int8.md)
  — −72.5% model size; INT8 ~3× *slower* on this small CPU graph (honest negative result).
- **Pruning axis:** `scripts/bench_cpu_pruning.py` → [`docs/results/phase4_cpu_pruning.md`](docs/results/phase4_cpu_pruning.md)
  — L1-mask sparsity sweep {20/40/60/80%}: raw ONNX size and dense CPU latency **unchanged**
  (masks ≠ channel removal — measured null result); gzip-9 size −69% at 80% sparsity;
  output fidelity (proxy, not accuracy) degrades cosine 0.9987 → 0.9223; pruned+INT8 keeps
  the −72.5% size cut. **Channel-removal lever now built and measured:**
  `pruning/structured_prune.py::channel_prune_conv_chain` truly removes channels —
  raw ONNX −78.8% and latency 0.545× vs FP32 at 80% removal, fidelity cost measured
  (cosine 0.9014, no retraining); structured 40%+INT8 compounds to −82.4% size.
  Remaining pruning stretch: dependency-graph pruning (torch-pruning) for branching
  architectures like RT-DETR proper, plus retrain-after-prune (needs the GPU lane).

---

## Current blocker

**No local RTX-class GPU during this work session.** Every CPU-runnable subset is
green (export → ONNX Runtime CPU → latency harness, ONNX QDQ, all the mock/
synthetic smokes, the Pareto aggregator, the dashboard). But the *headline*
numbers — the ones recruiters actually read — all require a CUDA GPU with
TensorRT and NVML:

- **RQ-E1** (PTQ accuracy retention) → needs TRT INT8 build + real COCO mAP.
- **RQ-E3** (compile-pipeline speed-up) → needs TRT FP16/INT8 latency on-device.
- **RQ-E2** (distillation gain) → needs a ~50-epoch COCO train2017 run.
- **RQ-E4** (power-latency frontier, **the headline plot**) → needs NVML watts/frame
  over a 15-min sustained load. **NVML only exists on NVIDIA GPUs.**

The target hardware is an **RTX 5080 (Blackwell, sm_100)**; any RTX-class card
(4090 / A100 / L40S) fills the same tables. The plan below assumes a rented
cloud GPU so this is unblockable today without buying hardware.

---

## Unblock plan

### Option A — Rent a cloud GPU (recommended, unblock today)

**Pick a box.** RunPod or Vast.ai, on-demand, CUDA 12.x + TensorRT 10.x:

| Provider | GPU | ~$/hr | Why |
|---|---|---|---|
| RunPod | RTX 4090 (24 GB) | ~$0.45–0.70 | Cheapest path to all of RQ-E1/E3/E4; Ada, not Blackwell, but TRT 10.x identical API |
| RunPod / Lambda | A100 (40/80 GB) | ~$1.10–1.80 | Use for the Phase-4 ~50-epoch distillation run (RQ-E2) where VRAM + throughput matter |
| Vast.ai | RTX 5080 (16 GB) | spot-priced | Only if you specifically want Blackwell sm_100 numbers to match the README hardware claim |

Use a PyTorch + CUDA base image (e.g. RunPod `runpod/pytorch:2.4.0-py3.11-cuda12.4`).

**Setup (once, on the box):**

```bash
git clone <this repo> && cd edge-vision
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gpu,trt]"           # torch CUDA wheels assumed in the base image
python -c "import tensorrt; print(tensorrt.__version__)"   # verify TRT is importable
python -c "import pynvml; pynvml.nvmlInit(); print('NVML OK')"

# COCO val2017 (~1 GB) + annotations — needed for real mAP + INT8 calibration
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip && unzip -q annotations_trainval2017.zip && cd -
```

**Run the headline pipeline (RQ-E1, E3, E4 — the most bang for the buck):**

```bash
# 1. Baseline: real RT-DETR-R50 on COCO val2017 (HF weights auto-download) -> RQ-E1/E3 baseline row
python scripts/run_baseline_smoke.py \
    --backend rtdetr \
    --coco-annotations data/coco/annotations/instances_val2017.json \
    --coco-images data/coco/val2017 \
    --max-images 5000 --eval-backend pycocotools

# 2. Compile: torch -> ONNX -> TRT (FP16) and verify parity
python scripts/run_compile_smoke.py --stage all --precision fp16

# 3. Quantize: ONNX QDQ + TRT INT8 (per-channel) + per-class drop table  -> RQ-E1
python scripts/run_quant_smoke.py \
    --coco-annotations data/coco/annotations/instances_val2017.json \
    --coco-images data/coco/val2017 \
    --max-images 500 --eval-backend pycocotools \
    --out-json docs/results/phase3_quantization.json

# 4. Latency sweep across all backends                                    -> RQ-E3
python scripts/run_latency_sweep.py \
    --backends torch-cpu torch-cuda onnxrt-cpu trt-fp32 trt-fp16 \
    --model rtdetr --n-runs 100 --n-warmup 20 \
    --out-json docs/results/phase2_latency.json

# 5. Power + thermal sweep: bind every engine and its own COCO metrics      -> RQ-E4
python scripts/run_power_sweep.py \
    --configs trt-fp32 trt-fp16 trt-int8 \
    --artifact trt-fp32=checkpoints/rtdetr_r50_fp32.engine \
    --artifact trt-fp16=checkpoints/rtdetr_r50_fp16.engine \
    --artifact trt-int8=checkpoints/rtdetr_r50_int8.engine \
    --metrics trt-fp32=docs/results/metrics_trt_fp32.json \
    --metrics trt-fp16=docs/results/metrics_trt_fp16.json \
    --metrics trt-int8=docs/results/metrics_trt_int8.json \
    --duration-sec 900 --sample-ms 100 \
    --out-json docs/results/phase5_power.json

# CPU dispatch/latency validation is separate: NVML cannot measure CPU watts.
# --mock-power is explicit and the resulting row is excluded from Pareto.
python scripts/run_power_sweep.py \
    --configs onnxrt-cpu \
    --artifact onnxrt-cpu=checkpoints/rtdetr_r50.onnx \
    --metrics onnxrt-cpu=docs/results/metrics_onnxrt_cpu.json \
    --mock-power --duration-sec 10 \
    --out-json docs/results/phase5_onnxrt_dispatch.json

# 6. Eyeball the Pareto frontier
make ui     # streamlit run dashboard/pareto_plot.py
```

> Equivalent Make shortcuts exist for the common steps: `make smoke`,
> `make export-onnx`, `make build-trt`, `make bench`, `make power-sweep`.

**Backend dispatch is now fail-closed.** `run_power_sweep.py` loads real ONNX
Runtime CPU sessions and TensorRT engines, but it refuses to start unless every
real label has both `--artifact LABEL=PATH` and
`--metrics LABEL=PATH[#selector]`. The selected metrics object must identify
`backend="pycocotools"` and contain numeric `mAP_50_95`; the simple/mock
fallback is rejected. Mock and CPU mock-power rows write `mAP=null` or
`power_measured=false` as appropriate and never enter the Pareto frontier.

**Cost & time (Option A, RQ-E1/E3/E4 only):**
- Steps 1–6 wall-clock: ~1.5–2.5 hr on a 4090 (most of it is the 15-min × N-config power sweep + COCO mAP).
- Cost: **~$1–3** on a 4090. Tear the pod down immediately after `scp`-ing the JSON back.

### Option B — Add the distillation run (RQ-E2)

Needs COCO **train2017** (~18 GB) and a longer GPU hold.

```bash
# Implement _make_dataloader in scripts/run_distill_full.py first
# (CocoDataset + RTDetrImageProcessor -> torch DataLoader; this is the remaining TODO).
python scripts/run_distill_full.py \
    --coco-annotations data/coco/annotations/instances_train2017.json \
    --coco-images data/coco/train2017 \
    --epochs 50 --lr 2e-4 --temperature 4.0 --alpha 0.7 \
    --out-dir checkpoints/distill_full
# then export + bench the student exactly like the teacher (steps 2/4/5 above)
```

**Cost & time (Option B):** ~50 epochs on an A100 is roughly **8–16 GPU-hours**;
**~$10–30**. Do this in a second session after the cheap RQ-E1/E3/E4 numbers are
banked — it's the highest-cost, highest-variance result.

### Option C — Phase 6 breadth (no GPU strictly required for the scaffolding)

Independent of the headline GPU runs, the next *breadth* phase from `BUILD_PLAN.md`:
- `src/edgevision/models/mobilesam_wrapper.py` — MobileSAM with detector-bbox prompts (RQ-E5).
- `src/edgevision/api/main.py` — FastAPI `/v1/detect`, `/v1/segment`, `/health`.
- Flesh out `hf_space/app.py` from placeholder → real ONNX-CPU demo (HF Spaces have no GPU).
- `docs/results/phase6_segmentation.md` — RQ-E5 cost-of-segmentation row.

---

## Expected outputs

| Run | Results file generated | README / report tables it fills |
|---|---|---|
| Step 1 (baseline) | `docs/results/phase1_baseline.md` (real RT-DETR row) | "Real RT-DETR-R50 on COCO val2017" table in `phase1_baseline.md` |
| Step 3 (quantize) | `docs/results/phase3_quantization.json` + `.md` rows | **RQ-E1** table (FP32/FP16/INT8 mAP, retained %, per-class drops) |
| Step 4 (latency) | `docs/results/phase2_latency.json` | **RQ-E3** per-stage latency table (p50/p95/p99, FPS) |
| Step 5 (power) | `docs/results/phase5_power.json` + `phase5_pareto.{md,json}` | **RQ-E4** Pareto table from rows with explicit, measured mAP and NVML power only |
| Option B (distill) | `checkpoints/distill_full/` + `docs/results/phase4_distillation.md` rows | **RQ-E2** distilled-student-vs-quantized-teacher table |
| Option C (Phase 6) | `docs/results/phase6_segmentation.md` | **RQ-E5** joint detect+segment FPS / watts/frame row |

These also replace the *"Further results pending GPU run"* line in `README.md`'s
**Headline results** callout, and let `docs/EVALUATION_REPORT.md` flip each
RQ-E section from *Pending* to a real number.

---

## Definition of done

The "headline GPU milestone" is complete when:

- [ ] Real RT-DETR-R50 COCO val2017 mAP reproduced within ~0.5 pp of the paper (RQ-E3 precondition).
- [ ] **RQ-E1** filled: FP32 / FP16 / INT8(per-channel) mAP + retained-% + per-class drop table, with the script-produced JSON checked in.
- [ ] **RQ-E3** filled: per-backend p50/p95/p99 + FPS table; TRT-FP16 vs PyTorch-eager speed-up quantified.
- [ ] **RQ-E4** filled (**headline**): watts/frame + p95 + throttle-rate across ≥4 configs; Pareto frontier renders in the dashboard with real points.
- [ ] `docs/EVALUATION_REPORT.md` RQ-E1/E3/E4 sections show numbers, not *Pending*.
- [ ] `README.md` **Headline results** callout updated with a real number (and the *pending* line removed once all four are in).
- [ ] Reproducibility checklist in `EVALUATION_REPORT.md` filled (checkpoint hash, ONNX hash, TRT builder version, CUDA/driver version, calibration image IDs, NVML interval).

Stretch (separate sessions): **RQ-E2** distillation table (Option B) and **RQ-E5**
MobileSAM bolt-on + live demo + HF Space (Option C).
