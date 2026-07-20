# Phase 4 — Distillation + Pruning (results)

> **Status:** Modules wired. CPU-runnable smoke (TinyDetector) verifies the
> loop converges. Real RT-DETR-R50 → R18 GPU training is blocked on the
> COCO train2017 DataLoader wiring + ~50-epoch RTX-5080 run.

## Module status

| Module | File | Tests | Notes |
|---|---|---|---|
| KD losses | `src/edgevision/distillation/loss.py` | 8/8 ✅ (torch-gated) | `LogitKDLoss` (KL+T²), `FeatureKDLoss` (MSE), `CombinedDetectionKDLoss` (α·KL + β·MSE) |
| Student trainer | `src/edgevision/distillation/student_train.py` | 2/2 ✅ (torch-gated) | CPU smoke (TinyDetector), full GPU path (RT-DETR via HF transformers) |
| Structured pruner | `src/edgevision/pruning/structured_prune.py` | 3/3 ✅ (torch-gated) | L1 + random channel pruning via `torch.nn.utils.prune`; `remove_pruning` for ONNX export |
| Config / schema | (both modules) | 9 ✅ (no torch) | `KDLossConfig`, `PruneConfig`, `DistillationConfig`, `DistillationResult` all validate cleanly |
| Distill smoke script | `scripts/run_distill_smoke.py` | 1 ✅ (torch-gated) | `--backend tiny` (CPU) / `rtdetr` (GPU) |
| Full distill script | `scripts/run_distill_full.py` | — | Scaffold for the ~50-epoch GPU run; DataLoader wiring is the TODO |

**Running totals: 103 tests passing, 23 skipping cleanly, 0 failing.**

## CPU smoke (TinyDetector, no COCO data needed)

Exercises the full training loop with a 2-layer conv model and synthetic
inputs. Expected output (with `--lr 1e-2 --epochs 3 --batches 4`):

```
[edge-vision] backend=tiny  epochs=3  T=4.0  α=0.7
  epoch  0  loss=X.XXXX
  epoch  1  loss=Y.YYYY   # should be < X
  epoch  2  loss=Z.ZZZZ   # should be < Y
  Converged      : True
  Wall seconds   : ~1-2s
```

## RQ-E2 — distillation gain at fixed latency (pending GPU run)

Reproduce the GPU row by running the full pipeline:

```bash
# 1. Phase 1 must have pulled RTDetr-R50 weights.
# 2. Wire the DataLoader in scripts/run_distill_full.py (implement
#    _make_dataloader with CocoDataset + RTDetrImageProcessor).
# 3. Train.
python scripts/run_distill_full.py \
    --coco-annotations data/coco/annotations/instances_train2017.json \
    --coco-images data/coco/train2017 \
    --epochs 50 \
    --lr 2e-4 \
    --temperature 4.0 \
    --alpha 0.7 \
    --out-dir checkpoints/distill_full

# 4. Export + bench the student exactly like Phase 2 teacher.
python scripts/run_compile_smoke.py --stage all --out-dir checkpoints/distill_student
python scripts/run_latency_sweep.py \
    --backends torch-cuda trt-fp16 \
    --model rtdetr  # (name a student checkpoint variant)
```

| Config | mAP@[0.5:0.95] | p95 (ms) | FPS | Size (MB) | vs teacher-FP16 |
|---|---|---|---|---|---|
| Teacher FP32 (RT-DETR-R50) | TBD | TBD | TBD | ~120 | — |
| Teacher FP16 (TRT) | TBD | TBD | TBD | ~60 | baseline |
| Teacher INT8 (TRT) | TBD | TBD | TBD | ~30 | TBD |
| **Student FP16 (distilled R18)** | TBD | ≤ 8ms target | TBD | ~30 | **RQ-E2 answer** |
| Student INT8 (distilled R18) | TBD | TBD | TBD | ~15 | TBD |

The RQ-E2 finding: at the same p95 budget (8 ms on RTX 5080), does the
distilled R18-FP16 student beat the quantized R50-INT8 teacher on mAP?

## Design notes

### Why temperature=4 for detection KD

Higher temperature makes the teacher's distribution softer, providing a
richer gradient signal to the student early in training. For detection,
where the class distribution across N queries is very sparse (most queries
predict "background"), T=2 collapses too fast; T=4 or T=8 keeps training
from becoming a hard-label CE exercise too early.

### Why only logit KD for now

Feature-KD on DETR-style models requires careful adapter layers (the R50
backbone has 2048 channels in the last stage; R18 has 512), and the adapter
parameters add ~12M new parameters to train. The logit-only baseline is
simpler and typically recovers 90%+ of the feature-KD gain on COCO-scale
datasets. If the logit-only student doesn't reach the target mAP, feature
KD is the first thing to try.

### Pruning after distillation (not before)

The recommended order is: distil first, prune second, fine-tune third.
Pruning before distillation removes channels the teacher is trying to match,
making the KD gradient noisy. The `structured_prune` module is ready to apply
after the student checkpoint lands.
