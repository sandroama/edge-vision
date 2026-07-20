# Phase 4 — CPU pruning lane (measured, no GPU)

> **Status:** MEASURED on CPU by
> [`scripts/bench_cpu_pruning.py`](../../scripts/bench_cpu_pruning.py), same
> protocol as the INT8 lane (200 timed passes/model, warmup, bootstrap 95%
> CIs from raw samples — all samples checked into `phase4_cpu_pruning.json`).
> **Primary finding is negative:** L1 *mask* pruning does **not** reduce raw
> ONNX size or dense CPU latency. **The structured channel-removal rows are
> the counterpoint:** actually removing channels
> (`edgevision.pruning.channel_prune_conv_chain`) cuts raw ONNX size by
> **78.8% at 80%** and moves dense latency
> (0.545x vs FP32) — at a steep fidelity
> cost on this random-init model.
> GPU/TensorRT latency and watts/frame stay **pending** ([`NEXT_STEPS.md`](../../NEXT_STEPS.md)).

## What was measured

- **Model:** `edgevision.models.tiny_model.make_tiny_model (RT-DETR-shaped CI stand-in, randomly initialized)` — input `[1, 3, 640, 640]`.
  **Randomly initialized** (no trained checkpoint exists), so the accuracy
  column is an **output-fidelity proxy** — mean cosine similarity of
  `logits` / `pred_boxes` vs the unpruned FP32 model on
  16 fixed random inputs — **NOT task accuracy**.
- **Pruning:** masks: torch.nn.utils.prune.l1_unstructured per Conv2d/Linear leaf module, baked via per-module prune.remove (mask wrappers bypassed due to documented torch>=2.12 drift; same L1 magnitude criterion). structured: edgevision.pruning.channel_prune_conv_chain — true L1 channel removal (rebuilt Conv2d/Linear modules, fewer params/FLOPs).
- **Inference:** `CPUExecutionProvider`, intra-op threads =
  1, graph optimization =
  `all`, 30 warmup +
  **200 timed** single-image passes per row.
- **Export:** torch legacy TorchScript exporter (dynamo=False, opset 17) for all rows — identical for every row, so sizes
  and latencies are apples-to-apples.

## Sweep (bracketed = bootstrap 95% CIs, 10,000 replicates)

| Config | Global sparsity | Nonzero params | ONNX KiB | gzip-9 KiB | p50 ms [95% CI] | p95 ms [95% CI] | latency vs FP32 [95% CI] | fidelity cos (logits / boxes) |
|---|---|---|---|---|---|---|---|---|
| fp32 baseline | 0% | 836,688 | 3270.47 | 3027.02 | 1.2433 [1.2271, 1.2676] | 1.4628 [1.4303, 1.5238] | 1.0x [1.0, 1.0] | 1.0 / 1.0 |
| pruned 20% (L1 mask, baked) | 19% | 674,400 | 3270.47 | 2593.98 | 1.2603 [1.2397, 1.2786] | 1.4496 [1.4287, 1.4854] | 1.0136x [0.9885, 1.0341] | 0.998666 / 0.999995 |
| pruned 40% (L1 mask, baked) | 39% | 512,112 | 3270.47 | 2095.7 | 1.2113 [1.1971, 1.2312] | 1.4108 [1.3851, 1.4257] | 0.9743x [0.9531, 0.9946] | 0.990171 / 0.999969 |
| pruned 60% (L1 mask, baked) | 58% | 349,824 | 3270.47 | 1534.78 | 1.2251 [1.2092, 1.236] | 1.434 [1.4144, 1.4686] | 0.9854x [0.9634, 1.0012] | 0.967358 / 0.999891 |
| pruned 80% (L1 mask, baked) | 78% | 187,536 | 3270.47 | 934.12 | 1.2233 [1.2025, 1.2436] | 1.413 [1.3697, 1.4438] | 0.9839x [0.9596, 1.0059] | 0.922278 / 0.999738 |
| structured 20% (channels removed) | 0% | 683,832 | 2673.37 | 2474.2 | 1.1263 [1.1105, 1.1521] | 1.3392 [1.3155, 1.4103] | 0.9059x [0.8864, 0.9286] | 0.987774 / 0.999958 |
| structured 40% (channels removed) | 0% | 506,009 | 1978.75 | 1831.07 | 1.0407 [1.0214, 1.0654] | 1.6452 [1.5067, 1.9628] | 0.8371x [0.817, 0.8602] | 0.940958 / 0.999798 |
| structured 60% (channels removed) | 0% | 353,683 | 1383.72 | 1280.28 | 0.8248 [0.814, 0.8377] | 1.0553 [0.9871, 1.2705] | 0.6634x [0.6481, 0.6783] | 0.921267 / 0.999734 |
| structured 80% (channels removed) | 0% | 176,652 | 692.2 | 640.01 | 0.6777 [0.6739, 0.6833] | 0.8398 [0.8034, 0.8796] | 0.545x [0.5345, 0.5537] | 0.901435 / 0.999667 |
| INT8 dynamic (unpruned) | 0% | 836,688 | 898.35 | 886.34 | 3.7678 [3.7312, 3.8057] | 5.7131 [4.4526, 9.299] | 3.0305x [2.9679, 3.0845] | 0.999996 / 1.0 |
| pruned 40% + INT8 dynamic | 39% | 512,112 | 898.35 | 684.71 | 3.7804 [3.7469, 3.8215] | 4.1457 [4.094, 4.1705] | 3.0406x [2.98, 3.0984] | 0.990185 / 0.999969 |
| structured 40% + INT8 dynamic | 0% | 506,009 | 575.37 | 563.26 | 2.6886 [2.6519, 2.7125] | 2.8755 [2.8462, 2.9293] | 2.1625x [2.1118, 2.1996] | 0.94094 / 0.999797 |

## Findings (exactly as measured)

1. **Mask pruning alone buys nothing on the dense CPU runtime — the negative
   result is the headline.** Raw ONNX size is flat
   (3270.47 KiB at every sparsity level) because zeroed
   weights are still stored dense. Latency ratios vs FP32 sit at
   0.9743–1.0136x with **no monotone trend in sparsity** — the pruned
   graphs are structurally identical to the baseline, so the few-percent
   wobble (some CIs exclude 1.0) is run-order/machine noise, **not** a pruning
   speed-up; ORT runs the same dense kernels regardless of the zeros.
2. **Channel removal is the lever that works — and the table shows its cost.**
   `channel_prune_conv_chain` (true structured surgery on the conv chain)
   shrinks raw ONNX size monotonically, down to
   692.2 KiB (−78.8%) at 80% removal with a
   0.545x latency ratio vs FP32 — the size
   axis masks cannot move. The price is fidelity: cosine(logits) falls to
   0.901435 at 80% (vs 0.922278
   for the 80% mask), because removing a channel discards its whole output.
   On a trained model this is exactly the retrain-after-prune trade the
   literature describes; measuring the *accuracy* consequence needs the GPU
   distillation run (RQ-E2, pending).
3. **The one real size signal from masks is compressibility:** gzip-9 size
   falls monotonically with sparsity (3027.02 →
   934.12 KiB at 80%), relevant only if models ship
   compressed.
4. **Fidelity degrades with sparsity** (cosine columns) — on a *trained* model
   this would be the mAP axis of the Pareto; measuring that needs the GPU
   distillation run (RQ-E2, pending).
5. **INT8 stacking:** mask-pruned-40%+INT8 keeps the ~73%
   dynamic-INT8 size reduction but mask sparsity adds nothing further;
   **structured-40%+INT8 compounds** to −82.4% of the FP32
   baseline (575.37 KiB) because channel removal
   and INT8 shrink different things (fewer weights × 1 byte each). The INT8
   latency *penalty* on this small CPU graph (phase 3's honest negative
   result) persists in all INT8 rows (3.0305x unpruned).

## Environment

- Python 3.11.15 · macOS-26.5.1-arm64-arm-64bit
- processor: `arm`
- torch 2.12.0 · onnxruntime 1.26.0 ·
  onnx 1.21.0 · numpy 1.26.4
- generated (UTC): 2026-07-18T17:36:26.379996+00:00

## Reproduce

```bash
python scripts/bench_cpu_pruning.py --n-runs 200 --n-warmup 30
```

## What this is NOT

- ❌ Not task accuracy / mAP — the model is randomly initialized; the fidelity
  column is output similarity only, and no retraining/fine-tuning follows the
  channel removal (a trained pipeline would recover fidelity after surgery).
- ❌ Not a GPU/TensorRT result — those rows stay **pending**.
- ✅ Is a real, reproducible CPU measurement showing mask-pruning's true (null)
  effect on dense size/latency, the structured channel-removal counterpoint
  (real size cut, real fidelity cost), the gzip compressibility win, and how
  both compose with dynamic INT8.
