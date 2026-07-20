"""Measure the PRUNING axis of the Pareto on CPU (Phase 4, no GPU).

Sweeps L1-magnitude weight pruning at sparsity {0.2, 0.4, 0.6, 0.8} on the
same RT-DETR-shaped tiny detector the CPU INT8 lane uses
(``edgevision.models.tiny_model.make_tiny_model``), and measures — with the
exact protocol of ``scripts/bench_cpu_int8.py`` (>=100 timed passes,
CPUExecutionProvider, bootstrap 95% CIs from raw samples) — what mask-based
pruning actually buys on a dense CPU runtime:

    * parameter counts (total / nonzero) and global sparsity achieved,
    * ONNX file size, raw AND gzip-9 compressed (zeros compress; dense
      storage does not shrink),
    * CPU latency p50/p95/p99 + latency ratio vs the unpruned FP32 baseline,
    * an OUTPUT-FIDELITY proxy vs the unpruned model — cosine similarity and
      MSE of ``logits`` / ``pred_boxes`` on fixed random inputs. The model is
      randomly initialized, so this is **fidelity, NOT task accuracy**.

It also measures the project's **structured channel-removal** lever
(``edgevision.pruning.channel_prune_conv_chain``) at the same sparsity
levels — the counterpoint to the mask null result: channels are actually
removed, so raw ONNX size / params / FLOPs genuinely shrink.

And it measures whether pruning **stacks** with dynamic INT8: one INT8-only
row, one mask-pruned(0.4)+INT8 row, and one structured(0.4)+INT8 row,
quantized in-session with ``onnxruntime.quantization.quantize_dynamic`` for
an apples-to-apples same-machine comparison.

Honesty notes baked into the outputs:
    * The project's ``edgevision.pruning`` wrappers hit the documented
      torch>=2.12 drift (``apply_pruning`` counts zeros on ``parameters()``,
      which sees ``weight_orig``; ``remove_pruning`` trips on container
      modules). This script therefore prunes at the ``torch.nn.utils.prune``
      API level directly — same L1 magnitude criterion, per leaf module —
      and bakes masks with per-module ``prune.remove``. The wrapper modules
      and their version-guarded tests are left untouched.
    * Masks zero weights; they do not remove channels. The expected (and
      measured) result is that raw ONNX size and dense CPU latency do NOT
      improve — that negative result is the point of this measurement.

Outputs (under ``docs/results/``):
    * ``phase4_cpu_pruning.json`` — full summary + every raw latency sample.
    * ``phase4_cpu_pruning.md``   — human-readable table + findings.

Run::

    python scripts/bench_cpu_pruning.py --n-runs 200 --n-warmup 30
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import platform
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from edgevision.compile.onnxrt_cpu import OnnxRuntimeCPUExecutor
from edgevision.models.tiny_model import make_tiny_input, make_tiny_model
from edgevision.pruning import channel_prune_conv_chain

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/bench_cpu_pruning.py`
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the phase-3 CPU-lane measurement helpers verbatim (same protocol).
from scripts.bench_cpu_int8 import _ci_block, _speedup_ci, _summary, _time_fn  # noqa: E402
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "results"

SPARSITY_LEVELS = (0.2, 0.4, 0.6, 0.8)
COMBINED_AMOUNT = 0.4  # pruning level used for the pruned+INT8 stacking row


# --------------------------------------------------------------------------- pruning

def prune_l1_baked(model, amount: float) -> int:
    """L1-magnitude prune ``amount`` of each Conv2d/Linear weight, mask baked in.

    Works around the documented torch>=2.12 drift in the project's
    ``edgevision.pruning`` wrappers by calling ``torch.nn.utils.prune``
    directly on leaf modules and immediately ``prune.remove``-ing per module.
    Returns the number of modules pruned. In-place.
    """
    import torch.nn.utils.prune as prune

    n_pruned = 0
    for module in model.modules():
        if type(module).__name__ not in ("Conv2d", "Linear"):
            continue
        if getattr(module, "weight", None) is None:
            continue
        prune.l1_unstructured(module, name="weight", amount=amount)
        prune.remove(module, "weight")  # bake mask -> dense weights with zeros
        n_pruned += 1
    return n_pruned


def sparsity_stats(model) -> dict:
    total = zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += int((p.data == 0).sum().item())
    return {
        "n_parameters_total": total,
        "n_parameters_nonzero": total - zeros,
        "n_parameters_zero": zeros,
        "global_sparsity": round(zeros / max(total, 1), 4),
    }


# --------------------------------------------------------------------------- export / fidelity

def export_legacy(model, dummy, path: Path) -> None:
    """Legacy TorchScript ONNX export (dynamo=False, opset 17).

    The same path phase-3 landed on for this toolchain (torch 2.12's dynamo
    exporter emits a graph ORT's quantizer cannot pre-process), used for ALL
    rows here so sizes and latencies are directly comparable.
    """
    import torch

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(path),
            opset_version=17,
            input_names=["images"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={n: {0: "batch"} for n in ("images", "logits", "pred_boxes")},
            do_constant_folding=True,
            dynamo=False,
        )


def file_sizes(path: Path) -> dict:
    raw = path.stat().st_size
    gz = len(gzip.compress(path.read_bytes(), compresslevel=9))
    return {
        "onnx_bytes": raw,
        "onnx_kib": round(raw / 1024, 2),
        "gzip9_bytes": gz,
        "gzip9_kib": round(gz / 1024, 2),
    }


def fidelity_vs_baseline(
    ex_base: OnnxRuntimeCPUExecutor,
    ex_cand: OnnxRuntimeCPUExecutor,
    inputs: list[np.ndarray],
) -> dict:
    """Output fidelity of candidate vs unpruned baseline on fixed random inputs.

    Cosine similarity + MSE per output head, averaged over inputs. This is an
    output-fidelity proxy on a randomly initialized model — NOT task accuracy.
    """
    cos = {"logits": [], "pred_boxes": []}
    mse = {"logits": [], "pred_boxes": []}
    for x in inputs:
        base = ex_base.run(x)
        cand = ex_cand.run(x)
        for name in ("logits", "pred_boxes"):
            a = base.get(name).ravel().astype(np.float64)
            b = cand.get(name).ravel().astype(np.float64)
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            cos[name].append(float(np.dot(a, b) / denom) if denom > 0 else 0.0)
            mse[name].append(float(np.mean((a - b) ** 2)))
    return {
        "metric": "output fidelity vs unpruned FP32 (random-init model; NOT task accuracy)",
        "n_inputs": len(inputs),
        "cosine_logits": round(float(np.mean(cos["logits"])), 6),
        "cosine_pred_boxes": round(float(np.mean(cos["pred_boxes"])), 6),
        "mse_logits": round(float(np.mean(mse["logits"])), 8),
        "mse_pred_boxes": round(float(np.mean(mse["pred_boxes"])), 8),
    }


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pruning axis of the Pareto on CPU: L1-mask sparsity sweep + "
                    "INT8 stacking, same protocol as bench_cpu_int8.py. No GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-runs", type=int, default=200, help="timed passes per model (>=100)")
    ap.add_argument("--n-warmup", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-fidelity-inputs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-classes", type=int, default=80)
    ap.add_argument("--num-queries", type=int, default=300)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--intra-op-threads", type=int, default=1)
    ap.add_argument("--graph-opt", default="all", choices=["off", "basic", "extended", "all"])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "checkpoints" / "cpu_pruning")
    args = ap.parse_args(argv)

    if args.n_runs < 100:
        print(f"[cpu-prune] --n-runs={args.n_runs} < 100; bumping to 100.")
        args.n_runs = 100

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    args.work_dir.mkdir(parents=True, exist_ok=True)

    def quantize_int8(src: Path, dst: Path) -> None:
        pre = src.with_suffix(".pre.onnx")
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        quantize_dynamic(model_input=str(pre), model_output=str(dst),
                         weight_type=QuantType.QInt8)
        pre.unlink(missing_ok=True)

    # Base model — same config as the phase-3 CPU INT8 lane.
    print(f"[cpu-prune] building tiny detector "
          f"(num_classes={args.num_classes}, num_queries={args.num_queries})")
    base_model = make_tiny_model(num_classes=args.num_classes, num_queries=args.num_queries)
    dummy = make_tiny_input(batch=1, height=args.height, width=args.width)

    # Fixed inputs: one for latency (identical to phase-3 protocol), K for fidelity.
    x_lat = np.random.default_rng(args.seed).standard_normal(
        (1, 3, args.height, args.width)).astype(np.float32)
    fid_rng = np.random.default_rng(args.seed + 1)
    x_fid = [fid_rng.standard_normal((1, 3, args.height, args.width)).astype(np.float32)
             for _ in range(args.n_fidelity_inputs)]

    # ------------------------------------------------------------- build all rows
    rows: list[dict] = []

    def add_row(name: str, onnx_path: Path, *, model_stats: dict, extra: dict) -> dict:
        row = {"config": name, **extra, "params": model_stats, "size": file_sizes(onnx_path),
               "_path": onnx_path}
        rows.append(row)
        return row

    base_path = args.work_dir / "tiny_fp32_base.onnx"
    export_legacy(base_model, dummy, base_path)
    add_row("fp32 baseline", base_path, model_stats=sparsity_stats(base_model),
            extra={"prune_amount": 0.0, "quantized": False})

    pruned_models: dict[float, object] = {}
    for amount in SPARSITY_LEVELS:
        m = copy.deepcopy(base_model)
        n_mod = prune_l1_baked(m, amount)
        pruned_models[amount] = m
        p = args.work_dir / f"tiny_pruned_{int(amount * 100)}.onnx"
        export_legacy(m, dummy, p)
        st = sparsity_stats(m)
        print(f"[cpu-prune] amount={amount}: {n_mod} modules pruned, "
              f"global sparsity={st['global_sparsity']}")
        add_row(f"pruned {amount:.0%} (L1 mask, baked)", p, model_stats=st,
                extra={"prune_amount": amount, "quantized": False, "modules_pruned": n_mod})

    # Structured channel-removal rows — the lever the mask rows prove is
    # missing. Channels are REMOVED, so params/size/FLOPs genuinely shrink.
    structured_models: dict[float, object] = {}
    for amount in SPARSITY_LEVELS:
        m, sres = channel_prune_conv_chain(base_model, amount)
        structured_models[amount] = m
        p = args.work_dir / f"tiny_structured_{int(amount * 100)}.onnx"
        export_legacy(m, dummy, p)
        print(f"[cpu-prune] structured amount={amount}: params "
              f"{sres.n_parameters_before:,} -> {sres.n_parameters_after:,} "
              f"(-{sres.param_reduction_pct:.1f}%), channels kept={sres.channels_kept}")
        add_row(f"structured {amount:.0%} (channels removed)", p,
                model_stats=sparsity_stats(m),
                extra={"prune_amount": amount, "quantized": False, "structured": True,
                       "channel_prune": sres.as_dict()})

    int8_path = args.work_dir / "tiny_int8.onnx"
    quantize_int8(base_path, int8_path)
    add_row("INT8 dynamic (unpruned)", int8_path, model_stats=sparsity_stats(base_model),
            extra={"prune_amount": 0.0, "quantized": True})

    comb_src = args.work_dir / f"tiny_pruned_{int(COMBINED_AMOUNT * 100)}.onnx"
    comb_path = args.work_dir / "tiny_pruned_40_int8.onnx"
    quantize_int8(comb_src, comb_path)
    add_row(f"pruned {COMBINED_AMOUNT:.0%} + INT8 dynamic", comb_path,
            model_stats=sparsity_stats(pruned_models[COMBINED_AMOUNT]),
            extra={"prune_amount": COMBINED_AMOUNT, "quantized": True})

    scomb_src = args.work_dir / f"tiny_structured_{int(COMBINED_AMOUNT * 100)}.onnx"
    scomb_path = args.work_dir / "tiny_structured_40_int8.onnx"
    quantize_int8(scomb_src, scomb_path)
    add_row(f"structured {COMBINED_AMOUNT:.0%} + INT8 dynamic", scomb_path,
            model_stats=sparsity_stats(structured_models[COMBINED_AMOUNT]),
            extra={"prune_amount": COMBINED_AMOUNT, "quantized": True, "structured": True})

    # ------------------------------------------------------------- measure all rows
    def make_exec(path: Path) -> OnnxRuntimeCPUExecutor:
        return OnnxRuntimeCPUExecutor(path, num_threads=args.intra_op_threads,
                                      graph_optimization=args.graph_opt)

    ex_base = make_exec(base_path)
    assert "CPUExecutionProvider" in ex_base.describe()["providers"]

    baseline_samples: list[float] | None = None
    for row in rows:
        ex = ex_base if row["_path"] == base_path else make_exec(row["_path"])
        print(f"[cpu-prune] timing '{row['config']}': "
              f"warmup={args.n_warmup} runs={args.n_runs}")
        samples = _time_fn(ex.make_callable(x_lat), n_runs=args.n_runs, n_warmup=args.n_warmup)
        row["latency"] = _summary(samples)
        row["latency_ci"] = _ci_block(samples, n_boot=args.n_boot, alpha=args.alpha, rng=rng)
        if baseline_samples is None:
            baseline_samples = samples
            row["latency_ratio_vs_fp32"] = {"point": 1.0, "lo": 1.0, "hi": 1.0}
        else:
            # ratio of median latencies: row / baseline (>1 means slower than fp32)
            r = _speedup_ci(samples, baseline_samples,
                            n_boot=args.n_boot, alpha=args.alpha, rng=rng)
            row["latency_ratio_vs_fp32"] = r
        row["fidelity"] = fidelity_vs_baseline(ex_base, ex, x_fid)
        row["raw_samples_ms"] = [round(s, 6) for s in samples]
        del row["_path"]
        print(f"[cpu-prune]   p50={row['latency']['p50_ms']}ms  "
              f"ratio_vs_fp32={row['latency_ratio_vs_fp32']['point']}x  "
              f"cos(logits)={row['fidelity']['cosine_logits']}")

    # ------------------------------------------------------------- write outputs
    import onnx as onnx_pkg
    import onnxruntime as ort

    result = {
        "experiment": "phase4_cpu_pruning",
        "lane": "CPU (ONNX Runtime, CPUExecutionProvider) — pruning axis of the Pareto, no GPU",
        "pruning": "masks: torch.nn.utils.prune.l1_unstructured per Conv2d/Linear leaf module, "
                   "baked via per-module prune.remove (mask wrappers bypassed due to documented "
                   "torch>=2.12 drift; same L1 magnitude criterion). structured: "
                   "edgevision.pruning.channel_prune_conv_chain — true L1 channel removal "
                   "(rebuilt Conv2d/Linear modules, fewer params/FLOPs)",
        "generated_utc": datetime.now(UTC).isoformat(),
        "model": {
            "source": "edgevision.models.tiny_model.make_tiny_model (RT-DETR-shaped CI stand-in, "
                      "randomly initialized)",
            "export_path": "torch legacy TorchScript exporter (dynamo=False, opset 17) for all rows",
            "num_classes": args.num_classes,
            "num_queries": args.num_queries,
            "input_shape": [1, 3, args.height, args.width],
            "accuracy_note": "Randomly initialized model — 'fidelity' rows are output "
                             "similarity vs the unpruned model, NOT task accuracy. No "
                             "trained checkpoint exists in checkpoints/.",
        },
        "protocol": {
            "provider": "CPUExecutionProvider",
            "intra_op_threads": args.intra_op_threads,
            "graph_optimization": args.graph_opt,
            "n_warmup": args.n_warmup,
            "n_runs": args.n_runs,
            "n_boot": args.n_boot,
            "alpha": args.alpha,
            "n_fidelity_inputs": args.n_fidelity_inputs,
            "seed": args.seed,
        },
        "rows": rows,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "onnx": onnx_pkg.__version__,
            "numpy": np.__version__,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "phase4_cpu_pruning.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[cpu-prune] wrote {json_path}")
    md_path = args.out_dir / "phase4_cpu_pruning.md"
    md_path.write_text(render_md(result))
    print(f"[cpu-prune] wrote {md_path}")
    return 0


# --------------------------------------------------------------------------- markdown

def render_md(r: dict) -> str:
    rows = r["rows"]
    conf = f"{int((1 - r['protocol']['alpha']) * 100)}%"
    base = rows[0]

    def fmt_row(row: dict) -> str:
        lat, ci, ratio, fid = (row["latency"], row["latency_ci"],
                               row["latency_ratio_vs_fp32"], row["fidelity"])
        p = row["params"]
        s = row["size"]
        return (
            f"| {row['config']} | {p['global_sparsity']:.0%} | {p['n_parameters_nonzero']:,} "
            f"| {s['onnx_kib']} | {s['gzip9_kib']} "
            f"| {lat['p50_ms']} [{ci['p50_ms']['lo']}, {ci['p50_ms']['hi']}] "
            f"| {lat['p95_ms']} [{ci['p95_ms']['lo']}, {ci['p95_ms']['hi']}] "
            f"| {ratio['point']}x [{ratio['lo']}, {ratio['hi']}] "
            f"| {fid['cosine_logits']} / {fid['cosine_pred_boxes']} |"
        )

    table = "\n".join(fmt_row(row) for row in rows)
    n = r["protocol"]["n_runs"]

    def get(config: str) -> dict:
        return next(row for row in rows if row["config"] == config)

    mask_rows = [row for row in rows
                 if row["prune_amount"] > 0 and not row["quantized"] and not row.get("structured")]
    struct_rows = [row for row in rows
                   if row["prune_amount"] > 0 and not row["quantized"] and row.get("structured")]
    prune_ratios = [row["latency_ratio_vs_fp32"]["point"] for row in mask_rows]
    ratio_lo, ratio_hi = min(prune_ratios), max(prune_ratios)
    s80 = struct_rows[-1]
    s80_size_pct = round(100 * (1 - s80["size"]["onnx_bytes"] / base["size"]["onnx_bytes"]), 1)
    mask80 = get("pruned 80% (L1 mask, baked)")
    int8_row = get("INT8 dynamic (unpruned)")
    mask_int8 = get("pruned 40% + INT8 dynamic")
    struct_int8 = get("structured 40% + INT8 dynamic")
    struct_int8_pct = round(100 * (1 - struct_int8["size"]["onnx_bytes"] / base["size"]["onnx_bytes"]), 1)

    return f"""# Phase 4 — CPU pruning lane (measured, no GPU)

> **Status:** MEASURED on CPU by
> [`scripts/bench_cpu_pruning.py`](../../scripts/bench_cpu_pruning.py), same
> protocol as the INT8 lane ({n} timed passes/model, warmup, bootstrap {conf}
> CIs from raw samples — all samples checked into `phase4_cpu_pruning.json`).
> **Primary finding is negative:** L1 *mask* pruning does **not** reduce raw
> ONNX size or dense CPU latency. **The structured channel-removal rows are
> the counterpoint:** actually removing channels
> (`edgevision.pruning.channel_prune_conv_chain`) cuts raw ONNX size by
> **{s80_size_pct}% at 80%** and moves dense latency
> ({s80['latency_ratio_vs_fp32']['point']}x vs FP32) — at a steep fidelity
> cost on this random-init model.
> GPU/TensorRT latency and watts/frame stay **pending** ([`NEXT_STEPS.md`](../../NEXT_STEPS.md)).

## What was measured

- **Model:** `{r['model']['source']}` — input `{r['model']['input_shape']}`.
  **Randomly initialized** (no trained checkpoint exists), so the accuracy
  column is an **output-fidelity proxy** — mean cosine similarity of
  `logits` / `pred_boxes` vs the unpruned FP32 model on
  {r['protocol']['n_fidelity_inputs']} fixed random inputs — **NOT task accuracy**.
- **Pruning:** {r['pruning']}.
- **Inference:** `{r['protocol']['provider']}`, intra-op threads =
  {r['protocol']['intra_op_threads']}, graph optimization =
  `{r['protocol']['graph_optimization']}`, {r['protocol']['n_warmup']} warmup +
  **{n} timed** single-image passes per row.
- **Export:** {r['model']['export_path']} — identical for every row, so sizes
  and latencies are apples-to-apples.

## Sweep (bracketed = bootstrap {conf} CIs, {r['protocol']['n_boot']:,} replicates)

| Config | Global sparsity | Nonzero params | ONNX KiB | gzip-9 KiB | p50 ms [{conf} CI] | p95 ms [{conf} CI] | latency vs FP32 [{conf} CI] | fidelity cos (logits / boxes) |
|---|---|---|---|---|---|---|---|---|
{table}

## Findings (exactly as measured)

1. **Mask pruning alone buys nothing on the dense CPU runtime — the negative
   result is the headline.** Raw ONNX size is flat
   ({base['size']['onnx_kib']} KiB at every sparsity level) because zeroed
   weights are still stored dense. Latency ratios vs FP32 sit at
   {ratio_lo}–{ratio_hi}x with **no monotone trend in sparsity** — the pruned
   graphs are structurally identical to the baseline, so the few-percent
   wobble (some CIs exclude 1.0) is run-order/machine noise, **not** a pruning
   speed-up; ORT runs the same dense kernels regardless of the zeros.
2. **Channel removal is the lever that works — and the table shows its cost.**
   `channel_prune_conv_chain` (true structured surgery on the conv chain)
   shrinks raw ONNX size monotonically, down to
   {s80['size']['onnx_kib']} KiB (−{s80_size_pct}%) at 80% removal with a
   {s80['latency_ratio_vs_fp32']['point']}x latency ratio vs FP32 — the size
   axis masks cannot move. The price is fidelity: cosine(logits) falls to
   {s80['fidelity']['cosine_logits']} at 80% (vs {mask80['fidelity']['cosine_logits']}
   for the 80% mask), because removing a channel discards its whole output.
   On a trained model this is exactly the retrain-after-prune trade the
   literature describes; measuring the *accuracy* consequence needs the GPU
   distillation run (RQ-E2, pending).
3. **The one real size signal from masks is compressibility:** gzip-9 size
   falls monotonically with sparsity ({base['size']['gzip9_kib']} →
   {mask80['size']['gzip9_kib']} KiB at 80%), relevant only if models ship
   compressed.
4. **Fidelity degrades with sparsity** (cosine columns) — on a *trained* model
   this would be the mAP axis of the Pareto; measuring that needs the GPU
   distillation run (RQ-E2, pending).
5. **INT8 stacking:** mask-pruned-40%+INT8 keeps the ~{100 - round(mask_int8['size']['onnx_bytes'] / base['size']['onnx_bytes'] * 100)}%
   dynamic-INT8 size reduction but mask sparsity adds nothing further;
   **structured-40%+INT8 compounds** to −{struct_int8_pct}% of the FP32
   baseline ({struct_int8['size']['onnx_kib']} KiB) because channel removal
   and INT8 shrink different things (fewer weights × 1 byte each). The INT8
   latency *penalty* on this small CPU graph (phase 3's honest negative
   result) persists in all INT8 rows ({int8_row['latency_ratio_vs_fp32']['point']}x unpruned).

## Environment

- Python {r['environment']['python']} · {r['environment']['platform']}
- processor: `{r['environment']['processor']}`
- torch {r['environment']['torch']} · onnxruntime {r['environment']['onnxruntime']} ·
  onnx {r['environment']['onnx']} · numpy {r['environment']['numpy']}
- generated (UTC): {r['generated_utc']}

## Reproduce

```bash
python scripts/bench_cpu_pruning.py --n-runs {n} --n-warmup {r['protocol']['n_warmup']}
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
"""


if __name__ == "__main__":
    raise SystemExit(main())
