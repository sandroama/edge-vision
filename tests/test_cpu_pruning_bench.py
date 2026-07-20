"""Drift guard for ``scripts/bench_cpu_pruning.py`` and its checked-in results.

Two layers:
    1. Unit checks on the script's pruning helper (fast, always run) — it must
       actually zero ~amount of the weights on this toolchain (this is the
       path that bypasses the torch>=2.12 wrapper drift).
    2. Consistency checks on ``docs/results/phase4_cpu_pruning.json``: every
       quoted statistic must be recomputable from the raw per-run samples
       stored alongside it, and the headline negative results must still hold
       (flat raw ONNX size across sparsity, monotone gzip size, monotone
       fidelity). Skips if the results file is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase4_cpu_pruning.json"

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- helper unit check

def test_prune_l1_baked_zeros_requested_fraction():
    from edgevision.models.tiny_model import make_tiny_model
    from scripts.bench_cpu_pruning import prune_l1_baked, sparsity_stats

    torch.manual_seed(0)
    model = make_tiny_model(num_classes=8, num_queries=4)
    n = prune_l1_baked(model, amount=0.5)
    assert n == 4  # 2 Conv2d + 2 Linear leaf modules
    st = sparsity_stats(model)
    # Weights are half-zeroed; biases are not pruned, so global sparsity is
    # a bit under 0.5 but must be well away from the 0.0 the drifted wrapper
    # reports.
    assert 0.35 < st["global_sparsity"] < 0.5
    # Masks are baked: no torch pruning bookkeeping left behind.
    assert not any(hasattr(m, "weight_orig") for m in model.modules())


def test_channel_prune_conv_chain_actually_shrinks():
    from edgevision.models.tiny_model import make_tiny_input, make_tiny_model
    from edgevision.pruning import channel_prune_conv_chain

    torch.manual_seed(0)
    model = make_tiny_model(num_classes=8, num_queries=4)
    n_before = sum(p.numel() for p in model.parameters())

    pruned, res = channel_prune_conv_chain(model, amount=0.5)
    n_after = sum(p.numel() for p in pruned.parameters())

    # Params genuinely removed (not masked): count drops, no zeros needed.
    assert res.n_parameters_before == n_before
    assert res.n_parameters_after == n_after < n_before
    assert res.channels_kept == {"backbone.0": 8, "backbone.2": 16}
    # Source model untouched.
    assert sum(p.numel() for p in model.parameters()) == n_before
    # Output contract preserved.
    out = pruned(make_tiny_input(1, 64, 64))
    assert out["logits"].shape == (1, 4, 8)
    assert out["pred_boxes"].shape == (1, 4, 4)


def test_channel_prune_conv_chain_fails_closed_on_non_chain():
    from edgevision.pruning import channel_prune_conv_chain
    from torch import nn

    class NotAChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3)
            self.head = nn.Linear(99, 4)  # not fed by pooled conv channels

    with pytest.raises(ValueError, match="refusing to prune"):
        channel_prune_conv_chain(NotAChain(), amount=0.5)

    with pytest.raises(ValueError, match="amount"):
        channel_prune_conv_chain(NotAChain(), amount=1.0)


# --------------------------------------------------------------------------- results drift guard

@pytest.fixture(scope="module")
def result() -> dict:
    if not RESULTS.exists():
        pytest.skip("phase4_cpu_pruning.json not generated yet")
    return json.loads(RESULTS.read_text())


def _mask_rows(result):
    return [r for r in result["rows"]
            if r["prune_amount"] > 0 and not r["quantized"] and not r.get("structured")]


def _structured_rows(result):
    return [r for r in result["rows"]
            if r["prune_amount"] > 0 and not r["quantized"] and r.get("structured")]


def test_rows_present_and_protocol_sane(result):
    configs = [r["config"] for r in result["rows"]]
    assert configs[0] == "fp32 baseline"
    amounts = [r["prune_amount"] for r in _mask_rows(result)]
    assert amounts == [0.2, 0.4, 0.6, 0.8]
    assert [r["prune_amount"] for r in _structured_rows(result)] == amounts
    # int8-only + mask40+int8 + structured40+int8
    assert sum(r["quantized"] for r in result["rows"]) == 3
    assert result["protocol"]["n_runs"] >= 100


def test_stats_match_raw_samples(result):
    for row in result["rows"]:
        samples = np.asarray(row["raw_samples_ms"], dtype=np.float64)
        assert samples.size == result["protocol"]["n_runs"] == row["latency"]["n"]
        assert row["latency"]["p50_ms"] == pytest.approx(np.percentile(samples, 50), abs=1e-3)
        assert row["latency"]["p95_ms"] == pytest.approx(np.percentile(samples, 95), abs=1e-3)
        assert row["latency"]["mean_ms"] == pytest.approx(samples.mean(), abs=1e-3)


def test_latency_ratio_matches_raw_samples(result):
    base = np.median(result["rows"][0]["raw_samples_ms"])
    for row in result["rows"][1:]:
        ratio = np.median(row["raw_samples_ms"]) / base
        assert row["latency_ratio_vs_fp32"]["point"] == pytest.approx(ratio, abs=1e-3)


def test_headline_negative_results_hold(result):
    pruned = _mask_rows(result)
    base = result["rows"][0]
    # Raw ONNX size flat: masks don't shrink dense storage.
    assert all(r["size"]["onnx_bytes"] == base["size"]["onnx_bytes"] for r in pruned)
    # gzip size strictly decreases with sparsity; logit fidelity degrades.
    gz = [base["size"]["gzip9_bytes"]] + [r["size"]["gzip9_bytes"] for r in pruned]
    assert gz == sorted(gz, reverse=True)
    cos = [r["fidelity"]["cosine_logits"] for r in pruned]
    assert cos == sorted(cos, reverse=True)
    assert base["fidelity"]["cosine_logits"] == 1.0
    # Nonzero param count decreases with sparsity.
    nz = [r["params"]["n_parameters_nonzero"] for r in pruned]
    assert nz == sorted(nz, reverse=True)


def test_int8_rows_keep_phase3_size_win(result):
    base = result["rows"][0]
    for row in result["rows"]:
        if row["quantized"]:
            reduction = 1 - row["size"]["onnx_bytes"] / base["size"]["onnx_bytes"]
            assert reduction > 0.6  # the ~72% dynamic-INT8 size cut survives pruning


def test_structured_rows_actually_shrink(result):
    """The channel-removal counterpoint: raw size and params genuinely drop."""
    base = result["rows"][0]
    structured = _structured_rows(result)
    assert structured, "structured channel-removal rows missing from results"
    sizes = [base["size"]["onnx_bytes"]] + [r["size"]["onnx_bytes"] for r in structured]
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(sizes)
    params = [base["params"]["n_parameters_total"]] + [
        r["params"]["n_parameters_total"] for r in structured
    ]
    assert params == sorted(params, reverse=True) and len(set(params)) == len(params)
    # Structured rows carry their surgery audit trail.
    for r in structured:
        assert r["channel_prune"]["n_parameters_after"] == r["params"]["n_parameters_total"]


def test_structured_int8_compound_size_cut(result):
    """structured 40% + INT8 must beat INT8-only on raw size (they compound)."""
    by_config = {r["config"]: r for r in result["rows"]}
    int8 = by_config["INT8 dynamic (unpruned)"]
    combo = by_config["structured 40% + INT8 dynamic"]
    assert combo["size"]["onnx_bytes"] < int8["size"]["onnx_bytes"]
