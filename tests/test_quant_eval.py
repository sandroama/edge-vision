"""Tests for ``edgevision.evaluation.quant_eval``.

CPU-only — no heavy ML deps. We construct ``CocoMetrics`` objects
directly and verify the diff arithmetic + sorting + table rendering.
"""

from __future__ import annotations

import json

import pytest

from edgevision.evaluation.coco_eval import CocoMetrics
from edgevision.evaluation.quant_eval import (
    PerClassDelta,
    QuantizationDelta,
    compare_metrics,
    summary_table,
)

# --------------------------------------------------------------------------- per-class delta


def test_per_class_delta_arithmetic():
    d = PerClassDelta(label="cat", reference=0.8, candidate=0.7)
    assert d.delta == pytest.approx(-0.1)
    assert d.retained_pct == pytest.approx(87.5)


def test_per_class_delta_zero_reference_is_nan_pct():
    d = PerClassDelta(label="rare", reference=0.0, candidate=0.0)
    assert d.retained_pct != d.retained_pct  # nan


# --------------------------------------------------------------------------- compare_metrics


def _pyc_metrics(map_50_95: float, map_50: float, per_class: dict[str, float]) -> CocoMetrics:
    """Tiny helper to build a pycocotools-style CocoMetrics."""
    return CocoMetrics(
        mAP_50_95=map_50_95,
        mAP_50=map_50,
        mAP_75=map_50,
        mAP_per_class=per_class,
        backend="pycocotools",
        n_images=10,
        n_predictions=10,
        n_ground_truth=10,
    )


def _simple_metrics(f1: float, per_class_f1: dict[str, float]) -> CocoMetrics:
    return CocoMetrics(
        mAP_50_95=0.0,
        mAP_50=0.0,
        mAP_75=0.0,
        mAP_per_class=per_class_f1,
        precision=0.9,
        recall=0.9,
        f1=f1,
        iou_mean=0.9,
        backend="simple",
        n_images=4,
        n_predictions=8,
        n_ground_truth=8,
    )


def test_compare_metrics_pycocotools_drop():
    ref = _pyc_metrics(0.50, 0.71, {"person": 0.55, "car": 0.45})
    cand = _pyc_metrics(0.45, 0.65, {"person": 0.50, "car": 0.40})

    delta = compare_metrics(ref, cand, reference_label="fp32", candidate_label="int8")
    assert delta.backend == "pycocotools"
    assert delta.mAP_50_95_drop == pytest.approx(-0.05)
    assert delta.mAP_50_drop == pytest.approx(-0.06)
    assert delta.retained_mAP_50_95_pct == pytest.approx(90.0)


def test_compare_metrics_simple_uses_f1_in_50_slot():
    ref = _simple_metrics(f1=0.85, per_class_f1={"person": 0.9, "car": 0.8})
    cand = _simple_metrics(f1=0.65, per_class_f1={"person": 0.7, "car": 0.6})

    delta = compare_metrics(ref, cand)
    assert delta.backend == "simple"
    # mAP_50 slot carries F1 for the simple backend.
    assert delta.overall_mAP_50_ref == pytest.approx(0.85)
    assert delta.overall_mAP_50_cand == pytest.approx(0.65)
    # The 0.5:0.95 slot is zero for simple backend.
    assert delta.overall_mAP_50_95_ref == 0.0
    assert delta.overall_mAP_50_95_cand == 0.0


def test_compare_metrics_rejects_backend_mismatch():
    ref = _pyc_metrics(0.5, 0.7, {"a": 0.5})
    cand = _simple_metrics(f1=0.5, per_class_f1={"a": 0.5})
    with pytest.raises(ValueError, match="Backend mismatch"):
        compare_metrics(ref, cand)


def test_compare_metrics_handles_class_union():
    """Classes present in only one of the two metrics should still surface."""
    ref = _pyc_metrics(0.5, 0.7, {"a": 0.6, "b": 0.5})
    cand = _pyc_metrics(0.4, 0.6, {"a": 0.5, "c": 0.4})
    delta = compare_metrics(ref, cand)
    labels = {d.label for d in delta.per_class}
    assert labels == {"a", "b", "c"}


# --------------------------------------------------------------------------- ranking


def test_worst_classes_are_largest_drop_first():
    ref = _pyc_metrics(
        0.5,
        0.7,
        {"untouched": 0.6, "small_drop": 0.6, "medium_drop": 0.6, "big_drop": 0.6},
    )
    cand = _pyc_metrics(
        0.4,
        0.6,
        {
            "untouched": 0.6,
            "small_drop": 0.55,
            "medium_drop": 0.4,
            "big_drop": 0.1,
        },
    )

    delta = compare_metrics(ref, cand)
    worst = delta.worst_classes(2)
    assert [d.label for d in worst] == ["big_drop", "medium_drop"]


def test_best_classes_are_smallest_drop_first():
    ref = _pyc_metrics(0.5, 0.7, {"a": 0.6, "b": 0.6})
    cand = _pyc_metrics(0.4, 0.6, {"a": 0.55, "b": 0.4})
    delta = compare_metrics(ref, cand)
    best = delta.best_classes(2)
    assert [d.label for d in best] == ["a", "b"]


def test_default_top_n_caps_at_list_length():
    ref = _pyc_metrics(0.5, 0.7, {"a": 0.6, "b": 0.6})
    cand = _pyc_metrics(0.4, 0.6, {"a": 0.5, "b": 0.55})
    delta = compare_metrics(ref, cand)
    assert len(delta.worst_classes(99)) == 2


# --------------------------------------------------------------------------- rendering


def test_summary_table_contains_class_rows():
    ref = _pyc_metrics(0.5, 0.7, {"person": 0.55})
    cand = _pyc_metrics(0.45, 0.65, {"person": 0.50})
    rendered = summary_table(compare_metrics(ref, cand))
    assert "Backend" in rendered
    assert "person" in rendered
    assert "delta=" in rendered


def test_summary_table_shows_simple_backend_headline():
    ref = _simple_metrics(f1=0.85, per_class_f1={"a": 0.9})
    cand = _simple_metrics(f1=0.75, per_class_f1={"a": 0.8})
    rendered = summary_table(compare_metrics(ref, cand))
    assert "F1" in rendered  # simple backend headline mentions F1


def test_as_dict_is_json_serialisable():
    ref = _pyc_metrics(0.5, 0.7, {"person": 0.55})
    cand = _pyc_metrics(0.45, 0.65, {"person": 0.50})
    payload = compare_metrics(ref, cand).as_dict()
    json.loads(json.dumps(payload))
    assert payload["n_classes"] == 1
    assert payload["per_class"][0]["label"] == "person"


def test_quantization_delta_zero_reference_retained_is_nan():
    """When the reference scored 0.0 there's no meaningful 'retained pct'."""
    ref = _pyc_metrics(0.0, 0.0, {"a": 0.0})
    cand = _pyc_metrics(0.0, 0.0, {"a": 0.0})
    delta = compare_metrics(ref, cand)
    assert delta.retained_mAP_50_95_pct != delta.retained_mAP_50_95_pct
    payload = delta.as_dict()
    assert payload["retained_mAP_50_95_pct"] is None


def test_quantization_delta_dataclass_fields_round_trip():
    """Construct the dataclass directly to exercise its public fields."""
    delta = QuantizationDelta(
        backend="pycocotools",
        reference_label="fp32",
        candidate_label="int8",
        overall_mAP_50_95_ref=0.5,
        overall_mAP_50_95_cand=0.45,
        overall_mAP_50_ref=0.7,
        overall_mAP_50_cand=0.65,
        per_class=[PerClassDelta("a", 0.6, 0.55)],
    )
    assert delta.mAP_50_95_drop == pytest.approx(-0.05)
    assert delta.retained_mAP_50_95_pct == pytest.approx(90.0)
