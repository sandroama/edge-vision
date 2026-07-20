"""Tests for ``edgevision.evaluation.coco_eval``."""

from __future__ import annotations

from edgevision.data.coco_loader import CocoDataset, gt_for_images
from edgevision.evaluation.coco_eval import (
    CocoMetrics,
    evaluate,
    evaluate_simple,
    iou_xyxy,
    summary_table,
)
from edgevision.models.rtdetr_wrapper import MockRTDetrDetector
from edgevision.schemas import BoundingBox

# --------------------------------------------------------------------------- IoU


def test_iou_identical_boxes_is_one():
    a = BoundingBox(0, 0, 10, 10)
    assert iou_xyxy(a, a) == 1.0


def test_iou_disjoint_is_zero():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(20, 20, 30, 30)
    assert iou_xyxy(a, b) == 0.0


def test_iou_half_overlap():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 0, 15, 10)
    # Intersection 5x10=50, union 100+100-50=150 → 1/3.
    assert abs(iou_xyxy(a, b) - 1 / 3) < 1e-6


# --------------------------------------------------------------------------- simple


def test_evaluate_simple_perfect_recall():
    ds = CocoDataset.synthetic(n_images=3, boxes_per_image=2)
    gt = gt_for_images(ds, ds.images)
    detector = MockRTDetrDetector(
        gt_by_image_id=gt, recall=1.0, false_positive_rate=0.0
    )
    preds = [detector.predict(img) for img in ds.images]
    metrics = evaluate_simple(preds, ds, iou_threshold=0.5)

    assert metrics.backend == "simple"
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.iou_mean == 1.0
    assert metrics.n_predictions == metrics.n_ground_truth


def test_evaluate_simple_with_drops_drops_recall():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=3, seed=0)
    gt = gt_for_images(ds, ds.images)
    detector = MockRTDetrDetector(
        gt_by_image_id=gt, recall=0.5, false_positive_rate=0.0, seed=42
    )
    preds = [detector.predict(img) for img in ds.images]
    metrics = evaluate_simple(preds, ds, iou_threshold=0.5)
    assert metrics.recall is not None and metrics.recall < 1.0
    # Drops only -> precision should still be 1.0 (no FPs).
    assert metrics.precision == 1.0


def test_evaluate_simple_with_fps_drops_precision():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=2, seed=0)
    gt = gt_for_images(ds, ds.images)
    detector = MockRTDetrDetector(
        gt_by_image_id=gt, recall=1.0, false_positive_rate=1.0, seed=0
    )
    preds = [detector.predict(img) for img in ds.images]
    metrics = evaluate_simple(preds, ds, iou_threshold=0.5)
    assert metrics.precision is not None and metrics.precision < 1.0


def test_evaluate_simple_per_class_breakdown():
    ds = CocoDataset.synthetic(n_images=4, n_classes=3, boxes_per_image=2)
    gt = gt_for_images(ds, ds.images)
    detector = MockRTDetrDetector(
        gt_by_image_id=gt, recall=1.0, false_positive_rate=0.0
    )
    preds = [detector.predict(img) for img in ds.images]
    metrics = evaluate_simple(preds, ds)
    # Every class should have an F1 of 1.0 with perfect recall + zero FPs.
    assert all(v == 1.0 for v in metrics.mAP_per_class.values())


def test_evaluate_auto_picks_a_backend_and_returns_metrics_shape():
    ds = CocoDataset.synthetic(n_images=2, boxes_per_image=2)
    gt = gt_for_images(ds, ds.images)
    detector = MockRTDetrDetector(
        gt_by_image_id=gt, recall=1.0, false_positive_rate=0.0
    )
    preds = [detector.predict(img) for img in ds.images]
    metrics = evaluate(preds, ds, backend="auto")
    assert isinstance(metrics, CocoMetrics)
    assert metrics.backend in {"pycocotools", "simple"}
    assert metrics.n_images == len(ds.images)
    assert metrics.n_predictions == len(ds.annotations)


def test_summary_table_renders_text():
    ds = CocoDataset.synthetic(n_images=2)
    gt = gt_for_images(ds, ds.images)
    detector = MockRTDetrDetector(gt_by_image_id=gt)
    preds = [detector.predict(img) for img in ds.images]
    metrics = evaluate_simple(preds, ds)
    rendered = summary_table(metrics)
    assert "Backend" in rendered
    assert "Images" in rendered


def test_metrics_as_dict_is_serialisable():
    ds = CocoDataset.synthetic(n_images=2, boxes_per_image=1)
    gt = gt_for_images(ds, ds.images)
    detector = MockRTDetrDetector(gt_by_image_id=gt)
    preds = [detector.predict(img) for img in ds.images]
    metrics = evaluate_simple(preds, ds)
    d = metrics.as_dict()
    assert d["backend"] == "simple"
    assert d["n_images"] == len(ds.images)
    assert "mAP_per_class" in d
