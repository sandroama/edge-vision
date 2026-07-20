"""Tests for ``edgevision.models.rtdetr_wrapper`` — the mock backend.

The real ``RTDetrDetector`` is exercised in Phase 2/3 GPU runs, not in CI
(it would require pulling 200+ MB of HF weights and torch GPU bits).
"""

from __future__ import annotations

import pytest

from edgevision.data.coco_loader import CocoDataset, gt_for_images
from edgevision.models.rtdetr_wrapper import (
    MockRTDetrDetector,
    make_detector,
)


def test_mock_with_perfect_recall_returns_all_gt():
    ds = CocoDataset.synthetic(n_images=3, n_classes=2, boxes_per_image=2)
    gt = gt_for_images(ds, ds.images)
    detector = MockRTDetrDetector(gt_by_image_id=gt, recall=1.0, false_positive_rate=0.0)

    total_dets = 0
    for img in ds.images:
        out = detector.predict(img)
        total_dets += len(out)
        assert out.image_id == img.image_id
        # Every detection bbox is exactly a GT bbox.
        gt_boxes = {g.bbox for g in gt[img.image_id]}
        for det in out.detections:
            assert det.bbox in gt_boxes
            assert det.confidence == 0.92

    assert total_dets == len(ds.annotations)


def test_mock_recall_drops_some():
    ds = CocoDataset.synthetic(n_images=4, n_classes=2, boxes_per_image=3)
    gt = gt_for_images(ds, ds.images)
    detector = MockRTDetrDetector(
        gt_by_image_id=gt, recall=0.5, false_positive_rate=0.0, seed=42
    )
    total_dets = sum(len(detector.predict(img)) for img in ds.images)
    assert total_dets < len(ds.annotations)
    # Half-ish (but rng-dependent) — just verify it's strictly less than full
    # and strictly greater than zero.
    assert total_dets > 0


def test_mock_false_positives_can_inflate_count():
    ds = CocoDataset.synthetic(n_images=4, n_classes=2, boxes_per_image=2)
    gt = gt_for_images(ds, ds.images)
    detector = MockRTDetrDetector(
        gt_by_image_id=gt, recall=1.0, false_positive_rate=1.0, seed=0
    )
    total_dets = sum(len(detector.predict(img)) for img in ds.images)
    # With FP rate of 1.0, every TP is followed by a noise detection => 2x.
    assert total_dets == 2 * len(ds.annotations)


def test_mock_without_gt_emits_one_detection():
    ds = CocoDataset.synthetic(n_images=2, n_classes=1, boxes_per_image=1)
    detector = MockRTDetrDetector(gt_by_image_id=None)
    for img in ds.images:
        out = detector.predict(img)
        assert len(out) == 1
        assert out.detections[0].label == "synthetic"


def test_mock_is_deterministic_with_same_seed():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=2, seed=0)
    gt = gt_for_images(ds, ds.images)
    a = MockRTDetrDetector(gt_by_image_id=gt, recall=0.7, seed=123)
    b = MockRTDetrDetector(gt_by_image_id=gt, recall=0.7, seed=123)
    out_a = [a.predict(img).detections for img in ds.images]
    out_b = [b.predict(img).detections for img in ds.images]
    assert out_a == out_b


def test_make_detector_factory_returns_mock():
    det = make_detector("mock")
    assert isinstance(det, MockRTDetrDetector)


def test_make_detector_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        make_detector("yolo")  # type: ignore[arg-type]
