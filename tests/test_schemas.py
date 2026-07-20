"""Tests for the shared schema dataclasses."""

from __future__ import annotations

from edgevision.schemas import (
    BoundingBox,
    Detection,
    GroundTruthBox,
    Image,
    ImageDetections,
)


def test_bbox_geometry():
    b = BoundingBox(10.0, 20.0, 110.0, 70.0)
    assert b.width == 100.0
    assert b.height == 50.0
    assert b.area == 5000.0


def test_bbox_xywh_roundtrip():
    b = BoundingBox(5.0, 10.0, 25.0, 40.0)
    x, y, w, h = b.to_xywh()
    assert (x, y, w, h) == (5.0, 10.0, 20.0, 30.0)
    b2 = BoundingBox.from_xywh(x, y, w, h)
    assert b2 == b


def test_bbox_negative_dims_clamped_in_area():
    # Degenerate box (x2 < x1) → area must be 0, not negative.
    b = BoundingBox(50.0, 50.0, 10.0, 10.0)
    assert b.area == 0.0


def test_image_metadata():
    img = Image(image_id=42, width=640, height=480, file_name="foo.jpg")
    assert img.shape_hw == (480, 640)
    assert img.path is None


def test_detection_and_image_detections():
    bbox = BoundingBox(0.0, 0.0, 100.0, 50.0)
    det = Detection(label="person", confidence=0.91, bbox=bbox, class_id=1)
    bag = ImageDetections(image_id=42, detections=[det])
    assert len(bag) == 1
    assert bag.detections[0].label == "person"


def test_ground_truth_box_defaults():
    gt = GroundTruthBox(
        image_id=1,
        label="cat",
        bbox=BoundingBox(0, 0, 10, 10),
        class_id=17,
    )
    assert gt.is_crowd is False
    assert gt.class_id == 17
