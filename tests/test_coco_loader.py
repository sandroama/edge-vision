"""Tests for ``edgevision.data.coco_loader``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgevision.data.coco_loader import CocoDataset, gt_for_images


def test_synthetic_dataset_has_expected_shape():
    ds = CocoDataset.synthetic(n_images=4, n_classes=3, boxes_per_image=2, seed=0)
    assert len(ds) == 4
    assert len(ds.categories) == 3
    assert len(ds.annotations) == 4 * 2

    # Image ids are 1..N.
    assert {img.image_id for img in ds.images} == {1, 2, 3, 4}

    # Every annotation references a valid image.
    image_ids = {img.image_id for img in ds.images}
    for ann in ds.annotations:
        assert ann.image_id in image_ids
        assert ann.bbox.area > 0
        assert ann.class_id is not None


def test_synthetic_dataset_is_deterministic():
    a = CocoDataset.synthetic(seed=42)
    b = CocoDataset.synthetic(seed=42)
    boxes_a = [(g.image_id, g.label, g.bbox) for g in a.annotations]
    boxes_b = [(g.image_id, g.label, g.bbox) for g in b.annotations]
    assert boxes_a == boxes_b


def test_to_coco_dict_roundtrip(tmp_path: Path):
    """Synthetic dataset -> coco json -> reload yields equivalent annotations."""
    ds = CocoDataset.synthetic(n_images=3, n_classes=2, boxes_per_image=2)
    payload = ds.to_coco_dict()

    # The raw JSON must be COCO-compatible: required top-level keys.
    assert {"images", "annotations", "categories"} <= set(payload.keys())
    assert all("id" in c for c in payload["categories"])
    assert all("bbox" in a and "image_id" in a for a in payload["annotations"])

    json_path = tmp_path / "instances_synth.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    with json_path.open("w") as f:
        json.dump(payload, f)

    reloaded = CocoDataset.from_json(json_path, images_dir, split="synth")
    assert len(reloaded.images) == len(ds.images)
    assert len(reloaded.annotations) == len(ds.annotations)
    assert {c.name for c in reloaded.categories} == {c.name for c in ds.categories}


def test_gt_for_images_groups_correctly():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=2)
    grouped = gt_for_images(ds, ds.images)
    assert set(grouped.keys()) == {img.image_id for img in ds.images}
    total = sum(len(v) for v in grouped.values())
    assert total == len(ds.annotations)


def test_from_json_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CocoDataset.from_json(tmp_path / "missing.json", tmp_path)


def test_max_images_truncates(tmp_path: Path):
    ds = CocoDataset.synthetic(n_images=8, boxes_per_image=1)
    payload = ds.to_coco_dict()
    json_path = tmp_path / "instances.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    with json_path.open("w") as f:
        json.dump(payload, f)

    truncated = CocoDataset.from_json(json_path, images_dir, max_images=3)
    assert len(truncated.images) == 3
    # Annotations only retained for kept images.
    assert all(a.image_id in {img.image_id for img in truncated.images} for a in truncated.annotations)
