"""Smoke test: package imports cleanly without pulling in heavy ML deps."""

from __future__ import annotations


def test_top_level_import():
    import edgevision

    assert edgevision.__version__ == "0.1.0"


def test_subpackages_importable():
    from edgevision import (
        api,
        compile,
        dashboard,
        data,
        distillation,
        evaluation,
        inference,
        models,
        profiling,
        pruning,
        quantization,
    )

    assert all(
        [
            api,
            compile,
            dashboard,
            data,
            distillation,
            evaluation,
            inference,
            models,
            profiling,
            pruning,
            quantization,
        ]
    )


def test_schemas_minimal():
    from edgevision.schemas import BoundingBox, Detection

    bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
    assert bbox.area == 5000
    assert bbox.width == 100
    assert bbox.height == 50

    det = Detection(label="person", confidence=0.9, bbox=bbox, class_id=0)
    assert det.label == "person"
    assert det.bbox.area == 5000
