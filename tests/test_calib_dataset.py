"""Tests for ``edgevision.quantization.calib_dataset``.

CPU-only — no torch / onnxruntime / tensorrt required.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgevision.data.coco_loader import CocoDataset
from edgevision.quantization.calib_dataset import (
    BatchProvider,
    build_calibration_provider,
    estimate_diversity,
    filter_existing_paths,
    load_image_rgb_synthetic,
    select_calibration_images,
)

# --------------------------------------------------------------------------- selection


def test_uniform_selection_returns_n_images():
    ds = CocoDataset.synthetic(n_images=20, boxes_per_image=2)
    chosen = select_calibration_images(ds, n=5, strategy="uniform", seed=0)
    assert len(chosen) == 5
    # All chosen images come from the dataset.
    ids = {img.image_id for img in ds.images}
    assert all(img.image_id in ids for img in chosen)


def test_uniform_selection_is_deterministic_with_seed():
    ds = CocoDataset.synthetic(n_images=20, boxes_per_image=2)
    a = select_calibration_images(ds, n=5, strategy="uniform", seed=42)
    b = select_calibration_images(ds, n=5, strategy="uniform", seed=42)
    assert [img.image_id for img in a] == [img.image_id for img in b]


def test_first_strategy_returns_in_order():
    ds = CocoDataset.synthetic(n_images=10, boxes_per_image=1)
    chosen = select_calibration_images(ds, n=4, strategy="first")
    assert [img.image_id for img in chosen] == [img.image_id for img in ds.images[:4]]


def test_stratified_covers_all_classes_when_possible():
    """Every class should be represented at least once if N >= n_classes."""
    ds = CocoDataset.synthetic(n_images=12, n_classes=3, boxes_per_image=1, seed=0)
    chosen = select_calibration_images(ds, n=6, strategy="stratified", seed=0)
    chosen_ids = {img.image_id for img in chosen}

    # Build a class-coverage view of the chosen set.
    classes_seen: set[int] = set()
    for ann in ds.annotations:
        if ann.image_id in chosen_ids:
            classes_seen.add(ann.class_id or -1)
    expected = {c.id for c in ds.categories}
    assert classes_seen == expected


def test_n_clamped_to_dataset_size():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=1)
    chosen = select_calibration_images(ds, n=100, strategy="uniform")
    assert len(chosen) == 4


def test_invalid_n_rejected():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=1)
    with pytest.raises(ValueError, match="n must be"):
        select_calibration_images(ds, n=0)


def test_unknown_strategy_rejected():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=1)
    with pytest.raises(ValueError, match="Unknown strategy"):
        select_calibration_images(ds, n=2, strategy="kmeans")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- BatchProvider


def test_batch_provider_emits_chw_float32_batches():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=1)
    provider = BatchProvider(
        images=ds.images,
        loader=load_image_rgb_synthetic,
        target_size=(160, 160),
        batch_size=2,
    )

    batches = list(provider)
    # 4 images / batch=2 = 2 batches.
    assert len(batches) == 2
    for b in batches:
        assert b.dtype == np.float32
        assert b.ndim == 4
        assert b.shape[1] == 3
        assert b.shape[2:] == (160, 160)


def test_batch_provider_handles_partial_last_batch():
    ds = CocoDataset.synthetic(n_images=5, boxes_per_image=1)
    provider = BatchProvider(
        images=ds.images,
        loader=load_image_rgb_synthetic,
        target_size=(160, 160),
        batch_size=2,
    )
    batches = list(provider)
    # 5 / 2 -> 3 batches: 2, 2, 1.
    assert len(batches) == 3
    assert batches[-1].shape[0] == 1


def test_batch_provider_len_matches_iter_count():
    ds = CocoDataset.synthetic(n_images=7, boxes_per_image=1)
    provider = BatchProvider(
        images=ds.images,
        loader=load_image_rgb_synthetic,
        target_size=(160, 160),
        batch_size=3,
    )
    assert len(provider) == 3
    assert sum(1 for _ in provider) == 3


def test_batch_provider_reset_allows_replay():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=1)
    provider = BatchProvider(
        images=ds.images,
        loader=load_image_rgb_synthetic,
        target_size=(160, 160),
        batch_size=2,
    )
    first = list(provider)
    provider.reset()
    second = list(provider)
    assert len(first) == len(second)
    np.testing.assert_array_equal(first[0], second[0])


def test_batch_provider_total_images():
    ds = CocoDataset.synthetic(n_images=6, boxes_per_image=1)
    provider = BatchProvider(
        images=ds.images,
        loader=load_image_rgb_synthetic,
        target_size=(160, 160),
        batch_size=2,
    )
    assert provider.total_images == 6


def test_load_image_rgb_synthetic_is_deterministic_per_image():
    ds = CocoDataset.synthetic(n_images=2, boxes_per_image=1)
    a = load_image_rgb_synthetic(ds.images[0])
    b = load_image_rgb_synthetic(ds.images[0])
    np.testing.assert_array_equal(a, b)


def test_load_image_rgb_synthetic_differs_across_images():
    ds = CocoDataset.synthetic(n_images=2, boxes_per_image=1)
    a = load_image_rgb_synthetic(ds.images[0])
    b = load_image_rgb_synthetic(ds.images[1])
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------- factory


def test_factory_uses_synthetic_loader_when_no_paths():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=1)
    # Synthetic dataset has all path=None, so factory should pick the
    # synthetic loader.
    provider = build_calibration_provider(ds, n=3, target_size=(160, 160))
    batches = list(provider)
    assert len(batches) == 3
    assert provider.total_images == 3


# --------------------------------------------------------------------------- diagnostics


def test_estimate_diversity_returns_finite_stats():
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=1)
    provider = build_calibration_provider(ds, n=4, target_size=(160, 160))
    stats = estimate_diversity(provider, max_batches=4)
    assert set(stats.keys()) == {"mean_r", "mean_g", "mean_b", "std_mean"}
    assert all(np.isfinite(v) for v in stats.values())


def test_filter_existing_paths_drops_missing(tmp_path):
    # Mark one image with a real path that does NOT exist.
    from edgevision.schemas import Image

    img_real = tmp_path / "real.jpg"
    img_real.write_bytes(b"x")  # not a real JPEG; existence is what we check.

    images = [
        Image(image_id=1, width=10, height=10, file_name="real.jpg", path=str(img_real)),
        Image(image_id=2, width=10, height=10, file_name="missing.jpg", path=str(tmp_path / "nope.jpg")),
        Image(image_id=3, width=10, height=10, file_name="no_path.jpg", path=None),
    ]
    kept = filter_existing_paths(images)
    assert {img.image_id for img in kept} == {1}
