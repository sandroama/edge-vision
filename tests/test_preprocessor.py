"""Tests for ``edgevision.data.preprocessor`` (numpy-only fallback path)."""

from __future__ import annotations

import numpy as np

from edgevision.data.preprocessor import (
    LetterboxParams,
    letterbox_resize,
    normalize_imagenet,
    preprocess_for_rtdetr,
    unletterbox_box,
)


def test_letterbox_preserves_aspect_ratio():
    src = np.zeros((480, 640, 3), dtype=np.uint8)
    out, params = letterbox_resize(src, target_size=(640, 640))
    assert out.shape == (640, 640, 3)
    # 640x480 -> scale to 640x480 fits inside 640x640 with vertical padding.
    assert params.scale == 1.0
    assert params.pad_x == 0
    assert params.pad_y == (640 - 480) // 2


def test_letterbox_handles_square():
    src = np.zeros((320, 320, 3), dtype=np.uint8)
    out, params = letterbox_resize(src, target_size=(640, 640))
    assert out.shape == (640, 640, 3)
    assert params.scale == 2.0


def test_normalize_imagenet_uint8_to_float():
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
    normed = normalize_imagenet(rgb)
    assert normed.dtype == np.float32
    # 128/255 = 0.502; (0.502 - mean) / std should be the same per channel.
    assert normed.shape == (4, 4, 3)


def test_normalize_imagenet_float_input_passes_through():
    rgb = np.full((4, 4, 3), 0.5, dtype=np.float32)
    normed = normalize_imagenet(rgb)
    # 0.5 input values mean small finite normalized values
    assert np.isfinite(normed).all()
    assert normed.shape == (4, 4, 3)


def test_preprocess_for_rtdetr_shape():
    rgb = np.full((360, 480, 3), 200, dtype=np.uint8)
    chw, params = preprocess_for_rtdetr(rgb, target_size=(640, 640))
    assert chw.shape == (3, 640, 640)
    assert chw.dtype == np.float32
    assert isinstance(params, LetterboxParams)


def test_unletterbox_box_inverse_of_letterbox():
    src = np.zeros((480, 640, 3), dtype=np.uint8)
    _, params = letterbox_resize(src, target_size=(640, 640))
    # Place a box at (10,10,100,100) in source coords; project to model coords
    # then back.
    x1, y1, x2, y2 = 10, 10, 100, 100
    mx1 = x1 * params.scale + params.pad_x
    my1 = y1 * params.scale + params.pad_y
    mx2 = x2 * params.scale + params.pad_x
    my2 = y2 * params.scale + params.pad_y
    rx1, ry1, rx2, ry2 = unletterbox_box((mx1, my1, mx2, my2), params)
    assert abs(rx1 - x1) < 1e-3
    assert abs(ry1 - y1) < 1e-3
    assert abs(rx2 - x2) < 1e-3
    assert abs(ry2 - y2) < 1e-3


def test_unletterbox_box_clips_to_source():
    params = LetterboxParams(
        scale=1.0, pad_x=0, pad_y=80, target_h=640, target_w=640, src_h=480, src_w=640
    )
    # A box that lies entirely in the top padding should clip to (0, 0, *, 0).
    x1, y1, x2, y2 = unletterbox_box((100, 10, 200, 50), params)
    assert y1 == 0
    assert y2 == 0  # 50 - 80 = -30 → clipped to 0
