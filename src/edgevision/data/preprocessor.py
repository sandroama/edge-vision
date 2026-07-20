"""RT-DETR input preprocessing.

The real RT-DETR via HF ``transformers.RTDetrImageProcessor`` handles its own
preprocessing (resize → normalize → pad). For mock and CI paths we keep a
dependency-free numpy-only fallback so tests don't pull torchvision.

This module's contract:
    preprocess_image_pil(pil_image, target_size) -> numpy array (H,W,3) float32 in [0,1]
    letterbox_resize(image, target_size) -> resized image + the (scale, pad) used
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ImageNet normalization — RT-DETR uses these.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class LetterboxParams:
    """Parameters used to letterbox-resize an image — needed to map model
    coordinates back to pixel space."""

    scale: float
    pad_x: int
    pad_y: int
    target_h: int
    target_w: int
    src_h: int
    src_w: int


def letterbox_resize(
    rgb: np.ndarray, target_size: tuple[int, int] = (640, 640)
) -> tuple[np.ndarray, LetterboxParams]:
    """Resize an RGB array to ``target_size`` preserving aspect ratio.

    Smaller dimension is zero-padded to keep the model input rectangular.
    Returns the resized image + parameters for un-letterboxing detections.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) RGB array, got shape {rgb.shape}")

    src_h, src_w = rgb.shape[:2]
    th, tw = target_size
    scale = min(tw / src_w, th / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    # Nearest-neighbour resize via stride indexing — keeps deps zero.
    # Acceptable for the mock path; the real path uses PIL/torchvision.
    if new_w == src_w and new_h == src_h:
        resized = rgb
    else:
        ys = (np.arange(new_h) * src_h / new_h).astype(np.int64)
        xs = (np.arange(new_w) * src_w / new_w).astype(np.int64)
        resized = rgb[ys][:, xs]

    pad_x = (tw - new_w) // 2
    pad_y = (th - new_h) // 2
    out = np.zeros((th, tw, 3), dtype=rgb.dtype)
    out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return out, LetterboxParams(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        target_h=th,
        target_w=tw,
        src_h=src_h,
        src_w=src_w,
    )


def normalize_imagenet(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB → float32 normalized with ImageNet stats."""
    if rgb.dtype != np.uint8:
        # Already float — just rescale to [0,1] if it looks like it's in [0,255]
        x = rgb.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
    else:
        x = rgb.astype(np.float32) / 255.0
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


def preprocess_for_rtdetr(
    rgb: np.ndarray, target_size: tuple[int, int] = (640, 640)
) -> tuple[np.ndarray, LetterboxParams]:
    """Full pipeline: letterbox resize → ImageNet normalize → CHW float32.

    Returns (CHW tensor as numpy, letterbox params for un-projecting boxes).
    """
    letterboxed, params = letterbox_resize(rgb, target_size)
    normalized = normalize_imagenet(letterboxed)
    chw = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
    return chw.astype(np.float32), params


def unletterbox_box(
    box_xyxy: tuple[float, float, float, float], params: LetterboxParams
) -> tuple[float, float, float, float]:
    """Map a model-space xyxy box back to source-image pixel coords."""
    x1, y1, x2, y2 = box_xyxy
    x1 = (x1 - params.pad_x) / params.scale
    y1 = (y1 - params.pad_y) / params.scale
    x2 = (x2 - params.pad_x) / params.scale
    y2 = (y2 - params.pad_y) / params.scale
    # Clip to source-image bounds.
    x1 = max(0.0, min(float(params.src_w), x1))
    y1 = max(0.0, min(float(params.src_h), y1))
    x2 = max(0.0, min(float(params.src_w), x2))
    y2 = max(0.0, min(float(params.src_h), y2))
    return (x1, y1, x2, y2)
