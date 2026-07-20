"""Calibration dataset abstraction for INT8 PTQ.

Both TensorRT (``IInt8EntropyCalibrator2``) and ONNX Runtime
(``CalibrationDataReader``) need an iterator that yields preprocessed
input tensors. The shapes match — only the wrapping API differs.

This module owns the *data path*:

    1. Build a ``CalibrationDataset`` from a real ``CocoDataset`` (sampling
       N images stratified by category coverage), or from a synthetic stub
       that emits zero-tensors (useful in CI).
    2. Hand it out as ``BatchProvider`` — a stateful iterator that returns
       ``np.ndarray`` of shape ``(batch, 3, H, W)`` until exhausted.
    3. The TRT calibrator (``trt_int8.py``) and ONNX QDQ data reader
       (``onnx_qdq.py``) both wrap a ``BatchProvider``.

Why size matters: smaller calibration sets are faster but capture less
activation diversity; larger sets give better INT8 fidelity at higher
calibration cost. 100–500 is the field consensus for COCO-class detectors.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from edgevision.data.coco_loader import CocoDataset
from edgevision.data.preprocessor import preprocess_for_rtdetr
from edgevision.schemas import Image

# --------------------------------------------------------------------------- sampling


def select_calibration_images(
    dataset: CocoDataset,
    *,
    n: int = 200,
    strategy: str = "uniform",
    seed: int = 42,
) -> list[Image]:
    """Pick ``n`` images for INT8 calibration.

    Strategies:
        ``"uniform"`` — random sample without replacement. Cheapest, and
            empirically close to stratified for COCO-scale sets.
        ``"stratified"`` — sample so each category is represented at least
            once (best-effort). Falls back to uniform once every category
            has at least one supporter.
        ``"first"`` — take the first ``n`` images (deterministic, useful
            for debugging).
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if n > len(dataset.images):
        n = len(dataset.images)

    rng = random.Random(seed)

    if strategy == "first":
        return dataset.images[:n]
    if strategy == "uniform":
        return rng.sample(dataset.images, n)
    if strategy == "stratified":
        # Group images by which categories appear in their annotations.
        ann_index: dict[int, set[int]] = {}
        for ann in dataset.annotations:
            ann_index.setdefault(ann.image_id, set()).add(ann.class_id or -1)

        category_ids = {c.id for c in dataset.categories}
        seen_categories: set[int] = set()
        chosen: list[Image] = []
        candidates = list(dataset.images)
        rng.shuffle(candidates)

        for img in candidates:
            if len(chosen) >= n:
                break
            new = ann_index.get(img.image_id, set()) - seen_categories
            if new or len(seen_categories) >= len(category_ids):
                chosen.append(img)
                seen_categories.update(ann_index.get(img.image_id, set()))

        # Top up to N from the remaining pool (keeping insertion order).
        remaining = [img for img in candidates if img not in chosen]
        chosen.extend(remaining[: n - len(chosen)])
        return chosen[:n]

    raise ValueError(f"Unknown strategy: {strategy!r}")


# --------------------------------------------------------------------------- batch provider


@dataclass
class BatchProvider:
    """Stateful iterator producing fixed-shape numpy batches for calibration.

    Args:
        images: list of ``Image`` records to draw from.
        loader: callable that takes an ``Image`` and returns an RGB ``ndarray``.
            For real COCO images this opens the file via PIL; for synthetic
            it returns zeros.
        target_size: model input (H, W). Defaults to RT-DETR's 640x640.
        batch_size: preferred batch size; the last batch may be smaller.

    Usage::

        provider = BatchProvider(
            images=select_calibration_images(ds, n=200),
            loader=load_image_rgb,
        )
        for batch in provider:
            calibrator.consume(batch)  # numpy array (B, 3, H, W) float32

    The provider can be reset (``reset()``) to iterate again — TRT calls
    ``get_batch`` until it returns None, then we may want a second pass.
    """

    images: list[Image]
    loader: Callable[[Image], np.ndarray]
    target_size: tuple[int, int] = (640, 640)
    batch_size: int = 1
    _cursor: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._cursor = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        self.reset()
        return self

    def __next__(self) -> np.ndarray:
        if self._cursor >= len(self.images):
            raise StopIteration
        chunk = self.images[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size
        tensors = [self._preprocess(img) for img in chunk]
        # If the chunk is short, return what we have (last partial batch).
        return np.stack(tensors, axis=0).astype(np.float32, copy=False)

    def __len__(self) -> int:
        # Number of batches (rounding up).
        return (len(self.images) + self.batch_size - 1) // self.batch_size

    @property
    def total_images(self) -> int:
        return len(self.images)

    def _preprocess(self, image: Image) -> np.ndarray:
        rgb = self.loader(image)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        if rgb.shape[-1] != 3:
            raise ValueError(f"Expected (H,W,3) RGB; got {rgb.shape}")
        chw, _ = preprocess_for_rtdetr(rgb, target_size=self.target_size)
        return chw


# --------------------------------------------------------------------------- loaders


def load_image_rgb_pil(image: Image) -> np.ndarray:
    """Open ``image.path`` with PIL and return an HxWx3 uint8 RGB array.

    Lazy-imports PIL.
    """
    if image.path is None:
        raise ValueError(f"Image {image.image_id} has no path; use load_image_rgb_synthetic.")
    try:
        from PIL import Image as PILImage
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "load_image_rgb_pil requires Pillow. Install with `pip install Pillow`."
        ) from e

    pil_img = PILImage.open(image.path).convert("RGB")
    return np.asarray(pil_img, dtype=np.uint8)


def load_image_rgb_synthetic(image: Image) -> np.ndarray:
    """Deterministic synthetic image — used in CI and unit tests.

    Returns a deterministic noise pattern keyed off ``image.image_id`` so
    different images produce different activation distributions (which is
    what the calibrator actually needs).
    """
    rng = np.random.default_rng(seed=image.image_id)
    return rng.integers(0, 256, size=(image.height, image.width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- factories


def build_calibration_provider(
    dataset: CocoDataset,
    *,
    n: int = 200,
    strategy: str = "uniform",
    seed: int = 42,
    target_size: tuple[int, int] = (640, 640),
    batch_size: int = 1,
    loader: Callable[[Image], np.ndarray] | None = None,
) -> BatchProvider:
    """One-call factory: dataset → ``BatchProvider`` ready for the calibrator.

    Auto-picks a loader: real-image PIL loader if every chosen image has a
    path, synthetic loader otherwise. Override ``loader`` to force a choice.
    """
    images = select_calibration_images(dataset, n=n, strategy=strategy, seed=seed)
    if loader is None:
        loader = (
            load_image_rgb_pil
            if all(img.path is not None for img in images)
            else load_image_rgb_synthetic
        )
    return BatchProvider(
        images=images,
        loader=loader,
        target_size=target_size,
        batch_size=batch_size,
    )


# --------------------------------------------------------------------------- helpers


def estimate_diversity(provider: BatchProvider, *, max_batches: int = 16) -> dict[str, float]:
    """Cheap activation-diversity proxy: per-channel mean/std across batches.

    A calibration set with all-zero or low-variance images produces poor
    INT8 scales; this helper lets a script sanity-check the set before
    committing to a 30-minute calibration run.
    """
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for i, batch in enumerate(provider):
        if i >= max_batches:
            break
        # Compute mean/std per channel.
        means.append(batch.mean(axis=(0, 2, 3)))
        stds.append(batch.std(axis=(0, 2, 3)))
    if not means:
        return {"mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0, "std_mean": 0.0}
    mean_arr = np.stack(means).mean(axis=0)
    std_arr = np.stack(stds).mean(axis=0)
    return {
        "mean_r": float(mean_arr[0]),
        "mean_g": float(mean_arr[1]),
        "mean_b": float(mean_arr[2]),
        "std_mean": float(std_arr.mean()),
    }


def filter_existing_paths(images: Iterable[Image]) -> list[Image]:
    """Drop images whose ``path`` doesn't exist on disk.

    Useful when the user pulls a partial COCO mirror — calibrating on
    paths that don't load is silent failure otherwise.
    """
    return [img for img in images if img.path is not None and Path(img.path).exists()]
