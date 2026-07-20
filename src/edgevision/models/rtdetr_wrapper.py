"""RT-DETR detector wrappers.

Two implementations behind one interface:
    1. ``MockRTDetrDetector`` — deterministic, offline, zero heavy deps.
       Reproduces a synthetic dataset's ground truth (with controllable
       noise / drop-rate) so eval-pipeline tests can target known mAP.
    2. ``RTDetrDetector`` — real HF ``transformers.RTDetrForObjectDetection``
       backed by PyTorch. Lazy-imported so the module is safe to import on a
       no-torch CI runner.

Both return ``ImageDetections`` with detection bboxes in **source-image pixel
coordinates** (i.e. already un-letterboxed).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from edgevision.schemas import (
    BoundingBox,
    Detection,
    GroundTruthBox,
    Image,
    ImageDetections,
)


class Detector(Protocol):
    """Common detector interface (Phase 1 — extends with batch APIs in Phase 2)."""

    def predict(self, image: Image) -> ImageDetections: ...


# --------------------------------------------------------------------------- mock


@dataclass
class MockRTDetrDetector:
    """Deterministic mock — replays a known ground truth as detections.

    Args:
        gt_by_image_id: optional ground-truth map. If provided, the mock
            returns each GT box as a Detection (with ``confidence``). If not
            provided, the mock returns a single fixed detection per image so
            tests don't need to wire up GT.
        confidence: confidence assigned to every emitted detection.
        recall: fraction of GT boxes to keep (1.0 = perfect, 0.5 = drop half).
        false_positive_rate: probability of inserting a noise detection per GT.
        seed: rng seed for reproducible drops/noise.
    """

    gt_by_image_id: dict[int, list[GroundTruthBox]] | None = None
    confidence: float = 0.92
    recall: float = 1.0
    false_positive_rate: float = 0.0
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def predict(self, image: Image) -> ImageDetections:
        detections: list[Detection] = []
        gts = (self.gt_by_image_id or {}).get(image.image_id, [])

        if not gts:
            # No GT — emit one centered detection so the pipeline still has
            # something to evaluate against.
            cx, cy = image.width / 2, image.height / 2
            box = BoundingBox(cx - 32, cy - 32, cx + 32, cy + 32)
            detections.append(
                Detection(
                    label="synthetic",
                    confidence=self.confidence,
                    bbox=box,
                    class_id=1,
                )
            )
        else:
            for gt in gts:
                if self._rng.random() > self.recall:
                    continue
                detections.append(
                    Detection(
                        label=gt.label,
                        confidence=self.confidence,
                        bbox=gt.bbox,
                        class_id=gt.class_id,
                    )
                )
                if self._rng.random() < self.false_positive_rate:
                    # Insert a slightly-shifted false positive.
                    shift = 50.0
                    fp = BoundingBox(
                        gt.bbox.x1 + shift,
                        gt.bbox.y1 + shift,
                        gt.bbox.x2 + shift,
                        gt.bbox.y2 + shift,
                    )
                    detections.append(
                        Detection(
                            label=gt.label,
                            confidence=self.confidence * 0.5,
                            bbox=fp,
                            class_id=gt.class_id,
                        )
                    )

        return ImageDetections(image_id=image.image_id, detections=detections)


# --------------------------------------------------------------------------- real


class RTDetrDetector:
    """Real RT-DETR via Hugging Face ``transformers``.

    Heavy deps (torch + transformers + Pillow) are imported lazily inside
    ``__init__`` so the module is safe to import on a no-torch CI runner.

    Usage:
        det = RTDetrDetector(model_name="PekingU/rtdetr_r50vd_coco_o365")
        result = det.predict(image)  # image.path must be set

    Args:
        model_name: HF model id.
        device: "auto" | "cuda" | "cpu" | "mps".
        confidence_threshold: filter low-confidence preds before returning.
    """

    def __init__(
        self,
        model_name: str = "PekingU/rtdetr_r50vd_coco_o365",
        device: str = "auto",
        confidence_threshold: float = 0.30,
    ) -> None:
        try:
            import torch
            from PIL import Image as PILImage  # noqa: F401  (used in predict)
            from transformers import (
                RTDetrForObjectDetection,
                RTDetrImageProcessor,
            )
        except ImportError as e:  # pragma: no cover - exercised only when deps absent
            raise ImportError(
                "RTDetrDetector requires torch + transformers + Pillow. "
                "Install with: pip install -e '.[dev]'"
            ) from e

        self._torch = torch
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.model = (
            RTDetrForObjectDetection.from_pretrained(model_name).to(device).eval()
        )
        self._id2label: dict[int, str] = self.model.config.id2label

    def predict(self, image: Image) -> ImageDetections:
        if image.path is None:
            raise ValueError(
                f"RTDetrDetector requires image.path to be set (image_id={image.image_id})"
            )

        from PIL import Image as PILImage

        pil_img = PILImage.open(image.path).convert("RGB")
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self._torch.tensor([(image.height, image.width)], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold,
        )[0]

        detections: list[Detection] = []
        for score, label_id, box in zip(
            results["scores"].cpu().tolist(),
            results["labels"].cpu().tolist(),
            results["boxes"].cpu().tolist(),
            strict=True,
        ):
            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    label=self._id2label.get(int(label_id), str(label_id)),
                    confidence=float(score),
                    bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                    class_id=int(label_id),
                )
            )

        return ImageDetections(image_id=image.image_id, detections=detections)


def make_detector(
    backend: str = "mock",
    *,
    gt_by_image_id: dict[int, list[GroundTruthBox]] | None = None,
    model_name: str = "PekingU/rtdetr_r50vd_coco_o365",
    device: str = "auto",
    confidence_threshold: float = 0.30,
) -> Detector:
    """Factory — picks ``MockRTDetrDetector`` or ``RTDetrDetector``.

    Use ``backend="mock"`` in tests / CI / smoke; ``backend="rtdetr"`` for the
    real Phase-1 baseline reproduction on COCO val2017.
    """
    if backend == "mock":
        return MockRTDetrDetector(gt_by_image_id=gt_by_image_id)
    if backend == "rtdetr":
        return RTDetrDetector(
            model_name=model_name,
            device=device,
            confidence_threshold=confidence_threshold,
        )
    raise ValueError(f"Unknown backend: {backend!r} (use 'mock' or 'rtdetr')")
