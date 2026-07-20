"""Shared schemas — the vocabulary every module speaks.

Kept dependency-light (stdlib only) so this module imports clean in any context,
including the no-torch CPU runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coords (xyxy, top-left origin)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """COCO results format: [x_topleft, y_topleft, w, h]."""
        return (self.x1, self.y1, self.width, self.height)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BoundingBox:
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)


@dataclass(frozen=True)
class Detection:
    """A single detection: bbox + label + confidence + class id."""

    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int | None = None


@dataclass(frozen=True)
class GroundTruthBox:
    """Reference annotation for one object in one image."""

    image_id: int
    label: str
    bbox: BoundingBox
    class_id: int | None = None
    is_crowd: bool = False


@dataclass(frozen=True)
class Image:
    """Image metadata + optional path. The actual pixel array stays out of the
    schema so we can pass these around cheaply.
    """

    image_id: int
    width: int
    height: int
    file_name: str
    path: str | None = None  # absolute path on disk; None for synthetic

    @property
    def shape_hw(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass
class ImageDetections:
    """All detections for one image."""

    image_id: int
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float | None = None  # populated by the latency harness

    def __len__(self) -> int:
        return len(self.detections)
