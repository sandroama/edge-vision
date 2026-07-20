"""COCO dataset loader.

Two modes:
    1. Real — point at a COCO-format ``instances_<split>.json`` + image dir.
       Used by the Phase 1 full baseline run on COCO val2017.
    2. Synthetic — programmatically build a tiny COCO-format dataset in memory
       (no disk, no downloads). Used by tests and the smoke script.

The loader deliberately does NOT decode pixels. It hands back ``Image``
metadata + ``GroundTruthBox`` annotations; the model wrapper opens the actual
pixels lazily, only for images it'll run inference on. Keeps memory bounded.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from edgevision.schemas import BoundingBox, GroundTruthBox, Image


@dataclass
class CocoCategory:
    id: int
    name: str
    supercategory: str | None = None


@dataclass
class CocoDataset:
    """In-memory COCO dataset — images + annotations + category map.

    Use ``CocoDataset.from_json`` to load a real split, or
    ``CocoDataset.synthetic`` to build a tiny dataset for tests.
    """

    images: list[Image]
    annotations: list[GroundTruthBox]
    categories: list[CocoCategory]
    images_dir: str | None = None
    split: str = "unknown"
    _id_to_image: dict[int, Image] = field(default_factory=dict, init=False, repr=False)
    _id_to_category: dict[int, CocoCategory] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_image = {img.image_id: img for img in self.images}
        self._id_to_category = {c.id: c for c in self.categories}

    # ------------------------------------------------------------------ access

    def get_image(self, image_id: int) -> Image:
        return self._id_to_image[image_id]

    def get_category(self, class_id: int) -> CocoCategory:
        return self._id_to_category[class_id]

    def annotations_for_image(self, image_id: int) -> list[GroundTruthBox]:
        return [a for a in self.annotations if a.image_id == image_id]

    def category_id_to_name(self) -> dict[int, str]:
        return {c.id: c.name for c in self.categories}

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Iterator[Image]:
        return iter(self.images)

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_json(
        cls,
        annotations_json: str | Path,
        images_dir: str | Path,
        split: str = "val2017",
        max_images: int | None = None,
    ) -> CocoDataset:
        """Load a real COCO split from ``instances_<split>.json``.

        Pass ``max_images`` to truncate (useful for the smoke run on val2017).
        """
        annotations_json = Path(annotations_json)
        images_dir = Path(images_dir)
        if not annotations_json.exists():
            raise FileNotFoundError(
                f"COCO annotations not found at {annotations_json}. "
                "Download from http://images.cocodataset.org/annotations/"
                "annotations_trainval2017.zip"
            )

        with annotations_json.open() as f:
            payload = json.load(f)

        categories = [
            CocoCategory(id=c["id"], name=c["name"], supercategory=c.get("supercategory"))
            for c in payload.get("categories", [])
        ]

        raw_images = payload.get("images", [])
        if max_images is not None:
            raw_images = raw_images[:max_images]
        keep_ids = {img["id"] for img in raw_images}

        images = [
            Image(
                image_id=img["id"],
                width=img["width"],
                height=img["height"],
                file_name=img["file_name"],
                path=str(images_dir / img["file_name"]),
            )
            for img in raw_images
        ]

        annotations: list[GroundTruthBox] = []
        cat_id_to_name = {c.id: c.name for c in categories}
        for a in payload.get("annotations", []):
            if a["image_id"] not in keep_ids:
                continue
            x, y, w, h = a["bbox"]
            annotations.append(
                GroundTruthBox(
                    image_id=a["image_id"],
                    label=cat_id_to_name.get(a["category_id"], str(a["category_id"])),
                    class_id=a["category_id"],
                    bbox=BoundingBox.from_xywh(x, y, w, h),
                    is_crowd=bool(a.get("iscrowd", 0)),
                )
            )

        return cls(
            images=images,
            annotations=annotations,
            categories=categories,
            images_dir=str(images_dir),
            split=split,
        )

    @classmethod
    def synthetic(
        cls,
        n_images: int = 4,
        n_classes: int = 3,
        boxes_per_image: int = 2,
        image_size: tuple[int, int] = (640, 640),
        seed: int = 0,
    ) -> CocoDataset:
        """Build a deterministic synthetic COCO-format dataset.

        No images on disk — used by tests and the smoke run. The generated
        ground-truth boxes are placed on a grid so that a "good" detector
        (the deterministic mock) can recover them exactly.
        """
        rng = random.Random(seed)
        h, w = image_size
        categories = [
            CocoCategory(id=i + 1, name=f"class_{i:02d}", supercategory="synthetic")
            for i in range(n_classes)
        ]
        images: list[Image] = []
        annotations: list[GroundTruthBox] = []

        for img_idx in range(n_images):
            image_id = img_idx + 1  # COCO ids are 1-indexed by convention
            images.append(
                Image(
                    image_id=image_id,
                    width=w,
                    height=h,
                    file_name=f"synth_{image_id:08d}.jpg",
                    path=None,
                )
            )

            # Tile boxes deterministically across the image so the mock
            # detector can reproduce them exactly.
            for b in range(boxes_per_image):
                cls_idx = (img_idx + b) % n_classes
                cat = categories[cls_idx]
                cell_w = w // (boxes_per_image + 1)
                cell_h = h // 2
                x1 = (b + 1) * cell_w - cell_w // 2 + rng.randint(-4, 4)
                y1 = h // 4 + rng.randint(-4, 4)
                x2 = x1 + cell_w - 8
                y2 = y1 + cell_h - 8
                annotations.append(
                    GroundTruthBox(
                        image_id=image_id,
                        label=cat.name,
                        class_id=cat.id,
                        bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                    )
                )

        return cls(
            images=images,
            annotations=annotations,
            categories=categories,
            images_dir=None,
            split="synthetic",
        )

    # ------------------------------------------------------------------ export

    def to_coco_dict(self) -> dict:
        """Serialise back to the COCO json schema (used by pycocotools)."""
        return {
            "info": {"description": f"edge-vision {self.split}"},
            "licenses": [],
            "categories": [
                {
                    "id": c.id,
                    "name": c.name,
                    "supercategory": c.supercategory or "none",
                }
                for c in self.categories
            ],
            "images": [
                {
                    "id": img.image_id,
                    "file_name": img.file_name,
                    "width": img.width,
                    "height": img.height,
                }
                for img in self.images
            ],
            "annotations": [
                {
                    "id": idx + 1,
                    "image_id": ann.image_id,
                    "category_id": (
                        ann.class_id
                        if ann.class_id is not None
                        else self._lookup_class_id_by_label(ann.label)
                    ),
                    "bbox": list(ann.bbox.to_xywh()),
                    "area": ann.bbox.area,
                    "iscrowd": int(ann.is_crowd),
                    "segmentation": [],
                }
                for idx, ann in enumerate(self.annotations)
            ],
        }

    def _lookup_class_id_by_label(self, label: str) -> int:
        for c in self.categories:
            if c.name == label:
                return c.id
        raise KeyError(f"Unknown label '{label}' (categories: {[c.name for c in self.categories]})")


def gt_for_images(
    dataset: CocoDataset, images: Iterable[Image]
) -> dict[int, list[GroundTruthBox]]:
    """Group ground-truth boxes by image id for the given subset."""
    keep_ids = {img.image_id for img in images}
    out: dict[int, list[GroundTruthBox]] = {iid: [] for iid in keep_ids}
    for ann in dataset.annotations:
        if ann.image_id in out:
            out[ann.image_id].append(ann)
    return out
