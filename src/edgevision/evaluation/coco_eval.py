"""COCO mAP evaluation.

Two paths:
    1. ``evaluate_pycocotools`` — uses pycocotools.cocoeval.COCOeval, the
       canonical implementation. mAP@[0.5:0.95], mAP@0.5, mAP@0.75, plus
       per-class AP. Used in Phase 1 full COCO val2017 runs.
    2. ``evaluate_simple`` — dependency-light fallback that does single-IoU-
       threshold matching and returns precision / recall / F1 + mean IoU.
       Used by tests and the smoke script when pycocotools is unavailable.

Both produce a ``CocoMetrics`` dataclass with the same shape; downstream
aggregators don't have to care which backend produced them.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from edgevision.data.coco_loader import CocoDataset
from edgevision.schemas import Detection, GroundTruthBox, ImageDetections


@dataclass
class CocoMetrics:
    """Unified metrics output across both eval backends.

    Field semantics depend on ``backend``:

    * ``backend == "pycocotools"`` — ``mAP_*`` fields are real PR-curve mAP
      values; ``mAP_per_class`` is per-class AP averaged over the IoU sweep.
      ``precision`` / ``recall`` / ``f1`` / ``iou_mean`` are unset.
    * ``backend == "simple"`` — ``mAP_*`` fields are all 0.0 (the simple
      backend cannot compute PR-curve AP). The real signal lives in
      ``precision`` / ``recall`` / ``f1`` / ``iou_mean``, and
      ``mAP_per_class`` carries per-class F1 instead of AP.

    Read the ``backend`` field before interpreting any specific number.
    """

    # "mAP" (mean Average Precision) is the universal COCO metric spelling and is
    # a serialized field name in docs/results/*.json — keep the casing intentionally.
    mAP_50_95: float = 0.0  # noqa: N815
    mAP_50: float = 0.0  # noqa: N815
    mAP_75: float = 0.0  # noqa: N815
    mAP_per_class: dict[str, float] = field(default_factory=dict)  # noqa: N815
    # Set only by the simple backend.
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    iou_mean: float | None = None
    n_predictions: int = 0
    n_ground_truth: int = 0
    n_images: int = 0
    backend: str = "unknown"  # "pycocotools" | "simple"

    def as_dict(self) -> dict:
        out: dict = {
            "backend": self.backend,
            "mAP_50_95": round(self.mAP_50_95, 4),
            "mAP_50": round(self.mAP_50, 4),
            "mAP_75": round(self.mAP_75, 4),
            "n_predictions": self.n_predictions,
            "n_ground_truth": self.n_ground_truth,
            "n_images": self.n_images,
            "mAP_per_class": {k: round(v, 4) for k, v in self.mAP_per_class.items()},
        }
        if self.precision is not None:
            out["precision"] = round(self.precision, 4)
        if self.recall is not None:
            out["recall"] = round(self.recall, 4)
        if self.f1 is not None:
            out["f1"] = round(self.f1, 4)
        if self.iou_mean is not None:
            out["iou_mean"] = round(self.iou_mean, 4)
        return out


# --------------------------------------------------------------------------- IoU


def iou_xyxy(a, b) -> float:
    """IoU between two BoundingBox-like objects (have x1,y1,x2,y2,area)."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------- helpers


def _predictions_to_coco_results(
    predictions: Iterable[ImageDetections],
    label_to_class_id: dict[str, int],
) -> list[dict]:
    """Convert ImageDetections → COCO results json array.

    Each entry: {image_id, category_id, bbox=[x,y,w,h], score}.
    """
    results = []
    for img_dets in predictions:
        for det in img_dets.detections:
            class_id = det.class_id
            if class_id is None:
                class_id = label_to_class_id.get(det.label)
                if class_id is None:
                    continue  # detection class not in dataset; skip
            x, y, w, h = det.bbox.to_xywh()
            results.append(
                {
                    "image_id": img_dets.image_id,
                    "category_id": class_id,
                    "bbox": [x, y, w, h],
                    "score": float(det.confidence),
                }
            )
    return results


# --------------------------------------------------------------------------- pycocotools backend


def evaluate_pycocotools(
    predictions: Iterable[ImageDetections],
    dataset: CocoDataset,
) -> CocoMetrics:
    """Run pycocotools.COCOeval. Raises ImportError if pycocotools is missing."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pycocotools is not installed. Run `pip install pycocotools` "
            "or use evaluate_simple as a fallback."
        ) from e

    predictions = list(predictions)
    label_to_id = {c.name: c.id for c in dataset.categories}

    # Materialise the dataset to a temp COCO json.
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_json_path = Path(tmpdir) / "instances.json"
        with gt_json_path.open("w") as f:
            json.dump(dataset.to_coco_dict(), f)

        coco_gt = COCO(str(gt_json_path))

        results = _predictions_to_coco_results(predictions, label_to_id)
        if not results:
            # COCOeval errors on an empty result set — return zeros.
            return CocoMetrics(
                backend="pycocotools",
                n_predictions=0,
                n_ground_truth=len(dataset.annotations),
                n_images=len(dataset.images),
                mAP_per_class={c.name: 0.0 for c in dataset.categories},
            )

        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats  # type: ignore[attr-defined]
    # COCO stats indexing (per pycocotools docs):
    # 0: AP @ IoU=0.50:0.95 | area=all | maxDets=100
    # 1: AP @ IoU=0.50      | area=all | maxDets=100
    # 2: AP @ IoU=0.75      | area=all | maxDets=100
    map_50_95 = float(stats[0]) if len(stats) > 0 else 0.0
    map_50 = float(stats[1]) if len(stats) > 1 else 0.0
    map_75 = float(stats[2]) if len(stats) > 2 else 0.0

    # Per-class AP — coco_eval.eval["precision"] has shape
    # (T, R, K, A, M) = (10 IoU thresh, 101 recall pts, K cats, 4 area, 3 maxDets)
    per_class: dict[str, float] = {}
    try:
        precision = coco_eval.eval["precision"]  # type: ignore[attr-defined]
        cat_ids = coco_eval.params.catIds  # type: ignore[attr-defined]
        id_to_name = {c.id: c.name for c in dataset.categories}
        for k, cid in enumerate(cat_ids):
            ap_slice = precision[:, :, k, 0, 2]  # area=all, maxDets=100
            ap_slice = ap_slice[ap_slice > -1]
            ap = float(ap_slice.mean()) if ap_slice.size else 0.0
            per_class[id_to_name.get(cid, str(cid))] = ap
    except Exception:  # pragma: no cover - defensive
        per_class = {c.name: 0.0 for c in dataset.categories}

    return CocoMetrics(
        backend="pycocotools",
        mAP_50_95=map_50_95,
        mAP_50=map_50,
        mAP_75=map_75,
        mAP_per_class=per_class,
        n_predictions=len(results),
        n_ground_truth=len(dataset.annotations),
        n_images=len(dataset.images),
    )


# --------------------------------------------------------------------------- simple backend


def evaluate_simple(
    predictions: Iterable[ImageDetections],
    dataset: CocoDataset,
    *,
    iou_threshold: float = 0.5,
) -> CocoMetrics:
    """Dependency-free single-IoU evaluator.

    Greedy per-image matching, ranked by descending confidence. Each GT
    matches at most one prediction. Used in tests + smoke; for real COCO mAP
    use ``evaluate_pycocotools``.
    """
    predictions = list(predictions)
    gts_by_image = {img.image_id: dataset.annotations_for_image(img.image_id) for img in dataset.images}

    n_pred = sum(len(p.detections) for p in predictions)
    n_gt = sum(len(v) for v in gts_by_image.values())

    tp = 0
    iou_sum = 0.0
    per_class_tp: dict[str, int] = {}
    per_class_pred: dict[str, int] = {}
    per_class_gt: dict[str, int] = {}

    for gt_list in gts_by_image.values():
        for g in gt_list:
            per_class_gt[g.label] = per_class_gt.get(g.label, 0) + 1

    for img_dets in predictions:
        gts: list[GroundTruthBox] = list(gts_by_image.get(img_dets.image_id, []))
        used: set[int] = set()
        ranked: list[Detection] = sorted(img_dets.detections, key=lambda d: -d.confidence)
        for det in ranked:
            per_class_pred[det.label] = per_class_pred.get(det.label, 0) + 1
            best_iou = 0.0
            best_idx = -1
            for j, g in enumerate(gts):
                if j in used or g.label != det.label:
                    continue
                cur = iou_xyxy(det.bbox, g.bbox)
                if cur > best_iou:
                    best_iou = cur
                    best_idx = j
            if best_idx >= 0 and best_iou >= iou_threshold:
                used.add(best_idx)
                tp += 1
                iou_sum += best_iou
                per_class_tp[det.label] = per_class_tp.get(det.label, 0) + 1

    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou_mean = iou_sum / tp if tp else 0.0

    # The simple backend cannot compute a proper PR-curve AP (no recall
    # thresholds), so per-class is reported as F1@iou_threshold. The mAP_*
    # fields are deliberately left at 0.0 — `f1` carries the real signal here.
    per_class_f1: dict[str, float] = {}
    for cls in {*per_class_pred.keys(), *per_class_gt.keys()}:
        ctp = per_class_tp.get(cls, 0)
        cp = per_class_pred.get(cls, 0)
        cg = per_class_gt.get(cls, 0)
        cprec = ctp / cp if cp else 0.0
        crec = ctp / cg if cg else 0.0
        per_class_f1[cls] = (
            2 * cprec * crec / (cprec + crec) if (cprec + crec) else 0.0
        )

    return CocoMetrics(
        backend="simple",
        mAP_50_95=0.0,  # simple fallback can't compute multi-IoU AP
        mAP_50=0.0,  # ditto — see `f1` below for the real signal
        mAP_75=0.0,
        mAP_per_class=per_class_f1,  # F1 per class, not mAP — see backend field
        precision=precision,
        recall=recall,
        f1=f1,
        iou_mean=iou_mean,
        n_predictions=n_pred,
        n_ground_truth=n_gt,
        n_images=len(dataset.images),
    )


# --------------------------------------------------------------------------- public entry


def evaluate(
    predictions: Iterable[ImageDetections],
    dataset: CocoDataset,
    *,
    backend: str = "auto",
    iou_threshold: float = 0.5,
) -> CocoMetrics:
    """Top-level dispatcher.

    Args:
        backend: "auto" | "pycocotools" | "simple"
        iou_threshold: only used by the simple backend.
    """
    if backend == "pycocotools":
        return evaluate_pycocotools(predictions, dataset)
    if backend == "simple":
        return evaluate_simple(predictions, dataset, iou_threshold=iou_threshold)
    if backend == "auto":
        try:
            return evaluate_pycocotools(predictions, dataset)
        except ImportError:
            return evaluate_simple(predictions, dataset, iou_threshold=iou_threshold)
    raise ValueError(f"Unknown backend: {backend!r}")


def summary_table(metrics: CocoMetrics) -> str:
    rows: list[tuple[str, str]] = [
        ("Backend", metrics.backend),
        ("Images", str(metrics.n_images)),
        ("Ground-truth boxes", str(metrics.n_ground_truth)),
        ("Predictions", str(metrics.n_predictions)),
    ]
    if metrics.backend == "pycocotools":
        rows += [
            ("mAP @ [0.5:0.95]", f"{metrics.mAP_50_95:.3f}"),
            ("mAP @ 0.5", f"{metrics.mAP_50:.3f}"),
            ("mAP @ 0.75", f"{metrics.mAP_75:.3f}"),
        ]
    else:
        rows += [
            ("Precision", f"{metrics.precision:.3f}" if metrics.precision is not None else "n/a"),
            ("Recall", f"{metrics.recall:.3f}" if metrics.recall is not None else "n/a"),
            ("F1", f"{metrics.f1:.3f}" if metrics.f1 is not None else "n/a"),
            ("IoU mean (TP)", f"{metrics.iou_mean:.3f}" if metrics.iou_mean is not None else "n/a"),
        ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k.ljust(width)} : {v}" for k, v in rows)
