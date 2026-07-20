"""Phase 1 baseline smoke — end-to-end CPU-only, no downloads.

What this does:
    1. Build a deterministic synthetic COCO-format dataset (4 images, 2 classes).
    2. Run the MockRTDetrDetector — replays GT as detections with a controllable
       drop-rate so the evaluator has something to score.
    3. Score with pycocotools.COCOeval (auto-falls-back to the simple matcher
       when pycocotools is missing).
    4. Print a summary table.

Run a real RT-DETR-R50 baseline over COCO val2017 with::

    python scripts/run_baseline_smoke.py --backend rtdetr \\
        --coco-annotations data/coco/annotations/instances_val2017.json \\
        --coco-images data/coco/val2017
"""

from __future__ import annotations

import argparse
import sys
import time

from edgevision.data.coco_loader import CocoDataset, gt_for_images
from edgevision.evaluation.coco_eval import evaluate, summary_table
from edgevision.models.rtdetr_wrapper import make_detector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="edge-vision Phase-1 baseline smoke")
    p.add_argument("--backend", choices=["mock", "rtdetr"], default="mock")
    p.add_argument(
        "--coco-annotations",
        type=str,
        default=None,
        help="Path to COCO instances_<split>.json (only with --backend rtdetr).",
    )
    p.add_argument(
        "--coco-images",
        type=str,
        default=None,
        help="Path to COCO image directory (only with --backend rtdetr).",
    )
    p.add_argument("--max-images", type=int, default=16)
    p.add_argument(
        "--eval-backend",
        choices=["auto", "pycocotools", "simple"],
        default="auto",
    )
    p.add_argument("--mock-recall", type=float, default=0.85)
    p.add_argument("--mock-fp-rate", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def build_dataset(args: argparse.Namespace) -> CocoDataset:
    if args.backend == "mock":
        return CocoDataset.synthetic(
            n_images=max(4, min(args.max_images, 8)),
            n_classes=3,
            boxes_per_image=2,
            seed=args.seed,
        )
    if not args.coco_annotations or not args.coco_images:
        raise SystemExit(
            "--backend rtdetr requires both --coco-annotations and --coco-images"
        )
    return CocoDataset.from_json(
        annotations_json=args.coco_annotations,
        images_dir=args.coco_images,
        split="val2017",
        max_images=args.max_images,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"[edge-vision] backend={args.backend} max_images={args.max_images}")

    dataset = build_dataset(args)
    print(
        f"[edge-vision] dataset: {len(dataset.images)} images, "
        f"{len(dataset.annotations)} GT boxes, "
        f"{len(dataset.categories)} classes"
    )

    if args.backend == "mock":
        from edgevision.models.rtdetr_wrapper import MockRTDetrDetector

        gt_map = gt_for_images(dataset, dataset.images)
        detector = MockRTDetrDetector(
            gt_by_image_id=gt_map,
            recall=args.mock_recall,
            false_positive_rate=args.mock_fp_rate,
            seed=args.seed,
        )
    else:
        detector = make_detector("rtdetr")

    t0 = time.perf_counter()
    predictions = []
    for img in dataset.images:
        t_img = time.perf_counter()
        result = detector.predict(img)
        result.inference_ms = (time.perf_counter() - t_img) * 1e3
        predictions.append(result)
    elapsed = time.perf_counter() - t0
    fps = len(dataset.images) / elapsed if elapsed > 0 else float("inf")

    n_det = sum(len(p) for p in predictions)
    print(f"[edge-vision] {n_det} detections in {elapsed:.3f}s ({fps:.1f} FPS)")

    metrics = evaluate(predictions, dataset, backend=args.eval_backend)
    print()
    print(summary_table(metrics))

    print()
    print("[edge-vision] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
