"""``edgevision-smoke`` console entry point — CPU-only mock baseline smoke.

This is the packaged, import-from-anywhere version of
``scripts/run_baseline_smoke.py``. It builds a deterministic synthetic
COCO-format dataset, replays ground truth through :class:`MockRTDetrDetector`
with a controllable recall / false-positive rate, scores the result with the
COCO evaluator (pycocotools when present, simple-IoU fallback otherwise), and
prints a summary table. No model download, no GPU.

Run the real RT-DETR-R50 baseline over COCO val2017 with the richer flags on
``scripts/run_baseline_smoke.py`` (``--backend rtdetr --coco-annotations ...``).
"""

from __future__ import annotations

import argparse
import sys
import time

from edgevision.data.coco_loader import CocoDataset, gt_for_images
from edgevision.evaluation.coco_eval import evaluate, summary_table
from edgevision.models.rtdetr_wrapper import MockRTDetrDetector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="edgevision-smoke",
        description="edge-vision CPU-only mock baseline smoke (no GPU, no downloads).",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Synthetic images to generate (clamped to [4, 8]).",
    )
    p.add_argument(
        "--eval-backend",
        choices=["auto", "pycocotools", "simple"],
        default="auto",
        help="mAP backend; 'auto' uses pycocotools when importable.",
    )
    p.add_argument(
        "--mock-recall",
        type=float,
        default=0.85,
        help="Fraction of ground-truth boxes the mock detector recovers.",
    )
    p.add_argument(
        "--mock-fp-rate",
        type=float,
        default=0.10,
        help="Mock detector false-positive rate.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("[edge-vision] edgevision-smoke (mock baseline, CPU-only)")

    dataset = CocoDataset.synthetic(
        n_images=max(4, min(args.max_images, 8)),
        n_classes=3,
        boxes_per_image=2,
        seed=args.seed,
    )
    print(
        f"[edge-vision] dataset: {len(dataset.images)} images, "
        f"{len(dataset.annotations)} GT boxes, "
        f"{len(dataset.categories)} classes"
    )

    gt_map = gt_for_images(dataset, dataset.images)
    detector = MockRTDetrDetector(
        gt_by_image_id=gt_map,
        recall=args.mock_recall,
        false_positive_rate=args.mock_fp_rate,
        seed=args.seed,
    )

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
