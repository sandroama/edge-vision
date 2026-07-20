"""Phase 3 quantization smoke — produces an RQ-E1 row.

This script answers RQ-E1 ("How does mAP@COCO degrade FP32 -> FP16 -> INT8?")
end-to-end on whatever data is available:

    * ``--coco-annotations`` + ``--coco-images`` -> real COCO val2017
    * Otherwise -> synthetic dataset (CI-friendly; numbers are illustrative)

Engines exercised (best-effort, skip-on-missing-deps):
    * ``mock`` -> MockRTDetrDetector replays GT (reference baseline).
    * ``mock-noisy`` -> mock with recall=0.85 + fp_rate=0.15 to simulate
      a noisy-quantized model. Used as the "candidate" in CI smoke runs
      since real INT8 needs a GPU.
    * ``int8-qdq`` (Phase-3-real) -> ONNX QDQ static quantization, then
      ORT-CPU inference. Requires ``onnxruntime.quantization`` and a
      previously-exported FP32 ONNX. Currently a placeholder noting how
      to wire the real pipeline.

Output:
    * ``docs/results/phase3_quantization.json`` — machine-readable.
    * Stdout summary via ``quant_eval.summary_table``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edgevision.data.coco_loader import CocoDataset, gt_for_images
from edgevision.evaluation.coco_eval import evaluate
from edgevision.evaluation.coco_eval import summary_table as eval_table
from edgevision.evaluation.quant_eval import compare_metrics, summary_table
from edgevision.models.rtdetr_wrapper import MockRTDetrDetector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="edge-vision Phase-3 quantization smoke")
    p.add_argument(
        "--coco-annotations",
        type=str,
        default=None,
        help="Path to instances_<split>.json. If omitted, uses synthetic dataset.",
    )
    p.add_argument(
        "--coco-images",
        type=str,
        default=None,
        help="Path to COCO image directory (only with --coco-annotations).",
    )
    p.add_argument("--max-images", type=int, default=8)
    p.add_argument(
        "--eval-backend",
        choices=["auto", "pycocotools", "simple"],
        default="auto",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default="docs/results/phase3_quantization.json",
    )
    p.add_argument(
        "--candidate-recall",
        type=float,
        default=0.85,
        help="Mock candidate's recall (degrades from 1.0 to simulate INT8 drops).",
    )
    p.add_argument(
        "--candidate-fp-rate",
        type=float,
        default=0.15,
        help="Mock candidate's false-positive rate.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def build_dataset(args: argparse.Namespace) -> CocoDataset:
    if args.coco_annotations and args.coco_images:
        return CocoDataset.from_json(
            annotations_json=args.coco_annotations,
            images_dir=args.coco_images,
            split="val2017",
            max_images=args.max_images,
        )
    return CocoDataset.synthetic(
        n_images=max(4, min(args.max_images, 8)),
        n_classes=3,
        boxes_per_image=2,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    dataset = build_dataset(args)
    print(
        f"[edge-vision] dataset: {len(dataset.images)} images, "
        f"{len(dataset.annotations)} GT boxes, "
        f"{len(dataset.categories)} classes"
    )

    gt_map = gt_for_images(dataset, dataset.images)

    # Reference: a "perfect" mock that replays GT exactly. Stands in for the
    # FP32 baseline here. In real Phase-3 GPU runs, the reference is the
    # FP16 RT-DETR engine.
    ref_detector = MockRTDetrDetector(
        gt_by_image_id=gt_map, recall=1.0, false_positive_rate=0.0, seed=args.seed
    )
    ref_preds = [ref_detector.predict(img) for img in dataset.images]
    ref_metrics = evaluate(ref_preds, dataset, backend=args.eval_backend)

    # Candidate: noisy mock standing in for an INT8-quantized model.
    cand_detector = MockRTDetrDetector(
        gt_by_image_id=gt_map,
        recall=args.candidate_recall,
        false_positive_rate=args.candidate_fp_rate,
        seed=args.seed,
    )
    cand_preds = [cand_detector.predict(img) for img in dataset.images]
    cand_metrics = evaluate(cand_preds, dataset, backend=args.eval_backend)

    print()
    print("[edge-vision] Reference (fp32 / mock-perfect):")
    print(eval_table(ref_metrics))
    print()
    print("[edge-vision] Candidate (int8 / mock-noisy):")
    print(eval_table(cand_metrics))
    print()

    delta = compare_metrics(
        ref_metrics,
        cand_metrics,
        reference_label="fp32-mock",
        candidate_label=f"int8-mock-r{args.candidate_recall:.2f}-fp{args.candidate_fp_rate:.2f}",
    )
    print("[edge-vision] RQ-E1 delta:")
    print(summary_table(delta, top_n=3))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "reference": ref_metrics.as_dict(),
                "candidate": cand_metrics.as_dict(),
                "delta": delta.as_dict(),
            },
            f,
            indent=2,
        )
    print()
    print(f"[edge-vision] wrote -> {out_path}")
    print("[edge-vision] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
