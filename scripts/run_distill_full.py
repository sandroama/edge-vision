"""Phase 4 full distillation — RT-DETR-R50 → RT-DETR-R18 on COCO train2017.

This is the production-scale run that generates the RQ-E2 checkpoint.
Designed to run on RTX 5080 for ~50 epochs (~8–12 hours).

Usage::

    pip install -e ".[dev,gpu]"

    python scripts/run_distill_full.py \\
        --coco-annotations data/coco/annotations/instances_train2017.json \\
        --coco-images data/coco/train2017 \\
        --epochs 50 \\
        --out-dir checkpoints/distill_full

Status: scaffold. The DataLoader wiring uses the Phase-1 CocoDataset
loader — fill in the ``_make_dataloader`` implementation when you start
the GPU run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="edge-vision Phase-4 full distillation")
    p.add_argument("--coco-annotations", type=str, required=True)
    p.add_argument("--coco-images", type=str, required=True)
    p.add_argument(
        "--teacher",
        type=str,
        default="PekingU/rtdetr_r50vd_coco_o365",
    )
    p.add_argument(
        "--student",
        type=str,
        default="PekingU/rtdetr_r18vd_coco_o365",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out-dir", type=str, default="checkpoints/distill_full")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _make_dataloader(coco_annotations: str, coco_images: str, batch_size: int):
    """Return a DataLoader that yields pixel_values tensors.

    TODO (Phase-4 GPU run): wrap CocoDataset + RTDetrImageProcessor in a
    proper torch DataLoader with multi-worker loading.
    """
    raise NotImplementedError(
        "DataLoader not yet wired for the GPU run. "
        "See BUILD_PLAN.md Phase 4 for the implementation checklist."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from edgevision.distillation import DistillationConfig, run_rtdetr_distillation
    except ImportError as e:
        print(f"[edge-vision] Missing dep: {e}")
        return 1

    cfg = DistillationConfig(
        teacher_model=args.teacher,
        student_model=args.student,
        output_dir=Path(args.out_dir),
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        alpha=args.alpha,
        seed=args.seed,
    )

    print(f"[edge-vision] Starting full distillation: {cfg.teacher_model} → {cfg.student_model}")
    print(f"              epochs={args.epochs}  lr={args.lr}  T={args.temperature}  α={args.alpha}")

    train_loader = _make_dataloader(args.coco_annotations, args.coco_images, args.batch_size)
    result = run_rtdetr_distillation(cfg, train_loader=train_loader)

    print(f"[edge-vision] Converged: {result.converged}")
    print(f"[edge-vision] Checkpoint: {result.checkpoint_path}")

    out = Path(args.out_dir) / "distill_summary.json"
    out.write_text(json.dumps(result.as_dict(), indent=2))
    print(f"[edge-vision] wrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
