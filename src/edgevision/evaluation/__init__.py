"""Evaluation — COCO mAP, quantization-eval, Pareto aggregation.

Phase 1: coco_eval (pycocotools mAP@[0.5:0.95]).
Phase 3: quant_eval (mAP delta + per-class drop).
Phase 5: pareto_aggregator (mAP x p95 x watts frontier).
"""
