# Phase 1 — baseline reproduction (results)

> **Status:** Phase 1 modules wired (mock backend tests green). Real RT-DETR-R50
> COCO val2017 reproduction is the GPU run, blocked on a ~120 MB HF weights
> pull + actual inference time. The harness is in place — only the numbers
> need to be filled.

## CPU smoke (mock backend)

The baseline smoke (`scripts/run_baseline_smoke.py --backend mock`) runs
end-to-end without any heavy ML deps. It builds a 4-image synthetic
COCO-format dataset, replays the GT through `MockRTDetrDetector`, and scores
via `evaluate_simple`.

| Config | n_images | n_gt | n_pred | precision | recall | F1 | mean IoU (TP) |
|---|---|---|---|---|---|---|---|
| mock recall=1.0, fp=0.0 | 4 | 8 | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| mock recall=0.6, fp=0.3 | 4 | 8 | 10 | 0.600 | 0.750 | 0.667 | 1.000 |

This is a **harness sanity check**, not a model accuracy claim. The mock
replays GT, so perfect-recall numbers are by construction. The noisy-recall
row exists to prove the eval drops precision when FPs appear and drops recall
when GTs are missed.

## Real RT-DETR-R50 on COCO val2017 — TBD

Hardware: RTX 5080 (Blackwell sm_100) + Ryzen 9 9950X.

To run::

    # 1. Get COCO val2017 + annotations.
    mkdir -p data/coco
    cd data/coco
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip val2017.zip
    unzip annotations_trainval2017.zip
    cd -

    # 2. Run the real baseline (auto eval-backend will pick pycocotools).
    python scripts/run_baseline_smoke.py \
        --backend rtdetr \
        --coco-annotations data/coco/annotations/instances_val2017.json \
        --coco-images data/coco/val2017 \
        --max-images 5000 \
        --eval-backend pycocotools

Expected published baseline (from the RT-DETR paper, R50 variant):

| Metric | Reported | edge-vision (TBD) |
|---|---|---|
| mAP@[0.5:0.95] | ~53.1 | (run pending) |
| mAP@0.5 | ~71.3 | (run pending) |
| FPS (PyTorch eager, single GPU) | ~108 | (run pending) |

## Module status

| Module | File | Tests | Notes |
|---|---|---|---|
| Schema | `src/edgevision/schemas.py` | 6/6 ✅ | BoundingBox / Detection / GT / Image / ImageDetections |
| COCO loader | `src/edgevision/data/coco_loader.py` | 6/6 ✅ | Real-json load + synthetic generator + COCO-roundtrip |
| Preprocessor | `src/edgevision/data/preprocessor.py` | 7/7 ✅ | Numpy-only letterbox + ImageNet normalize + un-letterbox |
| RT-DETR wrapper | `src/edgevision/models/rtdetr_wrapper.py` | 7/7 ✅ | Mock + Real (HF transformers, lazy-imported) |
| COCO eval | `src/edgevision/evaluation/coco_eval.py` | 10/10 ✅ | pycocotools + simple fallback, same `CocoMetrics` shape |
| Baseline smoke | `scripts/run_baseline_smoke.py` | 2/2 ✅ | CLI: `--backend mock|rtdetr`, `--eval-backend auto|pycocotools|simple` |

**Total: 41 tests, 41 passing.** All green on CPU with no torch / pycocotools
required (CI default).
