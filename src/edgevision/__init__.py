"""edgevision — Real-Time Detection + Segmentation on Real Hardware (Project Alpha-1).

Submodules:
    data         — COCO loaders + RT-DETR preprocessing
    models       — RT-DETR + MobileSAM wrappers
    distillation — Teacher/student KD training
    pruning      — Structured channel pruning
    quantization — TRT INT8 + ONNX QDQ
    compile      — ONNX export + TRT build + ORT exec
    inference    — Batch=1 latency harness (p50/p95/p99)
    profiling    — NVML power + thermal + CPU profile
    evaluation   — mAP + quant-eval + Pareto aggregation
    dashboard    — Streamlit Pareto + live-camera demo
    api          — FastAPI inference service
"""

__version__ = "0.1.0"
