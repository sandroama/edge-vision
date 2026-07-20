---
title: edge-vision measured-results dashboard
emoji: 🎯
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Measured CPU INT8/pruning results for RT-DETR edge compression
---

# edge-vision — HF Space (measured-results dashboard)

Streamlit dashboard for the [edge-vision](https://github.com/sandroama/edge-vision) project. It renders the project's **measured CPU lane** — ONNX Runtime dynamic-INT8 model-size reduction and CPU latency with bootstrap 95% CIs — read verbatim from `docs/results/*.json`. Nothing on the page is hardcoded or estimated, and every GPU-dependent number (TensorRT latency, real COCO mAP, NVML watts/frame) is explicitly labeled **pending** until a real RTX-class run happens (`NEXT_STEPS.md`).

This is intentionally *not* a live-inference demo: it shows real measurements instead of running an unvalidated model in the browser.

## Deploying to a Space

The Space repo root should contain:

```
app.py                      # this file's sibling
requirements.txt            # streamlit + plotly + pandas, nothing heavier
results/                    # copied from the project's docs/results/
    phase3_cpu_int8.json
    phase5_pareto.json
```

`app.py` looks for `../docs/results` first (repo layout) and falls back to `./results` (Space layout). Missing files degrade to a friendly empty state, never a traceback.

## Run locally

```bash
pip install -e '.[ui]'      # from the project root
streamlit run hf_space/app.py
```
