# Deployment — edge-vision

Three deployment targets, scaffolded around an **ONNX-first** workflow so any of them is a config change rather than a rewrite.

| Target | Status | When to use |
|---|---|---|
| RTX 5080 + TensorRT (FP16/INT8) | ✅ primary | Real-time GPU inference, Pareto-frontier benchmarks |
| CPU via ONNX Runtime | ✅ primary | HF Space demo, "no GPU" fallback path |
| Jetson Orin Nano | 🟡 stub | Phase 7 — populate when hardware is acquired |
| Apple Neural Engine via CoreML | 🟡 stub | Phase 7 — optional macOS demo path |

---

## 1) RTX 5080 (Blackwell sm_100) — primary target

### Prerequisites
- NVIDIA driver ≥ 555 (Blackwell support)
- CUDA Toolkit 12.4+
- TensorRT 10.x (matches CUDA 12.4)
- Python 3.11 or 3.12 with the project installed via `pip install -e ".[dev,gpu]"`

### One-time setup
```bash
# Verify driver + CUDA
nvidia-smi
nvcc --version

# Install TensorRT (one of):
#   a) System install via Debian/Ubuntu repo (recommended, matches CUDA toolkit)
#   b) pip install tensorrt  (works for the Python API, but you still need the runtime libs)
#   c) NGC container: nvcr.io/nvidia/tensorrt:24.10-py3 (if local install fights you)
python -c "import tensorrt; print(tensorrt.__version__)"
```

### Build + bench an engine
```bash
# Phase 1 baseline checkpoint
make smoke

# Phase 2 export + build
make export-onnx     # -> checkpoints/rtdetr_r50.onnx
make build-trt       # -> checkpoints/rtdetr_r50_fp16.engine
make bench           # latency p50/p95/p99 + FPS

# Phase 3 INT8 calibration
python scripts/run_quant_smoke.py --engines fp16 int8

# Phase 5 sustained 15-min power sweep
make power-sweep     # samples NVML every 100ms, logs throttle events
```

### Things that go wrong
- **`tensorrt.Builder` raises "no kernels available":** sm_100 not yet supported by the installed TRT version. Update to TRT 10.5+ or pull from NGC.
- **Engine builds but mAP collapses:** check `--workspace` flag (default ~1GB is too small for RT-DETR; bump to 4GB).
- **NVML reports 0 W:** the user running the script needs `gpu` group membership; check `nvidia-smi -q -d POWER`.

---

## 2) CPU via ONNX Runtime — fallback / HF Space target

### Prerequisites
- Any x86_64 / arm64 machine with Python 3.11+
- `pip install -e ".[dev]"` — no GPU extras needed

### Bench
```bash
python scripts/run_compile_smoke.py --backend onnxrt-cpu
python scripts/run_latency_sweep.py --backend onnxrt-cpu --num-images 100
```

### HF Space deploy
```bash
cd hf_space/
huggingface-cli upload <user>/edge-vision .
```

The HF Space `app.py` loads the ONNX checkpoint from a release artifact and serves a Streamlit demo. CPU-only execution; expect ~5 FPS on a Spaces CPU runtime — that's fine, the GPU FPS table is in the README anyway.

---

## 3) Jetson Orin Nano — stub (Phase 7)

`scripts/jetson_bench.py` is an empty stub. To populate:

1. Flash JetPack 6.x onto the Jetson Orin Nano.
2. Install TensorRT (ships with JetPack).
3. Copy `checkpoints/rtdetr_r50_fp16.engine` over and re-`build-trt` on-device (engines are not portable across GPUs — must rebuild on Jetson).
4. Run `python scripts/jetson_bench.py --engine fp16` — same harness as the desktop, prints latency + `tegrastats`-derived power.

Estimated effort once the board exists: 4–6 hours.

---

## 4) Apple CoreML — stub (Phase 7)

`src/edgevision/compile/coreml_export.py` will (when populated) use `coremltools.convert(model, source="pytorch")` to produce a `.mlpackage`. Useful as the "Apple AIML interview" demo. Requires macOS for the conversion step.

---

## Reproducibility checklist (copy into every release)

- [ ] PyTorch checkpoint hash logged
- [ ] ONNX model hash logged
- [ ] TRT engine builder version logged
- [ ] CUDA + driver version logged
- [ ] Calibration set (which 100–500 COCO images) logged with image IDs
- [ ] NVML sample interval + window logged
- [ ] CPU governor + thermal state logged for ONNX-CPU runs
