"""Phase 2 latency sweep — fill in the per-backend p50/p95/p99 + FPS table.

Backends measured (when their deps are installed):

    * ``torch-cpu``      — PyTorch eager forward on CPU
    * ``torch-cuda``     — PyTorch eager forward on the first CUDA device
    * ``onnxrt-cpu``     — ONNX Runtime CPU
    * ``trt-fp16``       — TensorRT FP16 engine
    * ``trt-fp32``       — TensorRT FP32 engine

Each row is one ``LatencyResult``, written to JSON for aggregation by
Phase 5's Pareto plotter. The sweep uses the in-repo tiny detector by
default so the script is fast in CI; pass ``--model rtdetr`` to use the
real RT-DETR-R50 (requires the ``[gpu]`` extras + GPU).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from edgevision.inference import LatencyResult


def _make_input_numpy(batch: int = 1, h: int = 640, w: int = 640) -> np.ndarray:
    return np.zeros((batch, 3, h, w), dtype=np.float32)


def _bench_onnxrt_cpu(onnx_path: Path, *, n_runs: int, n_warmup: int) -> LatencyResult:
    from edgevision.compile import OnnxRuntimeCPUExecutor
    from edgevision.inference import measure_latency_cpu

    executor = OnnxRuntimeCPUExecutor(onnx_path)
    x = _make_input_numpy()
    return measure_latency_cpu(
        executor.make_callable(x),
        n_runs=n_runs,
        n_warmup=n_warmup,
        device="cpu (ORT)",
    )


def _bench_torch(
    model, *, device: str, n_runs: int, n_warmup: int
) -> LatencyResult:
    import torch

    from edgevision.inference import (
        measure_latency_cpu,
        measure_latency_cuda,
    )

    model = model.to(device).eval()
    dummy = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=device)
    fn = lambda: model(dummy)  # noqa: E731

    if device == "cuda":
        return measure_latency_cuda(
            fn, n_runs=n_runs, n_warmup=n_warmup, device="cuda:0"
        )
    return measure_latency_cpu(fn, n_runs=n_runs, n_warmup=n_warmup, device=device)


def _bench_trt(engine_path: Path, *, n_runs: int, n_warmup: int) -> LatencyResult:
    """Bench a serialised TRT engine. Uses pycuda for buffer management."""
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "TRT bench requires `pycuda` + `tensorrt`. Install with `pip install -e '.[trt]'`."
        ) from e

    from edgevision.inference import measure_latency_cuda

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    with engine_path.open("rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Allocate input + output buffers.
    bindings: list[int] = []
    h_inputs: list[np.ndarray] = []
    d_inputs: list = []
    h_outputs: list[np.ndarray] = []
    d_outputs: list = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        if -1 in shape or 0 in shape:  # dynamic axis -> bind batch=1
            shape = (1, *shape[1:])
            context.set_input_shape(name, shape)
        size = int(np.prod(shape))
        dtype = np.float32
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            host = np.zeros(size, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            h_inputs.append(host)
            d_inputs.append(dev)
            cuda.memcpy_htod(dev, host)
            context.set_tensor_address(name, int(dev))
            bindings.append(int(dev))
        else:
            host = np.zeros(size, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            h_outputs.append(host)
            d_outputs.append(dev)
            context.set_tensor_address(name, int(dev))
            bindings.append(int(dev))

    stream = cuda.Stream()

    def _fn() -> None:
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    return measure_latency_cuda(
        _fn, n_runs=n_runs, n_warmup=n_warmup, device=f"cuda (TRT {engine_path.stem})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="edge-vision Phase-2 latency sweep")
    p.add_argument(
        "--backends",
        nargs="+",
        default=["torch-cpu", "onnxrt-cpu"],
        choices=[
            "torch-cpu",
            "torch-cuda",
            "onnxrt-cpu",
            "trt-fp16",
            "trt-fp32",
        ],
    )
    p.add_argument("--model", choices=["tiny", "rtdetr"], default="tiny")
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--out-json", type=str, default="docs/results/phase2_latency.json")
    p.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    return p.parse_args(argv)


def _ensure_artifacts(checkpoints_dir: Path, model: str) -> Path:
    """Make sure the ONNX (and optionally TRT) artifacts exist."""
    if model != "tiny":
        # Real RT-DETR artifacts are produced by Phase 1's smoke + Phase 2
        # full-model export, not by this script. Just expect them to exist.
        return checkpoints_dir / "rtdetr_r50.onnx"

    onnx_path = checkpoints_dir / "tiny_detector.onnx"
    if not onnx_path.exists():
        from edgevision.compile import export_to_onnx
        from edgevision.models.tiny_model import make_tiny_input, make_tiny_model

        export_to_onnx(make_tiny_model(), make_tiny_input(), onnx_path)
    return onnx_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    try:
        onnx_path = _ensure_artifacts(checkpoints_dir, args.model)
    except ImportError as e:
        print(f"[edge-vision] Missing dep for ensure-artifacts: {e}")
        return 1

    for backend in args.backends:
        t0 = time.perf_counter()
        print(f"[edge-vision] Benching backend={backend} ...")
        try:
            if backend == "torch-cpu" or backend == "torch-cuda":
                from edgevision.models.tiny_model import make_tiny_model

                model = make_tiny_model() if args.model == "tiny" else None
                if model is None:
                    print(f"  -> {backend} with --model rtdetr is Phase-2 GPU work; skipping.")
                    continue
                device = "cuda" if backend == "torch-cuda" else "cpu"
                result = _bench_torch(
                    model, device=device, n_runs=args.n_runs, n_warmup=args.n_warmup
                )
            elif backend == "onnxrt-cpu":
                result = _bench_onnxrt_cpu(
                    onnx_path, n_runs=args.n_runs, n_warmup=args.n_warmup
                )
            elif backend.startswith("trt-"):
                precision = backend.split("-", 1)[1]
                engine_path = checkpoints_dir / f"tiny_detector_{precision}.engine"
                if not engine_path.exists():
                    print(f"  -> engine missing: {engine_path}; run scripts/run_compile_smoke.py --stage trt first")
                    continue
                result = _bench_trt(
                    engine_path, n_runs=args.n_runs, n_warmup=args.n_warmup
                )
            else:
                print(f"  -> unknown backend: {backend}")
                continue
        except (ImportError, RuntimeError) as e:
            print(f"  -> {backend} failed: {e}")
            continue

        elapsed = time.perf_counter() - t0
        print("  -> " + result.as_row())
        rows.append(
            {
                "backend": backend,
                "model": args.model,
                "wall_seconds": round(elapsed, 2),
                **{k: v for k, v in result.as_dict().items()},
            }
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print()
    print(f"[edge-vision] wrote {len(rows)} rows -> {out_path}")
    print("[edge-vision] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
