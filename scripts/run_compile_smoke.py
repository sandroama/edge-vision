"""Phase 2 compile smoke — torch -> ONNX -> (optionally) TensorRT.

Stages:
    --stage onnx       export the tiny detector to an ONNX file and verify it
    --stage onnxrt     run a few forward passes through ONNX Runtime CPU
    --stage trt        build a TensorRT engine (requires GPU + tensorrt extras)
    --stage all        run every stage that the local env supports

This script intentionally uses the in-repo ``tiny_model`` rather than RT-DETR
so it runs in a few seconds and does not require any HF model downloads. The
goal is to verify the *pipeline* works end-to-end, not to produce a useful
detector. The Phase-3 quant smoke and Phase-1 baseline use the real RT-DETR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _stage_onnx(out_dir: Path, opset: int, *, dynamic_batch: bool) -> Path:
    from edgevision.compile import export_to_onnx, verify_onnx
    from edgevision.models.tiny_model import make_tiny_input, make_tiny_model

    print(f"[edge-vision] Exporting tiny detector to ONNX (opset={opset})")
    model = make_tiny_model()
    dummy = make_tiny_input()
    out_path = out_dir / "tiny_detector.onnx"
    export_to_onnx(
        model,
        dummy,
        out_path,
        opset=opset,
        dynamic_batch=dynamic_batch,
    )

    info = verify_onnx(out_path)
    size_mb = out_path.stat().st_size / (1 << 20)
    print(f"  -> {out_path} ({size_mb:.2f} MB)")
    print(f"     opset={info.opset}, ir={info.ir_version}, producer={info.producer}")
    print(f"     inputs={info.input_names}, outputs={info.output_names}")
    return out_path


def _stage_onnxrt(onnx_path: Path, n_iter: int) -> None:
    import numpy as np

    from edgevision.compile import OnnxRuntimeCPUExecutor
    from edgevision.inference import measure_latency_cpu

    print(f"[edge-vision] Running ONNX Runtime CPU on {onnx_path}")
    executor = OnnxRuntimeCPUExecutor(onnx_path)
    description = executor.describe()
    print(f"  -> {description}")

    # Use a fixed-shape input even though the export uses dynamic batch; ORT
    # binds the actual shape at runtime.
    fixed_shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in description["input_shape"])
    if len(fixed_shape) == 4:
        # Fall back to a (1,3,640,640) sentinel shape if any dim was None.
        fixed_shape = (1, 3, 640, 640)
    x = np.zeros(fixed_shape, dtype=np.float32)

    result = measure_latency_cpu(
        executor.make_callable(x),
        n_runs=n_iter,
        n_warmup=max(2, n_iter // 10),
        device="cpu (ORT)",
    )
    print("  -> " + result.as_row())


def _stage_trt(onnx_path: Path, out_dir: Path, precision: str) -> None:
    from edgevision.compile import TrtBuildConfig, build_engine, trt_available

    if not trt_available():
        print(
            "[edge-vision] TensorRT not installed — skipping TRT build. "
            "Install with `pip install -e '.[trt]'` to enable."
        )
        return

    engine_path = out_dir / f"tiny_detector_{precision}.engine"
    print(f"[edge-vision] Building TensorRT engine ({precision}) -> {engine_path}")
    result = build_engine(
        onnx_path,
        engine_path,
        config=TrtBuildConfig(precision=precision, workspace_mb=1024),
    )
    print("  -> " + str(result.as_dict()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="edge-vision Phase-2 compile smoke")
    p.add_argument(
        "--stage",
        choices=["onnx", "onnxrt", "trt", "all"],
        default="all",
        help="Which compile stage(s) to run.",
    )
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--no-dynamic-batch", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path: Path | None = None

    try:
        if args.stage in ("onnx", "all"):
            onnx_path = _stage_onnx(
                out_dir, opset=args.opset, dynamic_batch=not args.no_dynamic_batch
            )

        if args.stage in ("onnxrt", "all"):
            if onnx_path is None:
                onnx_path = out_dir / "tiny_detector.onnx"
            _stage_onnxrt(onnx_path, n_iter=args.n_iter)

        if args.stage in ("trt", "all"):
            if onnx_path is None:
                onnx_path = out_dir / "tiny_detector.onnx"
            _stage_trt(onnx_path, out_dir, precision=args.precision)
    except ImportError as e:
        print(f"[edge-vision] Missing dep: {e}")
        print("  -> Run `pip install -e '.[dev]'` (and `[gpu,trt]` for GPU stages).")
        return 1

    print()
    print("[edge-vision] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
