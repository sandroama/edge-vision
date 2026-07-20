"""TensorRT engine builder — the GPU compile target.

Reads an ONNX file, produces a serialised TRT engine. FP32 and FP16 paths
land here in Phase 2; INT8 (with the entropy calibrator) lands in Phase 3.

Engines are **not portable** across GPU architectures or TensorRT versions,
so we always rebuild on the deployment host and never check the resulting
``.engine`` file into git (see ``.gitignore``).

Lazy-imports ``tensorrt`` — it lives in the ``[trt]`` extra and won't be
installed on CPU-only machines. Importing this module without TRT is OK;
calling ``build_engine`` without TRT raises a descriptive ImportError.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrtBuildConfig:
    """Knobs for one engine build."""

    precision: str = "fp16"             # "fp32" | "fp16" | "int8"
    workspace_mb: int = 4096
    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 1
    int8_calibrator: Any = None         # IInt8EntropyCalibrator2 instance (Phase 3)
    strict_types: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class TrtBuildResult:
    """Bookkeeping for a successful engine build."""

    engine_path: str
    onnx_path: str
    precision: str
    build_seconds: float
    file_size_mb: float
    trt_version: str

    def as_dict(self) -> dict:
        return {
            "engine_path": self.engine_path,
            "onnx_path": self.onnx_path,
            "precision": self.precision,
            "build_seconds": round(self.build_seconds, 2),
            "file_size_mb": round(self.file_size_mb, 2),
            "trt_version": self.trt_version,
        }


def _import_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "TensorRT is not installed. Install with `pip install -e '.[trt]'` "
            "and ensure your CUDA/driver version matches. "
            "See DEPLOYMENT.md for the full setup."
        ) from e
    return trt


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    config: TrtBuildConfig | None = None,
) -> TrtBuildResult:
    """Build a serialised TRT engine from an ONNX graph.

    Args:
        onnx_path: source ONNX file.
        engine_path: destination ``.engine`` file (overwritten if exists).
        config: build knobs. Defaults to FP16, 4 GB workspace, batch=1.

    Returns:
        A ``TrtBuildResult`` with build time + engine size + TRT version.

    Raises:
        ImportError: if TensorRT isn't installed.
        FileNotFoundError: if ``onnx_path`` doesn't exist.
        RuntimeError: on engine-build failure (with the TRT log embedded).
    """
    config = config or TrtBuildConfig()
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    trt = _import_tensorrt()

    log_severity = trt.Logger.VERBOSE if config.verbose else trt.Logger.WARNING
    trt_logger = trt.Logger(log_severity)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"ONNX parse failed:\n{errors}")

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        config.workspace_mb * (1 << 20),
    )

    if config.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Platform does not support fast FP16.")
        builder_config.set_flag(trt.BuilderFlag.FP16)
    elif config.precision == "int8":
        # Validate the user-supplied calibrator BEFORE touching builder state.
        if config.int8_calibrator is None:
            raise ValueError(
                "INT8 build requires config.int8_calibrator (Phase 3). "
                "See edgevision.quantization.trt_int8 for the calibrator factory."
            )
        if not builder.platform_has_fast_int8:
            raise RuntimeError("Platform does not support fast INT8.")
        builder_config.set_flag(trt.BuilderFlag.INT8)
        builder_config.int8_calibrator = config.int8_calibrator
    elif config.precision != "fp32":
        raise ValueError(
            f"Unknown precision: {config.precision!r} (use fp32, fp16, or int8)"
        )

    if config.strict_types:
        builder_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # Build a single optimization profile for batch=[min,opt,max] over the
    # first input. RT-DETR-style models have a single (1,3,H,W) input.
    if network.num_inputs >= 1:
        input_tensor = network.get_input(0)
        shape = list(input_tensor.shape)
        if len(shape) == 4:
            min_shape = (config.min_batch, *shape[1:])
            opt_shape = (config.opt_batch, *shape[1:])
            max_shape = (config.max_batch, *shape[1:])
            profile = builder.create_optimization_profile()
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            builder_config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, builder_config)
    elapsed = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (returned None).")

    with engine_path.open("wb") as f:
        f.write(serialized)

    return TrtBuildResult(
        engine_path=str(engine_path),
        onnx_path=str(onnx_path),
        precision=config.precision,
        build_seconds=elapsed,
        file_size_mb=engine_path.stat().st_size / (1 << 20),
        trt_version=getattr(trt, "__version__", "unknown"),
    )


def trt_available() -> bool:
    """Quick check — useful in scripts that branch on TRT availability."""
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        return False
