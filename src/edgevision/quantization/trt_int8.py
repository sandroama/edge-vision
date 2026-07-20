"""TensorRT INT8 entropy calibrator (the GPU path for RQ-E1).

Wraps a ``BatchProvider`` in TRT's ``IInt8EntropyCalibrator2`` API. The
calibrator is what ``trt_build.build_engine`` consumes when
``config.precision == "int8"``.

What entropy calibration does, in one paragraph: TRT runs the model in
FP32 on the calibration set, watches activation histograms at every
quantizable tensor, and picks per-tensor scales that minimise the KL
divergence between FP32 and INT8 distributions. The output is a small
``.cache`` file mapping tensor names → INT8 scales. Re-builds reuse the
cache and skip the FP32 pass.

Lazy-imports tensorrt + pycuda. Calling :func:`make_int8_calibrator`
without those deps raises a clear ImportError.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from edgevision.quantization.calib_dataset import BatchProvider


def _import_trt_and_cuda():
    try:
        import tensorrt as trt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "TensorRT not installed. Install with `pip install -e '.[trt]'`."
        ) from e
    try:
        import pycuda.autoinit  # noqa: F401  (initialises CUDA context)
        import pycuda.driver as cuda
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pycuda not installed. Install with `pip install -e '.[trt]'`."
        ) from e
    return trt, cuda


def make_int8_calibrator(
    provider: BatchProvider,
    cache_path: str | Path,
    *,
    input_name: str = "images",
) -> Any:
    """Construct an ``IInt8EntropyCalibrator2`` bound to ``provider``.

    Args:
        provider: calibration batches; iterated once unless the cache exists.
        cache_path: ``.cache`` file location. Reused across builds — delete
            it to force fresh calibration.
        input_name: must match the engine's input tensor name.

    Returns:
        A TRT calibrator object. Pass it to
        ``TrtBuildConfig(precision="int8", int8_calibrator=<this>)``.
    """
    trt, cuda = _import_trt_and_cuda()

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self._provider = provider
            self._iter = iter(provider)
            self._cache_file = cache_path
            self._device_buffer: Any | None = None
            self._buffer_size: int = 0
            self._input_name = input_name

        def get_batch_size(self) -> int:
            return self._provider.batch_size

        def get_batch(self, names: list[str]) -> list[int] | None:
            try:
                batch = next(self._iter)
            except StopIteration:
                return None

            # Flatten + ensure float32 contiguous.
            host = np.ascontiguousarray(batch, dtype=np.float32)
            nbytes = host.nbytes
            if self._device_buffer is None or self._buffer_size < nbytes:
                if self._device_buffer is not None:
                    self._device_buffer.free()
                self._device_buffer = cuda.mem_alloc(nbytes)
                self._buffer_size = nbytes
            cuda.memcpy_htod(self._device_buffer, host)
            return [int(self._device_buffer)]

        def read_calibration_cache(self) -> bytes | None:
            if self._cache_file.exists():
                return self._cache_file.read_bytes()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            self._cache_file.write_bytes(cache)

    return _EntropyCalibrator()


def build_int8_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    provider: BatchProvider,
    cache_path: str | Path | None = None,
    *,
    workspace_mb: int = 4096,
    input_name: str = "images",
) -> Any:
    """High-level convenience: ONNX + calibration set → INT8 ``.engine``.

    Wraps :func:`make_int8_calibrator` and
    :func:`edgevision.compile.trt_build.build_engine` in one call. Use
    this from scripts; use the lower-level functions for finer control.
    """
    from edgevision.compile.trt_build import TrtBuildConfig, build_engine

    cache_path = Path(cache_path) if cache_path else Path(engine_path).with_suffix(".calib.cache")
    calibrator = make_int8_calibrator(provider, cache_path, input_name=input_name)
    return build_engine(
        onnx_path,
        engine_path,
        config=TrtBuildConfig(
            precision="int8",
            workspace_mb=workspace_mb,
            int8_calibrator=calibrator,
        ),
    )


def is_trt_int8_supported() -> bool:
    """Quick availability check — TRT + pycuda + INT8 platform support."""
    try:
        import tensorrt as trt

        try:
            import pycuda.driver  # noqa: F401
        except ImportError:
            return False
        # Constructing a Builder needs an initialised CUDA context, which
        # autoinit handles. We only return True if the build call would be
        # allowed to use INT8.
        try:
            import pycuda.autoinit  # noqa: F401
        except Exception:
            return False
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        return builder.platform_has_fast_int8
    except ImportError:
        return False
