"""TensorRT runtime executor for sustained latency and power workloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class TensorRTExecutor:
    """Load a TensorRT 10 engine and expose synchronized batch-1 inference."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        input_shape: tuple[int, ...] = (1, 3, 640, 640),
    ) -> None:
        engine_path = Path(engine_path)
        if not engine_path.is_file() or engine_path.stat().st_size == 0:
            raise FileNotFoundError(f"TensorRT engine not found or empty: {engine_path}")
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover - hardware environment
            raise ImportError(
                "TensorRT execution requires tensorrt and pycuda. Install with "
                "`pip install -e '.[trt]'` on a host with a compatible NVIDIA driver."
            ) from exc

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(
                f"TensorRT could not deserialize {engine_path}; rebuild it on this GPU/TRT host."
            )
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT failed to create an execution context")

        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_names = [
            name for name in tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        if len(input_names) != 1:
            raise ValueError(
                f"power sweep supports one-input engines; found {len(input_names)}: {input_names}"
            )
        input_name = input_names[0]
        declared = tuple(engine.get_tensor_shape(input_name))
        if any(dim <= 0 for dim in declared):
            if not context.set_input_shape(input_name, input_shape):
                raise ValueError(f"TensorRT rejected input shape {input_shape} for {input_name!r}")
        elif declared != input_shape:
            raise ValueError(
                f"engine expects input shape {declared}, but sweep requested {input_shape}"
            )

        self._trt = trt
        self._runtime = runtime
        self._engine = engine
        self._context = context
        self._stream = cuda.Stream()
        self._host_buffers: list[np.ndarray] = []
        self._device_buffers: list[Any] = []
        self.engine_path = engine_path
        self.input_shape = input_shape

        for name in tensor_names:
            shape = tuple(context.get_tensor_shape(name))
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"unresolved TensorRT shape for {name!r}: {shape}")
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            host = np.zeros(int(np.prod(shape)), dtype=dtype)
            device = cuda.mem_alloc(host.nbytes)
            self._host_buffers.append(host)
            self._device_buffers.append(device)
            context.set_tensor_address(name, int(device))
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                cuda.memcpy_htod(device, host)

    def run(self) -> None:
        """Execute one inference and synchronize before returning."""
        ok = self._context.execute_async_v3(stream_handle=self._stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        self._stream.synchronize()

    def make_callable(self):
        """Return the synchronized zero-argument inference callable."""
        return self.run

    def describe(self) -> dict[str, Any]:
        """Return runtime provenance for result artifacts."""
        return {
            "engine_path": str(self.engine_path),
            "input_shape": list(self.input_shape),
            "tensorrt_version": getattr(self._trt, "__version__", "unknown"),
        }
