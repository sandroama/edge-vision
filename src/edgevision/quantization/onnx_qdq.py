"""ONNX Runtime static INT8 quantization (the CPU path for RQ-E1).

ORT exposes a ``CalibrationDataReader`` interface — implement it once,
hand it to ``onnxruntime.quantization.quantize_static``, and out comes a
quantized ``.onnx`` ready for the ``CPUExecutionProvider``.

We do **static** (per-tensor) quantization here, not dynamic. Static gives
better accuracy by collecting activation ranges over a real calibration
set; dynamic just quantizes weights and computes scales at runtime.

The ``QDQQuantizationConfig`` knobs mirror the TRT side so RQ-E1 reports
apples-to-apples comparisons across CPU and GPU INT8 paths.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgevision.quantization.calib_dataset import BatchProvider


@dataclass
class QDQQuantizationConfig:
    """Knobs for ONNX Runtime static quantization."""

    # ORT's quantization mode — "QDQ" (recommended) or "QOperator".
    quant_format: str = "QDQ"
    # Per-channel weights = better accuracy, slightly slower. Worth it.
    per_channel: bool = True
    # Activation type: QInt8 (signed) or QUInt8 (unsigned).
    activation_type: str = "QInt8"
    weight_type: str = "QInt8"
    # Auto-resolve via ORT's preprocess (fold constants etc) before calibrating.
    preprocess: bool = True
    # Names of activation nodes to leave at FP32 (useful for output heads).
    nodes_to_exclude: tuple[str, ...] = ()
    # Optimisation passes ORT applies before quantization.
    extra_options: dict = None  # type: ignore[assignment]


@dataclass(frozen=True)
class QDQQuantizationResult:
    """Bookkeeping for a successful static-quant pass."""

    onnx_in_path: str
    onnx_out_path: str
    n_calibration_images: int
    output_size_mb: float
    input_size_mb: float
    size_reduction_pct: float

    def as_dict(self) -> dict:
        return {
            "onnx_in_path": self.onnx_in_path,
            "onnx_out_path": self.onnx_out_path,
            "n_calibration_images": self.n_calibration_images,
            "output_size_mb": round(self.output_size_mb, 2),
            "input_size_mb": round(self.input_size_mb, 2),
            "size_reduction_pct": round(self.size_reduction_pct, 1),
        }


# --------------------------------------------------------------------------- data reader


class _ProviderDataReader:
    """Adapt our ``BatchProvider`` to ORT's ``CalibrationDataReader`` interface.

    ORT calls ``get_next()`` repeatedly until it returns None. Each call
    must return a dict ``{input_name: np.ndarray}`` matching the model's
    input. We simply pull the next batch from the provider.
    """

    def __init__(self, provider: BatchProvider, input_name: str) -> None:
        self._provider = provider
        self._iter: Iterable[np.ndarray] | None = None
        self._input_name = input_name

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iter is None:
            self._iter = iter(self._provider)
        try:
            batch = next(self._iter)  # type: ignore[arg-type]
        except StopIteration:
            return None
        return {self._input_name: batch}

    def rewind(self) -> None:  # pragma: no cover - rarely used by ORT
        self._iter = None


# --------------------------------------------------------------------------- public API


def quantize_static(
    onnx_in: str | Path,
    onnx_out: str | Path,
    *,
    provider: BatchProvider,
    input_name: str = "images",
    config: QDQQuantizationConfig | None = None,
) -> QDQQuantizationResult:
    """Run ORT static INT8 quantization.

    Args:
        onnx_in: path to FP32 ONNX (e.g. the Phase-2 export).
        onnx_out: where to write the quantized ONNX.
        provider: calibration batches; iterated once during quant.
        input_name: graph input name to bind batches to. Defaults to
            ``"images"`` (matches ``onnx_export.DEFAULT_INPUT_NAMES``).
        config: knobs; default QDQ + per-channel int8.

    Returns:
        ``QDQQuantizationResult`` with file sizes + size reduction %.

    Raises:
        ImportError: if onnxruntime.quantization is not installed.
        FileNotFoundError: if ``onnx_in`` doesn't exist.
    """
    try:
        import onnxruntime as ort  # noqa: F401  (validates presence)
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
        )
        from onnxruntime.quantization import (
            quantize_static as _ort_quantize_static,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "onnx_qdq requires onnxruntime[quantization]. "
            "Install with: pip install -e '.[dev]'"
        ) from e

    config = config or QDQQuantizationConfig()
    onnx_in = Path(onnx_in)
    onnx_out = Path(onnx_out)
    if not onnx_in.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_in}")
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    quant_format = {
        "QDQ": QuantFormat.QDQ,
        "QOperator": QuantFormat.QOperator,
    }[config.quant_format]
    activation_type = {
        "QInt8": QuantType.QInt8,
        "QUInt8": QuantType.QUInt8,
    }[config.activation_type]
    weight_type = {
        "QInt8": QuantType.QInt8,
        "QUInt8": QuantType.QUInt8,
    }[config.weight_type]

    reader = _ProviderDataReader(provider, input_name=input_name)

    _ort_quantize_static(
        model_input=str(onnx_in),
        model_output=str(onnx_out),
        calibration_data_reader=reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=config.per_channel,
        nodes_to_exclude=list(config.nodes_to_exclude),
        extra_options=config.extra_options or {},
    )

    in_mb = onnx_in.stat().st_size / (1 << 20)
    out_mb = onnx_out.stat().st_size / (1 << 20)
    reduction = (1.0 - out_mb / in_mb) * 100 if in_mb > 0 else 0.0

    return QDQQuantizationResult(
        onnx_in_path=str(onnx_in),
        onnx_out_path=str(onnx_out),
        n_calibration_images=provider.total_images,
        output_size_mb=out_mb,
        input_size_mb=in_mb,
        size_reduction_pct=reduction,
    )


def is_qdq_supported() -> bool:
    """Quick availability check for the QDQ quantization import path."""
    try:
        import onnxruntime.quantization  # noqa: F401

        return True
    except ImportError:
        return False
