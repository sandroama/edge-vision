"""Tests for ``edgevision.quantization.onnx_qdq``.

The export → quantize round-trip needs torch + onnx + onnxruntime +
``onnxruntime.quantization``. All four are in the project's base deps but
the alpha-3.15 development environment may not have them; skip cleanly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edgevision.data.coco_loader import CocoDataset
from edgevision.quantization import (
    QDQQuantizationConfig,
    QDQQuantizationResult,
    build_calibration_provider,
    is_qdq_supported,
)


def test_is_qdq_supported_returns_bool():
    assert isinstance(is_qdq_supported(), bool)


def test_default_config_per_channel_qdq_int8():
    cfg = QDQQuantizationConfig()
    assert cfg.quant_format == "QDQ"
    assert cfg.per_channel is True
    assert cfg.activation_type == "QInt8"
    assert cfg.weight_type == "QInt8"


def test_quant_result_as_dict_round_trip():
    res = QDQQuantizationResult(
        onnx_in_path="in.onnx",
        onnx_out_path="out.onnx",
        n_calibration_images=128,
        output_size_mb=10.5,
        input_size_mb=42.0,
        size_reduction_pct=75.0,
    )
    payload = res.as_dict()
    import json

    json.loads(json.dumps(payload))
    assert payload["n_calibration_images"] == 128
    assert payload["size_reduction_pct"] == 75.0


# --------------------------------------------------------------------------- end-to-end


@pytest.mark.slow
def test_quantize_static_round_trip(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxruntime.quantization")

    from edgevision.compile import OnnxRuntimeCPUExecutor, export_to_onnx
    from edgevision.models.tiny_model import make_tiny_input, make_tiny_model
    from edgevision.quantization import quantize_static

    # 1. Export FP32.
    fp32 = tmp_path / "tiny.onnx"
    export_to_onnx(make_tiny_model(), make_tiny_input(), fp32)

    # 2. Build calibration provider on a synthetic dataset.
    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=1)
    provider = build_calibration_provider(ds, n=4, target_size=(640, 640), batch_size=1)

    # 3. Quantize.
    int8 = tmp_path / "tiny_int8.onnx"
    result = quantize_static(fp32, int8, provider=provider, input_name="images")
    assert int8.exists()
    assert int8.stat().st_size > 0
    assert isinstance(result, QDQQuantizationResult)
    assert result.n_calibration_images == 4

    # 4. Round-trip via ORT to make sure the quantized graph is loadable.
    executor = OnnxRuntimeCPUExecutor(int8)
    x = np.zeros((1, 3, 640, 640), dtype=np.float32)
    out = executor.run(x)
    assert "logits" in out.outputs
    assert "pred_boxes" in out.outputs


def test_quantize_static_raises_on_missing_input(tmp_path: Path):
    pytest.importorskip("onnxruntime.quantization")

    from edgevision.quantization import quantize_static

    ds = CocoDataset.synthetic(n_images=2, boxes_per_image=1)
    provider = build_calibration_provider(ds, n=2, target_size=(160, 160), batch_size=1)

    with pytest.raises(FileNotFoundError):
        quantize_static(
            tmp_path / "missing.onnx",
            tmp_path / "out.onnx",
            provider=provider,
        )
