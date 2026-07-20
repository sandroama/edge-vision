"""Tests for ``edgevision.quantization.trt_int8``.

Most checks here run anywhere — they exercise the function signatures and
the ``is_trt_int8_supported`` predicate. The actual calibrator construction
needs ``tensorrt`` + ``pycuda`` + a CUDA context; gated behind importorskip
+ ``@pytest.mark.gpu`` so CI doesn't try to allocate CUDA memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edgevision.data.coco_loader import CocoDataset
from edgevision.quantization import (
    build_calibration_provider,
    is_trt_int8_supported,
)


def test_is_trt_int8_supported_returns_bool():
    assert isinstance(is_trt_int8_supported(), bool)


def test_make_int8_calibrator_raises_without_tensorrt(tmp_path: Path, monkeypatch):
    """Without TRT installed, the calibrator factory should raise ImportError.

    We simulate the absence by importorskip'ing tensorrt — if it's
    available on the runner, the real test below covers it instead.
    """
    if is_trt_int8_supported():
        pytest.skip("tensorrt is installed; absence-of-dep test not applicable")

    from edgevision.quantization import make_int8_calibrator

    ds = CocoDataset.synthetic(n_images=2, boxes_per_image=1)
    provider = build_calibration_provider(ds, n=2, target_size=(160, 160), batch_size=1)
    with pytest.raises(ImportError, match="(?i)tensorrt|pycuda"):
        make_int8_calibrator(provider, tmp_path / "calib.cache")


# --------------------------------------------------------------------------- GPU paths


@pytest.mark.trt
@pytest.mark.gpu
@pytest.mark.slow
def test_make_int8_calibrator_constructs_calibrator(tmp_path: Path):
    pytest.importorskip("tensorrt")
    pytest.importorskip("pycuda.driver")

    from edgevision.quantization import make_int8_calibrator

    ds = CocoDataset.synthetic(n_images=2, boxes_per_image=1)
    provider = build_calibration_provider(ds, n=2, target_size=(160, 160), batch_size=1)
    calib = make_int8_calibrator(provider, tmp_path / "calib.cache")
    assert calib.get_batch_size() == 1
    assert calib.read_calibration_cache() is None  # cache file not created yet


@pytest.mark.trt
@pytest.mark.gpu
@pytest.mark.slow
def test_build_int8_engine_smoke(tmp_path: Path):
    pytest.importorskip("tensorrt")
    pytest.importorskip("pycuda.driver")
    pytest.importorskip("torch")

    from edgevision.compile import export_to_onnx
    from edgevision.models.tiny_model import make_tiny_input, make_tiny_model
    from edgevision.quantization import build_int8_engine

    onnx_path = tmp_path / "tiny.onnx"
    engine_path = tmp_path / "tiny_int8.engine"
    export_to_onnx(make_tiny_model(), make_tiny_input(), onnx_path)

    ds = CocoDataset.synthetic(n_images=4, boxes_per_image=1)
    provider = build_calibration_provider(
        ds, n=4, target_size=(640, 640), batch_size=1
    )

    result = build_int8_engine(onnx_path, engine_path, provider, workspace_mb=256)
    assert engine_path.exists()
    assert engine_path.stat().st_size > 0
    assert result.precision == "int8"
