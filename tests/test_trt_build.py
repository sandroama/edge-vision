"""Tests for ``edgevision.compile.trt_build``.

TRT installs only on hosts with NVIDIA driver + CUDA + matching wheel, so
the bulk of these tests are gated behind ``pytest.importorskip("tensorrt")``
and ``@pytest.mark.trt``. The module-level checks (config dataclass,
``trt_available()``) run everywhere — they exercise the public surface
without requiring TRT itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgevision.compile import TrtBuildConfig, TrtBuildResult, trt_available


def test_trt_available_returns_bool():
    assert isinstance(trt_available(), bool)


def test_default_config_is_fp16_batch1_4gb():
    cfg = TrtBuildConfig()
    assert cfg.precision == "fp16"
    assert cfg.workspace_mb == 4096
    assert cfg.min_batch == 1
    assert cfg.opt_batch == 1
    assert cfg.max_batch == 1
    assert cfg.int8_calibrator is None


def test_config_dataclass_accepts_overrides():
    cfg = TrtBuildConfig(
        precision="fp32", workspace_mb=2048, max_batch=8, verbose=True
    )
    assert cfg.precision == "fp32"
    assert cfg.workspace_mb == 2048
    assert cfg.max_batch == 8
    assert cfg.verbose is True


def test_build_result_as_dict_roundtrip():
    result = TrtBuildResult(
        engine_path="x.engine",
        onnx_path="x.onnx",
        precision="fp16",
        build_seconds=12.34,
        file_size_mb=42.0,
        trt_version="10.5.0",
    )
    payload = result.as_dict()
    json.loads(json.dumps(payload))
    assert payload["precision"] == "fp16"
    assert payload["build_seconds"] == 12.34


def test_build_engine_raises_on_missing_onnx(tmp_path: Path):
    pytest.importorskip("tensorrt")
    from edgevision.compile import build_engine

    with pytest.raises(FileNotFoundError):
        build_engine(tmp_path / "missing.onnx", tmp_path / "out.engine")


def test_build_engine_int8_requires_calibrator(tmp_path: Path):
    pytest.importorskip("tensorrt")
    from edgevision.compile import build_engine

    onnx_path = tmp_path / "in.onnx"
    onnx_path.write_bytes(b"unused")  # non-empty placeholder
    with pytest.raises((ValueError, RuntimeError)):
        build_engine(
            onnx_path,
            tmp_path / "out.engine",
            config=TrtBuildConfig(precision="int8"),
        )


# --------------------------------------------------------------------------- GPU paths

# These tests actually build engines — gated behind a GPU runner.


@pytest.mark.trt
@pytest.mark.gpu
@pytest.mark.slow
def test_build_engine_fp16_round_trip(tmp_path: Path):
    pytest.importorskip("tensorrt")
    pytest.importorskip("torch")

    from edgevision.compile import build_engine, export_to_onnx
    from edgevision.models.tiny_model import make_tiny_input, make_tiny_model

    onnx_path = tmp_path / "tiny.onnx"
    engine_path = tmp_path / "tiny_fp16.engine"
    export_to_onnx(make_tiny_model(), make_tiny_input(), onnx_path)

    result = build_engine(
        onnx_path,
        engine_path,
        config=TrtBuildConfig(precision="fp16", workspace_mb=512),
    )
    assert engine_path.exists()
    assert engine_path.stat().st_size > 0
    assert result.precision == "fp16"
    assert result.build_seconds > 0
