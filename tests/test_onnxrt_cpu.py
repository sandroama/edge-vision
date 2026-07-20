"""Tests for ``edgevision.compile.onnxrt_cpu``.

Uses an exported tiny model (round-trip via the export pipeline) so the
test exercises the real ORT load path. Needs ``torch`` + ``onnxruntime``;
otherwise the whole module is skipped via ``importorskip``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from edgevision.compile import (  # noqa: E402
    OnnxRuntimeCPUExecutor,
    OnnxRuntimeOutputs,
    export_to_onnx,
)
from edgevision.models.tiny_model import make_tiny_input, make_tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def tiny_onnx_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("onnxrt") / "tiny.onnx"
    export_to_onnx(make_tiny_model(), make_tiny_input(), out)
    return out


@pytest.mark.slow
def test_executor_describes_io(tiny_onnx_path: Path):
    ex = OnnxRuntimeCPUExecutor(tiny_onnx_path)
    desc = ex.describe()
    assert desc["input_name"] == "images"
    assert desc["output_names"] == ["logits", "pred_boxes"]
    assert "CPUExecutionProvider" in desc["providers"]


@pytest.mark.slow
def test_executor_runs_a_forward_pass(tiny_onnx_path: Path):
    ex = OnnxRuntimeCPUExecutor(tiny_onnx_path)
    x = np.zeros((1, 3, 640, 640), dtype=np.float32)
    out = ex.run(x)
    assert isinstance(out, OnnxRuntimeOutputs)
    assert "logits" in out.outputs
    assert "pred_boxes" in out.outputs
    assert out.get("logits").shape[0] == 1


@pytest.mark.slow
def test_executor_accepts_non_float32_input(tiny_onnx_path: Path):
    """The wrapper auto-casts to float32 if the user hands in a different dtype."""
    ex = OnnxRuntimeCPUExecutor(tiny_onnx_path)
    x = np.zeros((1, 3, 640, 640), dtype=np.float64)
    out = ex.run(x)
    assert "logits" in out.outputs


@pytest.mark.slow
def test_make_callable_works_with_latency_harness(tiny_onnx_path: Path):
    from edgevision.inference import measure_latency_cpu

    ex = OnnxRuntimeCPUExecutor(tiny_onnx_path)
    x = np.zeros((1, 3, 640, 640), dtype=np.float32)
    result = measure_latency_cpu(ex.make_callable(x), n_runs=3, n_warmup=2)
    assert result.n == 3
    assert result.p50_ms > 0


def test_executor_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        OnnxRuntimeCPUExecutor(tmp_path / "does-not-exist.onnx")


@pytest.mark.slow
def test_outputs_get_raises_on_unknown_name(tiny_onnx_path: Path):
    ex = OnnxRuntimeCPUExecutor(tiny_onnx_path)
    out = ex.run(np.zeros((1, 3, 640, 640), dtype=np.float32))
    with pytest.raises(KeyError, match="not produced"):
        out.get("nonsense")
