"""Tests for the Phase-2 compile-smoke script.

The script runs the export → ORT-CPU stages by default. We exercise them
when torch + onnxruntime are present (CI) and skip otherwise.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from scripts.run_compile_smoke import main as compile_main  # noqa: E402


@pytest.mark.slow
def test_compile_smoke_onnx_stage_only(tmp_path: Path):
    out_dir = tmp_path / "checkpoints"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = compile_main(["--stage", "onnx", "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "tiny_detector.onnx").exists()
    assert "Exporting" in buf.getvalue()
    assert "DONE" in buf.getvalue()


@pytest.mark.slow
def test_compile_smoke_onnxrt_stage_runs_inference(tmp_path: Path):
    out_dir = tmp_path / "checkpoints"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = compile_main(["--stage", "all", "--out-dir", str(out_dir), "--n-iter", "5"])
    assert rc == 0
    assert (out_dir / "tiny_detector.onnx").exists()
    assert "ONNX Runtime CPU" in buf.getvalue()
    # The TRT stage will print a "not installed" message in CI environments.
    assert "DONE" in buf.getvalue()
