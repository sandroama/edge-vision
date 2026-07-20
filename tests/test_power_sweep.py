"""CPU-safe validation for real Phase-5 backend and metrics dispatch."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts/run_power_sweep.py"
    spec = importlib.util.spec_from_file_location("run_power_sweep", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(path: Path, *, backend: str = "pycocotools", value: float = 0.51) -> Path:
    path.write_text(json.dumps({"backend": backend, "mAP_50_95": value}))
    return path


def test_metrics_binding_requires_real_pycocotools_map(tmp_path):
    power = _module()
    valid = power.load_accuracy_binding(str(_metrics(tmp_path / "real.json")))
    assert valid.value == pytest.approx(0.51)
    assert valid.backend == "pycocotools"

    simple = _metrics(tmp_path / "simple.json", backend="simple", value=0.0)
    with pytest.raises(power.SweepConfigurationError, match="requires a pycocotools"):
        power.load_accuracy_binding(str(simple))


def test_metrics_binding_supports_explicit_nested_selector(tmp_path):
    power = _module()
    path = tmp_path / "quant.json"
    path.write_text(
        json.dumps(
            {
                "reference": {"backend": "pycocotools", "mAP_50_95": 0.53},
                "candidate": {"backend": "pycocotools", "mAP_50_95": 0.50},
            }
        )
    )
    binding = power.load_accuracy_binding(f"{path}#candidate")
    assert binding.value == pytest.approx(0.50)
    assert binding.selector == "candidate"


def test_real_config_requires_both_artifact_and_metrics():
    power = _module()
    with pytest.raises(power.SweepConfigurationError, match="requires --artifact"):
        power.resolve_targets(
            ["onnxrt-cpu"],
            artifact_specs=[],
            metrics_specs=[],
            shape_specs=[],
            mock_power=True,
        )


def test_onnx_and_trt_dispatch_receive_explicit_artifacts(tmp_path, monkeypatch):
    power = _module()
    onnx = tmp_path / "model.onnx"
    engine = tmp_path / "model.engine"
    onnx.write_bytes(b"onnx")
    engine.write_bytes(b"engine")
    metrics = _metrics(tmp_path / "metrics.json")
    calls: list[tuple[str, Path, tuple[int, ...]]] = []

    monkeypatch.setattr(
        power,
        "_onnx_target",
        lambda path, shape: (calls.append(("onnx", path, shape)) or (lambda: None), {}),
    )
    monkeypatch.setattr(
        power,
        "_trt_target",
        lambda path, shape: (calls.append(("trt", path, shape)) or (lambda: None), {}),
    )
    targets = power.resolve_targets(
        ["onnxrt-cpu", "trt-fp16"],
        artifact_specs=[f"onnxrt-cpu={onnx}", f"trt-fp16={engine}"],
        metrics_specs=[f"onnxrt-cpu={metrics}", f"trt-fp16={metrics}"],
        shape_specs=["onnxrt-cpu=1,3,320,320"],
        mock_power=True,
    )

    assert [target.backend for target in targets] == ["onnxrt-cpu", "trt"]
    assert calls == [
        ("onnx", onnx, (1, 3, 320, 320)),
        ("trt", engine, (1, 3, 640, 640)),
    ]
    assert all(target.accuracy and target.accuracy.value == 0.51 for target in targets)


def test_cpu_backend_requires_explicit_mock_power_until_cpu_power_is_integrated(tmp_path):
    power = _module()
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"onnx")
    metrics = _metrics(tmp_path / "metrics.json")
    with pytest.raises(power.SweepConfigurationError, match="NVML cannot measure CPU power"):
        power.resolve_targets(
            ["onnxrt-cpu"],
            artifact_specs=[f"onnxrt-cpu={artifact}"],
            metrics_specs=[f"onnxrt-cpu={metrics}"],
            shape_specs=[],
            mock_power=False,
        )


def test_mock_sweep_never_reuses_published_map(tmp_path):
    power = _module()
    out = tmp_path / "power.json"
    rc = power.main(
        [
            "--configs",
            "mock-fp32",
            "--duration-sec",
            "0.01",
            "--warmup",
            "0",
            "--out-json",
            str(out),
        ]
    )
    assert rc == 0
    row = json.loads(out.read_text())["rows"][0]
    assert row["mAP_50_95"] is None
    assert row["accuracy_measured"] is False
    assert row["power_measured"] is False
    assert row["watts_per_frame"] is None


def test_tensorrt_executor_checks_artifact_before_optional_import(tmp_path):
    from edgevision.compile.trt_runtime import TensorRTExecutor

    with pytest.raises(FileNotFoundError, match="not found or empty"):
        TensorRTExecutor(tmp_path / "missing.engine")
