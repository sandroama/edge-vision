"""Phase 5 sustained backend sweep with explicit artifact and metric provenance.

Real configurations fail closed: each label must bind to an executable ONNX
or TensorRT artifact and to a pycocotools metrics JSON object produced for that
same configuration. Mock rows deliberately write ``mAP_50_95=null`` and are
excluded from Pareto aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from edgevision.profiling import run_sustained
from edgevision.profiling.thermal_runner import ThermalRunResult


class SweepConfigurationError(ValueError):
    """Raised before measurement when bindings cannot support a real run."""


@dataclass(frozen=True)
class AccuracyBinding:
    """A measured COCO mAP value and its exact artifact location."""

    value: float
    path: Path
    selector: str
    backend: str


@dataclass(frozen=True)
class SweepTarget:
    """Preflighted executable configuration."""

    label: str
    backend: str
    precision: str
    artifact_path: Path | None
    accuracy: AccuracyBinding | None
    fn: Any
    runtime: dict[str, Any]


def _bindings(values: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        label, separator, bound = value.partition("=")
        if not separator or not label or not bound:
            raise SweepConfigurationError(f"{option} expects LABEL=VALUE; got {value!r}")
        if label in result:
            raise SweepConfigurationError(f"duplicate {option} binding for {label!r}")
        result[label] = bound
    return result


def _backend_for(label: str) -> tuple[str, str]:
    precision = next((p for p in ("fp32", "fp16", "int8") if p in label), "fp32")
    if label.startswith("mock-"):
        return "mock", precision
    if "onnxrt" in label or label.endswith("-cpu"):
        return "onnxrt-cpu", precision
    if label.startswith("trt-") or label.startswith("distilled-"):
        return "trt", precision
    raise SweepConfigurationError(
        f"cannot infer backend for {label!r}; use onnxrt-*, trt-*, distilled-*, or mock-*"
    )


def _input_shape(spec: str | None) -> tuple[int, ...]:
    if spec is None:
        return (1, 3, 640, 640)
    try:
        shape = tuple(int(part) for part in spec.split(","))
    except ValueError as exc:
        raise SweepConfigurationError(
            f"input shape must be comma-separated integers: {spec!r}"
        ) from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise SweepConfigurationError(f"input shape dimensions must be positive: {spec!r}")
    return shape


def _select_json_object(data: Any, selector: str, source: Path) -> dict[str, Any]:
    current = data
    if selector:
        for key in selector.split("."):
            if not isinstance(current, dict) or key not in current:
                raise SweepConfigurationError(
                    f"metrics selector {selector!r} not found in {source}"
                )
            current = current[key]
    if not isinstance(current, dict):
        raise SweepConfigurationError(
            f"metrics selector {selector or '<root>'!r} in {source} is not an object"
        )
    return current


def load_accuracy_binding(spec: str) -> AccuracyBinding:
    """Load ``PATH[#dotted.selector]`` and require real pycocotools mAP."""
    path_text, separator, selector = spec.partition("#")
    path = Path(path_text)
    if not path.is_file() or path.stat().st_size == 0:
        raise SweepConfigurationError(f"metrics artifact not found or empty: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SweepConfigurationError(f"metrics artifact is not valid JSON: {path}") from exc
    selected = _select_json_object(data, selector if separator else "", path)
    backend = selected.get("backend")
    if backend != "pycocotools":
        raise SweepConfigurationError(
            f"{path}#{selector or '<root>'} uses backend={backend!r}; real Pareto mAP "
            "requires a pycocotools COCO evaluation, not simple/mock metrics"
        )
    value = selected.get("mAP_50_95")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepConfigurationError(f"{path}#{selector or '<root>'} has no numeric mAP_50_95")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise SweepConfigurationError(f"mAP_50_95 must be in [0,1], got {value}")
    return AccuracyBinding(value=value, path=path.resolve(), selector=selector, backend=backend)


def _onnx_target(path: Path, shape: tuple[int, ...]) -> tuple[Any, dict[str, Any]]:
    from edgevision.compile import OnnxRuntimeCPUExecutor

    executor = OnnxRuntimeCPUExecutor(path, graph_optimization="all")
    if len(executor.input_shape) != len(shape):
        raise SweepConfigurationError(
            f"ONNX input rank is {len(executor.input_shape)}; requested shape has rank {len(shape)}"
        )
    incompatible = [
        (index, expected, requested)
        for index, (expected, requested) in enumerate(zip(executor.input_shape, shape, strict=True))
        if isinstance(expected, int) and expected > 0 and expected != requested
    ]
    if incompatible:
        raise SweepConfigurationError(
            f"ONNX static dimensions do not match requested shape {shape}: {incompatible}"
        )
    x = np.zeros(shape, dtype=np.float32)
    return executor.make_callable(x), executor.describe()


def _trt_target(path: Path, shape: tuple[int, ...]) -> tuple[Any, dict[str, Any]]:
    from edgevision.compile import TensorRTExecutor

    executor = TensorRTExecutor(path, input_shape=shape)
    return executor.make_callable(), executor.describe()


def resolve_targets(
    labels: list[str],
    *,
    artifact_specs: list[str],
    metrics_specs: list[str],
    shape_specs: list[str],
    mock_power: bool,
) -> list[SweepTarget]:
    """Preflight all bindings before starting any sustained measurement."""
    artifacts = _bindings(artifact_specs, "--artifact")
    metrics = _bindings(metrics_specs, "--metrics")
    shapes = _bindings(shape_specs, "--input-shape")
    unknown = (set(artifacts) | set(metrics) | set(shapes)) - set(labels)
    if unknown:
        raise SweepConfigurationError(
            f"bindings supplied for configs not in --configs: {sorted(unknown)}"
        )

    targets: list[SweepTarget] = []
    for label in labels:
        backend, precision = _backend_for(label)
        if backend == "mock":
            targets.append(
                SweepTarget(
                    label=label,
                    backend=backend,
                    precision=precision,
                    artifact_path=None,
                    accuracy=None,
                    fn=lambda: time.sleep(0.002),
                    runtime={"kind": "mock", "accuracy_measured": False},
                )
            )
            continue

        if label not in artifacts:
            raise SweepConfigurationError(
                f"{label!r} requires --artifact {label}=PATH ({backend} executable)"
            )
        if label not in metrics:
            raise SweepConfigurationError(
                f"{label!r} requires --metrics {label}=PATH[#selector]; mAP is never inherited"
            )
        if backend == "onnxrt-cpu" and not mock_power:
            raise SweepConfigurationError(
                f"{label!r} is CPU inference. Pass --mock-power for a latency/dispatch "
                "validation row (marked power_measured=false); NVML cannot measure CPU power."
            )

        path = Path(artifacts[label])
        if not path.is_file() or path.stat().st_size == 0:
            raise SweepConfigurationError(f"artifact not found or empty for {label!r}: {path}")
        accuracy = load_accuracy_binding(metrics[label])
        shape = _input_shape(shapes.get(label))
        try:
            if backend == "onnxrt-cpu":
                fn, runtime = _onnx_target(path, shape)
            else:
                fn, runtime = _trt_target(path, shape)
        except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SweepConfigurationError(f"cannot initialize {label!r}: {exc}") from exc
        targets.append(
            SweepTarget(
                label=label,
                backend=backend,
                precision=precision,
                artifact_path=path.resolve(),
                accuracy=accuracy,
                fn=fn,
                runtime=runtime,
            )
        )
    return targets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="edge-vision Phase-5 power sweep")
    parser.add_argument("--configs", nargs="+", default=["mock-fp32"])
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Executable ONNX or TensorRT artifact; required once per real config.",
    )
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        metavar="LABEL=PATH[#SELECTOR]",
        help="Real pycocotools metrics object; required once per real config.",
    )
    parser.add_argument(
        "--input-shape",
        action="append",
        default=[],
        metavar="LABEL=N,C,H,W",
        help="Per-config runtime shape; defaults to 1,3,640,640.",
    )
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--sample-ms", type=float, default=100.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--mock-power",
        action="store_true",
        help="Explicit synthetic-power mode. Rows are marked power_measured=false.",
    )
    parser.add_argument("--out-json", default="docs/results/phase5_power.json")
    parser.add_argument("--resume", action="store_true", help="Preserve other existing labels.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        targets = resolve_targets(
            args.configs,
            artifact_specs=args.artifact,
            metrics_specs=args.metrics,
            shape_specs=args.input_shape,
            mock_power=args.mock_power,
        )
    except SweepConfigurationError as exc:
        print(f"[edge-vision] configuration error: {exc}", file=sys.stderr)
        return 2

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_label: dict[str, dict[str, Any]] = {}
    if args.resume and out_path.exists():
        try:
            rows_by_label = {
                str(row["config_label"]): row
                for row in json.loads(out_path.read_text()).get("rows", [])
            }
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[edge-vision] cannot resume malformed {out_path}: {exc}", file=sys.stderr)
            return 2

    for target in targets:
        print(f"[edge-vision] sweeping {target.label!r} for {args.duration_sec}s")
        use_mock_power = args.mock_power or target.backend == "mock"
        result: ThermalRunResult = run_sustained(
            target.fn,
            duration_s=args.duration_sec,
            sample_ms=args.sample_ms,
            device_index=args.device_index,
            config_label=target.label,
            warmup_iterations=args.warmup,
            use_mock_power=use_mock_power,
        )
        row = result.as_dict()
        row.update(
            {
                "mAP_50_95": target.accuracy.value if target.accuracy else None,
                "accuracy_measured": target.accuracy is not None,
                "metrics_artifact": str(target.accuracy.path) if target.accuracy else None,
                "metrics_selector": target.accuracy.selector if target.accuracy else None,
                "metrics_backend": target.accuracy.backend if target.accuracy else None,
                "artifact_path": str(target.artifact_path) if target.artifact_path else None,
                "size_mb": (
                    round(target.artifact_path.stat().st_size / (1 << 20), 3)
                    if target.artifact_path
                    else 0.0
                ),
                "backend": target.backend,
                "precision": target.precision,
                "runtime": target.runtime,
                "power_measured": not use_mock_power,
                "power_source": "mock" if use_mock_power else "nvml",
            }
        )
        if use_mock_power:
            row["watts_per_frame"] = None
        rows_by_label[target.label] = row
        out_path.write_text(json.dumps({"rows": list(rows_by_label.values())}, indent=2) + "\n")
        print("  -> " + result.as_row())

    from edgevision.evaluation.pareto_aggregator import load_configs_from_jsons, write_report

    configs = load_configs_from_jsons([out_path])
    if configs:
        print(f"[edge-vision] Pareto table -> {write_report(configs, out_dir=out_path.parent)}")
    else:
        print("[edge-vision] no fully measured accuracy+power rows; Pareto report not written")
    print(f"[edge-vision] wrote {len(rows_by_label)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
