"""ONNX export — the first stage of the compile pipeline.

The contract is small on purpose:

    1. ``export_to_onnx`` takes a torch ``nn.Module`` + a dummy input,
       and writes an ONNX graph to disk. Dynamic batch axis on by default.
    2. ``verify_onnx`` re-loads the file with the ``onnx`` package and
       returns a small metadata dict (opset, inputs, outputs, n_params)
       that downstream code can sanity-check.

Both functions lazy-import their heavy deps so importing this module on a
no-torch CI runner does not fail.

The exported graph is what the rest of the compile pipeline operates on —
TensorRT consumes it, ONNX Runtime CPU consumes it, CoreML (Phase 7) would
consume it. Keeping export pinned to opset 17 means we don't move under
TensorRT's feet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OPSET = 17
DEFAULT_INPUT_NAMES = ("images",)
DEFAULT_OUTPUT_NAMES = ("logits", "pred_boxes")


@dataclass(frozen=True)
class OnnxModelInfo:
    """Sanity-check metadata about an exported ONNX graph."""

    path: str
    opset: int
    ir_version: int
    producer: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_shapes: dict[str, tuple]   # ints or strings (for dynamic axes)
    output_shapes: dict[str, tuple]
    n_initializers: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "opset": self.opset,
            "ir_version": self.ir_version,
            "producer": self.producer,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "input_shapes": {k: list(v) for k, v in self.input_shapes.items()},
            "output_shapes": {k: list(v) for k, v in self.output_shapes.items()},
            "n_initializers": self.n_initializers,
        }


def export_to_onnx(
    model: Any,
    dummy_input: Any,
    output_path: str | Path,
    *,
    opset: int = DEFAULT_OPSET,
    dynamic_batch: bool = True,
    input_names: tuple[str, ...] = DEFAULT_INPUT_NAMES,
    output_names: tuple[str, ...] = DEFAULT_OUTPUT_NAMES,
    do_constant_folding: bool = True,
) -> Path:
    """Export ``model(dummy_input)`` to an ONNX file at ``output_path``.

    Args:
        model: a ``torch.nn.Module`` already on the device the dummy is on.
        dummy_input: a torch tensor (or tuple) matching the model's signature.
            For RT-DETR-style models the convention is a ``(1, 3, H, W)``
            float32 tensor.
        output_path: where the ``.onnx`` file is written. Parents created.
        opset: ONNX opset version. Pinned to 17 by default — TensorRT 10.x
            supports it stably and most ops we care about are emitted.
        dynamic_batch: if True, the batch axis is exported as dynamic. Lets
            us run batch=1 latency benches and batch=N throughput sweeps from
            the same engine.
        input_names / output_names: names attached to graph IO. Used by
            downstream loaders to bind tensors.
        do_constant_folding: standard torch-export hygiene; folds constant
            subgraphs at export time.

    Returns:
        Path to the written ``.onnx`` file.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "export_to_onnx requires PyTorch. Install with `pip install torch`."
        ) from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in (*input_names, *output_names)}

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
            do_constant_folding=do_constant_folding,
        )

    return output_path


def verify_onnx(onnx_path: str | Path) -> OnnxModelInfo:
    """Load an ONNX graph and return its core metadata.

    Used by tests to assert opset / IO / shape invariants after export.
    Lazy-imports the ``onnx`` package.
    """
    try:
        import onnx
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "verify_onnx requires the `onnx` package. Install with `pip install onnx`."
        ) from e

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    opset = max((opset.version for opset in model.opset_import), default=0)

    input_names = tuple(i.name for i in model.graph.input)
    output_names = tuple(o.name for o in model.graph.output)

    def _shape_of(value) -> tuple:
        dims = value.type.tensor_type.shape.dim
        return tuple(d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims)

    input_shapes = {i.name: _shape_of(i) for i in model.graph.input}
    output_shapes = {o.name: _shape_of(o) for o in model.graph.output}

    return OnnxModelInfo(
        path=str(onnx_path),
        opset=opset,
        ir_version=model.ir_version,
        producer=model.producer_name,
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        n_initializers=len(model.graph.initializer),
    )
