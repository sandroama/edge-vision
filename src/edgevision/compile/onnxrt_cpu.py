"""ONNX Runtime CPU executor — the "no GPU" deployment target.

Wraps an ``onnxruntime.InferenceSession`` configured for the
``CPUExecutionProvider``. The executor's job is to take a numpy array, push
it through the graph, and surface the raw output tensors. Decoding outputs
into ``ImageDetections`` happens *outside* this class — RT-DETR's output
shape (logits + pred_boxes) is one decoder, MobileSAM's is another, and
keeping that out of here means a single executor serves every model.

Lazy-imports onnxruntime so the module is safe to import without it. ORT
is in the project's base deps, but the lazy-import keeps the failure mode
explicit if a user installs a stripped-down environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OnnxRuntimeOutputs:
    """Raw ORT outputs tagged with the binding names from the ONNX graph."""

    outputs: dict[str, np.ndarray]

    def get(self, name: str) -> np.ndarray:
        if name not in self.outputs:
            raise KeyError(
                f"Output {name!r} not produced. Available: {list(self.outputs.keys())}"
            )
        return self.outputs[name]


class OnnxRuntimeCPUExecutor:
    """Bind one ONNX file, run forward passes on it.

    Args:
        onnx_path: path to the exported model.
        num_threads: intra-op thread count. ``None`` uses ORT defaults.
            On a 9950X-class CPU, leaving this unset and letting ORT pick
            beats most manual choices.
        input_name: optional override; auto-detected if not set.
        output_names: optional override; auto-detected if not set.
        graph_optimization: ORT optimization level. ``"basic"`` is a safe
            default. Use ``"all"`` for the bench-most-config path.

    Usage::

        ex = OnnxRuntimeCPUExecutor("checkpoints/rtdetr_r50.onnx")
        x = np.zeros((1, 3, 640, 640), dtype=np.float32)
        outs = ex.run(x)
        logits = outs.get("logits")
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        num_threads: int | None = None,
        input_name: str | None = None,
        output_names: list[str] | None = None,
        graph_optimization: str = "basic",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OnnxRuntimeCPUExecutor requires onnxruntime. "
                "Install with: pip install -e '.[dev]'"
            ) from e

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = {
            "off": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[graph_optimization]
        if num_threads is not None:
            sess_options.intra_op_num_threads = num_threads

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        sess_inputs = self._session.get_inputs()
        if input_name is None:
            if len(sess_inputs) != 1:
                raise ValueError(
                    f"ONNX has {len(sess_inputs)} inputs; please pass input_name explicitly."
                )
            input_name = sess_inputs[0].name
        self.input_name = input_name
        self.input_shape = tuple(sess_inputs[0].shape)
        self.input_dtype = sess_inputs[0].type

        if output_names is None:
            output_names = [o.name for o in self._session.get_outputs()]
        self.output_names = list(output_names)

    # ------------------------------------------------------------------ inference

    def run(self, x: np.ndarray) -> OnnxRuntimeOutputs:
        """Run one forward pass. ``x`` must match the model's input shape & dtype."""
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        outs = self._session.run(self.output_names, {self.input_name: x})
        return OnnxRuntimeOutputs(outputs=dict(zip(self.output_names, outs, strict=True)))

    def make_callable(self, x: np.ndarray):
        """Return a no-arg callable suitable for the latency harness."""
        return lambda: self._session.run(self.output_names, {self.input_name: x})

    # ------------------------------------------------------------------ inspect

    def describe(self) -> dict[str, Any]:
        return {
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "output_names": self.output_names,
            "providers": self._session.get_providers(),
        }
