"""Compilation pipeline — torch -> ONNX -> TensorRT / ONNX Runtime.

Phase 2:
    onnx_export  — torch.onnx export, opset 17, dynamic batch.
    trt_build    — TensorRT engine builder (FP32, FP16). INT8 in Phase 3.
    onnxrt_cpu   — ONNX Runtime CPU executor (CPUExecutionProvider).
Phase 3:
    trt_build extends with INT8 + entropy calibrator (defined in
    edgevision.quantization.trt_int8 to keep the calibration logic close
    to the dataset that drives it).
Phase 7 (stretch):
    coreml_export — optional Apple CoreML conversion.
"""

from edgevision.compile.onnx_export import (
    DEFAULT_OPSET,
    OnnxModelInfo,
    export_to_onnx,
    verify_onnx,
)
from edgevision.compile.onnxrt_cpu import (
    OnnxRuntimeCPUExecutor,
    OnnxRuntimeOutputs,
)
from edgevision.compile.trt_build import (
    TrtBuildConfig,
    TrtBuildResult,
    build_engine,
    trt_available,
)
from edgevision.compile.trt_runtime import TensorRTExecutor

__all__ = [
    "DEFAULT_OPSET",
    "OnnxModelInfo",
    "OnnxRuntimeCPUExecutor",
    "OnnxRuntimeOutputs",
    "TrtBuildConfig",
    "TrtBuildResult",
    "TensorRTExecutor",
    "build_engine",
    "export_to_onnx",
    "trt_available",
    "verify_onnx",
]
