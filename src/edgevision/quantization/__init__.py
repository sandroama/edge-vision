"""Post-training quantization — TRT INT8 + ONNX QDQ.

Phase 3 modules:
    calib_dataset — stratified COCO calibration sample + ``BatchProvider``.
    trt_int8      — TensorRT IInt8EntropyCalibrator2 + cache management.
    onnx_qdq      — ONNX Runtime static (QDQ) quantization for the CPU path.
"""

from edgevision.quantization.calib_dataset import (
    BatchProvider,
    build_calibration_provider,
    estimate_diversity,
    filter_existing_paths,
    load_image_rgb_pil,
    load_image_rgb_synthetic,
    select_calibration_images,
)
from edgevision.quantization.onnx_qdq import (
    QDQQuantizationConfig,
    QDQQuantizationResult,
    is_qdq_supported,
    quantize_static,
)
from edgevision.quantization.trt_int8 import (
    build_int8_engine,
    is_trt_int8_supported,
    make_int8_calibrator,
)

__all__ = [
    "BatchProvider",
    "QDQQuantizationConfig",
    "QDQQuantizationResult",
    "build_calibration_provider",
    "build_int8_engine",
    "estimate_diversity",
    "filter_existing_paths",
    "is_qdq_supported",
    "is_trt_int8_supported",
    "load_image_rgb_pil",
    "load_image_rgb_synthetic",
    "make_int8_calibrator",
    "quantize_static",
    "select_calibration_images",
]
