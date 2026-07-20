"""pytest configuration — registers custom markers.

Markers (defined in pyproject.toml):
    slow — long-running tests (deselect with `-m "not slow"`)
    gpu  — requires a CUDA-capable GPU
    trt  — requires TensorRT
"""

from __future__ import annotations
