"""Tiny detector-shaped model for compile-pipeline tests.

This module exists so the export → ONNX → ORT-CPU → TensorRT pipeline can be
exercised end-to-end *in CI*, without pulling 200 MB of HF weights and
spending 30 seconds loading them. The shape mimics RT-DETR's API enough for
the export path to be representative:

    * one (1, 3, 640, 640) float32 input
    * two outputs — ``logits`` (1, N, K) and ``pred_boxes`` (1, N, 4)
    * fully convolutional + global pool, so the export emits a real graph
      rather than a single op.

It is **not** a useful detector. Don't measure mAP on it. Use it to verify
the compile pipeline doesn't drop outputs, change opset, or add unsupported
ops.

Lazy-imports torch so the module stays cheap to import in non-torch contexts
(e.g. when tests are running the latency harness on CPU only).
"""

from __future__ import annotations


def make_tiny_model(num_classes: int = 4, num_queries: int = 8):
    """Construct a small detector-shaped torch.nn.Module.

    Returns a model whose ``forward(x)`` returns
    ``{"logits": (B, num_queries, num_classes), "pred_boxes": (B, num_queries, 4)}``
    so the same downstream code that handles RT-DETR outputs handles this too.
    """
    try:
        import torch
        from torch import nn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "make_tiny_model requires PyTorch. Install with `pip install torch`."
        ) from e

    class TinyDetector(nn.Module):
        def __init__(self, num_classes: int, num_queries: int) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.num_queries = num_queries
            # Two-stage tiny "backbone": stride-4 conv → stride-4 conv.
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=4, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.cls_head = nn.Linear(32, num_queries * num_classes)
            self.box_head = nn.Linear(32, num_queries * 4)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            features = self.backbone(x)
            logits = self.cls_head(features).reshape(-1, self.num_queries, self.num_classes)
            boxes = torch.sigmoid(
                self.box_head(features).reshape(-1, self.num_queries, 4)
            )
            return {"logits": logits, "pred_boxes": boxes}

    return TinyDetector(num_classes=num_classes, num_queries=num_queries)


def make_tiny_input(batch: int = 1, height: int = 640, width: int = 640):
    """A torch.float32 (B, 3, H, W) tensor of zeros, suitable as dummy input."""
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "make_tiny_input requires PyTorch. Install with `pip install torch`."
        ) from e
    return torch.zeros((batch, 3, height, width), dtype=torch.float32)
