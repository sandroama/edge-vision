"""Teacher/student knowledge distillation for detection models.

Phase 4 modules:
    loss           — LogitKDLoss, FeatureKDLoss, CombinedDetectionKDLoss.
    student_train  — CPU smoke (TinyDetector) + full GPU (RT-DETR-R50 → R18).
"""

from edgevision.distillation.loss import (
    CombinedDetectionKDLoss,
    FeatureKDLoss,
    KDLossConfig,
    LogitKDLoss,
)
from edgevision.distillation.student_train import (
    DistillationConfig,
    DistillationResult,
    EpochLoss,
    make_synthetic_batch,
    run_rtdetr_distillation,
    run_tiny_distillation_smoke,
)

__all__ = [
    "CombinedDetectionKDLoss",
    "DistillationConfig",
    "DistillationResult",
    "EpochLoss",
    "FeatureKDLoss",
    "KDLossConfig",
    "LogitKDLoss",
    "make_synthetic_batch",
    "run_rtdetr_distillation",
    "run_tiny_distillation_smoke",
]
