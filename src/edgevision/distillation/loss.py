"""Knowledge-distillation losses for detection models.

Three components, each independently usable:

    1. ``LogitKDLoss`` — temperature-scaled KL divergence between teacher
       and student classification logits. Works on any (B, N, C) output.
       This is the workhorse and the only loss needed for the CPU smoke.

    2. ``FeatureKDLoss`` — per-layer MSE between teacher and student
       intermediate features. Requires caller-managed forward hooks to
       collect features; the loss itself just sums MSE across registered
       (teacher_feat, student_feat) pairs.

    3. ``CombinedDetectionKDLoss`` — weighted sum of logit KD + feature KD.
       ``alpha`` controls logit-KD weight; ``beta`` controls feature-KD.

Reference: "DETRDistill: A Universal Knowledge Distillation Framework for
DETR-based Object Detectors" (Chang et al., 2023). We implement the
simpler logit path; the Hungarian-matched query distillation from that paper
can be plugged in at Phase 4 GPU time if it meaningfully moves mAP.

Lazy-imports torch — caller must install PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KDLossConfig:
    """Hyperparameters for the combined KD loss."""

    temperature: float = 4.0
    alpha: float = 0.7     # weight on logit KD term (1 - alpha = weight on CE)
    beta: float = 0.1      # weight on feature KD term (added to alpha sum)

    # Sanity-check invariant: alpha in (0, 1], beta >= 0.
    def __post_init__(self) -> None:
        if not (0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")


# --------------------------------------------------------------------------- logit KD


class LogitKDLoss:
    """Temperature-scaled KL divergence: KL(student || teacher).

    Assumes logits of shape ``(B, N, C)`` where N is the number of queries
    (or tokens) and C is num_classes. The loss is averaged over B and N.

    ``temperature`` controls how "soft" the teacher distribution is. Higher
    temperature → softer targets → more informative gradient signal in early
    training. The loss is scaled by T² to compensate for the 1/T² shrinkage
    in the KL gradient (Hinton et al. 2015).
    """

    def __init__(self, temperature: float = 4.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.T = temperature

    def __call__(
        self,
        student_logits: Any,
        teacher_logits: Any,
    ) -> Any:
        """
        Args:
            student_logits: ``(B, N, C)`` float tensor — raw logits, not softmax.
            teacher_logits: same shape as student_logits.

        Returns:
            scalar loss tensor.
        """
        try:
            import torch.nn.functional as F  # noqa: N812  — universal PyTorch convention
        except ImportError as e:  # pragma: no cover
            raise ImportError("LogitKDLoss requires PyTorch.") from e

        T = self.T  # noqa: N806  — "T" is the standard knowledge-distillation temperature symbol
        # Flatten B×N → (B*N, C) so kl_div sees a 2D input.
        s = student_logits.reshape(-1, student_logits.shape[-1])
        t = teacher_logits.reshape(-1, teacher_logits.shape[-1])

        s_log_p = F.log_softmax(s / T, dim=-1)
        t_p = F.softmax(t / T, dim=-1)
        # kl_div expects (input, target) = (log_probs, probs).
        kl = F.kl_div(s_log_p, t_p, reduction="batchmean")
        return kl * (T * T)


# --------------------------------------------------------------------------- feature KD


class FeatureKDLoss:
    """MSE between teacher and student intermediate feature maps.

    Usage:
        Register forward hooks yourself; pass the collected feature tensors
        as a list of (teacher_feat, student_feat) pairs.

        loss_fn = FeatureKDLoss()
        feature_pairs = [(t_feat1, s_feat1), (t_feat2, s_feat2)]
        loss = loss_fn(feature_pairs)

    If the teacher and student have different widths at a given layer
    (which they will — R50 has 2048 channels, R18 has 512) you need an
    adapter conv layer to project the student features up to the teacher
    width before calling this loss. Adapter training is on the same
    optimizer as the student backbone.
    """

    def __call__(self, feature_pairs: list[tuple[Any, Any]]) -> Any:
        try:
            import torch
            import torch.nn.functional as F  # noqa: N812  — universal PyTorch convention
        except ImportError as e:  # pragma: no cover
            raise ImportError("FeatureKDLoss requires PyTorch.") from e

        if not feature_pairs:
            return torch.tensor(0.0)

        total = torch.tensor(0.0, device=feature_pairs[0][0].device)
        for t_feat, s_feat in feature_pairs:
            # Detach teacher — we don't want gradients flowing into it.
            total = total + F.mse_loss(s_feat, t_feat.detach())
        return total / len(feature_pairs)


# --------------------------------------------------------------------------- combined


class CombinedDetectionKDLoss:
    """Weighted sum: α·KL_logit + β·MSE_feature + (1-α)·CE_task.

    In practice the CE task loss is the detector's own regression + classification
    loss, handled by the model's own forward. This class only handles the
    distillation terms; callers add the task loss externally with weight (1-α).
    """

    def __init__(self, config: KDLossConfig | None = None) -> None:
        self.cfg = config or KDLossConfig()
        self._logit_loss = LogitKDLoss(temperature=self.cfg.temperature)
        self._feat_loss = FeatureKDLoss()

    def __call__(
        self,
        student_logits: Any,
        teacher_logits: Any,
        feature_pairs: list[tuple[Any, Any]] | None = None,
    ) -> Any:
        kl = self._logit_loss(student_logits, teacher_logits)
        feat = self._feat_loss(feature_pairs or [])
        return self.cfg.alpha * kl + self.cfg.beta * feat

    def describe(self) -> str:
        c = self.cfg
        return (
            f"CombinedDetectionKDLoss("
            f"T={c.temperature}, α={c.alpha}, β={c.beta})"
        )
