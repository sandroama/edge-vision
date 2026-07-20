"""Quantization evaluation — answer the RQ-E1 question honestly.

The interesting finding for INT8 PTQ is rarely "mAP dropped X points
overall." It's "*these* classes broke first, by *that* much, and here's
why it makes sense (small objects, low-frequency classes, etc)." This
module turns two ``CocoMetrics`` into that story.

Inputs:
    * ``reference`` — the FP32 (or FP16) baseline metrics
    * ``candidate`` — the quantized metrics

Outputs:
    * Overall mAP delta + retained pct
    * Per-class drops sorted by magnitude (worst classes first)
    * "Top-N broken" classes for the writeup
    * A summary table renderer

Both metrics objects must share the same backend — comparing pycocotools
mAP to a simple-backend F1 is a category error.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from edgevision.evaluation.coco_eval import CocoMetrics

# --------------------------------------------------------------------------- per-class delta


@dataclass(frozen=True)
class PerClassDelta:
    """One class's drop between reference and candidate."""

    label: str
    reference: float
    candidate: float

    @property
    def delta(self) -> float:
        """Negative number = candidate is worse."""
        return self.candidate - self.reference

    @property
    def retained_pct(self) -> float:
        """100 = no drop, 0 = total loss. Returns nan if reference is 0."""
        if self.reference == 0:
            return float("nan")
        return 100.0 * (self.candidate / self.reference)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "reference": round(self.reference, 4),
            "candidate": round(self.candidate, 4),
            "delta": round(self.delta, 4),
            "retained_pct": round(self.retained_pct, 2)
            if self.retained_pct == self.retained_pct  # nan check
            else None,
        }


# --------------------------------------------------------------------------- top-level delta


@dataclass
class QuantizationDelta:
    """Reference vs candidate metrics — the core RQ-E1 deliverable."""

    backend: str
    reference_label: str
    candidate_label: str

    # "mAP" is the standard COCO metric spelling, kept intentionally in field /
    # property names that also serialize into docs/results/*.json.
    overall_mAP_50_95_ref: float  # noqa: N815
    overall_mAP_50_95_cand: float  # noqa: N815

    overall_mAP_50_ref: float  # noqa: N815
    overall_mAP_50_cand: float  # noqa: N815

    per_class: list[PerClassDelta] = field(default_factory=list)

    # Convenience accessors -----------------------------------------------

    @property
    def mAP_50_95_drop(self) -> float:  # noqa: N802  — "mAP" is the COCO metric spelling
        return self.overall_mAP_50_95_cand - self.overall_mAP_50_95_ref

    @property
    def mAP_50_drop(self) -> float:  # noqa: N802  — "mAP" is the COCO metric spelling
        return self.overall_mAP_50_cand - self.overall_mAP_50_ref

    @property
    def retained_mAP_50_95_pct(self) -> float:  # noqa: N802  — "mAP" is the COCO metric spelling
        if self.overall_mAP_50_95_ref == 0:
            return float("nan")
        return 100.0 * (self.overall_mAP_50_95_cand / self.overall_mAP_50_95_ref)

    def worst_classes(self, n: int = 5) -> list[PerClassDelta]:
        """Top-``n`` largest-drop classes, ascending delta (most-negative first)."""
        return sorted(self.per_class, key=lambda d: d.delta)[:n]

    def best_classes(self, n: int = 5) -> list[PerClassDelta]:
        """Top-``n`` classes that *survived* INT8 best (least-negative delta)."""
        return sorted(self.per_class, key=lambda d: -d.delta)[:n]

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "reference_label": self.reference_label,
            "candidate_label": self.candidate_label,
            "overall_mAP_50_95_ref": round(self.overall_mAP_50_95_ref, 4),
            "overall_mAP_50_95_cand": round(self.overall_mAP_50_95_cand, 4),
            "overall_mAP_50_ref": round(self.overall_mAP_50_ref, 4),
            "overall_mAP_50_cand": round(self.overall_mAP_50_cand, 4),
            "mAP_50_95_drop": round(self.mAP_50_95_drop, 4),
            "mAP_50_drop": round(self.mAP_50_drop, 4),
            "retained_mAP_50_95_pct": round(self.retained_mAP_50_95_pct, 2)
            if self.retained_mAP_50_95_pct == self.retained_mAP_50_95_pct
            else None,
            "per_class": [d.as_dict() for d in self.per_class],
            "n_classes": len(self.per_class),
        }


# --------------------------------------------------------------------------- public API


def compare_metrics(
    reference: CocoMetrics,
    candidate: CocoMetrics,
    *,
    reference_label: str = "fp32",
    candidate_label: str = "int8",
) -> QuantizationDelta:
    """Diff two ``CocoMetrics`` objects produced by the same eval backend.

    Args:
        reference: the baseline (typically FP32 or FP16).
        candidate: the quantized model's metrics.
        reference_label / candidate_label: arbitrary tags carried through
            into the report (e.g. ``"fp16"``, ``"int8-trt"``, ``"int8-qdq"``).

    Raises:
        ValueError: if the backends differ — comparing pycocotools mAP
            against simple-backend F1 is a category error and would mislead
            anyone reading the report.
    """
    if reference.backend != candidate.backend:
        raise ValueError(
            f"Backend mismatch: reference={reference.backend!r} vs "
            f"candidate={candidate.backend!r}. Both must use the same "
            "evaluation backend to produce a meaningful diff."
        )

    # Build per-class deltas from the union of class names. Missing classes
    # default to 0.0 — that's "no signal", not "no drop".
    all_labels = sorted(
        set(reference.mAP_per_class.keys()) | set(candidate.mAP_per_class.keys())
    )
    per_class = [
        PerClassDelta(
            label=label,
            reference=reference.mAP_per_class.get(label, 0.0),
            candidate=candidate.mAP_per_class.get(label, 0.0),
        )
        for label in all_labels
    ]

    # For the simple backend the mAP fields are always zero — surface F1 in
    # the mAP_50 slot so the report still shows a meaningful overall diff.
    if reference.backend == "simple":
        ref_50 = reference.f1 or 0.0
        cand_50 = candidate.f1 or 0.0
        ref_5095 = 0.0
        cand_5095 = 0.0
    else:
        ref_50 = reference.mAP_50
        cand_50 = candidate.mAP_50
        ref_5095 = reference.mAP_50_95
        cand_5095 = candidate.mAP_50_95

    return QuantizationDelta(
        backend=reference.backend,
        reference_label=reference_label,
        candidate_label=candidate_label,
        overall_mAP_50_95_ref=ref_5095,
        overall_mAP_50_95_cand=cand_5095,
        overall_mAP_50_ref=ref_50,
        overall_mAP_50_cand=cand_50,
        per_class=per_class,
    )


# --------------------------------------------------------------------------- text rendering


def summary_table(delta: QuantizationDelta, *, top_n: int = 5) -> str:
    """Multi-line text summary fit for ``docs/results/phase3_quantization.md``."""
    backend = delta.backend
    if backend == "simple":
        headline = (
            f"  Overall F1 ({delta.reference_label} vs {delta.candidate_label}) : "
            f"{delta.overall_mAP_50_ref:.3f} -> {delta.overall_mAP_50_cand:.3f}  "
            f"(delta={delta.mAP_50_drop:+.3f})"
        )
    else:
        retained = delta.retained_mAP_50_95_pct
        retained_str = f"{retained:.1f}%" if retained == retained else "n/a"
        headline = (
            f"  mAP@[0.5:0.95] ({delta.reference_label} vs {delta.candidate_label}) : "
            f"{delta.overall_mAP_50_95_ref:.3f} -> {delta.overall_mAP_50_95_cand:.3f}  "
            f"(delta={delta.mAP_50_95_drop:+.3f}, retained={retained_str})"
        )

    worst = delta.worst_classes(top_n)
    best = delta.best_classes(top_n)

    lines: list[str] = [
        f"  Backend                : {backend}",
        f"  Reference -> Candidate : {delta.reference_label} -> {delta.candidate_label}",
        f"  Classes diffed         : {len(delta.per_class)}",
        headline,
        "",
        f"  Worst {len(worst)} classes (largest drop):",
    ]
    lines.extend(_format_class_rows(worst))
    lines.append("")
    lines.append(f"  Best {len(best)} classes (smallest drop):")
    lines.extend(_format_class_rows(best))
    return "\n".join(lines)


def _format_class_rows(items: Iterable[PerClassDelta]) -> list[str]:
    rows: list[str] = []
    for d in items:
        retained = (
            f"{d.retained_pct:5.1f}%" if d.retained_pct == d.retained_pct else "  n/a"
        )
        rows.append(
            f"    - {d.label:<24}  "
            f"ref={d.reference:.3f}  cand={d.candidate:.3f}  "
            f"delta={d.delta:+.3f}  retained={retained}"
        )
    return rows
