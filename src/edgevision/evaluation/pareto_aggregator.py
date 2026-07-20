"""Pareto-frontier aggregation — RQ-E4's signature artefact.

Reads per-config JSON files (produced by ``run_power_sweep.py``,
``run_latency_sweep.py``, and ``run_quant_smoke.py``), computes derived
quantities (watts/frame, retained-mAP%), identifies dominated vs
Pareto-optimal configs, and writes the final ``phase5_pareto.md`` table.

This module is intentionally a *transform* — it has no I/O wiring of its
own beyond reading JSON. The script orchestrates the I/O; this module is
what tests can exercise in isolation.

Pareto dominance definition (lower-is-better for latency and power, higher
for mAP):

    Config A *dominates* config B if:
        A.mAP  >= B.mAP  AND
        A.p95  <= B.p95  AND
        A.watts_per_frame <= B.watts_per_frame
    with strict inequality on at least one axis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ParetoConfig:
    """One row in the Pareto table (one engine configuration)."""

    label: str  # e.g. "trt-fp16", "distilled-int8", "onnx-cpu"
    # "mAP" is the standard COCO metric spelling and a serialized JSON field name.
    mAP_50_95: float  # noqa: N815  — 0 → 1; higher is better
    p95_ms: float  # higher is worse (latency)
    watts_per_frame: float  # higher is worse (power/energy)
    fps: float
    size_mb: float  # model file size
    precision: str  # "fp32" | "fp16" | "int8"
    backend: str  # "trt" | "onnxrt-cpu" | "torch"
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mAP_50_95": round(self.mAP_50_95, 4),
            "p95_ms": round(self.p95_ms, 3),
            "watts_per_frame": round(self.watts_per_frame, 4),
            "fps": round(self.fps, 2),
            "size_mb": round(self.size_mb, 1),
            "precision": self.precision,
            "backend": self.backend,
            "notes": self.notes,
        }


def dominates(a: ParetoConfig, b: ParetoConfig) -> bool:
    """Return True if ``a`` weakly dominates ``b`` on all three axes,
    strictly on at least one.

    Higher mAP = better. Lower p95_ms = better. Lower watts_per_frame = better.
    """
    at_least_as_good = (
        a.mAP_50_95 >= b.mAP_50_95
        and a.p95_ms <= b.p95_ms
        and a.watts_per_frame <= b.watts_per_frame
    )
    strictly_better_on_one = (
        a.mAP_50_95 > b.mAP_50_95 or a.p95_ms < b.p95_ms or a.watts_per_frame < b.watts_per_frame
    )
    return at_least_as_good and strictly_better_on_one


def is_dominated(config: ParetoConfig, others: list[ParetoConfig]) -> bool:
    return any(dominates(o, config) for o in others if o is not config)


def pareto_frontier(configs: list[ParetoConfig]) -> list[ParetoConfig]:
    """Return only the non-dominated configs, sorted by ascending p95."""
    frontier = [c for c in configs if not is_dominated(c, configs)]
    return sorted(frontier, key=lambda c: c.p95_ms)


# --------------------------------------------------------------------------- JSON loading


def load_configs_from_jsons(json_paths: list[str | Path]) -> list[ParetoConfig]:
    """Load ``ParetoConfig`` objects from a list of Phase-5 JSON files.

    Expected JSON schema (produced by ``run_power_sweep.py``)::

        {
          "rows": [
            {
              "config_label": "trt-fp16",
              "fps": 234.5,
              "latency": {"p95_ms": 4.3, ...},
              "power": {"mean_power_w": 95.3, ...},
              "mAP_50_95": 0.530,
              "size_mb": 62.0,
              "precision": "fp16",
              "backend": "trt"
            }, ...
          ]
        }
    """
    configs: list[ParetoConfig] = []
    for p in json_paths:
        p = Path(p)
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for row in data.get("rows", []):
            # A Pareto point is only meaningful when both accuracy and power
            # were measured for this exact executable artifact. Mock-power and
            # latency-only rows remain in the sweep JSON for diagnostics but
            # must not become fake frontier points. Require the provenance flags
            # to be explicitly true: legacy rows written before `run_power_sweep`
            # emitted them carry no flags at all, and must fail closed rather
            # than inherit trust they never earned.
            if row.get("accuracy_measured") is not True or row.get("power_measured") is not True:
                continue
            if row.get("mAP_50_95") is None:
                continue
            fps = float(row.get("fps", 0.0))
            power_w = float(row.get("power", {}).get("mean_power_w", 0.0))
            p95_ms = float(row.get("latency", {}).get("p95_ms", 0.0))
            # Fail closed on truncated/legacy rows: a missing fps, latency, or
            # power block would otherwise load as 0.0 — a fake point with 0 ms
            # p95 and 0 W/frame that dominates every real config.
            if fps <= 0 or p95_ms <= 0 or power_w <= 0:
                continue
            watts_per_frame = power_w / fps
            configs.append(
                ParetoConfig(
                    label=str(row.get("config_label", "unknown")),
                    mAP_50_95=float(row.get("mAP_50_95", 0.0)),
                    p95_ms=p95_ms,
                    watts_per_frame=watts_per_frame,
                    fps=fps,
                    size_mb=float(row.get("size_mb", 0.0)),
                    precision=str(row.get("precision", "unknown")),
                    backend=str(row.get("backend", "unknown")),
                    notes=str(row.get("notes", "")),
                )
            )
    return configs


def load_configs_from_dicts(rows: list[dict]) -> list[ParetoConfig]:
    """Convenience: build configs directly from a list of dicts."""
    configs = []
    for row in rows:
        fps = float(row.get("fps", 0.0))
        power_w = (
            float(row.get("watts_per_frame", 0.0)) * fps
            if fps > 0
            else float(row.get("power_mean_w", 0.0))
        )
        wpf = float(row.get("watts_per_frame", power_w / fps if fps > 0 else 0.0))
        configs.append(
            ParetoConfig(
                label=str(row.get("label", "unknown")),
                mAP_50_95=float(row.get("mAP_50_95", 0.0)),
                p95_ms=float(row.get("p95_ms", 0.0)),
                watts_per_frame=wpf,
                fps=fps,
                size_mb=float(row.get("size_mb", 0.0)),
                precision=str(row.get("precision", "unknown")),
                backend=str(row.get("backend", "unknown")),
                notes=str(row.get("notes", "")),
            )
        )
    return configs


# --------------------------------------------------------------------------- reporting


def summary_table(configs: list[ParetoConfig]) -> str:
    """Markdown table of all configs with Pareto-frontier markers."""
    dominated = {c.label for c in configs if is_dominated(c, configs)}

    header = (
        "| Config | mAP@[0.5:0.95] | p95 (ms) | Watts/frame | FPS | Size (MB) | Backend | Pareto? |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for c in sorted(configs, key=lambda x: x.p95_ms):
        on_pareto = "✅" if c.label not in dominated else "—"
        rows.append(
            f"| `{c.label}` "
            f"| {c.mAP_50_95:.3f} "
            f"| {c.p95_ms:.2f} "
            f"| {c.watts_per_frame:.4f} "
            f"| {c.fps:.1f} "
            f"| {c.size_mb:.0f} "
            f"| {c.backend} "
            f"| {on_pareto} |"
        )
    return "\n".join(rows)


def write_report(
    configs: list[ParetoConfig],
    out_dir: str | Path = "docs/results",
) -> Path:
    """Write ``phase5_pareto.md`` + ``phase5_pareto.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Phase 5 — accuracy × latency × power Pareto (RQ-E4)",
        "",
        "Generated by `evaluation/pareto_aggregator.py`. GPU rows slot in as the "
        "``run_power_sweep.py`` JSONs land.",
        "",
        summary_table(configs),
    ]
    if not configs:
        md_lines.append(
            "\n> **No rows yet — this is the expected pre-GPU state, not a broken report.**\n"
            "> A config only earns a Pareto row once it has *both* a measured mAP and a\n"
            "> measured NVML power draw for the same artifact. Mock and CPU mock-power\n"
            "> rows are recorded in `phase5_power.json` for diagnostics and excluded here\n"
            "> by design. See [`NEXT_STEPS.md`](../../NEXT_STEPS.md) for the GPU runbook."
        )
    md_path = out_dir / "phase5_pareto.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    json_path = out_dir / "phase5_pareto.json"
    payload = {
        "configs": [c.as_dict() for c in configs],
        "frontier": [c.label for c in pareto_frontier(configs)],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    return md_path
