"""Tests for ``edgevision.evaluation.pareto_aggregator``.

CPU-only — no ML deps. Exercises the dominance logic, frontier extraction,
JSON round-trip, and table rendering with synthetic ParetoConfig objects.
"""

from __future__ import annotations

import json

from edgevision.evaluation.pareto_aggregator import (
    ParetoConfig,
    dominates,
    is_dominated,
    load_configs_from_dicts,
    pareto_frontier,
    summary_table,
    write_report,
)


def _cfg(label: str, mAP: float, p95: float, wpf: float) -> ParetoConfig:  # noqa: N803  — mirrors ParetoConfig.mAP_50_95
    return ParetoConfig(
        label=label,
        mAP_50_95=mAP,
        p95_ms=p95,
        watts_per_frame=wpf,
        fps=1000.0 / p95,
        size_mb=50.0,
        precision="fp16",
        backend="trt",
    )


# --------------------------------------------------------------------------- dominance


def test_dominates_strictly_better_on_one():
    a = _cfg("a", mAP=0.5, p95=5.0, wpf=0.1)
    b = _cfg("b", mAP=0.5, p95=6.0, wpf=0.1)
    assert dominates(a, b)  # a has lower latency
    assert not dominates(b, a)


def test_dominates_requires_at_least_equal_on_all_axes():
    a = _cfg("a", mAP=0.55, p95=5.0, wpf=0.1)
    b = _cfg("b", mAP=0.50, p95=4.5, wpf=0.09)
    # a is better on mAP; b is better on p95 and wpf — neither dominates.
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominates_not_self():
    a = _cfg("a", mAP=0.5, p95=5.0, wpf=0.1)
    assert not dominates(a, a)


def test_is_dominated_in_group():
    good = _cfg("good", mAP=0.5, p95=4.0, wpf=0.08)
    bad = _cfg("bad", mAP=0.4, p95=7.0, wpf=0.15)  # worse on all axes
    assert is_dominated(bad, [good, bad])
    assert not is_dominated(good, [good, bad])


# --------------------------------------------------------------------------- pareto_frontier


def test_pareto_frontier_removes_dominated():
    # b has the same mAP as a but strictly better p95 — so b dominates a.
    # c is dominated by both a and b on all axes.
    a = _cfg("a", mAP=0.5, p95=5.0, wpf=0.1)  # dominated by b
    b = _cfg("b", mAP=0.5, p95=4.0, wpf=0.1)  # Pareto-optimal
    c = _cfg("c", mAP=0.3, p95=9.0, wpf=0.2)  # dominated by both
    frontier = pareto_frontier([a, b, c])
    labels = {cfg.label for cfg in frontier}
    assert "c" not in labels
    assert "a" not in labels  # dominated by b
    assert "b" in labels


def test_pareto_frontier_sorted_by_p95():
    a = _cfg("a", mAP=0.5, p95=8.0, wpf=0.05)
    b = _cfg("b", mAP=0.4, p95=4.0, wpf=0.12)
    frontier = pareto_frontier([a, b])
    assert [c.p95_ms for c in frontier] == sorted(c.p95_ms for c in frontier)


def test_pareto_frontier_single_config():
    a = _cfg("only", mAP=0.5, p95=5.0, wpf=0.1)
    frontier = pareto_frontier([a])
    assert len(frontier) == 1


def test_pareto_frontier_empty():
    assert pareto_frontier([]) == []


# --------------------------------------------------------------------------- load_configs_from_dicts


def test_load_from_dicts_round_trip():
    rows = [
        {
            "label": "fp16",
            "mAP_50_95": 0.53,
            "p95_ms": 4.5,
            "watts_per_frame": 0.09,
            "fps": 222.0,
            "size_mb": 60.0,
            "precision": "fp16",
            "backend": "trt",
        },
        {
            "label": "int8",
            "mAP_50_95": 0.51,
            "p95_ms": 3.2,
            "watts_per_frame": 0.07,
            "fps": 312.0,
            "size_mb": 30.0,
            "precision": "int8",
            "backend": "trt",
        },
    ]
    configs = load_configs_from_dicts(rows)
    assert len(configs) == 2
    assert configs[0].label == "fp16"
    assert configs[1].precision == "int8"


def test_json_loader_skips_unmeasured_accuracy_or_power(tmp_path):
    from edgevision.evaluation.pareto_aggregator import load_configs_from_jsons

    path = tmp_path / "power.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "config_label": "mock",
                        "mAP_50_95": None,
                        "accuracy_measured": False,
                        "power_measured": False,
                        "fps": 10,
                    },
                    {
                        "config_label": "real",
                        "mAP_50_95": 0.5,
                        "accuracy_measured": True,
                        "power_measured": True,
                        "fps": 100,
                        "latency": {"p95_ms": 10},
                        "power": {"mean_power_w": 50},
                    },
                ]
            }
        )
    )
    configs = load_configs_from_jsons([path])
    assert [config.label for config in configs] == ["real"]


def test_json_loader_skips_truncated_rows_that_would_fake_dominate(tmp_path):
    """A row missing fps/latency/power must not load as a 0 ms / 0 W point.

    Such a point would Pareto-dominate every real config and flip genuine
    frontier rows to 'dominated' in the published table.
    """
    from edgevision.evaluation.pareto_aggregator import load_configs_from_jsons

    real = {
        "config_label": "real",
        "mAP_50_95": 0.5,
        "accuracy_measured": True,
        "power_measured": True,
        "fps": 100,
        "latency": {"p95_ms": 10},
        "power": {"mean_power_w": 50},
    }
    truncated = {
        "config_label": "truncated",
        "mAP_50_95": 0.6,
        "accuracy_measured": True,
        "power_measured": True,
        # fps / latency / power blocks missing (interrupted write, old schema)
    }
    path = tmp_path / "power.json"
    path.write_text(json.dumps({"rows": [real, truncated]}))

    configs = load_configs_from_jsons([path])
    assert [config.label for config in configs] == ["real"]
    frontier = pareto_frontier(configs)
    assert [config.label for config in frontier] == ["real"]


def test_json_loader_skips_legacy_rows_missing_provenance_flags(tmp_path):
    """Rows with no accuracy_measured/power_measured keys must fail closed.

    `run_power_sweep.py` always emits both flags. A row without them predates
    that contract (e.g. the mock rows in the checked-in phase5_power.json,
    which reuse RT-DETR's *published* mAP), so it cannot be trusted as a
    measured Pareto point just because its numbers happen to be positive.
    """
    from edgevision.evaluation.pareto_aggregator import load_configs_from_jsons

    legacy = {
        "config_label": "mock-a",
        "mAP_50_95": 0.531,  # published RT-DETR number, not measured here
        "fps": 345.14,
        "latency": {"p95_ms": 3.046},
        "power": {"mean_power_w": 200.4},
        "backend": "mock",
        # no accuracy_measured / power_measured keys at all
    }
    path = tmp_path / "power.json"
    path.write_text(json.dumps({"rows": [legacy]}))

    assert load_configs_from_jsons([path]) == []


# --------------------------------------------------------------------------- summary_table + write_report


def test_summary_table_contains_all_labels():
    configs = [
        _cfg("fp32", 0.53, 8.0, 0.12),
        _cfg("fp16", 0.53, 4.5, 0.07),
        _cfg("int8", 0.50, 3.0, 0.05),
    ]
    table = summary_table(configs)
    assert "fp32" in table
    assert "fp16" in table
    assert "int8" in table
    assert "✅" in table  # at least one Pareto-optimal config


def test_summary_table_marks_dominated():
    good = _cfg("good", mAP=0.5, p95=3.0, wpf=0.05)
    bad = _cfg("bad", mAP=0.3, p95=9.0, wpf=0.2)
    table = summary_table([good, bad])
    # "bad" is dominated — should have "—" not "✅".
    lines = [line for line in table.splitlines() if "bad" in line]
    assert lines and "—" in lines[0]


def test_write_report_creates_md_and_json(tmp_path):
    configs = [
        _cfg("fp16", 0.53, 4.5, 0.09),
        _cfg("int8", 0.51, 3.0, 0.06),
    ]
    md_path = write_report(configs, out_dir=tmp_path)
    assert md_path.exists()
    assert (tmp_path / "phase5_pareto.json").exists()

    payload = json.loads((tmp_path / "phase5_pareto.json").read_text())
    assert "configs" in payload
    assert "frontier" in payload
    assert len(payload["configs"]) == 2


def test_pareto_config_as_dict_is_json_serialisable():
    cfg = _cfg("test", 0.5, 5.0, 0.1)
    json.loads(json.dumps(cfg.as_dict()))
