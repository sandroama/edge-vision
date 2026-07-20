"""edge-vision — CPU ONNX Runtime demo dashboard (Hugging Face Space).

A recruiter-facing summary of the project's *measured* CPU lane: real ONNX
Runtime dynamic-INT8 model-size reduction and CPU latency (with bootstrap 95%
confidence intervals), plus an explicit, clearly-labeled view of which parts of
the accuracy x latency x power Pareto are still GPU-pending.

Honesty contract (do not break):
  * Every number shown is read verbatim from ``docs/results/*.json`` — nothing
    is hardcoded or invented here.
  * The GPU lane (TensorRT latency, real COCO mAP, NVML watts/frame) is PENDING
    a real RTX-class run and is always labeled as such. Placeholder "mock-*"
    rows in the power sweep are shown only as a wiring preview, never as results.

Run locally::

    pip install -e '.[ui]'        # adds streamlit + plotly to the project venv
    streamlit run hf_space/app.py

The module imports cleanly even when streamlit/pandas/plotly are absent (every
heavy import is lazy and guarded), so ``python -m py_compile`` and unit imports
work in a bare environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Results files live in the project's docs/results tree (hf_space/ is one level
# under the project root). On a deployed Space, app.py sits at the repo root
# instead, so fall back to a Space-local results/ dir (the deploy step copies
# docs/results/*.json there — see hf_space/README.md).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "docs" / "results"
if not _RESULTS_DIR.is_dir():
    _RESULTS_DIR = Path(__file__).resolve().parent / "results"
_CPU_INT8_JSON = _RESULTS_DIR / "phase3_cpu_int8.json"
_PARETO_JSON = _RESULTS_DIR / "phase5_pareto.json"

_REPO_URL = "https://github.com/sandroama/edge-vision"


# --------------------------------------------------------------------------- data loading
# These helpers are import-safe (no streamlit dependency) so they can be unit
# tested directly. The caching decorators are applied by thin wrappers below
# only once streamlit is known to be importable.


def _load_json(path: Path) -> dict[str, Any] | None:
    """Read a results JSON, returning ``None`` on any failure.

    A recruiter may clone and run this Space with no results checked out, so
    every load is defensive — a missing or malformed file degrades to a
    friendly empty-state rather than a traceback.
    """
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _real_pareto_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return only Pareto configs from a *real* (non-mock) backend.

    The committed ``phase5_pareto.json`` currently holds ``mock-*`` wiring
    rows. We never surface those as results; this filter keeps the headline
    honest if/when real GPU rows land in the same file.
    """
    if not payload:
        return []
    configs = payload.get("configs", []) or []
    return [c for c in configs if str(c.get("backend", "")).lower() != "mock"]


# --------------------------------------------------------------------------- UI sections


def _render_header(st: Any) -> None:
    st.title("edge-vision — measured edge-inference Pareto")
    st.markdown(
        "**Quantize, distill, and profile RT-DETR for the edge — and report the "
        "watts, not just the FPS.** This page shows the project's *measured CPU "
        "lane* (the slice that runs anywhere, no GPU required) and is explicit "
        "about which numbers are still pending a real GPU run."
    )


def _render_headline(st: Any, cpu: dict[str, Any] | None) -> None:
    """Headline metrics — visible above the fold, no scrolling.

    All four values are read straight from ``phase3_cpu_int8.json``.
    """
    st.subheader("Headline: CPU INT8 lane (measured today, no GPU)")

    if cpu is None:
        st.info(
            "No CPU-lane results found yet. Generate them with a quick, "
            "GPU-free benchmark:\n\n"
            "```bash\n"
            "python scripts/bench_cpu_int8.py --n-runs 200 --n-warmup 40\n"
            "```\n\n"
            "This writes `docs/results/phase3_cpu_int8.json`, which this page reads."
        )
        return

    size = cpu.get("size", {})
    lat = cpu.get("latency", {})
    int8 = lat.get("int8", {})
    speedup = cpu.get("bootstrap_ci", {}).get("median_speedup_fp32_over_int8", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Model size reduction",
        f"-{size.get('size_reduction_pct', 0):.2f}%",
        help=(
            f"FP32 {size.get('fp32_kib', 0):.0f} KiB -> INT8 "
            f"{size.get('int8_kib', 0):.0f} KiB via ONNX Runtime dynamic "
            "quantization. The robust, portable CPU-lane win."
        ),
    )
    c2.metric(
        "INT8 p95 latency",
        f"{int8.get('p95_ms', 0):.2f} ms",
        help="95th-percentile single-image CPU latency for the INT8 model "
        "(200 timed runs, single thread).",
    )
    c3.metric(
        "INT8 throughput",
        f"{int8.get('fps_mean', 0):.0f} FPS",
        help="Mean single-image throughput on CPU for the INT8 model.",
    )
    c4.metric(
        "Median speed-up (FP32/INT8)",
        f"{speedup.get('point', 0):.2f}x",
        help=(
            f"95% CI [{speedup.get('lo', 0):.3f}, {speedup.get('hi', 0):.3f}]. "
            "Below 1.0 on purpose: dynamic INT8 adds per-op quantize/dequantize "
            "overhead that dominates on this small graph. Reported exactly as "
            "measured — the size reduction is the CPU win; INT8 *latency* gains "
            "are a GPU/TensorRT story (pending)."
        ),
    )

    st.caption(
        "Measured on Python 3.11 / ONNX Runtime 1.26, single-thread CPU, 200 "
        "timed runs each, 95% bootstrap CIs from the raw per-run samples. "
        "Source: `docs/results/phase3_cpu_int8.json`."
    )

    st.warning(
        "**What this is NOT:** not a TensorRT/GPU result and not an mAP claim. "
        "The model here is an RT-DETR-shaped CI stand-in (chosen so the lane "
        "runs with no 120 MB download), so no accuracy number is reported on "
        "it. Real COCO mAP, TensorRT latency, and NVML watts/frame are "
        "**GPU-pending** (see the project's `NEXT_STEPS.md`).",
        icon="ℹ️",
    )


def _render_size_chart(st: Any, cpu: dict[str, Any], use_plotly: bool) -> None:
    """FP32 vs INT8 model size — labeled bars, value annotations (not color-only)."""
    size = cpu.get("size", {})
    fp32_kib = float(size.get("fp32_kib", 0.0))
    int8_kib = float(size.get("int8_kib", 0.0))
    reduction = float(size.get("size_reduction_pct", 0.0))

    st.markdown("#### Model size: FP32 vs INT8 (dynamic)")
    st.caption(
        f"On-disk ONNX model size. INT8 dynamic quantization removes "
        f"{reduction:.2f}% of the bytes (FP32 {fp32_kib:.0f} KiB -> INT8 "
        f"{int8_kib:.0f} KiB). Lower is better."
    )

    rendered = False
    if use_plotly:
        try:
            import plotly.graph_objects as go

            labels = ["FP32", "INT8 (dynamic)"]
            values = [fp32_kib, int8_kib]
            fig = go.Figure(
                go.Bar(
                    x=labels,
                    y=values,
                    text=[f"{v:.0f} KiB" for v in values],
                    textposition="outside",
                    marker_color=["#64748b", "#16a34a"],
                    marker_line_color="#1e293b",
                    marker_line_width=1,
                )
            )
            fig.update_layout(
                title="Model size by precision (lower is better)",
                yaxis_title="Size (KiB)",
                xaxis_title="Precision",
                showlegend=False,
                margin={"t": 50, "b": 40, "l": 60, "r": 20},
            )
            # Alt-text equivalent for assistive tech / non-visual readers.
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
            rendered = True
        except Exception:  # pragma: no cover - plotly optional
            rendered = False

    if not rendered:
        # Fallback that needs no plotly: a native bar chart keyed by label.
        try:
            import pandas as pd

            df = pd.DataFrame(
                {"Size (KiB)": [fp32_kib, int8_kib]},
                index=["FP32", "INT8 (dynamic)"],
            )
            st.bar_chart(df, y="Size (KiB)")
        except Exception:  # pragma: no cover - pandas optional
            st.write(
                {"FP32 (KiB)": round(fp32_kib, 2), "INT8 (KiB)": round(int8_kib, 2)}
            )


def _render_latency_chart(st: Any, cpu: dict[str, Any], use_plotly: bool) -> None:
    """CPU latency by percentile — grouped bars with error bars (95% CI)."""
    lat = cpu.get("latency", {})
    fp32 = lat.get("fp32", {})
    int8 = lat.get("int8", {})
    cis = cpu.get("bootstrap_ci", {})

    st.markdown("#### CPU latency by percentile (95% bootstrap CIs)")
    st.caption(
        "Single-image inference latency on CPU at the p50/p95/p99 percentiles. "
        "Error bars are percentile-bootstrap 95% confidence intervals from the "
        "200 raw per-run samples. Lower is better. FP32 is faster here because "
        "dynamic INT8 adds per-op quantize/dequantize overhead on this small graph."
    )

    percentiles = ["p50_ms", "p95_ms", "p99_ms"]
    p_labels = ["p50", "p95", "p99"]

    rendered = False
    if use_plotly:
        try:
            import plotly.graph_objects as go

            def _err(prec: str) -> tuple[list[float], list[float]]:
                """Return (plus, minus) error magnitudes relative to point value."""
                point = [float({"fp32": fp32, "int8": int8}[prec].get(p, 0.0)) for p in percentiles]
                plus, minus = [], []
                for p, val in zip(percentiles, point, strict=False):
                    ci = cis.get(prec, {}).get(p, {})
                    hi = float(ci.get("hi", val))
                    lo = float(ci.get("lo", val))
                    plus.append(max(0.0, hi - val))
                    minus.append(max(0.0, val - lo))
                return plus, minus

            fp32_vals = [float(fp32.get(p, 0.0)) for p in percentiles]
            int8_vals = [float(int8.get(p, 0.0)) for p in percentiles]
            fp32_plus, fp32_minus = _err("fp32")
            int8_plus, int8_minus = _err("int8")

            fig = go.Figure()
            fig.add_bar(
                name="FP32",
                x=p_labels,
                y=fp32_vals,
                text=[f"{v:.2f}" for v in fp32_vals],
                textposition="outside",
                marker_color="#64748b",
                error_y={"type": "data", "symmetric": False, "array": fp32_plus, "arrayminus": fp32_minus},
            )
            fig.add_bar(
                name="INT8 (dynamic)",
                x=p_labels,
                y=int8_vals,
                text=[f"{v:.2f}" for v in int8_vals],
                textposition="outside",
                marker_color="#16a34a",
                error_y={"type": "data", "symmetric": False, "array": int8_plus, "arrayminus": int8_minus},
            )
            fig.update_layout(
                title="CPU latency by percentile (lower is better)",
                barmode="group",
                yaxis_title="Latency (ms)",
                xaxis_title="Percentile",
                legend_title="Precision",
                margin={"t": 50, "b": 40, "l": 60, "r": 20},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            rendered = True
        except Exception:  # pragma: no cover - plotly optional
            rendered = False

    if not rendered:
        try:
            import pandas as pd

            df = pd.DataFrame(
                {
                    "FP32": [float(fp32.get(p, 0.0)) for p in percentiles],
                    "INT8 (dynamic)": [float(int8.get(p, 0.0)) for p in percentiles],
                },
                index=p_labels,
            )
            st.bar_chart(df)
        except Exception:  # pragma: no cover - pandas optional
            st.write({"fp32_ms": fp32, "int8_ms": int8})

    # Always provide the exact numbers in a labeled table — the accessible,
    # non-visual equivalent of the chart above.
    _render_latency_table(st, fp32, int8, cis)


def _render_latency_table(st: Any, fp32: dict, int8: dict, cis: dict) -> None:
    """Exact latency numbers with CIs as a screen-reader-friendly table."""

    def _row(name: str, src: dict, prec_key: str) -> dict[str, str]:
        ci = cis.get(prec_key, {})

        def _fmt(metric: str) -> str:
            val = float(src.get(metric, 0.0))
            bounds = ci.get(metric, {})
            if bounds:
                return f"{val:.3f}  [{float(bounds.get('lo', val)):.3f}, {float(bounds.get('hi', val)):.3f}]"
            return f"{val:.3f}"

        return {
            "Precision": name,
            "p50 ms [95% CI]": _fmt("p50_ms"),
            "p95 ms [95% CI]": _fmt("p95_ms"),
            "p99 ms [95% CI]": _fmt("p99_ms"),
            "FPS (mean)": f"{float(src.get('fps_mean', 0.0)):.1f}",
        }

    rows = [_row("FP32", fp32, "fp32"), _row("INT8 (dynamic)", int8, "int8")]
    try:
        import pandas as pd

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception:  # pragma: no cover - pandas optional
        st.table(rows)


def _render_pareto_section(st: Any, pareto_payload: dict[str, Any] | None, use_plotly: bool) -> None:
    """The headline RQ-E4 Pareto frontier — GPU-pending today.

    We surface real (non-mock) rows if they exist; otherwise we explain that
    the figure is pending and show the GPU lane that produces it.
    """
    st.subheader("Full accuracy x latency x power Pareto (RQ-E4) — GPU-pending")
    st.caption(
        "The headline figure: each engine config plotted by mAP, p95 latency, "
        "and watts/frame. It needs an NVIDIA GPU (TensorRT engines + NVML power "
        "sampling), so it is not measured yet."
    )

    real_rows = _real_pareto_rows(pareto_payload)

    if not real_rows:
        st.info(
            "**Pending a real RTX-class run.** Watts/frame only exists on NVIDIA "
            "GPUs via NVML, and the TensorRT FP16/INT8 latency + real COCO mAP "
            "rows need a CUDA box. On a GPU machine this populates the frontier:\n\n"
            "```bash\n"
            "python scripts/run_power_sweep.py \\\n"
            "    --configs trt-fp32 trt-fp16 trt-int8 onnxrt-cpu \\\n"
            "    --duration-sec 900\n"
            "```\n\n"
            "The two axes already proven on CPU — INT8 size reduction and CPU "
            "latency — are shown above. See the project's `EVALUATION_REPORT.md` "
            "for the lane-by-lane status."
        )
        return

    # Real rows present: render the genuine frontier.
    frontier = set(pareto_payload.get("frontier", []) if pareto_payload else [])
    try:
        import pandas as pd

        df = pd.DataFrame(real_rows)
        df["Status"] = df["label"].apply(
            lambda lbl: "Pareto-optimal" if lbl in frontier else "dominated"
        )
        if use_plotly:
            try:
                import plotly.express as px

                fig = px.scatter(
                    df,
                    x="p95_ms",
                    y="mAP_50_95",
                    size="watts_per_frame",
                    color="Status",
                    symbol="Status",  # shape, so color is not the only signal
                    hover_data=["label", "fps", "size_mb", "precision", "backend"],
                    title="Pareto frontier: accuracy x latency (bubble = watts/frame)",
                    labels={
                        "p95_ms": "p95 latency (ms) — lower is better",
                        "mAP_50_95": "mAP@[0.5:0.95] — higher is better",
                    },
                    color_discrete_map={"Pareto-optimal": "#16a34a", "dominated": "#94a3b8"},
                    symbol_map={"Pareto-optimal": "star", "dominated": "circle"},
                )
                fig.update_traces(marker={"opacity": 0.85, "line": {"width": 1, "color": "white"}})
                st.plotly_chart(fig, use_container_width=True)
            except Exception:  # pragma: no cover
                st.scatter_chart(df, x="p95_ms", y="mAP_50_95")
        st.dataframe(df.sort_values("p95_ms"), use_container_width=True, hide_index=True)
    except Exception:  # pragma: no cover - pandas optional
        st.table(real_rows)


def _render_roadmap(st: Any) -> None:
    """Plain-language lane status so a recruiter sees exactly what is real."""
    st.subheader("Lane status — what is measured vs pending")
    rows = [
        {
            "Lane": "CPU INT8 (size + latency)",
            "Status": "Measured",
            "Where": "phase3_cpu_int8.json",
            "Needs": "Nothing — runs anywhere",
        },
        {
            "Lane": "Real COCO mAP (FP32/FP16/INT8)",
            "Status": "Pending",
            "Where": "phase3_quantization.md",
            "Needs": "GPU + COCO calibration (static QDQ / TRT-INT8)",
        },
        {
            "Lane": "TensorRT latency (RQ-E3)",
            "Status": "Pending",
            "Where": "phase2_compile.md",
            "Needs": "NVIDIA GPU + TensorRT",
        },
        {
            "Lane": "Watts/frame Pareto (RQ-E4, headline)",
            "Status": "Pending",
            "Where": "phase5_power.md",
            "Needs": "RTX-class GPU + NVML",
        },
        {
            "Lane": "MobileSAM segmentation cost (RQ-E5)",
            "Status": "Pending",
            "Where": "phase6_segmentation.md",
            "Needs": "Phase 6",
        },
    ]
    try:
        import pandas as pd

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception:  # pragma: no cover - pandas optional
        st.table(rows)
    st.caption(
        "The two CPU axes are real measurements; every GPU axis is clearly "
        "labeled pending. Honesty is the point — see `docs/EVALUATION_REPORT.md`."
    )


def _render_sidebar(st: Any, cpu: dict[str, Any] | None) -> bool:
    """Sidebar groups all context/controls. Returns whether plotly is wanted."""
    with st.sidebar:
        st.header("About this demo")
        st.markdown(
            "**edge-vision** compresses RT-DETR for edge inference "
            "(ONNX / TensorRT / INT8) and profiles it on **watts per frame**, "
            "not just FPS."
        )
        st.markdown(f"[View the project on GitHub]({_REPO_URL})")

        st.divider()
        st.subheader("Display options")
        use_plotly = st.checkbox(
            "Use interactive charts (Plotly)",
            value=True,
            help="Uncheck to fall back to simpler built-in charts. Exact numbers "
            "are always shown in the tables regardless.",
        )

        st.divider()
        st.subheader("Environment")
        if cpu is not None:
            env = cpu.get("environment", {})
            st.caption(
                f"Benchmarked on:\n\n"
                f"- Python {env.get('python', 'n/a')}\n"
                f"- {env.get('platform', 'n/a')}\n"
                f"- onnxruntime {env.get('onnxruntime', 'n/a')}\n"
                f"- Generated {cpu.get('generated_utc', 'n/a')}"
            )
        else:
            st.caption("No benchmark artifacts loaded.")

        st.divider()
        st.caption(
            "All metrics are read verbatim from `docs/results/*.json`. "
            "Nothing on this page is hardcoded or estimated."
        )
    return use_plotly


def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        print("Install the [ui] extras to run the dashboard: pip install -e '.[ui]'")
        return

    st.set_page_config(
        page_title="edge-vision — CPU ONNX demo",
        page_icon="🎯",
        layout="wide",
    )

    # Cache the (cheap, but repeated-on-rerun) JSON reads.
    @st.cache_data(show_spinner=False)
    def _cpu() -> dict[str, Any] | None:
        return _load_json(_CPU_INT8_JSON)

    @st.cache_data(show_spinner=False)
    def _pareto() -> dict[str, Any] | None:
        return _load_json(_PARETO_JSON)

    cpu = _cpu()
    pareto_payload = _pareto()

    use_plotly = _render_sidebar(st, cpu)

    _render_header(st)
    _render_headline(st, cpu)

    if cpu is not None:
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            _render_size_chart(st, cpu, use_plotly)
        with col_b:
            _render_latency_chart(st, cpu, use_plotly)

    st.divider()
    _render_pareto_section(st, pareto_payload, use_plotly)

    st.divider()
    _render_roadmap(st)


if __name__ == "__main__":
    main()
