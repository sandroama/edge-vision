"""Streamlit Pareto-frontier dashboard (Phase 5).

Renders the (mAP × p95-latency × watts/frame) comparison table and an
interactive scatter plot. Loads from ``docs/results/phase5_pareto.json``
produced by ``scripts/run_power_sweep.py``.

Run::

    streamlit run dashboard/pareto_plot.py

When no JSON is present it shows a placeholder with instructions.
"""

from __future__ import annotations

from pathlib import Path

_PARETO_JSON = Path(__file__).parent.parent / "docs" / "results" / "phase5_pareto.json"


def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        print("Install the [ui] extras: pip install -e '.[dev,ui]'")
        return

    st.set_page_config(
        page_title="edge-vision — Pareto frontier",
        page_icon="🎯",
        layout="wide",
    )
    st.title("edge-vision — accuracy × latency × power")
    st.caption(
        "RQ-E4 from the edge-vision project. Each point is one (model, precision, backend) "
        "configuration profiled with NVML over a 15-minute sustained run."
    )

    if not _PARETO_JSON.exists():
        st.info(
            "No Pareto data yet. Run the power sweep to populate:\n\n"
            "```bash\n"
            "python scripts/run_power_sweep.py \\\n"
            "    --configs trt-fp32 trt-fp16 trt-int8 onnxrt-cpu \\\n"
            "    --duration-sec 900 \\\n"
            "    --out-json docs/results/phase5_power.json\n"
            "```\n\n"
            "For a quick CI smoke:\n\n"
            "```bash\n"
            "python scripts/run_power_sweep.py \\\n"
            "    --configs mock-fp32 mock-fp16 \\\n"
            "    --duration-sec 30 --mock-power\n"
            "```"
        )
        return

    import json

    import pandas as pd

    payload = json.loads(_PARETO_JSON.read_text())
    configs = payload.get("configs", [])
    frontier_labels = set(payload.get("frontier", []))

    if not configs:
        st.warning("phase5_pareto.json exists but contains no configs.")
        return

    df = pd.DataFrame(configs)
    df["on_frontier"] = df["label"].apply(
        lambda lbl: "✅ Pareto-optimal" if lbl in frontier_labels else "dominated"
    )

    # Key metrics summary.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Configs evaluated", len(df))
    col2.metric("Pareto-optimal", df["on_frontier"].str.startswith("✅").sum())
    best_fps_row = df.loc[df["fps"].idxmax()]
    col3.metric("Best FPS", f"{best_fps_row['fps']:.0f}", help=best_fps_row["label"])
    best_eff_row = df.loc[df["watts_per_frame"].idxmin()]
    col4.metric("Best watts/frame", f"{best_eff_row['watts_per_frame']:.4f}", help=best_eff_row["label"])

    st.divider()

    # Scatter: p95 latency vs mAP, sized by watts/frame.
    try:
        import plotly.express as px

        fig = px.scatter(
            df,
            x="p95_ms",
            y="mAP_50_95",
            size="watts_per_frame",
            color="on_frontier",
            hover_data=["label", "fps", "size_mb", "precision", "backend"],
            title="Pareto frontier: accuracy × latency (bubble size = watts/frame)",
            labels={
                "p95_ms": "p95 latency (ms) ↓ is better",
                "mAP_50_95": "mAP@[0.5:0.95] ↑ is better",
                "on_frontier": "",
            },
            color_discrete_map={
                "✅ Pareto-optimal": "#16a34a",
                "dominated": "#94a3b8",
            },
        )
        fig.update_traces(marker={"opacity": 0.8, "line": {"width": 1, "color": "white"}})
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Install plotly (`pip install -e '.[ui]'`) for the scatter plot.")

    st.divider()

    # Full table.
    st.subheader("All configurations")
    display_cols = [
        "label", "mAP_50_95", "p95_ms", "watts_per_frame", "fps",
        "size_mb", "precision", "backend", "on_frontier",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[existing_cols].sort_values("p95_ms"),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
