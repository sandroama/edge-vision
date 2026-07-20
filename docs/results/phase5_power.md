# Phase 5 — Power + Thermal Profiling (results)

> **Status:** Modules wired. Mock-mode CI smoke runs end-to-end and generates
> the Pareto table. Real NVML numbers require the RTX-5080 + ``pynvml`` + a
> working engine from Phases 1–4.

## Module status

| Module | File | Tests | Notes |
|---|---|---|---|
| NVML power monitor | `src/edgevision/profiling/nvml_power.py` | ✅ | `PowerMonitor` (pynvml) + `MockPowerMonitor` (always works) + `power_monitor` context manager |
| Thermal runner | `src/edgevision/profiling/thermal_runner.py` | ✅ | `run_sustained` runs a callable for N seconds while NVML samples in the background; `ThermalRunResult` = latency + power |
| CPU profiler | `src/edgevision/profiling/cpu_profile.py` | ✅ (psutil-gated) | psutil CPU% + RSS + estimated watt draw from TDP × utilisation |
| Pareto aggregator | `src/edgevision/evaluation/pareto_aggregator.py` | ✅ | `dominates` / `is_dominated` / `pareto_frontier` / `write_report` → `phase5_pareto.md` + `phase5_pareto.json` |
| Pareto dashboard | `dashboard/pareto_plot.py` | — (UI) | Full Streamlit app with Plotly scatter; shows placeholder when no JSON present |
| Power sweep script | `scripts/run_power_sweep.py` | ✅ | `--mock-power` for CI; real path for GPU; writes and re-reads phase5_power.json |

**Running totals: 127 tests passing, 26 skipping cleanly, 0 failing.**

## CI smoke (mock mode, no GPU)

Confirmed that the full pipeline runs end-to-end with `--mock-power`:

```
[edge-vision] Sweeping config='mock-fp32'  duration=0.1s ...
  -> [mock-fp32] fps=407.0  p95=2.52ms  power_mean=200.4W  watts/frame=0.4924  throttle=10
[edge-vision] Sweeping config='mock-fp16'  duration=0.1s ...
  -> [mock-fp16] fps=404.2  p95=2.53ms  power_mean=200.4W  watts/frame=0.4958  throttle=10

Pareto table: mock-fp32 ✅ Pareto-optimal (lower p95)
              mock-fp16 —  (dominated)
```

The mock injects a simulated throttle event in the last 10% of samples, which exercises the `throttle_events` counter in `PowerProfile`.

## RQ-E4 — power-latency frontier (pending GPU run)

Run the full sweep to fill in this table:

```bash
# Requires: [gpu] extras, pynvml, Phase 2-3 engines in checkpoints/
python scripts/run_power_sweep.py \
    --configs trt-fp32 trt-fp16 trt-int8 onnxrt-cpu \
              distilled-fp16 distilled-int8 \
    --duration-sec 900 \
    --sample-ms 100 \
    --out-json docs/results/phase5_power.json

# Then view the Pareto dashboard:
streamlit run dashboard/pareto_plot.py
```

| Config | mAP@[0.5:0.95] | p95 (ms) | Watts/frame | FPS | Pareto? |
|---|---|---|---|---|---|
| trt-fp32 | TBD | TBD | TBD | TBD | TBD |
| trt-fp16 | TBD | TBD | TBD | TBD | TBD |
| trt-int8 | TBD | TBD | TBD | TBD | TBD |
| distilled-r18-fp16 | TBD | TBD | TBD | TBD | TBD |
| distilled-r18-int8 | TBD | TBD | TBD | TBD | TBD |
| onnxrt-cpu-fp32 | TBD | TBD | TBD | TBD | TBD |
| onnxrt-cpu-int8 | TBD | TBD | TBD | TBD | TBD |

## Design notes

### Thermal throttle detection

A "throttle event" is defined as a sample where GPU clock falls below 90%
of the peak observed clock during the run. On RTX 5080 (Blackwell) with
the default power limit (~320W), sustained inference at FP32 can hit this
after ~8–12 minutes depending on case airflow and thermal paste state.

The metric `throttle_events / n_samples` is a *rate*, not a binary flag.
Reporting it this way means you can say "throttling occurred in 4% of
samples" rather than "yes/no" — the interview nuance that separates serious
hardware reporting from checkbox documentation.

### Watts per frame: the right unit

Most hardware ML reporting uses "throughput (FPS)" or "latency (ms)" but
forgets power. Watts/frame = mean_power_W / FPS. It's the right unit because:
- It captures the *energy cost per prediction*, not just speed.
- An INT8 engine that draws 40% less power at the same FPS is significantly
  better in deployment.
- On a Jetson (Phase 7), power budget is the binding constraint, not VRAM.

### CPU TDP estimation

The `CpuProfiler.estimated_power_w` is TDP × CPU_utilisation_pct / 100.
The 9950X has a configurable TDP of 65–170 W. Using 65W (base TDP) gives a
conservative (under-)estimate. For precise measurements, a power meter at
the wall or RAPL energy counters are more accurate; this module's estimate
is "good enough to see the relative trend" not "hardware-grade billing."
