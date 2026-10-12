#!/usr/bin/env python
"""Phase 4 figures (static PNGs for the markdown report)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P4 = ROOT / "interpretability" / "phase4"
FIG = P4 / "figures"
SEEDS = (42,)   # Phase 4 is single-seed

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
ORDER = ["Original", "S1-FixedMeanFusion", "S2-NoGraphPrior", "S3-UniformRouter",
         "S4-SimplifiedAll", "GlobalConcatFusion"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "figure.dpi": 140,
})


def runs():
    return pd.read_csv(P4 / "tables" / "phase4_runs_long.csv")


def mean_std():
    return pd.read_csv(P4 / "tables" / "phase4_summary_mean_std.csv")


def paired():
    return pd.read_csv(P4 / "tables" / "paired_differences.csv")


def present():
    r = runs()
    return [m for m in ORDER if m in set(r.model_label)]


def fig01_mean_huber():
    """Dot plot: near-equal values, so bars from zero would hide every difference."""
    df = runs()
    models = present()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vals = [float(df[df.model_label == m].VAL_Huber.iloc[0]) for m in models]
    base = float(df[df.model_label == "Original"].VAL_Huber.iloc[0])
    for i, (m, v) in enumerate(zip(models, vals)):
        ax.plot([i - 0.22, i + 0.22], [v, v], color=SERIES[i % len(SERIES)],
                linewidth=3, solid_capstyle="round", zorder=3)
        ax.scatter([i], [v], s=70, color=SERIES[i % len(SERIES)], zorder=4,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(f"{100*(base-v)/base:+.2f}%", (i, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color=INK)
    ax.axhline(base, color=INK2, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xticks(range(len(models)), models, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("VAL Huber (lower is better, axis does not start at zero)")
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.6, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title("Figure 1 - Validation Huber by model, seed 42 (labels give the "
                 "change vs Original; dashed line = Original)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_mean_val_huber.png")
    plt.close(fig)


def fig02_horizon_mae():
    ms = mean_std()
    models = present()
    hs = ["VAL_MAE_1", "VAL_MAE_5", "VAL_MAE_10", "VAL_MAE_20"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / max(len(models), 1)
    idx = np.arange(len(hs))
    for i, m in enumerate(models):
        vals = [ms[(ms.model == m) & (ms.metric == h)]["mean"].iloc[0] for h in hs]
        ax.bar(idx + (i - len(models) / 2 + 0.5) * w, vals, width=w,
               color=SERIES[i % len(SERIES)], label=m)
    ax.set_xticks(idx, [h.replace("VAL_MAE_", "MAE@") + "d" for h in hs])
    ax.set_ylabel("mean MAE")
    ax.set_title("Figure 2 - Horizon MAE by model, seed 42", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_horizon_mae.png")
    plt.close(fig)


def fig03_paired_improvement():
    p = paired()
    p = p[p.reference == "Original"]
    models = [m for m in present() if m != "Original"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.25
    for i, s in enumerate(SEEDS):
        vals = []
        for m in models:
            row = p[(p.candidate == m) & (p.metric == "VAL_Huber")]
            if row.empty:
                vals.append(np.nan); continue
            df = runs()
            c = df[(df.model_label == m) & (df.seed == s)].VAL_Huber
            r = df[(df.model_label == "Original") & (df.seed == s)].VAL_Huber
            vals.append(float(100.0 * (r.iloc[0] - c.iloc[0]) / r.iloc[0]))
        ax.bar(np.arange(len(models)) + (i - 1) * w, vals, width=w,
               color=SERIES[i], label=f"seed{s}")
    ax.axhline(0.0, color=INK2, linewidth=1.0, linestyle="--")
    ax.set_xticks(range(len(models)), models, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("relative VAL Huber improvement vs Original (%)")
    ax.set_title("Figure 3 - Paired improvement over Original (positive = better)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_paired_improvement.png")
    plt.close(fig)


def fig04_complexity_vs_performance():
    df, ms = runs(), mean_std()
    models = present()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, m in enumerate(models):
        n = df[df.model_label == m].parameter_count.iloc[0]
        y = ms[(ms.model == m) & (ms.metric == "VAL_Huber")]["mean"].iloc[0]
        e = ms[(ms.model == m) & (ms.metric == "VAL_Huber")]["sample_std"].iloc[0]
        ax.errorbar(n, y, yerr=e, fmt="o", color=SERIES[i % len(SERIES)],
                    markersize=8, capsize=3)
        ax.annotate(m, (n, y), textcoords="offset points", xytext=(6, 5), fontsize=7)
    ax.set_xlabel("total parameters")
    ax.set_ylabel("mean VAL Huber")
    ax.set_title("Figure 4 - Complexity vs performance (descriptive only; "
                 "no efficiency claim)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_complexity_vs_performance.png")
    plt.close(fig)


def fig05_frozen_vs_retrained():
    """Phase 3 frozen intervention effect against Phase 4 retrained effect."""
    p3 = ROOT / "interpretability" / "phase3"
    g = pd.read_csv(p3 / "gate" / "gate_intervention_metrics.csv")
    gr = pd.read_csv(p3 / "graph" / "graph_intervention_metrics.csv")
    rg = pd.read_csv(p3 / "regime" / "regime_intervention_metrics.csv")

    def frozen_abs(df, key):
        sub = df[df.intervention == key]["relative_VAL_Huber_pct"].abs()
        return float(sub.mean()) if len(sub) else float("nan")

    mapping = [("gate", "S1-FixedMeanFusion", frozen_abs(g, "gate_fixed_0.5")),
               ("graph prior", "S2-NoGraphPrior",
                frozen_abs(gr, "graph_content_only")),
               ("regime router", "S3-UniformRouter",
                frozen_abs(rg, "regime_uniform_router"))]
    pr = paired()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [m[0] for m in mapping]
    frozen = [m[2] for m in mapping]
    retr = []
    for _, variant, _ in mapping:
        row = pr[(pr.candidate == variant) & (pr.reference == "Original")
                 & (pr.metric == "VAL_Huber")]
        retr.append(abs(float(row["improvement_of_model_means_pct"].iloc[0]))
                    if not row.empty else np.nan)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, frozen, width=0.4, color=SERIES[0],
           label="Phase 3 frozen intervention (|VAL Huber change|, %)")
    ax.bar(x + 0.2, retr, width=0.4, color=SERIES[1],
           label="Phase 4 retrained removal (|model-mean change|, %)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("absolute effect size (%)")
    ax.set_yscale("log")
    ax.set_title("Figure 5 - Frozen intervention vs retrained removal "
                 "(different quantities; not one scale)", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_frozen_vs_retrained.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (fig01_mean_huber, fig02_horizon_mae, fig03_paired_improvement,
               fig04_complexity_vs_performance, fig05_frozen_vs_retrained):
        fn()
        print("wrote", fn.__name__)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
