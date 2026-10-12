#!/usr/bin/env python
"""Phase 3 figures for the functional mechanism audit (static PNGs)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "interpretability" / "phase3"
FIG = P3 / "figures"
SEEDS = (42, 43, 44)
HORIZONS = (1, 5, 10, 20)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "figure.dpi": 140,
})

GATE_ORDER = ["gate_original", "gate_fixed_0.5", "gate_temporal_only",
              "gate_spatial_only", "gate_sample_shuffle_mean",
              "gate_channel_shuffle_mean"]
GATE_LABEL = {"gate_original": "original", "gate_fixed_0.5": "g=0.5",
              "gate_temporal_only": "temporal only", "gate_spatial_only": "spatial only",
              "gate_sample_shuffle_mean": "gate shuffled (sample)",
              "gate_channel_shuffle_mean": "gate shuffled (channel)"}


def _vec(v):
    """Accept a JSON string/object, a parquet struct, or a plain sequence."""
    import json as _j
    if v is None:
        return np.array([])
    if isinstance(v, str):
        v = _j.loads(v)
    if isinstance(v, dict):
        keys = [k for k in ("1", "5", "10", "20") if k in v] or sorted(v)
        return np.asarray([v[k] for k in keys], dtype=float)
    return np.asarray(v, dtype=float)


def gate_metrics():
    return pd.read_csv(P3 / "gate" / "gate_intervention_metrics.csv")


def fig01_gate_performance():
    d = gate_metrics()
    cols = ["VAL_Huber", "MAE@1", "MAE@5", "MAE@10", "MAE@20"]
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4))
    for ax, c in zip(axes, cols):
        for i, s in enumerate(SEEDS):
            sub = d[d.seed == s].set_index("intervention")
            vals = [sub.loc[g, c] if g in sub.index else np.nan for g in GATE_ORDER]
            ax.plot(range(len(GATE_ORDER)), vals, "o-", color=SERIES[i],
                    markersize=6, label=f"seed{s}")
        ax.set_xticks(range(len(GATE_ORDER)),
                      [GATE_LABEL[g] for g in GATE_ORDER], rotation=40,
                      ha="right", fontsize=7)
        ax.set_title(c, fontsize=9)
    axes[0].set_ylabel("metric value")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Figure 1 - Gate intervention performance (frozen model; no "
                 "retraining)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_gate_performance.png")
    plt.close(fig)


def fig02_gate_deviation():
    d = pd.read_parquet(P3 / "gate" / "gate_prediction_changes.parquet")
    sub = d[d.intervention.isin([g for g in GATE_ORDER if g in set(d.intervention)])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [g for g in GATE_ORDER if g in set(sub.intervention)]
    w = 0.2
    for hi, h in enumerate(HORIZONS):
        vals = []
        for g in labels:
            row = sub[(sub.intervention == g)]
            vecs = [_vec(x) for x in row.per_horizon_mean_abs_change]
            vecs = [a for a in vecs if a.size > hi]
            v = float(np.mean([a[hi] for a in vecs])) if vecs else np.nan
            vals.append(v)
        ax.bar(np.arange(len(labels)) + (hi - 1.5) * w, vals, width=w,
               color=SERIES[hi], label=f"{h}d")
    ax.set_xticks(range(len(labels)), [GATE_LABEL[g] for g in labels],
                  rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean |prediction change| per horizon")
    ax.set_title("Figure 2 - Gate prediction deviation from the original "
                 "(mean over 3 seeds)", fontsize=10)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_gate_deviation.png")
    plt.close(fig)


def fig03_prior_sweep():
    m = pd.read_csv(P3 / "graph" / "prior_sweep_metrics.csv")
    st = pd.read_csv(P3 / "graph" / "prior_sweep_structure.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, s in enumerate(SEEDS):
        sub = m[m.seed == s].drop_duplicates("lambda_value").sort_values("lambda_value")
        axes[0].plot(sub.lambda_value, sub["VAL_Huber"], "o-", color=SERIES[i],
                     markersize=5, label=f"seed{s}")
        axes[0].plot(sub.lambda_value, sub["MAE@5"], "s--", color=SERIES[i],
                     markersize=4, alpha=0.7)
    axes[0].set_xlabel("prior strength lambda")
    axes[0].set_ylabel("Huber (solid) / MAE@5 (dashed)")
    axes[0].set_title("prediction performance vs prior strength", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    for i, s in enumerate(SEEDS):
        sub = st[st.seed == s].sort_values("lambda_value")
        axes[1].plot(sub.lambda_value, sub.attention_concentration, "o-",
                     color=SERIES[i], markersize=5, label=f"seed{s} concentration")
        axes[1].plot(sub.lambda_value, sub.content_attention_spearman, "s--",
                     color=SERIES[i], markersize=4, alpha=0.7)
    axes[1].axhline(0.0, color=INK2, linewidth=1.0, linestyle=":")
    axes[1].set_xlabel("prior strength lambda")
    axes[1].set_ylabel("concentration (solid) / content-attention Spearman (dashed)")
    axes[1].set_title("attention structure vs prior strength", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Figure 3 - Prior-strength sweep (lambda=0.5 is the original "
                 "model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_prior_sweep.png")
    plt.close(fig)


def fig04_graph_interventions():
    m = pd.read_csv(P3 / "graph" / "graph_intervention_metrics.csv")
    ch = pd.read_parquet(P3 / "graph" / "graph_prediction_changes.parquet")
    keep = ["graph_original", "graph_prior_only", "graph_content_only", "graph_uniform"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, s in enumerate(SEEDS):
        sub = m[m.seed == s].set_index("intervention")
        axes[0].plot(range(len(keep)), [sub.loc[k, "VAL_Huber"] for k in keep],
                     "o-", color=SERIES[i], markersize=6, label=f"seed{s}")
        c = ch[(ch.seed == s)].set_index("intervention")
        axes[1].plot(range(len(keep)),
                     [c.loc[k, "mean_abs_prediction_change"] for k in keep],
                     "o-", color=SERIES[i], markersize=6, label=f"seed{s}")
    for ax, ttl, ylab in ((axes[0], "validation performance", "VAL Huber"),
                          (axes[1], "deviation from original",
                           "mean |prediction change|")):
        ax.set_xticks(range(len(keep)),
                      [k.replace("graph_", "") for k in keep], rotation=25,
                      ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Figure 4 - Original vs prior-only vs content-only vs uniform "
                 "attention", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_graph_interventions.png")
    plt.close(fig)


def fig05_generator_similarity():
    d = pd.read_csv(P3 / "regime" / "generator_similarity.csv")
    pairs = sorted(d.pair.unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.25
    for i, s in enumerate(SEEDS):
        sub = d[d.seed == s].set_index("pair")
        ax.bar(np.arange(len(pairs)) + (i - 1) * w,
               [sub.loc[p, "cosine_mean"] for p in pairs], width=w,
               color=SERIES[i], label=f"seed{s}")
        ax.errorbar(np.arange(len(pairs)) + (i - 1) * w,
                    [sub.loc[p, "cosine_mean"] for p in pairs],
                    yerr=[sub.loc[p, "cosine_std"] for p in pairs],
                    fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.axhline(1.0, color=INK2, linewidth=1.0, linestyle="--")
    ax.set_xticks(range(len(pairs)), pairs)
    ax.set_ylabel("mean pairwise cosine (bars = std)")
    ax.set_title("Figure 5 - State-generator similarity (1.0 = identical "
                 "direction)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_generator_similarity.png")
    plt.close(fig)


def fig06_regime_interventions():
    m = pd.read_csv(P3 / "regime" / "regime_intervention_metrics.csv")
    keep = ["regime_original", "regime_uniform_router", "regime_expert0_only",
            "regime_expert1_only", "regime_expert2_only"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, s in enumerate(SEEDS):
        sub = m[m.seed == s].set_index("intervention")
        axes[0].plot(range(len(keep)), [sub.loc[k, "VAL_Huber"] for k in keep],
                     "o-", color=SERIES[i], markersize=6, label=f"seed{s}")
        ch = pd.read_parquet(P3 / "regime" / "regime_prediction_changes.parquet")
        c = ch[ch.seed == s].set_index("intervention")
        axes[1].plot(range(len(keep)),
                     [c.loc[k, "mean_abs_prediction_change"] for k in keep],
                     "o-", color=SERIES[i], markersize=6, label=f"seed{s}")
    for ax, ttl, ylab in ((axes[0], "validation performance", "VAL Huber"),
                          (axes[1], "deviation from original",
                           "mean |prediction change|")):
        ax.set_xticks(range(len(keep)),
                      [k.replace("regime_", "") for k in keep], rotation=25,
                      ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Figure 6 - Regime routing interventions (full Z recursion "
                 "re-walked from t=0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_regime_interventions.png")
    plt.close(fig)


def fig07_sensitivity_summary():
    g = gate_metrics()
    gr = pd.read_csv(P3 / "graph" / "graph_intervention_metrics.csv")
    rg = pd.read_csv(P3 / "regime" / "regime_intervention_metrics.csv")
    core = [("Gate neutralised", g, "gate_fixed_0.5"),
            ("Prior removed", gr, "graph_content_only"),
            ("Content removed", gr, "graph_prior_only"),
            ("Routing uniform", rg, "regime_uniform_router")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.25
    for i, s in enumerate(SEEDS):
        vals = []
        for _, df, key in core:
            sub = df[df.seed == s].set_index("intervention")
            vals.append(sub.loc[key, "relative_VAL_Huber_pct"])
        ax.bar(np.arange(len(core)) + (i - 1) * w, vals, width=w,
               color=SERIES[i], label=f"seed{s}")
    ax.axhline(0.0, color=INK2, linewidth=1.0, linestyle="--")
    ax.set_xticks(range(len(core)), [c[0] for c in core], rotation=20, ha="right",
                  fontsize=8)
    ax.set_ylabel("relative VAL Huber degradation (%)")
    ax.set_title("Figure 7 - Mechanism sensitivity summary (0 = no change; "
                 "negative = better than original)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_sensitivity_summary.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (fig01_gate_performance, fig02_gate_deviation, fig03_prior_sweep,
               fig04_graph_interventions, fig05_generator_similarity,
               fig06_regime_interventions, fig07_sensitivity_summary):
        fn()
        print("wrote", fn.__name__)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
