#!/usr/bin/env python
"""Phase 5 figures (static PNGs)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / "interpretability" / "phase5"
FD = P5 / "frozen_diagnostics"
FC = P5 / "fusion_controls"
FIG = P5 / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "figure.dpi": 140})


def pipe(stages, labels, fname, title):
    dim = pd.read_csv(FD / "effective_dimension.csv")
    dyn = pd.read_csv(FD / "stage_dynamics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    xs = np.arange(len(stages))
    for i, s in enumerate((42, 43, 44)):
        d = dim[dim.seed == s].set_index("stage")
        y = dim[dim.seed == s].set_index("stage")
        axes[0].plot(xs, [d.loc[st, "normalized"] if st in d.index else np.nan for st in stages],
                     "o-", color=SERIES[i], markersize=5, label=f"seed{s}")
        axes[1].plot(xs, [y.loc[st, "normalized"] if st in y.index else np.nan for st in stages],
                     "o-", color=SERIES[i], markersize=5, alpha=0.0)
    axes[1].clear()
    for i, s in enumerate((42, 43, 44)):
        dd = dyn[dyn.seed == s].set_index("stage")
        axes[1].plot(xs, [dd.loc[st, "mean"] if st in dd.index else np.nan for st in stages],
                     "o-", color=SERIES[i], markersize=5, label=f"seed{s}")
    for ax, ttl, ylab in ((axes[0], "normalised effective dimension (PR/D)",
                           "participation ratio / width"),
                          (axes[1], "consecutive-sample dynamics", "mean step / sqrt(d)")):
        ax.set_xticks(xs, labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def geo():
    g = pd.read_csv(FD / "geometry_preservation.csv")
    k = pd.read_csv(FD / "knn_retention.csv")
    order = ["S_node_prgnn->S_node_prepool", "S_node_prepool->S_pool_stock",
             "S_node_prepool->S_hs", "S_node_prepool->S_s", "S_hs->S_s",
             "T_cat->T_ht", "T_ht->T_t", "F1_s_t->F2_fused", "F0_hs_ht->F1_s_t"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    gm = g.groupby("pair").spearman.mean().reindex([o for o in order if o in set(g.pair)])
    axes[0].barh(range(len(gm)), gm.values, color=SERIES[0], height=0.6)
    axes[0].set_yticks(range(len(gm)), gm.index, fontsize=7)
    axes[0].set_xlabel("Spearman of pairwise sample distances")
    axes[0].set_xlim(0.8, 1.0)
    axes[0].set_title("geometry preservation", fontsize=9)
    kp = k[k.k > 0].groupby(["pair", "k"]).retention.mean().unstack()
    kp = kp.reindex([o for o in order if o in kp.index])
    w = 0.25
    for i, kk in enumerate([5, 10, 20]):
        axes[1].barh(np.arange(len(kp)) + (i - 1) * w, kp[kk].values, height=w,
                     color=SERIES[i], label=f"k={kk}")
    axes[1].set_yticks(range(len(kp)), kp.index, fontsize=7)
    axes[1].set_xlabel("kNN neighbourhood retention")
    axes[1].set_title("neighbourhood retention", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_xlim(0.6, 1.0)
    fig.suptitle("Figure 3 - Geometry and neighbourhood retention through each "
                 "compression stage (3-seed mean)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_geometry_retention.png")
    plt.close(fig)


def probes():
    d = pd.read_csv(FD / "linear_probe_metrics.csv")
    d = d[d.alpha == 1.0]
    order = ["S_node_prepool_stock_pca64", "S_node_prepool_commodity_pca64",
             "T_ht", "T_Z", "T_cat", "T_H", "S_hs", "F1_s_t", "F0_hs_ht", "F2_fused"]
    m = d.groupby("representation").probe_Huber.mean().reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [SERIES[0] if "T_" in r else SERIES[1] if "S" in r.split("_")[0] else SERIES[2]
              for r in m.index]
    ax.barh(range(len(m)), m.values, color=colors, height=0.62)
    ax.set_yticks(range(len(m)), m.index, fontsize=8)
    ax.set_xlabel("ridge probe Huber on VAL (lower is better; fit on TRAIN)")
    ax.axvline(0.00163, color=INK2, linewidth=1.2, linestyle="--")
    ax.annotate("trained full model 0.00163", (0.00163, len(m) - 0.5),
                fontsize=7, color=INK2, xytext=(4, 0), textcoords="offset points")
    ax.set_title("Figure 4 - Linear predictive accessibility of frozen "
                 "representations (diagnostic probe, not main-model performance)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_probe_accessibility.png")
    plt.close(fig)


def controls():
    m = pd.read_csv(FC / "fusion_control_metrics.csv")
    order = ["GlobalConcatFusion", "Original", "GatedConcat128", "S1-FixedMeanFusion",
             "MeanConcat128", "ConcatLinear64"]
    lab = {"GlobalConcatFusion": "GCF (concat, GELU, 64)",
           "Original": "Original (gated sum, 64)",
           "GatedConcat128": "C2 gated concat, 128",
           "S1-FixedMeanFusion": "S1 mean sum, 64",
           "MeanConcat128": "C3 mean concat, 128",
           "ConcatLinear64": "C1 concat, linear, 64"}
    m = m.set_index("model").reindex([o for o in order if o in set(m.model)])
    base = float(m.loc["Original", "VAL_Huber"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (name, r) in enumerate(m.iterrows()):
        v = float(r["VAL_Huber"])
        ax.plot([i - 0.22, i + 0.22], [v, v], color=SERIES[i % len(SERIES)], linewidth=3,
                solid_capstyle="round")
        ax.scatter([i], [v], s=70, color=SERIES[i % len(SERIES)], zorder=4,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(f"{100*(base-v)/base:+.2f}%", (i, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8)
    ax.axhline(base, color=INK2, linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(m)), [lab.get(x, x) for x in m.index], rotation=20,
                  ha="right", fontsize=8)
    ax.set_ylabel("seed42 VAL Huber (lower is better; axis not from zero)")
    lo, hi = m.VAL_Huber.min(), m.VAL_Huber.max()
    ax.set_ylim(lo - (hi - lo) * 0.6, hi + (hi - lo) * 0.6)
    ax.set_title("Figure 5 - Fusion controls, seed 42 (n = 1; labels give change vs "
                 "Original; dashed = Original)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_fusion_controls.png")
    plt.close(fig)


def width_operator():
    m = pd.read_csv(FC / "fusion_control_metrics.csv").set_index("model")
    base = float(m.loc["Original", "VAL_Huber"])
    pts = [("Original", 64, "gated sum"), ("S1-FixedMeanFusion", 64, "mean sum"),
           ("ConcatLinear64", 64, "concat+linear"), ("GlobalConcatFusion", 64, "concat+GELU"),
           ("GatedConcat128", 128, "gated concat"), ("MeanConcat128", 128, "mean concat")]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, (name, w, op) in enumerate(pts):
        if name not in m.index:
            continue
        y = 100.0 * (base - float(m.loc[name, "VAL_Huber"])) / base
        mk = "o" if w == 64 else "s"
        ax.scatter([i], [y], s=90, marker=mk, color=SERIES[i % len(SERIES)], zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(f"{op}\n{w}d", (i, y), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=7)
    ax.axhline(0.0, color=INK2, linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(pts)), [""] * len(pts))
    ax.set_ylabel("improvement vs Original (%)")
    ax.set_title("Figure 6 - Fusion operator and width map (seed 42, n = 1). "
                 "Circles = 64d output, squares = 128d output", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_width_operator_map.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pipe(["S_node_prgnn", "S_node_prepool", "S_pool_stock", "S_hs", "S_s"],
         ["pre-GNN\nnodes", "pre-pool\nnodes", "stock\npool", "h_s", "s"],
         "fig01_spatial_pipeline.png",
         "Figure 1 - Spatial compression pipeline (flattened node tensors)")
    pipe(["T_token", "T_H", "T_Z", "T_cat", "T_ht", "T_t"],
         ["token", "H_T", "Z_T", "[H_T;Z_T]", "h_t", "t"],
         "fig02_temporal_pipeline.png",
         "Figure 2 - Temporal compression pipeline")
    geo(); probes(); controls(); width_operator()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
