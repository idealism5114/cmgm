#!/usr/bin/env python
"""Phase 6 figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "interpretability" / "phase6"
FIG = P6 / "figures"
REPR = P6 / "representations"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
ORDER = ["Original", "GlobalConcatFusion", "ResidualFusion", "BranchPreservingFusion"]
LABEL = {"Original": "Original\n(gated sum)", "GlobalConcatFusion": "GCF\n(concat+GELU)",
         "ResidualFusion": "ResidualFusion\n(nonlin + residual)",
         "BranchPreservingFusion": "BranchPreserving\n(3-way)"}
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "figure.dpi": 140})


def main_results():
    return pd.read_csv(P6 / "tables" / "phase6_main_results.csv")


def fig1():
    d = main_results().set_index("model")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    base = float(d.loc["Original", "VAL_Huber"])
    for i, m in enumerate(ORDER):
        v = float(d.loc[m, "VAL_Huber"])
        ax.plot([i - 0.22, i + 0.22], [v, v], color=SERIES[i], linewidth=3,
                solid_capstyle="round")
        ax.scatter([i], [v], s=80, color=SERIES[i], zorder=4, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(f"{100*(base-v)/base:+.2f}%", (i, v), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9)
    ax.axhline(base, color=INK2, linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(ORDER)), [LABEL[m] for m in ORDER], fontsize=8)
    lo = d.VAL_Huber.min(); hi = d.VAL_Huber.max()
    ax.set_ylim(lo - (hi - lo) * 0.5, hi + (hi - lo) * 0.6)
    ax.set_ylabel("VAL Huber (lower is better; axis not from zero)")
    ax.set_title("Figure 1 - Fusion architectures, seed 42 (n = 1). Labels give change "
                 "vs Original; dashed = Original", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig01_val_huber.png"); plt.close(fig)


def fig2():
    d = main_results().set_index("model")
    hs = ["MAE@1", "MAE@5", "MAE@10", "MAE@20"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.2
    idx = np.arange(len(hs))
    for i, m in enumerate(ORDER):
        ax.bar(idx + (i - 1.5) * w, [float(d.loc[m, h]) for h in hs], width=w,
               color=SERIES[i], label=LABEL[m].replace("\n", " "))
    ax.set_xticks(idx, [h.replace("MAE@", "MAE@") + "d" for h in hs])
    ax.set_ylabel("mean MAE")
    ax.set_title("Figure 2 - Horizon MAE by fusion architecture, seed 42 (n = 1)", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig02_horizon_mae.png"); plt.close(fig)


def fig3():
    d = main_results().set_index("model")
    base = float(d.loc["Original", "VAL_Huber"])
    ms = [m for m in ("GlobalConcatFusion", "ResidualFusion", "BranchPreservingFusion")]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    vals = [100 * (base - float(d.loc[m, "VAL_Huber"])) / base for m in ms]
    ax.bar(range(len(ms)), vals, color=[SERIES[1], SERIES[2], SERIES[3]], width=0.55)
    ax.axhline(0.0, color=INK2, linestyle="--", linewidth=1.0)
    for band, lab in ((0.061, "reproducibility floor 0.061%"),):
        ax.axhline(band, color=INK2, linestyle=":", linewidth=1.0)
        ax.axhline(-band, color=INK2, linestyle=":", linewidth=1.0)
        ax.annotate(lab, (len(ms) - 0.5, band), fontsize=7, color=INK2,
                    xytext=(2, 3), textcoords="offset points")
    ax.set_xticks(range(len(ms)), [LABEL[m].replace("\n", " ") for m in ms], fontsize=8)
    ax.set_ylabel("relative VAL Huber improvement vs Original (%)")
    ax.set_title("Figure 3 - Improvement over Original, seed 42 (n = 1)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig03_improvement.png"); plt.close(fig)


def fig4():
    d = main_results().set_index("model")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, m in enumerate(ORDER):
        ax.scatter([float(d.loc[m, "parameter_count"])], [float(d.loc[m, "VAL_Huber"])],
                   s=90, color=SERIES[i], zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(m, (float(d.loc[m, "parameter_count"]), float(d.loc[m, "VAL_Huber"])),
                    textcoords="offset points", xytext=(7, 5), fontsize=7)
    ax.set_xlabel("total parameters")
    ax.set_ylabel("seed42 VAL Huber")
    ax.set_title("Figure 4 - Parameters vs validation Huber (descriptive only; "
                 "not a parameter-matched study)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig04_params_vs_huber.png"); plt.close(fig)


def fig5():
    st = pd.read_csv(REPR / "fusion_stage_diagnostics.csv")
    kn = pd.read_csv(REPR / "fusion_knn_retention.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for i, m in enumerate(ORDER):
        s = st[(st.model == m) & (st.stage == "fusion_output")]
        if len(s):
            axes[0].bar(i, float(s.participation_ratio.iloc[0]), color=SERIES[i], width=0.6)
            axes[2].bar(i, float(s["mean"].iloc[0]), color=SERIES[i], width=0.6)
        k = kn[kn.model == m]
        if len(k):
            axes[1].bar(i, float(k.retention.iloc[0]), color=SERIES[i], width=0.6)
    for ax, ttl, ylab in ((axes[0], "effective dimension of fusion output",
                           "participation ratio"), (axes[1], "kNN retention k=10",
                           "input -> output retention"),
                          (axes[2], "consecutive-sample dynamics of fusion output",
                           "mean step / sqrt(d)")):
        ax.set_xticks(range(len(ORDER)), [m.replace("Fusion", "\nFusion") for m in ORDER],
                      fontsize=7)
        ax.set_title(ttl, fontsize=9); ax.set_ylabel(ylab)
    fig.suptitle("Figure 5 - Fusion representation context (auxiliary; Phase 5 showed "
                 "geometry preservation does not rank performance)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig05_representation_diagnostics.png"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5):
        f(); print("wrote", f.__name__)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
