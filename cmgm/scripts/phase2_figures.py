#!/usr/bin/env python
"""Phase 2 figures. Reads only the frozen Phase 1 artifacts and the Phase 2
tables; draws nothing that implies causality. Static PNGs for the markdown
report, so no hover layer and a single (light) mode.

    python -m cmgm.scripts.phase2_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "interpretability" / "phase2"
FIG = P2 / "figures"
SEEDS = (42, 43, 44)
MARKETS = ("stock", "bond", "commodity")
CAPTURES = ("L1H1", "L1H2", "L2H1", "L2H2")
HORIZONS = (1, 5, 10, 20)

# validated reference palette (light mode), slots used in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seqblue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                "#184f95", "#0d366b"])
DIV_BR = LinearSegmentedColormap.from_list(
    "divbr", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
              "#eda3a2", "#e34948", "#8f1f1e"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "figure.dpi": 140,
})


def _dt(s):
    return pd.to_datetime(s)


def _align_map():
    import json
    a = json.loads((P2 / "regime" / "regime_alignment.json").read_text())
    return {int(k): {int(i): int(j) for i, j in v.items()}
            for k, v in a["alignment"].items()}, a


def fig01_regime_timeline():
    al, _ = _align_map()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, seed in zip(axes, SEEDS):
        d = pd.read_parquet(P2 / "regime" / f"regime_sample_metrics_seed{seed}.parquet")
        x = _dt(d["date"])
        for i in range(3):
            col = f"p_state{seed_aligned_col(al[seed], i)}"
            ax.plot(x, d[col], color=SERIES[i], label=f"R{i+1}",
                    linewidth=1.6)
        ax.set_ylim(0, 1)
        ax.set_title(f"seed{seed}  aligned state probabilities "
                     f"(filtered posterior, final timestep)", fontsize=9)
        ax.set_ylabel("p")
        ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("validation date (last observed day of the window)")
    fig.suptitle("Figure 1 - Regime timeline, aligned across seeds "
                 "(labels aligned by Hungarian matching; no economic naming)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_regime_timeline.png")
    plt.close(fig)


def seed_aligned_col(perm, i):
    """original state index whose aligned label is R{i+1}."""
    for k, v in perm.items():
        if v == i:
            return k
    return i


def fig02_regime_profiles():
    al, _ = _align_map()
    prof = pd.read_csv(P2 / "regime" / "regime_profiles.csv")
    desc = [c for c in prof.descriptor.unique()]
    desc = [c for c in desc]
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        d = prof[prof.seed == seed]
        M = np.full((len(desc), 3), np.nan)
        for i in range(3):
            orig = seed_aligned_col(al[seed], i)
            for r, c in enumerate(desc):
                v = d[(d.state == orig) & (d.descriptor == c)]["mean"]
                if len(v):
                    M[r, i] = v.iloc[0]
        lim = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap=DIV_BR, norm=TwoSlopeNorm(0, -lim, lim),
                       aspect="auto")
        ax.set_xticks(range(3), [f"R{i+1}" for i in range(3)])
        ax.set_yticks(range(len(desc)), desc, fontsize=7)
        ax.set_title(f"seed{seed}", fontsize=9)
        ax.grid(False)
        for r in range(len(desc)):
            for c in range(3):
                n = d[(d.state == seed_aligned_col(al[seed], c))
                      & (d.descriptor == desc[r])]["n_samples"]
                if len(n) and n.iloc[0] < 10:
                    ax.text(c, r, "n<10", ha="center", va="center",
                            fontsize=5, color=INK)
    fig.colorbar(im, ax=axes, shrink=0.6, label="z-scored mean (descriptors "
                 "standardised within seed)")
    fig.suptitle("Figure 2 - Aligned regime profiles on backward-looking market "
                 "descriptors (no future information)", fontsize=10)
    fig.savefig(FIG / "fig02_regime_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig03_market_matrices():
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for j, cap in enumerate(CAPTURES):
        share = pd.read_parquet(P2 / "graph" /
                                f"market_attention_share_{cap}_seed42.parquet")
        enr = pd.read_parquet(P2 / "graph" /
                              f"market_attention_enrichment_{cap}_seed42.parquet")
        M = np.array([[share[f"M_{n}_{m}"].mean() for m in MARKETS]
                      for n in MARKETS])
        E = np.array([[enr[f"E_{n}_{m}"].mean() for m in MARKETS]
                      for n in MARKETS])
        for row, (mat, cmap, norm, ttl) in enumerate((
                (M, SEQ_BLUE, None, "raw attention share"),
                (E, DIV_BR, TwoSlopeNorm(1.0, 0.5, 1.5), "enrichment vs node-count baseline"))):
            ax = axes[row, j]
            im = ax.imshow(mat, cmap=cmap, norm=norm, vmin=None if norm else 0,
                           vmax=None if norm else max(M.max(), 1e-9))
            ax.set_xticks(range(3), MARKETS, fontsize=7)
            ax.set_yticks(range(3), MARKETS, fontsize=7)
            ax.set_xlabel("source market", fontsize=7)
            ax.set_ylabel("target market", fontsize=7)
            ax.set_title(f"{cap} - {ttl}", fontsize=8)
            ax.grid(False)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, f"{mat[a, b]:.3f}", ha="center", va="center",
                            fontsize=6,
                            color=INK if (row == 0 or norm is not None) else INK)
    fig.suptitle("Figure 3 - Market-to-market attention matrices per capture "
                 "(seed42; rows: target, columns: source; rows sum to 1)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_market_matrices.png")
    plt.close(fig)


def fig04_dynamic_dependency():
    pairs = [(n, m) for n in MARKETS for m in MARKETS if n != m]
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=True)
    for ax, cap in zip(axes.ravel(), CAPTURES):
        ens = {s: pd.read_parquet(P2 / "graph" /
                                  f"market_attention_enrichment_{cap}_seed{s}.parquet")
               for s in SEEDS}
        x = _dt(ens[42]["date"])
        for i, (n, m) in enumerate(pairs):
            v = np.mean([ens[s][f"E_{n}_{m}"].values for s in SEEDS], axis=0)
            ax.plot(x, v, color=SERIES[i], linewidth=1.4, label=f"{n}<={m}")
        ax.axhline(1.0, color=INK2, linewidth=1.0, linestyle="--")
        ax.set_title(f"{cap}", fontsize=9)
        ax.set_ylabel("enrichment", fontsize=8)
        if cap == CAPTURES[0]:
            ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=7)
    fig.suptitle("Figure 4 - Dynamic cross-market dependency: baseline-adjusted "
                 "enrichment over time (mean of 3 seeds; dashed line = node-count "
                 "baseline of 1.0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_dynamic_market_dependency.png")
    plt.close(fig)


def fig05_source_influence():
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=True)
    for ax, cap in zip(axes.ravel(), CAPTURES):
        ens = {s: pd.read_parquet(P2 / "graph" /
                                  f"market_attention_enrichment_{cap}_seed{s}.parquet")
               for s in SEEDS}
        x = _dt(ens[42]["date"])
        for i, m in enumerate(MARKETS):
            adj = np.mean([[ens[s][f"E_{n}_{m}"].values / 3.0 for n in MARKETS]
                           for s in SEEDS], axis=(0, 1))
            ax.plot(x, adj, color=SERIES[i], linewidth=1.4, label=m)
        ax.axhline(1.0 / 3.0, color=INK2, linewidth=1.0, linestyle="--")
        ax.set_title(f"{cap}", fontsize=9)
        ax.set_ylabel("baseline-adjusted\nsource influence", fontsize=8)
        if cap == CAPTURES[0]:
            ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=7)
    fig.suptitle("Figure 5 - Model-implied source influence by market "
                 "(mean over targets and seeds; dashed = equal influence)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_source_influence.png")
    plt.close(fig)


def fig06_concentration_turnover():
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    ax = axes[0]
    for i, cap in enumerate(CAPTURES):
        v = np.mean([pd.read_parquet(P2 / "graph" /
                                     f"graph_concentration_{cap}_seed{s}.parquet")
                     ["concentration_all"].values for s in SEEDS], axis=0)
        d = pd.read_parquet(P2 / "graph" /
                            f"graph_concentration_{cap}_seed42.parquet")
        ax.plot(_dt(d["date"]), v, color=SERIES[i], linewidth=1.4, label=cap)
    ax.set_ylabel("concentration\n(1 - normalised entropy)", fontsize=8)
    ax.set_title("Attention concentration per target (mean of 3 seeds)",
                 fontsize=9)
    ax.legend(loc="upper right", frameon=False, ncol=4, fontsize=7)
    ax = axes[1]
    for i, cap in enumerate(CAPTURES):
        d = pd.read_parquet(P2 / "graph" / f"graph_turnover_{cap}_seed42.parquet")
        v = np.mean([pd.read_parquet(P2 / "graph" /
                                     f"graph_turnover_{cap}_seed{s}.parquet")
                     ["market_graph_turnover"].values for s in SEEDS], axis=0)
        ax.plot(_dt(d["date"]), v, color=SERIES[i], linewidth=1.4, label=cap)
    ax.set_ylabel("market-graph turnover\n(0.5 * L1 of 3x3 matrix)",
                  fontsize=8)
    ax.set_title("Market-level graph turnover", fontsize=9)
    ax.set_xlabel("validation date")
    ax.legend(loc="upper right", frameon=False, ncol=4, fontsize=7)
    fig.suptitle("Figure 6 - Graph concentration and turnover (descriptive; "
                 "no perturbation performed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_concentration_turnover.png")
    plt.close(fig)


def fig07_gate_behavior():
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    specs = [("mean_gate", "mean gate over 64 channels"),
             ("gate_entropy_norm", "normalised gate entropy"),
             ("decisiveness_norm", "fusion decisiveness (2*|g-0.5|)")]
    for ax, (col, ttl) in zip(axes, specs):
        for i, s in enumerate(SEEDS):
            d = pd.read_parquet(P2 / "gate" / f"gate_sample_metrics_seed{s}.parquet")
            ax.plot(_dt(d["date"]), d[col], color=SERIES[i], linewidth=1.4,
                    label=f"seed{s}")
        if col == "mean_gate":
            ax.axhline(0.5, color=INK2, linewidth=1.0, linestyle="--")
            ax.set_ylim(0.4, 0.6)
        ax.set_ylabel(ttl, fontsize=8)
        ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=7)
    axes[0].set_title("Figure 7 - Fusion gate behaviour over the validation "
                      "dates", fontsize=10)
    axes[-1].set_xlabel("validation date")
    fig.tight_layout()
    fig.savefig(FIG / "fig07_gate_behavior.png")
    plt.close(fig)


def fig08_joint_relations():
    al, _ = _align_map()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    for i, s in enumerate(SEEDS):
        d = pd.read_parquet(P2 / "joint" / "phase2_sample_metrics_all_seeds.parquet")
        d = d[d.seed == s]
        means = [d[d.aligned_state == k]["mean_gate"].mean() for k in range(3)]
        ax.bar(np.arange(3) + (i - 1) * 0.26, means, width=0.25,
               color=SERIES[i], label=f"seed{s}")
    ax.set_xticks(range(3), [f"R{k+1}" for k in range(3)])
    ax.set_ylabel("mean gate", fontsize=8)
    ax.set_title("aligned regime vs mean gate", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[0, 1]
    for i, s in enumerate(SEEDS):
        d = pd.read_parquet(P2 / "joint" / "phase2_sample_metrics_all_seeds.parquet")
        d = d[d.seed == s]
        means = [d[d.aligned_state == k]["cross_avg"].mean() for k in range(3)]
        ax.bar(np.arange(3) + (i - 1) * 0.26, means, width=0.25,
               color=SERIES[i], label=f"seed{s}")
    ax.set_xticks(range(3), [f"R{k+1}" for k in range(3)])
    ax.set_ylabel("cross-market share", fontsize=8)
    ax.set_title("aligned regime vs cross-market share", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    d = pd.read_parquet(P2 / "joint" / "phase2_sample_metrics_all_seeds.parquet")
    ax = axes[1, 0]
    for i, s in enumerate(SEEDS):
        dd = d[d.seed == s]
        ax.scatter(dd["conc_avg"], dd["mean_gate"], s=8, color=SERIES[i],
                   label=f"seed{s}", alpha=0.8)
    ax.set_xlabel("graph concentration", fontsize=8)
    ax.set_ylabel("mean gate", fontsize=8)
    ax.set_title("concentration vs mean gate", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 1]
    for i, s in enumerate(SEEDS):
        dd = d[d.seed == s]
        ax.scatter(dd["state_entropy_final"], dd["err_5"], s=8,
                   color=SERIES[i], label=f"seed{s}", alpha=0.8)
    ax.set_xlabel("state entropy (final timestep)", fontsize=8)
    ax.set_ylabel("MAE@5d", fontsize=8)
    ax.set_title("latent-state uncertainty vs error", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Figure 8 - Joint descriptive relations (association only; no "
                 "causal direction is implied or tested)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_joint_relations.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (fig01_regime_timeline, fig02_regime_profiles, fig03_market_matrices,
               fig04_dynamic_dependency, fig05_source_influence,
               fig06_concentration_turnover, fig07_gate_behavior,
               fig08_joint_relations):
        fn()
        print("wrote", fn.__name__)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
