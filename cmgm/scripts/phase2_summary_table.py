#!/usr/bin/env python
"""Phase 2 cross-seed summary table (report section 6 / deliverable 51)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "interpretability" / "phase2"
SEEDS = (42, 43, 44)
CAPTURES = ("L1H1", "L1H2", "L2H1", "L2H2")


def main() -> None:
    rows = []

    def add(component, metric, per_seed, consistency):
        rows.append(dict(component=component, metric=metric,
                         **{f"seed{s}": per_seed[s] for s in SEEDS},
                         cross_seed_consistency=consistency))

    reg = {s: pd.read_parquet(P2 / "regime" / f"regime_sample_metrics_seed{s}.parquet")
           for s in SEEDS}
    add("Regime", "state entropy (final)",
        {s: float(reg[s]["state_entropy_final"].mean()) for s in SEEDS},
        "see per-seed values")
    add("Regime", "max state occupancy",
        {s: float(max((reg[s].argmax_state == k).mean() for k in range(3)))
         for s in SEEDS}, "degenerate in every seed")
    rc = pd.read_csv(P2 / "regime" / "regime_cross_seed_consistency.csv")
    add("Regime", "hard-state ARI (pairwise)",
        {s: None for s in SEEDS},
        f"mean {rc[rc.metric == 'adjusted_rand_index'].value.mean():+.3f}")
    add("Regime", "soft JSD (aligned, mean)",
        {s: None for s in SEEDS},
        f"mean {rc[rc.metric == 'jsd_mean'].value.mean():.4f}")

    for cap in CAPTURES:
        within, cross, conc, turn = {}, {}, {}, {}
        for s in SEEDS:
            sh = pd.read_parquet(P2 / "graph" / f"market_attention_share_{cap}_seed{s}.parquet")
            M = np.array([[sh[f"M_{n}_{m}"].mean() for m in ("stock", "bond", "commodity")]
                          for n in ("stock", "bond", "commodity")])
            within[s] = float(np.mean([M[i, i] for i in range(3)]))
            cross[s] = 1.0 - within[s]
            cc = pd.read_parquet(P2 / "graph" / f"graph_concentration_{cap}_seed{s}.parquet")
            conc[s] = float(cc["concentration_all"].mean())
            tv = pd.read_parquet(P2 / "graph" / f"graph_turnover_{cap}_seed{s}.parquet")
            turn[s] = float(tv["market_graph_turnover"].mean())
        gc = pd.read_csv(P2 / "graph" / "graph_cross_seed_consistency.csv")
        gcs = gc[(gc.capture == cap) & (gc.kind == "market_matrix")
                 & (gc.metric == "cosine_mean")]
        add("Graph", f"within-market share ({cap})", within, "n/a")
        add("Graph", f"cross-market share ({cap})", cross, "n/a")
        add("Graph", f"concentration ({cap})", conc, "n/a")
        add("Graph", f"turnover ({cap})", turn, "n/a")
        add("Graph", f"3x3 matrix cosine ({cap})",
            {s: None for s in SEEDS},
            f"pairwise mean {gcs.value.mean():.4f}")

    g = {s: pd.read_parquet(P2 / "gate" / f"gate_sample_metrics_seed{s}.parquet")
         for s in SEEDS}
    add("Gate", "mean gate", {s: float(g[s]["mean_gate"].mean()) for s in SEEDS},
        "collapsed near 0.5")
    add("Gate", "gate entropy", {s: float(g[s]["gate_entropy_norm"].mean()) for s in SEEDS},
        "near maximum 1.0")
    add("Gate", "decisiveness", {s: float(g[s]["decisiveness_norm"].mean()) for s in SEEDS},
        "near 0 (max possible 1.0)")
    gcns = pd.read_csv(P2 / "gate" / "gate_cross_seed_consistency.csv")
    sub = gcns[(gcns.kind == "summary_stat") & (gcns.metric == "mean_gate")]
    add("Gate", "mean-gate cross-seed similarity",
        {s: None for s in SEEDS}, f"pairwise Pearson mean {sub.pearson.mean():.4f}")

    df = pd.DataFrame(rows)
    (P2 / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(P2 / "tables" / "phase2_cross_seed_summary.csv", index=False)

    hdr = ["Component", "Metric", "Seed42", "Seed43", "Seed44",
           "Cross-seed consistency"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        cells = []
        for s in SEEDS:
            v = r[f"seed{s}"]
            cells.append("" if v is None or (isinstance(v, float) and np.isnan(v))
                         else f"{v:.4f}")
        out.append("| " + " | ".join([r["component"], r["metric"], *cells,
                                      r["cross_seed_consistency"]]) + " |")
    md = "\n".join(out)

    p = P2 / "PHASE2_INTERPRETABILITY_REPORT.md"
    txt = p.read_text()
    anchor = "## 5c. A single validation date"
    assert anchor in txt
    block = ("## 5b. Cross-seed summary table\n\n"
             "Values are means over the 268 validation samples unless stated. "
             "Blank cells mean the metric is a cross-seed quantity rather than a "
             "per-seed one.\n\n" + md + "\n\n")
    txt = txt.replace(anchor, block + anchor, 1)
    p.write_text(txt)
    print("cross-seed summary table written and inserted into the report")


if __name__ == "__main__":
    main()
