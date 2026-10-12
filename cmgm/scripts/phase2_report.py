#!/usr/bin/env python
"""Phase 2 human-readable report. Assembles PHASE2_INTERPRETABILITY_REPORT.md
from the tables written by cmgm.scripts.phase2_analysis.

    python -m cmgm.scripts.phase2_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "interpretability" / "phase2"
SEEDS = (42, 43, 44)
MARKETS = ("stock", "bond", "commodity")
CAPTURES = ("L1H1", "L1H2", "L2H1", "L2H2")
HORIZONS = (1, 5, 10, 20)


def rd(*p):
    return pd.read_csv(P2.joinpath(*p))


def pq(*p):
    return pd.read_parquet(P2.joinpath(*p))


def tbl(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    audit = json.loads((P2 / "audit" / "phase2_audit.json").read_text())
    al = json.loads((P2 / "regime" / "regime_alignment.json").read_text())
    rcons = rd("regime", "regime_cross_seed_consistency.csv")
    gcons = rd("graph", "graph_cross_seed_consistency.csv")
    hdiv = rd("graph", "graph_head_diversity.csv")
    pva = rd("graph", "prior_vs_attention.csv")
    selfatt = rd("graph", "self_attention_summary.csv")
    gconsg = rd("gate", "gate_cross_seed_consistency.csv")
    chan = rd("gate", "gate_channel_summary.csv")
    jo = pq("joint", "phase2_sample_metrics_all_seeds.parquet")
    rga = rd("joint", "regime_graph_association.csv")
    rgat = rd("joint", "regime_gate_association.csv")
    gga = rd("joint", "graph_gate_association.csv")
    iea = rd("joint", "interpretability_error_association.csv")
    outa = rd("joint", "outcome_association.csv")

    L = []
    L.append("# Phase 2 - Descriptive Interpretability and Cross-seed Consistency")
    L.append("")
    L.append(f"Generated from the frozen Phase 1 artifacts; analysis run "
             f"`{audit['generated_utc']}`.")
    L.append("")
    L.append("## 1. Scope")
    L.append("")
    L.append("- **VAL only** - the 268 validation samples, dates "
             f"{audit['date_range'][0]} to {audit['date_range'][1]}. TEST was never read.")
    L.append("- **Descriptive only** - no perturbation, masking, deletion, "
             "intervention, gradient attribution or branch ablation was performed.")
    L.append("- **No retraining**, no model change, no checkpoint change, no "
             "best-epoch reselection; Phase 1 artifacts were not modified.")
    L.append("- Regime states are called R1/R2/R3 only. They are aligned labels, "
             "not training labels, and carry no economic name.")
    L.append("- Every p-value below is descriptive and does not account for "
             "temporal dependence; effect sizes and cross-seed consistency are "
             "the primary evidence.")
    L.append("- Rules fixed before looking at results (see "
             "`audit/phase2_audit.json`): "
             + "; ".join(f"**{k}** = {v}" for k, v in audit["rules"].items()
                         if k in ("head_aggregation_rule", "capture_analysis_rule",
                                  "market_baseline_rule", "regime_alignment_rule",
                                  "effect_size_rule", "correlation_rule")))
    L.append("")

    # ---------------- 2. temporal regime
    L.append("## 2. Temporal regime interpretation")
    L.append("")
    rows = []
    for s in SEEDS:
        d = pq("regime", f"regime_sample_metrics_seed{s}.parquet")
        occ = [int((d.argmax_state == k).sum()) for k in range(3)]
        rows.append((f"seed{s}", occ[0], occ[1], occ[2],
                     f"{np.mean(occ)/len(d):.3f}",
                     f"{d.state_entropy_final.mean():.4f}",
                     f"{d.state_confidence_final.mean():.4f}",
                     f"{d.max_probability.mean():.4f}"))
    L.append(tbl(rows, ["Seed", "occ state0", "occ state1", "occ state2",
                        "max occupancy", "mean entropy", "mean confidence",
                        "mean max-prob"]))
    L.append("")
    L.append("**[FINDING] The hard-state occupancy is degenerate.** One state is "
             "almost never the argmax in each seed, while the mean posterior over "
             "states stays near-uniform (~0.33 each). The 3-state filter is "
             "therefore effectively two-state under `argmax` even though its soft "
             "mixture is not collapsed. Occupancy was never equalised by hand.")
    L.append("")
    trans = rd("regime", "regime_transition_matrices.csv")
    rows = []
    for s in SEEDS:
        emp = trans[(trans.seed == s) & (trans.kind == "empirical_argmax")
                    & (trans.from_state >= 0)]
        mod = trans[(trans.seed == s) & (trans.kind == "model_parameter")]
        sw = trans[(trans.seed == s) & (trans.kind == "empirical_argmax")
                   & (trans.from_state == -1)]
        selfemp = np.mean([emp[(emp.from_state == i) & (emp.to_state == i)]
                           .probability.iloc[0] for i in range(3)])
        selfmod = np.mean([mod[(mod.from_state == i) & (mod.to_state == i)]
                           .probability.iloc[0] for i in range(3)])
        rows.append((f"seed{s}", f"{selfmod:.3f}", f"{selfemp:.3f}",
                     f"{sw.probability.iloc[0]:.3f}"))
    L.append(tbl(rows, ["Seed", "model self-transition (parameter)",
                        "empirical self-transition (argmax)",
                        "switching frequency"]))
    L.append("")
    L.append("*Model transition parameter and empirical realised transition are "
             "different quantities and are never conflated.*")
    L.append("")
    dur = rd("regime", "regime_duration.csv")
    rows = []
    for s in SEEDS:
        for k in range(3):
            sel = dur[(dur.seed == s) & (dur.state == k) & dur.n_runs.notna()]
            if len(sel):
                r = sel.iloc[0]
                rows.append((f"seed{s}", f"state{k}", int(r.n_runs),
                             f"{r.mean_duration:.2f}", f"{r.median_duration:.1f}",
                             int(r.max_duration)))
    L.append(tbl(rows, ["Seed", "State", "runs", "mean duration",
                        "median duration", "max duration"]))
    L.append("")
    L.append("### Alignment and cross-seed consistency")
    L.append("")
    rows = []
    for s in SEEDS:
        d = al["degenerate_states"][str(s)]
        rows.append((f"seed{s}",
                     " -> ".join(f"state{k}=R{i+1}" for k, i in
                                 sorted(al["alignment"][str(s)].items())),
                     d["occupancy"], d["empty_states"]))
    L.append(tbl(rows, ["Seed", "seed state -> aligned label", "occupancy",
                        "empty (degenerate) states"]))
    L.append("")
    keep = ["adjusted_rand_index", "normalized_mutual_info",
            "aligned_hard_agreement", "jsd_mean", "jsd_median",
            "raw_hard_agreement_uncalibrated"]
    rows = []
    for p in ("42-43", "42-44", "43-44"):
        sub = rcons[rcons.pair == p]
        rows.append((p,) + tuple(f"{sub[sub.metric == k].value.iloc[0]:.4f}"
                                 if len(sub[sub.metric == k]) else "n/a"
                                 for k in keep))
    L.append(tbl(rows, ["Pair"] + [k.replace("_", " ") for k in keep]))
    L.append("")
    L.append("**[FINDING] Hard-state regimes do not replicate across seeds.** "
             "ARI is at or below zero for every seed pair and NMI is ~0.01 or "
             "less, so the argmax state sequence is essentially unrelated between "
             "seeds. Soft agreement is also very high in JSD terms only because "
             "both posteriors stay near-uniform; it does not indicate shared "
             "structure. The degenerate occupancy above makes any agreement "
             "figure hard to interpret.")
    L.append("")
    prof = rd("regime", "regime_profiles.csv")
    sep = (prof.dropna(subset=["effect_size_epsilon_squared"])
           .groupby("descriptor").effect_size_epsilon_squared.max()
           .sort_values(ascending=False))
    rows = [(k, f"{v:.3f}") for k, v in sep.items()]
    L.append("Descriptor separation between aligned states (largest "
             "epsilon-squared across seeds):")
    L.append("")
    L.append(tbl(rows, ["Descriptor", "max epsilon-squared"]))
    L.append("")
    L.append("![Regime timeline](figures/fig01_regime_timeline.png)")
    L.append("")
    L.append("![Regime profiles](figures/fig02_regime_profiles.png)")
    L.append("")

    # ---------------- 3. spatial
    L.append("## 3. Spatial dependency interpretation")
    L.append("")
    rows, diag_e = [], {}
    for cap in CAPTURES:
        Ms, Es = [], []
        for s in SEEDS:
            sh = pq("graph", f"market_attention_share_{cap}_seed{s}.parquet")
            en = pq("graph", f"market_attention_enrichment_{cap}_seed{s}.parquet")
            Ms.append(np.array([[sh[f"M_{n}_{m}"].mean() for m in MARKETS]
                                for n in MARKETS]))
            Es.append(np.array([[en[f"E_{n}_{m}"].mean() for m in MARKETS]
                                for n in MARKETS]))
        M, E = np.mean(Ms, axis=0), np.mean(Es, axis=0)
        diag_e[cap] = {m: E[i, i] for i, m in enumerate(MARKETS)}
        off = sorted(((E[n, m], f"{MARKETS[n]}<={MARKETS[m]}")
                      for n in range(3) for m in range(3) if n != m), reverse=True)
        rows.append((cap, " | ".join(f"{M[2, j]:.3f}" for j in range(3)),
                     " | ".join(f"{E[i, i]:.3f}" for i in range(3)),
                     f"{off[0][0]:.3f} ({off[0][1]})",
                     f"{np.mean([M[i, i] for i in range(3)]):.3f}"))
    pse = {sd: {c: {m: float(pq("graph", f"market_attention_enrichment_{c}_seed{sd}.parquet")
                             [f"E_{m}_{m}"].mean()) for m in MARKETS}
                for c in CAPTURES} for sd in SEEDS}
    for sd in SEEDS:
        for c in CAPTURES:
            pse[sd][c]["stock<=bond"] = float(
                pq("graph", f"market_attention_enrichment_{c}_seed{sd}.parquet")
                ["E_stock_bond"].mean())
    L.append(tbl(rows, ["Capture", "raw share, commodity target (S|B|C)",
                        "within-market enrichment (S|B|C)",
                        "largest off-diagonal enrichment",
                        "mean within-market share"]))
    L.append("")
    L.append("**[FINDING] Raw share is a node-count artefact; after correction "
             "stocks are under-attended and bonds are over-attended.** Stock "
             "holds 248 of 284 sources, so the raw stock share must be read "
             "against the 0.873 baseline, not compared with bond and commodity "
             "directly. Corrected, E[stock<=stock] is "
             + ", ".join(f"{diag_e[c]['stock']:.2f}" for c in CAPTURES)
             + " and E[commodity<=commodity] is "
             + ", ".join(f"{diag_e[c]['commodity']:.2f}" for c in CAPTURES)
             + ": stock sits at or below baseline, commodity is mixed "
             "(0.67-1.54), and only attention to bond sources is several times "
             "baseline. The effect rests on only 12 bond nodes out of 284.")
    L.append("")
    L.append("**[FINDING] That bond over-attention is a seeds-43/44 effect that "
             "seed42 barely shows.** Per seed, E[bond<=bond] is "
             + "; ".join(f"seed{sd}: " + ", ".join(
                 f"{pse[sd][c]['bond']:.2f} ({c})" for c in CAPTURES)
                 for sd in SEEDS)
             + ". E[stock<=bond] splits the same way: "
             + ", ".join(f"{np.mean([pse[sd][c]['stock<=bond'] for c in CAPTURES]):.2f}"
                         f" (seed{sd})" for sd in SEEDS)
             + ". This is the same seed42-versus-43/44 split found in graph "
             "concentration, and it is the largest cross-seed inconsistency in "
             "Phase 2. With only 12 bond nodes the estimate is also fragile, so "
             "the bond result should be treated as seed- and capture-dependent "
             "rather than as a stable property of the model.")
    L.append("")
    L.append("Rows of the raw matrix sum to 1 by construction (verified to "
             f"{gcons[gcons.metric == 'max_row_sum_deviation'].value.max():.1e} "
             "in every capture). The node-count baseline is "
             f"stock {audit['market_baseline']['stock']:.4f}, "
             f"bond {audit['market_baseline']['bond']:.4f}, "
             f"commodity {audit['market_baseline']['commodity']:.4f}.")
    L.append("")
    L.append("The uncorrected raw share would suggest stocks dominate simply "
             "because 248 of 284 nodes are stocks; the baseline correction is "
             "what makes the bond result visible at all.")
    L.append("")
    rows = []
    for cap in CAPTURES:
        sub = selfatt[selfatt.capture == cap]
        rows.append((cap,
                     f"{sub.loc[sub.market == 'stock', 'mean'].iloc[0]:.5f}",
                     f"{sub.loc[sub.market == 'commodity', 'mean'].iloc[0]:.5f}",
                     f"{sub.loc[sub.market == 'same_market_nonself', 'mean'].iloc[0]:.5f}",
                     f"{sub.loc[sub.market == 'cross_market', 'mean'].iloc[0]:.5f}"))
    L.append(tbl(rows, ["Capture", "self (stock)", "self (commodity)",
                        "same-market non-self", "cross-market"]))
    L.append("")
    L.append("Uniform attention would give 1/284 = 0.00352 per source. Self-edges "
             "sit at roughly half that because the adaptive adjacency has an "
             "exactly zero diagonal, so the structure prior contributes "
             "`log(1e-6)*0.5` to every self logit. Self-attention is suppressed, "
             "not removed.")
    L.append("")
    hd = hdiv.groupby(["seed"]).mean(numeric_only=True)
    rows = [(f"seed{s}", f"{hd.loc[s, 'mean_pairwise_head_jsd']:.3e}",
             f"{hd.loc[s, 'mean_pairwise_head_cosine']:.4f}",
             f"{hd.loc[s, 'mean_std_across_heads_of_market_share']:.2e}")
            for s in SEEDS]
    L.append(tbl(rows, ["Seed", "mean pairwise head JSD",
                        "mean pairwise head cosine",
                        "std across heads of market share"]))
    L.append("")
    L.append("**[FINDING] Graph attention is sample-dependent in seeds 43 and 44 "
             "but effectively frozen in seed42.** Averaged per-sample standard "
             "deviation of the head-mean attention is 1e-5 for seed42 versus "
             "5.8e-3 for seed43. Mean concentration (1 - normalised entropy) at "
             "L2H1 is 0.0353 with std 0.0004 for seed42, against 0.4853 "
             "(std 0.2950) for seed43 and 0.3821 (std 0.2283) for seed44. "
             "Seed42's attention barely responds to the input at all, so 'the "
             "graph is sample-adaptive' is not a seed-invariant statement.")
    L.append("")
    pv = pva[pva.kind == "spearman_prior_vs_attention"]
    pc = pva[pva.kind == "spearman_content_vs_attention"]
    L.append("**[FINDING] The graph attention is prior-dominated.** Spearman "
             f"between the structure prior and the final attention is "
             f"{pv.value.mean():.3f} on average across captures and seeds, while "
             f"the content logits correlate with final attention at "
             f"{pc.value.mean():+.4f}. Content logits exceed 0.01 in under 0.5% "
             "of entries. This is a structural association, not a causal "
             "decomposition: softmax is non-linear, so no share of the final "
             "attention is attributed to prior versus content.")
    L.append("")
    rows = []
    for cap in CAPTURES:
        sub = gcons[(gcons.capture == cap) & (gcons.kind == "market_matrix")
                    & (gcons.metric.isin(["pearson_mean", "cosine_mean", "l1_mean"]))]
        if len(sub):
            rows.append((cap,
                         f"{sub[sub.metric == 'pearson_mean'].value.mean():.4f}",
                         f"{sub[sub.metric == 'cosine_mean'].value.mean():.4f}",
                         f"{sub[sub.metric == 'l1_mean'].value.mean():.4f}"))
    L.append(tbl(rows, ["Capture", "mean pairwise Pearson (3x3 matrix)",
                        "mean cosine", "mean L1"]))
    L.append("")
    L.append("![Market matrices](figures/fig03_market_matrices.png)")
    L.append("")
    L.append("![Dynamic dependency](figures/fig04_dynamic_market_dependency.png)")
    L.append("")
    L.append("![Source influence](figures/fig05_source_influence.png)")
    L.append("")
    L.append("![Concentration and turnover](figures/fig06_concentration_turnover.png)")
    L.append("")

    # ---------------- 4. gate
    L.append("## 4. Fusion gate interpretation")
    L.append("")
    rows = []
    for s in SEEDS:
        d = pq("gate", f"gate_sample_metrics_seed{s}.parquet")
        rows.append((f"seed{s}", f"{d.mean_gate.mean():.4f}",
                     f"{d.mean_gate.std():.4f}", f"{d.gate_entropy_norm.mean():.4f}",
                     f"{d.decisiveness_norm.mean():.4f}",
                     f"{d['frac_gt_0.5'].mean():.3f}",
                     f"{d.temporal_leaning_channels.max():.0f}",
                     f"{d.spatial_leaning_channels.max():.0f}"))
    L.append(tbl(rows, ["Seed", "mean gate", "std", "gate entropy",
                        "decisiveness", "frac >0.5", "max temporal-leaning ch",
                        "max spatial-leaning ch"]))
    L.append("")
    lt = int(max(pq("gate", f"gate_sample_metrics_seed{sd}.parquet")
                   ["temporal_leaning_channels"].max() for sd in SEEDS))
    ls = int(max(pq("gate", f"gate_sample_metrics_seed{sd}.parquet")
                 ["spatial_leaning_channels"].max() for sd in SEEDS))
    L.append("**[FINDING] The gate is collapsed near 0.5.** Mean gate is within "
             "0.014 of 0.5, normalised entropy is ~0.996 of its maximum, and "
             "decisiveness is ~0.05-0.07 out of a possible 1.0. Across all 268 "
             f"validation samples and all three seeds, at most {lt} of 64 channels "
             f"ever exceed the 0.6 temporal-leaning threshold and at most {ls} "
             "fall below the 0.4 spatial-leaning threshold, and both occur only "
             "in seed43. The gate therefore carries almost no branch-preference "
             "signal, and this is the strongest negative result of Phase 2.")
    L.append("")
    dyn = chan.groupby("seed").mean(numeric_only=True)
    L.append(f"Channel-level variability is also small: mean per-channel gate "
             f"std averaged over seeds is "
             f"{dyn['std_gate'].mean():.5f} across 64 channels.")
    L.append("")
    L.append("Cross-seed gate consistency (permutation-insensitive summaries):")
    L.append("")
    sub = gconsg[gconsg.kind == "summary_stat"]
    rows = []
    for m in ("mean_gate", "gate_entropy_norm", "decisiveness_norm", "frac_gt_0.5"):
        s2 = sub[sub.metric == m]
        rows.append((m, f"{s2.pearson.mean():.4f}", f"{s2.spearman.mean():.4f}",
                     f"{s2.mae.mean():.6f}"))
    L.append(tbl(rows, ["Summary statistic", "mean Pearson across pairs",
                        "mean Spearman", "mean MAE"]))
    L.append("")
    L.append("Full 64-d vectors are compared only as a supplement and are "
             "labelled index-aligned, because latent channels are not guaranteed "
             "to have the same semantics across seeds.")
    L.append("")
    L.append("![Gate behaviour](figures/fig07_gate_behavior.png)")
    L.append("")

    # ---------------- 5. joint
    L.append("## 5. Joint association")
    L.append("")
    L.append("All statements in this section are **association, not causation**. "
             "Regime and graph are parallel explanation objects; nothing here "
             "implies that a regime changes the graph.")
    L.append("")
    strong = rga[(rga.epsilon_squared.notna()) & (rga.epsilon_squared > 0.06)]
    L.append(f"Regime x graph: {len(strong)} of {len(rga)} metric-seed "
             "combinations reach a conventional medium effect size "
             "(epsilon-squared > 0.06). Largest:")
    L.append("")
    rows = [(r.metric, f"seed{r.seed}", f"{r.epsilon_squared:.3f}",
             f"{r.R1_mean:.4f}", f"{r.R2_mean:.4f}", f"{r.R3_mean:.4f}")
            for r in strong.sort_values("epsilon_squared", ascending=False).head(8).itertuples()]
    L.append(tbl(rows, ["Graph metric", "Seed", "epsilon-squared", "R1", "R2",
                        "R3"]) if rows else "_(none)_")
    L.append("")
    strong2 = rgat[(rgat.epsilon_squared.notna()) & (rgat.epsilon_squared > 0.06)]
    L.append(f"Regime x gate: {len(strong2)} of {len(rgat)} combinations exceed "
             "epsilon-squared 0.06.")
    L.append("")
    gg = gga[gga.gate_metric.isin(["mean_gate", "gate_entropy_norm",
                                   "decisiveness_norm"])]
    L.append("Graph x gate (Spearman, per seed): strongest associations")
    L.append("")
    rows = [(r.graph_metric, r.gate_metric, f"seed{r.seed}", f"{r.spearman_rho:+.3f}")
            for r in gg.reindex(gg.spearman_rho.abs().sort_values(ascending=False).index)
            .head(8).itertuples()]
    L.append(tbl(rows, ["Graph metric", "Gate metric", "Seed", "Spearman rho"]))
    L.append("")
    L.append("Interpretability x prediction error (Spearman, per horizon, never "
             "pooled across seeds):")
    L.append("")
    ie = iea[iea.error_metric != "MAE_all_horizons"]
    rows = [(r.interpretability_metric, r.error_metric, f"seed{r.seed}",
             f"{r.spearman_rho:+.3f}")
            for r in ie.reindex(ie.spearman_rho.abs().sort_values(ascending=False).index)
            .head(10).itertuples()]
    L.append(tbl(rows, ["Interpretability metric", "Error metric", "Seed",
                        "Spearman rho"]))
    L.append("")
    L.append("### Outcome association")
    L.append("")
    L.append("Explicitly separated: regimes are **not** defined from future "
             "returns. The table below is descriptive only.")
    L.append("")
    rows = [(f"seed{r.seed}", r.aligned_state, int(r.n_samples),
             f"{r.mean_realised_return_5d:+.5f}", f"{r.mean_abs_error_5d:.5f}")
            for r in outa.itertuples()]
    L.append(tbl(rows, ["Seed", "Aligned state", "n", "mean realised 5d return",
                        "mean MAE@5d"]))
    L.append("")
    L.append("![Joint relations](figures/fig08_joint_relations.png)")
    L.append("")

    L.append("## 5c. A single validation date dominates the dynamic variation")
    L.append("")
    L.append("2023-05-24 is simultaneously the largest one-step change in gate "
             "entropy and in fusion decisiveness (seeds 43 and 44), in graph "
             "concentration, and in market-graph turnover, while seed42 shows no "
             "change at that date. The 20-day input window ending 2023-05-24 "
             "spans 2023-04-25 to 2023-05-25 and contains eight commodity "
             "(day, commodity) pairs moving more than 5% in one day, including "
             "crude oil -6.5% on 2023-05-04 and nickel -5.5% on 2023-05-11.")
    L.append("")
    L.append("The step is therefore consistent with an extreme-input event "
             "entering the lookback window, not with a change in the model. It "
             "matters for Phase 3: one date carries a large share of the "
             "observed dynamic variation, so any intervention study should "
             "exclude it or analyse it explicitly.")
    L.append("")
    L.append("## 6. Robust findings")
    L.append("")
    for t in [
        "**Graph attention is prior-dominated** (prior-vs-attention Spearman "
        "~0.83 in every seed and capture) and the raw market share is set by "
        "node count, not by market importance.",
        "**The gate is collapsed near 0.5** with near-maximal entropy in every "
        "seed. At most 1-2 of 64 channels ever cross the descriptive 0.6 / 0.4 "
        "leaning thresholds, and only in seed43.",
        "**Self-attention is suppressed to about half of uniform** because the "
        "adaptive adjacency has an exactly zero diagonal, in every capture and seed.",
        "**Hard-state occupancy is degenerate** in all three seeds, with one "
        "state almost never argmax.",
    ]:
        L.append(f"- {t}")
    L.append("")
    L.append("## 7. Unstable findings")
    L.append("")
    for t in [
        "**Regime hard-state sequences do not replicate across seeds** (ARI <= 0, "
        "NMI <= ~0.01). Any regime-conditioned statement in section 5 inherits "
        "this instability.",
        "**State alignment is unreliable.** It is degenerate in seed43 (state 1 "
        "is never the argmax, so its profile comes from 0 samples) and it pairs "
        "seed44's dominant state (177 of 268 samples) with a seed42 state that "
        "holds a single sample. The aligned label scheme should therefore not be "
        "trusted beyond giving the states a common ordering; "
        "`regime_alignment.json` records the cost matrices and the degenerate "
        "states.",
        "**Capture sensitivity is small on average but not zero in the tail**: "
        "captures differ by up to 1.0 in max absolute attention on rare one-hot "
        "entries while their mean absolute difference is ~2e-5.",
        "**Head sensitivity**: raw per-head attention contains rare one-hot "
        "entries (0.07% above 0.9); these average out of the head-mean used for "
        "the main analysis but are not absent.",
        "**Cross-market enrichment varies by two orders of magnitude between "
        "seeds** (E[bond<=bond] 1.01-1.44 in seed42 versus 2.77-10.58 in seeds "
        "43/44), so the spatial market structure is the least reproducible part "
        "of Phase 2.",
        "**Mean gate is anti-correlated across seeds** (pairwise Pearson "
        "-0.27 on the per-date mean gate), so even the summary gate trajectory "
        "does not reproduce across seeds.",
    ]:
        L.append(f"- {t}")
    L.append("")

    # ---------------- 8. Phase 3 candidates
    L.append("## 8. Phase 3 candidates")
    L.append("")
    L.append("Eligibility follows the pre-registered criteria, not preference:")
    L.append("")
    occs = {sd: [int((pq("regime", f"regime_sample_metrics_seed{sd}.parquet")
                        .argmax_state == k).sum()) for k in range(3)] for sd in SEEDS}
    L.append("- **Regime: NOT eligible as a 3-state object.** Hard-state "
             "occupancy is degenerate in every seed ("
             + "; ".join(f"seed{sd}: {occs[sd]}" for sd in SEEDS)
             + "), and cross-seed alignment is not stable, so a faithfulness "
             "test of a 3-state regime claim would not be interpretable. A "
             "2-state collapse would have to be re-established and re-audited "
             "first.")
    L.append("- **Graph: WEAKLY eligible.** The market-level structure is "
             "non-trivial only in the sense of being *at* the node-count "
             "baseline; the honest Phase 3 target is not 'which market matters' "
             "but 'does removing the prior domination change predictions', which "
             "is an intervention question. Cross-seed similarity is high for the "
             "matrix but the matrix is near-degenerate, so a faithfulness test "
             "would be testing a near-null object.")
    L.append("- **Gate: NOT eligible.** The gate is collapsed near 0.5 with "
             f"near-maximal entropy; at most {lt} of 64 channels cross the 0.6 "
             f"threshold and at most {ls} fall below 0.4, so there is little "
             "variation for an intervention to explain.")
    L.append("")
    L.append("Recommendation: before any Phase 3 intervention, re-examine why the "
             "content attention is dominated by the prior (prior_scale=0.5 "
             "against a near-flat adaptive adjacency) and why the gate is "
             "neutral, since both are properties of the frozen model rather than "
             "of the analysis.")
    L.append("")
    (P2 / "PHASE2_INTERPRETABILITY_REPORT.md").write_text("\n".join(L) + "\n",
                                                          encoding="utf-8")
    print("report written:", P2 / "PHASE2_INTERPRETABILITY_REPORT.md",
          f"({len(L)} lines)")


if __name__ == "__main__":
    main()
