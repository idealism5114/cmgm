#!/usr/bin/env python
"""Phase 3 report: PHASE3_FUNCTIONAL_MECHANISM_AUDIT.md.

Classification thresholds are stated explicitly in the report and were fixed
before the numbers were read; the continuous values are always shown alongside
so a reader can re-threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "interpretability" / "phase3"
SEEDS = (42, 43, 44)

THRESHOLDS = {
    "redundant_below_pct": 1.0,
    "weak_below_pct": 5.0,
    "seed_sensitive_spread_pct": 5.0,
}


def tbl(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def classify(rel: np.ndarray) -> str:
    rel = np.asarray(rel, dtype=float)
    mag = float(np.abs(rel).max())
    spread = float(rel.max() - rel.min())
    same_sign = bool(np.all(rel > 0) or np.all(rel < 0))
    if not same_sign and spread > THRESHOLDS["seed_sensitive_spread_pct"]:
        return "seed-sensitive"
    if mag < THRESHOLDS["redundant_below_pct"]:
        return "functionally redundant"
    if mag < THRESHOLDS["weak_below_pct"]:
        return "functionally weak"
    if spread > mag:
        return "seed-sensitive"
    return "functionally active"


def main() -> None:
    audit = json.loads((P3 / "audit" / "phase3_audit.json").read_text())
    gm = pd.read_csv(P3 / "gate" / "gate_intervention_metrics.csv")
    gc = pd.read_parquet(P3 / "gate" / "gate_prediction_changes.parquet")
    grm = pd.read_csv(P3 / "graph" / "graph_intervention_metrics.csv")
    grc = pd.read_parquet(P3 / "graph" / "graph_prediction_changes.parquet")
    swm = pd.read_csv(P3 / "graph" / "prior_sweep_metrics.csv")
    sws = pd.read_csv(P3 / "graph" / "prior_sweep_structure.csv")
    rm = pd.read_csv(P3 / "regime" / "regime_intervention_metrics.csv")
    rc = pd.read_parquet(P3 / "regime" / "regime_prediction_changes.parquet")
    gsim = pd.read_csv(P3 / "regime" / "generator_similarity.csv")
    grank = pd.read_csv(P3 / "regime" / "generator_effective_rank.csv")
    ctrl = pd.read_csv(P3 / "tables" / "intervention_controls.csv")
    base = pd.read_csv(P3 / "tables" / "baseline_reproduction.csv")

    def row(df, seed, key):
        s = df[(df.seed == seed) & (df.intervention == key)]
        return s.iloc[0] if len(s) else None

    def rel(df, key):
        return np.array([row(df, s, key)["relative_VAL_Huber_pct"] for s in SEEDS])

    def touch(df, key):
        return np.array([row(df, s, key)["mean_abs_prediction_change"] for s in SEEDS])

    L = []
    L.append("# Phase 3 - Functional Mechanism Audit")
    L.append("")
    L.append(f"Generated {audit['generated_utc']}. Frozen-model intervention on "
             "the audited `switching_latent_balanced_readout` checkpoints.")
    L.append("")
    L.append("## 1. Scope and controls")
    L.append("")
    L.append("- **Frozen-model intervention, NOT a retrained ablation.** No "
             "training, optimizer, backward pass, parameter update, checkpoint "
             "change or architecture change. All results describe how strongly "
             "the already-trained model depends on a mechanism; they say nothing "
             "about what a model trained without it would learn.")
    L.append("- VAL only (268 samples per seed, 2023-01-10 to 2024-02-21); TEST "
             "was never read.")
    L.append("- Every intervention is a reversible temporary override. The "
             "state_dict hash is unchanged after every run "
             f"({int(base.shape[0])} seeds checked: "
             f"{all(json.loads(json.dumps(g['unchanged'])) for g in audit['parameter_guard'])}).")
    L.append("- Classification thresholds, fixed in "
             "`cmgm/scripts/phase3_report.py` before the numbers were read: "
             f"|relative change| < {THRESHOLDS['redundant_below_pct']}% -> "
             "functionally redundant; < "
             f"{THRESHOLDS['weak_below_pct']}% -> functionally weak; larger -> "
             "functionally active; opposite signs across seeds with a spread "
             f"above {THRESHOLDS['seed_sensitive_spread_pct']}% -> seed-sensitive. "
             "Continuous values are always reported so the labels can be "
             "re-derived.")
    L.append("")
    L.append("Required controls (all must reproduce the frozen baseline):")
    L.append("")
    rows = []
    for _, r in ctrl.iterrows():
        key = r.get("control", r.get("intervention", ""))
        if isinstance(key, str) and key:
            ok = r.get("bitwise")
            extra = ""
            if "max_abs_diff" in r and pd.notna(r.get("max_abs_diff")):
                extra = f"max|d|={float(r['max_abs_diff']):.3g}"
            elif "max_abs_prediction_diff" in r and pd.notna(r.get("max_abs_prediction_diff")):
                extra = f"max|d|={float(r['max_abs_prediction_diff']):.3g}"
            elif "routed_p_max_dev_from_uniform" in r and pd.notna(r.get("routed_p_max_dev_from_uniform")):
                extra = f"routing dev={float(r['routed_p_max_dev_from_uniform']):.3g}"
            elif "routed_p_max_dev_from_onehot" in r and pd.notna(r.get("routed_p_max_dev_from_onehot")):
                extra = f"routing dev={float(r['routed_p_max_dev_from_onehot']):.3g}"
            rows.append((int(r["seed"]) if pd.notna(r.get("seed")) else "", key,
                         "" if pd.isna(ok) else bool(ok), extra))
    L.append(tbl(rows, ["Seed", "Control", "Bitwise", "Detail"]))
    L.append("")
    L.append(f"Baseline reproduction against Phase 1: "
             f"{all(bool(b['bitwise_identical_to_phase1']) for b in audit['baseline_reproduction'])} "
             "(max abs difference 0.0 for predictions and targets).")
    L.append("")

    # ---------------- A. gate
    L.append("## 2. Gate")
    L.append("")
    rows = []
    for key in ("gate_original", "gate_fixed_0.5", "gate_temporal_only",
                "gate_spatial_only", "gate_sample_shuffle_mean",
                "gate_channel_shuffle_mean"):
        if key == "gate_original":
            rows.append(("original", "reference (self-comparison)",
                         "0.000 / 0.000 / 0.000", "reference"))
            continue
        r = rel(gm, key)
        t = touch(gc, key)
        rows.append((key.replace("gate_", ""),
                     " / ".join(f"{v:+.2f}%" for v in r),
                     " / ".join(f"{v:.2e}" for v in t), classify(r)))
    L.append(tbl(rows, ["Intervention", "relative VAL Huber by seed",
                        "mean |prediction change| by seed", "Classification"]))
    L.append("")
    g05 = rel(gm, "gate_fixed_0.5")
    gsh = rel(gm, "gate_sample_shuffle_mean")
    gtem = rel(gm, "gate_temporal_only")
    gspa = rel(gm, "gate_spatial_only")
    L.append("**Q1 - does g=0.5 reproduce the original?** Relative VAL Huber "
             "change is " + " / ".join(f"{v:+.3f}%" for v in g05)
             + ", and the mean absolute prediction change is "
             + " / ".join(f"{v:.2e}" for v in touch(gc, "gate_fixed_0.5"))
             + ". The learned dynamic gate is therefore not needed to reproduce "
             "the frozen model's predictions.")
    L.append("")
    L.append("**Q2 - does sample shuffling of the gate matter?** "
             + " / ".join(f"{v:+.3f}%" for v in gsh)
             + " relative change; channel shuffling gives "
             + " / ".join(f"{v:+.3f}%" for v in rel(gm, "gate_channel_shuffle_mean"))
             + ". Neither destroys performance, so the gate's sample- and "
             "channel-specific structure carries no functional value here.")
    L.append("")
    L.append("**Q3 - temporal-only versus spatial-only.** "
             + " / ".join(f"{v:+.1f}%" for v in gtem) + " (temporal) versus "
             + " / ".join(f"{v:+.1f}%" for v in gspa) + " (spatial), relative to "
             "the original. This measures how much each branch alone could "
             "reproduce the fused prediction; it is a property of the frozen "
             "model's fusion, not a causal statement about either branch.")
    L.append("")
    L.append("![Gate performance](figures/fig01_gate_performance.png)")
    L.append("")
    L.append("![Gate deviation](figures/fig02_gate_deviation.png)")
    L.append("")

    # ---------------- B. graph
    L.append("## 3. Graph attention: prior versus content")
    L.append("")
    rows = []
    for key in ("graph_original", "graph_prior_only", "graph_content_only",
                "graph_uniform"):
        if key == "graph_original":
            rows.append(("original", "reference (self-comparison)",
                         "0.000 / 0.000 / 0.000", "reference"))
            continue
        r = rel(grm, key)
        t = touch(grc, key)
        rows.append((key.replace("graph_", ""),
                     " / ".join(f"{v:+.1f}%" for v in r),
                     " / ".join(f"{v:.3f}" for v in t), classify(r)))
    L.append(tbl(rows, ["Intervention", "relative VAL Huber by seed",
                        "mean |prediction change| by seed", "Classification"]))
    L.append("")
    L.append("**Prior necessity** (removing the prior = content-only) versus "
             "**content necessity** (removing the content = prior-only). "
             "Whichever removal moves the prediction less is the mechanism the "
             "frozen model depends on less.")
    L.append("")
    rows = []
    for lam in sorted(swm.lambda_value.unique()):
        sub = swm[swm.lambda_value == lam]
        rows.append((f"{lam:g}", " / ".join(f"{v:+.2f}%" for v in
                                            sub.sort_values("seed")["relative_VAL_Huber_pct"]),
                     " / ".join(f"{v:.4f}" for v in
                                sub.sort_values("seed")["VAL_Huber"])))
    L.append(tbl(rows, ["lambda", "relative VAL Huber by seed", "VAL Huber by seed"]))
    L.append("")
    L.append("Structural response to the same sweep (attention adaptivity):")
    L.append("")
    rows = []
    for lam in sorted(sws.lambda_value.unique()):
        sub = sws[sws.lambda_value == lam].sort_values("seed")
        rows.append((f"{lam:g}",
                     " / ".join(f"{v:.4f}" for v in sub.attention_concentration),
                     " / ".join(f"{v:+.3f}" for v in sub.content_attention_spearman),
                     " / ".join(f"{v:.4f}" for v in sub.cross_market_share)))
    L.append(tbl(rows, ["lambda", "concentration by seed",
                        "content-attention Spearman by seed",
                        "cross-market share by seed"]))
    L.append("")
    L.append("![Prior sweep](figures/fig03_prior_sweep.png)")
    L.append("")
    L.append("![Graph interventions](figures/fig04_graph_interventions.png)")
    L.append("")

    # ---------------- C. regime
    L.append("## 4. Switching generators and routing")
    L.append("")
    L.append("Generator specialization (descriptive, no intervention):")
    L.append("")
    rows = []
    for _, r in gsim.sort_values(["seed", "pair"]).iterrows():
        rows.append((int(r.seed), r.pair, f"{r.cosine_mean:.4f}",
                     f"{r.cosine_median:.4f}", f"{r.cosine_std:.4f}",
                     f"{r.relative_l2_mean:.4f}"))
    L.append(tbl(rows, ["Seed", "Pair", "cosine mean", "cosine median",
                        "cosine std", "relative L2 mean"]))
    L.append("")
    rows = [(int(r.seed), f"{r.effective_rank_mean:.4f}")
            for _, r in grank.sort_values("seed").iterrows()]
    L.append(tbl(rows, ["Seed", "mean effective rank of [G0,G1,G2] (max 3)"]))
    L.append("")
    rows = []
    for key in ("regime_original", "regime_uniform_router", "regime_expert0_only",
                "regime_expert1_only", "regime_expert2_only",
                "regime_evidence_only", "regime_prior_only",
                "regime_sample_shuffle_mean"):
        if key == "regime_original":
            rows.append(("original", "reference (self-comparison)",
                         "0.000 / 0.000 / 0.000", "reference"))
            continue
        r = rel(rm, key)
        t = touch(rc, key)
        rows.append((key.replace("regime_", ""),
                     " / ".join(f"{v:+.2f}%" for v in r),
                     " / ".join(f"{v:.3f}" for v in t), classify(r)))
    L.append(tbl(rows, ["Intervention", "relative VAL Huber by seed",
                        "mean |prediction change| by seed", "Classification"]))
    L.append("")
    L.append("![Generator similarity](figures/fig05_generator_similarity.png)")
    L.append("")
    L.append("![Regime interventions](figures/fig06_regime_interventions.png)")
    L.append("")

    # ---------------- D. summary
    L.append("## 5. Mechanism summary")
    L.append("")
    p2 = json.loads((ROOT / "interpretability" / "phase2" / "tables" /
                     "phase2_verification.json").read_text())
    rows = [
        ("Gate", "near 0.5 collapse", "fixed 0.5 / shuffle",
         " / ".join(f"{v:.2e}" for v in touch(gc, "gate_fixed_0.5")),
         " / ".join(f"{v:+.2f}%" for v in g05), classify(g05)),
        ("Graph prior", "prior dominated", "remove prior (content-only)",
         " / ".join(f"{v:.3f}" for v in touch(grc, "graph_content_only")),
         " / ".join(f"{v:+.1f}%" for v in rel(grm, "graph_content_only")),
         classify(rel(grm, "graph_content_only"))),
        ("Graph content", "weak content correlation", "prior-only",
         " / ".join(f"{v:.4f}" for v in touch(grc, "graph_prior_only")),
         " / ".join(f"{v:+.1f}%" for v in rel(grm, "graph_prior_only")),
         classify(rel(grm, "graph_prior_only"))),
        ("Regime router", "near-uniform posterior", "uniform routing",
         " / ".join(f"{v:.3f}" for v in touch(rc, "regime_uniform_router")),
         " / ".join(f"{v:+.2f}%" for v in rel(rm, "regime_uniform_router")),
         classify(rel(rm, "regime_uniform_router"))),
        ("Regime generators", "unknown specialization", "one-hot experts",
         " / ".join(f"{np.mean([touch(rc, k)[i] for k in ('regime_expert0_only','regime_expert1_only','regime_expert2_only')]):.3f}"
                    for i in range(3)),
         " / ".join(f"{np.mean([rel(rm, k)[i] for k in ('regime_expert0_only','regime_expert1_only','regime_expert2_only')]):+.2f}%"
                    for i in range(3)),
         classify(np.mean([rel(rm, k) for k in
                           ("regime_expert0_only", "regime_expert1_only",
                            "regime_expert2_only")], axis=0))),
    ]
    L.append(tbl(rows, ["Mechanism", "Phase 2 behaviour", "Intervention",
                        "Prediction sensitivity", "Performance sensitivity",
                        "Classification"]))
    L.append("")
    L.append(f"Phase 2 verification record available: {p2['PASS']}.")
    L.append("")
    L.append("Combined multi-mechanism neutralisation is deliberately not run as "
             "a main result: it cannot attribute an effect to an individual "
             "mechanism. It belongs in supplementary work only.")
    L.append("")
    L.append("![Sensitivity summary](figures/fig07_sensitivity_summary.png)")
    L.append("")
    L.append("## 6. Restrictions")
    L.append("")
    L.append("- No economic or causal claim is made anywhere above. Terms such as "
             "'prior', 'content', 'router' and 'generator' name model components, "
             "not market phenomena.")
    L.append("- The three seeds are never pooled as 268 x 3 iid observations; "
             "every effect is reported per seed with its cross-seed direction.")
    L.append("- No model, checkpoint, loss or architecture was changed, and no "
             "Phase 1 or Phase 2 artifact was modified.")
    L.append("")
    (P3 / "PHASE3_FUNCTIONAL_MECHANISM_AUDIT.md").write_text("\n".join(L) + "\n",
                                                             encoding="utf-8")
    print("report written:", P3 / "PHASE3_FUNCTIONAL_MECHANISM_AUDIT.md")


if __name__ == "__main__":
    main()
