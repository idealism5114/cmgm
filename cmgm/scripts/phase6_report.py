#!/usr/bin/env python
"""Phase 6 report and audit."""
from __future__ import annotations
import hashlib, json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "interpretability" / "phase6"
REPR = P6 / "representations"
FLOOR_ABS = 9.955838322639465e-07
FLOOR_PCT = 0.061
ORDER = ["Original", "GlobalConcatFusion", "ResidualFusion", "BranchPreservingFusion"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tbl(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    d = pd.read_csv(P6 / "tables" / "phase6_main_results.csv").set_index("model")
    pp = pd.read_csv(P6 / "tables" / "phase6_paired_comparisons.csv")
    st = pd.read_csv(REPR / "fusion_stage_diagnostics.csv")
    kn = pd.read_csv(REPR / "fusion_knn_retention.csv")
    ck = pd.read_csv(REPR / "fusion_cka.csv")
    pth = pd.read_csv(REPR / "fusion_path_descriptives.csv")
    meta = json.loads((REPR / "extraction_meta.json").read_text())

    def row(cand, ref):
        s = pp[(pp.candidate == cand) & (pp.reference == ref)]
        return s.iloc[0] if len(s) else None

    def line(cand, ref):
        r = row(cand, ref)
        if r is None:
            return f"{cand} vs {ref}: pending"
        return (f"{cand} vs {ref}: absolute delta {r.absolute_delta:+.8f} VAL Huber, "
                f"{r.relative_improvement_pct:+.3f}% relative, {r.times_floor:+.1f}x the "
                f"reproducibility floor, tier = {r.tier}")

    L = []
    L.append("# Phase 6 - Nonlinear Cross-Branch Fusion Optimization")
    L.append("")
    L.append(f"Generated {datetime.utcnow().isoformat()}Z.")
    L.append("")
    L.append("## 1. Motivation")
    L.append("")
    L.append("Phase 5 established three facts that motivate this experiment. First, "
             "the width hypothesis is contradicted: GlobalConcatFusion is not a "
             "128-dimensional model, it is `GELU(Linear(128,64)([s;t]))` feeding the "
             "same 64-input head as the Original, and the 128-dimensional control C2 "
             "was indistinguishable from the Original. Second, the Original's fusion "
             "is channel-wise, so output channel *j* can only combine *t_j* and "
             "*s_j* and can never mix channels across branches. Third, the only model "
             "that beat the Original was GCF, the only one that mixes channels "
             "*and* applies a nonlinearity; the purely linear channel-mixing control "
             "C1 was the worst model tested.")
    L.append("")
    L.append("Phase 6 therefore tests two pre-registered refinements of the GCF "
             "operator. It is a targeted architecture experiment, not an open search: "
             "no cross attention, no commodity-specific gate, no width sweep, no "
             "activation sweep and no graph, temporal or pooling change.")
    L.append("")
    L.append("## 2. Model definitions")
    L.append("")
    L.append("Fusion input convention, confirmed from the frozen source and recorded "
             "in `fusion_input_semantics.md`: all models consume the **branch-projected** "
             "representations `x_s = gcn_proj(h_s)`, `x_t = lstm_proj(h_t)`, "
             "concatenated as `c = [x_s; x_t] in R^128`. This is exactly what the "
             "audited GlobalConcatFusion uses.")
    L.append("")
    L.append("```")
    L.append("M1 GlobalConcatFusion (reused, NOT retrained here)")
    L.append("    f = GELU(W_c c + b_c)                                     -> R^64")
    L.append("")
    L.append("M2 ResidualFusion")
    L.append("    n = GELU(W_1 c + b_1)")
    L.append("    r = W_r c + b_r")
    L.append("    f = LayerNorm(n + r)                                      -> R^64")
    L.append("")
    L.append("M3 BranchPreservingFusion")
    L.append("    s' = W_s x_s + b_s")
    L.append("    t' = W_t x_t + b_t")
    L.append("    c' = GELU(W_c c + b_c)")
    L.append("    f  = LayerNorm(W_o [s'; t'; c'] + b_o)                    -> R^64")
    L.append("```")
    L.append("")
    L.append("All three feed the unchanged prediction head "
             "`64 -> 64 -> ReLU -> Dropout -> H*24`. No gate, dropout, second hidden "
             "layer or attention is added. M2's residual path is a **learned linear "
             "residual projection**, not an identity shortcut, because 128 to 64 "
             "cannot be an identity. M3 deliberately carries no gate, since the "
             "experiment tests branch identity plus cross-branch interaction, not a "
             "new routing mechanism.")
    L.append("")
    L.append("## 3. Audit and protocol")
    L.append("")
    L.append("- seed 42 only, n = 1. No robustness claim is made anywhere.")
    L.append("- VAL only. TEST_READ = false; no TEST path was opened.")
    L.append("- 2 new runs (ResidualFusion, BranchPreservingFusion) and 2 reused "
             "audited references (Original, GlobalConcatFusion).")
    L.append("- The audited protocol is reused unchanged: same TRAIN/VAL, targets, "
             "feature and sample order, batch size, optimizer, learning rate, "
             "scheduler, epoch limit, patience, early stopping, metric code, "
             "determinism settings and seed. Checkpoint selection is the "
             "prediction-only four-horizon VAL Huber; VAL5 never selects.")
    L.append(f"- Numerical reference: the Phase 4 same-seed reproducibility floor of "
             f"{FLOOR_ABS:.3e} VAL Huber, about {FLOOR_PCT}%.")
    L.append("- Source drift from earlier external edits is recorded but not "
             "reverted, and none of the drifted-in variants are used here.")
    L.append("")
    L.append("## 4. Main results")
    L.append("")
    rows = []
    for m in ORDER:
        if m not in d.index:
            continue
        r = d.loc[m]
        rows.append((m, r.fusion, f"{r.VAL_Huber:.8f}", f"{r['MAE@1']:.6f}",
                     f"{r['MAE@5']:.6f}", f"{r['MAE@10']:.6f}", f"{r['MAE@20']:.6f}",
                     f"{r['RMSE@5']:.6f}", f"{100*r['Hit@5']:.2f}",
                     int(r.best_epoch), int(r.parameter_count), r.source))
    L.append(tbl(rows, ["Model", "Fusion", "Huber", "MAE1", "MAE5", "MAE10", "MAE20",
                        "RMSE5", "Hit5%", "Best ep", "Params", "Source"]))
    L.append("")
    L.append("## 5. Paired comparisons")
    L.append("")
    for c, r in (("GlobalConcatFusion", "Original"), ("ResidualFusion", "Original"),
                 ("BranchPreservingFusion", "Original"),
                 ("ResidualFusion", "GlobalConcatFusion"),
                 ("BranchPreservingFusion", "GlobalConcatFusion")):
        L.append(f"- {line(c, r)}")
    L.append("")
    L.append(f"Decision tiers follow the pre-registered rule: below {FLOOR_PCT}% is "
             f"within the observed reproducibility floor, {FLOOR_PCT}-0.2% is a small "
             f"seed42-only difference, and above 0.2% is a clearer seed42 effect "
             f"relative to that floor. None of these is a significance statement.")
    L.append("")
    L.append("## 6. ResidualFusion")
    L.append("")
    L.append(f"Question: does a learned linear residual projection alongside the "
             f"nonlinear interaction improve on GCF? Result: {line('ResidualFusion', 'GlobalConcatFusion')}. "
             f"ResidualFusion carries {int(d.loc['ResidualFusion','parameter_count']) - int(d.loc['GlobalConcatFusion','parameter_count']):+d} "
             f"parameters relative to GCF, so any improvement reflects both the fusion "
             f"structure and the additional fusion capacity, while a shortfall means "
             f"the extra capacity alone did not help.")
    L.append("")
    res = pth[pth.model == "ResidualFusion"]
    if len(res):
        for q in ("norm_ratio_nonlinear_over_residual", "cka_nonlinear_vs_residual",
                  "cosine_nonlinear_vs_residual"):
            v = res[res.quantity == q]
            if len(v):
                L.append(f"- {q}: {float(v.value.iloc[0]):.4f}")
        L.append("")
        L.append("These describe the relative scale and similarity of the two paths "
                 "only. No path ablation was performed, so no contribution share is "
                 "claimed.")
        L.append("")
    L.append("## 7. BranchPreservingFusion")
    L.append("")
    L.append(f"Question: does keeping spatial-specific, temporal-specific and "
             f"interaction paths improve on GCF? Result: "
             f"{line('BranchPreservingFusion', 'GlobalConcatFusion')}. Parameter delta "
             f"vs GCF: {int(d.loc['BranchPreservingFusion','parameter_count']) - int(d.loc['GlobalConcatFusion','parameter_count']):+d}.")
    L.append("")
    bp = pth[pth.model == "BranchPreservingFusion"]
    if len(bp):
        rows = []
        for tag in ("spatial_specific", "temporal_specific", "interaction"):
            g = bp[bp.quantity.str.startswith(f"l2_{tag}")]
            if len(g):
                vals = {q.rsplit("_", 1)[1]: float(v) for q, v in zip(g.quantity, g.value)}
                rows.append((tag, f"{vals.get('mean', float('nan')):.4f}",
                             f"{vals.get('median', float('nan')):.4f}",
                             f"{vals.get('std', float('nan')):.4f}"))
        L.append(tbl(rows, ["Path", "mean L2", "median L2", "std L2"]))
        L.append("")
        L.append("These are representation scales, not contribution shares.")
        L.append("")
    L.append("## 8. Comparison with GCF")
    L.append("")
    L.append("The question is whether simple GCF is already sufficient or whether a "
             "more structured nonlinear fusion helps. The answer is read directly "
             "from the two comparisons against GCF in section 5.")
    L.append("")
    L.append("## 9. Representation diagnostics")
    L.append("")
    rows = []
    for m in ORDER:
        s = st[(st.model == m) & (st.stage == "fusion_output")]
        k = kn[kn.model == m]
        c = ck[ck.model == m]
        if not len(s):
            continue
        rows.append((m, f"{float(s.participation_ratio.iloc[0]):.4f}",
                     f"{float(k.retention.iloc[0]):.4f}" if len(k) else "-",
                     f"{float(c.cka_input_vs_output.iloc[0]):.4f}" if len(c) else "-",
                     f"{float(c.geometry_spearman.iloc[0]):.4f}" if len(c) else "-",
                     f"{float(s['mean'].iloc[0]):.5f}"))
    L.append(tbl(rows, ["Model", "effective dim of fusion output", "kNN retention k=10",
                        "CKA input vs output", "geometry Spearman", "dynamics"]))
    L.append("")
    L.append("Auxiliary context only. Phase 5 showed that geometry preservation does "
             "not rank models by performance, so these numbers are not used to order "
             "the architectures.")
    L.append("")
    L.append("## 10. Limitations")
    L.append("")
    for t in ("n = 1 (seed 42). No robustness, significance or generalisation claim "
              "is made; each difference is reported in multiples of the observed "
              "same-seed reproducibility floor.",
              "This is an architecture comparison, not a parameter-matched study: "
              "ResidualFusion and BranchPreservingFusion carry more fusion parameters "
              "than GCF.",
              "VAL only; TEST was never evaluated.",
              "Runtimes are descriptive and not an efficiency comparison, because the "
              "host runs unrelated GPU work and early stopping changes epoch counts.",
              "Frozen model source files had already been modified by another party "
              "before this phase; the drift is recorded in the audit and none of the "
              "drifted-in variants are used."):
        L.append(f"- {t}")
    L.append("")
    L.append("## 11. Recommendation")
    L.append("")
    g_res = row("ResidualFusion", "GlobalConcatFusion")
    g_bp = row("BranchPreservingFusion", "GlobalConcatFusion")
    rec = []
    for label, r in (("ResidualFusion", g_res), ("BranchPreservingFusion", g_bp)):
        if r is None:
            rec.append(f"- {label}: pending.")
        elif "within" in r.tier:
            rec.append(f"- {label}: inconclusive at seed42 (within the floor).")
        elif r.relative_improvement_pct > 0:
            rec.append(f"- {label}: improves on GCF by "
                       f"{r.relative_improvement_pct:.3f}% ({r.times_floor:+.1f}x floor). "
                       f"Candidate to advance to seed43/44 replication at your "
                       f"decision; not started here.")
        else:
            rec.append(f"- {label}: did not improve on GCF "
                       f"({r.relative_improvement_pct:+.3f}%, {r.times_floor:+.1f}x "
                       f"floor). Keep GCF on this evidence.")
    L.extend(rec)
    L.append("")
    L.append("No seed was extended, no fourth architecture was designed, and no "
             "LayerNorm, width, gate or activation was tuned after seeing these results.")
    L.append("")
    L.append("![VAL Huber](figures/fig01_val_huber.png)")
    L.append("")
    L.append("![Horizon MAE](figures/fig02_horizon_mae.png)")
    L.append("")
    L.append("![Improvement](figures/fig03_improvement.png)")
    L.append("")
    L.append("![Params vs Huber](figures/fig04_params_vs_huber.png)")
    L.append("")
    L.append("![Representation diagnostics](figures/fig05_representation_diagnostics.png)")
    L.append("")
    (P6 / "PHASE6_NONLINEAR_FUSION_OPTIMIZATION.md").write_text("\n".join(L) + "\n",
                                                               encoding="utf-8")

    new_runs = []
    for d_ in sorted((ROOT / "experiments" / "phase6_fusion").glob("*_seed*")):
        a = json.loads((d_ / "audit.json").read_text())
        new_runs.append(dict(variant=a["variant"], seed=a["seed"],
                             run_dir=str(d_.relative_to(ROOT)),
                             checkpoint=a["checkpoint"],
                             checkpoint_sha256=a["checkpoint_sha256"],
                             best_epoch=a["best_epoch"],
                             checkpoint_is_argmin=a["checkpoint_is_argmin"],
                             best_val_huber=a["best_val_huber"],
                             parameter_count=a["parameter_count"],
                             history_sha256=a["history_sha256"]))
    aud = dict(phase=6, generated_utc=datetime.utcnow().isoformat() + "Z",
               scope="targeted nonlinear cross-branch fusion architecture experiment",
               seed=42, n=1, TEST_READ=False, TRAINING=True,
               NEW_RUNS=len(new_runs), REUSED_RUNS=2, new_runs=new_runs,
               reused=[{"model": "Original", "run": str(d.loc["Original", "run_dir"])},
                       {"model": "GlobalConcatFusion",
                        "run": str(d.loc["GlobalConcatFusion", "run_dir"])}],
               model_definitions={m: str(d.loc[m, "fusion"]) for m in ORDER if m in d.index},
               parameter_counts={m: int(d.loc[m, "parameter_count"]) for m in ORDER if m in d.index},
               protocol=dict(optimizer="Adam", lr=1e-4, weight_decay=1e-5, batch_size=64,
                             max_epochs=200, patience=10,
                             selection="prediction-only four-horizon VAL Huber",
                             train_shuffle=False),
               reproducibility_floor=dict(val_huber_abs=FLOOR_ABS, pct=FLOOR_PCT),
               source_hashes={f: sha(ROOT / f) for f in
                              ("cmgm/models/phase6_variants.py",
                               "cmgm/models/hetero_mixhop_model.py",
                               "cmgm/models/switching_latent_transformer.py",
                               "cmgm/scripts/phase6_fusion.py",
                               "cmgm/training/train.py")},
               figures=sorted(p.name for p in (P6 / "figures").glob("*.png")),
               output_hashes={str(p.relative_to(ROOT)): sha(p) for p in sorted(P6.rglob("*"))
                              if p.is_file() and "audit" not in p.parts})
    aud["PASS"] = bool(len(new_runs) == 2
                       and all(r["checkpoint_is_argmin"] for r in new_runs)
                       and not aud["TEST_READ"] and len(aud["figures"]) == 5)
    (P6 / "audit").mkdir(parents=True, exist_ok=True)
    (P6 / "audit" / "phase6_audit.json").write_text(
        json.dumps(aud, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report + audit written; PASS =", aud["PASS"])


if __name__ == "__main__":
    main()
