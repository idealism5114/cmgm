#!/usr/bin/env python
"""Phase 4 human-readable report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P4 = ROOT / "interpretability" / "phase4"
ORDER = ["Original", "S1-FixedMeanFusion", "S2-NoGraphPrior", "S3-UniformRouter",
         "S4-SimplifiedAll", "GlobalConcatFusion"]


def tbl(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _classify(improvement_pct: float, floor_pct: float) -> str:
    """improvement_pct > 0 means the candidate has LOWER error than the reference."""
    if abs(improvement_pct) <= floor_pct:
        return "performance-preserving (within the reproducibility floor)"
    if improvement_pct > 0:
        return f"improved by {improvement_pct:.2f}%"
    return f"cost {abs(improvement_pct):.2f}%"


def main() -> None:
    runs = pd.read_csv(P4 / "tables" / "phase4_runs_long.csv")
    ms = pd.read_csv(P4 / "tables" / "phase4_summary_mean_std.csv")
    paired = pd.read_csv(P4 / "tables" / "paired_differences.csv")
    params = pd.read_csv(P4 / "tables" / "parameter_summary.csv")
    nec = pd.read_csv(P4 / "tables" / "mechanism_necessity.csv")
    tab_audit = json.loads((P4 / "audit" / "phase4_tables_audit.json").read_text())
    floor = tab_audit.get("reproducibility_floor") or {}
    base_huber = float(runs[runs.model_label == "Original"].VAL_Huber.iloc[0])
    floor_pct = (100.0 * float(floor["best_val_huber_abs_diff"]) / base_huber
                 if floor.get("best_val_huber_abs_diff") else 0.0)
    models = [m for m in ORDER if m in set(runs.model_label)]

    def m_of(m, k, col="mean"):
        r = ms[(ms.model == m) & (ms.metric == k)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    def pr(c, r, k):
        q = paired[(paired.candidate == c) & (paired.reference == r) & (paired.metric == k)]
        return q.iloc[0] if len(q) else None

    L = []
    L.append("# Phase 4 - Retrained Simplification Ablation")
    L.append("")
    L.append(f"Generated {tab_audit['generated_utc']}.")
    L.append("")
    L.append("## 1. Scope")
    L.append("")
    L.append("- **Retrained structural ablation.** Unlike Phases 1-3, parameters "
             "change during these runs because each variant is fitted from "
             "scratch. A frozen-model intervention and a retrained ablation "
             "answer different questions and are never conflated below.")
    L.append("- VAL only, **a single seed (42)**, no TEST. TEST_READ = false. Seed 42 is fixed by decision, not chosen after seeing any result, and because n = 1 no seed-level variability can be estimated and no robustness claim is made.")
    L.append("- The audited protocol is reused unchanged: same data, split, "
             "features, targets, optimizer, schedule, batch size, epoch limit, "
             "early stopping, selection objective (prediction-only four-horizon "
             "VAL Huber), metric code, seeds and determinism settings.")
    L.append("- No post-hoc model search: the four variants were pre-registered "
             "before any result was seen, and no variant, coefficient or "
             "hyper-parameter was added or tuned afterwards.")
    L.append(f"- New runs: {tab_audit['n_new_runs']} (target 4: four variants at seed 42). Reused audited "
             f"runs: {tab_audit['n_reused_runs']} (Original and "
             "GlobalConcatFusion are never retrained here).")
    if tab_audit["models_missing"]:
        L.append(f"- **Missing models:** {tab_audit['models_missing']}.")
    L.append("")
    L.append("## 2. Why Phase 4 was run")
    L.append("")
    L.append("Phase 3 intervened on the frozen checkpoint and found three "
             "mechanisms whose removal barely moves the prediction: the learned "
             "fusion gate, the adaptive graph prior, and the learned regime "
             "router. That establishes *frozen functional redundancy*, which "
             "does not by itself imply that the mechanisms are unnecessary when "
             "training from scratch. Phase 4 tests the training-time question.")
    L.append("")
    L.append("## 3. Variants")
    L.append("")
    rows = []
    for m in models:
        if m in ("Original", "GlobalConcatFusion"):
            rows.append((m, "reference (reused audited run)", "-", int(params[params.model == m].total_parameters.iloc[0])))
            continue
        desc = {
            "S1-FixedMeanFusion": "fused = 0.5*t + 0.5*s; gate_fc removed",
            "S2-NoGraphPrior": "e_ij = q.k/sqrt(d); adaptive prior and its learner removed",
            "S3-UniformRouter": "p_t = (1/3,1/3,1/3) every step; evidence and transition parameters removed",
            "S4-SimplifiedAll": "S1 + S2 + S3 together",
        }.get(m, "")
        rows.append((m, desc, "new run", int(params[params.model == m].total_parameters.iloc[0])))
    L.append(tbl(rows, ["Model", "Simplification", "Training", "Parameters"]))
    L.append("")
    L.append(tbl([(r.model, int(r.total_parameters), f"{int(r.delta_vs_baseline):+d}",
                   f"{r.relative_param_change_pct:+.2f}%")
                  for r in params.itertuples()],
                 ["Model", "Params", "Delta vs baseline", "% change"]))
    L.append("")
    L.append("Every variant constructs the complete native D0B model first and "
             "only then deletes the replaced modules, so all shared parameters "
             "keep the same seeded initialization as the baseline and the "
             "deleted parameters never consume RNG. The initialization audit "
             "confirms this per run.")
    L.append("")

    L.append("## 4. Per-seed results")
    L.append("")
    rows = []
    for r in runs.sort_values(["model_label", "seed"]).itertuples():
        rows.append((r.model_label, r.seed, f"{r.VAL_Huber:.8f}", f"{r.VAL_MAE_1:.6f}",
                     f"{r.VAL_MAE_5:.6f}", f"{r.VAL_MAE_10:.6f}", f"{r.VAL_MAE_20:.6f}",
                     f"{r.VAL_RMSE_5:.6f}", f"{r.VAL_Hit5_percent:.2f}",
                     int(r.best_epoch), f"{r.training_seconds:.0f}",
                     int(r.parameter_count)))
    L.append(tbl(rows, ["Model", "Seed", "Huber", "MAE1", "MAE5", "MAE10", "MAE20",
                        "RMSE5", "Hit5%", "Best ep", "Sec", "Params"]))
    L.append("")
    L.append("## 5. Mean over models (single seed; sample SD not estimable)")
    L.append("")
    rows = [(m, k, int(ms[(ms.model == m) & (ms.metric == k)].n.iloc[0]),
             f"{m_of(m, k):.8f}" if not np.isnan(m_of(m, k)) else "-",
             f"{m_of(m, k, 'sample_std'):.3e}" if not np.isnan(m_of(m, k, "sample_std")) else "-")
            for m in models for k in ("VAL_Huber", "VAL_MAE_1", "VAL_MAE_5",
                                      "VAL_MAE_10", "VAL_MAE_20", "VAL_RMSE_5")]
    L.append(tbl([(m, k, n, mean, sd if n > 1 else "n/a (n=1)")
                  for m, k, n, mean, sd in rows],
                 ["Model", "Metric", "n", "Mean", "Sample SD"]))
    L.append("")
    L.append("## 6. Paired comparisons")
    L.append("")
    rows = []
    for r in paired[paired.metric == "VAL_Huber"].itertuples():
        rows.append((f"{r.candidate} vs {r.reference}", "VAL_Huber", r.improved_seeds,
                     f"{r.mean_paired_delta:+.8f}", f"{r.mean_seed_improvement_pct:+.2f}%",
                     f"{r.improvement_of_model_means_pct:+.2f}%"))
    L.append(tbl(rows, ["Comparison", "Metric", "Improved seeds", "Mean paired delta",
                        "Mean seed improvement", "Improvement of model means"]))
    L.append("")
    L.append("Positive improvement means the candidate has *lower* error. Each "
             "comparison is the paired difference at seed 42 against the seed-42 "
             "baseline, so `improved seeds` can only read 1/1 or 0/1.")
    L.append("")
    all_h = [r for r in paired.itertuples() if r.metric != "VAL_Huber"]
    if all_h:
        L.append("Same comparison across the horizon metrics:")
        L.append("")
        L.append(tbl([(f"{r.candidate} vs {r.reference}", r.metric, r.improved_seeds,
                       f"{r.mean_paired_delta:+.8f}", f"{r.improvement_of_model_means_pct:+.2f}%")
                      for r in all_h],
                     ["Comparison", "Metric", "Improved seeds", "Mean paired delta",
                      "Improvement of model means"]))
        L.append("")

    L.append("## 7. Mechanism necessity")
    L.append("")
    for name, variant in (("learned gate", "S1-FixedMeanFusion"),
                          ("graph prior", "S2-NoGraphPrior"),
                          ("regime router", "S3-UniformRouter"),
                          ("all three jointly", "S4-SimplifiedAll")):
        row = pr(variant, "Original", "VAL_Huber")
        if row is None:
            L.append(f"- **{name} ({variant}): pending.**")
            continue
        imp = float(row.improvement_of_model_means_pct)
        L.append(f"- **{name} ({variant}):** {row.improved_seeds} seeds improved, "
                 f"mean paired Huber delta {row.mean_paired_delta:+.8f} "
                 f"(candidate minus reference, so positive means worse), model-mean "
                 f"change {imp:+.2f}%: **{_classify(imp, floor_pct)}**. The "
                 f"same-architecture reproducibility floor is {floor_pct:.3f}%, and "
                 "with n = 1 this is a single-seed observation, not evidence of "
                 "seed robustness.")
    L.append("")
    L.append("Frozen functional redundancy does not automatically imply "
             "training-time redundancy: Phase 3 asks whether a mechanism still "
             "acts inside the current checkpoint, while Phase 4 asks whether the "
             "model can reach the same performance without it when trained from "
             "scratch. The two are separate claims and the table below keeps them "
             "separate.")
    L.append("")
    L.append(tbl([(r.mechanism_removed, r.retrained_variant, r.frozen_phase3_result,
                   r.retrained_result, r.interpretation) for r in nec.itertuples()],
                 ["Mechanism removed", "Retrained variant", "Frozen Phase 3 result",
                  "Retrained result", "Interpretation"]))
    L.append("")

    L.append("## 8. Fusion bottleneck question")
    L.append("")
    s1 = pr("S1-FixedMeanFusion", "Original", "VAL_Huber")
    g1 = pr("GlobalConcatFusion", "Original", "VAL_Huber")
    gs1 = pr("GlobalConcatFusion", "S1-FixedMeanFusion", "VAL_Huber")
    gs4 = pr("GlobalConcatFusion", "S4-SimplifiedAll", "VAL_Huber")
    L.append("GlobalConcatFusion keeps both branch representations at 128 "
             "dimensions before the head, whereas S1 still compresses to 64 "
             "dimensions after simply replacing the gate with a fixed 0.5. If "
             "concatenation clearly beats fixed mean fusion, the binding "
             "constraint is the early 64-dimensional compression rather than the "
             "dynamic gate itself.")
    L.append("")
    rows = []
    for label, row in (("GlobalConcatFusion vs Original", g1),
                       ("GlobalConcatFusion vs S1-FixedMeanFusion", gs1),
                       ("GlobalConcatFusion vs S4-SimplifiedAll", gs4),
                       ("S1-FixedMeanFusion vs Original", s1)):
        if row is None:
            rows.append((label, "pending", "-", "-"))
        else:
            rows.append((label, row.improved_seeds, f"{row.mean_paired_delta:+.8f}",
                         f"{row.improvement_of_model_means_pct:+.2f}%"))
    L.append(tbl(rows, ["Comparison", "Improved seeds", "Mean paired delta",
                        "Improvement of model means"]))
    L.append("")
    L.append(f"Read against the {floor_pct:.3f}% same-architecture reproducibility "
             "floor: only differences larger than that are interpretable.")
    L.append("")

    L.append("## 9. Optimization behaviour")
    L.append("")
    L.append(tbl([(m, int(ms[(ms.model == m) & (ms.metric == "best_epoch")].n.iloc[0]),
                   f"{m_of(m, 'best_epoch'):.1f}",
                   f"{m_of(m, 'best_epoch', 'sample_std'):.1f}",
                   f"{m_of(m, 'training_seconds'):.0f}")
                  for m in models],
                 ["Model", "n", "mean best epoch", "SD", "mean seconds"]))
    L.append("")
    L.append("Epoch counts and runtimes are descriptive only. Runtimes are not a "
             "controlled efficiency measurement: the host runs unrelated GPU work "
             "and the number of epochs differs by early stopping. A smaller epoch "
             "count is not evidence of greater efficiency.")
    L.append("")

    L.append("## 9b. Same-seed reproducibility floor")
    L.append("")
    try:
        floor = json.loads((P4 / "tables" / "reproducibility_floor.json").read_text())
        L.append("S1 shares its architecture with the already audited "
                 "`switching_latent_balanced_mean_fusion` variant: loading one "
                 "model's weights into the other gives **bitwise identical** "
                 "outputs. Training them separately from the same seed therefore "
                 "isolates pure numerical reproducibility, and it is not exact on "
                 "this host:")
        L.append("")
        L.append(f"- epoch 1 absolute difference in validation loss: "
                 f"{floor['epoch1_abs_diff']:.2e}")
        L.append(f"- maximum absolute difference across "
                 f"{max(floor['best_epoch_mine'], floor['best_epoch_audited'])} epochs: "
                 f"{floor['max_abs_val_loss_diff']:.2e}")
        L.append(f"- best VAL Huber: {floor['best_val_huber_mine']:.8f} (this run, "
                 f"epoch {floor['best_epoch_mine']}) vs "
                 f"{floor['best_val_huber_audited']:.8f} (audited, epoch "
                 f"{floor['best_epoch_audited']})")
        L.append("")
        L.append(f"- difference in the *selected* metric between these two "
                 f"same-architecture runs: "
                 f"{floor.get('best_val_huber_abs_diff', float('nan')):.2e} in VAL Huber")
        L.append("")
        L.append("Two runs of the *same* architecture at the same seed therefore "
                 "differ by up to ~1e-4 in validation loss, because cuDNN "
                 "determinism covers convolutions but not every cuBLAS reduction "
                 "path and the GPUs are shared. Differences between variants "
                 "smaller than roughly "
                 f"{floor.get('best_val_huber_abs_diff', 1e-5):.0e} in VAL Huber are "
                 "inside this floor and must not be read as mechanism effects.")
        L.append("")
    except Exception as exc:                                  # noqa: BLE001
        L.append(f"Reproducibility control unavailable: {exc}.")
        L.append("")
    L.append("## 10. Limitations")
    L.append("")
    for t in ("n = 1 seed (42) per variant. Seed-level variability cannot be "
              "estimated and no significance, robustness or generalisation claim "
              "is made; a result here is one observation, not a repeatable effect.",
              "Validation split only. No TEST evaluation was performed and no "
              "model or seed was selected on TEST.",
              "Runtime and epoch counts are uncontrolled and descriptive.",
              "No cross-period or out-of-sample generalisation claim is made.",
              "The seed42 historical baseline run did not record its software "
              "environment; that provenance remains unknown and is not imputed.",
              "The deletion-based variants alter parameter counts, so comparisons "
              "are between complete training configurations rather than "
              "parameter-matched models."):
        L.append(f"- {t}")
    L.append("")

    L.append("## 11. Recommendation")
    L.append("")
    rec = []
    for name, variant in (("learned fusion gate", "S1-FixedMeanFusion"),
                          ("adaptive graph prior", "S2-NoGraphPrior"),
                          ("learned regime router", "S3-UniformRouter"),
                          ("all three together", "S4-SimplifiedAll")):
        row = pr(variant, "Original", "VAL_Huber")
        if row is None:
            rec.append(f"- {name}: inconclusive (run pending).")
        else:
            imp = float(row.improvement_of_model_means_pct)
            kind = _classify(imp, floor_pct)
            if abs(imp) <= floor_pct:
                verdict = (f"removal is performance-preserving at seed 42 "
                           f"({imp:+.2f}%, inside the {floor_pct:.3f}% floor): "
                           f"supported by this single-seed validation evidence")
            elif imp > 0:
                verdict = f"removal improved performance at seed 42 ({imp:+.2f}%)"
            else:
                verdict = (f"removal cost {abs(imp):.2f}% at seed 42, above the "
                           f"{floor_pct:.3f}% floor: retain on current evidence")
            rec.append(f"- {name}: {kind}; {verdict}. Single seed, so no "
                       f"robustness claim.")
    L.extend(rec)
    L.append("")
    L.append("No decision here involves TEST, and no architecture was added or "
             "tuned after seeing these results.")
    L.append("")
    (P4 / "PHASE4_RETRAINED_SIMPLIFICATION_ABLATION.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")
    print("report written:", P4 / "PHASE4_RETRAINED_SIMPLIFICATION_ABLATION.md")


if __name__ == "__main__":
    main()
