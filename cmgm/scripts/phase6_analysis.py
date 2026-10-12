#!/usr/bin/env python
"""Phase 6 - main table, paired comparisons and decision-rule tiers."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "interpretability" / "phase6"
EXP = ROOT / "experiments" / "phase6_fusion"
REF = ROOT / "experiments" / "d0b_node_history_readout"
HORIZONS = (1, 5, 10, 20)
FLOOR_ABS = 9.955838322639465e-07      # Phase 4 measured same-seed floor
FLOOR_PCT = 0.061
MODELS = ("Original", "GlobalConcatFusion", "ResidualFusion", "BranchPreservingFusion")
FUSION_LABEL = {
    "Original": "gated channel-wise sum",
    "GlobalConcatFusion": "concat -> linear -> GELU",
    "ResidualFusion": "nonlinear cross-channel + learned linear residual projection",
    "BranchPreservingFusion": "branch-specific + interaction (3-way)",
}
RUNS = {
    "Original": ("experiments/d0b_node_history_readout/switching_latent_balanced_readout_seed42_*", "reused audited"),
    "GlobalConcatFusion": ("experiments/d0b_node_history_readout/switching_latent_balanced_concat_fusion_seed42_*", "reused audited"),
    "ResidualFusion": ("experiments/phase6_fusion/residual_cross_branch_fusion_seed42_*", "Phase 6 new run"),
    "BranchPreservingFusion": ("experiments/phase6_fusion/branch_preserving_fusion_seed42_*", "Phase 6 new run"),
}


def latest(pat):
    hits = sorted(ROOT.glob(pat))
    if not hits:
        raise FileNotFoundError(pat)
    return hits[-1]


def load(name):
    d = latest(RUNS[name][0])
    r = json.loads((d / "result.json").read_text())
    met = json.loads((d / "metrics.json").read_text())["val"]
    rec = dict(model=name, source=RUNS[name][1], run_dir=str(d.relative_to(ROOT)),
               checkpoint=r["checkpoint"], VAL_Huber=float(r["best_val_prediction_loss"]),
               best_epoch=int(r["best_epoch"]), training_seconds=float(r["training_seconds"]),
               parameter_count=int(r["model"]["parameter_count"]))
    if str(HORIZONS[0]) in met:                       # audited nested format
        for h in HORIZONS:
            rec[f"MAE@{h}"] = float(met[str(h)]["MAE"])
            rec[f"RMSE@{h}"] = float(met[str(h)]["RMSE"])
            rec[f"Hit@{h}"] = float(met[str(h)]["Hit"])
    else:                                             # Phase 4/5/6 flat format
        for h in HORIZONS:
            rec[f"MAE@{h}"] = float(met[f"VAL_MAE_{h}"])
            rec[f"RMSE@{h}"] = float(met[f"VAL_RMSE_{h}"])
            rec[f"Hit@{h}"] = float(met[f"VAL_Hit{h}_percent"]) / 100.0
    return rec


def tier(pct: float) -> str:
    a = abs(pct)
    if a < FLOOR_PCT:
        return "within observed reproducibility floor"
    if a <= 0.2:
        return "small seed42-only difference"
    return "clearer seed42 effect relative to observed floor"


def main() -> None:
    rows = [load(m) for m in MODELS]
    df = pd.DataFrame(rows)
    P6.mkdir(parents=True, exist_ok=True)
    (P6 / "tables").mkdir(parents=True, exist_ok=True)
    base = float(df[df.model == "Original"].VAL_Huber.iloc[0])
    df["delta_vs_original"] = df.VAL_Huber - base
    df["improvement_vs_original_pct"] = 100.0 * (base - df.VAL_Huber) / base
    df["fusion"] = df.model.map(FUSION_LABEL)
    df.to_csv(P6 / "tables" / "phase6_main_results.csv", index=False)

    gcf = float(df[df.model == "GlobalConcatFusion"].VAL_Huber.iloc[0])
    pairs = []
    for cand in ("GlobalConcatFusion", "ResidualFusion", "BranchPreservingFusion"):
        c = float(df[df.model == cand].VAL_Huber.iloc[0])
        pairs.append(dict(comparison=f"{cand} vs Original", reference="Original",
                          candidate=cand, reference_huber=base, candidate_huber=c,
                          absolute_delta=c - base,
                          relative_improvement_pct=100.0 * (base - c) / base,
                          times_floor=(base - c) / FLOOR_ABS,
                          tier=tier(100.0 * (base - c) / base)))
    for cand in ("ResidualFusion", "BranchPreservingFusion"):
        c = float(df[df.model == cand].VAL_Huber.iloc[0])
        pairs.append(dict(comparison=f"{cand} vs GCF", reference="GlobalConcatFusion",
                          candidate=cand, reference_huber=gcf, candidate_huber=c,
                          absolute_delta=c - gcf,
                          relative_improvement_pct=100.0 * (gcf - c) / gcf,
                          times_floor=(gcf - c) / FLOOR_ABS,
                          tier=tier(100.0 * (gcf - c) / gcf)))
    pp = pd.DataFrame(pairs)
    pp.to_csv(P6 / "tables" / "phase6_paired_comparisons.csv", index=False)

    (P6 / "tables" / "phase6_parameter_counts.csv").write_text(
        df[["model", "parameter_count", "source"]].to_csv(index=False), encoding="utf-8")
    print(df[["model", "VAL_Huber", "MAE@1", "MAE@5", "MAE@10", "MAE@20", "RMSE@5",
              "Hit@5", "best_epoch", "parameter_count", "source"]].round(6).to_string(index=False))
    print()
    print(pp[["comparison", "absolute_delta", "relative_improvement_pct",
              "times_floor", "tier"]].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
