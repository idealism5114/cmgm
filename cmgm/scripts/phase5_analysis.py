#!/usr/bin/env python
"""Phase 5 - aggregate the fusion-control metrics and parameter counts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / "interpretability" / "phase5"
FC = P5 / "fusion_controls"
ROOT_EXP = ROOT / "experiments"
HORIZONS = (1, 5, 10, 20)


def latest(pat: str) -> Path:
    hits = sorted(ROOT.glob(pat))
    if not hits:
        raise FileNotFoundError(pat)
    return hits[-1]


def from_audited(run_dir: Path, label: str) -> dict:
    r = json.loads((run_dir / "result.json").read_text())
    met = json.loads((run_dir / "metrics.json").read_text())["val"]
    rec = dict(model=label, source="reused audited run",
               run_dir=str(run_dir.relative_to(ROOT)),
               checkpoint=r["checkpoint"], VAL_Huber=float(r["best_val_prediction_loss"]),
               best_epoch=int(r["best_epoch"]), training_seconds=float(r["training_seconds"]),
               parameter_count=int(r["model"]["parameter_count"]))
    for h in HORIZONS:
        rec[f"MAE@{h}"] = float(met[str(h)]["MAE"])
        rec[f"RMSE@{h}"] = float(met[str(h)]["RMSE"])
        rec[f"Hit@{h}"] = float(met[str(h)]["Hit"])
    return rec


def from_phase4(run_dir: Path, label: str) -> dict:
    r = json.loads((run_dir / "result.json").read_text())
    met = r["metrics"]["val"]
    return dict(model=label, source="Phase 4 run",
                run_dir=str(run_dir.relative_to(ROOT)), checkpoint=r["checkpoint"],
                VAL_Huber=float(r["best_val_prediction_loss"]),
                best_epoch=int(r["best_epoch"]), training_seconds=float(r["training_seconds"]),
                parameter_count=int(r["model"]["parameter_count"]),
                **{f"MAE@{h}": float(met[f"VAL_MAE_{h}"]) for h in HORIZONS},
                **{f"RMSE@{h}": float(met[f"VAL_RMSE_{h}"]) for h in HORIZONS},
                **{f"Hit@{h}": float(met[f"VAL_Hit{h}_percent"]) / 100.0 for h in HORIZONS})


def from_phase5(run_dir: Path, label: str) -> dict:
    d = from_phase4(run_dir, label)
    d["source"] = "Phase 5B new run"
    return d


def main() -> None:
    rows = [
        from_audited(latest("experiments/d0b_node_history_readout/switching_latent_balanced_readout_seed42_*"), "Original"),
        from_audited(latest("experiments/d0b_node_history_readout/switching_latent_balanced_concat_fusion_seed42_*"), "GlobalConcatFusion"),
        from_phase4(latest("experiments/phase4_ablation/switching_latent_fixed_mean_fusion_seed42_*"), "S1-FixedMeanFusion"),
    ]
    for label in ("ConcatLinear64", "GatedConcat128", "MeanConcat128"):
        rows.append(from_phase5(latest(f"experiments/phase5b_controls/{label}_seed42_*"), label))
    df = pd.DataFrame(rows)
    FC.mkdir(parents=True, exist_ok=True)
    df.to_csv(FC / "fusion_control_metrics.csv", index=False)

    base = float(df[df.model == "Original"].VAL_Huber.iloc[0])
    df["delta_vs_original"] = df.VAL_Huber - base
    df["improvement_pct"] = 100.0 * (base - df.VAL_Huber) / base
    df["fusion_width"] = [64, 64, 64, 64, 128, 128]
    df["has_dynamic_gate"] = [True, False, False, False, True, False]
    df["additive_mixing"] = [True, False, False, False, False, False]
    df["parameter_delta_vs_original"] = df.parameter_count - int(
        df[df.model == "Original"].parameter_count.iloc[0])
    df.to_csv(FC / "fusion_control_metrics.csv", index=False)
    df[["model", "parameter_count", "parameter_delta_vs_original", "source"]].to_csv(
        FC / "fusion_control_parameter_counts.csv", index=False)
    pd.DataFrame([dict(model=r["model"], MAE_1=r["MAE@1"], MAE_5=r["MAE@5"],
                       MAE_10=r["MAE@10"], MAE_20=r["MAE@20"], RMSE_5=r["RMSE@5"],
                       Hit_5=r["Hit@5"], VAL_Huber=r["VAL_Huber"],
                       best_epoch=r["best_epoch"], training_seconds=r["training_seconds"])
                  for r in df.to_dict("records")]).to_csv(
        FC / "fusion_control_training_history.csv", index=False)
    print(df[["model", "fusion_width", "has_dynamic_gate", "additive_mixing",
              "VAL_Huber", "improvement_pct", "parameter_count"]].round(8).to_string(index=False))


if __name__ == "__main__":
    main()
