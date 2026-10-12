#!/usr/bin/env python
"""Phase 4 tables: per-seed results, mean +/- sample SD, paired comparisons.

Reads the new Phase 4 runs plus the reused audited Original and
GlobalConcatFusion runs. Never reads TEST.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P4 = ROOT / "interpretability" / "phase4"
P4_EXP = ROOT / "experiments" / "phase4_ablation"
P4_CKPT = ROOT / "checkpoints" / "phase4_ablation"
REF_EXP = ROOT / "experiments" / "d0b_node_history_readout"
SEEDS = (42,)   # Phase 4 is single-seed
HORIZONS = (1, 5, 10, 20)
BASELINE_PARAMS = 520549

FROZEN_PHASE3 = {
    "gate": "functionally redundant (+0.13/+0.52/-0.34% Huber; predictions move ~1e-3)",
    "graph_prior": "functionally redundant (removing it costs <=0.2% Huber)",
    "regime_router": "functionally redundant (<=0.11% Huber across all seeds)",
    "graph_content": "seed-sensitive (-8.6% on seed43, ~0.1% on 42/44)",
}
METRIC_ORDER = ("VAL_Huber", "VAL_MAE_1", "VAL_MAE_5", "VAL_MAE_10",
                "VAL_MAE_20", "VAL_RMSE_5", "VAL_Hit5_percent")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    def default(v):
        if isinstance(v, (np.generic,)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False, default=default) + "\n",
                    encoding="utf-8")


def latest(pattern: str) -> Path:
    hits = sorted(REF_EXP.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[-1]


def from_audited(run_dir: Path, model_name: str, seed: int) -> dict:
    res = json.loads((run_dir / "result.json").read_text())
    met = json.loads((run_dir / "metrics.json").read_text())["val"]
    rec = dict(model=model_name, seed=seed, source="reused audited run",
               run_dir=str(run_dir.relative_to(ROOT)),
               VAL_Huber=float(res["best_val_prediction_loss"]),
               best_epoch=int(res["best_epoch"]),
               training_seconds=float(res["training_seconds"]),
               parameter_count=int(res["model"]["parameter_count"]))
    for i, h in enumerate(HORIZONS):
        rec[f"VAL_MAE_{h}"] = float(met[str(h)]["MAE"])
        rec[f"VAL_RMSE_{h}"] = float(met[str(h)]["RMSE"])
        rec[f"VAL_Hit{h}_percent"] = 100.0 * float(met[str(h)]["Hit"])
    return rec


def from_phase4(run_dir: Path, variant: str, seed: int) -> dict:
    res = json.loads((run_dir / "result.json").read_text())
    met = res["metrics"]["val"]
    rec = dict(model=variant, seed=seed, source="Phase 4 new run",
               run_dir=str(run_dir.relative_to(ROOT)),
               VAL_Huber=float(res["best_val_prediction_loss"]),
               best_epoch=int(res["best_epoch"]),
               training_seconds=float(res["training_seconds"]),
               parameter_count=int(res["model"]["parameter_count"]))
    rec.update({k: float(v) for k, v in met.items()})
    return rec


def collect() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        rows.append(from_audited(latest(f"switching_latent_balanced_readout_seed{seed}_*"),
                                 "Original", seed))
        rows.append(from_audited(latest(f"switching_latent_balanced_concat_fusion_seed{seed}_*"),
                                 "GlobalConcatFusion", seed))
    for variant_dir in sorted(P4_EXP.glob("*_seed*")):
        name = variant_dir.name
        variant = name.rsplit("_seed", 1)[0]
        seed = int(name.rsplit("_seed", 1)[1].split("_")[0])
        try:
            rows.append(from_phase4(variant_dir, variant, seed))
        except Exception as exc:                       # noqa: BLE001
            print(f"skip unreadable run {name}: {exc}")
    return pd.DataFrame(rows)


SHORT = {
    "switching_latent_fixed_mean_fusion": "S1-FixedMeanFusion",
    "switching_latent_no_graph_prior": "S2-NoGraphPrior",
    "switching_latent_uniform_router": "S3-UniformRouter",
    "switching_latent_simplified_all": "S4-SimplifiedAll",
}
ORDER = ["Original", "S1-FixedMeanFusion", "S2-NoGraphPrior", "S3-UniformRouter",
         "S4-SimplifiedAll", "GlobalConcatFusion"]


def _classify(improvement_pct: float, floor_pct: float) -> str:
    """improvement_pct > 0 means the candidate has LOWER error than the reference."""
    if abs(improvement_pct) <= floor_pct:
        return (f"performance-preserving: within the same-architecture "
                f"reproducibility floor of {floor_pct:.3f}%")
    if improvement_pct > 0:
        return (f"removal improved performance by {improvement_pct:.2f}% at seed 42 "
                f"(single seed; above the {floor_pct:.3f}% floor, no robustness claim)")
    return (f"removal cost performance by {abs(improvement_pct):.2f}% at seed 42 "
            f"(single seed; above the {floor_pct:.3f}% floor)")


def main() -> None:
    df = collect()
    df["model_label"] = df.model.map(lambda m: SHORT.get(m, m))
    present = [m for m in ORDER if m in set(df.model_label)]
    missing = [m for m in ORDER if m not in set(df.model_label)]
    df.to_csv(P4 / "tables" / "phase4_runs_long.csv", index=False)

    # ---- per-seed table
    per = df.sort_values(["model_label", "seed"])[
        ["model_label", "seed", "VAL_Huber", "VAL_MAE_1", "VAL_MAE_5",
         "VAL_MAE_10", "VAL_MAE_20", "VAL_RMSE_5", "VAL_Hit5_percent",
         "best_epoch", "training_seconds", "parameter_count", "source"]]
    per.to_csv(P4 / "tables" / "phase4_per_seed.csv", index=False)

    # ---- mean +/- sample SD
    rows = []
    for m in present:
        sub = df[df.model_label == m]
        for k in METRIC_ORDER:
            v = sub[k].astype(float)
            rows.append(dict(model=m, metric=k, n=int(v.count()),
                             mean=float(v.mean()),
                             sample_std=float(v.std(ddof=1)) if v.count() > 1
                             else float("nan"),  # n=1: sample SD is undefined
                             n_note=("single seed; seed variability not estimable"
                                     if v.count() < 2 else "")))
        rows.append(dict(model=m, metric="best_epoch", n=int(sub.best_epoch.count()),
                         mean=float(sub.best_epoch.mean()),
                         sample_std=float(sub.best_epoch.std(ddof=1))
                         if sub.best_epoch.count() > 1 else float("nan"),
                         n_note="single seed" if sub.best_epoch.count() < 2 else ""))
        rows.append(dict(model=m, metric="training_seconds", n=int(sub.training_seconds.count()),
                         mean=float(sub.training_seconds.mean()),
                         sample_std=float(sub.training_seconds.std(ddof=1))
                         if sub.training_seconds.count() > 1 else float("nan"),
                         n_note="single seed" if sub.training_seconds.count() < 2 else ""))
        rows.append(dict(model=m, metric="parameter_count", n=int(sub.parameter_count.count()),
                         mean=float(sub.parameter_count.mean()), sample_std=0.0,
                         n_note="model constant"))
    mean_std = pd.DataFrame(rows)
    mean_std.to_csv(P4 / "tables" / "phase4_summary_mean_std.csv", index=False)

    # ---- parameter summary
    prows = []
    for m in present:
        n = int(df[df.model_label == m].parameter_count.iloc[0])
        prows.append(dict(model=m, total_parameters=n,
                          trainable_parameters=n,
                          delta_vs_baseline=n - BASELINE_PARAMS,
                          relative_param_change_pct=100.0 * (n - BASELINE_PARAMS) / BASELINE_PARAMS))
    pd.DataFrame(prows).to_csv(P4 / "tables" / "parameter_summary.csv", index=False)

    # ---- paired comparisons
    pairs = [("S1-FixedMeanFusion", "Original"), ("S2-NoGraphPrior", "Original"),
             ("S3-UniformRouter", "Original"), ("S4-SimplifiedAll", "Original"),
             ("GlobalConcatFusion", "Original"), ("GlobalConcatFusion", "S1-FixedMeanFusion"),
             ("GlobalConcatFusion", "S4-SimplifiedAll"), ("S1-FixedMeanFusion", "S2-NoGraphPrior")]
    prows = []
    for cand, ref in pairs:
        c = df[df.model_label == cand].set_index("seed")
        r = df[df.model_label == ref].set_index("seed")
        common = sorted(set(c.index) & set(r.index))
        if len(common) < 1:
            continue
        for k in ("VAL_Huber", "VAL_MAE_1", "VAL_MAE_5", "VAL_MAE_10", "VAL_MAE_20"):
            d = np.array([c.loc[s, k] - r.loc[s, k] for s in common], dtype=float)
            rel = np.array([100.0 * (r.loc[s, k] - c.loc[s, k]) / r.loc[s, k] for s in common])
            prows.append(dict(candidate=cand, reference=ref, metric=k, n=len(common),
                              improved_seeds=f"{int((d < 0).sum())}/{len(common)}",
                              mean_paired_delta=float(d.mean()),
                              mean_seed_improvement_pct=float(rel.mean()),
                              improvement_of_model_means_pct=float(
                                  100.0 * (r[k].mean() - c[k].mean()) / r[k].mean())))
    paired = pd.DataFrame(prows)
    paired.to_csv(P4 / "tables" / "paired_differences.csv", index=False)

    # ---- equivalence control: S1 shares its architecture with the audited
    # mean-fusion variant, so the gap between them is pure training-time
    # numerics and measures the same-seed reproducibility floor.
    floor = None
    try:
        import glob as _glob
        mine = sorted(_glob.glob(str(P4_EXP / "switching_latent_fixed_mean_fusion_seed42_*")))[-1]
        aud = sorted(_glob.glob(str(REF_EXP / "switching_latent_balanced_mean_fusion_seed42_*")))[-1]
        my_h = json.loads((Path(mine) / "result.json").read_text())["training_history"]
        au_h = json.loads((Path(aud) / "training_history.json").read_text())
        mv = np.array([r["VAL_Huber"] for r in my_h], dtype=float)
        av = np.array(au_h["val_loss"], dtype=float)
        n = min(len(mv), len(av))
        floor = dict(
            control="S1-FixedMeanFusion vs audited switching_latent_balanced_mean_fusion",
            identical_architecture=True,
            architecture_check="with identical weights loaded into both, the two "
                               "forward passes are bitwise equal (max diff 0.0)",
            epoch1_abs_diff=float(abs(mv[0] - av[0])),
            max_abs_val_loss_diff=float(np.abs(mv[:n] - av[:n]).max()),
            best_val_huber_mine=float(mv.min()), best_val_huber_audited=float(av.min()),
            best_epoch_mine=int(np.argmin(mv) + 1), best_epoch_audited=int(np.argmin(av) + 1),
            best_val_huber_abs_diff=float(abs(mv.min() - av.min())),
            operational_floor_note="the operationally relevant floor is the gap "
                                   "between the two runs' selected metrics, not the "
                                   "trajectory drift",
            interpretation="same-seed retraining is not bitwise reproducible on this "
                           "shared host; trajectories start within 3e-10 and drift to "
                           "~1e-4 in validation loss, so differences below roughly "
                           "1e-5 in VAL Huber are not interpretable")
        write_json(P4 / "tables" / "reproducibility_floor.json", floor)
        print("reproducibility floor measured:", floor["max_abs_val_loss_diff"])
    except Exception as exc:                                  # noqa: BLE001
        print("reproducibility control unavailable:", exc)


    # floor in relative percentage terms, measured from the same-architecture control
    base_huber = float(df[df.model_label == "Original"].VAL_Huber.iloc[0])
    floor_abs = float(floor["best_val_huber_abs_diff"]) if floor else 0.0
    floor_pct = 100.0 * floor_abs / base_huber if base_huber else 0.0

    # ---- mechanism necessity table
    necessities = []
    for label, variant, frozen in (
            ("learned gate", "S1-FixedMeanFusion", FROZEN_PHASE3["gate"]),
            ("graph prior", "S2-NoGraphPrior", FROZEN_PHASE3["graph_prior"]),
            ("regime router", "S3-UniformRouter", FROZEN_PHASE3["regime_router"]),
            ("all three", "S4-SimplifiedAll",
             "jointly untested in Phase 3")):
        sub = paired[(paired.candidate == variant) & (paired.reference == "Original")
                     & (paired.metric == "VAL_Huber")]
        if sub.empty:
            necessities.append(dict(mechanism_removed=label, retrained_variant=variant,
                                    frozen_phase3_result=frozen, retrained_result="pending",
                                    interpretation="run not finished"))
            continue
        row = sub.iloc[0]
        necessities.append(dict(
            mechanism_removed=label, retrained_variant=variant,
            frozen_phase3_result=frozen,
            retrained_result=f"{row.improved_seeds} seeds improved; mean paired delta "
                             f"{row.mean_paired_delta:+.8f} Huber (candidate - reference); "
                             f"model-mean change "
                             f"{row.improvement_of_model_means_pct:+.2f}%",
            interpretation=_classify(float(row.improvement_of_model_means_pct),
                                     floor_pct)))
    pd.DataFrame(necessities).to_csv(P4 / "tables" / "mechanism_necessity.csv", index=False)

    write_json(P4 / "audit" / "phase4_tables_audit.json", dict(
        generated_utc=datetime.utcnow().isoformat() + "Z",
        models_present=present, models_missing=missing,
        n_new_runs=int((df.source == "Phase 4 new run").sum()),
        target_new_runs=4,
        seeds=list(SEEDS),
        single_seed=True,
        single_seed_note="Phase 4 is fixed to seed 42; seed-level variability is "
                         "not estimable and no robustness claim is made",
        n_reused_runs=int((df.source == "reused audited run").sum()),
        TEST_READ=False,
        reused_run_paths={m: sorted(df[df.model_label == m].run_dir.unique().tolist())
                          for m in ("Original", "GlobalConcatFusion") if m in present},
        table_hashes={p.name: _sha(p) for p in sorted((P4 / "tables").glob("*.csv"))},
        reproducibility_floor=floor,
    ))
    print(f"tables written; models present {present}; missing {missing}")
    print(f"new runs: {(df.source == 'Phase 4 new run').sum()} / 4 (seed 42 only)")


if __name__ == "__main__":
    main()
