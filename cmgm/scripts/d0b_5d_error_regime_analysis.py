"""Pure array/statistical analysis for the checkpoint-only error diagnostic."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cmgm.training.metric_standard import population_metrics

TOL = 1e-8
QUANTILES = {
    "target_magnitude": [0.25, 0.5, 0.75, 0.9, 0.95],
    "vol20": [0.25, 0.5, 0.75, 0.9], "vol5": [0.25, 0.5, 0.75, 0.9],
    **{k: [0.25, 0.75, 0.9] for k in ("dispersion_commodity", "dispersion_stock", "dispersion_bond",
        "trend_strength", "confidence", "entropy", "kl_surprise", "l1_surprise", "delta_z", "mean_delta_z5")},
    **{k: [0.5, 0.75, 0.9] for k in ("movement", "transition_score", "transition_max")},
}
EX_POST = {"target_magnitude", "target_tail", "reversal", "direction", "target_sign",
           "trend_x_reversal", "vol_x_reversal", "target_x_transition"}
SAMPLE_COLUMNS = set(QUANTILES) - {"target_magnitude", "trend_strength"}


def metrics(pred, target):
    if len(pred) == 0:
        return {key: None for key in ("MAE", "MSE", "RMSE", "Hit", "RMSE_squared_minus_MSE_abs")}
    m = population_metrics(pred, target)
    m["RMSE_squared_minus_MSE_abs"] = abs(m["RMSE"] ** 2 - m["MSE"])
    return m


def finite_corr(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return {"Pearson": None, "Spearman": None}
    return {"Pearson": float(np.corrcoef(a, b)[0, 1]), "Spearman": float(spearmanr(a, b).statistic)}


def percentile_labels(q):
    edges = [0] + [int(100 * x) for x in q] + [100]
    return [f"Q{lo}-Q{hi}" for lo, hi in zip(edges, edges[1:])]


def bin_values(values, cutpoints, labels):
    # Right-closed intervals: equality belongs to the lower bin, also if cuts tie.
    return np.asarray(labels, dtype=object)[np.searchsorted(cutpoints, values, side="left")]


def fit_thresholds(train, models):
    assert set(train["split"]) == {"train"}, "Thresholds must use TRAIN only"
    thresholds = {}
    for key, quantiles in QUANTILES.items():
        values = train.drop_duplicates("sample_index")[key] if key in SAMPLE_COLUMNS else train[key]
        thresholds[key] = {"quantiles": quantiles, "cuts": np.quantile(values, quantiles).tolist(),
            "labels": percentile_labels(quantiles), "unit": "forecast_origin" if key in SAMPLE_COLUMNS else "observation"}
    for model in models:
        for name, values, q in (
            ("prediction_magnitude_" + model, train[model].abs(), [.25, .5, .75, .9]),
            ("large_error_" + model, (train[model] - train.target).abs(), [.9, .95])):
            thresholds[name] = {"quantiles": q, "cuts": np.quantile(values, q).tolist(), "labels": percentile_labels(q), "unit": "observation"}
    thresholds["flat_past_p10"] = float(np.quantile(train.past5.abs(), .1))
    thresholds["near_zero_target_p10"] = float(np.quantile(train.target.abs(), .1))
    thresholds["source"] = "TRAIN ONLY; equal-to-cut observations in lower interval; tied cuts may create empty bins"
    return thresholds


def assign_groups(frame, thresholds, models):
    groups, categories = {}, {}
    for key in QUANTILES:
        spec = thresholds[key]
        groups[key] = bin_values(frame[key], spec["cuts"], spec["labels"])
        categories[key] = spec["labels"]
    for model in models:
        key = "prediction_magnitude_" + model
        spec = thresholds[key]
        groups[key] = bin_values(frame[model].abs(), spec["cuts"], spec["labels"])
        categories[key] = spec["labels"]
    zero = (frame.past5 == 0) | (frame.target == 0)
    flat = frame.past5.abs() < thresholds["flat_past_p10"]
    continuation = np.sign(frame.past5) == np.sign(frame.target)
    groups["reversal"] = np.select([zero, flat, continuation], ["Zero", "Flat/weak", "Continuation"], default="Reversal")
    categories["reversal"] = ["Continuation", "Reversal", "Flat/weak", "Zero"]
    groups["direction"] = np.select([zero, flat, continuation & (frame.past5 > 0), continuation,
                                     frame.past5 > 0],
        ["Zero", "Flat/weak", "Positive continuation", "Negative continuation", "Up -> Down"], default="Down -> Up")
    categories["direction"] = ["Positive continuation", "Negative continuation", "Up -> Down", "Down -> Up", "Flat/weak", "Zero"]
    groups["state"] = np.asarray([f"state {int(s)}" for s in frame.state])
    categories["state"] = ["state 0", "state 1", "state 2"]
    groups["target_sign"] = np.select([frame.target.abs() <= thresholds["near_zero_target_p10"], frame.target > 0],
                                         ["Near-zero", "Positive"], default="Negative")
    categories["target_sign"] = ["Positive", "Negative", "Near-zero"]
    large = frame.target_magnitude > thresholds["target_magnitude"]["cuts"][2]
    active = frame.transition_score > thresholds["transition_score"]["cuts"][1]
    high_vol = frame.vol20 > thresholds["vol20"]["cuts"][2]
    groups["target_x_transition"] = np.char.add(np.where(large, "large + ", "normal + "), np.where(active, "active", "stable"))
    categories["target_x_transition"] = [f"{t} + {s}" for t in ("normal", "large") for s in ("stable", "active")]
    groups["vol_x_reversal"] = np.array([("high-vol + " if h else "normal-vol + ") + r for h, r in zip(high_vol, groups["reversal"])])
    categories["vol_x_reversal"] = [v + r for v in ("normal-vol + ", "high-vol + ") for r in categories["reversal"]]
    groups["trend_x_reversal"] = np.array([str(t) + " + " + r for t, r in zip(groups["trend_strength"], groups["reversal"])])
    categories["trend_x_reversal"] = [t + " + " + r for t in categories["trend_strength"] for r in categories["reversal"]]
    for key, mask, labels in (("binary_target", large, ["normal", "large"]),
                             ("binary_transition", active, ["stable", "active"]),
                             ("binary_vol", high_vol, ["normal-vol", "high-vol"])):
        groups[key], categories[key] = np.where(mask, labels[1], labels[0]), labels
    for market in ("stock", "bond"):
        key = market + "_commodity_quadrant"
        groups[key] = np.array([f"{market}:{np.sign(a):.0f}/commodity:{np.sign(b):.0f}"
            for a, b in zip(frame["past5_" + market], frame.past5_commodity)])
        categories[key] = [f"{market}:{a}/commodity:{b}" for a in (-1, 0, 1) for b in (-1, 0, 1)]
    return groups, categories


def paired_win_stats(d, t, y):
    ad, at = np.abs(np.asarray(d)-y), np.abs(np.asarray(t)-y)
    difference = ad-at
    tie = np.abs(difference) < TOL
    dw, tw = (difference < 0) & ~tie, (difference > 0) & ~tie
    # Same tie set for squared-error comparison: magnitude squaring preserves rank.
    assert np.array_equal((ad**2 < at**2) & ~tie, dw)
    assert np.array_equal((ad**2 > at**2) & ~tie, tw)
    result = {"D0B_win": float(dw.mean()), "TempWeighted_win": float(tw.mean()), "Tie": float(tie.mean()),
        "MSE_D0B_win": float(dw.mean()), "MSE_TempWeighted_win": float(tw.mean()),
        "MSE_win_consistency": True}
    for model, gain in (("D0B", -difference[dw]), ("TempWeighted", difference[tw])):
        result[model + "_gain"] = {"mean": float(gain.mean()), "median": float(np.median(gain)), "P90": float(np.quantile(gain, .9))} if len(gain) else None
    return result


def group_row(frame, mask, dimension, group, models, totals):
    subset = frame.loc[mask]
    count = len(subset)
    origins = subset.sample_index.nunique()
    row = {"split": str(frame.split.iloc[0]), "dimension": dimension, "group": group,
        "count": count, "forecast_origins": origins, "observation_share": count/len(frame),
        "warning": "LOW SAMPLE SIZE" if origins < 20 else "",
        "availability": "ex-post only" if dimension in EX_POST or dimension == "binary_target" else "pre-forecast",
        "mean_abs_target": float(subset.target.abs().mean()) if count else None,
        "target_std": float(subset.target.std(ddof=0)) if count else None,
        "mean_group_value": float(subset[dimension].mean()) if count and dimension in QUANTILES else None}
    for col in ("p0", "p1", "p2", "confidence", "entropy", "transition_score"):
        row["mean_" + col] = float(subset[col].mean()) if count else None
    for model in models:
        m = metrics(subset[model].to_numpy(), subset.target.to_numpy())
        row.update({model + "_" + k: v for k, v in m.items()})
        e = subset[model].to_numpy(dtype=float)-subset.target.to_numpy(dtype=float)
        row[model + "_MAE_share"] = float(np.abs(e).sum()/totals[model][0]) if totals[model][0] else None
        row[model + "_MSE_share"] = float(np.square(e).sum()/totals[model][1]) if totals[model][1] else None
    if "TempWeighted" in models and count:
        row.update(delta_MAE=row["D0B_MAE"]-row["TempWeighted_MAE"], delta_MSE=row["D0B_MSE"]-row["TempWeighted_MSE"])
        row.update({k: v for k, v in paired_win_stats(subset.D0B.to_numpy(), subset.TempWeighted.to_numpy(), subset.target.to_numpy()).items() if "gain" not in k})
    return row


def clustered_bootstrap(frame, mask, draws=1000, seed=42):
    """Paired resampling of all origins; every selected origin keeps all its assets.

    Mask is frozen before resampling. Denominator is the resampled number of
    subgroup observations, not number of origins. Empty draws are explicit.
    """
    assert "TempWeighted" in frame
    delta = np.abs(frame.D0B.to_numpy()-frame.target.to_numpy())-np.abs(frame.TempWeighted.to_numpy()-frame.target.to_numpy())
    work = pd.DataFrame({"origin": frame.sample_index, "sum": delta*np.asarray(mask), "count": np.asarray(mask, dtype=int)})
    clusters = work.groupby("origin")[["sum", "count"]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(clusters), size=(draws, len(clusters)))
    numerators, denominators = clusters[ids, 0].sum(1), clusters[ids, 1].sum(1)
    valid = denominators > 0
    values = numerators[valid]/denominators[valid]
    n = int((clusters[:, 1] > 0).sum())
    return {"delta_MAE": float(clusters[:, 0].sum()/clusters[:, 1].sum()) if clusters[:, 1].sum() else None,
        "CI95": np.quantile(values, [.025, .975]).tolist() if len(values) else None,
        "draws": draws, "valid_draws": int(valid.sum()), "seed": seed, "forecast_origins": n,
        "warning": "LOW SAMPLE SIZE" if n < 20 else "",
        "unit": "paired forecast-origin cluster, all commodities kept together",
        "limitation": "IID origins; overlapping 5d windows retain serial dependence. CI is approximate, not a time-block CI."}


def complementarity(frame, thresholds):
    y = frame.target.to_numpy(dtype=float)
    d, t = frame.D0B.to_numpy(dtype=float), frame.TempWeighted.to_numpy(dtype=float)
    ed, et = d-y, t-y
    ad, at = np.abs(ed), np.abs(et)
    result = {"win_rates": paired_win_stats(d, t, y), "correlations": {}}
    for level, id_col in (("observation", None), ("sample", "sample_index"), ("commodity", "commodity_index")):
        v = pd.DataFrame({"Dabs": ad, "Tabs": at, "Dsq": ed**2, "Tsq": et**2})
        if id_col:
            v["id"] = frame[id_col].to_numpy()
            v = v.groupby("id").mean()
        result["correlations"][level] = {"absolute": finite_corr(v.Dabs, v.Tabs), "squared": finite_corr(v.Dsq, v.Tsq)}
    result["error_sign"] = {"agreement": float((np.sign(ed) == np.sign(et)).mean()),
        "both_overpredict": float(((ed > 0) & (et > 0)).mean()),
        "both_underpredict": float(((ed < 0) & (et < 0)).mean()),
        "opposite": float((ed*et < 0).mean()), "at_least_one_zero": float(((ed == 0) | (et == 0)).mean())}
    result["tail_overlap"] = {}
    for k, percentile in enumerate((90, 95)):
        ld = ad > thresholds["large_error_D0B"]["cuts"][k]
        lt = at > thresholds["large_error_TempWeighted"]["cuts"][k]
        result["tail_overlap"][str(percentile)] = {"both": int((ld & lt).sum()), "D0B_only": int((ld & ~lt).sum()),
            "TempWeighted_only": int((lt & ~ld).sum()), "neither": int((~ld & ~lt).sum()),
            "Jaccard": float((ld & lt).sum()/(ld | lt).sum()) if (ld | lt).any() else None}
    oracle_pred = np.where(ad <= at, d, t)
    result["oracle"] = metrics(oracle_pred, y)
    result["oracle"]["label"] = "Oracle upper-bound diagnostic only; non-deployable"
    md, mt = metrics(d, y), metrics(t, y)
    result["oracle_gap"] = {key: (min(md[key], mt[key])-result["oracle"][key])/min(md[key], mt[key]) if min(md[key], mt[key]) else None for key in ("MAE", "MSE")}
    result["average"] = metrics(.5*(d+t), y)
    result["advantage_correlations"] = {key: finite_corr(at-ad, frame[key]) for key in (
        "target_magnitude", "vol20", "dispersion_commodity", "trend_strength", "entropy", "confidence",
        "transition_score", "kl_surprise", "delta_z", "past5_stock", "past5_bond", "past5_commodity")}
    result["bootstrap"] = {"overall": clustered_bootstrap(frame, np.ones(len(frame), dtype=bool)),
        "high_vol": clustered_bootstrap(frame, frame.vol20 > thresholds["vol20"]["cuts"][2]),
        "reversal": clustered_bootstrap(frame, frame.group_reversal == "Reversal"),
        "high_transition": clustered_bootstrap(frame, frame.transition_score > thresholds["transition_score"]["cuts"][1])}
    return result


def analyze(observations, output):
    frame = observations.copy()
    for model in ("D0B", "TempWeighted"):
        if model in frame:
            frame[model] = frame[model].astype(np.float64)
    frame["target"] = frame.target.astype(np.float64)
    frame["target_magnitude"] = frame.target.abs()
    frame["trend_strength"] = frame.past5.abs()
    models = [m for m in ("D0B", "TempWeighted") if m in frame]
    thresholds = fit_thresholds(frame[frame.split == "train"], models)
    groups, categories = assign_groups(frame, thresholds, models)
    for key, values in groups.items():
        frame["group_" + key] = values
    observation_rows, sample_rows, commodity_rows, group_rows = [], [], [], []
    overall, paired, rankings, underreaction, worst, where_fail = {}, {}, {}, {}, {}, {}
    for split in ("train", "val", "test"):
        f = frame[frame.split == split].reset_index(drop=True)
        y = f.target.to_numpy()
        overall[split], underreaction[split] = {}, {}
        totals = {m: (float(np.abs(f[m]-y).sum()), float(np.square(f[m]-y).sum())) for m in models}
        for model in models:
            overall[split][model] = metrics(f[model].to_numpy(), y)
            columns = [c for c in f if c not in models]
            rec = f[columns].copy()
            rec["model"], rec["prediction"] = model, f[model]
            rec["residual"] = rec.prediction-rec.target
            rec["abs_residual"], rec["squared_residual"] = rec.residual.abs(), rec.residual**2
            rec["hit"] = (np.sign(rec.prediction) == np.sign(rec.target)).astype(int)
            observation_rows.append(rec)
            for sample, subset in f.groupby("sample_index", sort=True):
                sr = {"split": split, "model": model, "sample_index": int(sample),
                    "forecast_origin": subset.forecast_origin.iloc[0],
                    "target_magnitude": float(subset.target_magnitude.mean()),
                    **metrics(subset[model].to_numpy(), subset.target.to_numpy())}
                sr.update({key: float(subset[key].iloc[0]) for key in SAMPLE_COLUMNS | {"state", "margin", "p0", "p1", "p2"}})
                sample_rows.append(sr)
            mask = f.target_magnitude > thresholds["target_magnitude"]["cuts"][3]
            ratios = f.loc[mask, model].abs().to_numpy()/(f.loc[mask, "target"].abs().to_numpy()+1e-12)
            underreaction[split][model] = {"count": int(mask.sum()), "mean": float(ratios.mean()),
                "median": float(np.median(ratios)), "P25": float(np.quantile(ratios, .25)), "P75": float(np.quantile(ratios, .75))} if len(ratios) else {"count": 0}
        for c, name in f[["commodity_index", "commodity"]].drop_duplicates().itertuples(index=False, name=None):
            row = group_row(f, f.commodity_index == c, "commodity", name, models, totals)
            row["commodity_index"] = int(c)
            commodity_rows.append(row)
        comm = [r for r in commodity_rows if r["split"] == split]
        rankings[split] = {"highest_MSE": sorted(comm, key=lambda r: r["D0B_MSE"], reverse=True)[:5],
                           "lowest_MSE": sorted(comm, key=lambda r: r["D0B_MSE"])[:5]}
        if "TempWeighted" in models:
            rankings[split].update(D0B_advantage=sorted(comm, key=lambda r: r["delta_MAE"])[:5],
                                  TempWeighted_advantage=sorted(comm, key=lambda r: r["delta_MAE"], reverse=True)[:5])
        for dimension, labels in categories.items():
            for group in labels:
                group_rows.append(group_row(f, f["group_" + dimension] == group, dimension, group, models, totals))
        for q, cut in zip((90, 95), thresholds["target_magnitude"]["cuts"][-2:]):
            group_rows.append(group_row(f, f.target_magnitude > cut, "target_tail", f">TRAIN P{q}", models, totals))
        for commodity in rankings[split]["highest_MSE"]:
            for dimension in ("binary_target", "reversal", "binary_transition", "binary_vol"):
                for group in categories[dimension]:
                    mask = (f.commodity_index == commodity["commodity_index"]) & (f["group_" + dimension] == group)
                    row = group_row(f, mask, dimension, group, models, totals)
                    row["dimension"] = "commodity_specific_" + dimension
                    row["commodity"] = commodity["group"]
                    group_rows.append(row)
        comparisons = {
            "target magnitude": ("target_magnitude", "Q0-Q25", "Q95-Q100"),
            "volatility": ("vol20", "Q0-Q25", "Q90-Q100"),
            "reversal": ("reversal", "Continuation", "Reversal"),
            "regime transition": ("transition_score", "Q0-Q50", "Q90-Q100"),
        }
        where_fail[split] = []
        for dimension, (key, low, high) in comparisons.items():
            lo = next(r for r in group_rows if r["split"] == split and r["dimension"] == key and r["group"] == low)
            hi = next(r for r in group_rows if r["split"] == split and r["dimension"] == key and r["group"] == high)
            where_fail[split].append({"dimension": dimension, "easy_group": low, "hard_group": high,
                "easy_MAE": lo["D0B_MAE"], "hard_MAE": hi["D0B_MAE"],
                "MAE_ratio": hi["D0B_MAE"]/lo["D0B_MAE"] if lo["D0B_MAE"] and hi["count"] else None,
                "MSE_ratio": hi["D0B_MSE"]/lo["D0B_MSE"] if lo["D0B_MSE"] and hi["count"] else None,
                "hard_MSE_share": hi["D0B_MSE_share"], "hard_observation_share": hi["observation_share"],
                "warning": lo["warning"] or hi["warning"]})
        worst[split] = {}
        for model in models:
            samples = [r for r in sample_rows if r["split"] == split and r["model"] == model]
            worst[split][model] = {metric: sorted(samples, key=lambda r: r[metric], reverse=True)[:10] for metric in ("MAE", "MSE")}
        if "TempWeighted" in models:
            paired[split] = complementarity(f, thresholds)
            overall[split]["Average"] = paired[split]["average"]
            overall[split]["Oracle"] = paired[split]["oracle"]
            worst[split]["overlap"] = {metric: len({r["sample_index"] for r in worst[split]["D0B"][metric]} & {r["sample_index"] for r in worst[split]["TempWeighted"][metric]}) for metric in ("MAE", "MSE")}
            for metric in ("MAE", "MSE"):
                for row in worst[split]["D0B"][metric]:
                    tw = next(r for r in sample_rows if r["split"] == split and r["model"] == "TempWeighted" and r["sample_index"] == row["sample_index"])
                    row["TempWeighted_MAE"], row["TempWeighted_MSE"] = tw["MAE"], tw["MSE"]
    pd.concat(observation_rows, ignore_index=True).to_csv(output / "observation_errors.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output / "sample_errors.csv", index=False)
    pd.DataFrame(commodity_rows).to_csv(output / "commodity_errors.csv", index=False)
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(output / "all_groups.csv", index=False)
    for filename, dimensions in {
        "target_magnitude_groups": ["target_magnitude", "target_tail"],
        "volatility_groups": ["vol20", "vol5", "vol_x_reversal"],
        "reversal_groups": ["reversal", "direction", "trend_x_reversal"],
        "regime_transition_groups": ["state", "confidence", "entropy", "movement", "transition_score", "transition_max", "kl_surprise", "l1_surprise", "delta_z", "mean_delta_z5", "target_x_transition"],
    }.items():
        group_df[group_df.dimension.isin(dimensions)].to_csv(output / (filename + ".csv"), index=False)
    if paired:
        pd.json_normalize([{"split": split, **values} for split, values in paired.items()]).to_csv(output / "model_complementarity.csv", index=False)
        group_df.to_csv(output / "conditional_model_complementarity.csv", index=False)
    else:
        pd.DataFrame([{"status": "NOT COMPUTABLE", "reason": "Official TempWeighted checkpoint was not saved"}]).to_csv(output / "model_complementarity.csv", index=False)
    frame.to_csv(output / "aligned_observations.csv", index=False)
    return {"models": models, "thresholds": thresholds, "overall": overall, "groups": group_rows,
        "commodities": commodity_rows, "rankings": rankings, "complementarity": paired,
        "underreaction": underreaction, "worst_origins": worst, "where_fail": where_fail,
        "metric_definitions": {"pooling": "float64 sample × commodity population",
            "MAE": "mean(abs(pred-target))", "MSE": "mean((pred-target)^2)", "RMSE": "sqrt(MSE)",
            "Hit": "unmasked mean(sign(pred)==sign(target)); JSON fraction, report percent",
            "delta": "D0B minus TempWeighted; positive favors TempWeighted",
            "advantage": "TempWeighted absolute error minus D0B absolute error; positive favors D0B",
            "ties": f"absolute-error difference < {TOL}; identical tie mask for MSE wins"},
        "causal_definitions": {"volatility": "Existing feature_builder raw-unit vol_5d/vol_20d, per-commodity rolling std (ddof=1), commodity mean; recomputed identically for diagnostic only",
            "dispersion": "Cross-sectional std (ddof=0) of each market's 1d returns, trailing 5-day mean",
            "past5": "Observed close/close.shift(5)-1, same ret_5d feature clipping",
            "transition_score": "Mean last-5 L1 posterior changes within causal input window; recent observed movement, NOT future transition",
            "latent_state": "argmax p_T is a descriptive latent state 0/1/2, no economic regime label",
            "source_data": "Unchanged official loader including its existing fill/alignment behavior; no new model inputs",
            "TRAIN_posterior_thresholds": "Best D0B checkpoint applied to TRAIN; no labels used in latent cutpoints"},
        "primary_classification": None,
        "classification_status": "Requires paired evidence and interpretation" if paired else "BLOCKED: comparator checkpoint absent; not evidence for Case G"}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_results(result, path):
    path.write_text(json.dumps(clean_json(result), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def fmt(x):
    if x is None:
        return "N/A"
    if isinstance(x, (float, np.floating)):
        return f"{x:.9g}"
    return str(x).replace("|", "/")


def d0b_findings(result):
    """Descriptive statements with coverage, avoiding inference from empty bins."""
    lines = []
    for split in ("val", "test"):
        rows = [r for r in result["groups"] if r["split"] == split]
        def row(dim, name):
            return next(r for r in rows if r["dimension"] == dim and r["group"] == name)
        tails = [row("target_tail", ">TRAIN P90"), row("target_tail", ">TRAIN P95")]
        lines.append(f"{split.upper()} target tail: " + "; ".join(
            f"{r['group']} covers {100*r['observation_share']:.2f}% observations, contributes {100*r['D0B_MSE_share']:.2f}% MSE"
            for r in tails) + ". These are frozen TRAIN thresholds, not the top 10%/5% of this split.")
        vol = [row("vol20", label) for label in ("Q75-Q90", "Q90-Q100")]
        lines.append(f"{split.upper()} Vol20 high/extreme coverage: " + ", ".join(
            f"{r['group']}: {r['forecast_origins']} origins ({r['warning'] or 'adequate count'})" for r in vol)
            + ". Empty/sparse bins cannot establish a high-volatility bottleneck.")
        reversal = result["where_fail"][split][2]
        lines.append(f"{split.upper()} reversal/continuation: MAE ratio {fmt(reversal['MAE_ratio'])}, MSE ratio {fmt(reversal['MSE_ratio'])}.")
        for dim in ("movement", "transition_score", "delta_z"):
            candidates = [r for r in rows if r["dimension"] == dim]
            low, high = candidates[0], candidates[-1]
            ratio = high["D0B_MAE"]/low["D0B_MAE"] if high["count"] and low["D0B_MAE"] else None
            lines.append(f"{split.upper()} {dim} highest/lowest TRAIN bin MAE ratio: {fmt(ratio)}; {low['forecast_origins']}/{high['forecast_origins']} origins.")
        kl_high = [r for r in rows if r["dimension"] == "kl_surprise" and r["group"] in ("Q75-Q90", "Q90-Q100")]
        lines.append(f"{split.upper()} high KL surprise coverage: {sum(r['forecast_origins'] for r in kl_high)} origins; do not interpret empty high-surprise bins as evidence of good or bad forecasts.")
        top = result["rankings"][split]["highest_MSE"]
        share = sum(r["D0B_MSE_share"] for r in top)
        lines.append(f"{split.upper()} five highest-MSE commodities contribute {share:.2%} total MSE: " + ", ".join(r["group"] for r in top) + ".")
    lines.append("Large future moves dominate squared error, but target magnitude contains future information and error scales mechanically with target magnitude. This alone does not identify a causal model-design fix or a deployable volatility signal.")
    lines.append("Cross-split changes of sign in movement/error differences do not support a stable transition bottleneck. Sample-size and TRAIN-to-evaluation distribution shifts must be considered before an A–G decision.")
    return lines


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows])


def write_report(result, output):
    paired = bool(result["complementarity"])
    lines = ["# D0B-5dErrorRegimeDecomposition", "", result["status"], "",
        "本轮仅 eval + no_grad。没有训练、backward、参数更新、模型修改或新 variant。",
        "TRAIN 仅定义阈值与参考统计；VAL/TEST 使用冻结阈值。TRAIN inference 保留最后一个不满 batch。",
        "", f"Diagnostic Git SHA: `{result['diagnostic_git_sha']}`；fixed TEST batch: `{result['fixed_batch_shape']}`。",
        "未提交诊断代码的 SHA256 与数据/预处理源码指纹保存在 results.json。", "",
        table(["Model", "Checkpoint", "Best epoch", "Seed", "Training SHA", "Params", "Unchanged"], [
            [m, v["checkpoint_path"], v["best_epoch"], v["seed"], v["checkpoint_git_sha"], v["parameter_count"], v["checkpoint_and_state_unchanged"]]
            for m, v in result["checkpoints"].items()]), "",
        "缺失 checkpoint metadata 标为 N/A，不从当前 seed/SHA 倒推训练 provenance。",
        ""]
    if not paired:
        lines += ["**关键限制：用户确认 +TempWeighted 当时未保存权重。旧 cmgm_best.pth 属于其他架构。**",
            "因此本报告只包含实际 D0B inference；历史 MAE 不能重建逐样本比较。双模型指标、相关性、win rate、oracle、平均预测与配对 bootstrap 均不可计算。",
            "不补训，不用随机初始化模型替代。最终 A–G 分类暂停；checkpoint 缺失本身不构成 Case G。", ""]
    lines += ["## 1. Overall performance", "", "MAE/MSE 在所有 sample × commodity 上直接汇总；RMSE=sqrt(MSE)；Hit 为未屏蔽的符号一致率。", "",
        table(["Split", "Model", "MAE", "MSE", "RMSE", "Hit%", "|RMSE²−MSE|"], [
            [s.upper(), m, v["MAE"], v["MSE"], v["RMSE"], 100*v["Hit"], v["RMSE_squared_minus_MSE_abs"]]
            for s in ("val", "test") for m, v in result["overall"][s].items()]), "",
        "Oracle（如可计算）为 **Oracle upper-bound diagnostic only — non-deployable**；其 gap 不是可实现 ensemble 收益。Average 固定 0.5/0.5。", "",
        "## 2. Where does D0B fail?", "",
        *[line + "\n" for line in d0b_findings(result)],
        table(["Split", "Dimension (low vs high TRAIN bin)", "Easy MAE", "Hard MAE", "MAE ratio", "MSE ratio", "Hard MSE share", "Hard obs share", "Warning"], [
            [s.upper(), r["dimension"], r["easy_MAE"], r["hard_MAE"], r["MAE_ratio"], r["MSE_ratio"], r["hard_MSE_share"], r["hard_observation_share"], r["warning"]]
            for s in ("val", "test") for r in result["where_fail"][s]]), "",
        "此表的 easy/hard 是预先固定的低/高组标签，不按 VAL/TEST 的误差重新排序。", ""]
    for dimension in ("target_magnitude", "target_tail", "vol20", "vol5", "dispersion_commodity", "reversal", "direction", "vol_x_reversal",
                      "state", "confidence", "entropy", "movement", "transition_score", "kl_surprise", "l1_surprise", "delta_z", "mean_delta_z5", "target_x_transition"):
        rows = [r for r in result["groups"] if r["split"] in ("val", "test") and r["dimension"] == dimension]
        lines += [f"### {dimension}", "", table(["Split", "Group", "Obs", "Origins", "MAE", "MSE", "RMSE", "Hit%", "MSE share", "Warning"], [
            [r["split"].upper(), r["group"], r["count"], r["forecast_origins"], r["D0B_MAE"], r["D0B_MSE"], r["D0B_RMSE"],
             100*r["D0B_Hit"] if r["D0B_Hit"] is not None else None, r["D0B_MSE_share"], r["warning"]] for r in rows]), ""]
    lines += ["### Commodity rankings", ""]
    for s in ("val", "test"):
        for ranking, rows in result["rankings"][s].items():
            lines += [f"{s.upper()} {ranking}", "", table(["Commodity", "D0B MAE", "D0B MSE", "MSE share", "Delta MAE"], [
                [r["group"], r["D0B_MAE"], r["D0B_MSE"], r["D0B_MSE_share"], r.get("delta_MAE")] for r in rows]), ""]
    lines += ["每个 split 的 top-5 难商品细分见 all_groups.csv 的 commodity_specific_*；不将 TEST 排名用作训练或阈值。", "",
        "### Extreme-target magnitude response", "", table(["Split", "Model", "Count", "Mean |pred|/|y|", "Median", "P25", "P75"], [
            [s.upper(), m, v["count"], v.get("mean"), v.get("median"), v.get("P25"), v.get("P75")]
            for s in ("val", "test") for m, v in result["underreaction"][s].items()]), "",
        "### Worst TEST forecast origins (ranked by sample MSE)", "",
        table(["Date", "Index", "MAE", "MSE", "Mean |y|", "Vol20", "Dispersion", "State", "Entropy", "Transition", "KL", "DeltaZ", "TW MSE"], [
            [r["forecast_origin"], r["sample_index"], r["MAE"], r["MSE"], r["target_magnitude"], r["vol20"], r["dispersion_commodity"],
             r["state"], r["entropy"], r["transition_score"], r["kl_surprise"], r["delta_z"], r.get("TempWeighted_MSE")]
            for r in result["worst_origins"]["test"]["D0B"]["MSE"]]), "",
        "MAE 排序及两模型 top-10/overlap（若可计算）另存 results.json/worst_origins。", "",
        "## 3. D0B vs TempWeighted complementarity", ""]
    if not paired:
        lines += ["不可计算：+TempWeighted 正式 checkpoint 未保存。所有比较保持缺失，不以历史汇总值补齐。", ""]
    else:
        for s in ("val", "test"):
            p = result["complementarity"][s]
            lines += [f"### {s.upper()}", "", table(["Level", "Error", "Pearson", "Spearman"], [
                [lev, kind, v["Pearson"], v["Spearman"]] for lev, vals in p["correlations"].items() for kind, v in vals.items()]), "",
                f"Win rates: {p['win_rates']}; tail overlap: {p['tail_overlap']}; oracle relative gaps: {p['oracle_gap']}.", "",
                table(["Paired MAE difference", "Delta D0B−TW", "95% CI", "Origins", "Warning"], [
                    [g, b["delta_MAE"], b["CI95"], b["forecast_origins"], b["warning"]] for g, b in p["bootstrap"].items()]), ""]
        lines += ["Bootstrap: 1000 draws, seed 42, paired forecast-origin clusters, all commodities kept together. Overlapping windows retain serial dependence; IID-origin CI is approximate.", ""]
    lines += ["## 4. Deployable vs ex-post signals", "",
        table(["Pre-forecast variable", "Definition"], [[k, v] for k, v in result["causal_definitions"].items()]), "",
        table(["Ex-post only", "Restriction"], [
            ["Future |5d target| / future sign", "Requires future target; cannot select model at forecast origin"],
            ["Continuation/reversal and its interactions", "Contains future target; causal reversal precursor would require a separate future study"]]), "",
        "Zero past/future returns are a separate Zero group; near-zero past is Flat/weak. Latent state 0/1/2 has no asserted economic meaning. Low counts (<20 forecast origins) are flagged. Group thresholds and exact definitions are in results.json.", "",
        f"Causal feature validation: {result.get('causal_feature_sanity', 'see validation log')}", "",
        "## 5. Required questions", ""]
    answers = required_answers(result)
    lines.extend([f"{i}. {answer}" for i, answer in enumerate(answers, 1)])
    lines += ["", "## Decision", "",
        f"Primary classification: {result['primary_classification'] or 'NOT ASSIGNED'}. {result['classification_status']}.", "",
        "Observed bottleneck: 以上 D0B 分组结果给出实际误差集中位置；双模型共同错误/互补性的证据尚不完整。" if not paired else "Observed bottleneck: See the paired group evidence above; interpretation must consider both splits and uncertainty.", "",
        "Evidence: 全量 VAL/TEST，TRAIN 固定阈值，原 checkpoint 与全部 state_dict 前后 SHA256 一致。", "",
        "Therefore the next experiment should test: 暂不推荐新模型实验；完成双模型机制结论需要已有正式 TempWeighted 权重。本轮不会通过补训获得它。" if not paired else "Therefore the next experiment should test: No automatic experiment is launched; review the full paired evidence first.", "",
        "Do NOT yet implement: ensemble/gate、长窗口训练、volatility model、transition-aware module、任何新架构或 loss tuning。", "", "STOP.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def required_answers(result):
    def desc(dim, metric="D0B_MAE"):
        return "; ".join(s.upper() + ": " + ", ".join(f"{r['group']}={fmt(r[metric])}" for r in result["groups"] if r["split"] == s and r["dimension"] == dim) for s in ("val", "test"))
    def mechanism(dim):
        return desc(dim) + "。按 TRAIN 区间比较；趋势若两 split 不一致，不解释为稳定瓶颈。"
    p = result["complementarity"]
    missing = "不可计算，+TempWeighted 正式 checkpoint 未保存。"
    answers = [
        f"D0B 当前统一 TEST 5d 指标：{result['overall']['test']['D0B']}（Hit 为比例）。",
        f"TempWeighted：{result['overall']['test'].get('TempWeighted', missing)}",
        "Overall winner：" + (str({s: min(('D0B', 'TempWeighted'), key=lambda m: result['overall'][s][m]['MAE']) for s in ('val','test')}) if p else missing),
        "Target magnitude 与误差：" + desc("target_magnitude"),
        "TRAIN top-10%/top-5% target 尾部总 MSE 份额：" + desc("target_tail", "D0B_MSE_share"),
        "Past volatility：" + mechanism("vol20"),
        "Reversal/continuation 难度：" + "; ".join(f"{s.upper()} MAE ratio={fmt(result['where_fail'][s][2]['MAE_ratio'])}, MSE ratio={fmt(result['where_fail'][s][2]['MSE_ratio'])}" for s in ('val','test')),
        "High-volatility reversal：" + desc("vol_x_reversal"),
        "Latent state 间误差：" + mechanism("state"),
        "Posterior entropy：" + mechanism("entropy"),
        "p movement：" + mechanism("movement"),
        "KL surprise：" + mechanism("kl_surprise"),
        "Z movement：" + mechanism("delta_z"),
        "最大 MSE 商品：" + "; ".join(s.upper() + ': ' + ', '.join(r['group'] for r in result['rankings'][s]['highest_MSE']) for s in ('val','test')),
        "Observation error correlation：" + (str({s:v['correlations']['observation'] for s,v in p.items() if s != 'train'}) if p else missing),
        "Sample error correlation：" + (str({s:v['correlations']['sample'] for s,v in p.items() if s != 'train'}) if p else missing),
        "P90/P95 large-error overlap：" + (str({s:v['tail_overlap'] for s,v in p.items() if s != 'train'}) if p else missing),
        "Oracle MAE/MSE relative gap：" + (str({s:v['oracle_gap'] for s,v in p.items() if s != 'train'}) if p else missing),
        "0.5 average：" + (str({s:v['average'] for s,v in p.items() if s != 'train'}) if p else missing),
        "Primary A–G classification：" + result['classification_status'],
    ]
    return answers
