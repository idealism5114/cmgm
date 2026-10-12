#!/usr/bin/env python
"""Phase 2 - descriptive interpretability and cross-seed consistency.

Reads only the frozen, audited Phase 1 artifacts under
``interpretability/phase1/`` and writes descriptive tables under
``interpretability/phase2/``.

Phase 2 scope: describe what the frozen model learned, and measure whether
those descriptions are stable across seeds.

  * no model change, no retraining, no checkpoint change, no best-epoch reselect
  * TEST is never read; only the 268 validation samples
  * no masking, perturbation, branch ablation, gradient attribution,
    attention deletion or any other intervention
  * no explanation rule is chosen after seeing results; every rule below is
    fixed in CONFIG and written verbatim into the audit

Usage
-----
    python -m cmgm.scripts.phase2_analysis --smoke        # fast code check
    python -m cmgm.scripts.phase2_analysis                # full run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal, spearmanr, pearsonr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "interpretability" / "phase1"
P2 = ROOT / "interpretability" / "phase2"
SEEDS = (42, 43, 44)
MARKETS = ("stock", "bond", "commodity")
# capture = (layer, hop); analysis is per capture, never merged by default
CAPTURES = (("layer_1", 1, "L1H1"), ("layer_1", 2, "L1H2"),
            ("layer_2", 1, "L2H1"), ("layer_2", 2, "L2H2"))
HORIZONS = (1, 5, 10, 20)

CONFIG = {
    "phase": 2,
    "scope": "VAL only; descriptive only; no perturbation; no TEST; no retraining",
    "head_aggregation_rule": "simple arithmetic mean over all 8 heads",
    "head_aggregation_rationale":
        "all 8 heads are structurally equivalent in this variant; the "
        "attn_self_heads=4 split is inert because cross_mask is None; no head "
        "may be selected after inspecting results",
    "capture_analysis_rule":
        "the 4 captures (2 layers x 2 hops) are always reported separately; "
        "an all-capture mean is reported only as an explicitly labelled "
        "descriptive summary",
    "market_baseline_rule":
        "B_m = |V_m| / 284 (uniform-attention expected source share)",
    "enrichment_epsilon": 1e-12,
    "regime_alignment_rule":
        "Hungarian assignment (scipy linear_sum_assignment) on the Euclidean "
        "distance between z-scored, non-forward-looking state profile vectors; "
        "seed42 is the label reference only and is not claimed to be best",
    "state_profile_assignment_rule":
        "hard state = argmax of p_final (the final-timestep filtered posterior)",
    "effect_size_rule":
        "Kruskal-Wallis with epsilon-squared = (H - k + 1) / (n - k); "
        "distribution-weak by choice",
    "correlation_rule": "Spearman is primary; Pearson reported alongside",
    "p_value_caveat":
        "descriptive only; temporal dependence not corrected",
    "gate_categorical_thresholds": {"temporal_leaning": 0.6, "spatial_leaning": 0.4},
    "horizon_error_rule":
        "per-horizon mean absolute error over the 24 commodities; horizons are "
        "never pooled before association",
    "no_iid_pooling_rule": "the three seeds are never concatenated into 804 iid rows",
    "leakage_rule":
        "all regime market descriptors use only the current and trailing rows "
        "of the input window (rows r-19..r); no forward-looking target is used "
        "to characterise or name a state",
    "descriptor_window": 20,
    "outcome_association_label":
        "OUTCOME ASSOCIATION - descriptive; regimes are never defined from "
        "future returns",
}

DESCRIPTORS = (
    "ret_stock_5", "ret_stock_20", "ret_bond_5", "ret_bond_20",
    "ret_commodity_5", "ret_commodity_20",
    "vol_stock_20", "vol_bond_20", "vol_commodity_20",
    "disp_stock_20", "disp_bond_20", "disp_commodity_20",
    "corr_stock_bond_20", "corr_stock_commodity_20", "corr_bond_commodity_20",
    "state_confidence",
)


# --------------------------------------------------------------- utilities
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(v):
    """Recursively make a value JSON-safe; non-finite floats become null.

    A degenerate state has an undefined profile, so NaN is a real result here
    and is written as null rather than silently dropped or coerced to zero.
    """
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.str_):
        return str(v)
    if isinstance(v, Path):
        return str(v)
    return v


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def p1z(seed: int, name: str):
    return np.load(P1 / f"seed{seed}" / name)


def load_json(p: Path):
    return json.loads(p.read_text())


def entropy_norm(p, axis, n_states):
    p = np.asarray(p, dtype=np.float64)
    h = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=axis)
    return h / np.log(n_states)


def jsd(p, q, axis=-1):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, None)
    m = 0.5 * (p + q)
    return 0.5 * (p * np.log(p / m)).sum(axis) + 0.5 * (q * np.log(q / m)).sum(axis)


def _safe_stats(v):
    """Descriptive stats that tolerate a degenerate (empty) state."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return dict(n_samples=0, mean=float("nan"), median=float("nan"),
                    std=float("nan"), q25=float("nan"), q75=float("nan"),
                    iqr=float("nan"))
    return dict(n_samples=int(v.size), mean=float(np.mean(v)),
                median=float(np.median(v)),
                std=float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                iqr=float(np.percentile(v, 75) - np.percentile(v, 25)))


def _mean_or_nan(v):
    v = np.asarray(v, dtype=np.float64)
    return float(np.mean(v)) if v.size else float("nan")


def epsilon_squared(groups):
    """Kruskal-Wallis epsilon-squared over a list of 1-D arrays."""
    groups = [np.asarray(g, dtype=np.float64) for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n - k <= 0 or any(np.ptp(g) == 0 for g in groups):
        return float("nan"), float("nan")
    h, p = kruskal(*groups)
    return float((h - k + 1) / (n - k)), float(p)


def market_indices():
    n = load_json(P1 / "seed42" / "node_metadata.json")
    mi = {k: tuple(v) for k, v in n["market_indices"].items()}
    return mi, n


def raw_price_matrix():
    from cmgm import config
    from cmgm.data.data_loader import (align_markets, load_bond_prices,
                                       load_commodity_prices, load_stock_prices)
    all_prices, _ = align_markets(load_stock_prices(config.STOCK_FILE),
                                  load_bond_prices(config.BOND_FILE),
                                  load_commodity_prices(config.COMMODITY_FILE))
    return all_prices


# ------------------------------------------------------- market descriptors
def market_descriptor_frame(all_prices, mi) -> pd.DataFrame:
    """Backward-looking market descriptors for every validation sample date."""
    P = np.asarray(all_prices.values, dtype=np.float64)
    R = np.zeros_like(P)
    R[1:] = P[1:] / np.maximum(np.abs(P[:-1]), 1e-8) - 1.0
    R = np.clip(R, -0.5, 0.5)
    agg, disp = {}, {}
    for m in MARKETS:
        a, b = mi[m]
        agg[m] = R[:, a:b].mean(axis=1)
        disp[m] = R[:, a:b].std(axis=1, ddof=1)

    dates = all_prices.index
    rows = []
    for seed in SEEDS:
        z = p1z(seed, "predictions.npz")
        p_final = p1z(seed, "temporal_signals.npz")["p_final"].astype(np.float64)
        conf = 1.0 - entropy_norm(p_final, 1, 3)
        for i, d in enumerate(z["date_last_observed"]):
            r = int(dates.get_loc(pd.Timestamp(str(d))))
            w = np.arange(max(0, r - 19), r + 1)
            row = {"seed": seed, "sample_index": int(i), "date": str(d)}
            for m in MARKETS:
                row[f"ret_{m}_5"] = float(agg[m][r - 4:r + 1].sum())
                row[f"ret_{m}_20"] = float(agg[m][w].sum())
                row[f"vol_{m}_20"] = float(agg[m][w].std(ddof=1))
                row[f"disp_{m}_20"] = float(disp[m][w].mean())
            for x, y in (("stock", "bond"), ("stock", "commodity"),
                         ("bond", "commodity")):
                c = np.corrcoef(agg[x][w], agg[y][w])[0, 1]
                row[f"corr_{x}_{y}_20"] = float(c) if np.isfinite(c) else 0.0
            row["state_confidence"] = float(conf[i])
            row["state_entropy"] = float(entropy_norm(p_final, 1, 3)[i])
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- Part A
def part_a_regime(descriptors: pd.DataFrame, out_dir: Path) -> dict:
    sample_frames, occ_rows, dur_rows, trans_rows, profile_rows = [], [], [], [], []
    hard_by_seed, prob_by_seed, trans_param = {}, {}, {}

    for seed in SEEDS:
        ts = p1z(seed, "temporal_signals.npz")
        z = p1z(seed, "predictions.npz")
        p = ts["regime_probabilities"].astype(np.float64)      # (N,T,3)
        p_final = ts["p_final"].astype(np.float64)             # (N,3)
        dates = z["date_last_observed"]
        hard = p_final.argmax(1)
        hard_by_seed[seed], prob_by_seed[seed] = hard, p_final
        trans_param[seed] = ts["transition_matrix"][0].astype(np.float64)

        ent_t = entropy_norm(p, 2, 3)                          # (N,T)
        df = pd.DataFrame({
            "seed": seed, "sample_index": z["sample_index"], "date": dates,
            "p_state0": p_final[:, 0], "p_state1": p_final[:, 1],
            "p_state2": p_final[:, 2],
            "argmax_state": hard, "max_probability": p_final.max(1),
            "state_entropy_final": ent_t[:, -1],
            "state_confidence_final": 1.0 - ent_t[:, -1],
            "state_entropy_mean_over_window": ent_t.mean(1),
            "state_confidence_mean_over_window": 1.0 - ent_t.mean(1),
        })
        sample_frames.append(df)

        for k in range(3):
            occ_rows.append(dict(seed=seed, state=k,
                                 occupancy_count=int((hard == k).sum()),
                                 occupancy_fraction=float((hard == k).mean())))
            run_len, runs = [], 0
            for i, v in enumerate(hard):
                if v == k:
                    runs += 1
                elif runs:
                    run_len.append(runs); runs = 0
            if runs:
                run_len.append(runs)
            segs = 0
            start = None
            for i, v in enumerate(hard):
                if v == k and start is None:
                    start = i
                if start is not None and (v != k or i == len(hard) - 1):
                    end = i - 1 if v != k else i
                    dur_rows.append(dict(seed=seed, state=k,
                                         start_date=str(dates[start]),
                                         end_date=str(dates[end]),
                                         duration=int(end - start + 1)))
                    segs += 1
                    start = None
            dur_rows.append(dict(seed=seed, state=k, start_date="__SUMMARY__",
                                 end_date="__SUMMARY__", duration=None,
                                 n_runs=len(run_len),
                                 mean_duration=float(np.mean(run_len)) if run_len else 0.0,
                                 median_duration=float(np.median(run_len)) if run_len else 0.0,
                                 std_duration=float(np.std(run_len, ddof=1)) if len(run_len) > 1 else 0.0,
                                 max_duration=int(max(run_len)) if run_len else 0,
                                 n_segments=segs))

        T_count = np.zeros((3, 3))
        for a, b in zip(hard[:-1], hard[1:]):
            T_count[a, b] += 1
        Phat = T_count / np.clip(T_count.sum(1, keepdims=True), 1, None)
        for i in range(3):
            for j in range(3):
                trans_rows.append(dict(seed=seed, kind="empirical_argmax",
                                       from_state=i, to_state=j,
                                       probability=float(Phat[i, j])))
        for i in range(3):
            for j in range(3):
                trans_rows.append(dict(seed=seed, kind="model_parameter",
                                       from_state=i, to_state=j,
                                       probability=float(trans_param[seed][i, j])))
        trans_rows.append(dict(seed=seed, kind="empirical_argmax", from_state=-1,
                               to_state=-1,
                               probability=float((hard[:-1] != hard[1:]).mean()),
                               note="switching_frequency (diagonal-agnostic)"))

    sample_df = pd.concat(sample_frames, ignore_index=True)
    for seed in SEEDS:
        d = sample_df[sample_df.seed == seed]
        d.to_parquet(out_dir / f"seed{seed}" / "regime_sample_metrics.parquet", index=False)
        d.to_parquet(out_dir / f"regime_sample_metrics_seed{seed}.parquet", index=False)

    # state profiles on z-scored descriptors
    prof = descriptors.copy()
    scols = [c for c in DESCRIPTORS if c != "state_confidence"]
    for c in scols:
        prof[c] = prof.groupby("seed")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))
    for seed in SEEDS:
        hard = hard_by_seed[seed]
        d = prof[prof.seed == seed]
        for k in range(3):
            m = hard == k
            for c in DESCRIPTORS:
                st = _safe_stats(d[c].values[m])
                e2, pv = epsilon_squared(
                    [prof[prof.seed == seed][c].values[hard == kk] for kk in range(3)])
                profile_rows.append(dict(
                    seed=seed, state=k, descriptor=c,
                    effect_size_epsilon_squared=e2, kw_p_value_descriptive=pv,
                    low_n=bool(st["n_samples"] < 10), **st))
    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(out_dir / "regime_profiles.csv", index=False)
    pd.DataFrame(dur_rows).to_csv(out_dir / "regime_duration.csv", index=False)
    pd.DataFrame(trans_rows).to_csv(out_dir / "regime_transition_matrices.csv", index=False)

    # ---------------- alignment (Hungarian on z-scored profile vectors)
    ref = 42
    ref_hard = hard_by_seed[ref]
    ref_prof = {}
    for k in range(3):
        ref_prof[k] = np.array([
            _mean_or_nan(prof[(prof.seed == ref)][c].values[ref_hard == k])
            for c in DESCRIPTORS])
    align, costs, degenerate = {}, {}, {}
    for seed in SEEDS:
        h = hard_by_seed[seed]
        empty = [k for k in range(3) if int((h == k).sum()) == 0]
        degenerate[seed] = dict(empty_states=empty,
                                occupancy=[int((h == k).sum()) for k in range(3)])
        if seed == ref:
            align[seed] = {k: k for k in range(3)}
            costs[seed] = np.zeros((3, 3))
            continue
        sp = {k: np.array([_mean_or_nan(prof[prof.seed == seed][c].values[h == k])
                           for c in DESCRIPTORS]) for k in range(3)}
        C = np.array([[np.linalg.norm(ref_prof[i] - sp[j]) for j in range(3)]
                      for i in range(3)])
        finite = np.isfinite(C)
        # an empty state has an undefined profile; its cost is a sentinel so the
        # assignment still returns, and the degeneracy is reported explicitly
        sentinel = float(np.nanmax(C[finite])) + 1.0 if finite.any() else 1.0
        r, c = linear_sum_assignment(np.where(finite, C, sentinel))
        align[seed] = {int(i): int(j) for i, j in zip(r, c)}
        costs[seed] = C
    write_json(out_dir / "regime_alignment.json", dict(
        reference_seed=ref, rule=CONFIG["regime_alignment_rule"],
        alignment={str(s): {str(k): int(v) for k, v in align[s].items()} for s in SEEDS},
        cost_matrices={str(s): costs[s].tolist() for s in SEEDS},
        semantics="R1/R2/R3 are aligned labels; they are not training labels and "
                  "carry no economic name in Phase 2",
        degenerate_states={str(k): v for k, v in degenerate.items()},
        nan_cost_sentinel_rule="empty states get a sentinel cost above the largest "
                               "finite cost so the assignment still returns; the "
                               "resulting mapping for a degenerate state is not "
                               "meaningful and is flagged in degenerate_states",
        descriptor_order=list(DESCRIPTORS)))

    # ---------------- cross-seed consistency
    rows = []
    for a, b in ((42, 43), (42, 44), (43, 44)):
        ha, hb = hard_by_seed[a], hard_by_seed[b]
        rows.append(dict(pair=f"{a}-{b}", metric="adjusted_rand_index",
                         value=float(adjusted_rand_score(ha, hb))))
        rows.append(dict(pair=f"{a}-{b}", metric="normalized_mutual_info",
                         value=float(normalized_mutual_info_score(ha, hb))))
        rows.append(dict(pair=f"{a}-{b}", metric="raw_hard_agreement_uncalibrated",
                         value=float((ha == hb).mean())))
    for a, b in ((42, 43), (42, 44), (43, 44)):
        pa, pb = prob_by_seed[a], prob_by_seed[b]
        perm = np.zeros(3, dtype=int)
        for i in range(3):
            perm[i] = align[b][i]
        pb_al = pb[:, perm]
        rows.append(dict(pair=f"{a}-{b}", metric="aligned_hard_agreement",
                         value=float((pa.argmax(1) == pb_al.argmax(1)).mean())))
        d = jsd(pa, pb_al, axis=1)
        rows.append(dict(pair=f"{a}-{b}", metric="jsd_mean", value=float(d.mean())))
        rows.append(dict(pair=f"{a}-{b}", metric="jsd_median", value=float(np.median(d))))
        for k in range(3):
            r = pearsonr(pa[:, k], pb_al[:, k])[0]
            rows.append(dict(pair=f"{a}-{b}", metric=f"statewise_prob_pearson_R{k+1}",
                             value=float(r)))
    for a, b in ((42, 43), (42, 44), (43, 44)):
        for i in range(3):
            j = align[b][i]
            ha = hard_by_seed[a]
            va = np.array([_mean_or_nan(prof[prof.seed == a][c].values[ha == i])
                           for c in DESCRIPTORS])
            hb = hard_by_seed[b]
            vb = np.array([_mean_or_nan(prof[prof.seed == b][c].values[hb == j])
                           for c in DESCRIPTORS])
            rows.append(dict(pair=f"{a}-{b}", metric=f"profile_cosine_R{i+1}",
                             value=float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))))
            rows.append(dict(pair=f"{a}-{b}", metric=f"profile_spearman_R{i+1}",
                             value=float(spearmanr(va, vb)[0])))
            rows.append(dict(pair=f"{a}-{b}", metric=f"profile_euclidean_z_R{i+1}",
                             value=float(np.linalg.norm(va - vb))))
    cons = pd.DataFrame(rows)
    cons.to_csv(out_dir / "regime_cross_seed_consistency.csv", index=False)

    return dict(sample_df=sample_df, hard=hard_by_seed, probs=prob_by_seed,
                align=align, profile_df=profile_df, consistency=cons,
                descriptors=prof, trans_param=trans_param)


# --------------------------------------------------------------- Part B
def _pairwise_head_diversity(A_raw: np.ndarray):
    """Mean pairwise JSD and cosine similarity between the 8 heads."""
    eps = 1e-12
    n_heads = A_raw.shape[-1]
    js, cs = [], []
    for h1 in range(n_heads):
        p = A_raw[..., h1].astype(np.float64)
        np_ = p / np.clip(p.sum(-1, keepdims=True), eps, None)
        for h2 in range(h1 + 1, n_heads):
            q = A_raw[..., h2].astype(np.float64)
            qn = q / np.clip(q.sum(-1, keepdims=True), eps, None)
            m = 0.5 * (np_ + qn)
            js.append(float((0.5 * (np_ * np.log((np_ + eps) / (m + eps))).sum(-1)
                             + 0.5 * (qn * np.log((qn + eps) / (m + eps))).sum(-1)).mean()))
            cs.append(float((np_ * qn).sum(-1).mean()))
    del A_raw
    return (float(np.mean(js)), float(np.mean(cs)))


def part_b_graph(out_dir: Path, smoke=None):
    mi, node_meta = market_indices()
    Bvec = np.array([(mi[m][1] - mi[m][0]) / 284.0 for m in MARKETS])
    idx = {m: np.arange(mi[m][0], mi[m][1]) for m in MARKETS}
    eps = CONFIG["enrichment_epsilon"]
    cons_rows, head_rows, prior_rows, self_rows = [], [], [], []
    per_seed_capture_cache = {}

    for seed in SEEDS:
        z = p1z(seed, "spatial_signals.npz")
        dates = p1z(seed, "predictions.npz")["date_last_observed"]
        for layer, hop, cname in CAPTURES:
            base = f"graph_attention_{layer}_hop{hop}"
            A_raw = z[f"{base}_attention"]
            if smoke:
                A_raw = A_raw[:smoke]
                dates_c = dates[:smoke]
            else:
                dates_c = dates
            N = A_raw.shape[0]

            jsd_mean, cos_mean = _pairwise_head_diversity(A_raw)
            A = np.asarray(A_raw.mean(axis=3), dtype=np.float64)
            # per-head market matrices, for the head-sensitivity check
            per_head_M = []
            for h in range(A_raw.shape[3]):
                Ah = A_raw[..., h].astype(np.float64)
                Sm = {m: Ah[:, :, idx[m]].sum(axis=2) for m in MARKETS}
                per_head_M.append(np.stack(
                    [np.stack([Sm[m][:, idx[n]].mean(1)
                               for m in MARKETS], axis=1) for n in MARKETS], axis=1))
                del Sm
            head_share_std = float(np.mean(np.std(np.stack(per_head_M), axis=0)))
            del per_head_M

            S = {m: A[:, :, idx[m]].sum(axis=2) for m in MARKETS}      # (N,284)
            M = np.stack([np.stack([S[m][:, idx[n]].mean(axis=1)
                                    for m in MARKETS], axis=1) for n in MARKETS], axis=1)
            E = M / Bvec[None, None, :]
            row_sum_dev = float(np.abs(M.sum(2) - 1).max())

            pd.DataFrame({
                "seed": seed, "capture": cname, "date": dates_c,
                **{f"M_{n}_{m}": M[:, i, j] for i, n in enumerate(MARKETS)
                   for j, m in enumerate(MARKETS)},
            }).to_parquet(out_dir / f"market_attention_share_{cname}_seed{seed}.parquet",
                          index=False)
            pd.DataFrame({
                "seed": seed, "capture": cname, "date": dates_c,
                **{f"E_{n}_{m}": E[:, i, j] for i, n in enumerate(MARKETS)
                   for j, m in enumerate(MARKETS)},
            }).to_parquet(out_dir / f"market_attention_enrichment_{cname}_seed{seed}.parquet",
                          index=False)

            SI = A.mean(axis=1)                                        # (N,284)
            pd.DataFrame({
                "node_index": np.arange(284),
                "node_name": [n["node_name"] for n in node_meta["nodes"]],
                "market": [n["market"] for n in node_meta["nodes"]],
                "mean_source_influence": SI.mean(0),
                "std_source_influence": SI.std(0, ddof=1),
                "seed": seed, "capture": cname,
            }).to_parquet(out_dir / f"node_source_influence_{cname}_seed{seed}.parquet",
                          index=False)

            ent = -(A * np.log(np.clip(A, eps, None))).sum(2) / np.log(284)
            conc, hhi = 1.0 - ent, (A ** 2).sum(2)
            srt = np.sort(A, axis=2)[:, :, ::-1]
            top1, top5, top10 = srt[:, :, 0], srt[:, :, :5].sum(2), srt[:, :, :10].sum(2)
            cdf = {"seed": seed, "capture": cname, "date": dates_c}
            for nm, arr in (("concentration", conc), ("hhi", hhi), ("top1", top1),
                            ("top5", top5), ("top10", top10)):
                for m in MARKETS:
                    cdf[f"{nm}_{m}"] = arr[:, idx[m]].mean(1)
                cdf[f"{nm}_all"] = arr.mean(1)
            pd.DataFrame(cdf).to_parquet(
                out_dir / f"graph_concentration_{cname}_seed{seed}.parquet", index=False)

            vec = M.reshape(N, 9)
            tv = 0.5 * np.abs(np.diff(vec, axis=0)).sum(1)
            pd.DataFrame({"seed": seed, "capture": cname,
                          "date": dates_c[1:], "market_graph_turnover": tv}
                         ).to_parquet(
                out_dir / f"graph_turnover_{cname}_seed{seed}.parquet", index=False)

            selfw = np.diagonal(A, axis1=1, axis2=2)                   # (N,284)
            for m in MARKETS:
                v = selfw[:, idx[m]]
                self_rows.append(dict(seed=seed, capture=cname, market=m,
                                      mean=float(v.mean()), median=float(np.median(v)),
                                      p95=float(np.percentile(v, 95))))
            same_vals, cross_vals = [], []
            for m in MARKETS:
                sub = A[:, idx[m]][:, :, idx[m]]                       # (N, nm, nm)
                mask = ~np.eye(len(idx[m]), dtype=bool)
                same_vals.append(float(sub[:, mask].mean()))
                other = np.setdiff1d(np.arange(284), idx[m])
                cross_vals.append(float(A[:, idx[m]][:, :, other].mean()))
                del sub
            self_rows.append(dict(seed=seed, capture=cname, market="same_market_nonself",
                                  mean=float(np.mean(same_vals)), median=None, p95=None))
            self_rows.append(dict(seed=seed, capture=cname, market="cross_market",
                                  mean=float(np.mean(cross_vals)), median=None, p95=None))

            # mean-over-samples attention, for structural association only
            A_mean = A.mean(axis=0)

            pb = z[f"{base}_prior_bias"].astype(np.float64).mean(axis=2)   # (284,284)
            lg = z[f"{base}_logits"]
            if smoke:
                lg = lg[:smoke]
            lg = np.asarray(lg, dtype=np.float64).mean(axis=3).mean(0)     # (284,284)
            content = lg - pb
            prior_alpha = np.exp(pb - pb.max(1, keepdims=True))
            prior_alpha /= prior_alpha.sum(1, keepdims=True)
            prior_market = np.stack(
                [np.stack([prior_alpha[idx[n]][:, idx[m]].mean()
                           for m in MARKETS], axis=0) for n in MARKETS], axis=0)
            for n in range(3):
                for m in range(3):
                    prior_rows.append(dict(
                        seed=seed, capture=cname, kind="prior_market_share",
                        target_market=MARKETS[n], source_market=MARKETS[m],
                        value=float(prior_market[n, m])))
            prior_rows.append(dict(seed=seed, capture=cname,
                                   kind="spearman_prior_vs_attention",
                                   target_market=None, source_market=None,
                                   value=float(spearmanr(pb.ravel(), A_mean.ravel())[0])))
            prior_rows.append(dict(seed=seed, capture=cname,
                                   kind="spearman_content_vs_attention",
                                   target_market=None, source_market=None,
                                   value=float(spearmanr(content.ravel(), A_mean.ravel())[0])))
            prior_rows.append(dict(seed=seed, capture=cname,
                                   kind="spearman_prior_vs_content",
                                   target_market=None, source_market=None,
                                   value=float(spearmanr(pb.ravel(), content.ravel())[0])))

            within = float(np.mean([M[:, i, i].mean() for i in range(3)]))
            head_rows.append(dict(seed=seed, capture=cname,
                                  mean_pairwise_head_jsd=jsd_mean,
                                  mean_pairwise_head_cosine=cos_mean,
                                  mean_std_across_heads_of_market_share=head_share_std,
                                  within_market_share=within))
            per_seed_capture_cache[(seed, cname)] = dict(
                M=M, E=E, conc=conc, tv=tv, dates=dates_c, SI=SI,
                row_sum_dev=row_sum_dev,
                within=M[:, np.arange(3), np.arange(3)].mean(1),
                cross=1.0 - M[:, np.arange(3), np.arange(3)].mean(1))
            del A_raw, A, S, M, E, ent, conc, hhi, srt, top1, top5, top10, selfw, A_mean

    pd.DataFrame(self_rows).to_csv(out_dir / "self_attention_summary.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(out_dir / "prior_vs_attention.csv", index=False)
    pd.DataFrame(head_rows).to_csv(out_dir / "graph_head_diversity.csv", index=False)

    # ---------------- cross-seed consistency (per capture, never merged)
    for layer, hop, cname in CAPTURES:
        for a, b in ((42, 43), (42, 44), (43, 44)):
            ca, cb = per_seed_capture_cache[(a, cname)], per_seed_capture_cache[(b, cname)]
            for tag, Ka, Kb in (("market_matrix", ca["M"], cb["M"]),
                                ("enrichment_matrix", ca["E"], cb["E"])):
                va, vb = Ka.reshape(len(Ka), 9), Kb.reshape(len(Kb), 9)
                cons_rows.append(dict(capture=cname, pair=f"{a}-{b}", kind=tag,
                                      metric="pearson_mean", value=float(np.mean(
                                          [pearsonr(x, y)[0] for x, y in zip(va, vb)]))))
                cons_rows.append(dict(capture=cname, pair=f"{a}-{b}", kind=tag,
                                      metric="spearman_mean", value=float(np.mean(
                                          [spearmanr(x, y)[0] for x, y in zip(va, vb)]))))
                cons_rows.append(dict(capture=cname, pair=f"{a}-{b}", kind=tag,
                                      metric="cosine_mean", value=float(np.mean(
                                          [x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
                                           for x, y in zip(va, vb)]))))
                cons_rows.append(dict(capture=cname, pair=f"{a}-{b}", kind=tag,
                                      metric="l1_mean", value=float(np.mean(
                                          np.abs(va - vb).sum(1)))))
            for k in (10, 20):
                ia = np.argsort(-ca["SI"].mean(0))[:k]
                ib = np.argsort(-cb["SI"].mean(0))[:k]
                cons_rows.append(dict(capture=cname, pair=f"{a}-{b}",
                                      kind=f"top{k}_source_influence",
                                      metric="jaccard", value=float(
                                          len(set(ia) & set(ib)) / len(set(ia) | set(ib)))))
                common = np.union1d(ia, ib)
                cons_rows.append(dict(capture=cname, pair=f"{a}-{b}",
                                      kind=f"top{k}_source_influence",
                                      metric="spearman_on_common_nodes", value=float(
                                          spearmanr(ca["SI"].mean(0)[common],
                                                    cb["SI"].mean(0)[common])[0])))
            cons_rows.append(dict(capture=cname, pair=f"{a}-{b}", kind="row_sum_check",
                                  metric="max_row_sum_deviation",
                                  value=max(ca["row_sum_dev"], cb["row_sum_dev"])))
    pd.DataFrame(cons_rows).to_csv(out_dir / "graph_cross_seed_consistency.csv", index=False)
    return per_seed_capture_cache


# --------------------------------------------------------------- Part C
def part_c_gate(out_dir: Path):
    chan_rows, cons_rows, cache = [], [], {}
    summary_metrics = ["mean_gate", "gate_entropy_norm", "decisiveness_norm",
                       "frac_gt_0.5", "frac_gt_0.75", "frac_lt_0.25",
                       "gate_median", "gate_q25", "gate_q75", "gate_std"]
    for seed in SEEDS:
        g = p1z(seed, "fusion_signals.npz")["fusion_gate"].astype(np.float64)
        dates = p1z(seed, "predictions.npz")["date_last_observed"]
        gg = np.clip(g, 1e-12, 1 - 1e-12)
        h = -(gg * np.log(gg) + (1 - gg) * np.log(1 - gg)) / np.log(2)
        df = pd.DataFrame({
            "seed": seed, "date": dates,
            "mean_gate": g.mean(1), "gate_std": g.std(1, ddof=1),
            "gate_median": np.median(g, axis=1),
            "gate_q25": np.percentile(g, 25, axis=1),
            "gate_q75": np.percentile(g, 75, axis=1),
            "gate_entropy_norm": h.mean(1),
            "decisiveness_norm": 2 * np.abs(g - 0.5).mean(1),
            "frac_gt_0.5": (g > 0.5).mean(1), "frac_gt_0.75": (g > 0.75).mean(1),
            "frac_lt_0.25": (g < 0.25).mean(1),
            "temporal_leaning_channels": (g > 0.6).sum(1),
            "spatial_leaning_channels": (g < 0.4).sum(1)})
        df.to_parquet(out_dir / f"seed{seed}" / "gate_sample_metrics.parquet", index=False)
        df.to_parquet(out_dir / f"gate_sample_metrics_seed{seed}.parquet", index=False)
        mc, sc = g.mean(0), g.std(0, ddof=1)
        thr = CONFIG["gate_categorical_thresholds"]
        for j in range(g.shape[1]):
            chan_rows.append(dict(
                seed=seed, channel=j, mean_gate=float(mc[j]), std_gate=float(sc[j]),
                temporal_leaning=bool(mc[j] > thr["temporal_leaning"]),
                spatial_leaning=bool(mc[j] < thr["spatial_leaning"]),
                dynamic=bool(sc[j] > np.median(sc))))
        cache[seed] = dict(g=g, df=df)
    chan = pd.DataFrame(chan_rows)
    chan.to_csv(out_dir / "gate_channel_summary.csv", index=False)
    for s_ in SEEDS:
        chan[chan.seed == s_].to_csv(out_dir / f"gate_channel_summary_seed{s_}.csv",
                                     index=False)
    for a, b in ((42, 43), (42, 44), (43, 44)):
        da, db = cache[a]["df"], cache[b]["df"]
        for m in summary_metrics:
            cons_rows.append(dict(pair=f"{a}-{b}", kind="summary_stat", metric=m,
                                  pearson=float(pearsonr(da[m], db[m])[0]),
                                  spearman=float(spearmanr(da[m], db[m])[0]),
                                  mae=float(np.abs(da[m] - db[m]).mean())))
        ga, gb = cache[a]["g"], cache[b]["g"]
        cons_rows.append(dict(
            pair=f"{a}-{b}", kind="full_64d_index_aligned_semantics_not_guaranteed",
            metric="cosine_mean",
            pearson=float(np.mean([x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
                                   for x, y in zip(ga, gb)])),
            spearman=float(np.mean([spearmanr(x, y)[0] for x, y in zip(ga, gb)])),
            mae=float(np.abs(ga - gb).sum(1).mean())))
    pd.DataFrame(cons_rows).to_csv(out_dir / "gate_cross_seed_consistency.csv",
                                   index=False)
    return cache


# --------------------------------------------------------------- Part D
def _merged_frame(seed, regime, graph_cache, gate_cache, n=None):
    df = regime["sample_df"]
    df = df[df.seed == seed].reset_index(drop=True)
    if n is not None:
        df = df.iloc[:n].reset_index(drop=True)
    inv = {v: k for k, v in regime["align"][seed].items()}
    hard = regime["hard"][seed]
    if n is not None:
        hard = hard[:n]
    out = pd.DataFrame({
        "seed": seed, "date": df["date"], "sample_index": df["sample_index"],
        "aligned_state": [inv[h] for h in hard],
        "state_entropy_final": df["state_entropy_final"],
        "state_confidence_final": df["state_confidence_final"]})
    z = p1z(seed, "predictions.npz")
    pred, targ = z["prediction"], z["target"]
    if n is not None:
        pred, targ = pred[:n], targ[:n]
    for i, h in enumerate(HORIZONS):
        out[f"err_{h}"] = np.abs(pred[:, i] - targ[:, i]).mean(axis=1)
    out["err_all"] = np.mean([out[f"err_{h}"] for h in HORIZONS], axis=0)
    for i, h in enumerate(HORIZONS):
        out[f"target_ret_{h}"] = targ[:, i].mean(axis=1)
    g = gate_cache[seed]["df"]
    if n is not None:
        g = g.iloc[:n].reset_index(drop=True)
    for c in ("mean_gate", "gate_entropy_norm", "decisiveness_norm", "frac_gt_0.5"):
        out[c] = g[c].values
    for layer, hop, cname in CAPTURES:
        cc = graph_cache[(seed, cname)]
        out[f"within_{cname}"] = cc["within"]
        out[f"cross_{cname}"] = cc["cross"]
        out[f"conc_{cname}"] = cc["conc"].mean(1)
        out[f"turnover_{cname}"] = np.concatenate([[np.nan], cc["tv"]])
        for i, n in enumerate(MARKETS):
            for j, m in enumerate(MARKETS):
                out[f"E_{n}_{m}_{cname}"] = cc["E"][:, i, j]
    for name in ("within", "cross", "conc", "turnover"):
        cols = [f"{name}_{c[2]}" for c in CAPTURES]
        out[name + "_avg"] = out[cols].mean(axis=1)
    for n in MARKETS:
        for m in MARKETS:
            cols = [f"E_{n}_{m}_{c[2]}" for c in CAPTURES]
            out[f"E_{n}_{m}_avg"] = out[cols].mean(axis=1)
    return out


def part_d_joint(out_dir: Path, regime, graph_cache, gate_cache, smoke=None):
    frames = {s: _merged_frame(s, regime, graph_cache, gate_cache,
                               n=smoke) for s in SEEDS}
    pd.concat(frames.values(), ignore_index=True).to_parquet(
        out_dir / "phase2_sample_metrics_all_seeds.parquet", index=False)

    graph_metrics = (["within_avg", "cross_avg", "conc_avg", "turnover_avg"]
                     + [f"within_{c[2]}" for c in CAPTURES]
                     + [f"cross_{c[2]}" for c in CAPTURES]
                     + [f"conc_{c[2]}" for c in CAPTURES]
                     + [f"turnover_{c[2]}" for c in CAPTURES]
                     + [f"E_{n}_{m}_avg" for n in MARKETS for m in MARKETS])
    gate_metrics = ["mean_gate", "gate_entropy_norm", "decisiveness_norm", "frac_gt_0.5"]

    # --- regime x graph
    rows = []
    for s in SEEDS:
        d = frames[s]
        for gm in graph_metrics:
            groups = [d[gm].values[d.aligned_state == k] for k in range(3)]
            e2, pv = epsilon_squared(groups)
            r = dict(seed=s, metric=gm,
                     kw_p_value_descriptive=pv, epsilon_squared=e2)
            for k in range(3):
                v = groups[k]
                r[f"R{k+1}_mean"] = float(np.nanmean(v)) if len(v) else float("nan")
                r[f"R{k+1}_n"] = int(len(v))
            rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "regime_graph_association.csv", index=False)

    # --- regime x gate
    rows = []
    for s in SEEDS:
        d = frames[s]
        for gm in gate_metrics:
            groups = [d[gm].values[d.aligned_state == k] for k in range(3)]
            e2, pv = epsilon_squared(groups)
            r = dict(seed=s, metric=gm, kw_p_value_descriptive=pv, epsilon_squared=e2)
            for k in range(3):
                v = groups[k]
                r[f"R{k+1}_mean"] = float(np.nanmean(v)) if len(v) else float("nan")
                r[f"R{k+1}_n"] = int(len(v))
            rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "regime_gate_association.csv", index=False)

    # --- graph x gate (Spearman primary)
    rows = []
    for s in SEEDS:
        d = frames[s]
        for gm in graph_metrics:
            for gate in gate_metrics:
                a, b = d[gm].values, d[gate].values
                m = np.isfinite(a) & np.isfinite(b)
                rho, p = spearmanr(a[m], b[m])
                rows.append(dict(seed=s, graph_metric=gm, gate_metric=gate,
                                 spearman_rho=float(rho),
                                 pearson_r=float(pearsonr(a[m], b[m])[0]),
                                 n=int(m.sum()), p_value_descriptive=float(p)))
    pd.DataFrame(rows).to_csv(out_dir / "graph_gate_association.csv", index=False)

    # --- interpretability x prediction error, per horizon, per seed
    rows = []
    interp = (["state_entropy_final", "state_confidence_final"] + gate_metrics
              + ["conc_avg", "turnover_avg", "within_avg", "cross_avg"])
    for s in SEEDS:
        d = frames[s]
        for im in interp:
            a = d[im].values
            for h in HORIZONS:
                b = d[f"err_{h}"].values
                m = np.isfinite(a) & np.isfinite(b)
                rho, p = spearmanr(a[m], b[m])
                rows.append(dict(seed=s, interpretability_metric=im, error_metric=f"MAE@{h}d",
                                 spearman_rho=float(rho), n=int(m.sum()),
                                 p_value_descriptive=float(p),
                                 p_value_caveat=CONFIG["p_value_caveat"]))
            b = d["err_all"].values
            rho, p = spearmanr(a, b)
            rows.append(dict(seed=s, interpretability_metric=im, error_metric="MAE_all_horizons",
                             spearman_rho=float(rho), n=int(len(a)),
                             p_value_descriptive=float(p),
                             p_value_caveat=CONFIG["p_value_caveat"]))
    pd.DataFrame(rows).to_csv(out_dir / "interpretability_error_association.csv",
                              index=False)

    # --- OUTCOME ASSOCIATION (clearly separated; regimes never defined from this)
    rows = []
    for s in SEEDS:
        d = frames[s]
        for k in range(3):
            sel = d.aligned_state == k
            r = dict(seed=s, aligned_state=f"R{k+1}", n_samples=int(sel.sum()),
                     label="OUTCOME ASSOCIATION - regimes are not defined from "
                           "future returns")
            for h in HORIZONS:
                r[f"mean_realised_return_{h}d"] = float(d.loc[sel, f"target_ret_{h}"].mean())
                r[f"mean_abs_error_{h}d"] = float(d.loc[sel, f"err_{h}"].mean())
            r["mean_state_entropy"] = float(d.loc[sel, "state_entropy_final"].mean())
            rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "outcome_association.csv", index=False)
    return frames


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=int, default=None,
                    help="analyse only the first N samples (code check only)")
    ap.add_argument("--phase2-root", type=Path, default=P2)
    args = ap.parse_args()
    root = Path(args.phase2_root).resolve()
    parts = ("regime", "graph", "gate", "joint", "figures", "tables", "audit")
    for p in parts:
        (root / p).mkdir(parents=True, exist_ok=True)
    for p in ("regime", "graph", "gate"):
        for s in SEEDS:
            (root / p / f"seed{s}").mkdir(exist_ok=True)

    all_prices = raw_price_matrix()
    mi, node_meta = market_indices()
    desc = market_descriptor_frame(all_prices, mi)
    desc.to_parquet(root / "tables" / "market_descriptors.parquet", index=False)

    regime = part_a_regime(desc, root / "regime")
    graph_cache = part_b_graph(root / "graph", smoke=args.smoke)
    gate_cache = part_c_gate(root / "gate")
    part_d_joint(root / "joint", regime, graph_cache, gate_cache, smoke=args.smoke)

    import matplotlib, pandas, scipy, sklearn
    inputs = {}
    for s in SEEDS:
        for f in ("predictions.npz", "temporal_signals.npz", "spatial_signals.npz",
                  "fusion_signals.npz", "audit.json", "node_metadata.json"):
            p = P1 / f"seed{s}" / f
            inputs[str(p.relative_to(ROOT))] = sha256_file(p)
    outputs = {str(p.relative_to(ROOT)): sha256_file(p)
               for p in sorted(root.rglob("*"))
               if p.is_file() and "audit" not in p.parts}
    write_json(root / "audit" / "phase2_audit.json", dict(
        phase=2,
        generated_utc=datetime.utcnow().isoformat() + "Z",
        seeds=list(SEEDS),
        sample_count={str(s): int((regime["sample_df"].seed == s).sum()) for s in SEEDS},
        date_range=[str(regime["sample_df"].date.min()), str(regime["sample_df"].date.max())],
        scope=CONFIG["scope"],
        rules=CONFIG,
        node_to_market_mapping_sha256=sha256_file(P1 / "seed42" / "node_metadata.json"),
        market_sizes=node_meta["market_sizes"],
        market_baseline={m: (mi[m][1] - mi[m][0]) / 284.0 for m in MARKETS},
        captures=[c[2] for c in CAPTURES],
        input_phase1_hashes=inputs,
        output_hashes=outputs,
        software=dict(python=__import__("platform").python_version(),
                      numpy=np.__version__, pandas=pandas.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__,
                      matplotlib=matplotlib.__version__),
        smoke_mode=bool(args.smoke is not None),
        smoke_samples=args.smoke,
        no_test_used=True,
        phase1_artifacts_modified=False,
    ))
    print(f"Phase 2 analysis written under {root}"
          + (f"  [SMOKE {args.smoke} samples]" if args.smoke else ""))


if __name__ == "__main__":
    main()
