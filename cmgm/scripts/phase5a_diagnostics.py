#!/usr/bin/env python
"""Phase 5A - frozen representation diagnostics.

Locates where in the frozen D0B forward pass sample-level structure is
attenuated: spatial pooling, the temporal readout, or the fusion compression.

Read-only: Phase 1 artifacts are used directly for VAL, and TRAIN
representations are re-extracted with the same frozen checkpoints in eval mode
under no_grad, with no model update. TEST is never read.

Vocabulary note: variance and effective dimensionality are NOT information.
Every statement here is about representation compression, variation
attenuation, geometry distortion or predictive accessibility, never about a
mutual-information quantity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.main_ablation import build_data
from cmgm.training.metric_standard import population_metrics

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "interpretability" / "phase1"
P5 = ROOT / "interpretability" / "phase5"
OUT = P5 / "frozen_diagnostics"
SEEDS = (42, 43, 44)
HORIZONS = (1, 5, 10, 20)
K_VALUES = (5, 10, 20)          # pre-registered
RIDGE_ALPHA = 1.0               # pre-registered primary
RIDGE_SENSITIVITY = (0.1, 1.0, 10.0)
PCA_DIMS = (64, 128)            # pre-registered
MARKETS = ("stock", "bond", "commodity")

VARIANTS = {
    42: "switching_latent_balanced_readout_seed42_20261006_231320_782593",
    43: "switching_latent_balanced_readout_seed43_20261007_123714_3172316",
    44: "switching_latent_balanced_readout_seed44_20261007_125331_3172316",
}
CKPT_ROOT = ROOT / "checkpoints" / "d0b_node_history_readout"

# Stage definitions, in TRUE forward order. The spec's SP/TP labels are mapped
# onto them and the ordering difference is recorded in the audit.
STAGE_NOTE = (
    "Spatial stages are listed in true forward order. The spec's SP2 "
    "(spatial_aggregated) actually precedes the graph layers, and the spec's SP0 "
    "(type_pool_input) is the post-LayerNorm node tensor entering pooling; the "
    "numbering therefore does not follow the forward pass and is documented here."
)


def sha256_file(path: Path) -> str:
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


# ------------------------------------------------------------ diagnostics
def variation_stats(X: np.ndarray) -> dict:
    """Per-dimension dispersion of a (B, D) representation, then summarised."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("expected a 2-D sample representation")
    std = X.std(axis=0, ddof=1)
    q75, q25 = np.percentile(X, 75, axis=0), np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    mad = np.median(np.abs(X - np.median(X, axis=0, keepdims=True)), axis=0)
    scale = np.abs(X).mean() + 1e-12
    return dict(dim=int(X.shape[1]), n=int(X.shape[0]),
                median_dim_std=float(np.median(std)),
                mean_dim_std=float(np.mean(std)),
                median_dim_iqr=float(np.median(iqr)),
                median_dim_mad=float(np.median(mad)),
                median_dim_std_normalized=float(np.median(std / scale)),
                overall_rms=float(np.sqrt((X ** 2).mean())))


def dynamics_stats(X: np.ndarray) -> dict:
    """Consecutive-sample displacement, normalized by sqrt(d)."""
    X = np.asarray(X, dtype=np.float64)
    d = 1.0 / np.sqrt(X.shape[1])
    step = np.linalg.norm(np.diff(X, axis=0), axis=1) * d
    return dict(mean=float(step.mean()), median=float(np.median(step)),
                p5=float(np.percentile(step, 5)), p95=float(np.percentile(step, 95)),
                max=float(step.max()))


def effective_dimension(X: np.ndarray) -> dict:
    """Participation ratio and PCA dimensions at fixed variance thresholds."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    sv = np.linalg.svd(Xc, compute_uv=False)
    lam = sv ** 2
    tot = lam.sum()
    if tot <= 0:
        return dict(participation_ratio=float("nan"), normalized=float("nan"),
                    dim_50=0, dim_80=0, dim_90=0, dim_95=0, rank_max=int(n - 1))
    pr = float((lam.sum() ** 2) / (lam ** 2).sum())
    cum = np.cumsum(lam) / tot
    def dim_at(th):
        return int(np.searchsorted(cum, th) + 1)
    return dict(participation_ratio=pr, normalized=pr / X.shape[1],
                dim_50=dim_at(0.50), dim_80=dim_at(0.80),
                dim_90=dim_at(0.90), dim_95=dim_at(0.95),
                rank_max=int(min(n - 1, X.shape[1])))


def _pdist(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sq = (X ** 2).sum(axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.sqrt(np.clip(D2, 0.0, None))


def geometry_preservation(Xa: np.ndarray, Xb: np.ndarray) -> dict:
    """Compare sample pairwise distance structures of two representations."""
    a, b = _pdist(Xa), _pdist(Xb)
    iu = np.triu_indices(a.shape[0], k=1)
    va, vb = a[iu], b[iu]
    return dict(spearman=float(spearmanr(va, vb)[0]),
                pearson=float(pearsonr(va, vb)[0]),
                mean_dist_a=float(va.mean()), mean_dist_b=float(vb.mean()))


def knn_sets(X: np.ndarray, k: int) -> np.ndarray:
    D = _pdist(X)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def knn_retention(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    A, B = knn_sets(Xa, k), knn_sets(Xb, k)
    return float(np.mean([len(set(A[i]) & set(B[i])) / k for i in range(A.shape[0])]))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den) if den > 0 else float("nan")


def flatten_market(node_tensor: np.ndarray, mi: dict) -> dict:
    """Per-market flattening of a (B, N, D) node tensor."""
    out = {}
    for m in MARKETS:
        a, b = mi[m]
        out[m] = node_tensor[:, a:b, :].reshape(node_tensor.shape[0], -1)
    return out


def cross_sectional_dispersion(node_tensor: np.ndarray, mi: dict) -> pd.DataFrame:
    rows = {}
    for m in MARKETS:
        a, b = mi[m]
        sub = node_tensor[:, a:b, :].astype(np.float64)
        mean = sub.mean(axis=1, keepdims=True)
        rows[f"dispersion_{m}"] = np.linalg.norm(sub - mean, axis=2).mean(axis=1)
        rows[f"mean_node_variance_{m}"] = sub.var(axis=1, ddof=1).mean(axis=1)
    return pd.DataFrame(rows)


# ------------------------------------------------------- stage assembly
def load_val_stages(seed: int) -> dict:
    """Assemble every diagnostic stage for one seed from Phase 1 artifacts."""
    sp = np.load(P1 / f"seed{seed}" / "spatial_signals.npz")
    tp = np.load(P1 / f"seed{seed}" / "temporal_signals.npz")
    fu = np.load(P1 / f"seed{seed}" / "fusion_signals.npz")
    pr = np.load(P1 / f"seed{seed}" / "predictions.npz")
    mi, _ = market_indices()
    node_pre = sp["spatial_aggregated"]      # time-weighted, pre-GNN
    node_post = sp["type_pool_input"]        # post-LayerNorm, pre-pooling (spec SP0)
    flat_pre = node_pre.reshape(node_pre.shape[0], -1)
    flat_post = node_post.reshape(node_post.shape[0], -1)
    mk_post = flatten_market(node_post, mi)
    hs, ht = sp["spatial_readout"], tp["temporal_readout"]
    s_proj, t_proj = fu["spatial_projected"], fu["temporal_projected"]
    stages = {
        # spatial
        "S_node_prgnn": node_pre.reshape(node_pre.shape[0], -1),
        "S_node_prepool": flat_post,
        "S_node_prepool_stock": mk_post["stock"],
        "S_node_prepool_bond": mk_post["bond"],
        "S_node_prepool_commodity": mk_post["commodity"],
        "S_pool_stock": sp["stock_pool"], "S_pool_bond": sp["bond_pool"],
        "S_pool_commodity": sp["commodity_pool"],
        "S_hs": hs, "S_s": s_proj,
        # temporal
        "T_token": tp["market_tokens"][:, -1, :], "T_H": tp["H_final"],
        "T_Z": tp["Z_final"], "T_uH": tp["u_H"], "T_uZ": tp["u_Z"],
        "T_cat": tp["balanced_concat"], "T_ht": ht, "T_t": t_proj,
        # fusion
        "F0_hs_ht": np.concatenate([hs, ht], axis=1),
        "F1_s_t": np.concatenate([s_proj, t_proj], axis=1),
        "F2_fused": fu["fused_representation"],
        "F2_fused_alt_gate": fu["fusion_gate"],
    }
    meta = dict(prediction=pr["prediction"], target=pr["target"],
                date=pr["date_last_observed"], node_pre=node_pre,
                node_post=node_post, mi=mi)
    return dict(stages=stages, meta=meta)


_MI_CACHE = {}


def market_indices():
    if "mi" not in _MI_CACHE:
        n = json.loads((P1 / "seed42" / "node_metadata.json").read_text())
        _MI_CACHE["mi"] = ({k: tuple(v) for k, v in n["market_indices"].items()}, n)
    return _MI_CACHE["mi"]


def run_val_diagnostics() -> dict:
    var_rows, dyn_rows, dim_rows, geo_rows, knn_rows, cka_rows, disp_rows = ([] for _ in range(7))
    pairs = [("S_node_prgnn", "S_node_prepool"), ("S_node_prepool", "S_pool_stock"),
             ("S_node_prepool", "S_hs"), ("S_node_prepool", "S_s"),
             ("S_hs", "S_s"), ("T_cat", "T_ht"), ("T_ht", "T_t"),
             ("F1_s_t", "F2_fused"), ("F0_hs_ht", "F1_s_t")]
    cka_pairs = [("S_hs", "T_ht"), ("S_s", "T_t"), ("T_uH", "T_uZ"),
                 ("F1_s_t", "F2_fused"), ("S_node_prepool", "S_hs"),
                 ("T_cat", "T_ht"), ("S_hs", "F2_fused")]
    for seed in SEEDS:
        loaded = load_val_stages(seed)
        stages, meta = loaded["stages"], loaded["meta"]
        for name, X in stages.items():
            var_rows.append(dict(seed=seed, stage=name, **variation_stats(X)))
            dyn_rows.append(dict(seed=seed, stage=name, **dynamics_stats(X)))
            dim_rows.append(dict(seed=seed, stage=name, **effective_dimension(X)))
        for a, b in pairs:
            geo_rows.append(dict(seed=seed, pair=f"{a}->{b}",
                                 **geometry_preservation(stages[a], stages[b])))
            for k in K_VALUES:
                knn_rows.append(dict(seed=seed, pair=f"{a}->{b}", k=k,
                                     retention=knn_retention(stages[a], stages[b], k)))
        for a, b in cka_pairs:
            cka_rows.append(dict(seed=seed, pair=f"{a}|{b}",
                                 cka=linear_cka(stages[a], stages[b])))
        disp = cross_sectional_dispersion(meta["node_post"], meta["mi"])
        disp.insert(0, "seed", seed)
        disp.insert(1, "date", meta["date"])
        disp_rows.append(disp)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(var_rows).to_csv(OUT / "stage_variation.csv", index=False)
    pd.DataFrame(dyn_rows).to_csv(OUT / "stage_dynamics.csv", index=False)
    pd.DataFrame(dim_rows).to_csv(OUT / "effective_dimension.csv", index=False)
    pd.DataFrame(geo_rows).to_csv(OUT / "geometry_preservation.csv", index=False)
    pd.DataFrame(knn_rows).to_csv(OUT / "knn_retention.csv", index=False)
    pd.DataFrame(cka_rows).to_csv(OUT / "representation_cka.csv", index=False)
    pd.concat(disp_rows, ignore_index=True).to_csv(
        OUT / "spatial_cross_sectional_dispersion.csv", index=False)
    return dict(var=var_rows, dyn=dyn_rows, dim=dim_rows, geo=geo_rows,
                knn=knn_rows, cka=cka_rows)


# ------------------------------------------------ frozen TRAIN extraction
def extract_split(seed: int, split: str, device) -> dict:
    """Frozen stage extraction for one split (used for the probe TRAIN side)."""
    data = build_data(SimpleNamespace(batch_size=config.BATCH_SIZE,
                                      seq_len=config.SEQ_LEN, seed=seed))
    mi = {k: tuple(v) for k, v in data["market_indices"].items()}
    model = HeteroMixHopCMGM(num_nodes=int(data["n_nodes"]),
                             n_commodities=int(data["n_commodities"]),
                             n_stock=int(mi["stock"][1] - mi["stock"][0]),
                             n_bond=int(mi["bond"][1] - mi["bond"][0]),
                             commodity_indices=(int(mi["commodity"][0]),
                                                int(mi["commodity"][1])),
                             variant="switching_latent_balanced_readout")
    ck = CKPT_ROOT / VARIANTS[seed] / "best.pt"
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()
    buf = {}
    handles = [
        model.type_pool.register_forward_pre_hook(
            lambda m, i: buf.__setitem__("node_post", i[0].detach())),
        model.type_pool.register_forward_hook(
            lambda m, i, o: buf.__setitem__("hs", o.detach())),
        model.gcn_proj.register_forward_hook(
            lambda m, i, o: buf.__setitem__("s", o.detach())),
        model.lstm_proj.register_forward_hook(
            lambda m, i, o: buf.__setitem__("t", o.detach())),
        model.head.register_forward_pre_hook(
            lambda m, i: buf.__setitem__("fused", i[0].detach())),
    ]
    b = model.switching_latent_transformer
    acc = {k: [] for k in ("node_post", "hs", "s", "t", "fused", "H", "Z", "cat", "ht")}
    y_acc = []
    loader = DataLoader(data["loaders"][split].dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, drop_last=False, num_workers=0)
    try:
        with torch.no_grad():
            for x, y in loader:
                buf.clear()
                out = model(x.to(device))
                acc["node_post"].append(buf["node_post"].cpu().numpy().astype(np.float32))
                acc["hs"].append(buf["hs"].cpu().numpy().astype(np.float32))
                acc["s"].append(buf["s"].cpu().numpy().astype(np.float32))
                acc["t"].append(buf["t"].cpu().numpy().astype(np.float32))
                acc["fused"].append(buf["fused"].cpu().numpy().astype(np.float32))
                acc["H"].append(b.last_h_last.cpu().numpy().astype(np.float32))
                acc["Z"].append(b.last_z_last.cpu().numpy().astype(np.float32))
                acc["cat"].append(b.last_balanced_concat.cpu().numpy().astype(np.float32))
                acc["ht"].append(b.last_h_temporal.cpu().numpy().astype(np.float32))
                y_acc.append(y.numpy().astype(np.float32))
    finally:
        for h in handles:
            h.remove()
    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out["target"] = np.concatenate(y_acc, axis=0)
    out["mi"] = mi
    return out


def verify_val_extraction(seed: int, ext: dict) -> dict:
    """The re-extracted VAL stages must match Phase 1 bitwise."""
    sp = np.load(P1 / f"seed{seed}" / "spatial_signals.npz")
    tp = np.load(P1 / f"seed{seed}" / "temporal_signals.npz")
    fu = np.load(P1 / f"seed{seed}" / "fusion_signals.npz")
    pr = np.load(P1 / f"seed{seed}" / "predictions.npz")
    checks = {
        "node_post_vs_type_pool_input": float(np.abs(ext["node_post"] - sp["type_pool_input"]).max()),
        "hs_vs_spatial_readout": float(np.abs(ext["hs"] - sp["spatial_readout"]).max()),
        "s_vs_spatial_projected": float(np.abs(ext["s"] - fu["spatial_projected"]).max()),
        "t_vs_temporal_projected": float(np.abs(ext["t"] - fu["temporal_projected"]).max()),
        "fused_vs_fused_representation": float(np.abs(ext["fused"] - fu["fused_representation"]).max()),
        "H_vs_H_final": float(np.abs(ext["H"] - tp["H_final"]).max()),
        "Z_vs_Z_final": float(np.abs(ext["Z"] - tp["Z_final"]).max()),
        "cat_vs_balanced_concat": float(np.abs(ext["cat"] - tp["balanced_concat"]).max()),
        "ht_vs_temporal_readout": float(np.abs(ext["ht"] - tp["temporal_readout"]).max()),
        "target_vs_phase1_target": float(np.abs(ext["target"] - pr["target"]).max()),
    }
    checks["all_bitwise_equal"] = bool(all(v == 0.0 for v in checks.values()))
    return checks


# ------------------------------------------------------------ linear probes
def ridge_probe(Xtr, Ytr, Xva, Yva, alpha: float, n_commodities: int) -> dict:
    """Closed-form ridge on standardized features; no VAL-based selection.

    The target is stored flat as (B, H*Nc) for regression, so predictions are
    reshaped back to (B, H, Nc) before any metric is taken. Indexing a flat
    (B, H*Nc) array by horizon would select the first few commodity columns
    instead of a horizon block.
    """
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    A = (Xtr - mu) / sd
    B = (Xva - mu) / sd
    d = A.shape[1]
    W = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ Ytr)
    pred = (B @ W).reshape(-1, len(HORIZONS), n_commodities)
    tgt = Yva.reshape(-1, len(HORIZONS), n_commodities)
    out = {}
    with torch.no_grad():
        p, t = torch.from_numpy(pred.astype(np.float32)), torch.from_numpy(tgt.astype(np.float32))
        crit = torch.nn.HuberLoss(delta=config.HUBER_DELTA, reduction="none")
        out["probe_Huber"] = float(sum(crit(p[:, i], t[:, i]).mean()
                                       for i in range(len(HORIZONS))))
    for i, h in enumerate(HORIZONS):
        m = population_metrics(pred[:, i], tgt[:, i])
        out[f"probe_MAE_{h}"] = float(m["MAE"])
        out[f"probe_RMSE_{h}"] = float(m["RMSE"])
        out[f"probe_Hit{h}_percent"] = float(100.0 * m["Hit"])
    return out


def y_targets(target: np.ndarray) -> np.ndarray:
    return target.reshape(target.shape[0], -1)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    diag = run_val_diagnostics()
    print(f"VAL diagnostics written: {len(diag['var'])} stage rows across {len(SEEDS)} seeds")

    audit = dict(
        phase="5A", generated_utc=datetime.utcnow().isoformat() + "Z",
        stage_note=STAGE_NOTE, seeds=list(SEEDS),
        test_read=False, training=False,
        vocabulary_note="variance/effective dimension are not information; the "
                        "diagnostics describe compression, attenuation, geometry "
                        "and predictive accessibility only",
        phase1_input_hashes={f"seed{s}/{f}": sha256_file(P1 / f"seed{s}" / f)
                             for s in SEEDS for f in
                             ("predictions.npz", "temporal_signals.npz",
                              "spatial_signals.npz", "fusion_signals.npz")},
    )

    if not args.skip_probes:
        val_checks, probe_rows = {}, []
        # TRAIN representations for fitting; VAL re-extracted and verified bitwise
        for seed in SEEDS:
            tr = extract_split(seed, "train", device)
            va = extract_split(seed, "val", device)
            val_checks[str(seed)] = verify_val_extraction(seed, va)
            mi = tr["mi"]
            n_comm = int(tr["target"].shape[-1])
            Ytr, Yva = y_targets(tr["target"]), y_targets(va["target"])
            reps = {
                "S_node_prepool_stock":
                    tr["node_post"][:, mi["stock"][0]:mi["stock"][1], :].reshape(len(tr["target"]), -1),
                "S_node_prepool_commodity":
                    tr["node_post"][:, mi["commodity"][0]:mi["commodity"][1], :].reshape(len(tr["target"]), -1),
                "S_hs": tr["hs"], "T_H": tr["H"], "T_Z": tr["Z"],
                "T_cat": tr["cat"], "T_ht": tr["ht"],
                "F0_hs_ht": np.concatenate([tr["hs"], tr["ht"]], axis=1),
                "F1_s_t": np.concatenate([tr["s"], tr["t"]], axis=1),
                "F2_fused": tr["fused"],
            }
            reps_val = {
                "S_node_prepool_stock":
                    va["node_post"][:, mi["stock"][0]:mi["stock"][1], :].reshape(len(va["target"]), -1),
                "S_node_prepool_commodity":
                    va["node_post"][:, mi["commodity"][0]:mi["commodity"][1], :].reshape(len(va["target"]), -1),
                "S_hs": va["hs"], "T_H": va["H"], "T_Z": va["Z"],
                "T_cat": va["cat"], "T_ht": va["ht"],
                "F0_hs_ht": np.concatenate([va["hs"], va["ht"]], axis=1),
                "F1_s_t": np.concatenate([va["s"], va["t"]], axis=1),
                "F2_fused": va["fused"],
            }
            for name in reps:
                m_tr, m_va = reps[name].astype(np.float64), reps_val[name].astype(np.float64)
                # pre-registered PCA for the wide node representations, fit on TRAIN
                if m_tr.shape[1] > 512:
                    mu = m_tr.mean(0, keepdims=True)
                    _, _, Vt = np.linalg.svd(m_tr - mu, full_matrices=False)
                    for dim in PCA_DIMS:
                        d = min(dim, Vt.shape[0])
                        Ztr = (m_tr - mu) @ Vt[:d].T
                        Zva = (m_va - mu) @ Vt[:d].T
                        for alpha in RIDGE_SENSITIVITY:
                            probe_rows.append(dict(
                                seed=seed, representation=f"{name}_pca{d}",
                                pca_dim=d, alpha=alpha, is_primary=alpha == RIDGE_ALPHA,
                                **ridge_probe(Ztr, Ytr, Zva, Yva, alpha, n_comm)))
                else:
                    for alpha in RIDGE_SENSITIVITY:
                        probe_rows.append(dict(
                            seed=seed, representation=name, pca_dim=None,
                            alpha=alpha, is_primary=alpha == RIDGE_ALPHA,
                            **ridge_probe(m_tr, Ytr, m_va, Yva, alpha, n_comm)))
            print(f"seed{seed}: probes done; VAL extraction bitwise-identical to "
                  f"Phase 1 = {val_checks[str(seed)]['all_bitwise_equal']}")
        pd.DataFrame(probe_rows).to_csv(OUT / "linear_probe_metrics.csv", index=False)
        audit["probe_protocol"] = dict(
            model="ridge regression (closed form), no MLP/forest/kernel",
            alpha_primary=RIDGE_ALPHA, alpha_sensitivity=list(RIDGE_SENSITIVITY),
            fit_on="TRAIN representations", evaluate_on="VAL representations",
            pca_dims=list(PCA_DIMS),
            pca_fit="TRAIN only, unsupervised, pre-registered dimensions",
            note="diagnostic linear probe; not comparable to main-model performance")
        audit["val_extraction_bitwise_checks"] = val_checks
        audit["val_extraction_verified"] = all(v["all_bitwise_equal"] for v in val_checks.values())

    audit["phase1_hashes_after"] = {k: sha256_file(P1 / k) for k in audit["phase1_input_hashes"]}
    audit["phase1_unchanged"] = all(
        audit["phase1_input_hashes"][k] == audit["phase1_hashes_after"][k]
        for k in audit["phase1_input_hashes"])
    write_json(P5 / "audit" / "phase5a_audit.json", audit)
    print("Phase 5A complete; phase1_unchanged =", audit["phase1_unchanged"])


if __name__ == "__main__":
    main()
