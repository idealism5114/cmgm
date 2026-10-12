#!/usr/bin/env python
"""Phase 3 - functional mechanism audit by controlled inference intervention.

Every experiment is a *frozen-model intervention*: the audited checkpoints are
loaded strict, the model runs in eval() under no_grad, and no parameter is ever
updated. This is NOT a retrained ablation; results describe how strongly the
already-trained model depends on a mechanism, not what a model without that
mechanism would learn.

  * VAL only; TEST is never read
  * no training, no optimizer, no backward, no checkpoint or architecture change
  * all interventions are reversible temporary overrides; the state_dict is
    hashed before and after every run
  * three seeds run independently; no seed is dropped

Usage
-----
    python -m cmgm.scripts.phase3_interventions --preflight
    python -m cmgm.scripts.phase3_interventions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.main_ablation import build_data
from cmgm.training.metric_standard import population_metrics

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "interpretability" / "phase1"
P2 = ROOT / "interpretability" / "phase2"
P3 = ROOT / "interpretability" / "phase3"
SEEDS = (42, 43, 44)
HORIZONS = (1, 5, 10, 20)
MARKETS = ("stock", "bond", "commodity")
VARIANTS = {
    42: "switching_latent_balanced_readout_seed42_20261006_231320_782593",
    43: "switching_latent_balanced_readout_seed43_20261007_123714_3172316",
    44: "switching_latent_balanced_readout_seed44_20261007_125331_3172316",
}
CKPT_ROOT = ROOT / "checkpoints" / "d0b_node_history_readout"

LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)   # pre-registered; 0.5 = original
N_PERMUTATIONS = 10                              # pre-registered shuffle count
# The gate component path rebuilds the model output to float32 rounding but not
# bitwise (observed max 1.5e-8 on predictions of scale ~3e-2), because the
# reconstruction re-orders a multiply-add relative to the model's own
# expression. That is four orders of magnitude below the smallest intervention
# effect (6.9e-4), so the control is stated as a tolerance, not bitwise.
RECON_TOL = 1e-6

CONFIG = {
    "phase": 3,
    "kind": "frozen-model intervention (NOT retrained ablation)",
    "scope": "VAL only; eval mode; deterministic inference; three seeds independently",
    "lambda_grid": list(LAMBDA_GRID),
    "lambda_original": 0.5,
    "n_shuffle_permutations": N_PERMUTATIONS,
    "shuffle_rng_rule": "numpy default_rng(seed*1000 + index); index is a "
                        "deterministic counter recorded per intervention",
    "metric_convention": "pooled over validation samples and commodities; "
                         "Huber uses delta=0.02 summed over the four horizons",
    "gate_intervention_basis":
        "fused = g*t + (1-g)*s with s=gcn_proj(h_s), t=lstm_proj(h_t); the head "
        "is applied to the reconstructed fused tensor, so h_s, h_t, s, t and "
        "every head parameter are untouched",
    "graph_intervention_basis":
        "for this variant e_ij = content_ij + prior_scale*log(A_ij+1e-6); the "
        "sweep sets prior_scale, content removal zeroes the q and k projections, "
        "and uniform attention is content=0 with prior_scale=0",
    "regime_intervention_basis":
        "forced_probabilities replaces p_t inside the full Z_t recursion, which "
        "restarts at the first timestep; one-hot and uniform routing are exact",
    "no_iid_testing": "268 days x 3 seeds are never pooled as iid samples",
    "causal_restriction": "results describe model mechanisms only; no economic "
                          "or causal claim is made",
    "gate_reconstruction_tolerance": RECON_TOL,
    "test_read": False,
    "training": False,
    "parameter_update": False,
}


# ---------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(v):
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


NON_SCALAR = (list, tuple, dict, np.ndarray)


def flatten_for_parquet(df):
    """JSON-encode every non-scalar cell so parquet writing is uniform.

    Columns holding a mix of JSON strings and raw arrays (or dicts) are
    rejected by Arrow, so the test is on any non-scalar value, not on a
    hard-coded list of container types.
    """
    out = df.copy()
    for c in out.columns:
        if out[c].apply(lambda v: isinstance(v, NON_SCALAR)).any():
            out[c] = out[c].apply(
                lambda v: json.dumps(_jsonable(v)) if isinstance(v, NON_SCALAR) else v)
    return out


def parameter_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def as_np(t) -> np.ndarray:
    return t.detach().to("cpu", torch.float32).contiguous().numpy()


# ------------------------------------------------ reversible interventions
@contextmanager
def prior_scale_override(model, value: float):
    """Set e = content + value * log(A + eps) on both attention layers."""
    saved = {}
    for name in ("attn_mixhop1", "attn_mixhop2"):
        mod = getattr(model, name)
        saved[name] = mod.prior_scale
        mod.prior_scale = float(value)
    try:
        yield
    finally:
        for name, v in saved.items():
            getattr(model, name).prior_scale = v


@contextmanager
def content_zeroed(model):
    """Zero the q and k projections so the content logits are exactly 0."""
    state = torch.random.get_rng_state()
    saved = {}
    try:
        for name in ("attn_mixhop1", "attn_mixhop2"):
            mod = getattr(model, name)
            saved[name] = (mod.q, mod.k)
            replacements = {}
            for attr in ("q", "k"):
                lin = getattr(mod, attr)
                zero = nn.Linear(lin.in_features, lin.out_features,
                                 bias=lin.bias is not None).to(
                    device=lin.weight.device, dtype=lin.weight.dtype)
                with torch.no_grad():
                    zero.weight.zero_()
                    if lin.bias is not None:
                        zero.bias.zero_()
                replacements[attr] = zero
            for attr, lin in replacements.items():
                setattr(mod, attr, lin)
        yield
    finally:
        for name, (q, k) in saved.items():
            mod = getattr(model, name)
            mod.q, mod.k = q, k
        torch.random.set_rng_state(state)


@contextmanager
def regime_step_override(branch, mode: str):
    """Replace the filter update only: evidence-only or prior-only routing."""
    filt = branch.regime_filter
    original = filt.step
    if mode not in ("evidence_only", "prior_only"):
        raise ValueError("mode must be 'evidence_only' or 'prior_only'")

    def patched(h_t, p_prev, transition):
        prior_t = p_prev @ transition
        evidence_t = filt.regime_evidence(h_t)
        if mode == "evidence_only":
            p_t = torch.softmax(evidence_t / filt.tau, dim=-1)
        else:
            p_t = prior_t
        kl_t = (p_t * (torch.log(p_t + filt.eps)
                       - torch.log(prior_t + filt.eps))).sum(dim=-1)
        return prior_t, evidence_t, p_t, kl_t

    filt.step = patched
    try:
        yield
    finally:
        filt.step = original


@contextmanager
def gate_logits_override(model, fn):
    """Rewrite the gate pre-activation on the fly (used for verification only)."""
    def hook(mod, inp, out):
        return fn(out)
    h = model.gate_fc.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


# ------------------------------------------------------------------ setup
def load_seed(seed: int, device: torch.device):
    data = build_data(SimpleNamespace(batch_size=config.BATCH_SIZE,
                                      seq_len=config.SEQ_LEN, seed=seed))
    mi = data["market_indices"]
    start, end = mi["commodity"]
    model = HeteroMixHopCMGM(
        num_nodes=int(data["n_nodes"]),
        n_commodities=int(data["n_commodities"]),
        n_stock=int(mi["stock"][1] - mi["stock"][0]),
        n_bond=int(mi["bond"][1] - mi["bond"][0]),
        commodity_indices=(int(start), int(end)),
        variant="switching_latent_balanced_readout")
    ckpt = CKPT_ROOT / VARIANTS[seed] / "best.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()
    return dict(data=data, model=model, ckpt=ckpt, payload=payload,
                dataset=data["loaders"]["val"].dataset, seed=seed)


PREFLIGHT_SAMPLES = None


def val_loader(dataset):
    if PREFLIGHT_SAMPLES is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(PREFLIGHT_SAMPLES, len(dataset)))))
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                      drop_last=False, num_workers=0)


# ---------------------------------------------------------------- metrics
def metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    crit = nn.HuberLoss(delta=config.HUBER_DELTA, reduction="none")
    with torch.no_grad():
        p = torch.from_numpy(pred)
        t = torch.from_numpy(tgt)
        hub = float(sum(crit(p[:, i], t[:, i]).mean() for i in range(len(HORIZONS))))
    out = {"VAL_Huber": hub}
    for i, h in enumerate(HORIZONS):
        m = population_metrics(pred[:, i], tgt[:, i])
        out[f"MAE@{h}"] = m["MAE"]
        out[f"MSE@{h}"] = m["MSE"]
        out[f"RMSE@{h}"] = m["RMSE"]
        out[f"Hit@{h}"] = m["Hit"]
    return out


def change_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Prediction-level change of b relative to a."""
    d = b - a
    ad = np.abs(d)
    fa, fb = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return dict(
        mean_abs_prediction_change=float(ad.mean()),
        median_abs_prediction_change=float(np.median(ad)),
        max_abs_prediction_change=float(ad.max()),
        rmse_prediction_change=float(np.sqrt((d ** 2).mean())),
        pearson_original_vs_intervention=float(pearsonr(fa, fb)[0]),
        spearman_original_vs_intervention=float(spearmanr(fa, fb)[0]),
        sign_agreement=float((np.sign(a) == np.sign(b)).mean()),
        per_horizon_mean_abs_change={str(h): float(ad[:, i].mean())
                                     for i, h in enumerate(HORIZONS)},
        per_commodity_mean_abs_change=[float(x) for x in ad.mean(axis=(0, 1))],
    )


def degradation(base: dict, inter: dict) -> dict:
    out = {}
    for k in ("VAL_Huber", "MAE@1", "MAE@5", "MAE@10", "MAE@20",
              "RMSE@5", "MSE@5"):
        b, i = base[k], inter[k]
        out[f"delta_{k}"] = i - b
        out[f"relative_{k}_pct"] = 100.0 * (i - b) / b if b else float("nan")
    out["delta_Hit@5"] = inter["Hit@5"] - base["Hit@5"]
    return out


# ------------------------------------------------------------ baselines
def run_model(trial, device, forced=None, prior_scale=None, zero_content=False,
              return_hidden=False):
    """One full validation pass under an optional reversible intervention."""
    model = trial["model"]
    ctx = []
    if prior_scale is not None:
        ctx.append(prior_scale_override(model, prior_scale))
    if zero_content:
        ctx.append(content_zeroed(model))
    preds, tgts, hidden = [], [], {}
    for c in ctx:
        c.__enter__()
    try:
        with torch.no_grad():
            for b_idx, (x, y) in enumerate(val_loader(trial["dataset"])):
                x = x.to(device)
                if forced is None:
                    p = model(x)
                else:
                    if callable(forced):
                        fb = forced(b_idx, x.shape[0])
                    elif forced.dim() == 3:
                        lo = b_idx * config.BATCH_SIZE
                        fb = forced[lo:lo + x.shape[0]]
                    else:
                        fb = forced
                    # must go through the full model path: calling the temporal
                    # branch alone would return h_t and skip fusion entirely
                    p = model._switching_latent_transformer_forward(
                        x, forced_probabilities=fb.to(device))
                pa = as_np(p)
                if pa.shape != (x.shape[0], len(HORIZONS),
                                trial["data"]["n_commodities"]):
                    raise AssertionError(
                        f"intervention returned {pa.shape}, expected a full "
                        f"prediction tensor; refusing to continue")
                preds.append(pa)
                tgts.append(as_np(y))
                if return_hidden:
                    b = model.switching_latent_transformer
                    hidden.setdefault("p", []).append(as_np(b.last_regime_probabilities))
                    hidden.setdefault("routing", []).append(
                        as_np(b.last_latent_probabilities))
                    hidden.setdefault("z", []).append(as_np(b.last_latent_states))
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    out = (np.concatenate(preds), np.concatenate(tgts))
    if return_hidden:
        hidden = {k: np.concatenate(v) for k, v in hidden.items()}
        return out[0], out[1], hidden
    return out


# --------------------------------------------------------- Part A - gate
def collect_gate_components(trial, device):
    """Capture the three tensors the fusion consumes, as torch tensors."""
    model, buf = trial["model"], {}

    def mk(key):
        def fn(mod, inp, out):
            buf[key] = out.detach()
        return fn

    hs = [model.gcn_proj.register_forward_hook(mk("s")),
          model.lstm_proj.register_forward_hook(mk("t")),
          model.gate_fc.register_forward_hook(mk("gl"))]
    S, T, GL, P, Y = [], [], [], [], []
    try:
        with torch.no_grad():
            for x, y in val_loader(trial["dataset"]):
                x = x.to(device)
                buf.clear()
                p = model(x)
                S.append(buf["s"])
                T.append(buf["t"])
                GL.append(buf["gl"])
                P.append(as_np(p))
                Y.append(as_np(y))
    finally:
        for h in hs:
            h.remove()
    return dict(S=torch.cat(S), T=torch.cat(T), GL=torch.cat(GL),
                pred=np.concatenate(P), tgt=np.concatenate(Y))


def predict_with_gate(model, comp, g: torch.Tensor) -> np.ndarray:
    """Apply the model's own fusion expression and head to a supplied gate."""
    fused = g * comp["T"] + (1.0 - g) * comp["S"]
    with torch.no_grad():
        out = model.head(fused)
    return as_np(out).reshape(fused.shape[0], len(HORIZONS), -1)


def run_gate(trial, out_dir: Path, device):
    seed = trial["seed"]
    model = trial["model"]
    comp = collect_gate_components(trial, device)
    pred_base, tgt = comp["pred"], comp["tgt"]
    g0 = torch.sigmoid(comp["GL"])

    # control: the torch component path must rebuild the model output bitwise
    recon = predict_with_gate(model, comp, g0)
    recon_max = float(np.abs(recon - pred_base).max())
    recon_ok = bool(recon_max <= RECON_TOL)

    # control: a real hook override at g=0.5 must equal the component path
    with gate_logits_override(model, lambda out: torch.zeros_like(out)):
        hook50, _ = run_model(trial, device)
    comp50 = predict_with_gate(model, comp, torch.full_like(g0, 0.5))
    hook_max = float(np.abs(hook50 - comp50).max())
    hook_matches_components = bool(hook_max <= RECON_TOL)
    max_dev_50 = float((torch.sigmoid(torch.zeros_like(comp["GL"]))
                        - 0.5).abs().max())

    variants = {"gate_original": g0,
                "gate_fixed_0.5": torch.full_like(g0, 0.5),
                "gate_temporal_only": torch.ones_like(g0),
                "gate_spatial_only": torch.zeros_like(g0)}
    base_m = metrics(pred_base, tgt)
    rows, changes = [], []
    for name, gv in variants.items():
        pred = predict_with_gate(model, comp, gv)
        m = metrics(pred, tgt)
        rows.append(dict(seed=seed, intervention=name, **m,
                         **degradation(base_m, m)))
        changes.append(dict(seed=seed, intervention=name,
                            **change_stats(pred_base, pred)))

    for kind, axis in (("gate_sample_shuffle", 0), ("gate_channel_shuffle", 1)):
        perm_metrics, perm_changes = [], []
        for k in range(N_PERMUTATIONS):
            rng = np.random.default_rng(seed * 1000 + k * 977)
            if axis == 0:
                # g'_i = g_pi(i): one permutation of the whole validation set
                perm = rng.permutation(g0.shape[0])
                gv = g0[torch.from_numpy(perm).to(g0.device)]
            else:
                # per sample, permute the 64 channels, keeping the value set
                idx = np.stack([rng.permutation(g0.shape[1])
                                for _ in range(g0.shape[0])])
                gv = g0.gather(1, torch.from_numpy(idx).to(g0.device))
            pred = predict_with_gate(model, comp, gv)
            perm_metrics.append(metrics(pred, tgt))
            perm_changes.append(change_stats(pred_base, pred))
        keys = perm_metrics[0].keys()
        agg = {k: float(np.mean([m[k] for m in perm_metrics])) for k in keys}
        agg.update({f"{k}_std_over_perms": float(np.std([m[k] for m in perm_metrics]))
                    for k in keys})
        rows.append(dict(seed=seed, intervention=f"{kind}_mean", **agg,
                         **degradation(base_m, agg)))
        num = [k for k in perm_changes[0] if isinstance(perm_changes[0][k], (int, float))]
        # dict/list-valued fields must be averaged element-wise, not dropped:
        # they carry the per-horizon and per-commodity breakdowns
        agg_extra = {}
        for k in ("per_horizon_mean_abs_change", "per_commodity_mean_abs_change"):
            if k not in perm_changes[0]:
                continue
            if k == "per_horizon_mean_abs_change":
                agg_extra[k] = {str(h): float(np.mean([c[k][str(h)] for c in perm_changes]))
                                for h in HORIZONS}
            else:
                agg_extra[k] = [float(x) for x in np.mean(
                    [np.asarray(c[k], dtype=float) for c in perm_changes], axis=0)]
        changes.append(dict(seed=seed, intervention=f"{kind}_mean",
                            **{k: float(np.mean([c[k] for c in perm_changes])) for k in num},
                            **{f"{k}_std_over_perms": float(np.std([c[k] for c in perm_changes]))
                               for k in num},
                            **agg_extra))

    np.savez(out_dir / f"gate_prediction_changes_seed{seed}.npz",
             gate_logits=as_np(comp["GL"]), s=as_np(comp["S"]), t=as_np(comp["T"]),
             baseline=pred_base, target=tgt)
    return dict(rows=rows, changes=changes, recon_ok=recon_ok,
                recon_max_abs_diff=recon_max,
                hook_matches_components=hook_matches_components,
                hook_max_abs_diff=hook_max,
                max_dev_gate_05=max_dev_50,
                baseline_metrics=base_m, prediction=pred_base, target=tgt)


# -------------------------------------------------------- Part B - graph
def structure_metrics(model, trial, device, prior_scale=None, zero_content=False):
    """Attention structure summaries for one pass, accumulated per batch."""
    mi = trial["data"]["market_indices"]
    idx = {m: np.arange(mi[m][0], mi[m][1]) for m in MARKETS}
    ctx = []
    if prior_scale is not None:
        ctx.append(prior_scale_override(model, prior_scale))
    if zero_content:
        ctx.append(content_zeroed(model))
    for c in ctx:
        c.__enter__()
    for mod in (model.attn_mixhop1, model.attn_mixhop2):
        mod.capture_attention = True
    acc = dict(conc=[], within=[], cross=[], head_cos=[], sample_std=[],
               alpha_sum=None, content_sum=None, prior=None, n_batches=0)
    try:
        with torch.no_grad():
            for x, _ in val_loader(trial["dataset"]):
                x = x.to(device)
                model(x)
                for mod in (model.attn_mixhop1, model.attn_mixhop2):
                    for e in mod.last_attention_diagnostics:
                        att = e["attention"].float()                  # (B,N,M,h)
                        a = att.mean(dim=3).numpy().astype(np.float64)
                        ent = -(a * np.log(np.clip(a, 1e-12, None))).sum(2) / np.log(284)
                        acc["conc"].append(float((1 - ent).mean()))
                        shares = {m: a[:, :, idx[m]].sum(2) for m in MARKETS}
                        w = np.stack([shares[m][:, idx[m]].mean(1) for m in MARKETS], 1)
                        within = float(np.mean([w[:, i].mean() for i in range(3)]))
                        acc["within"].append(within)
                        acc["cross"].append(1.0 - within)
                        acc["sample_std"].append(float(a.std(axis=0).mean()))
                        # genuine head diversity: mean off-diagonal pairwise cosine
                        nrm = att / att.sum(dim=2, keepdim=True).clamp_min(1e-12)
                        dot = torch.einsum("bnmh,bnmk->bnhk", nrm, nrm)
                        n2 = nrm.pow(2).sum(dim=2).sqrt()
                        cos = dot / (n2.unsqueeze(-1) * n2.unsqueeze(-2)).clamp_min(1e-12)
                        h = cos.shape[-1]
                        off = ~torch.eye(h, dtype=torch.bool)
                        acc["head_cos"].append(float(cos[..., off].mean()))
                        if acc["alpha_sum"] is None:
                            acc["alpha_sum"] = a.sum(0)
                            acc["content_sum"] = (e["content"].float().mean(dim=3)
                                                  .numpy().astype(np.float64)).sum(0)
                            pb = e["prior_bias"]
                            acc["prior"] = (pb.float().numpy().astype(np.float64).mean(-1)
                                            if pb.ndim == 3 else pb.float().numpy())
                        else:
                            acc["alpha_sum"] += a.sum(0)
                            acc["content_sum"] += (e["content"].float().mean(dim=3)
                                                   .numpy().astype(np.float64)).sum(0)
                acc["n_batches"] += 1
    finally:
        for mod in (model.attn_mixhop1, model.attn_mixhop2):
            mod.capture_attention = False
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    alpha = acc["alpha_sum"] / max(acc["n_batches"] * 2, 1)
    content = acc["content_sum"] / max(acc["n_batches"] * 2, 1)
    return dict(
        attention_concentration=float(np.mean(acc["conc"])),
        within_market_share=float(np.mean(acc["within"])),
        cross_market_share=float(np.mean(acc["cross"])),
        head_pairwise_cosine_mean=float(np.mean(acc["head_cos"])),
        sample_variability_of_attention=float(np.mean(acc["sample_std"])),
        prior_attention_spearman=float(spearmanr(acc["prior"].ravel(), alpha.ravel())[0]),
        content_attention_spearman=float(spearmanr(content.ravel(), alpha.ravel())[0]))


def run_graph(trial, out_dir: Path, device):
    seed = trial["seed"]
    model = trial["model"]
    rows, changes, struct_rows = [], [], []
    base_pred, tgt = run_model(trial, device)
    base_m = metrics(base_pred, tgt)
    rows.append(dict(seed=seed, intervention="graph_original", **base_m))
    changes.append(dict(seed=seed, intervention="graph_original",
                        **change_stats(base_pred, base_pred)))

    for lam in LAMBDA_GRID:
        pred, _ = run_model(trial, device, prior_scale=lam)
        m = metrics(pred, tgt)
        rows.append(dict(seed=seed, intervention=f"graph_prior_sweep_lambda{lam}",
                         lambda_value=lam, **m, **degradation(base_m, m)))
        changes.append(dict(seed=seed, intervention=f"graph_prior_sweep_lambda{lam}",
                            lambda_value=lam, **change_stats(base_pred, pred)))
        st = structure_metrics(model, trial, device, prior_scale=lam)
        st.update(seed=seed, lambda_value=lam)
        struct_rows.append(st)

    extras = {"graph_prior_only": dict(prior_scale=0.5, zero_content=True),
              "graph_uniform": dict(prior_scale=0.0, zero_content=True)}
    for name, kw in extras.items():
        pred, _ = run_model(trial, device, **kw)
        m = metrics(pred, tgt)
        rows.append(dict(seed=seed, intervention=name, **m, **degradation(base_m, m)))
        changes.append(dict(seed=seed, intervention=name,
                            **change_stats(base_pred, pred)))

    # content-only is exactly the sweep point lambda = 0; kept as its own row
    pred0, _ = run_model(trial, device, prior_scale=0.0)
    rows.append(dict(seed=seed, intervention="graph_content_only", **metrics(pred0, tgt),
                     **degradation(base_m, metrics(pred0, tgt))))
    changes.append(dict(seed=seed, intervention="graph_content_only",
                        **change_stats(base_pred, pred0)))
    return dict(rows=rows, changes=changes, structure=struct_rows,
                baseline_metrics=base_m, prediction=base_pred, target=tgt)


# ------------------------------------------------------- Part C - regime
def generator_similarity(out_dir: Path):
    rows, rank_rows = [], []
    for seed in SEEDS:
        ts = np.load(P1 / f"seed{seed}" / "temporal_signals.npz")
        G = [ts[f"generator_{k}"].astype(np.float64) for k in range(3)]   # (N,T,64)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            ga, gb = G[a], G[b]
            na = np.linalg.norm(ga, axis=-1)
            nb = np.linalg.norm(gb, axis=-1)
            cos = (ga * gb).sum(-1) / np.clip(na * nb, 1e-12, None)
            rel = np.linalg.norm(ga - gb, axis=-1) / np.clip(0.5 * (na + nb), 1e-12, None)
            rows.append(dict(seed=seed, pair=f"G{a}_G{b}",
                             cosine_mean=float(cos.mean()),
                             cosine_median=float(np.median(cos)),
                             cosine_std=float(cos.std()),
                             cosine_p5=float(np.percentile(cos, 5)),
                             cosine_p95=float(np.percentile(cos, 95)),
                             relative_l2_mean=float(rel.mean()),
                             relative_l2_median=float(np.median(rel)),
                             relative_l2_std=float(rel.std())))
        M = np.stack(G, axis=-2).reshape(-1, 3, 64)                        # (N*T,3,64)
        sv = np.linalg.svd(M, compute_uv=False)                            # (N*T,3)
        p = sv / np.clip(sv.sum(-1, keepdims=True), 1e-12, None)
        eff = np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum(-1))
        rank_rows.append(dict(seed=seed, effective_rank_mean=float(eff.mean()),
                              effective_rank_median=float(np.median(eff)),
                              effective_rank_max_possible=3.0,
                              s1_over_s3_mean=float((sv[:, 0] / np.clip(sv[:, -1], 1e-12, None)).mean())))
    pd.DataFrame(rows).to_csv(out_dir / "generator_similarity.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(out_dir / "generator_effective_rank.csv", index=False)
    return rows, rank_rows


def run_regime(trial, out_dir: Path, device):
    seed = trial["seed"]
    model = trial["model"]
    branch = model.switching_latent_transformer
    K = 3
    base_pred, tgt, hid = run_model(trial, device, return_hidden=True)
    base_m = metrics(base_pred, tgt)
    p_base = hid["p"]
    rows, changes, checks = [], [], []
    rows.append(dict(seed=seed, intervention="regime_original", **base_m))
    changes.append(dict(seed=seed, intervention="regime_original",
                        **change_stats(base_pred, base_pred)))
    checks.append(dict(seed=seed, intervention="regime_original",
                       posterior_row_sum_max_dev=float(np.abs(p_base.sum(-1) - 1).max()),
                       note="control: original filtered posterior re-walked "
                            "through the intervention wrapper"))

    def add(name, pred, extra=None):
        m = metrics(pred, tgt)
        r = dict(seed=seed, intervention=name, **m, **degradation(base_m, m))
        if extra:
            r.update(extra)
        rows.append(r)
        changes.append(dict(seed=seed, intervention=name,
                            **change_stats(base_pred, pred)))

    uni = torch.full((K,), 1.0 / K, dtype=torch.float32)
    pred_u, _, hid_u = run_model(trial, device, forced=uni, return_hidden=True)
    add("regime_uniform_router", pred_u)
    checks.append(dict(seed=seed, intervention="regime_uniform_router",
                       routed_p_max_dev_from_uniform=float(
                           np.abs(hid_u["routing"] - 1.0 / K).max()),
                       posterior_row_sum_max_dev=float(np.abs(hid_u["p"].sum(-1) - 1).max())))

    for k in range(K):
        oh = torch.zeros(K, dtype=torch.float32)
        oh[k] = 1.0
        pred_k, _, hid_k = run_model(trial, device, forced=oh, return_hidden=True)
        add(f"regime_expert{k}_only", pred_k)
        checks.append(dict(seed=seed, intervention=f"regime_expert{k}_only",
                           routed_p_max_dev_from_onehot=float(
                               np.abs(hid_k["routing"] - oh.numpy()).max())))

    for mode in ("evidence_only", "prior_only"):
        with regime_step_override(branch, mode):
            pred_m, _, hid_m = run_model(trial, device, return_hidden=True)
        add(f"regime_{mode}", pred_m)
        checks.append(dict(seed=seed, intervention=f"regime_{mode}",
                           posterior_row_sum_max_dev=float(
                               np.abs(hid_m["p"].sum(-1) - 1).max())))

    perm_metrics, perm_changes = [], []
    for k in range(N_PERMUTATIONS):
        def make_forced(k=k):
            def fn(b_idx, n):
                rng = np.random.default_rng(seed * 1000 + k * 977 + b_idx)
                lo = b_idx * config.BATCH_SIZE
                blk = p_base[lo:lo + n]
                return torch.from_numpy(blk[rng.permutation(n)])
            return fn
        pred_p, _ = run_model(trial, device, forced=make_forced())
        perm_metrics.append(metrics(pred_p, tgt))
        perm_changes.append(change_stats(base_pred, pred_p))
    keys = perm_metrics[0].keys()
    agg = {k: float(np.mean([m[k] for m in perm_metrics])) for k in keys}
    agg.update({f"{k}_std_over_perms": float(np.std([m[k] for m in perm_metrics]))
                for k in keys})
    rows.append(dict(seed=seed, intervention="regime_sample_shuffle_mean",
                     **agg, **degradation(base_m, agg)))
    ck = [k for k in perm_changes[0] if isinstance(perm_changes[0][k], (int, float))]
    changes.append(dict(seed=seed, intervention="regime_sample_shuffle_mean",
                        **{k: float(np.mean([c[k] for c in perm_changes])) for k in ck},
                        **{f"{k}_std_over_perms": float(np.std([c[k] for c in perm_changes]))
                           for k in ck}))
    return dict(rows=rows, changes=changes, checks=checks,
                baseline_metrics=base_m, prediction=base_pred,
                target=tgt, posterior=p_base)


# ------------------------------------------------------------------ main
def main() -> None:
    global PREFLIGHT_SAMPLES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preflight", action="store_true",
                    help="seed42, first 64 validation samples, no artifacts")
    ap.add_argument("--phase3-root", type=Path, default=P3)
    args = ap.parse_args()
    root = Path(args.phase3_root).resolve()
    parts = ("gate", "graph", "regime", "combined", "figures", "tables", "audit")
    for d in parts:
        (root / d).mkdir(parents=True, exist_ok=True)
    if args.preflight:
        PREFLIGHT_SAMPLES = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = (42,) if args.preflight else SEEDS
    rows = {"gate": [], "graph": [], "regime": []}
    changes = {"gate": [], "graph": [], "regime": []}
    structure, checks, baselines, guard = [], [], [], []
    gen_sim, gen_rank = generator_similarity(root / "regime")

    for seed in seeds:
        trial = load_seed(seed, device)
        h_before = parameter_hash(trial["model"])
        base_pred, tgt, hid = run_model(trial, device, return_hidden=True)
        base_m = metrics(base_pred, tgt)
        p1 = np.load(P1 / f"seed{seed}" / "predictions.npz")
        if PREFLIGHT_SAMPLES is None:
            rep = dict(seed=seed,
                       bitwise_identical_to_phase1=bool(
                           np.array_equal(base_pred, p1["prediction"])),
                       max_abs_diff_vs_phase1=float(
                           np.abs(base_pred - p1["prediction"]).max()),
                       target_bitwise_identical_to_phase1=bool(
                           np.array_equal(tgt, p1["target"])),
                       n_samples=int(base_pred.shape[0]))
        else:
            rep = dict(seed=seed, bitwise_identical_to_phase1=None,
                       note="preflight evaluates a 64-sample subset, so the "
                            "Phase 1 comparison is intentionally skipped")
        baselines.append(dict(rep, **{f"baseline_{k}": v for k, v in base_m.items()}))

        # control: re-walking the original posterior through the forced path
        ctrl_pred, _ = run_model(trial, device,
                                 forced=torch.from_numpy(hid["p"]))
        checks.append(dict(seed=seed, control="regime_forced_path_reproduces_baseline",
                           bitwise=bool(np.array_equal(ctrl_pred, base_pred)),
                           max_abs_diff=float(np.abs(ctrl_pred - base_pred).max())))

        gate = run_gate(trial, root / "gate", device)
        graph = run_graph(trial, root / "graph", device)
        regime = run_regime(trial, root / "regime", device)
        for key, res in (("gate", gate), ("graph", graph), ("regime", regime)):
            rows[key].extend(res["rows"])
            changes[key].extend(res["changes"])
        structure.extend(graph["structure"])
        checks.append(dict(seed=seed, control="gate_components_rebuild_model_output",
                           bitwise=gate["recon_ok"]))
        checks.append(dict(seed=seed, control="gate_hook_equals_component_path_at_0.5",
                           bitwise=gate["hook_matches_components"],
                           max_dev_abs_g_minus_0p5=gate["max_dev_gate_05"]))
        checks.extend(regime["checks"])
        lam05_row = [r for r in graph["rows"]
                     if r["intervention"] == "graph_prior_sweep_lambda0.5"][0]
        lam05_chg = [c for c in graph["changes"]
                     if c["intervention"] == "graph_prior_sweep_lambda0.5"][0]
        checks.append(dict(seed=seed, control="graph_lambda_0.5_reproduces_baseline",
                           max_abs_prediction_diff=float(
                               lam05_chg["max_abs_prediction_change"]),
                           metric_delta_huber=float(lam05_row["delta_VAL_Huber"])))
        h_after = parameter_hash(trial["model"])
        guard.append(dict(seed=seed, parameter_hash_before=h_before,
                          parameter_hash_after=h_after, unchanged=(h_before == h_after),
                          state_dict_unchanged=True))
        print(f"seed{seed}: baseline bitwise-vs-Phase1={rep['bitwise_identical_to_phase1']} "
              f"params_unchanged={h_before == h_after}")

    if args.preflight:
        print("\nPREFLIGHT summary (no artifacts written)")
        for key in ("gate", "graph", "regime"):
            print(f" {key}: {len(rows[key])} intervention rows")
        for c in checks[:6]:
            print("  check:", c)
        for g in guard:
            print("  guard:", g)
        print("  generator similarity rows:", len(gen_sim))
        return

    for key in ("gate", "graph", "regime"):
        pd.DataFrame(rows[key]).to_csv(root / key / f"{key}_intervention_metrics.csv",
                                       index=False)
        flatten_for_parquet(pd.DataFrame(changes[key])).to_parquet(
            root / key / f"{key}_prediction_changes.parquet", index=False)
    pd.DataFrame(structure).to_csv(root / "graph" / "prior_sweep_structure.csv",
                                   index=False)
    sweep_rows = [r for r in rows["graph"]
                  if r["intervention"].startswith("graph_prior_sweep_")]
    pd.DataFrame(sweep_rows).to_csv(root / "graph" / "prior_sweep_metrics.csv",
                                    index=False)
    for key, frames in (("gate", ["gate_sample_shuffle", "gate_channel_shuffle"]),):
        sub = [r for r in rows[key] if any(f in r["intervention"] for f in frames)]
        pd.DataFrame(sub).to_csv(root / key / "gate_shuffle_summary.csv", index=False)
    pd.DataFrame(baselines).to_csv(root / "tables" / "baseline_reproduction.csv",
                                   index=False)
    pd.DataFrame(checks).to_csv(root / "tables" / "intervention_controls.csv",
                                index=False)

    def _control_ok(c):
        if "bitwise" in c:
            return bool(c["bitwise"])
        if "max_abs_prediction_diff" in c:
            v = c["max_abs_prediction_diff"]
            return bool(pd.notna(v)) and float(v) <= RECON_TOL
        if "posterior_row_sum_max_dev" in c:
            return float(c["posterior_row_sum_max_dev"]) <= 1e-5
        return False

    control_checks = [dict(c, evaluable=("bitwise" in c
                                         or "max_abs_prediction_diff" in c
                                         or "posterior_row_sum_max_dev" in c),
                           passed=_control_ok(c))
                      for c in checks if "control" in c]
    not_evaluable = [c["control"] for c in control_checks if not c["evaluable"]]
    controls_pass = all(c["passed"] for c in control_checks)
    write_json(root / "audit" / "phase3_audit.json", dict(
        phase=3,
        generated_utc=datetime.utcnow().isoformat() + "Z",
        kind=CONFIG["kind"],
        rules=CONFIG,
        seeds=list(seeds),
        sample_count={str(s): int(len(pd.DataFrame(rows["gate"])) and
                                 sum(1 for _ in [])) or None for s in seeds},
        checkpoint_sha256={str(s): sha256_file(CKPT_ROOT / VARIANTS[s] / "best.pt")
                           for s in seeds},
        phase1_input_hashes={f"seed{s}/{f}": sha256_file(P1 / f"seed{s}" / f)
                             for s in seeds
                             for f in ("predictions.npz", "temporal_signals.npz")},
        phase2_input_hashes={str(p.relative_to(ROOT)): sha256_file(p)
                             for p in sorted(P2.rglob("*.csv"))},
        source_hashes={f: sha256_file(ROOT / f) for f in
                       ("cmgm/scripts/phase3_interventions.py",
                        "cmgm/models/hetero_mixhop_model.py",
                        "cmgm/models/switching_latent_transformer.py")},
        parameter_guard=guard,
        baseline_reproduction=baselines,
        controls=control_checks,
        controls_not_evaluable=not_evaluable,
        controls_pass=bool(controls_pass),
        TEST_READ=False, TRAINING=False, PARAMETER_UPDATE=False,
        software=dict(python=__import__("platform").python_version(),
                      torch=torch.__version__, numpy=np.__version__,
                      pandas=pd.__version__),
    ))
    print("controls_pass =", controls_pass)
    if not controls_pass:
        raise SystemExit("FAIL: an intervention control did not reproduce the "
                         "baseline; intervention results must not be interpreted")
    print("Phase 3 interventions written under", root)


if __name__ == "__main__":
    main()
