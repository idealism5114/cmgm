#!/usr/bin/env python
"""Phase 6 - frozen fusion-stage representation extraction and diagnostics.

For Original, GCF, ResidualFusion and BranchPreservingFusion at seed 42, extract
the fusion inputs, any internal paths the architecture defines, and the fusion
output, then compute the same compression diagnostics as Phase 5.

Frozen extraction only: eval mode, no_grad, strict checkpoint load, no update.
Geometry preservation is reported as auxiliary context; Phase 5 showed it does
not rank models by performance, so it is never used to order them here.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.phase6_variants import (M2_RESIDUAL, M3_BRANCH_PRESERVING,
                                         Phase6Fusion)
from cmgm.scripts.main_ablation import build_data
from cmgm.scripts.phase5a_diagnostics import (dynamics_stats, effective_dimension,
                                              geometry_preservation, knn_retention,
                                              linear_cka)

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "interpretability" / "phase6"
REPR = P6 / "representations"
SEED = 42
MODELS = ("Original", "GlobalConcatFusion", "ResidualFusion", "BranchPreservingFusion")


def _ckpt(pattern: str) -> Path:
    hits = sorted(ROOT.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[-1] / "best.pt"


def build(name: str, kw: dict):
    if name == "Original":
        return (HeteroMixHopCMGM(**kw, variant="switching_latent_balanced_readout"),
                _ckpt("checkpoints/d0b_node_history_readout/switching_latent_balanced_readout_seed42_*"))
    if name == "GlobalConcatFusion":
        return (HeteroMixHopCMGM(**kw, variant="switching_latent_balanced_concat_fusion"),
                _ckpt("checkpoints/d0b_node_history_readout/switching_latent_balanced_concat_fusion_seed42_*"))
    if name == "ResidualFusion":
        return (Phase6Fusion(M2_RESIDUAL, **kw), _ckpt("checkpoints/phase6_fusion/residual_cross_branch_fusion_seed42_*"))
    if name == "BranchPreservingFusion":
        return (Phase6Fusion(M3_BRANCH_PRESERVING, **kw), _ckpt("checkpoints/phase6_fusion/branch_preserving_fusion_seed42_*"))
    raise ValueError(name)


_DATA = {}


def _data():
    if "d" not in _DATA:
        _DATA["d"] = build_data(SimpleNamespace(batch_size=config.BATCH_SIZE,
                                                seq_len=config.SEQ_LEN, seed=SEED))
    return _DATA["d"]


@torch.no_grad()
def extract(name: str, device) -> dict:
    data = _data()
    mi = {k: tuple(v) for k, v in data["market_indices"].items()}
    kw = dict(num_nodes=int(data["n_nodes"]), n_commodities=int(data["n_commodities"]),
              n_stock=int(mi["stock"][1] - mi["stock"][0]),
              n_bond=int(mi["bond"][1] - mi["bond"][0]),
              commodity_indices=(int(mi["commodity"][0]), int(mi["commodity"][1])))
    model, ck = build(name, kw)
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()

    buf = {}
    handles = [
        model.gcn_proj.register_forward_hook(lambda m, i, o: buf.__setitem__("x_s", o.detach())),
        model.lstm_proj.register_forward_hook(lambda m, i, o: buf.__setitem__("x_t", o.detach())),
        model.head.register_forward_pre_hook(lambda m, i: buf.__setitem__("fused", i[0].detach())),
    ]
    if name == "ResidualFusion":
        handles += [
            model.fusion.main.register_forward_hook(lambda m, i, o: buf.__setitem__("m2_pre", o.detach())),
            model.fusion.residual.register_forward_hook(lambda m, i, o: buf.__setitem__("m2_res", o.detach())),
        ]
    if name == "BranchPreservingFusion":
        handles += [
            model.fusion.spatial_proj.register_forward_hook(lambda m, i, o: buf.__setitem__("m3_s", o.detach())),
            model.fusion.temporal_proj.register_forward_hook(lambda m, i, o: buf.__setitem__("m3_t", o.detach())),
            model.fusion.interaction.register_forward_hook(lambda m, i, o: buf.__setitem__("m3_pre", o.detach())),
            model.fusion.out.register_forward_pre_hook(lambda m, i: buf.__setitem__("m3_cat", i[0].detach())),
        ]
    keys = ["x_s", "x_t", "fused"]
    if name == "ResidualFusion":
        keys += ["m2_pre", "m2_res"]
    if name == "BranchPreservingFusion":
        keys += ["m3_s", "m3_t", "m3_pre", "m3_cat"]
    acc = {k: [] for k in keys}
    preds, tgts = [], []
    loader = DataLoader(data["loaders"]["val"].dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, drop_last=False, num_workers=0)
    try:
        for x, y in loader:
            buf.clear()
            p = model(x.to(device))
            for k in keys:
                acc[k].append(buf[k].cpu().numpy().astype(np.float32))
            preds.append(p.cpu().numpy().astype(np.float32))
            tgts.append(y.numpy().astype(np.float32))
    finally:
        for h in handles:
            h.remove()
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["prediction"] = np.concatenate(preds)
    out["target"] = np.concatenate(tgts)
    out["checkpoint"] = str(ck.relative_to(ROOT))
    out["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPR.mkdir(parents=True, exist_ok=True)
    stage_rows, knn_rows, cka_rows, path_rows, meta = [], [], [], [], {}
    for name in MODELS:
        e = extract(name, device)
        pre = np.concatenate([e["x_s"], e["x_t"]], axis=1)
        post = e["fused"]
        np.savez_compressed(REPR / f"{name}_seed42.npz",
                            fusion_input_spatial=e["x_s"], fusion_input_temporal=e["x_t"],
                            fusion_output=post, prediction=e["prediction"],
                            target=e["target"],
                            **{k: e[k] for k in e if k.startswith("m2_") or k.startswith("m3_")})
        for tag, X in (("fusion_input", pre), ("fusion_output", post)):
            r = dict(model=name, stage=tag, seed=SEED)
            r.update(effective_dimension(X)); r.update(dynamics_stats(X))
            stage_rows.append(r)
        g = geometry_preservation(pre, post)
        knn_rows.append(dict(model=name, k=10,
                             retention=knn_retention(pre, post, 10)))
        cka_rows.append(dict(model=name, cka_input_vs_output=linear_cka(pre, post),
                             geometry_spearman=g["spearman"]))
        if name == "ResidualFusion":
            n = F.gelu(torch.from_numpy(e["m2_pre"])).numpy()
            r = e["m2_res"]
            nr = np.linalg.norm(n, axis=1); rr = np.linalg.norm(r, axis=1)
            path_rows.append(dict(model=name, quantity="norm_ratio_nonlinear_over_residual",
                                  value=float(np.mean(nr / np.clip(rr, 1e-8, None))),
                                  note="descriptive scale ratio only"))
            path_rows.append(dict(model=name, quantity="cka_nonlinear_vs_residual",
                                  value=float(linear_cka(n, r)), note=""))
            cos = (n * r).sum(1) / np.clip(nr * rr, 1e-8, None)
            path_rows.append(dict(model=name, quantity="cosine_nonlinear_vs_residual",
                                  value=float(cos.mean()), note=""))
        if name == "BranchPreservingFusion":
            s_b = e["m3_s"]; t_b = e["m3_t"]
            c_b = F.gelu(torch.from_numpy(e["m3_pre"])).numpy()
            for tag, X in (("spatial_specific", s_b), ("temporal_specific", t_b),
                           ("interaction", c_b)):
                v = np.linalg.norm(X, axis=1)
                path_rows.append(dict(model=name, quantity=f"l2_{tag}_mean",
                                      value=float(v.mean()), note="descriptive scale only"))
                path_rows.append(dict(model=name, quantity=f"l2_{tag}_median",
                                      value=float(np.median(v)), note="descriptive scale only"))
                path_rows.append(dict(model=name, quantity=f"l2_{tag}_std",
                                      value=float(v.std()), note="descriptive scale only"))
        meta[name] = dict(checkpoint=e["checkpoint"], parameter_count=e["parameter_count"],
                          geometry_spearman=g["spearman"], n_samples=int(post.shape[0]))
        print(f"{name:24s} pre {pre.shape} post {post.shape} "
              f"geom {g['spearman']:.4f} kNN10 {knn_rows[-1]['retention']:.4f} "
              f"CKA {cka_rows[-1]['cka_input_vs_output']:.4f}")
    pd.DataFrame(stage_rows).to_csv(REPR / "fusion_stage_diagnostics.csv", index=False)
    pd.DataFrame(knn_rows).to_csv(REPR / "fusion_knn_retention.csv", index=False)
    pd.DataFrame(cka_rows).to_csv(REPR / "fusion_cka.csv", index=False)
    pd.DataFrame(path_rows).to_csv(REPR / "fusion_path_descriptives.csv", index=False)
    (REPR / "extraction_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("representation diagnostics written")


if __name__ == "__main__":
    main()
