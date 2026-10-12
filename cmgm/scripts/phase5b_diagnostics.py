#!/usr/bin/env python
"""Phase 5B - representation diagnostics for the fusion controls.

For every fusion model at seed 42 (Original, S1, GCF, C1, C2, C3), extract the
pre-fusion tensor [s; t] and the post-fusion tensor entering the head, then run
the same compression diagnostics as Phase 5A. This connects the architecture to
the representation change it causes.

Frozen extraction only: eval mode, no_grad, strict checkpoint load, no updates.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.phase4_variants import S1_FIXED_MEAN_FUSION, Phase4Simplification
from cmgm.models.phase5_variants import PHASE5_VARIANTS, Phase5FusionControl
from cmgm.scripts.main_ablation import build_data

ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / "interpretability" / "phase5"
OUT = P5 / "frozen_diagnostics"
SEED = 42
REF = ROOT / "experiments" / "d0b_node_history_readout"

MODELS = ("Original", "S1-FixedMeanFusion", "GCF", "ConcatLinear64",
          "GatedConcat128", "MeanConcat128")


def _ckpt(pattern: str) -> Path:
    hits = sorted(ROOT.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    best = hits[-1] / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    return best


def build(name: str, kw: dict):
    if name == "Original":
        m = HeteroMixHopCMGM(**kw, variant="switching_latent_balanced_readout")
        return m, _ckpt("checkpoints/d0b_node_history_readout/switching_latent_balanced_readout_seed42_*")
    if name == "S1-FixedMeanFusion":
        m = Phase4Simplification(S1_FIXED_MEAN_FUSION, **kw)
        return m, _ckpt("checkpoints/phase4_ablation/switching_latent_fixed_mean_fusion_seed42_*")
    if name == "GCF":
        m = HeteroMixHopCMGM(**kw, variant="switching_latent_balanced_concat_fusion")
        return m, _ckpt("checkpoints/d0b_node_history_readout/switching_latent_balanced_concat_fusion_seed42_*")
    if name in PHASE5_VARIANTS:
        m = Phase5FusionControl(name, **kw)
        return m, _ckpt(f"checkpoints/phase5b_controls/{name}_seed42_*")
    raise ValueError(name)


_DATA = {}


def _data():
    """build_data ignores its seed, so build the pipeline once for all models."""
    if "d" not in _DATA:
        _DATA["d"] = build_data(SimpleNamespace(batch_size=config.BATCH_SIZE,
                                                seq_len=config.SEQ_LEN, seed=SEED))
    return _DATA["d"]


@torch.no_grad()
def extract(name: str, device) -> dict:
    data = _data()
    mi = {k: tuple(v) for k, v in data["market_indices"].items()}
    kw = dict(num_nodes=int(data["n_nodes"]),
              n_commodities=int(data["n_commodities"]),
              n_stock=int(mi["stock"][1] - mi["stock"][0]),
              n_bond=int(mi["bond"][1] - mi["bond"][0]),
              commodity_indices=(int(mi["commodity"][0]), int(mi["commodity"][1])))
    model, ck = build(name, kw)
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()
    buf = {}
    handles = [
        model.gcn_proj.register_forward_hook(lambda m, i, o: buf.__setitem__("s", o.detach())),
        model.lstm_proj.register_forward_hook(lambda m, i, o: buf.__setitem__("t", o.detach())),
        model.head.register_forward_pre_hook(lambda m, i: buf.__setitem__("fused", i[0].detach())),
    ]
    pre, post, preds = [], [], []
    loader = DataLoader(data["loaders"]["val"].dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, drop_last=False, num_workers=0)
    try:
        for x, y in loader:
            buf.clear()
            p = model(x.to(device))
            pre.append(torch.cat([buf["s"], buf["t"]], dim=-1).cpu().numpy().astype(np.float32))
            post.append(buf["fused"].cpu().numpy().astype(np.float32))
            preds.append(p.cpu().numpy().astype(np.float32))
    finally:
        for h in handles:
            h.remove()
    return dict(pre=np.concatenate(pre), post=np.concatenate(post),
                pred=np.concatenate(preds), checkpoint=str(ck.relative_to(ROOT)),
                parameter_count=int(sum(p.numel() for p in model.parameters())))


def main() -> None:
    from cmgm.scripts.phase5a_diagnostics import (dynamics_stats, effective_dimension,
                                                  geometry_preservation, knn_retention)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, knn_rows, meta = [], [], {}
    cache = {}
    for name in MODELS:
        ext = extract(name, device)
        cache[name] = ext
        pre, post = ext["pre"], ext["post"]
        for tag, X in (("pre_fusion", pre), ("post_fusion", post)):
            r = dict(model=name, stage=tag, seed=SEED)
            r.update(effective_dimension(X)); r.update(dynamics_stats(X))
            rows.append(r)
        g = geometry_preservation(pre, post)
        for k in (5, 10, 20):
            knn_rows.append(dict(model=name, k=k, retention=knn_retention(pre, post, k)))
        meta[name] = dict(pre_width=int(pre.shape[1]), post_width=int(post.shape[1]),
                          checkpoint=ext["checkpoint"], parameter_count=ext["parameter_count"],
                          geometry_spearman=g["spearman"])
        print(f"{name:18s} pre {pre.shape} post {post.shape} "
              f"geom_spearman {g['spearman']:.4f} params {ext['parameter_count']}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "fusion_control_representation_metrics.csv", index=False)
    pd.DataFrame(knn_rows).to_csv(OUT / "fusion_control_knn.csv", index=False)
    (P5 / "fusion_controls").mkdir(parents=True, exist_ok=True)
    (P5 / "fusion_controls" / "representation_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote fusion_control_representation_metrics.csv")


if __name__ == "__main__":
    main()
