#!/usr/bin/env python
"""Phase 1 - frozen-model interpretability signal extraction for D0B.

Read-only extraction of internal tensors from the already trained and already
audited ``switching_latent_balanced_readout`` checkpoint.

Phase 1 scope is extract + verify + audit only:
  * no training, no optimizer step, no backward, no parameter update
  * model source, checkpoint file, data split and checkpoint selection are
    unchanged; the recorded best checkpoint is loaded with strict=True
  * TEST is never used for model selection, interpretation tuning or reporting
  * the original ``model(x)`` call path and its outputs are untouched: every
    internal tensor is captured with pure forward hooks plus the model's own
    pre-existing, opt-in, RNG-free ``capture_attention`` diagnostic flag.

Usage
-----
    python -m cmgm.scripts.phase1_interpretability_extraction --seed 42 --preflight
    python -m cmgm.scripts.phase1_interpretability_extraction --seed 42 --device cuda
    python -m cmgm.scripts.phase1_interpretability_extraction --all-seeds --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import (
    align_markets, load_bond_prices, load_commodity_prices,
    load_stock_prices, set_seed,
)
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.main_ablation import build_data
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import make_loss, validate_epoch

ROOT = Path(__file__).resolve().parents[2]
VARIANT = "switching_latent_balanced_readout"
DISPLAY = "D0B-BalancedLatentReadout"
HORIZONS = tuple(int(h) for h in config.MULTI_HORIZONS)
N_HEADS = 8
MARKETS = ("stock", "bond", "commodity")

CANONICAL_RUNS = {
    42: "switching_latent_balanced_readout_seed42_20261006_231320_782593",
    43: "switching_latent_balanced_readout_seed43_20261007_123714_3172316",
    44: "switching_latent_balanced_readout_seed44_20261007_125331_3172316",
}
EXPERIMENT_ROOT = ROOT / "experiments" / "d0b_node_history_readout"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "d0b_node_history_readout"

RECORDED_SOURCE_FILES = (
    "cmgm/config.py", "cmgm/models/hetero_mixhop_model.py",
    "cmgm/models/switching_latent_transformer.py",
    "cmgm/models/node_history_readout.py", "cmgm/training/train.py",
    "cmgm/training/metric_standard.py", "cmgm/data/data_loader.py",
    "cmgm/data/feature_builder.py", "cmgm/scripts/main_ablation.py",
    "cmgm/scripts/d0b_node_history_readout.py",
)
PHASE1_ADDITIONAL_SOURCE_FILES = (
    "cmgm/models/switching_transformer.py", "cmgm/graph/adaptive_graph.py",
    "cmgm/models/model.py", "cmgm/scripts/phase1_interpretability_extraction.py",
)
PHASE1_SOURCE_FILES = RECORDED_SOURCE_FILES + PHASE1_ADDITIONAL_SOURCE_FILES

TEMPORAL_KEYS = (
    "market_tokens", "H_sequence", "H_final", "Z_sequence", "Z_final",
    "latent_states", "latent_candidates", "regime_probabilities",
    "regime_priors", "regime_evidence", "regime_evidence_step",
    "p_final", "transition_matrix",
    "temporal_readout", "u_H", "u_Z", "WH_H_T", "WZ_Z_T", "balanced_concat",
    "generator_0", "generator_1", "generator_2",
    "long_memory_layer1_causal_attention", "long_memory_layer2_causal_attention",
    "long_memory_layer1_qk_logits", "long_memory_layer2_qk_logits",
)
SPATIAL_KEYS = (
    "adaptive_adjacency", "temporal_score", "spatial_time_alpha",
    "spatial_H_seq", "spatial_aggregated", "spatial_mixhop1_pre_relu",
    "spatial_nodes", "type_pool_input", "spatial_readout",
    "stock_pool", "bond_pool", "commodity_pool",
    "market_node_attention_stock", "market_node_attention_bond",
    "market_node_attention_commodity",
    "graph_attention_layer_1_hop1_attention", "graph_attention_layer_1_hop2_attention",
    "graph_attention_layer_2_hop1_attention", "graph_attention_layer_2_hop2_attention",
    "graph_attention_layer_1_hop1_logits", "graph_attention_layer_1_hop2_logits",
    "graph_attention_layer_2_hop1_logits", "graph_attention_layer_2_hop2_logits",
    "graph_attention_layer_1_hop1_prior_bias", "graph_attention_layer_1_hop2_prior_bias",
    "graph_attention_layer_2_hop1_prior_bias", "graph_attention_layer_2_hop2_prior_bias",
    "graph_attention_layer_1_hop1_content", "graph_attention_layer_1_hop2_content",
    "graph_attention_layer_2_hop1_content", "graph_attention_layer_2_hop2_content",
)
# Tensors produced once per forward and shared by every sample.
STATIC_TENSORS = {"adaptive_adjacency"}
STATIC_ATTENTION_FIELDS = {"prior_bias"}

FUSION_KEYS = (
    "gate_input", "gate_logits", "fusion_gate", "spatial_readout_pre_proj",
    "spatial_projected", "temporal_readout_pre_proj", "temporal_projected",
    "fused_representation", "head_output",
)


# ---------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().tolist()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(f"not JSON serializable: {type(v).__name__}")

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False, default=default) + "\n",
                    encoding="utf-8")


def as_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu", torch.float32).contiguous().numpy()


def split_bounds(total: int):
    train_end = int(total * config.TRAIN_RATIO)
    return train_end, train_end + int(total * config.VAL_RATIO)


def model_kwargs(data: dict) -> dict:
    mi = data["market_indices"]
    start, end = mi["commodity"]
    return dict(num_nodes=int(data["n_nodes"]),
                n_commodities=int(data["n_commodities"]),
                n_stock=int(mi["stock"][1] - mi["stock"][0]),
                n_bond=int(mi["bond"][1] - mi["bond"][0]),
                commodity_indices=(int(start), int(end)))


def deterministic_loader(dataset, batch_size: int) -> DataLoader:
    """Identical ordering to the audited evaluation: no shuffle, no drop_last."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      drop_last=False, num_workers=0)


def data_fingerprint(dataset) -> str:
    h = hashlib.sha256()
    for field in ("feature_matrix", "prices", "raw_prices"):
        value = getattr(dataset, field, None)
        if value is not None:
            h.update(field.encode())
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def softmax_np(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def environment_mismatch(recorded: dict, current: dict) -> dict:
    """Compare the recorded run environment with this extraction environment."""
    if not recorded:
        return dict(recorded_environment_available=False,
                    differing_components={}, identical=None,
                    note="original artifacts did not record software versions; "
                         "nothing is asserted about the original environment")
    pairs = {"python": ("python", "python"), "pytorch": ("pytorch", "torch"),
             "numpy": ("numpy", "numpy"), "pandas": ("pandas", "pandas")}
    diffs = {}
    for name, (rk, ck) in pairs.items():
        rv, cv = recorded.get(rk), current.get(ck)
        if rv is not None and cv is not None and str(rv) != str(cv):
            diffs[name] = dict(recorded=str(rv), current=str(cv))
    return dict(recorded_environment_available=True,
                recorded=recorded, differing_components=diffs,
                identical=not diffs,
                note="the data pipeline is numeric and version-sensitive; "
                     "differing numpy/pandas/python versions change the "
                     "floating-point bytes of the feature matrix, so the data "
                     "fingerprint is not expected to match across environments")


def environment_info(device: torch.device) -> dict:
    import pandas, scipy
    env = dict(python=platform.python_version(), torch=torch.__version__,
               numpy=np.__version__, pandas=pandas.__version__,
               scipy=scipy.__version__, platform=platform.platform(),
               cuda_available=torch.cuda.is_available(),
               device=str(device),
               cudnn=torch.backends.cudnn.version())
    if device.type == "cuda":
        env["gpu_name"] = torch.cuda.get_device_name(device)
        env["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        env["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    return env


# ---------------------------------------------------------- hook recorder
class Capture:
    """Pure forward-hook recorder. Mutates nothing and draws no RNG."""

    def __init__(self):
        self.handles = []
        self.set_ = {}
        self.seq = {}

    def reset_sequences(self):
        self.seq = {}

    def _store(self, key, tensor, kind):
        arr = as_np(tensor)
        if kind == "set":
            self.set_[key] = arr
        else:
            self.seq.setdefault(key, []).append(arr)

    def on_output(self, module, key, kind="set"):
        def fn(mod, inp, out):
            self._store(key, out, kind)
        self.handles.append(module.register_forward_hook(fn))

    def on_input(self, module, key, kind="set"):
        def fn(mod, inp):
            self._store(key, inp[0], kind)
        self.handles.append(module.register_forward_pre_hook(fn))

    def stacked(self, key):
        return np.stack(self.seq[key], axis=0)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def register_hooks(model, cap: Capture) -> None:
    branch = model.switching_latent_transformer
    enc = branch.market_encoder

    # --- temporal branch ---
    cap.on_output(enc, "market_tokens")                          # (B,T,128)
    cap.on_output(branch.long_memory, "H_sequence")              # (B,T,128)
    cap.on_output(branch.long_memory_readout, "WH_H_T")          # (B,64)
    cap.on_output(branch.long_memory_norm, "u_H")                # (B,64)
    cap.on_output(branch.micro_state_readout, "WZ_Z_T")          # (B,64)
    cap.on_output(branch.micro_state_norm, "u_Z")                # (B,64)
    cap.on_input(branch.state_readout, "balanced_concat")        # (B,128)
    cap.on_output(branch.state_readout, "temporal_readout")      # h_t (B,64)
    cap.on_output(branch.regime_filter.regime_evidence,
                  "regime_evidence_step", kind="seq")            # (B,3) x T
    for k, gen in enumerate(branch.latent_transition.generators):
        cap.on_output(gen, f"generator_{k}", kind="seq")         # (B,64) x T

    # --- spatial branch ---
    cap.on_output(model.graph_learner, "adaptive_adjacency")     # (N,N)
    cap.on_input(model.temporal_score, "spatial_H_seq")          # (B,T,N,64)
    cap.on_output(model.temporal_score, "temporal_score")        # (B,T,N,1)
    cap.on_input(model.attn_mixhop1, "spatial_aggregated")       # (B,N,64)
    cap.on_output(model.attn_mixhop1, "spatial_mixhop1_pre_relu")  # (B,N,64)
    cap.on_output(model.gcn_norm, "spatial_nodes")               # h  (B,N,64)
    cap.on_input(model.type_pool, "type_pool_input")             # (B,N,64)
    cap.on_output(model.type_pool, "spatial_readout")            # h_s (B,64)

    # --- fusion ---
    cap.on_input(model.gcn_proj, "spatial_readout_pre_proj")
    cap.on_output(model.gcn_proj, "spatial_projected")           # s (B,64)
    cap.on_input(model.lstm_proj, "temporal_readout_pre_proj")
    cap.on_output(model.lstm_proj, "temporal_projected")         # t (B,64)
    cap.on_input(model.gate_fc, "gate_input")                    # [h_s; h_t]
    cap.on_output(model.gate_fc, "gate_logits")                  # pre-sigmoid
    cap.on_input(model.head, "fused_representation")             # f (B,64)
    cap.on_output(model.head, "head_output")                     # (B,96)


# ------------------------------------------------------------- extraction
def reference_predictions(model, dataset, device, batch_size, limit):
    preds = []
    with torch.no_grad():
        for i, (x, _) in enumerate(deterministic_loader(dataset, batch_size)):
            if limit is not None and i >= limit:
                break
            preds.append(as_np(model(x.to(device))))
    return np.concatenate(preds, axis=0)


def extraction_pass(model, dataset, device, cap, batch_size, limit,
                    capture_attention, n_stock, n_bond):
    collected = {}
    predictions, targets = [], []
    diag = dict(content_vs_logits_minus_prior_max_dev=0.0,
                prior_bias_variants=set(), prior_bias_captures=0,
                transition_matrix=as_np(
                    model.switching_latent_transformer.regime_filter
                    .transition_matrix()))

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(deterministic_loader(dataset, batch_size)):
            if limit is not None and b_idx >= limit:
                break
            x = x.to(device)
            cap.reset_sequences()
            pred = model(x)
            branch = model.switching_latent_transformer

            predictions.append(as_np(pred))
            targets.append(as_np(y))
            collected.setdefault("prediction", []).append(as_np(pred))

            for key, arr in cap.set_.items():
                if key in STATIC_TENSORS and b_idx > 0:
                    continue
                collected.setdefault(key, []).append(arr)
            for key in list(cap.seq):
                # hooks fire per timestep: (T, B, ...) -> (B, T, ...)
                collected.setdefault(key, []).append(
                    np.swapaxes(cap.stacked(key), 0, 1))

            # regime/latent tensors are produced inside a python loop, so they
            # come from the branch's own caches (set unconditionally in
            # latent_forward); the per-step hooks above are cross-checks.
            collected.setdefault("regime_probabilities", []).append(
                as_np(branch.last_regime_probabilities))
            collected.setdefault("regime_priors", []).append(
                as_np(branch.last_regime_priors))
            collected.setdefault("regime_evidence", []).append(
                as_np(branch.last_regime_evidence))
            collected.setdefault("latent_states", []).append(
                as_np(branch.last_latent_states))
            collected.setdefault("latent_candidates", []).append(
                as_np(branch.last_latent_candidates))

            for i, blk in enumerate(branch.long_memory.layers):
                collected.setdefault(
                    f"long_memory_layer{i + 1}_causal_attention", []).append(
                    as_np(blk.attention.last_attention))
                collected.setdefault(
                    f"long_memory_layer{i + 1}_qk_logits", []).append(
                    as_np(blk.attention.last_qk_logits))

            if capture_attention:
                for layer_name, mod in (("layer_1", model.attn_mixhop1),
                                        ("layer_2", model.attn_mixhop2)):
                    for entry in mod.last_attention_diagnostics:
                        hop = entry["hop"]
                        content = as_np(entry["content"])
                        logits = as_np(entry["logits"])
                        prior = as_np(entry["prior_bias"])
                        diag["content_vs_logits_minus_prior_max_dev"] = max(
                            diag["content_vs_logits_minus_prior_max_dev"],
                            float(np.abs(content - (logits - prior[None])).max()))
                        key = hashlib.sha256(prior.tobytes()).hexdigest()[:16]
                        diag["prior_bias_variants"].add(key)
                        diag["prior_bias_captures"] += 1
                        for name, value in (("attention", entry["attention"]),
                                            ("logits", entry["logits"]),
                                            ("prior_bias", entry["prior_bias"]),
                                            ("content", entry["content"])):
                            if name in STATIC_ATTENTION_FIELDS and b_idx > 0:
                                continue
                            collected.setdefault(
                                f"graph_attention_{layer_name}_hop{hop}_{name}",
                                []).append(as_np(value))

            for name, value in branch.market_encoder.last_market_attentions.items():
                collected.setdefault(f"market_node_attention_{name}",
                                     []).append(as_np(value))

    out = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)

    # ---- derived tensors (same arithmetic the model performs; verified) ----
    out["p_final"] = out["regime_probabilities"][:, -1, :]
    out["H_final"] = out["H_sequence"][:, -1, :]
    out["Z_sequence"] = out["latent_states"]
    out["Z_final"] = out["latent_states"][:, -1, :]
    out["fusion_gate"] = 1.0 / (1.0 + np.exp(-out["gate_logits"]))
    out["spatial_time_alpha"] = softmax_np(out["temporal_score"], axis=1)[..., 0]
    tp = out["type_pool_input"]
    out["stock_pool"] = tp[:, :n_stock, :].mean(axis=1)
    out["bond_pool"] = tp[:, n_stock:n_stock + n_bond, :].mean(axis=1)
    out["commodity_pool"] = tp[:, n_stock + n_bond:, :].mean(axis=1)
    out["transition_matrix"] = np.repeat(
        diag["transition_matrix"][None, :, :], prediction.shape[0], axis=0)
    diag["prior_bias_variants"] = sorted(diag["prior_bias_variants"])
    return out, prediction, target, diag


def _layernorm_np(x, ln: torch.nn.LayerNorm):
    w = ln.weight.detach().cpu().numpy()
    b = ln.bias.detach().cpu().numpy()
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + ln.eps) * w + b


def semantic_checks(model, signals, prediction, target, device, diag):
    p = signals["regime_probabilities"]
    att = signals["graph_attention_layer_1_hop1_attention"]
    logits = signals["graph_attention_layer_1_hop1_logits"]
    prior = signals["graph_attention_layer_1_hop1_prior_bias"]
    A = signals["adaptive_adjacency"]
    branch = model.switching_latent_transformer
    h = checks = {}

    h["regime_probability_row_sum_max_dev"] = float(np.abs(p.sum(-1) - 1).max())
    h["regime_probability_min"] = float(p.min())
    h["regime_probability_max"] = float(p.max())
    h["regime_posterior_equals_softmax_log_prior_plus_evidence_max_dev"] = float(
        np.abs(p - softmax_np(np.log(signals["regime_priors"] + 1e-8)
                              + signals["regime_evidence"], axis=-1)).max())
    h["regime_evidence_hook_matches_cache_max_dev"] = float(
        np.abs(signals["regime_evidence_step"]
               - signals["regime_evidence"]).max())
    h["transition_matrix_rows_sum_to_one_max_dev"] = float(
        np.abs(signals["transition_matrix"].sum(-1) - 1).max())
    h["transition_matrix_sticky_alpha"] = float(branch.regime_filter.sticky_alpha)
    h["transition_matrix_softmax_temp_tau"] = float(branch.regime_filter.tau)

    h["attention_source_dim_sum_max_dev"] = float(np.abs(att.sum(2) - 1).max())
    h["attention_equals_softmax_of_logits_max_dev"] = float(
        np.abs(att - softmax_np(logits, axis=2)).max())
    h["logits_minus_prior_bias_equals_content_max_dev"] = float(
        diag["content_vs_logits_minus_prior_max_dev"])
    h["prior_bias_capture_count"] = diag["prior_bias_captures"]
    h["prior_bias_distinct_tensor_count"] = len(diag["prior_bias_variants"])
    h["attention_self_edge_min"] = float(np.diagonal(att, axis1=1, axis2=2).min())
    h["attention_has_no_all_zero_row"] = bool((att.sum(2) > 0).all())
    h["adaptive_adjacency_symmetric"] = bool(np.allclose(A, A.T, atol=1e-6))
    h["adaptive_adjacency_diagonal_min"] = float(np.diag(A).min())
    h["adaptive_adjacency_zero_fraction"] = float((A == 0).mean())
    h["attention_differs_between_samples"] = bool(
        att.std(axis=0).max() > 0)
    h["attention_differs_between_layers"] = bool(
        np.abs(signals["graph_attention_layer_1_hop1_attention"]
               - signals["graph_attention_layer_2_hop1_attention"]).max() > 0)
    h["attention_differs_between_hops"] = bool(
        np.abs(signals["graph_attention_layer_1_hop1_attention"]
               - signals["graph_attention_layer_1_hop2_attention"]).max() > 0)
    h["attention_head_dim"] = int(att.shape[-1])
    h["spatial_time_alpha_row_sum_max_dev"] = float(
        np.abs(signals["spatial_time_alpha"].sum(1) - 1).max())

    z_recon = np.einsum("btk,btkd->btd", p, signals["latent_candidates"])
    h["Z_equals_probability_weighted_candidates_max_dev"] = float(
        np.abs(z_recon - signals["Z_sequence"]).max())
    gen = np.stack([signals[f"generator_{k}"] for k in range(3)], axis=2)
    h["latent_candidates_equals_generator_outputs_max_dev"] = float(
        np.abs(gen - signals["latent_candidates"]).max())

    with torch.no_grad():
        rec = branch.state_readout(
            torch.from_numpy(signals["balanced_concat"]).to(device)).cpu().numpy()
    h["temporal_readout_equals_state_readout_of_concat_max_dev"] = float(
        np.abs(rec - signals["temporal_readout"]).max())
    h["u_H_equals_LN_of_WH_H_final_max_dev"] = float(
        np.abs(signals["u_H"] - _layernorm_np(signals["WH_H_T"],
                                              branch.long_memory_norm)).max())
    h["u_Z_equals_LN_of_WZ_Z_final_max_dev"] = float(
        np.abs(signals["u_Z"] - _layernorm_np(signals["WZ_Z_T"],
                                              branch.micro_state_norm)).max())
    h["H_final_equals_H_sequence_last_max_dev"] = float(
        np.abs(signals["H_final"] - signals["H_sequence"][:, -1]).max())
    h["Z_final_equals_Z_sequence_last_max_dev"] = float(
        np.abs(signals["Z_final"] - signals["Z_sequence"][:, -1]).max())

    with torch.no_grad():
        concat = torch.from_numpy(np.concatenate(
            [signals["stock_pool"], signals["bond_pool"],
             signals["commodity_pool"]], axis=-1)).to(device)
        pooled = model.type_pool.type_agg(concat).cpu().numpy()
    h["spatial_readout_equals_type_agg_of_pools_max_dev"] = float(
        np.abs(pooled - signals["spatial_readout"]).max())
    h["type_pool_input_equals_gcn_norm_output_max_dev"] = float(
        np.abs(signals["type_pool_input"] - signals["spatial_nodes"]).max())
    h["spatial_aggregated_equals_time_weighted_H_seq_max_dev"] = float(
        np.abs((signals["spatial_H_seq"]
                * signals["spatial_time_alpha"][..., None]).sum(1)
               - signals["spatial_aggregated"]).max())

    g = signals["fusion_gate"]
    h["fused_equals_gate_temporal_plus_one_minus_gate_spatial_max_dev"] = float(
        np.abs(g * signals["temporal_projected"]
               + (1 - g) * signals["spatial_projected"]
               - signals["fused_representation"]).max())
    h["gate_equals_sigmoid_of_logits_max_dev"] = float(
        np.abs(1.0 / (1.0 + np.exp(-signals["gate_logits"]))
               - signals["fusion_gate"]).max())
    h["gate_input_equals_concat_spatial_temporal_max_dev"] = float(
        np.abs(np.concatenate([signals["spatial_readout"],
                               signals["temporal_readout"]], axis=-1)
               - signals["gate_input"]).max())
    h["spatial_proj_input_equals_spatial_readout_max_dev"] = float(
        np.abs(signals["spatial_readout_pre_proj"]
               - signals["spatial_readout"]).max())
    h["temporal_proj_input_equals_temporal_readout_max_dev"] = float(
        np.abs(signals["temporal_readout_pre_proj"]
               - signals["temporal_readout"]).max())

    with torch.no_grad():
        flat = model.head(torch.from_numpy(
            signals["fused_representation"]).to(device)).cpu().numpy()
    h["head_output_equals_head_of_fused_max_dev"] = float(
        np.abs(flat - signals["head_output"]).max())
    h["prediction_equals_reshape_of_head_output_max_dev"] = float(
        np.abs(flat.reshape(prediction.shape) - prediction).max())
    h["prediction_shape"] = list(prediction.shape)
    h["target_shape"] = list(target.shape)
    return h


# ------------------------------------------------------------- artifacts
def save_artifacts(result, output_root: Path, store_content: bool,
                   runtime: dict) -> None:
    seed = result["seed"]
    out_dir = Path(output_root) / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = result["signals"]

    np.savez(out_dir / "predictions.npz",
             prediction=result["prediction"], target=result["target"],
             sample_index=result["sample_index"],
             date_last_observed=result["date_last"],
             **{f"date_target_{h}": result["date_target"][h] for h in HORIZONS})

    np.savez(out_dir / "temporal_signals.npz",
             **{k: sig[k] for k in TEMPORAL_KEYS if k in sig})
    spatial_keys = [k for k in SPATIAL_KEYS if k in sig]
    if not store_content:
        spatial_keys = [k for k in spatial_keys if not k.endswith("_content")]
    np.savez(out_dir / "spatial_signals.npz", **{k: sig[k] for k in spatial_keys})
    np.savez(out_dir / "fusion_signals.npz",
             **{k: sig[k] for k in FUSION_KEYS if k in sig})

    mi = result["market_indices"]
    names = result["feature_names"]
    nodes = []
    for idx in range(len(names)):
        if idx < mi["stock"][1]:
            market = "stock"
        elif idx < mi["bond"][1]:
            market = "bond"
        else:
            market = "commodity"
        nodes.append(dict(node_index=int(idx), node_name=str(names[idx]),
                          market=market))
    c0, c1 = mi["commodity"]
    write_json(out_dir / "node_metadata.json", dict(
        node_count=len(names),
        market_indices={k: list(map(int, v)) for k, v in mi.items()},
        market_sizes={k: int(mi[k][1] - mi[k][0]) for k in MARKETS},
        commodity_node_interval=[int(c0), int(c1)],
        node_order="stock (0:n_stock) then bond then commodity, as produced by align_markets",
        graph_attention_node_axis="node index into this list; source axis = -2, target axis = -3",
        nodes=nodes,
        commodity_mapping=[dict(commodity=str(names[c0 + i]), node_index=int(c0 + i),
                                target_index=int(i), output_index=int(i))
                           for i in range(c1 - c0)],
        input_feature_names=[str(x) for x in result["feature_names"]],
    ))

    write_json(out_dir / "metric_reproduction.json", dict(
        seed=seed, split=result["split"],
        metrics=result["reproduction"],
        val_selection_loss=dict(
            stored_metric=result["stored_val_loss"],
            recomputed_metric=result["val_loss"],
            absolute_difference=abs(result["stored_val_loss"] - result["val_loss"]),
            definition="batch-mean of the four-horizon Huber (delta=0.02) prediction "
                       "loss, mean over validation batches; the D0B selection criterion"),
    ))

    write_json(out_dir / "metadata.json", dict(
        seed=seed, variant=VARIANT, display_name=DISPLAY, split=result["split"],
        checkpoint=str(result["ckpt_path"]),
        checkpoint_sha256=sha256_file(result["ckpt_path"]),
        experiment_result=str(result["exp_dir"] / "result.json"),
        stored_best_epoch=int(result["payload"]["best_epoch"]),
        stored_best_val_loss=float(result["payload"]["best_val_loss"]),
        source_hashes_recorded=result["stored_source_hashes"],
        source_hashes_phase1={f: sha256_file(ROOT / f) for f in PHASE1_SOURCE_FILES},
        source_hash_mismatches=[f for f in PHASE1_SOURCE_FILES
                                if f in result["stored_source_hashes"]
                                and sha256_file(ROOT / f) != result["stored_source_hashes"][f]],
        data_fingerprint_recorded=result["stored_data_fingerprint"],
        data_fingerprint_recomputed=result["data_fingerprints"],
        data_fingerprints_match_recorded={
            s: result["data_fingerprints"].get(s)
               == result["stored_data_fingerprint"][s]["sha256"]
            for s in result["data_fingerprints"]},
        recorded_environment=result["recorded_environment"],
        recorded_environment_path=result["recorded_environment_path"],
        environment_mismatch=environment_mismatch(
            result["recorded_environment"], runtime["environment"]),
        n_samples=int(result["prediction"].shape[0]),
        split_row_bounds=[int(result["start_row"]), int(result["stop_row"])],
        sample_ordering="ascending global row of the last observed day; "
                        "X row i covers global rows [start+i, start+i+19]; "
                        "sample order preserved from the audited loader (shuffle=False)",
        date_first=str(result["date_last"][0]),
        date_last=str(result["date_last"][-1]),
        horizons=list(HORIZONS),
        extraction_timestamp_utc=datetime.utcnow().isoformat() + "Z",
        runtime=runtime,
    ))

    write_json(out_dir / "audit.json", dict(
        seed=seed, split=result["split"],
        checkpoint=str(result["ckpt_path"]),
        checkpoint_sha256=sha256_file(result["ckpt_path"]),
        strict_load=dict(passed=True,
                         missing=list(result["load_info"].missing_keys),
                         unexpected=list(result["load_info"].unexpected_keys)),
        phase1_source_hashes={f: sha256_file(ROOT / f) for f in PHASE1_SOURCE_FILES},
        recorded_source_hashes=result["stored_source_hashes"],
        data_fingerprint_recorded=result["stored_data_fingerprint"],
        data_fingerprint_recomputed=result["data_fingerprints"],
        data_fingerprints_match_recorded={
            s: result["data_fingerprints"].get(s)
               == result["stored_data_fingerprint"][s]["sha256"]
            for s in result["data_fingerprints"]},
        recorded_environment=result["recorded_environment"],
        recorded_environment_path=result["recorded_environment_path"],
        environment_mismatch=environment_mismatch(
            result["recorded_environment"], runtime["environment"]),
        n_validation_samples=int(result["prediction"].shape[0]),
        sample_index_first_last=[int(result["sample_index"][0]),
                                 int(result["sample_index"][-1])],
        date_ordering_monotonic=bool(np.all(result["date_last"][1:]
                                            >= result["date_last"][:-1])),
        date_first=str(result["date_last"][0]),
        date_last=str(result["date_last"][-1]),
        tensor_shapes=result["shapes"],
        nan_inf_counts=result["nan_inf"],
        prediction_equivalence=result["equivalence"],
        metric_reproduction=result["reproduction"],
        val_selection_loss_reproduction=dict(
            stored=result["stored_val_loss"], recomputed=result["val_loss"],
            absolute_difference=abs(result["stored_val_loss"] - result["val_loss"])),
        semantic_checks=result["checks"],
        extraction_timestamp_utc=datetime.utcnow().isoformat() + "Z",
        environment=runtime["environment"],
        gpu_contention_note="GPU utilisation is shared with unrelated jobs on "
                            "this host; the extraction is deterministic and does "
                            "not depend on scheduling",
    ))


# ------------------------------------------------------------------ main
def run(seed: int, split: str, output_root: Path, device_name: str,
        limit=None, preflight=False, capture_attention=True,
        checkpoint=None, experiment=None, store_content=False) -> dict:
    if split not in ("val", "train"):
        raise ValueError("Phase 1 extracts 'val' (default) or 'train'; TEST is out of scope")
    run_name = CANONICAL_RUNS[seed]
    ckpt_path = Path(checkpoint) if checkpoint else CHECKPOINT_ROOT / run_name / "best.pt"
    exp_dir = Path(experiment) if experiment else EXPERIMENT_ROOT / run_name
    stored_result = json.loads((exp_dir / "result.json").read_text())
    stored_metrics = json.loads((exp_dir / "metrics.json").read_text())
    env_path = exp_dir / "environment.json"
    recorded_environment = (json.loads(env_path.read_text())
                            if env_path.exists() else None)

    all_prices, market_indices = align_markets(
        load_stock_prices(config.STOCK_FILE),
        load_bond_prices(config.BOND_FILE),
        load_commodity_prices(config.COMMODITY_FILE))
    dates = all_prices.index
    total_rows = len(dates)
    train_end, val_end = split_bounds(total_rows)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    runtime = dict(environment=environment_info(device))

    data = build_data(SimpleNamespace(batch_size=config.BATCH_SIZE,
                                      seq_len=config.SEQ_LEN, seed=seed))
    kwargs = model_kwargs(data)
    dataset = data["loaders"][split].dataset
    data_fingerprints = {name: data_fingerprint(data["loaders"][name].dataset)
                         for name in ("train", "val")}

    set_seed(seed)
    model = HeteroMixHopCMGM(**kwargs, variant=VARIANT)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_info = model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()

    start_row = 0 if split == "train" else train_end
    stop_row = train_end if split == "train" else val_end
    n_expected = (stop_row - start_row) - config.SEQ_LEN - max(HORIZONS) + 1
    if len(dataset) != n_expected:
        raise ValueError(f"split origin count {len(dataset)} != expected {n_expected}")

    # 1. reference forward: diagnostics off and no hooks attached
    model.attn_mixhop1.capture_attention = False
    model.attn_mixhop2.capture_attention = False
    reference = reference_predictions(model, dataset, device, config.BATCH_SIZE, limit)

    # 2. extraction forward: hooks attached + the model's own attention capture
    cap = Capture()
    register_hooks(model, cap)
    model.attn_mixhop1.capture_attention = bool(capture_attention)
    model.attn_mixhop2.capture_attention = bool(capture_attention)
    signals, prediction, target, diag = extraction_pass(
        model, dataset, device, cap, config.BATCH_SIZE, limit,
        capture_attention,
        int(market_indices["stock"][1]),
        int(market_indices["bond"][1] - market_indices["bond"][0]))
    model.attn_mixhop1.capture_attention = False
    model.attn_mixhop2.capture_attention = False
    cap.close()

    diff = np.abs(reference - prediction)
    equivalence = dict(
        bitwise_identical=bool(np.array_equal(reference, prediction)),
        max_abs_difference=float(diff.max()),
        mean_abs_difference=float(diff.mean()),
        allclose_rtol1e5_atol1e8=bool(np.allclose(reference, prediction,
                                                  rtol=1e-5, atol=1e-8)),
        reference="model(x), no hooks, capture_attention=False",
        candidate="model(x), hooks attached, capture_attention=True",
    )

    checks = semantic_checks(model, signals, prediction, target, device, diag)

    criterion = make_loss()
    val_loader = deterministic_loader(data["loaders"]["val"].dataset, config.BATCH_SIZE)
    val_loss = float(validate_epoch(model, val_loader,
                                    torch.empty(2, 0, dtype=torch.long),
                                    torch.empty(0), criterion, device))

    metrics = {str(h): population_metrics(prediction[:, i], target[:, i])
               for i, h in enumerate(HORIZONS)}
    reproduction = {}
    for h in HORIZONS:
        for name in ("MAE", "MSE", "RMSE", "Hit"):
            s = float(stored_metrics[split][str(h)][name])
            r = float(metrics[str(h)][name])
            reproduction[f"{split}_{h}d_{name}"] = dict(
                stored_metric=s, recomputed_metric=r, absolute_difference=abs(s - r))

    nan_inf = {k: dict(nan=int(np.isnan(v).sum()), inf=int(np.isinf(v).sum()))
               for k, v in signals.items() if v.dtype.kind == "f"}
    shapes = {k: list(v.shape) for k, v in signals.items()}
    shapes["prediction"] = list(prediction.shape)
    shapes["target"] = list(target.shape)

    n = prediction.shape[0]
    sample_index = np.arange(n, dtype=np.int64)
    row_of_sample = start_row + sample_index + config.SEQ_LEN - 1
    date_last = np.array([dates[r].strftime("%Y-%m-%d") for r in row_of_sample])
    date_target = {h: np.array([dates[r + h].strftime("%Y-%m-%d")
                                for r in row_of_sample]) for h in HORIZONS}

    result = dict(
        seed=seed, split=split, n_samples=int(n), signals=signals,
        prediction=prediction, target=target, equivalence=equivalence,
        checks=checks, metrics=metrics, reproduction=reproduction,
        val_loss=val_loss,
        stored_val_loss=float(stored_result["best_val_prediction_loss"]),
        shapes=shapes, nan_inf=nan_inf, sample_index=sample_index,
        date_last=date_last, date_target=date_target, payload=payload,
        ckpt_path=ckpt_path, exp_dir=exp_dir, dataset=dataset, model=model,
        market_indices=market_indices, dates=dates, feature_names=data["feature_names"],
        start_row=start_row, stop_row=stop_row,
        recorded_environment=recorded_environment,
        recorded_environment_path=(str(env_path) if env_path.exists() else None),
        data_fingerprints=data_fingerprints,
        stored_source_hashes=stored_result["model"]["source_hashes"],
        stored_data_fingerprint=stored_result["model"]["data_fingerprint"]["splits"],
        load_info=load_info)

    if preflight:
        print(f"=== PREFLIGHT seed={seed} split={split} (no artifacts written) ===")
        for k in sorted(shapes):
            print(f"  {k}: {shapes[k]}")
        print(" equivalence:", equivalence)
        print(" metric reproduction max abs diff:",
              max(v["absolute_difference"] for v in reproduction.values()))
        print(" val selection loss:", val_loss, "stored:", result["stored_val_loss"])
        print(" data fingerprint recomputed:",
              data_fingerprint(dataset), "recorded:",
              result["stored_data_fingerprint"][split]["sha256"])
        for k in ("regime_probability_row_sum_max_dev",
                  "attention_source_dim_sum_max_dev",
                  "attention_equals_softmax_of_logits_max_dev",
                  "Z_equals_probability_weighted_candidates_max_dev",
                  "fused_equals_gate_temporal_plus_one_minus_gate_spatial_max_dev",
                  "prediction_equals_reshape_of_head_output_max_dev"):
            print(f"  {k} = {checks[k]}")
        return result

    save_artifacts(result, output_root, store_content, runtime)
    print(f"seed{seed}: {n} samples, equivalence max|d|={equivalence['max_abs_difference']:.3e}, "
          f"val-loss reproduction |d|={abs(result['stored_val_loss'] - val_loss):.3e}, "
          f"artifacts -> {Path(output_root) / f'seed{seed}'}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, choices=sorted(CANONICAL_RUNS), default=42)
    p.add_argument("--all-seeds", action="store_true",
                   help="run every frozen seed (42, 43, 44) sequentially")
    p.add_argument("--split", default="val", choices=("val", "train"))
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "interpretability" / "phase1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--experiment", type=Path, default=None)
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--no-attention-capture", action="store_true")
    p.add_argument("--store-content-logits", action="store_true",
                   help="also persist pre-prior content logits (redundant with "
                        "logits - prior_bias, which is verified exact)")
    args = p.parse_args()

    if args.all_seeds and (args.checkpoint or args.experiment):
        raise ValueError("--all-seeds cannot be combined with a single-run override")
    seeds = sorted(CANONICAL_RUNS) if args.all_seeds else [args.seed]
    for seed in seeds:
        run(seed, args.split, args.output_root, args.device,
            limit=args.limit_batches, preflight=args.preflight,
            capture_attention=not args.no_attention_capture,
            checkpoint=args.checkpoint, experiment=args.experiment,
            store_content=args.store_content_logits)


if __name__ == "__main__":
    main()
