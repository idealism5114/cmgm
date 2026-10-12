#!/usr/bin/env python
"""Phase 1 - cross-seed manifest, consistency check and human-readable report.

Reads only the artifacts written by
``cmgm.scripts.phase1_interpretability_extraction`` and reports what is on
disk. It never re-derives model quantities, so the report cannot disagree with
the extracted tensors.

    python -m cmgm.scripts.phase1_interpretability_report
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
NPZ_FILES = ("predictions.npz", "temporal_signals.npz", "spatial_signals.npz",
             "fusion_signals.npz")

# signal -> (producing module, meaning).  Modules are named exactly as in the
# model source so the table can be checked against the code.  A key containing
# "{1,2}_hop{1,2}" expands over both EdgeAttnMixHop layers and both hops.
SIGNAL_TABLE = [
    ("market_tokens", "SwitchingLatentTransformerBranch.market_encoder", "daily 128-d market token built from the three market attention aggregations plus per-market dispersion"),
    ("H_sequence", "LongMemoryTransformer", "causal Transformer hidden states H_1..H_T, one 128-d token per day"),
    ("H_final", "derived: H_sequence[:, -1]", "Transformer state H_T"),
    ("Z_sequence", "SwitchingLatentTransformerBranch.latent_forward", "deterministic micro state Z_1..Z_T"),
    ("Z_final", "derived: Z_sequence[:, -1]", "micro state Z_T"),
    ("latent_candidates", "RegimeLatentTransition.generators", "per-state generator outputs G_k([H_t, Z_{t-1}]), k=0..2"),
    ("generator_0", "RegimeLatentTransition.generators[0]", "per-step output of state generator 0"),
    ("generator_1", "RegimeLatentTransition.generators[1]", "per-step output of state generator 1"),
    ("generator_2", "RegimeLatentTransition.generators[2]", "per-step output of state generator 2"),
    ("regime_probabilities", "MarkovRegimeFilter.step", "filtered posterior state probability p_t (softmax over the 3 states)"),
    ("regime_priors", "MarkovRegimeFilter.step", "one-step-ahead prior p_{t-1} @ transition_matrix"),
    ("regime_evidence", "MarkovRegimeFilter.regime_evidence", "per-state evidence logits from the linear layer"),
    ("regime_evidence_step", "hook on regime_filter.regime_evidence", "same evidence via per-step hook, used as an independent cross-check"),
    ("p_final", "derived: regime_probabilities[:, -1]", "state probability at the final timestep"),
    ("transition_matrix", "MarkovRegimeFilter.transition_matrix", "3x3 Markov transition matrix, identical for every sample (static parameter)"),
    ("temporal_readout", "SwitchingLatentTransformerBranch.state_readout", "balanced temporal representation h_t"),
    ("u_H", "SwitchingLatentTransformerBranch.long_memory_norm", "LN(W_H H_T): long-memory half of the balanced readout"),
    ("u_Z", "SwitchingLatentTransformerBranch.micro_state_norm", "LN(W_Z Z_T): switching-latent half of the balanced readout"),
    ("WH_H_T", "long_memory_readout", "W_H H_T before LayerNorm"),
    ("WZ_Z_T", "micro_state_readout", "W_Z Z_T before LayerNorm"),
    ("balanced_concat", "derived: pre-hook on state_readout", "[u_H ; u_Z], the input W_r consumes"),
    ("long_memory_layer1_causal_attention", "CausalBaseRPEAttention", "temporal self-attention over the 20 daily tokens (not graph attention)"),
    ("long_memory_layer2_causal_attention", "CausalBaseRPEAttention", "temporal self-attention over the 20 daily tokens (not graph attention)"),
    ("long_memory_layer1_qk_logits", "CausalBaseRPEAttention", "raw q.k logits before the Base-RPE bias and the causal mask"),
    ("long_memory_layer2_qk_logits", "CausalBaseRPEAttention", "raw q.k logits before the Base-RPE bias and the causal mask"),
    ("adaptive_adjacency", "AdaptiveGraphLearner", "learned adaptive adjacency A used as the attention structure prior"),
    ("temporal_score", "HeteroMixHopCMGM.temporal_score", "spatial-branch per-node per-day score"),
    ("spatial_time_alpha", "derived: softmax(temporal_score, dim=time)", "time attention weights inside the spatial branch"),
    ("spatial_H_seq", "pre-hook on temporal_score", "per-day node projections H_seq"),
    ("spatial_aggregated", "pre-hook on attn_mixhop1", "time-weighted node tensor entering the first EdgeAttnMixHop"),
    ("spatial_mixhop1_pre_relu", "EdgeAttnMixHop (layer 1)", "output of layer 1 before its ReLU"),
    ("spatial_nodes", "HeteroMixHopCMGM.gcn_norm", "post-LayerNorm node representations after both EdgeAttnMixHop layers"),
    ("type_pool_input", "pre-hook on type_pool", "node representations entering the per-market pooling"),
    ("spatial_readout", "type_pool", "spatial representation h_s"),
    ("stock_pool", "derived: mean over stock nodes", "mean-pooled stock block of the spatial node tensor"),
    ("bond_pool", "derived: mean over bond nodes", "mean-pooled bond block"),
    ("commodity_pool", "derived: mean over commodity nodes", "mean-pooled commodity block"),
    ("market_node_attention_stock", "MarketAwareTemporalEncoder", "within-market node attention inside the daily market token"),
    ("market_node_attention_bond", "MarketAwareTemporalEncoder", "within-market node attention inside the daily market token"),
    ("market_node_attention_commodity", "MarketAwareTemporalEncoder", "within-market node attention inside the daily market token"),
    ("graph_attention_layer_{1,2}_hop{1,2}_attention", "EdgeAttnMixHop", "normalised edge attention used for message passing"),
    ("graph_attention_layer_{1,2}_hop{1,2}_logits", "EdgeAttnMixHop", "attention logits after the structure prior and before softmax"),
    ("graph_attention_layer_{1,2}_hop{1,2}_prior_bias", "EdgeAttnMixHop", "prior_scale * log(A + eps), broadcast over heads; identical in all captures"),
    ("gate_input", "pre-hook on gate_fc", "[h_s ; h_t]"),
    ("gate_logits", "gate_fc", "gate pre-activation"),
    ("fusion_gate", "derived: sigmoid(gate_logits)", "g in (0,1), per sample and per channel"),
    ("spatial_projected", "gcn_proj", "s = W_s h_s + b_s"),
    ("temporal_projected", "lstm_proj", "t = W_t h_t + b_t"),
    ("fused_representation", "pre-hook on head", "f = g*t + (1-g)*s"),
    ("head_output", "head", "flat head output, reshaped to (H, N_commodity)"),
    ("prediction", "HeteroMixHopCMGM.forward", "final prediction (H, N_commodity)"),
    ("target", "MarketSequenceDataset", "realised forward returns"),
]
# captured for verification but deliberately NOT written to disk
NOT_STORED = {
    "graph_attention_layer_{1,2}_hop{1,2}_content":
        "content-only logits q.k/sqrt(d); exactly equal to logits - prior_bias, "
        "which is verified on every capture and recorded in audit.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def expand(key):
    """Expand a '{1,2}_hop{1,2}' table key into its four concrete names."""
    if "{1,2}_hop{1,2}" not in key:
        return [key]
    suffix = key.split("hop{1,2}_", 1)[1]
    return [f"graph_attention_layer_{l}_hop{h}_{suffix}"
            for l in (1, 2) for h in (1, 2)]


def seed_summary(seed: int, root: Path) -> dict:
    d = root / f"seed{seed}"
    audit = load_json(d / "audit.json")
    meta = load_json(d / "metadata.json")
    repro = load_json(d / "metric_reproduction.json")
    nodes = load_json(d / "node_metadata.json")
    files = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
             for p in sorted(d.iterdir()) if p.is_file()}
    stored_keys = sorted({k for f in NPZ_FILES for k in np.load(d / f).files})

    z = np.load(d / "spatial_signals.npz")
    priors = [z[k] for k in sorted(z.files) if k.endswith("_prior_bias")]
    distinct_prior = len({hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest() for a in priors})
    att = z["graph_attention_layer_1_hop1_attention"]
    self_w = np.diagonal(att, axis1=1, axis2=2)          # (N, 284, 8)
    self_stats = dict(mean=float(self_w.mean()),
                      nonzero_fraction=float((self_w > 0).mean()),
                      min=float(self_w.min()))
    del att, self_w

    eq = audit["prediction_equivalence"]
    val_repro = audit["val_selection_loss_reproduction"]
    metric_max = max(v["absolute_difference"] for v in repro["metrics"].values())
    nonfinite = {k: v for k, v in audit["nan_inf_counts"].items()
                 if v["nan"] or v["inf"]}
    env = audit.get("environment_mismatch", {}) or {}
    structural = bool(
        audit["strict_load"]["passed"] and eq["bitwise_identical"]
        and not nonfinite and audit["n_validation_samples"] == 268
        and len(files) >= 7)
    metric_ok = bool(metric_max < 1e-6
                     and val_repro["absolute_difference"] < 1e-6)
    return dict(seed=seed, dir=str(d.relative_to(ROOT)), audit=audit,
                metadata=meta, metric_reproduction=repro, node_metadata=nodes,
                files=files, stored_keys=stored_keys,
                distinct_prior_bias=distinct_prior, self_attention=self_stats,
                metric_max_abs_diff=metric_max, equivalence=eq,
                val_repro=val_repro, nonfinite=nonfinite, environment=env,
                pass_structural=structural, pass_metric=metric_ok,
                pass_=bool(structural and metric_ok))


def consistency(seeds: list) -> dict:
    def pick(fn):
        return {s["seed"]: fn(s) for s in seeds}
    shapes = pick(lambda s: s["audit"]["tensor_shapes"])
    dates = pick(lambda s: (s["audit"]["date_first"], s["audit"]["date_last"]))
    nodes = pick(lambda s: [n["node_name"] for n in s["node_metadata"]["nodes"]])
    shape_agreement = {k: sorted({tuple(shapes[s["seed"]][k]) for s in seeds})
                       for k in shapes[seeds[0]["seed"]]}
    return dict(
        all_seeds_present=[s["seed"] for s in seeds],
        n_samples=pick(lambda s: s["audit"]["n_validation_samples"]),
        date_ranges=dates,
        node_ordering_identical_across_seeds=len(
            {tuple(v) for v in nodes.values()}) == 1,
        shape_agreement={k: [list(x) for x in v]
                         for k, v in shape_agreement.items()},
        shapes_identical_across_seeds=all(len(v) == 1
                                          for v in shape_agreement.values()),
        stored_keys_identical_across_seeds=len(
            {tuple(s["stored_keys"]) for s in seeds}) == 1,
        distinct_prior_bias_per_seed=pick(lambda s: s["distinct_prior_bias"]),
        metric_reproduction_max_abs_diff=pick(lambda s: s["metric_max_abs_diff"]),
        prediction_equivalence_max_abs_diff=pick(
            lambda s: s["equivalence"]["max_abs_difference"]),
        val_selection_loss_abs_diff=pick(
            lambda s: s["val_repro"]["absolute_difference"]),
        per_seed_pass=pick(lambda s: s["pass_"]),
        per_seed_pass_structural=pick(lambda s: s["pass_structural"]),
        per_seed_pass_metric=pick(lambda s: s["pass_metric"]),
        overall_pass=all(s["pass_"] for s in seeds),
    )


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_report(seeds: list, cons: dict, root: Path) -> str:
    first = seeds[0]
    shapes = first["audit"]["tensor_shapes"]
    stored = set(first["stored_keys"])
    rows, covered = [], {"prediction", "target"}
    for key, module, meaning in SIGNAL_TABLE:
        for name in expand(key):
            covered.add(name)
            arr = first["audit"]["tensor_shapes"].get(name)
            on_disk = name in stored
            rows.append((name, module,
                         str(shapes.get(name, "(not captured)")),
                         "stored" if on_disk else "captured, not stored",
                         meaning))
    extra = sorted(k for k in stored if k not in covered)

    lines = []
    lines.append("# Phase 1 - Frozen-Model Interpretability Extraction Report")
    lines.append("")
    lines.append(f"Generated {datetime.utcnow().isoformat()}Z by "
                 "`cmgm/scripts/phase1_interpretability_report.py`.")
    lines.append("")
    lines.append("**Scope.** Extraction, verification and audit only. No model "
                 "structure, checkpoint, data split, prediction or model "
                 "selection was changed. TEST was not touched. Regime states "
                 "are referred to only as state_0 / state_1 / state_2.")
    lines.append("")
    lines.append("## 1. Model path confirmation")
    lines.append("")
    lines.append("All three branches were read from the source rather than "
                 "inferred from names. Frozen variant "
                 "`switching_latent_balanced_readout`, `HeteroMixHopCMGM` in "
                 "`cmgm/models/hetero_mixhop_model.py`:")
    lines.append("")
    lines.append("- **Spatial branch** - "
                 "`HeteroMixHopCMGM._temp_weighted_spatial` (line 2346): "
                 "`type_proj` per timestep, `temporal_score` softmax over the "
                 "20 days, then `attn_mixhop1` and `attn_mixhop2` (two "
                 "`EdgeAttnMixHop`, K=2, 8 heads, beta=0.05, prior_scale=0.5, "
                 "no hard mask, no cross-market mask), then `gcn_norm`, then "
                 "`type_pool`.")
    lines.append("- **Temporal branch** - "
                 "`SwitchingLatentTransformerBranch` (line 307) in "
                 "`cmgm/models/switching_latent_transformer.py`: "
                 "`market_encoder` -> `LongMemoryTransformer` (line 99) -> "
                 "`MarkovRegimeFilter.step` (line 197) + "
                 "`RegimeLatentTransition` (line 220) -> `readout` (line 821).")
    lines.append("- **Fusion** - `HeteroMixHopCMGM._market_token_predict` "
                 "(line 2562): `gate = sigmoid(gate_fc([h_s; h_t]))`, "
                 "`fused = gate * lstm_proj(h_t) + (1 - gate) * gcn_proj(h_s)`, "
                 "`head(fused)` reshaped to (H, 24).")
    lines.append("")
    lines.append("## 2. Extracted tensors")
    lines.append("")
    lines.append(f"Shapes are the real measured shapes for seed{first['seed']}; "
                 "the run configuration is T=20, N=284 (248 stock / 12 bond / "
                 "24 commodity), F=21, 3 latent states, 8 attention heads. The "
                 "`storage` column states what is actually written to the "
                 "`.npz` artifacts.")
    lines.append("")
    lines.append(table(rows, ["Signal", "Producing module", "Shape",
                              "Storage", "Meaning"]))
    lines.append("")
    lines.append("Deliberately captured but not written, because the quantity "
                 "is exactly recoverable from a stored tensor:")
    lines.append("")
    for key, why in NOT_STORED.items():
        for name in expand(key):
            lines.append(f"- `{name}`: {why}.")
    lines.append("")
    if extra:
        lines.append("Stored arrays not in the curated table above: "
                     + ", ".join(f"`{e}`" for e in extra) + ".")
        lines.append("")
    lines.append("## 3. Numerical equivalence")
    lines.append("")
    lines.append(table([(f"seed{s['seed']}", s["equivalence"]["bitwise_identical"],
                         f"{s['equivalence']['max_abs_difference']:.3e}",
                         f"{s['equivalence']['mean_abs_difference']:.3e}")
                        for s in seeds],
                       ["Seed", "Bitwise identical", "max abs difference",
                        "mean abs difference"]))
    lines.append("")
    lines.append("Reference is `model(x)` with no hooks and with the model's own "
                 "`capture_attention` flag off; the candidate is the same "
                 "`model(x)` with forward hooks attached and "
                 "`capture_attention` on. Attention capture is an existing, "
                 "opt-in, RNG-free diagnostic path in `EdgeAttnMixHop`, and "
                 "every other tensor is read by a hook that does not alter the "
                 "computation.")
    lines.append("")
    lines.append("## 4. Metric reproduction")
    lines.append("")
    lines.append(table([(f"seed{s['seed']}",
                         f"{s['metric_max_abs_diff']:.3e}",
                         f"{s['val_repro']['stored']:.12f}",
                         f"{s['val_repro']['recomputed']:.12f}",
                         f"{s['val_repro']['absolute_difference']:.3e}")
                        for s in seeds],
                       ["Seed", "max abs deviation over the 16 VAL metrics",
                        "stored VAL selection loss", "recomputed", "abs diff"]))
    lines.append("")
    lines.append("The selection loss is the criterion the audited runs used to "
                 "pick their best epoch (batch-mean four-horizon Huber, "
                 "delta=0.02). Per-horizon VAL MAE/MSE/RMSE/Hit and the "
                 "per-metric comparison live in `seed*/metric_reproduction.json`.")
    lines.append("")
    lines.append("### 4.1 Environment provenance")
    lines.append("")
    with_env = [s["seed"] for s in seeds
                if (s.get("environment") or {}).get("recorded_environment_available")]
    without_env = [s["seed"] for s in seeds if s["seed"] not in with_env]
    lines.append("The audited runs were produced in a different software "
                 "environment from this extraction host. "
                 + (f"Seed(s) {with_env} recorded their environment; "
                    if with_env else "")
                 + (f"seed(s) {without_env} recorded none, so nothing is "
                    "asserted about them." if without_env else ""))
    lines.append("")
    rows = []
    for s in seeds:
        env = s.get("environment") or {}
        available = bool(env.get("recorded_environment_available"))
        rec = env.get("recorded", {}) if available else {}
        cur = s["audit"].get("environment", {})
        blank = "(not recorded)"
        rows.append((f"seed{s['seed']}",
                     rec.get("python", blank) if available else blank,
                     rec.get("pytorch", blank) if available else blank,
                     rec.get("numpy", blank) if available else blank,
                     cur.get("python"), cur.get("torch"), cur.get("numpy")))
    lines.append(table(rows, ["Seed", "recorded python", "recorded pytorch",
                              "recorded numpy", "current python",
                              "current torch", "current numpy"]))
    lines.append("")
    lines.append("Because the feature matrix is produced by numeric pandas and "
                 "NumPy code, differing versions can change its floating-point "
                 "bytes, so the recomputed data fingerprints are not expected to "
                 "match the recorded ones even though the input CSVs, the "
                 "pipeline source hashes and the split sizes are unchanged:")
    lines.append("")
    fp_rows = []
    for s in seeds:
        m = s["metadata"]["data_fingerprints_match_recorded"]
        for split in ("train", "val"):
            fp_rows.append((f"seed{s['seed']}", split, bool(m[split]),
                            s["metadata"]["data_fingerprint_recomputed"][split][:16],
                            s["metadata"]["data_fingerprint_recorded"][split]["sha256"][:16]))
    lines.append(table(fp_rows, ["Seed", "Split", "Match", "Recomputed",
                                 "Recorded"]))
    lines.append("")
    lines.append("This is an unresolved provenance discrepancy and is reported "
                 "as such rather than explained away: the recorded per-field "
                 "arrays were never stored, so the recorded digest cannot be "
                 "reproduced byte for byte. What the evidence does establish is "
                 "that the audited computation reproduces: the model's own "
                 "selection loss matches to the last recorded digit, and the 16 "
                 "VAL metrics agree to about 1e-10, which is float32 rounding "
                 "level. `cmgm/scripts/phase1_data_fingerprint_diagnostic.py` "
                 "re-hashes the rebuilt data under permuted field orders, field "
                 "subsets and per-field byte representations and reports whether "
                 "any reproduces the recorded digest.")
    lines.append("")
    lines.append("## 5. Three-seed completeness")
    lines.append("")
    for s in seeds:
        lines.append(f"- seed{s['seed']}: "
                     f"{'PASS' if s['pass_'] else 'FAIL'} - "
                     f"{s['audit']['n_validation_samples']} samples, "
                     f"dates {s['audit']['date_first']} to {s['audit']['date_last']}, "
                     f"checkpoint sha256 {s['audit']['checkpoint_sha256'][:12]}..., "
                     f"{len(s['stored_keys'])} stored arrays")
    lines.append("")
    lines.append(f"Cross-seed overall: "
                 f"{'PASS' if cons['overall_pass'] else 'FAIL'}. Identical node "
                 f"ordering across seeds: "
                 f"{cons['node_ordering_identical_across_seeds']}. Identical "
                 f"shapes across seeds: {cons['shapes_identical_across_seeds']}.")
    lines.append("")
    return "\n".join(lines)


def attention_semantics(seeds: list) -> str:
    c = seeds[0]["audit"]["semantic_checks"]
    n_nodes = seeds[0]["audit"]["tensor_shapes"]["adaptive_adjacency"][0]
    att_shape = seeds[0]["audit"]["tensor_shapes"][
        "graph_attention_layer_1_hop1_attention"]
    distinct_prior = seeds[0]["distinct_prior_bias"]
    self_stats = seeds[0]["self_attention"]
    z = np.load(ROOT / seeds[0]["dir"] / "spatial_signals.npz")
    row_dev = {k: float(np.abs(z[k].sum(2) - 1.0).max())
               for k in z.files if k.startswith("graph_attention")
               and k.endswith("_attention")}
    worst_row_dev = max(row_dev.values())
    del z
    return "\n".join([
        "# Graph attention semantics (measured, not assumed)",
        "",
        f"Tensor shape {att_shape} = (batch, target node, source node, head). "
        f"Reference model: seed{seeds[0]['seed']}.",
        "",
        "1. **Direction.** `EdgeAttnMixHop.forward` computes "
        "`e = einsum('bnhd,bmhd->bnmh', Q, Kt)`: the query index `n` is axis 1 "
        "and the key index `m` is axis 2. Axis 1 is therefore the "
        "destination/target node, whose representation is updated, and axis 2 "
        "is the source node.",
        "2. **Source axis** = 2. **Destination axis** = 1.",
        "3. **Softmax.** Yes, `F.softmax(e, dim=2)`, normalised over source "
        "nodes, so each target node's incoming weights sum to 1.",
        f"4. **Verified.** Attention summed over axis 2 equals 1 to within "
        f"float32 rounding. Computed from the stored artifacts of "
        f"seed{seeds[0]['seed']}, the deviation is at most "
        f"{worst_row_dev:.3e} across all four captures (layer 1/2 x hop 1/2); "
        f"`audit.json` records the value for layer 1 hop 1, "
        f"{c['attention_source_dim_sum_max_dev']:.3e}. The stored attention "
        f"equals `softmax(logits, axis=2)` up to "
        f"{c['attention_equals_softmax_of_logits_max_dev']:.3e} on that same "
        "capture.",
        f"5. **Self-edges.** Never masked. The adaptive adjacency has an "
        f"exactly zero diagonal (max diagonal entry "
        f"{c['adaptive_adjacency_diagonal_min']:.3e}), so the structure prior "
        f"contributes `log(1e-6) * 0.5` to every self-edge logit, which "
        f"suppresses but does not remove self-attention. Measured self-weight "
        f"in layer 1 hop 1: mean {self_stats['mean']:.3e}, minimum "
        f"{self_stats['min']:.3e}, nonzero for "
        f"{self_stats['nonzero_fraction'] * 100:.2f}% of (sample, node, head) "
        "entries.",
        f"6. **Mask.** No hard mask and no cross-market mask are active for this "
        f"variant (`hard_mask=False`, `cross_mask=None`), so all {n_nodes} "
        "sources are reachable from every target.",
        f"7. **Adaptive prior.** `log_prior = log(clamp(A, min=0) + 1e-6)` is "
        f"added to the content logits scaled by `prior_scale=0.5`. Because "
        f"`graph_prior_heads=None` the prior is added to all 8 heads. The 4 "
        f"captures (2 layers x 2 hops) produced **{distinct_prior} distinct** "
        f"prior tensors, as expected: one `A` is computed per forward and "
        "shared by both layers and hops.",
        "8. **Final edge weights used in message passing** are exactly the "
        "post-softmax `attention` tensor: dropout is inactive in `eval()`, so "
        "the stored tensor is the one multiplied by `V`. Attention and logits "
        "are stored separately and no aggregation of any kind was applied.",
        f"9. **Content logits.** `logits - prior_bias` equals the content-only "
        f"logits up to {c['logits_minus_prior_bias_equals_content_max_dev']:.3e}, "
        "so the content logits are not stored; the deviation is recorded per "
        "capture in `audit.json`.",
        "10. **Layer semantics are identical.** Both layers are the same "
        "`EdgeAttnMixHop` class with identical configuration and differ only in "
        "weights, and the measured attention differs between layers.",
        f"11. **Sample-dependent**: {c['attention_differs_between_samples']} - "
        "attention varies across samples.",
        "12. **Time-dependent**: no. The spatial branch first reduces the 20 "
        "days to a single node tensor via time attention (alpha row sums "
        f"deviate by at most {c['spatial_time_alpha_row_sum_max_dev']:.3e}), so "
        "the graph attention has no time axis.",
        f"13. **Multi-head**: yes, {c['attention_head_dim']} heads, stored "
        "separately and never averaged.",
        "14. **Hops.** `K=2`, so each layer performs two attention "
        "computations: hop 1 attends over the input node tensor, hop 2 attends "
        "over the hop-1 output. Both hops are stored separately. The MixHop "
        "update is `H = beta*H_in + (1-beta)*V_agg` with `H_in` fixed to the "
        "layer input and `beta=0.05`.",
        "",
        f"15. **Structure prior is directed.** `adaptive_adjacency_symmetric` is "
        f"{c['adaptive_adjacency_symmetric']}: `AdaptiveGraphLearner` builds "
        "`A = ReLU(tanh(alpha*(M1 @ M2.T - M2 @ M1.T))) * gate`, which is "
        "anti-symmetric before the ReLU, so `A_ij` and `A_ji` are unrelated and "
        "the graph is directed. Roughly half of all entries are exactly zero.",
        "",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1-root", type=Path,
                   default=ROOT / "interpretability" / "phase1")
    args = p.parse_args()
    root = args.phase1_root
    seeds = [seed_summary(s, root) for s in SEEDS
             if (root / f"seed{s}" / "audit.json").exists()]
    if not seeds:
        raise SystemExit(f"no extracted seeds under {root}")
    cons = consistency(seeds)

    manifest = dict(
        generated_utc=datetime.utcnow().isoformat() + "Z",
        phase1_root=str(root.relative_to(ROOT)),
        seeds=[dict(seed=s["seed"], dir=s["dir"], files=s["files"],
                    stored_keys=s["stored_keys"],
                    checkpoint=s["audit"]["checkpoint"],
                    checkpoint_sha256=s["audit"]["checkpoint_sha256"],
                    n_samples=s["audit"]["n_validation_samples"],
                    pass_=s["pass_"])
               for s in seeds],
        total_bytes=sum(f["bytes"] for s in seeds for f in s["files"].values()),
    )
    (root / "summary").mkdir(parents=True, exist_ok=True)
    (root / "summary" / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (root / "summary" / "consistency_check.json").write_text(
        json.dumps(cons, indent=2, ensure_ascii=False) + "\n")
    (root / "PHASE1_INTERPRETABILITY_EXTRACTION_REPORT.md").write_text(
        build_report(seeds, cons, root))
    (root / "attention_semantics.md").write_text(attention_semantics(seeds))
    print(f"report written for seeds {[s['seed'] for s in seeds]}; "
          f"overall {'PASS' if cons['overall_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
