"""Checkpoint-only D0B routing diagnostics; never imported by normal forward.

Run from the repository root with ``python -m cmgm.scripts.d0b_regime_routing_diagnostics``.
No optimizer, scheduler, training entrypoint, or checkpoint writer is used.
All q interventions preserve the recipient's soft posterior recursion.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cmgm.config import FEATURE_DIM, HUBER_DELTA, MULTI_HORIZONS, SEQ_LEN
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_previous_state_diagnostics import _load_checkpoint

VARIANT = "switching_latent_balanced_readout"
TEMPERATURES = (0.50, 0.75, 1.00, 1.25, 2.00)
ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)
TIME_POINTS = (0, 2, 5, 10, 15, 19)
ROOT = Path(__file__).resolve().parents[2]


def assert_d0b(branch):
    if (not branch.balanced_readout or branch.use_latent_memory
            or branch.use_dynamic_slope or branch.use_balanced_transition_input
            or branch.use_regime_relative_memory):
        raise ValueError("Only unmodified D0B BalancedLatentReadout is allowed")


def routing_distribution(p, temperature=1.0, mode="native", previous=None,
                         permutation=None, sample_starts=None):
    """q is a generator weight only. Causal shuffle rejects later donor dates.

    For the chronological sliding windows, donor_start <= recipient_start is
    necessary for same-relative-timestep routing to be calendar-causal. An
    unavailable donor falls back to the recipient's p. This is explicitly a
    masked shuffle control, not an unrestricted batch permutation.
    """
    if mode == "native":
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return p if temperature == 1.0 else F.softmax(p.clamp_min(1e-8).log() / temperature, -1)
    if mode == "hard":
        return F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
    if mode == "uniform":
        return torch.full_like(p, 1.0 / p.shape[-1])
    if mode == "lag":
        return torch.full_like(p, 1.0 / p.shape[-1]) if previous is None else previous
    if mode in ("shuffle", "shuffle_unrestricted"):
        if permutation is None:
            raise ValueError("shuffle requires an explicit deterministic permutation")
        shuffled = p.index_select(0, permutation)
        if mode == "shuffle_unrestricted":
            return shuffled
        if sample_starts is None:
            raise ValueError("causal shuffle requires window start timestamps/indices")
        allowed = sample_starts[permutation] <= sample_starts
        return torch.where(allowed[:, None], shuffled, p)
    if mode.startswith("state"):
        state = int(mode[5:])
        if state not in range(p.shape[-1]):
            raise ValueError("invalid forced state")
        return F.one_hot(torch.full_like(p[..., 0], state, dtype=torch.long), p.shape[-1]).to(p.dtype)
    raise ValueError(f"unknown routing mode: {mode}")


def latent_forward_with_routing_intervention(
    branch, long_memory, routing_temperature=1.0, mode="native",
    sticky_alpha_override=None, permutation=None, sample_starts=None,
    posterior_reference=None,
):
    """Small diagnostic replay using the real filter and generator modules.

    Defaults reproduce D0B exactly. For routing-only experiments a previously
    recorded native posterior may be reused: it depends on H and p_prev, never
    Z or q. Alpha overrides always run the actual filter again. No module
    attribute or trained parameter is replaced by an intervention.
    """
    assert_d0b(branch)
    B, T, _ = long_memory.shape
    if sticky_alpha_override is None:
        A = branch.regime_filter.transition_matrix()
    else:
        if not 0 <= sticky_alpha_override <= 1:
            raise ValueError("alpha outside [0,1]")
        learned = branch.regime_filter.transition_logits.softmax(-1)
        A = sticky_alpha_override * torch.eye(branch.K, device=learned.device, dtype=learned.dtype) + (1 - sticky_alpha_override) * learned
        if posterior_reference is not None:
            raise ValueError("alpha intervention must recompute the posterior")
    p_prev = long_memory.new_full((B, branch.K), 1 / branch.K)
    z_prev = long_memory.new_zeros(B, branch.z_dim)
    trace = {key: [] for key in ("p", "q", "prior", "evidence", "Z", "candidates")}
    for t in range(T):
        if posterior_reference is None:
            prior, evidence, p, _ = branch.regime_filter.step(long_memory[:, t], p_prev, A)
        else:
            prior, evidence, p = (posterior_reference[key][:, t] for key in ("prior", "evidence", "p"))
        q = routing_distribution(p, routing_temperature, mode,
                                 previous=p_prev if t > 0 else None,
                                 permutation=permutation, sample_starts=sample_starts)
        # D0B passes H_t and its OWN intervened Z_(t-1) to the existing G_k.
        z, candidates = branch.latent_transition(long_memory[:, t], z_prev, q)
        for key, value in zip(trace, (p, q, prior, evidence, z, candidates)):
            trace[key].append(value)
        p_prev = p  # CRITICAL: q must never enter Markov recursion.
        z_prev = z
    result = {key: torch.stack(values, dim=1) for key, values in trace.items()}
    result["A"] = A
    result["H"] = long_memory
    return result


def complete_readout(model, spatial, trace, zero_component=None):
    branch = model.switching_latent_transformer
    if getattr(branch, "horizon_specific_state_readout", False):
        temporal = branch.readout_by_horizon(trace["H"][:, -1], trace["Z"][:, -1], zero_component=zero_component)
        return dict(trace, h_long=branch.last_h_long.clone(), h_micro=branch.last_h_micro.clone(),
                    h_long_effective=branch.last_h_long_effective.clone(),
                    h_micro_effective=branch.last_h_micro_effective.clone(),
                    h_temporal=temporal, prediction=model._market_token_predict_by_horizon(spatial, temporal))
    temporal = branch.readout(trace["H"][:, -1], trace["Z"][:, -1], zero_component=zero_component)
    return dict(trace, h_long=branch.last_h_long.clone(), h_micro=branch.last_h_micro.clone(),
                h_long_effective=branch.last_h_long_effective.clone(),
                h_micro_effective=branch.last_h_micro_effective.clone(),
                h_temporal=temporal, prediction=model._market_token_predict(spatial, temporal))


def normal_reference(model, x):
    """Record the actual, unchanged trained model forward before any helper."""
    prediction = model(x)
    branch = model.switching_latent_transformer
    trace = {key: getattr(branch, attr).clone() for key, attr in {
        "H": "last_long_memory", "p": "last_regime_probabilities",
        "prior": "last_regime_priors", "evidence": "last_regime_evidence",
        "Z": "last_latent_states", "candidates": "last_latent_candidates",
        "h_long": "last_h_long", "h_micro": "last_h_micro",
        "h_long_effective": "last_h_long_effective", "h_micro_effective": "last_h_micro_effective",
        "h_temporal": "last_h_temporal",
    }.items()}
    trace.update(q=trace["p"], A=branch.regime_filter.transition_matrix(), prediction=prediction)
    return trace, model._temp_weighted_spatial(x)


def fixed_permutation(size, seed, device):
    # Random cycle: deterministic bijection with no fixed points (unless B=1).
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    permutation = torch.empty(size, dtype=torch.long)
    permutation[order] = order.roll(1)
    return permutation.to(device)


def specifications(shuffle_mode="shuffle"):
    modes = {f"T={value:.2f}": {"routing_temperature": value} for value in TEMPERATURES}
    modes.update({"hard top1": {"mode": "hard"}, "uniform": {"mode": "uniform"},
                  "batch-shuffled p": {"mode": shuffle_mode}, "lagged p": {"mode": "lag"}})
    modes.update({f"alpha={alpha:.2f}": {"sticky_alpha_override": alpha} for alpha in ALPHAS})
    modes.update({f"state{k}": {"mode": f"state{k}"} for k in range(3)})
    modes.update({"zero-micro": {"zero_component": "Z"}, "zero-long": {"zero_component": "H"}})
    return modes


def run_intervention(model, spatial, native, specification, permutation, starts):
    spec = dict(specification)
    zero = spec.pop("zero_component", None)
    if zero is not None:
        trace = native
    else:
        trace = latent_forward_with_routing_intervention(
            model.switching_latent_transformer, native["H"],
            permutation=permutation, sample_starts=starts,
            posterior_reference=native if "sticky_alpha_override" not in spec else None, **spec)
    return complete_readout(model, spatial, trace, zero)


def array(tensor):
    return tensor.detach().cpu().numpy()


def difference(a, b):
    d = (a - b).abs()
    return {"mean": d.mean().item(), "max": d.max().item()}


def impacts(trace, native):
    result = {key: difference(trace[key], native[key]) for key in
              ("prediction", "h_micro", "h_temporal", "p")}
    result["Z_T"] = difference(trace["Z"][:, -1], native["Z"][:, -1])
    result["Z_trajectory"] = {str(t): difference(trace["Z"][:, t], native["Z"][:, t])
                              for t in TIME_POINTS if t < trace["Z"].shape[1]}
    result["per_horizon"] = {str(h): difference(trace["prediction"][:, i], native["prediction"][:, i])
                             for i, h in enumerate(MULTI_HORIZONS)}
    return result


def entropy(p):
    return -(p * np.log(np.maximum(p, 1e-8))).sum(-1)


def probability_stats(q, p=None, prior=None):
    q = np.asarray(q, dtype=np.float64)
    sorted_q = np.sort(q, axis=-1)
    result = {"mean": q.mean(axis=(0, 1)).tolist(), "entropy": float(entropy(q).mean()),
              "min_probability": float(q.min()), "min_probability_by_state": q.min(axis=(0, 1)).tolist(),
              "mean_max": float(sorted_q[..., -1].mean()),
              "margin": float((sorted_q[..., -1] - sorted_q[..., -2]).mean()),
              "occupancy": np.bincount(q.argmax(-1).ravel(), minlength=q.shape[-1]).astype(float).__truediv__(q.shape[0] * q.shape[1]).tolist(),
              "temporal_L1": float(np.abs(np.diff(q, axis=1)).sum(-1).mean())}
    if p is not None:
        p = np.asarray(p, dtype=np.float64)
        result["q_p_L1"] = float(np.abs(q - p).sum(-1).mean())
        result["q_p_KL"] = float((q * (np.log(np.maximum(q, 1e-8)) - np.log(np.maximum(p, 1e-8)))).sum(-1).mean())
    if prior is not None:
        result["prior_entropy"] = float(entropy(prior).mean())
        result["posterior_prior_L1"] = float(np.abs(q - prior).sum(-1).mean())
        result["posterior_prior_KL"] = float((q * (np.log(q + 1e-8) - np.log(prior + 1e-8))).sum(-1).mean())
    return result


def correlation(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-12 and y.std() > 1e-12 else None


def specialization(candidates, p):
    candidates, p = candidates.astype(np.float64), p.astype(np.float64)
    pairs, disagreements = {}, []
    for i, j in itertools.combinations(range(p.shape[-1]), 2):
        left, right = candidates[:, :, i], candidates[:, :, j]
        absolute = np.abs(left - right)
        l1 = absolute.sum(-1)
        cosine = (left * right).sum(-1) / np.maximum(np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), 1e-8)
        pairs[f"{i}-{j}"] = {"L1": float(l1.mean()), "mean_abs": float(absolute.mean()), "cosine": float(cosine.mean())}
        disagreements.append(l1)
    disagreement = np.mean(disagreements, axis=0)
    mixture = (p[..., None] * candidates).sum(-2)
    dominant = np.take_along_axis(candidates, p.argmax(-1)[..., None, None], axis=2).squeeze(2)
    dominant_gap = np.abs(dominant - mixture).sum(-1)
    sorted_p = np.sort(p, axis=-1)
    margin = sorted_p[..., -1] - sorted_p[..., -2]
    lo, hi = np.quantile(margin, [0.25, 0.75])
    low, high = margin <= lo, margin >= hi
    return {"candidate_L2_norm": np.linalg.norm(candidates, axis=-1).mean(axis=(0, 1)).tolist(),
            "weighted_contribution_L2_norm": np.linalg.norm(p[..., None] * candidates, axis=-1).mean(axis=(0, 1)).tolist(),
            "pairwise": pairs, "disagreement_mean_L1": float(disagreement.mean()),
            "margin_dominant_gap_correlation": correlation(margin, dominant_gap),
            "margin_disagreement_correlation": correlation(margin, disagreement),
            "dominant_soft_gap_L1": float(dominant_gap.mean()),
            "confidence_groups": {"definition": "bottom/top quartile of native top1-top2 margin; overlapping windows",
                                  "q25": float(lo), "q75": float(hi),
                                  "low_disagreement_L1": float(disagreement[low].mean()),
                                  "high_disagreement_L1": float(disagreement[high].mean()),
                                  "low_count": int(low.sum()), "high_count": int(high.sum())}}


def metrics(pred, target):
    from cmgm.training.metric_standard import population_metrics
    return population_metrics(pred, target)


def horizon_metrics(pred, target):
    return {str(h): metrics(pred[:, i], target[:, i]) for i, h in enumerate(MULTI_HORIZONS)}


def merge_impacts(items, counts):
    first = items[0]
    if "mean" in first and "max" in first:
        return {"mean": float(np.average([v["mean"] for v in items], weights=counts)),
                "max": max(v["max"] for v in items)}
    return {key: merge_impacts([v[key] for v in items], counts) for key in first}


def evaluate_split(model, loader, split, specs, device, seed, output):
    """Native full loaders; TRAIN only describes routing, with no loss/backward."""
    p_values, prior_values, evidence_values, candidate_values, targets, native_predictions = [], [], [], [], [], []
    collected = {name: {"q": [], "p": [], "prior": [], "pred": [], "impact": []} for name in specs}
    counts, allowed_count, shuffled_count = [], 0, 0
    for index, batch in enumerate(loader):
        x, target = batch[:2]
        x = x.to(device)
        with torch.no_grad():
            native, spatial = normal_reference(model, x)
            permutation = fixed_permutation(len(x), seed + index, device)
            starts = torch.arange(len(x), device=device) + sum(counts)
            allowed_count += int((starts[permutation] <= starts).sum())
            shuffled_count += len(x)
            p_values.append(array(native["p"]))
            prior_values.append(array(native["prior"]))
            evidence_values.append(array(native["evidence"]))
            candidate_values.append(array(native["candidates"]))
            counts.append(len(x))
            targets.append(array(target))
            native_predictions.append(array(native["prediction"]))
            for name, spec in specs.items():
                if split == "TRAIN":
                    q = torch.stack([routing_distribution(native["p"][:, t], spec.get("routing_temperature", 1.0),
                                                        spec.get("mode", "native"),
                                                        native["p"][:, t - 1] if t else None,
                                                        permutation, starts) for t in range(x.shape[1])], 1)
                    collected[name]["q"].append(array(q))
                    continue
                trace = run_intervention(model, spatial, native, spec, permutation, starts)
                for key in ("q", "p", "prior"):
                    collected[name][key].append(array(trace[key]))
                collected[name]["pred"].append(array(trace["prediction"]))
                collected[name]["impact"].append(impacts(trace, native))
        print(f"[{split}] batch {index + 1}/{len(loader)} complete", flush=True)
    p, prior, evidence, candidates, target, native_pred = map(np.concatenate,
        (p_values, prior_values, evidence_values, candidate_values, targets, native_predictions))
    result = {"samples": sum(counts), "batches": len(counts),
              "native_metrics": horizon_metrics(native_pred, target),
              "normal_regime": probability_stats(p, prior=prior),
              "evidence_mean": evidence.mean(axis=(0, 1)).tolist(),
              "evidence_std": evidence.std(axis=(0, 1)).tolist(),
              "candidate_specialization": specialization(candidates, p),
              "causal_shuffle_donor_fraction": allowed_count / shuffled_count, "modes": {}}
    saved_predictions = {"target": target, "native": native_pred}
    for name, values in collected.items():
        q = np.concatenate(values["q"])
        item = {"routing": probability_stats(q, p)}
        if split != "TRAIN":
            pred = np.concatenate(values["pred"])
            posterior, priors = np.concatenate(values["p"]), np.concatenate(values["prior"])
            item.update(routing=probability_stats(q, posterior), metrics=horizon_metrics(pred, target),
                        impact=merge_impacts(values["impact"], counts),
                        posterior=probability_stats(posterior, prior=priors))
            native_distance = probability_stats(q, p)
            item["routing"]["q_native_p_L1"] = native_distance["q_p_L1"]
            item["routing"]["q_native_p_KL"] = native_distance["q_p_KL"]
            saved_predictions[name] = pred
        result["modes"][name] = item
    np.savez_compressed(output / f"{split.lower()}_predictions.npz", **saved_predictions)
    return result


def gradient_diagnostics(model, x, y):
    """Prediction-only eval-mode backward. No optimizer object is constructed."""
    branch = model.switching_latent_transformer
    groups = {"regime evidence": list(branch.regime_filter.regime_evidence.parameters()),
              "transition logits": [branch.regime_filter.transition_logits],
              **{f"G{k}": list(g.parameters()) for k, g in enumerate(branch.latent_transition.generators)},
              "long readout": list(branch.long_memory_readout.parameters()) + list(branch.long_memory_norm.parameters()),
              "micro readout": list(branch.micro_state_readout.parameters()) + list(branch.micro_state_norm.parameters())}
    groups["generators"] = list(branch.latent_transition.parameters())
    groups["balanced readouts"] = groups["long readout"] + groups["micro readout"] + list(branch.state_readout.parameters())
    norms, vectors = {}, {}
    for i, h in enumerate(MULTI_HORIZONS):
        model.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.huber_loss(pred[:, i], y[:, i], delta=HUBER_DELTA)
        loss.backward()
        flat = {name: torch.cat([(param.grad if param.grad is not None else torch.zeros_like(param)).detach().flatten()
                                 for param in params]).cpu() for name, params in groups.items()}
        norms[str(h)] = {name: v.norm().item() for name, v in flat.items()}
        norms[str(h)]["loss"] = loss.item()
        vectors[str(h)] = flat
    cosines = {name: {f"5d-vs-{h}d": F.cosine_similarity(vectors["5"][name], vectors[str(h)][name], dim=0, eps=1e-12).item()
                      for h in (1, 10, 20)} for name in ("regime evidence", "transition logits", "generators", "balanced readouts")}
    model.zero_grad(set_to_none=True)
    return {"norms": norms, "cosines": cosines, "Huber_delta": HUBER_DELTA,
            "method": "model.zero_grad -> normal eval forward -> single-horizon Huber backward; no auxiliary loss, no optimizer"}


def causality_sanity(model, x, native, specs, seed):
    cutoff = x.shape[1] // 2
    changed = x.clone()
    changed[:, cutoff:] = changed[:, cutoff:] * -2 + 7
    branch = model.switching_latent_transformer
    permutation = fixed_permutation(len(x), seed, x.device)
    starts = torch.arange(len(x), device=x.device)
    with torch.no_grad():
        altered, _ = normal_reference(model, changed)
        result = {"prefix_length": cutoff, "normal_prefix_max_diff": {
            key: (native[key][:, :cutoff] - altered[key][:, :cutoff]).abs().max().item()
            for key in ("H", "p", "prior", "evidence", "Z", "candidates")}}
        mode_diffs = {}
        for name, spec in specs.items():
            if "zero_component" in spec:
                continue
            original = latent_forward_with_routing_intervention(branch, native["H"], permutation=permutation, sample_starts=starts, **spec)
            future_changed = latent_forward_with_routing_intervention(branch, altered["H"], permutation=permutation, sample_starts=starts, **spec)
            mode_diffs[name] = {key: (original[key][:, :cutoff] - future_changed[key][:, :cutoff]).abs().max().item()
                                for key in ("p", "q", "Z")}
        result["intervention_prefix_max_diff"] = mode_diffs
        single = model(x[:1])
        result["native_batch_independence_max_diff"] = (single - native["prediction"][:1]).abs().max().item()
    maximum = max(list(result["normal_prefix_max_diff"].values()) + [v for values in mode_diffs.values() for v in values.values()])
    if maximum > 2e-6:
        raise AssertionError(f"future perturbation changed prefix: {maximum}")
    result["window_relative_prefix_passed"] = True
    result["calendar_causality_exception"] = (
        "User-authorized unrestricted batch permutation is a non-calendar-causal stress test; same relative t only."
        if any(spec.get("mode") == "shuffle_unrestricted" for spec in specs.values()) else None
    )
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_markdown_report(report, destination):
    """Render every measured diagnostic, keeping interpretation separate."""
    lines = ["# D0B Regime Routing Diagnostics", "",
             "Checkpoint-only inference。没有训练、optimizer step、scheduler step 或 checkpoint 更新。",
             "所有误差均为收益率空间；Hit 以百分比显示。Δ 为对 native 的平均绝对差；熵用自然对数。",
             "逐批记录统计按样本数加权。概率统计覆盖滑动窗口全部相对时刻（重复日期有重复计数），不是独立交易日样本。", ""]

    def paragraph(text):
        lines.extend([text, ""])

    def heading(text):
        paragraph("## " + text)

    def cell(value):
        if isinstance(value, float):
            return f"{value:.8g}"
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
        lines.append("")

    def metric_cells(item, h="5"):
        m = item["metrics"][h]
        return [m["MAE"], m["RMSE"], m["Hit"] * 100]

    provenance, fixed, splits = report["provenance"], report["fixed"], report["splits"]
    heading("1. 来源、固定批次和实验边界")
    table(["项目", "值"], [[k, v] for k, v in provenance.items() if k != "model_source_sha256"])
    paragraph("`native` 来自真实 `model(x)`；空间分支和 H 在 eval 模式下缓存，干预复用这些不变量。"
              "温度/forced/uniform/shuffle/lag 只改 q；α 只改诊断 helper 的局部 A 并重新递推 p。"
              "T=1 直接令 q=p，另检查 log-softmax 公式的浮点误差。"
              "zero-micro/zero-long 在最终投影和 LayerNorm 之后置零，经过原有融合门和预测头。")
    paragraph("Batch shuffle 按用户明确授权作为**非日历时间因果的压力测试**：固定随机循环置换，同一相对 t，"
              "每批固定 donor，p 的递推仍使用 recipient 自己的 p。不得把此对照解释为可部署的因果模型。")
    heading("2. 本次 native baseline（全部 horizon）")
    table(["Split", "样本数", "Horizon", "MAE", "MSE", "RMSE", "Hit%"],
          [[split, splits[split]["samples"], h, m["MAE"], m["MSE"], m["RMSE"], m["Hit"] * 100]
           for split in ("VAL", "TEST") for h, m in splits[split]["native_metrics"].items()])
    paragraph("本轮起 pooled RMSE=sqrt(MSE)、Hit 不掩码；历史 reference 使用商品平均 RMSE 和掩码 Hit，二者差值不能解释为模型变化。")
    reference = provenance["published_primary_reference"]
    observed = splits["TEST"]["native_metrics"]["5"]
    table(["5d TEST", "给定 reference", "本次 checkpoint", "本次减 reference"],
          [[key, reference[key], observed[key], observed[key] - reference[key]] for key in ("MAE", "RMSE", "Hit")])
    heading("3. 默认等价、因果性与无修改校验")
    paragraph("[D0B routing temperature sanity]\n```json\n" + json.dumps(report["routing_temperature_sanity"], indent=2) + "\n```")
    paragraph("[D0B alpha=0.5 sanity]\n```json\n" + json.dumps(report["alpha_0.5_sanity"], indent=2) + "\n```")
    paragraph("```json\n" + json.dumps(report["causality"], indent=2, ensure_ascii=False) + "\n```")
    paragraph("```json\n" + json.dumps(report["integrity"], indent=2) + "\n```")
    routing_names = [f"T={t:.2f}" for t in TEMPERATURES] + ["hard top1", "uniform", "batch-shuffled p", "lagged p"]
    heading("4. Routing 核心对比（Δ 来自 fixed TEST batch；熵来自完整 TEST）")
    table(["Routing mode", "q entropy", "Z_T Δ", "Prediction Δ", "VAL MAE", "TEST MAE"],
          [[name, splits["TEST"]["modes"][name]["routing"]["entropy"],
            fixed["modes"][name]["impact"]["Z_T"]["mean"], fixed["modes"][name]["impact"]["prediction"]["mean"],
            splits["VAL"]["modes"][name]["metrics"]["5"]["MAE"], splits["TEST"]["modes"][name]["metrics"]["5"]["MAE"]]
           for name in routing_names])
    table(["Mode", "VAL MAE", "VAL RMSE", "VAL Hit%", "TEST MAE", "TEST RMSE", "TEST Hit%"],
          [[name] + metric_cells(splits["VAL"]["modes"][name]) + metric_cells(splits["TEST"]["modes"][name]) for name in routing_names])
    paragraph("T=.50 strong sharpen；T=.75 mild sharpen；T=1 native；T=1.25 mild flatten；T=2 strong flatten。"
              "所有列出的温度均为预先指定的 intervention；没有调参循环、训练或择优重训。")
    heading("5. TRAIN/VAL/TEST routing entropy、margin、occupancy、L1/KL")
    table(["Split", "Mode", "Entropy", "Mean max", "Margin", "Occupancy (0/1/2)", "L1(q,p)", "KL(q||p)"],
          [[split, name] + [splits[split]["modes"][name]["routing"][key] for key in
                           ("entropy", "mean_max", "margin", "occupancy", "q_p_L1", "q_p_KL")]
           for split in ("TRAIN", "VAL", "TEST") for name in routing_names])
    paragraph("Occupancy 仅为 argmax 的描述统计，不能据此将 soft regime 解释为已识别的真实市场标签。"
              "uniform 的三个概率相等；argmax 返回首个索引，所以其 occupancy=[1,0,0] 是 tie-breaking，绝不表示 state0 被选中。")
    heading("6. 固定批次递归影响（mean / max）")
    table(["Mode", "Z_T mean", "Z_T max", "micro mean", "micro max", "temporal mean", "temporal max", "pred mean", "pred max"],
          [[name] + [fixed["modes"][name]["impact"][key][stat] for key in ("Z_T", "h_micro", "h_temporal", "prediction")
                     for stat in ("mean", "max")] for name in routing_names])
    table(["Mode"] + [f"t={t}" for t in TIME_POINTS],
          [[name] + [fixed["modes"][name]["impact"]["Z_trajectory"][str(t)]["mean"] for t in TIME_POINTS] for name in routing_names])
    heading("7. Uniform null control / RoutingFraction")
    paragraph("[D0B uniform-generator-routing] RoutingFraction = mean|native−uniform| / (mean|native−zeroMicro|+1e−8)。"
              "这是干预响应之比，**不是可加性贡献分解或解释方差比例**；门控与预测头有非线性。")
    for scope, modes in [("fixed TEST", fixed["modes"])] + [(split, splits[split]["modes"]) for split in ("VAL", "TEST")]:
        rows = []
        for h in map(str, MULTI_HORIZONS):
            impact = lambda name: modes[name]["impact"]["per_horizon"][h]["mean"]
            rows.append([h, impact("uniform"), impact("zero-micro"), impact("uniform") / (impact("zero-micro") + 1e-8)] + [impact(f"state{k}") for k in range(3)])
        paragraph(scope)
        table(["Horizon", "I_routing", "I_zeroMicro", "RoutingFraction", "state0 Δ", "state1 Δ", "state2 Δ"], rows)
        all_routing = modes["uniform"]["impact"]["prediction"]["mean"]
        all_micro = modes["zero-micro"]["impact"]["prediction"]["mean"]
        paragraph(f"All-horizon I_routing={all_routing:.9g}; I_zeroMicro={all_micro:.9g}; RoutingFraction={all_routing/(all_micro+1e-8):.9g}.")
    heading("8. Persistence α 对比")
    table(["Alpha", "Mean diag(A)", "TEST p entropy", "TEST p temporal L1", "VAL MAE", "TEST MAE"],
          [[a, fixed["modes"][f"alpha={a:.2f}"]["mean_diagonal"],
            splits["TEST"]["modes"][f"alpha={a:.2f}"]["posterior"]["entropy"],
            splits["TEST"]["modes"][f"alpha={a:.2f}"]["posterior"]["temporal_L1"],
            splits["VAL"]["modes"][f"alpha={a:.2f}"]["metrics"]["5"]["MAE"],
            splits["TEST"]["modes"][f"alpha={a:.2f}"]["metrics"]["5"]["MAE"]] for a in ALPHAS])
    for a in ALPHAS:
        name = f"alpha={a:.2f}"
        paragraph(name + " transition matrix")
        table(["From / To", "0", "1", "2"], [[i] + row for i, row in enumerate(fixed["modes"][name]["transition_matrix"])])
    table(["Split", "Alpha", "Prior H", "Posterior H", "Temporal L1", "Mean max p", "Margin", "Occupancy", "p Δ", "Prediction Δ", "MAE", "RMSE", "Hit%"],
          [[split, a] + [splits[split]["modes"][f"alpha={a:.2f}"]["posterior"][key]
                         for key in ("prior_entropy", "entropy", "temporal_L1", "mean_max", "margin", "occupancy")]
           + [splits[split]["modes"][f"alpha={a:.2f}"]["impact"][key]["mean"] for key in ("p", "prediction")]
           + metric_cells(splits[split]["modes"][f"alpha={a:.2f}"])
           for split in ("VAL", "TEST") for a in ALPHAS])
    paragraph("α=0 仍使用 learned softmax(L) Markov prior。α=1 令 A=I，prior 保留上一分布，但 evidence 仍逐时更新 posterior，不能解释成状态永不变化。"
              "所有递推遵循原 D0B：每个 20 日滑窗开始时 p_prev=uniform、Z_prev=0。因此 α=1 是窗口内持续累积 evidence，而非跨整个数据集保持一个永久状态。")
    heading("9. Per-horizon mechanism")
    for scope, modes in [("fixed TEST", fixed["modes"])] + [(split, splits[split]["modes"]) for split in ("VAL", "TEST")]:
        paragraph(scope)
        rows = []
        for h in map(str, MULTI_HORIZONS):
            impact = lambda name: modes[name]["impact"]["per_horizon"][h]["mean"]
            rows.append([h, impact("zero-micro"), impact("zero-long"), impact("zero-micro")/(impact("zero-long")+1e-8), impact("uniform")])
        table(["Horizon", "zero-micro Δ", "zero-long Δ", "micro/long", "uniform Δ"], rows)
        table(["Horizon", "state0 Δ", "state1 Δ", "state2 Δ", "T=.75 Δ", "T=.50 Δ"],
              [[h] + [modes[name]["impact"]["per_horizon"][h]["mean"] for name in ("state0", "state1", "state2", "T=0.75", "T=0.50")]
               for h in map(str, MULTI_HORIZONS)])
    heading("10. 完整逐 horizon 指标（所有干预，无逐 horizon 选温度）")
    for split in ("VAL", "TEST"):
        paragraph(split)
        table(["Mode", "Horizon", "MAE", "RMSE", "Hit%"],
              [[name, h] + metric_cells(item, h) for name, item in splits[split]["modes"].items() for h in map(str, MULTI_HORIZONS)])
    heading("11. 单 horizon prediction-only 梯度与冲突")
    paragraph(report["gradients"]["method"] + f"; Huber delta={report['gradients']['Huber_delta']}。固定 TEST batch 的目标仅用于此局部梯度测量，不改变权重或 inference routing。")
    keys = list(report["gradients"]["norms"]["1"])
    table(["Horizon"] + keys, [[h] + [report["gradients"]["norms"][h][key] for key in keys] for h in map(str, MULTI_HORIZONS)])
    table(["Module", "5d vs 1d", "5d vs 10d", "5d vs 20d"],
          [[name] + list(values.values()) for name, values in report["gradients"]["cosines"].items()])
    heading("12. Native regime 与 candidate specialization")
    table(["Split", "p mean", "p entropy", "mean max", "margin", "occupancy", "prior entropy", "KL(p||prior)", "L1(p,prior)", "temporal L1"],
          [[split] + [splits[split]["normal_regime"][key] for key in ("mean", "entropy", "mean_max", "margin", "occupancy", "prior_entropy", "posterior_prior_KL", "posterior_prior_L1", "temporal_L1")]
           for split in ("TRAIN", "VAL", "TEST")])
    paragraph("Transition / drift:\n```json\n" + json.dumps(report["transition"], indent=2, ensure_ascii=False) + "\n```")
    table(["Split", "candidate L2 norms", "p-weighted contribution L2 norms", "margin vs dominant-soft gap corr", "margin vs disagreement corr"],
          [[split] + [splits[split]["candidate_specialization"][key] for key in ("candidate_L2_norm", "weighted_contribution_L2_norm", "margin_dominant_gap_correlation", "margin_disagreement_correlation")]
           for split in ("TRAIN", "VAL", "TEST")])
    table(["Split", "Pair", "L1 sum", "Mean absolute", "Cosine"],
          [[split, pair, values["L1"], values["mean_abs"], values["cosine"]]
           for split in ("TRAIN", "VAL", "TEST") for pair, values in splits[split]["candidate_specialization"]["pairwise"].items()])
    table(["Split", "margin q25", "margin q75", "Low confidence D", "High confidence D", "n low", "n high"],
          [[split] + [splits[split]["candidate_specialization"]["confidence_groups"][key]
                      for key in ("q25", "q75", "low_disagreement_L1", "high_disagreement_L1", "low_count", "high_count")]
           for split in ("TRAIN", "VAL", "TEST")])
    paragraph("D 为三个 candidate pair 的 L1（对 latent 维求和）再取平均；confidence 用 margin 的底/顶四分位。相关系数仅作描述；相邻滑窗并非独立样本。"
              "Candidate 不同说明函数输出有分化，不足以证明已学到有经济含义的 regime specialization。")
    heading("13. 产物与限制")
    paragraph("`results.json` 保存完整数值；`fixed_native_trace.npz` 保存 H/p/prior/evidence/A/candidates/Z、balanced readout、prediction 和固定 X/y；"
              "`fixed_intervention_traces.npz` 保存各干预递归轨迹；`val_predictions.npz` / `test_predictions.npz` 保存全部目标及所有干预的逐 horizon 预测。")
    paragraph("这是单 checkpoint 的确定性 intervention，未做多 seed 训练或统计显著性验证。VAL/TEST 改善方向一致只作为机制线索，不能视为泛化收益已被确认。"
              "旧数据管线的 bfill、完整区间资产筛选等问题未在本轮更改。")
    paragraph(f"执行时间 {report['elapsed_seconds']:.2f}s。机制归类与唯一优先级结论见同目录 `CONCLUSIONS.md`。完成后停止，不自动实现新模型。")
    destination.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/switching_latent_balanced_readout_best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/d0b_regime_routing_diagnostics")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shuffle-mode", choices=("causal", "unrestricted"), default="causal",
                        help="Unrestricted is a NON-calendar-causal stress control; requires explicit research authorization.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    started = time.time()
    original_hash = sha256(args.checkpoint)
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "cmgm/models").glob("*.py"))}
    from cmgm.scripts.main_ablation import build_data
    data = build_data(SimpleNamespace(batch_size=args.batch_size, seq_len=SEQ_LEN, seed=args.seed))
    # Include TRAIN tail for descriptive coverage; VAL/TEST match existing loaders exactly.
    loaders = {key.upper(): DataLoader(value.dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
               for key, value in data["loaders"].items()}
    n_stock = data["market_indices"]["stock"][1]
    n_bond = data["market_indices"]["bond"][1] - data["market_indices"]["bond"][0]
    model = HeteroMixHopCMGM(data["n_nodes"], data["n_commodities"], n_stock=n_stock, n_bond=n_bond,
                           feat_dim=FEATURE_DIM, variant=VARIANT).to(device).eval()
    payload = _load_checkpoint(model, args.checkpoint, device)
    branch = model.switching_latent_transformer
    assert_d0b(branch)
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fixed = next((batch for index, batch in enumerate(loaders["TEST"]) if index == args.batch_index), None)
    if fixed is None:
        raise IndexError("fixed TEST batch does not exist")
    x, y = fixed[:2]
    x, y = x.to(device), y.to(device)
    specs = specifications("shuffle" if args.shuffle_mode == "causal" else "shuffle_unrestricted")
    report = {"provenance": {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": original_hash,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "checkpoint_best_epoch": payload.get("best_epoch"), "checkpoint_metadata": payload.get("metadata"),
              "checkpoint_best_val_loss": payload.get("best_val_loss"), "variant": VARIANT,
              "fixed_batch_shape": list(x.shape), "fixed_batch_index": args.batch_index,
              "seed": args.seed, "batch_size": args.batch_size, "seq_len": SEQ_LEN,
              "horizons": MULTI_HORIZONS, "torch": torch.__version__, "device": str(device),
              "threads": args.threads, "shuffle_mode": args.shuffle_mode, "model_source_sha256": source_hashes,
              "published_primary_reference": {"MAE": 0.021995, "RMSE": 0.028486, "Hit": 0.491},
              "aggregation": "sample-weighted; pooled RMSE=sqrt(MSE); unmasked sign Hit; entropy natural log; overlapping windows included",
              "preprocessing_note": "Reuses checkpoint-era preprocessing including existing ffill/bfill; intervention causality does not certify historical preprocessing."},
              "fixed": {}, "splits": {}}
    with torch.no_grad():
        native, spatial = normal_reference(model, x)
        permutation = fixed_permutation(len(x), args.seed, device)
        starts = torch.arange(len(x), device=device) + args.batch_index * args.batch_size
        np.savez_compressed(args.output / "fixed_native_trace.npz", **{key: array(value) for key, value in native.items()},
                            h_spatial=array(spatial), X=array(x), target=array(y), permutation=array(permutation), sample_starts=array(starts))
        report["fixed"]["normal_regime"] = probability_stats(array(native["p"]), prior=array(native["prior"]))
        report["fixed"]["candidate_specialization"] = specialization(array(native["candidates"]), array(native["p"]))
        report["fixed"]["modes"] = {}
        traces = {}
        for name, spec in specs.items():
            trace = run_intervention(model, spatial, native, spec, permutation, starts)
            report["fixed"]["modes"][name] = {"impact": impacts(trace, native),
                "routing": probability_stats(array(trace["q"]), array(trace["p"])),
                "posterior": probability_stats(array(trace["p"]), prior=array(trace["prior"])),
                "transition_matrix": array(trace["A"]).tolist(), "mean_diagonal": trace["A"].diag().mean().item()}
            traces.update({f"{name}/{key}": array(trace[key]) for key in ("q", "p", "Z", "candidates", "h_micro", "h_temporal", "prediction")})
            if "sticky_alpha_override" not in spec:
                torch.testing.assert_close(trace["p"], native["p"], rtol=0, atol=0)
        # Independent default helper, no posterior cache, compared with real forward.
        default = complete_readout(model, spatial, latent_forward_with_routing_intervention(branch, native["H"]))
        sanity = {"q_p_max": (default["q"] - native["p"]).abs().max().item(),
                  "formula_T1_q_p_max": (native["p"].clamp_min(1e-8).log().softmax(-1) - native["p"]).abs().max().item(),
                  "prediction_max": (default["prediction"] - native["prediction"]).abs().max().item(),
                  "Z_max": (default["Z"] - native["Z"]).abs().max().item()}
        for key in ("p", "prior", "evidence", "candidates", "Z", "h_long", "h_micro", "h_temporal", "prediction"):
            torch.testing.assert_close(default[key], native[key], rtol=0, atol=2e-6)
        report["routing_temperature_sanity"] = sanity
        alpha_sanity = report["fixed"]["modes"]["alpha=0.50"]["impact"]
        report["alpha_0.5_sanity"] = {key: alpha_sanity[key]["max"] for key in ("p", "Z_T", "prediction")}
        report["alpha_0.5_sanity"]["Z_full_trajectory"] = float(np.abs(traces["alpha=0.50/Z"] - array(native["Z"])).max())
        for value in report["alpha_0.5_sanity"].values():
            assert value <= 2e-6
        logits = branch.regime_filter.transition_logits
        report["transition"] = {"logits": array(logits).tolist(), "native_matrix": array(native["A"]).tolist(),
             "mean_diagonal": native["A"].diag().mean().item(),
             "drift_from_recorded_initial": None,
             "drift_note": "Checkpoint has no initial-logit snapshot/variant metadata. Current source initializes logits to zero; below is a source-assumed comparison only.",
             "source_assumed_zero_init_L2": logits.norm().item(), "source_assumed_zero_init_max_abs": logits.abs().max().item()}
        np.savez_compressed(args.output / "fixed_intervention_traces.npz", **traces)
        report["causality"] = causality_sanity(model, x, native, specs, args.seed)
    print("[D0B routing temperature sanity]", sanity, flush=True)
    print("[D0B alpha=0.5 sanity]", report["alpha_0.5_sanity"], flush=True)
    routing_specs = {name: spec for name, spec in specs.items() if name.startswith("T=") or name in ("hard top1", "uniform", "batch-shuffled p", "lagged p")}
    for split, loader in loaders.items():
        report["splits"][split] = evaluate_split(model, loader, split, routing_specs if split == "TRAIN" else specs,
                                                device, args.seed, args.output)
        (args.output / "results.partial.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["gradients"] = gradient_diagnostics(model, x, y)
    with torch.no_grad():
        after = model(x)
    for key, value in model.state_dict().items():
        if not torch.equal(value, initial_state[key]):
            raise AssertionError(f"model state changed: {key}")
    if sha256(args.checkpoint) != original_hash:
        raise AssertionError("checkpoint bytes changed")
    for name, digest in source_hashes.items():
        if sha256(ROOT / name) != digest:
            raise AssertionError(f"model source changed: {name}")
    report["integrity"] = {"state_dict_bitwise_unchanged": True, "checkpoint_sha256_unchanged": True,
                           "model_sources_unchanged": True,
                           "normal_prediction_max_diff_after_all_diagnostics": (after - native["prediction"]).abs().max().item(),
                           "parameter_gradients_cleared": all(p.grad is None for p in model.parameters()),
                           "optimizer_steps": 0, "scheduler_steps": 0, "training_epochs": 0}
    report["elapsed_seconds"] = time.time() - started
    (args.output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(report, args.output / "REPORT.md")
    print("[D0B uniform-generator-routing]", report["splits"]["TEST"]["modes"]["uniform"], flush=True)
    print(f"[DONE] {args.output / 'results.json'}; elapsed={report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
