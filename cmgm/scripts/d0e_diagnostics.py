"""D0E training integration and checkpoint-only persistence diagnostics.

The only new trainable model parameter is the global upstream sticky logit.
Uniform/forced routing exists exclusively in checkpoint diagnostics, never in
the model or training.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_previous_state_diagnostics import _load_checkpoint
from cmgm.scripts.d0b_regime_routing_diagnostics import (
    ROOT, array, difference, evaluate_split, impacts,
    normal_reference, run_intervention, sha256,
)

VARIANT = "switching_latent_learnable_persistence"
BASE_VARIANT = "switching_latent_balanced_readout"
DISPLAY = "D0E-LearnablePersistence"
LOGIT_KEY = "switching_latent_transformer.regime_filter.sticky_logit"
CONTROLS = {"uniform": {"mode": "uniform"},
            **{f"state{k}": {"mode": f"state{k}"} for k in range(3)},
            "zero-micro": {"zero_component": "Z"}, "zero-long": {"zero_component": "H"}}


@contextlib.contextmanager
def diagnostic_context(model):
    """Probes do not consume training RNG or change parameter .grad fields."""
    was_training = model.training
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        model.eval()
        try:
            yield
        finally:
            model.train(was_training)


def model_arguments(model):
    return dict(num_nodes=model.num_nodes, n_commodities=model.n_commodities,
                n_stock=model.n_stock, n_bond=model.n_bond, feat_dim=model.feat_dim,
                attn_heads=model.attn_heads, attn_dropout=model.attn_dropout,
                attn_prior_scale=model.attn_prior_scale, attn_self_heads=model.attn_self_heads,
                graph_cfg=model.graph_cfg, use_embedding=model.use_embedding, relations=model.relations)


def shared_initialization_check(model, fixed_x, seed=42):
    """Compare an untouched D0E initialization against independently seeded D0B."""
    if model.variant != VARIANT:
        raise ValueError("D0E initialization check requires D0E")
    with diagnostic_context(model), torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        baseline = HeteroMixHopCMGM(variant=BASE_VARIANT, **model_arguments(model)).to(fixed_x.device).eval()
        left, right = dict(baseline.named_parameters()), dict(model.named_parameters())
        assert set(right) - set(left) == {LOGIT_KEY}
        assert set(left) - set(right) == set()
        differences = {name: (left[name] - right[name]).abs().max().item() for name in left}
        mismatches = sum(value != 0 for value in differences.values())
        counts = [sum(p.numel() for p in m.parameters()) for m in (baseline, model)]
        native, _ = normal_reference(baseline, fixed_x)
        actual, _ = normal_reference(model, fixed_x)
        # E is the shared level+dispersion daily token.
        native["E"] = baseline.switching_latent_transformer.last_market_tokens
        actual["E"] = model.switching_latent_transformer.last_market_tokens
        forward = {key: difference(actual[key], native[key])["max"] for key in
                   ("E", "H", "p", "prior", "evidence", "Z", "h_long", "h_micro", "h_temporal", "prediction")}
        transition_diff = difference(actual["A"], native["A"])["max"]
        result = {"max_abs_diff": max(differences.values()), "mismatch_count": mismatches,
                  "D0B_params": counts[0], "D0E_params": counts[1], "difference": counts[1] - counts[0],
                  "alpha": model.switching_latent_transformer.regime_filter.sticky_alpha_value().item(),
                  "D0B_A": array(native["A"]).tolist(), "D0E_A": array(actual["A"]).tolist(),
                  "transition_max_diff": transition_diff, "forward_max_diffs": forward}
        result["PASS"] = mismatches == 0 and counts[1] - counts[0] == 1 and result["alpha"] == .5 and max(forward.values()) <= 2e-6 and transition_diff == 0
        print("[D0E shared init]", json.dumps(result), flush=True)
        if not result["PASS"]:
            raise AssertionError("D0E is not initialization-equivalent to D0B")
        return result


def gradient_groups(model):
    branch = model.switching_latent_transformer
    groups = {"regime evidence": list(branch.regime_filter.regime_evidence.parameters()),
              "transition logits": [branch.regime_filter.transition_logits],
              **{f"G{k}": list(g.parameters()) for k, g in enumerate(branch.latent_transition.generators)},
              "long readout": list(branch.long_memory_readout.parameters()) + list(branch.long_memory_norm.parameters()),
              "micro readout": list(branch.micro_state_readout.parameters()) + list(branch.micro_state_norm.parameters())}
    if branch.regime_filter.learnable_sticky_alpha:
        groups["sticky_logit"] = [branch.regime_filter.sticky_logit]
    groups["generators"] = list(branch.latent_transition.parameters())
    groups["balanced readouts"] = groups["long readout"] + groups["micro readout"] + list(branch.state_readout.parameters())
    return groups


def flattened_gradients(loss, groups, retain_graph=False):
    params = list(dict.fromkeys(p for values in groups.values() for p in values))
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    by_id = {id(p): g.detach() if g is not None else torch.zeros_like(p) for p, g in zip(params, grads)}
    return {name: torch.cat([by_id[id(p)].flatten() for p in values]).cpu() for name, values in groups.items()}


def prediction_loss(pred, target, horizon=None):
    from cmgm.training.train import make_loss
    criterion = make_loss()
    if horizon is not None:
        return criterion(pred[:, horizon], target[:, horizon])
    return sum(criterion(pred[:, i], target[:, i]) for i in range(pred.shape[1]))


def gradient_probe(model, batch, stage, per_horizon=False):
    """No optimizer, backward side effects, auxiliary-loss changes or RNG drift."""
    device = next(model.parameters()).device
    x, target = (v.to(device) for v in batch[:2])
    branch = model.switching_latent_transformer
    groups = gradient_groups(model)
    with diagnostic_context(model), torch.enable_grad():
        pred = model(x)
        primary = prediction_loss(pred, target)
        total = primary + branch.switch_loss() + model.regime_diversity_loss()
        pred_grads = flattened_gradients(primary, groups, retain_graph=True)
        total_grads = flattened_gradients(total, groups)
        result = {"stage": stage, "batch_shape": list(x.shape), "mode": "eval (fixed dropout-free probe)",
                  "prediction_loss": primary.item(), "total_loss": total.item(),
                  "switch_beta": branch.regime_filter.current_beta,
                  "gradients": {name: {"prediction_only_norm": v.norm().item(),
                                       "total_loss_norm": total_grads[name].norm().item()}
                                for name, v in pred_grads.items()}}
        value = branch.regime_filter.sticky_alpha_value()
        alpha = value.detach().item() if isinstance(value, torch.Tensor) else value
        derivative = alpha * (1 - alpha)
        if "sticky_logit" in pred_grads:
            result.update(alpha=alpha, sticky_logit=branch.regime_filter.sticky_logit.item(),
                          sigmoid_derivative=derivative,
                          sticky_prediction_signed=pred_grads["sticky_logit"].item(),
                          sticky_total_signed=total_grads["sticky_logit"].item())
        if per_horizon:
            vectors, horizon_results = {}, {}
            for i, h in enumerate(config.MULTI_HORIZONS):
                loss = prediction_loss(model(x), target, i)
                vector = flattened_gradients(loss, groups)
                vectors[h] = vector
                item = {"loss": loss.item(), "norms": {name: v.norm().item() for name, v in vector.items()}}
                if "sticky_logit" in vector:
                    grad_a = vector["sticky_logit"].item()
                    item.update(dL_d_sticky_logit=grad_a, sign=float(np.sign(grad_a)),
                                magnitude=abs(grad_a), dL_d_alpha=grad_a / derivative if derivative > 0 else None)
                horizon_results[str(h)] = item
            result["per_horizon"] = horizon_results
            result["cosines"] = {name: {f"5d-vs-{h}d": F.cosine_similarity(vectors[5][name], vectors[h][name], dim=0, eps=1e-12).item()
                                        for h in (1, 10, 20)}
                                 for name in ("regime evidence", "transition logits", "generators", "balanced readouts")}
            if "sticky_logit" in groups:
                signs = [horizon_results[str(h)]["sign"] for h in (1, 5, 10, 20)]
                result["global_persistence_horizon_conflict"] = signs[0] != 0 and all(s == -signs[0] for s in signs[1:])
    print(f"[D0E gradients {stage}] {json.dumps(result)}", flush=True)
    return result


def transition_diagnostics(model, initial_logits=None):
    filtering = model.switching_latent_transformer.regime_filter
    with torch.no_grad():
        alpha = float(filtering.sticky_alpha_value())
        L = filtering.transition_logits
        S = L.softmax(-1)
        A = filtering.transition_matrix()
        L0 = torch.zeros_like(L) if initial_logits is None else L.new_tensor(initial_logits)
        S0 = L0.softmax(-1)
        I = torch.eye(len(L), device=L.device, dtype=L.dtype)
        A0 = .5 * I + .5 * S0
        alpha_effect = (alpha - .5) * (I - S0)
        logits_effect = (1 - alpha) * (S - S0)
        return {"alpha": alpha, "sticky_logit": filtering.sticky_logit.item() if filtering.learnable_sticky_alpha else None,
                "sigmoid_derivative": alpha * (1 - alpha), "S": array(S).tolist(), "A": array(A).tolist(),
                "logits": array(L).tolist(), "mean_diagonal": A.diag().mean().item(),
                "row_entropy": array(-(A * A.clamp_min(1e-8).log()).sum(-1)).tolist(),
                "initial_reference": "recorded initialization" if initial_logits is not None else "current-source zero-init assumption; no checkpoint initial snapshot",
                "transition_logits_drift_L2": (L - L0).norm().item(),
                "effective_A_change_L2": (A - A0).norm().item(),
                "alpha_only_change_L2": alpha_effect.norm().item(), "logits_change_at_current_alpha_L2": logits_effect.norm().item(),
                "decomposition": "A-A0 = (alpha-.5)(I-S0) + (1-alpha)(S-S0)",
                "decomposition_max_error": (A - A0 - alpha_effect - logits_effect).abs().max().item()}


def fixed_sanity(model, x, label="D0E", raise_on_failure=True):
    """Causal temporal prefixes, batch independence, and node relabeling."""
    branch = model.switching_latent_transformer
    cutoff = x.shape[1] // 2

    def trajectory(values):
        branch(values)
        trace = {"E": branch.last_market_tokens.clone(), "H": branch.last_long_memory.clone(),
                 "p": branch.last_regime_probabilities.clone(), "Z": branch.last_latent_states.clone()}
        readouts = {key: [] for key in ("h_long", "h_micro", "h_temporal")}
        for t in range(values.shape[1]):
            readout = (branch.readout_by_horizon if getattr(branch, "horizon_specific_state_readout", False)
                       else branch.readout)
            h = readout(trace["H"][:, t], trace["Z"][:, t])
            readouts["h_long"].append(branch.last_h_long.clone())
            readouts["h_micro"].append(branch.last_h_micro.clone())
            readouts["h_temporal"].append(h)
        trace.update({key: torch.stack(v, 1) for key, v in readouts.items()})
        if getattr(branch, "horizon_specific_state_readout", False):
            trace.update({f"h_temporal_{h}": trace["h_temporal"][:, :, i]
                          for i, h in enumerate(branch.forecast_horizons)})
        return trace

    with diagnostic_context(model), torch.no_grad():
        before = trajectory(x)
        changed = x.clone()
        changed[:, cutoff:] = changed[:, cutoff:] * -2 + 7
        after = trajectory(changed)
        causality = {key: difference(before[key][:, :cutoff], after[key][:, :cutoff])["max"] for key in before}
        pred = model(x)
        order = torch.arange(len(x) - 1, -1, -1, device=x.device)
        batch = difference(model(x[order]), pred[order])["max"]
        single = difference(model(x[:1]), pred[:1])["max"]
        # Temporal branch is invariant to within-market asset permutation.
        # Full graph also requires relabeling its asset-specific embeddings.
        market_checks = {}
        sizes = (model.n_stock, model.n_bond, model.n_commodities)
        start = 0
        for name, size in zip(("stock", "bond", "commodity"), sizes):
            node_order = torch.arange(model.num_nodes, device=x.device)
            node_order[start:start + size] = node_order[start:start + size].flip(0)
            permuted = trajectory(x[:, :, node_order])
            temporal = max(difference(before[key], permuted[key])["max"] for key in before)
            relabeled = HeteroMixHopCMGM(variant=model.variant, **model_arguments(model)).to(device=x.device, dtype=x.dtype).eval()
            state = {key: v.clone() for key, v in model.state_dict().items()}
            for key in ("graph_learner.E1", "graph_learner.E2"):
                state[key] = state[key][node_order]
            relabeled.load_state_dict(state, strict=True)
            full = difference(relabeled(x[:, :, node_order]), pred)["max"]
            market_checks[name] = {"temporal_invariance_max": temporal, "full_model_with_graph_relabeling_max": full}
            start += size
        values = list(causality.values()) + [batch, single] + [v for item in market_checks.values() for v in item.values()]
        result = {"prefix_cutoff": cutoff, "causality": causality, "batch_permutation_max": batch,
                  "single_sample_max": single, "within_market": market_checks,
                  "readout_note": "Readout applied to H_t/Z_t at each t; full-window spatial pooling is not a prefix forecast",
                  "PASS": max(values) <= 3e-6}
        if not result["PASS"] and raise_on_failure:
            raise AssertionError(f"{label} invariance check failed: {result}")
        print(f"[{label} causality/batch/market sanity]", json.dumps(result), flush=True)
        return result


def micro_diagnostics(model, native):
    branch = model.switching_latent_transformer
    z, H = native["Z"], native["H"]
    long_norm = native["h_long"].norm(dim=-1).mean().item()
    micro_norm = native["h_micro"].norm(dim=-1).mean().item()
    W = branch.state_readout.weight.detach()
    half = W.shape[1] // 2
    return {"mean_Z_norm": z.norm(dim=-1).mean().item(),
            "mean_delta_Z_norm": (z[:, 1:] - z[:, :-1]).norm(dim=-1).mean().item(),
            "consecutive_cosine": F.cosine_similarity(z[:, 1:], z[:, :-1], dim=-1).mean().item(),
            "Z_T_H_T_raw_ratio": (z[:, -1].norm(dim=-1) / H[:, -1].norm(dim=-1).clamp_min(1e-8)).mean().item(),
            "h_long_norm": long_norm, "h_micro_norm": micro_norm, "micro_long_norm_ratio": micro_norm / (long_norm + 1e-8),
            "W_micro_W_long": (W[:, half:].norm() / W[:, :half].norm().clamp_min(1e-8)).item(),
            "W_definition": "Frobenius norm ratio of micro/long halves of final balanced state_readout.weight"}


def collect_checkpoint(model, payload, data, output, label, fixed_batch):
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    x = fixed_batch[0].to(device)
    with diagnostic_context(model):
        with torch.no_grad():
            native, spatial = normal_reference(model, x)
            fixed = {"micro": micro_diagnostics(model, native), "controls": {}}
            for name, spec in CONTROLS.items():
                trace = run_intervention(model, spatial, native, spec, None, None)
                torch.testing.assert_close(trace["p"], native["p"], rtol=0, atol=0)
                fixed["controls"][name] = impacts(trace, native)
            np.savez_compressed(output / "fixed_native.npz", **{k: array(v) for k, v in native.items()})
        splits = {}
        for name, loader in data["loaders"].items():
            # TRAIN full tail is descriptive only, never changes training.
            full = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False, drop_last=False)
            print(f"[D0E {label} {name.upper()} checkpoint diagnostics]", flush=True)
            splits[name.upper()] = evaluate_split(model, full, name.upper(), {"uniform": CONTROLS["uniform"]} if name == "train" else CONTROLS,
                                                   device, 42, output)
        metadata = payload.get("metadata", {})
        result = {"variant": model.variant, "params": sum(p.numel() for p in model.parameters()),
                  "best_epoch": payload.get("best_epoch"), "history": payload.get("history", {}),
                  "transition": transition_diagnostics(model, metadata.get("initial_transition_logits")),
                  "fixed": fixed, "splits": splits, "sanity": fixed_sanity(model, x),
                  "gradients": gradient_probe(model, fixed_batch, f"{label} best checkpoint", per_horizon=True)}
    return result


def comparison_report(d0e, payload, data, d0b_checkpoint, output, seed=42, initialization=None,
                      checkpoint_path=None):
    """Called after training, or explicitly with --checkpoint. Never trains."""
    output.mkdir(parents=True, exist_ok=False)
    d0b_hash = sha256(d0b_checkpoint)
    state_before = {k: v.detach().clone() for k, v in d0e.state_dict().items()}
    with diagnostic_context(d0e):
        fixed = next(iter(data["loaders"]["test"]))
        baseline = HeteroMixHopCMGM(variant=BASE_VARIANT, **model_arguments(d0e)).to(next(d0e.parameters()).device)
        baseline_payload = _load_checkpoint(baseline, Path(d0b_checkpoint), next(d0e.parameters()).device)
        for model, saved in ((d0e, payload), (baseline, baseline_payload)):
            epoch = saved.get("best_epoch", 1)
            model.switching_latent_transformer.set_epoch(epoch)
        report = {"seed": seed, "fixed_batch_shape": list(fixed[0].shape), "initialization": initialization,
                  "metadata": payload.get("metadata", {}), "D0B_checkpoint": str(Path(d0b_checkpoint).resolve()),
                  "D0B_checkpoint_sha256": d0b_hash, "models": {}}
        if checkpoint_path is not None:
            report["D0E_checkpoint"] = str(Path(checkpoint_path).resolve())
            report["D0E_checkpoint_sha256"] = sha256(checkpoint_path)
        for label, model, saved in (("D0B", baseline, baseline_payload), ("D0E", d0e, payload)):
            report["models"][label] = collect_checkpoint(model, saved, data, output / label, label, fixed)
    assert sha256(d0b_checkpoint) == d0b_hash
    if checkpoint_path is not None:
        assert sha256(checkpoint_path) == report["D0E_checkpoint_sha256"]
    assert all(torch.equal(value, state_before[key]) for key, value in d0e.state_dict().items())
    report["integrity"] = {"checkpoint_diagnostics_did_not_update_parameters": True, "D0B_checkpoint_unchanged": True}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    write_report(report, output / "REPORT.md")
    print(f"[D0E REPORT] {output / 'REPORT.md'}", flush=True)
    return report


def functional_values(model_result, split, horizon=None):
    modes = model_result["splits"][split]["modes"]
    def value(name):
        impact = modes[name]["impact"]
        return (impact["per_horizon"][str(horizon)] if horizon is not None else impact["prediction"])["mean"]
    micro, long, routing = value("zero-micro"), value("zero-long"), value("uniform")
    return [micro, long, micro / (long + 1e-8), routing, routing / (micro + 1e-8)]


def write_report(report, path):
    models = report["models"]
    history = models["D0E"]["history"]
    lines = ["# D0E-LearnablePersistence", "",
             "唯一新增训练参数为 sticky_logit，初始化为 0；alpha=sigmoid(sticky_logit)。"
             "D0E 改变 upstream posterior Markov prior；D0D 改变 generator 输入尺度；routing temperature 改变 downstream mixture。本变体仅研究前者。", ""]
    def paragraph(text):
        lines.extend([text, ""])
    def heading(text):
        paragraph("## " + text)
    def fmt(value):
        if isinstance(value, float): return f"{value:.9g}"
        if isinstance(value, (dict, list)): return json.dumps(value, ensure_ascii=False)
        return str(value)
    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        lines.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in rows)
        lines.append("")
    def metric(item, split, horizon="5", mode=None):
        source = item["splits"][split]
        m = source["native_metrics"][horizon] if mode is None else source["modes"][mode]["metrics"][horizon]
        return [m["MAE"], m["MSE"], m["RMSE"], m["Hit"] * 100]
    heading("来源与初始化校验")
    table(["字段", "值"], [[k, v] for k, v in report.items() if k != "models"])
    paragraph("Checkpoint final_learned_alpha 指恢复 best epoch 后保存在权重中的 alpha；last_training_epoch_alpha 为 early stopping 前最后一轮，两者分开记录。"
              "D0B 按本次实际加载 checkpoint 计算，不使用 inference override 作为训练目标。")
    heading("表 1：Performance（primary 5d）")
    table(["Variant", "Params", "BestEpoch", "Alpha", "VAL MAE/MSE/RMSE/Hit%", "TEST MAE/MSE/RMSE/Hit%"],
          [[label, m["params"], m["best_epoch"], m["transition"]["alpha"], metric(m,"VAL"), metric(m,"TEST")] for label,m in models.items()])
    heading("表 2：Alpha trajectory")
    rows=[]
    for label, epoch in [("1",1),("5",5),("10",10),("best",history.get("best_epoch",0)),("final",history.get("final_epoch",0))]:
        if epoch and epoch <= len(history.get("alpha_history", [])):
            i=epoch-1
            rows.append([label,epoch,history["sticky_logit_history"][i],history["alpha_history"][i],history["train_loss"][i],history["val_loss"][i],history["lr_history"][i],history["switch_beta"][i]])
    table(["Label","Epoch","sticky_logit","alpha","train loss","val loss","LR","switch beta"],rows)
    paragraph("完整 alpha_history / sticky_logit_history / train_loss / val_loss / lr_history / switch_beta 保存在 checkpoint 和 results.json；没有对 sticky_logit 单独设置 optimizer group 或 weight decay。")
    heading("表 3：Regime behavior")
    table(["Variant","Split","alpha","p mean","prior H","posterior H","mean max p","margin","occupancy","temporal L1","KL(p,prior)","L1(p,prior)","A mean diagonal","L drift"],
          [[label,split,m["transition"]["alpha"]] + [m["splits"][split]["normal_regime"][key] for key in
             ("mean","prior_entropy","entropy","mean_max","margin","occupancy","temporal_L1","posterior_prior_KL","posterior_prior_L1")]
           + [m["transition"]["mean_diagonal"],m["transition"]["transition_logits_drift_L2"]]
           for label,m in models.items() for split in ("TRAIN","VAL","TEST")])
    paragraph("概率按全部滑动窗口的相对时刻汇总，重复日期重复计数。Occupancy 仅为描述统计。Alpha 增加不必导致 posterior temporal movement 降低。")
    for label,m in models.items():
        heading(label + " transition decomposition / saturation")
        paragraph("```json\n" + json.dumps(m["transition"],indent=2,ensure_ascii=False) + "\n```")
    heading("表 4：Functional utilization（全部 horizons 平均）")
    table(["Variant","Split","zero-micro","zero-long","micro/long","native-vs-uniform","RoutingFraction"],
          [[label,split]+functional_values(m,split) for label,m in models.items() for split in ("VAL","TEST")])
    paragraph("D0B 既有 RoutingFraction reference：TEST≈2.09%，VAL≈2.32%；实际对照以上方重新加载结果为准。"
              "RoutingFraction 是非线性干预响应比，不是可加性贡献。Uniform 仅在诊断中替换 generator weighting，p 递推不变，Z 完整递推。")
    table(["Variant","Split","uniform VAL/TEST 5d MAE/MSE/RMSE/Hit%","uniform prediction mean/max diff"],
          [[label,split,metric(m,split,mode="uniform"),m["splits"][split]["modes"]["uniform"]["impact"]["prediction"]]
           for label,m in models.items() for split in ("VAL","TEST")])
    heading("表 5：Per horizon")
    for split in ("VAL","TEST"):
        paragraph(split)
        table(["Horizon","D0B MAE/MSE/RMSE/Hit%","D0E MAE/MSE/RMSE/Hit%","MAE delta E-B","zero-micro E","zero-long E","micro/long E","routing impact E","RoutingFraction E"],
              [[h,metric(models["D0B"],split,str(h)),metric(models["D0E"],split,str(h)),
                metric(models["D0E"],split,str(h))[0]-metric(models["D0B"],split,str(h))[0]] + functional_values(models["D0E"],split,h)
               for h in config.MULTI_HORIZONS])
        table(["Variant","Horizon","zero-micro","zero-long","micro/long","routing impact","RoutingFraction","state0 impact","state1 impact","state2 impact"],
              [[label,h]+functional_values(m,split,h)+[m["splits"][split]["modes"][f"state{k}"]["impact"]["per_horizon"][str(h)]["mean"] for k in range(3)]
               for label,m in models.items() for h in config.MULTI_HORIZONS])
    heading("表 6：Checkpoint gradients（固定 TEST batch）")
    table(["Variant","Module","prediction-only norm","total-loss norm"],
          [[label,key,v["prediction_only_norm"],v["total_loss_norm"]] for label,m in models.items() for key,v in m["gradients"]["gradients"].items()])
    paragraph("Prediction-only = 原多 horizon Huber 之和；total = 加上原 switch regularizer 和原 diversity loss。"
              "诊断采用 eval 去掉 dropout；autograd.grad 不写入参数 .grad，不消费训练 RNG，不执行 optimizer step。")
    heading("Epoch 1/5/10/best 梯度（固定 TRAIN batch）")
    table(["Stage","alpha","sticky_logit","dLpred/da","dLtotal/da","transition grad pred/total"],
          [[stage,v.get("alpha"),v.get("sticky_logit"),v.get("sticky_prediction_signed"),v.get("sticky_total_signed"),v["gradients"]["transition logits"]]
           for stage,v in history.get("epoch_diagnostics",{}).items()])
    heading("Per-horizon sticky gradients / conflict")
    gradients=models["D0E"]["gradients"]
    table(["Horizon","dL/d(sticky_logit)","sign","magnitude","dL/dalpha"],
          [[h]+[v[k] for k in ("dL_d_sticky_logit","sign","magnitude","dL_d_alpha")] for h,v in gradients["per_horizon"].items()])
    table(["Variant","Module","cos(5d,1d)","cos(5d,10d)","cos(5d,20d)"],
          [[label,key]+list(v.values()) for label,m in models.items() for key,v in m["gradients"]["cosines"].items()])
    paragraph(f"Global persistence horizon conflict (fixed batch): {gradients['global_persistence_horizon_conflict']}. "
              "标量梯度报告符号和幅值；dL/dalpha 由 sigmoid 导数换算，导数为 0 时报告 null，不进行 clamp 或改变参数化。")
    heading("Candidates / micro state / balanced readout / forced controls")
    for label,m in models.items():
        paragraph(label + " fixed TEST micro/readout:\n```json\n" + json.dumps(m["fixed"]["micro"],indent=2) + "\n```")
        table(["State","Z_T mean/max","h_micro mean/max","h_temporal mean/max","prediction mean/max"],
              [[k]+[m["fixed"]["controls"][f"state{k}"][key] for key in ("Z_T","h_micro","h_temporal","prediction")] for k in range(3)])
        for split in ("TRAIN","VAL","TEST"):
            paragraph(label + " " + split + " candidates:\n```json\n" + json.dumps(m["splits"][split]["candidate_specialization"],indent=2,ensure_ascii=False) + "\n```")
        paragraph(label + " sanity:\n```json\n" + json.dumps(m["sanity"],indent=2,ensure_ascii=False) + "\n```")
    heading("六个问题：数值观察，非自动模型替换")
    alpha=models["D0E"]["transition"]["alpha"]
    paragraph(f"1–2. Best alpha={alpha:.9g}，相对 .5 的差为 {alpha-.5:+.9g}；方向={'更强 persistence' if alpha>.5 else ('更强 transition mixing' if alpha<.5 else '未改变')}。完整轨迹见表2；不能以接近1作为成功条件。")
    improvements={s: all(e<b for e,b in zip(metric(models["D0E"],s)[:2],metric(models["D0B"],s)[:2])) for s in ("VAL","TEST")}
    paragraph(f"3. 5d MAE 与 RMSE 是否都降低：{improvements}。细微数值差异不自动构成显著泛化改善。")
    paragraph("4. RoutingFraction D0E−D0B：" + fmt({s:functional_values(models["D0E"],s)[-1]-functional_values(models["D0B"],s)[-1] for s in ("VAL","TEST")}) + "。需要结合 entropy、candidate 分化和误差方向解释功能性利用。")
    paragraph("5. Transition logits 的实际作用需结合权重 (1−alpha)、A 分解和 prediction gradient：" + fmt({"1-alpha":1-alpha,"transition_grad":gradients["gradients"]["transition logits"],"sticky_prediction_grad":gradients["sticky_prediction_signed"],"sigmoid_derivative":gradients["sigmoid_derivative"]}) + "。非零梯度只说明路径可达，不等于主要预测贡献。")
    paragraph(f"6. 单一 global persistence 的 horizon 梯度冲突：{gradients['global_persistence_horizon_conflict']}。若1d方向与5/10/20d相反且指标呈相应权衡，标记 multi-horizon persistence conflict；不改变global alpha。")
    paragraph("Case A 需 alpha 实际偏移、VAL/TEST 同方向改善及健康机制；仅浓度变化而误差近似不变对应 B；训练后变差对应 C；alpha≈.5 且性能相当对应 D；alpha≈1 且L梯度衰减需检查 E；跨horizon的权衡与梯度冲突对应 F。"
              "Case 归类应结合效应大小人工复核。本脚本不自动替换 D0B、不调参、不启动下一变体。")
    paragraph("本次 baseline 由实际 D0B checkpoint 重新计算。历史 inference alpha=.5/.75/1 TEST MAE 为 .021994783/.021984348/.021917269，VAL 为 .023200622/.023188258/.023169838，仅作对照，不能要求训练alpha向1收敛。")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_d0e(args, device, data):
    """One formal variant, using the existing train() with unchanged defaults."""
    from cmgm.scripts.main_ablation import _checkpoint_path_for_variant, evaluate_primary_horizon, print_diagnostics
    from cmgm.training.train import train
    if not args.d0b_checkpoint.is_file():
        raise FileNotFoundError(f"D0B comparison checkpoint missing: {args.d0b_checkpoint}")
    if args.epochs < 1:
        raise ValueError("D0E training requires at least one epoch")
    set_seed(args.seed)
    kwargs = dict(num_nodes=data["n_nodes"], n_commodities=data["n_commodities"],
                  n_stock=data["market_indices"]["stock"][1],
                  n_bond=data["market_indices"]["bond"][1] - data["market_indices"]["bond"][0], feat_dim=config.FEATURE_DIM)
    model = HeteroMixHopCMGM(variant=VARIANT, **kwargs)
    # Dataset sampling is CPU-only; do not initialize or snapshot CUDA devices.
    with torch.random.fork_rng(devices=[]):
        fixed_test = next(iter(data["loaders"]["test"]))
        fixed_train = next(iter(data["loaders"]["train"]))
    initial = shared_initialization_check(model, fixed_test[0], args.seed)
    model = model.to(device)
    model._experiment_seed = args.seed
    model.switching_latent_transformer._initial_transition_logits = (
        model.switching_latent_transformer.regime_filter.transition_logits.detach().cpu().clone()
    )
    initial_logits = array(model.switching_latent_transformer.regime_filter.transition_logits).tolist()
    gradient_probe(model, fixed_train, "initial")
    path = _checkpoint_path_for_variant(VARIANT, args.checkpoint_dir)
    metadata = {"variant": VARIANT, "display_name": DISPLAY,
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
                "seed": args.seed, "seq_len": args.seq_len, "initial_alpha": .5,
                "initial_transition_logits": initial_logits, "initialization_check": initial,
                "protocol": {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
                             "optimizer": "Adam", "lr": config.LEARNING_RATE, "weight_decay": config.WEIGHT_DECAY,
                             "loss": config.LOSS_TYPE, "Huber_delta": config.HUBER_DELTA,
                             "horizons": config.MULTI_HORIZONS, "scheduler": "ReduceLROnPlateau"}}
    started = time.time()
    history = train(model, data["loaders"]["train"], data["loaders"]["val"],
                    torch.empty(2, 0, dtype=torch.long), torch.zeros(0), device,
                    num_epochs=args.epochs, patience=args.patience,
                    checkpoint_path=str(path), checkpoint_metadata=metadata,
                    epoch_diagnostic=lambda active, stage: gradient_probe(active, fixed_train, stage))
    # Reload the serialized best checkpoint strictly before final diagnostics.
    payload = _load_checkpoint(model, path, device)
    normalized, original, target = evaluate_primary_horizon(model, data["loaders"]["test"], data, device)
    legacy = print_diagnostics(model, VARIANT, data["loaders"], device)
    output = args.d0e_report_dir / time.strftime("%Y%m%d_%H%M%S")
    report = comparison_report(model, payload, data, args.d0b_checkpoint, output, args.seed, initial, path)
    from cmgm.training.evaluate import compute_metrics
    zero = compute_metrics(np.zeros_like(target), target)
    return {"variant": DISPLAY, "params": initial["D0E_params"], "time": time.time() - started,
            "MAE": normalized["MAE"], "MSE": normalized["MSE"], "RMSE": normalized["RMSE"], "Hit_Ratio": normalized["Hit_Ratio"],
            "vs_zero_pct": (normalized["MAE"] / zero["MAE"] - 1) * 100,
            "mn": normalized, "mo": original, "diagnostics": legacy,
            "report_path": str(output / "REPORT.md"), "alpha": report["models"]["D0E"]["transition"]["alpha"]}


def main():
    parser = argparse.ArgumentParser(description="D0E checkpoint-only diagnostics; no training")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d0b-checkpoint", type=Path, default=ROOT / "checkpoints/switching_latent_balanced_readout_best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    from cmgm.scripts.main_ablation import build_data
    data = build_data(SimpleNamespace(batch_size=args.batch_size, seq_len=config.SEQ_LEN, seed=args.seed))
    model = HeteroMixHopCMGM(data["n_nodes"], data["n_commodities"], n_stock=data["market_indices"]["stock"][1],
                           n_bond=data["market_indices"]["bond"][1] - data["market_indices"]["bond"][0], variant=VARIANT).to(device)
    digest = sha256(args.checkpoint)
    payload = _load_checkpoint(model, args.checkpoint, device)
    comparison_report(model, payload, data, args.d0b_checkpoint, args.output, args.seed,
                      payload.get("metadata", {}).get("initialization_check"), args.checkpoint)
    assert sha256(args.checkpoint) == digest


if __name__ == "__main__":
    main()
