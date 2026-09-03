"""Inference-only previous-micro-state utilization diagnostics for D0B/D0D.

This module deliberately does not alter the model forward path.  It loads a
trained D0B or D0D checkpoint, freezes H/p from one fixed test
batch, and replays only the existing regime-specific latent generators under
controlled interventions to their previous-state input.

Usage:
    python -m cmgm.scripts.d0b_previous_state_diagnostics \
        --checkpoint /path/to/d0b_best.pt --batch-index 0
    python -m cmgm.scripts.d0b_previous_state_diagnostics \
        --variant switching_latent_balanced_transition \
        --checkpoint /path/to/d0d_best.pt --batch-index 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch

from cmgm.config import BATCH_SIZE, FEATURE_DIM, RANDOM_SEED, SEQ_LEN
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM


@dataclass
class ReplayResult:
    """Outputs of one recursively self-consistent latent intervention."""

    name: str
    long_memory: torch.Tensor
    probabilities: torch.Tensor
    z: torch.Tensor
    candidates: torch.Tensor
    h_micro: torch.Tensor
    h_temporal: torch.Tensor
    prediction: torch.Tensor


def _mean_max(left: torch.Tensor, right: torch.Tensor):
    difference = (left - right).abs()
    return difference.mean().item(), difference.max().item()


def _persistence(z: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    norms = z.norm(dim=-1)
    if z.shape[1] < 2:
        return {
            "mean_norm": norms.mean().item(),
            "mean_change": 0.0,
            "consecutive_cosine": float("nan"),
        }
    change = (z[:, 1:] - z[:, :-1]).norm(dim=-1)
    cosine = torch.nn.functional.cosine_similarity(
        z[:, 1:], z[:, :-1], dim=-1, eps=eps
    )
    return {
        "mean_norm": norms.mean().item(),
        "mean_change": change.mean().item(),
        "consecutive_cosine": cosine.mean().item(),
    }


def _fixed_derangement(batch_size: int, seed: int, device: torch.device):
    """Return a reproducible non-identity batch permutation when B > 1."""
    if batch_size <= 1:
        return torch.arange(batch_size, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    identity = torch.arange(batch_size)
    for _ in range(256):
        order = torch.randperm(batch_size, generator=generator)
        if not torch.any(order == identity):
            return order.to(device)
    # Guaranteed derangement fallback; still deterministic.
    return torch.roll(identity, shifts=1).to(device)


def replay_latent_trajectory(
    branch,
    long_memory: torch.Tensor,
    probabilities: torch.Tensor,
    mode: str = "normal",
    scale: float = 1.0,
    permutation: Optional[torch.Tensor] = None,
    feature_permutation: Optional[torch.Tensor] = None,
    forced_state: Optional[int] = None,
):
    """Replay D0B generators without changing H, p, or trained parameters.

    Every intervention is recursive on its own generated trajectory.  The
    lagged definition is t=0 -> 0, t=1 -> Z0, t>=2 -> Z_(t-2).  No future
    element is ever read.
    """
    allowed = {
        "normal", "zero", "sign_flip", "scaled", "shuffle", "lagged",
        "feature_permute",
    }
    if mode not in allowed:
        raise ValueError(f"unknown previous-state intervention: {mode}")
    batch_size, time_steps, _ = long_memory.shape
    if forced_state is not None and not 0 <= forced_state < branch.K:
        raise ValueError("forced_state is outside [0, K)")
    if mode == "shuffle" and permutation is None:
        raise ValueError("shuffle mode requires a fixed permutation")
    if mode == "feature_permute" and feature_permutation is None:
        raise ValueError("feature_permute mode requires a fixed permutation")

    z_prev = long_memory.new_zeros(batch_size, branch.z_dim)
    states = []
    all_candidates = []
    for step in range(time_steps):
        if mode == "zero":
            generator_prev = torch.zeros_like(z_prev)
        elif mode == "sign_flip":
            generator_prev = -z_prev
        elif mode == "scaled":
            generator_prev = scale * z_prev
        elif mode == "shuffle":
            generator_prev = z_prev.index_select(0, permutation)
        elif mode == "feature_permute":
            generator_prev = z_prev.index_select(-1, feature_permutation)
        elif mode == "lagged":
            if step == 0:
                generator_prev = torch.zeros_like(z_prev)
            elif step == 1:
                generator_prev = states[0]
            else:
                generator_prev = states[step - 2]
        else:
            generator_prev = z_prev

        h_input, z_input = branch.transition_inputs(
            long_memory[:, step], generator_prev, step
        )
        transition_input = torch.cat([h_input, z_input], dim=-1)
        candidates = torch.stack(
            [generator(transition_input) for generator in branch.latent_transition.generators],
            dim=1,
        )
        if forced_state is None:
            mixing_probability = probabilities[:, step]
        else:
            mixing_probability = long_memory.new_zeros(batch_size, branch.K)
            mixing_probability[:, forced_state] = 1.0
        z_t = torch.einsum("bk,bkd->bd", mixing_probability, candidates)
        all_candidates.append(candidates)
        states.append(z_t)
        z_prev = z_t
    return torch.stack(states, dim=1), torch.stack(all_candidates, dim=1)


def _complete_replay(model, h_spatial, long_memory, probabilities,
                     name: str, **replay_kwargs) -> ReplayResult:
    branch = model.switching_latent_transformer
    z, candidates = replay_latent_trajectory(
        branch, long_memory, probabilities, **replay_kwargs
    )
    h_temporal = branch.readout(long_memory[:, -1], z[:, -1])
    h_micro = branch.last_h_micro.clone()
    prediction = model._market_token_predict(h_spatial, h_temporal)
    return ReplayResult(
        name=name,
        long_memory=long_memory,
        probabilities=probabilities,
        z=z,
        candidates=candidates,
        h_micro=h_micro,
        h_temporal=h_temporal,
        prediction=prediction,
    )


def _local_generator_diagnostics(branch, H, normal_z, eps=1e-8):
    z_previous = torch.cat(
        [torch.zeros_like(normal_z[:, :1]), normal_z[:, :-1]], dim=1
    )
    transition_h = []
    transition_z = []
    for step in range(H.shape[1]):
        h_input, z_input = branch.transition_inputs(
            H[:, step], z_previous[:, step], step
        )
        transition_h.append(h_input)
        transition_z.append(z_input)
    flat_h = torch.stack(transition_h, dim=1).reshape(-1, H.shape[-1])
    flat_z = torch.stack(transition_z, dim=1).reshape(
        -1, z_previous.shape[-1]
    )
    metrics = []
    for state, generator in enumerate(branch.latent_transition.generators):
        normal_input = torch.cat([flat_h, flat_z], dim=-1)
        zero_input = torch.cat([flat_h, torch.zeros_like(flat_z)], dim=-1)
        normal_candidate = generator(normal_input)
        zero_candidate = generator(zero_input)
        difference = normal_candidate - zero_candidate
        relative = difference.norm(dim=-1) / (normal_candidate.norm(dim=-1) + eps)

        fc1 = generator[0]
        h_dim = H.shape[-1]
        w_h = fc1.weight[:, :h_dim]
        w_z = fc1.weight[:, h_dim:]
        contribution_h = torch.nn.functional.linear(flat_h, w_h)
        contribution_z = torch.nn.functional.linear(flat_z, w_z)
        h_norm = contribution_h.norm(dim=-1).mean()
        z_norm = contribution_z.norm(dim=-1).mean()
        metrics.append({
            "state": state,
            "mean_diff": difference.abs().mean().item(),
            "max_diff": difference.abs().max().item(),
            "relative_norm": relative.mean().item(),
            "whh_norm": h_norm.item(),
            "wzz_norm": z_norm.item(),
            "effective_ratio": (z_norm / (h_norm + eps)).item(),
            "w_h_norm": w_h.norm().item(),
            "w_z_norm": w_z.norm().item(),
            "weight_ratio": (w_z.norm() / (w_h.norm() + eps)).item(),
            "bias_norm": fc1.bias.norm().item() if fc1.bias is not None else 0.0,
        })
    return metrics


def _local_input_gradients(branch, H, normal_z, selected_steps, eps=1e-8):
    """Generator-local gradients of candidate energy, isolated from other H paths."""
    values = []
    for step in selected_steps:
        h_t = H[:, step].detach().clone().requires_grad_(True)
        if step == 0:
            z_prev = torch.zeros_like(normal_z[:, 0]).requires_grad_(True)
        else:
            z_prev = normal_z[:, step - 1].detach().clone().requires_grad_(True)
        h_input, z_input = branch.transition_inputs(h_t, z_prev, step)
        transition_input = torch.cat([h_input, z_input], dim=-1)
        candidates = torch.stack(
            [generator(transition_input) for generator in branch.latent_transition.generators],
            dim=1,
        )
        # A local scalar avoids contamination from H's regime/readout paths.
        local_energy = candidates.square().mean()
        grad_h_input, grad_z_input = torch.autograd.grad(
            local_energy, (h_input, z_input), retain_graph=True
        )
        grad_h, grad_z = torch.autograd.grad(local_energy, (h_t, z_prev))
        h_norm = grad_h_input.norm(dim=-1).mean()
        z_norm = grad_z_input.norm(dim=-1).mean()
        raw_h_norm = grad_h.norm(dim=-1).mean()
        raw_z_norm = grad_z.norm(dim=-1).mean()
        values.append({
            "time": step,
            "grad_h": h_norm.item(),
            "grad_zprev": z_norm.item(),
            "ratio": (z_norm / (h_norm + eps)).item(),
            "raw_grad_h": raw_h_norm.item(),
            "raw_grad_zprev": raw_z_norm.item(),
            "raw_ratio": (raw_z_norm / (raw_h_norm + eps)).item(),
        })
    return values


def run_previous_state_diagnostics(model, x_batch: torch.Tensor,
                                   seed: int = RANDOM_SEED):
    """Run all D0B previous-state interventions on one immutable test batch."""
    supported = {
        "switching_latent_balanced_readout": "D0B",
        "switching_latent_balanced_transition": "D0D",
    }
    if model.variant not in supported:
        raise ValueError("diagnostics require the exact D0B or D0D variant")
    label = supported[model.variant]
    branch = model.switching_latent_transformer
    if not branch.balanced_readout or branch.use_latent_memory or branch.use_dynamic_slope:
        raise ValueError("diagnostics require plain D0B/D0D, not another D-series variant")
    model.eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with torch.no_grad():
        normal_prediction = model(x_batch)
        H = branch.last_long_memory.clone()
        p = branch.last_regime_probabilities.clone()
        normal_z_forward = branch.last_latent_states.clone()
        normal_candidates_forward = branch.last_latent_candidates.clone()
        h_spatial = model._temp_weighted_spatial(x_batch)

        normal = _complete_replay(
            model, h_spatial, H, p, "Normal", mode="normal"
        )
        torch.testing.assert_close(normal.z, normal_z_forward, rtol=0.0, atol=1e-7)
        torch.testing.assert_close(
            normal.candidates, normal_candidates_forward, rtol=0.0, atol=1e-7
        )
        torch.testing.assert_close(
            normal.prediction, normal_prediction, rtol=0.0, atol=1e-7
        )

        permutation = _fixed_derangement(x_batch.shape[0], seed, x_batch.device)
        feature_permutation = torch.roll(
            torch.arange(branch.z_dim, device=x_batch.device), shifts=1
        )
        interventions = {
            "zero_previous": _complete_replay(
                model, h_spatial, H, p, "Zero previous state", mode="zero"
            ),
            "sign_flip": _complete_replay(
                model, h_spatial, H, p, "Sign-flip previous state", mode="sign_flip"
            ),
            "scale_0.5": _complete_replay(
                model, h_spatial, H, p, "0.5x previous state", mode="scaled", scale=0.5
            ),
            "scale_1.5": _complete_replay(
                model, h_spatial, H, p, "1.5x previous state", mode="scaled", scale=1.5
            ),
            "shuffle": _complete_replay(
                model, h_spatial, H, p, "Batch-shuffled previous state",
                mode="shuffle", permutation=permutation,
            ),
            "lagged": _complete_replay(
                model, h_spatial, H, p, "Lagged previous state", mode="lagged"
            ),
            "feature_permute": _complete_replay(
                model, h_spatial, H, p, "Feature-permuted previous state",
                mode="feature_permute", feature_permutation=feature_permutation,
            ),
        }

        # Reference that removes the entire post-normalization micro readout.
        zero_micro_temporal = branch.readout(H[:, -1], normal.z[:, -1], zero_component="Z")
        zero_micro_prediction = model._market_token_predict(h_spatial, zero_micro_temporal)

        forced = {}
        for state in range(branch.K):
            forced_normal = _complete_replay(
                model, h_spatial, H, p, f"forced state {state} normal previous",
                mode="normal", forced_state=state,
            )
            forced_zero = _complete_replay(
                model, h_spatial, H, p, f"forced state {state} zero previous",
                mode="zero", forced_state=state,
            )
            forced[state] = {
                "z": _mean_max(forced_normal.z[:, -1], forced_zero.z[:, -1]),
                "prediction": _mean_max(
                    forced_normal.prediction, forced_zero.prediction
                ),
            }

        local = _local_generator_diagnostics(branch, H, normal.z)

    time_points = sorted(set(
        min(step, H.shape[1] - 1) for step in (0, 2, 5, 10, 15, 19)
    ))
    gradient_points = sorted(set(
        min(step, H.shape[1] - 1) for step in (1, 5, 10, 15, 19)
    ))
    local_gradients = _local_input_gradients(
        branch, H, normal.z, gradient_points
    )

    zero_previous = interventions["zero_previous"]
    zero_micro_impact = _mean_max(normal.prediction, zero_micro_prediction)
    zero_previous_impact = _mean_max(normal.prediction, zero_previous.prediction)
    shuffle_impact = _mean_max(normal.prediction, interventions["shuffle"].prediction)
    eps = branch.regime_filter.eps
    recurrence_fraction = zero_previous_impact[0] / (zero_micro_impact[0] + eps)
    identity_fraction = shuffle_impact[0] / (zero_micro_impact[0] + eps)

    comparison = {}
    for key, result in interventions.items():
        comparison[key] = {
            "z_last": _mean_max(normal.z[:, -1], result.z[:, -1]),
            "h_micro": _mean_max(normal.h_micro, result.h_micro),
            "h_temporal": _mean_max(normal.h_temporal, result.h_temporal),
            "prediction": _mean_max(normal.prediction, result.prediction),
            "persistence": _persistence(result.z),
        }
    comparison["normal"] = {
        "z_last": (0.0, 0.0), "h_micro": (0.0, 0.0),
        "h_temporal": (0.0, 0.0), "prediction": (0.0, 0.0),
        "persistence": _persistence(normal.z),
    }
    comparison["zero_micro"] = {
        "z_last": (0.0, 0.0),
        "h_micro": _mean_max(normal.h_micro, torch.zeros_like(normal.h_micro)),
        "h_temporal": _mean_max(normal.h_temporal, zero_micro_temporal),
        "prediction": zero_micro_impact,
        "persistence": _persistence(normal.z),
    }

    result = {
        "normal": normal,
        "interventions": interventions,
        "comparison": comparison,
        "H_sanity_max_diff": _mean_max(
            normal.long_memory, zero_previous.long_memory
        )[1],
        "p_sanity_max_diff": _mean_max(
            normal.probabilities, zero_previous.probabilities
        )[1],
        "zero_previous_candidate_diffs": [
            _mean_max(normal.candidates[:, :, state], zero_previous.candidates[:, :, state])
            for state in range(branch.K)
        ],
        "zero_previous_trajectory_mean_diff": _mean_max(normal.z, zero_previous.z)[0],
        "zero_previous_time_diffs": {
            step: _mean_max(normal.z[:, step], zero_previous.z[:, step])[0]
            for step in time_points
        },
        "permutation": permutation.detach().cpu(),
        "shuffle_unchanged_fraction": (
            (permutation == torch.arange(x_batch.shape[0], device=x_batch.device))
            .float().mean().item()
        ),
        "local": local,
        "local_gradients": local_gradients,
        "forced": forced,
        "zero_micro_prediction": zero_micro_prediction,
        "zero_micro_impact": zero_micro_impact,
        "recurrence_fraction": recurrence_fraction,
        "identity_fraction": identity_fraction,
        "label": label,
        "raw_z_h_ratio": (
            normal.z.norm(dim=-1).mean()
            / (H.norm(dim=-1).mean() + eps)
        ).item(),
        "transition_input_stats": {
            "raw_h": H[:, 1:].norm(dim=-1).mean().item(),
            "raw_z": normal.z[:, :-1].norm(dim=-1).mean().item(),
            "balanced_h": branch.last_transition_h_inputs[:, 1:].norm(
                dim=-1
            ).mean().item(),
            "balanced_z": branch.last_transition_z_inputs[:, 1:].norm(
                dim=-1
            ).mean().item(),
        },
    }

    after = model.state_dict()
    for name, value in before.items():
        torch.testing.assert_close(value, after[name], rtol=0.0, atol=0.0)
    _print_report(result, H, p, x_batch.shape)
    return result


def _print_report(result, H, p, batch_shape):
    normal = result["normal"]
    zero = result["interventions"]["zero_previous"]
    label = result["label"]
    print(f"\n[{label} previous-state diagnostics: inference only]")
    print(f"  fixed TEST batch shape={tuple(batch_shape)}")
    print(
        f"  [normal reference] H={tuple(H.shape)} p={tuple(p.shape)} "
        f"Z={tuple(normal.z.shape)} prediction={tuple(normal.prediction.shape)}"
    )
    print(
        f"  [{label} zero-Zprev sanity] max H diff="
        f"{result['H_sanity_max_diff']:.3e} max p diff={result['p_sanity_max_diff']:.3e}"
    )
    for state, (mean, maximum) in enumerate(result["zero_previous_candidate_diffs"]):
        print(
            f"  [{label} zero-previous-state] candidate state{state} "
            f"mean/max diff={mean:.6e}/{maximum:.6e}"
        )
    print(
        f"  [{label} zero-previous-state] Z trajectory mean diff="
        f"{result['zero_previous_trajectory_mean_diff']:.6e}"
    )
    for step, value in result["zero_previous_time_diffs"].items():
        print(f"  [{label} zero-previous-state t={step}] mean |delta Z_t|={value:.6e}")

    print("\n  Intervention                         Z_T mean/max       h_micro mean/max   h_temporal mean/max prediction mean/max")
    for key in (
        "normal", "zero_previous", "sign_flip", "scale_0.5", "scale_1.5",
        "shuffle", "feature_permute", "lagged", "zero_micro",
    ):
        row = result["comparison"][key]
        print(
            f"  {key:<35} "
            f"{row['z_last'][0]:.3e}/{row['z_last'][1]:.3e}  "
            f"{row['h_micro'][0]:.3e}/{row['h_micro'][1]:.3e}  "
            f"{row['h_temporal'][0]:.3e}/{row['h_temporal'][1]:.3e}  "
            f"{row['prediction'][0]:.3e}/{row['prediction'][1]:.3e}"
        )
    print(
        f"  [{label} shuffled-previous-state] unchanged batch positions="
        f"{result['shuffle_unchanged_fraction']:.2%} permutation="
        f"{result['permutation'].tolist()}"
    )

    stats = result["transition_input_stats"]
    print(
        f"\n  [{label} transition input] raw H/Z={stats['raw_h']:.6e}/"
        f"{stats['raw_z']:.6e} ratio={stats['raw_z']/(stats['raw_h']+1e-8):.6e}; "
        f"balanced H/Z={stats['balanced_h']:.6e}/{stats['balanced_z']:.6e} "
        f"ratio={stats['balanced_z']/(stats['balanced_h']+1e-8):.6e}"
    )
    if label == "D0D":
        print(
            "  [D0B effective Z/H reference] state0=1.002e-02 "
            "state1=1.377e-02 state2=9.511e-03"
        )
        print(
            "  [D0B local relative sensitivity reference] state0=4.66e-03 "
            "state1=9.44e-03 state2=5.25e-03; "
            "generator-local Zprev/H grad≈8.5e-02"
        )
    print(f"\n  [{label} local previous-state sensitivity]")
    for item in result["local"]:
        print(
            f"    state {item['state']}: candidate mean/max diff="
            f"{item['mean_diff']:.3e}/{item['max_diff']:.3e} "
            f"relative={item['relative_norm']:.3e}; "
            f"||W_H H||={item['whh_norm']:.3e} ||W_Z Z||={item['wzz_norm']:.3e} "
            f"effective Z/H={item['effective_ratio']:.3e}; "
            f"||W_H||={item['w_h_norm']:.3e} ||W_Z||={item['w_z_norm']:.3e} "
            f"weight Z/H={item['weight_ratio']:.3e} bias={item['bias_norm']:.3e}"
        )
    print(f"  [{label} generator-local candidate-energy gradients]")
    for item in result["local_gradients"]:
        print(
            f"    t={item['time']}: grad H={item['grad_h']:.3e} "
            f"grad Zprev={item['grad_zprev']:.3e} Zprev/H={item['ratio']:.3e}; "
            f"raw grad H/Zprev={item['raw_grad_h']:.3e}/"
            f"{item['raw_grad_zprev']:.3e} raw ratio={item['raw_ratio']:.3e}"
        )
    for state, values in result["forced"].items():
        print(
            f"  [{label} recurrence interaction state {state}] "
            f"Z_T mean/max={values['z'][0]:.3e}/{values['z'][1]:.3e} "
            f"prediction mean/max={values['prediction'][0]:.3e}/"
            f"{values['prediction'][1]:.3e}"
        )

    print(f"\n  [{label} Z persistence]")
    for key in ("normal", "zero_previous", "sign_flip", "shuffle"):
        values = result["comparison"][key]["persistence"]
        print(
            f"    {key}: mean ||Z||={values['mean_norm']:.6e} "
            f"mean ||delta Z||={values['mean_change']:.6e} "
            f"consecutive cosine={values['consecutive_cosine']:.6f}"
        )
    print(f"\n  [{label} recurrence utilization]")
    print(
        "    I_zeroPrev="
        f"{result['comparison']['zero_previous']['prediction'][0]:.6e} "
        f"I_shufflePrev={result['comparison']['shuffle']['prediction'][0]:.6e} "
        f"I_zeroMicro={result['zero_micro_impact'][0]:.6e}"
    )
    print(
        f"    recurrence/micro={result['recurrence_fraction']:.6f} "
        f"identity/micro={result['identity_fraction']:.6f}"
    )
    if label == "D0D":
        print(
            "    [D0B recurrence reference] I_zeroPrev=1.113513e-05 "
            "I_shufflePrev=1.883487e-06 I_zeroMicro=8.547919e-03 "
            "recurrence/micro=1.303e-03 identity/micro=2.20e-04"
        )
        print(
            "    [D0B forced recurrence reference] state0=1.193e-05 "
            "state1=4.532e-05 state2=1.379e-05; lagged=8.009e-07"
        )
    print(f"    raw mean ||Z||/||H||={result['raw_z_h_ratio']:.6e}")
    local_effective = sum(
        item["effective_ratio"] for item in result["local"]
    ) / len(result["local"])
    forced_impacts = [
        values["prediction"][0] for values in result["forced"].values()
    ]
    forced_mean = sum(forced_impacts) / len(forced_impacts)
    forced_spread = (
        (max(forced_impacts) - min(forced_impacts)) / (forced_mean + 1e-12)
    )
    conclusions = []
    if result["recurrence_fraction"] >= 0.05 and result["identity_fraction"] >= 0.05:
        conclusions.append("Case A supported: recurrence and sample identity have measurable prediction effects")
    if result["recurrence_fraction"] < 0.05 and result["identity_fraction"] < 0.05:
        conclusions.append("Case B supported: the final Z representation matters more than its recurrence")
    if result["raw_z_h_ratio"] < 0.05 and local_effective >= 0.05:
        conclusions.append("Case C supported: learned generator weights partly compensate the raw Z/H scale gap")
    if result["recurrence_fraction"] >= 0.05 and result["identity_fraction"] < 0.02:
        conclusions.append("Case D supported: previous-state distribution matters more than sample identity")
    if forced_spread >= 0.25:
        conclusions.append("Case E supported: forced regimes show materially different recurrence effects")
    if not conclusions:
        conclusions.append("No Case A-E heuristic is decisive; inspect the reported continuous effect sizes")
    print("  [mechanism conclusion; descriptive 5%/25% heuristics]")
    for conclusion in conclusions:
        print(f"    {conclusion}")


def _load_checkpoint(model, checkpoint_path: Path, device: torch.device):
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch.
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a state_dict or training checkpoint dict")
    metadata = payload.get("metadata", {})
    stored_variant = metadata.get("variant") if isinstance(metadata, dict) else None
    if stored_variant is not None and stored_variant != model.variant:
        raise ValueError(
            f"checkpoint variant is {stored_variant!r}, but requested model is "
            f"{model.variant!r}"
        )
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    model.load_state_dict(state, strict=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "switching_latent_balanced_readout",
            "switching_latent_balanced_transition",
        ),
        default="switching_latent_balanced_readout",
    )
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_index < 0:
        raise ValueError("--batch-index must be non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    from cmgm.scripts.main_ablation import build_data

    data_args = SimpleNamespace(
        batch_size=args.batch_size, seq_len=args.seq_len, seed=args.seed
    )
    data = build_data(data_args)
    n_stock = data["market_indices"]["stock"][1] - data["market_indices"]["stock"][0]
    n_bond = data["market_indices"]["bond"][1] - data["market_indices"]["bond"][0]
    model = HeteroMixHopCMGM(
        data["n_nodes"], data["n_commodities"],
        n_stock=n_stock, n_bond=n_bond, feat_dim=FEATURE_DIM,
        variant=args.variant,
    ).to(device)
    payload = _load_checkpoint(model, args.checkpoint, device)
    best_epoch = payload.get("best_epoch") if isinstance(payload, dict) else None
    print(
        f"[Checkpoint] {args.variant} best={args.checkpoint.resolve()} "
        f"best_epoch={best_epoch if best_epoch is not None else 'not recorded'}"
    )
    selected = None
    for index, batch in enumerate(data["loaders"]["test"]):
        if index == args.batch_index:
            selected = batch
            break
    if selected is None:
        raise IndexError(f"test batch index {args.batch_index} is out of range")
    x_batch = selected[0].to(device)
    print(
        f"[Fixed TEST batch] index={args.batch_index} size={x_batch.shape[0]} "
        f"seed={args.seed} shuffle=False"
    )
    run_previous_state_diagnostics(model, x_batch, seed=args.seed)


if __name__ == "__main__":
    main()
