"""Run retained CMGM ablations plus the independent S0 temporal baseline.

Usage:
    python -m cmgm.scripts.main_ablation
    python -m cmgm.scripts.main_ablation --variants +TempWeighted,F-RegimeDynamic
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cmgm.config import (
    BATCH_SIZE,
    FEATURE_DIM,
    FEAT_ZSCORE_EPS,
    MULTI_HORIZONS,
    NUM_EPOCHS,
    PATIENCE,
    RANDOM_SEED,
    SEQ_LEN,
    TARGET_HORIZON,
    TARGET_TYPE,
)
from cmgm.data.data_loader import (
    MarketSequenceDataset,
    build_market_descriptor_timeline,
    create_data_loaders,
    set_seed,
)
from cmgm.data.feature_builder import build_feature_matrix
from cmgm.experiment_logger import ExperimentLogger
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.evaluate import compute_metrics, inverse_transform_predictions
from cmgm.training.train import make_loss, train


VARIANTS = [
    ("+TempWeighted", "temporal_weighted_graph"),
    ("B-Transformer", "transformer_temporal"),
    ("C-TransformerRPE", "transformer_rpe"),
    ("S0-MarketTokenTransformer", "market_token_transformer"),
    ("S0D-MarketDispersionTransformer", "market_dispersion_transformer"),
    ("S1-SwitchingTransformer", "switching_transformer"),
    ("S1C-NullSwitchControl", "switching_null_control"),
    ("S2F-SwitchingFilterRPE", "switching_filter_rpe"),
    ("D0-SwitchingLatentTransformer", "switching_latent_transformer"),
    ("D0B-BalancedLatentReadout", "switching_latent_balanced_readout"),
    ("D0C-DynamicSlopeTransition", "switching_latent_dynamic_slope"),
    ("D1A-LatentMemory", "switching_latent_memory"),
    ("D1A2-ActiveLatentMemory", "switching_active_latent_memory"),
    ("D1-RegimeRelativeLatentMemory", "switching_regime_relative_latent_memory"),
    ("F-RegimeDynamic", "regime_dynamic_transformer"),
    ("F2-RegimeSemantic", "regime_dynamic_semantic"),
    ("G-SemanticRouter", "semantic_router"),
    ("H-LossRebalance", "loss_rebalance"),
    ("I-RoutingStrength", "routing_strength"),
    ("J-ContextCalibrated", "context_calibrated"),
    ("K-ContextBalance", "context_balance"),
    ("L-AdapterOrthogonal", "adapter_orthogonal"),
]

D_SERIES_VARIANTS = frozenset({
    "switching_latent_transformer",
    "switching_latent_balanced_readout",
    "switching_latent_dynamic_slope",
    "switching_latent_memory",
    "switching_active_latent_memory",
    "switching_regime_relative_latent_memory",
})


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "HeteroMixHop A/B/C/S0/S0D/S1/S1C/S2F/D0/D0B/D0C/D1A/D1A2/D1 "
            "and retained F/F2/G/H/I/J/K/L ablations"
        )
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help=(
            "Directory for best checkpoints of D0/D0B/D0C/D1A/D1A2/D1 "
            "(default: ./checkpoints). Other variants are not saved."
        ),
    )
    parser.add_argument(
        "--variants",
        help=(
            "Comma-separated display or internal names; defaults to "
            "A/B/C/S0/S0D/S1/S1C/S2F/D0/D0B/D0C/D1A/D1A2/D1/F/F2/G/H/I/J/K/L"
        ),
    )
    return parser.parse_args()


def _checkpoint_path_for_variant(variant, checkpoint_dir):
    """Create and return this D-series variant's deterministic best path."""
    if variant not in D_SERIES_VARIANTS:
        return None
    directory = Path(checkpoint_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{variant}_best.pt"


def build_data(args):
    """Build the shared 21-feature train/validation/test pipeline once."""
    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    raw_full = np.concatenate(
        [data["raw_prices_train"], data["raw_prices_val"], data["raw_prices_test"]],
        axis=0,
    )
    feat_raw, _ = build_feature_matrix(raw_full)
    train_size = data["raw_prices_train"].shape[0]
    feat_mean = feat_raw[:train_size].mean(axis=0, keepdims=True)
    feat_std = np.maximum(
        feat_raw[:train_size].std(axis=0, keepdims=True), FEAT_ZSCORE_EPS
    )
    features = (feat_raw - feat_mean) / feat_std

    norm_mean = data["norm_stats"]["mean"]
    norm_std = data["norm_stats"]["std"]
    normalized = (raw_full - norm_mean) / norm_std
    train_end = int(len(raw_full) * 0.7)
    val_end = train_end + int(len(raw_full) * 0.15)
    feature_splits = [features[:train_end], features[train_end:val_end], features[val_end:]]
    norm_splits = [normalized[:train_end], normalized[train_end:val_end], normalized[val_end:]]
    raw_splits = [
        data["raw_prices_train"],
        data["raw_prices_val"],
        data["raw_prices_test"],
    ]
    descriptors, descriptors_raw, descriptor_stats = build_market_descriptor_timeline(
        raw_full,
        data["market_indices"],
        train_end=train_size,
        lookback=5,
    )
    descriptor_splits = [
        descriptors[:train_end],
        descriptors[train_end:val_end],
        descriptors[val_end:],
    ]
    horizons = MULTI_HORIZONS if TARGET_TYPE == "return" else [TARGET_HORIZON]
    datasets = {
        split: MarketSequenceDataset(
            norm,
            data["market_indices"],
            args.seq_len,
            feature_matrix=feature,
            raw_prices=raw,
            target_type=TARGET_TYPE,
            horizons=horizons,
        )
        for split, norm, feature, raw in zip(
            ("train", "val", "test"),
            norm_splits,
            feature_splits,
            raw_splits,
        )
    }
    semantic_datasets = {
        split: MarketSequenceDataset(
            norm,
            data["market_indices"],
            args.seq_len,
            feature_matrix=feature,
            raw_prices=raw,
            target_type=TARGET_TYPE,
            horizons=horizons,
            market_descriptors=descriptor,
        )
        for split, norm, feature, raw, descriptor in zip(
            ("train", "val", "test"),
            norm_splits,
            feature_splits,
            raw_splits,
            descriptor_splits,
        )
    }
    data["loaders"] = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=split == "train",
        )
        for split, dataset in datasets.items()
    }
    data["semantic_loaders"] = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=split == "train",
        )
        for split, dataset in semantic_datasets.items()
    }
    data["descriptor_timeline"] = descriptors
    data["descriptor_timeline_raw"] = descriptors_raw
    data["descriptor_stats"] = descriptor_stats
    return data


def evaluate_primary_horizon(model, loader, data, device):
    model.eval()
    predictions, targets = [], []
    horizon_index = MULTI_HORIZONS.index(TARGET_HORIZON)
    with torch.no_grad():
        for batch in loader:
            x_batch, y_batch = batch[:2]
            descriptor = batch[2].to(device) if len(batch) == 3 else None
            pred = model(
                x_batch.to(device), market_descriptor=descriptor
            ).cpu().numpy()
            target = y_batch.numpy()
            if pred.ndim == 3:
                pred = pred[:, horizon_index, :]
            if target.ndim == 3:
                target = target[:, horizon_index, :]
            predictions.append(pred)
            targets.append(target)

    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    normalized_metrics = compute_metrics(pred, target)
    pred_original, target_original = inverse_transform_predictions(
        pred,
        target,
        data["norm_stats"],
        data["raw_prices_test"],
        data["market_indices"],
        target_type=TARGET_TYPE,
    )
    return normalized_metrics, compute_metrics(pred_original, target_original), target


def _prediction_loss(prediction, target, criterion):
    if prediction.dim() == 3:
        return sum(
            criterion(prediction[:, horizon], target[:, horizon])
            for horizon in range(prediction.size(1))
        )
    return criterion(prediction, target)


def _event_test(label, name, values, probabilities):
    count = max(1, int(np.ceil(len(values) * 0.1)))
    order = np.argsort(values)
    low = probabilities[order[:count]].mean(axis=0)
    high = probabilities[order[-count:]].mean(axis=0)
    print(f"  [{label} event/{name}] high p={np.round(high, 4)}")
    print(f"  [{label} event/{name}] low  p={np.round(low, 4)}")
    print(f"  [{label} event/{name}] L1 difference={np.abs(high - low).sum():.6f}")


def _market_token_diagnostics(model, loader, device):
    """Inference-only diagnostics for the independent S0/S0D temporal branches."""
    branch = model.market_token_transformer
    encoder = branch.market_encoder
    transformer = branch.transformer
    route_label = "S0D" if encoder.use_dispersion else "S0"
    market_attentions = {name: [] for name in encoder.MARKET_NAMES}
    market_entropies = {name: [] for name in encoder.MARKET_NAMES}
    node_norms = {name: [] for name in encoder.MARKET_NAMES}
    market_norms = {name: [] for name in encoder.MARKET_NAMES}
    dispersion_values = {name: [] for name in encoder.MARKET_NAMES}
    temporal_attentions = []
    market_concat_norms = []
    daily_norms = []
    transformer_norms = []
    temporal_output_norms = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_batch = batch[0].to(device)
            branch(x_batch)
            for name in encoder.MARKET_NAMES:
                attention = encoder.last_market_attentions[name]
                entropy = -(attention * (attention + 1e-12).log()).sum(dim=-1)
                market_attentions[name].append(attention.cpu().reshape(-1))
                market_entropies[name].append(entropy.cpu().reshape(-1))
                node_norms[name].append(
                    encoder.last_node_encodings[name].norm(dim=-1).cpu().reshape(-1)
                )
                market_norms[name].append(
                    encoder.last_market_tokens[name].norm(dim=-1).cpu().reshape(-1)
                )
                if encoder.use_dispersion:
                    dispersion_values[name].append(
                        encoder.last_market_dispersions[name].cpu().reshape(
                            -1, encoder.node_dim
                        )
                    )
            time_attention = transformer.last_temporal_attention
            temporal_attentions.append(time_attention.cpu().reshape(-1))
            market_concat_norms.append(
                encoder.last_market_concat.norm(dim=-1).cpu().reshape(-1)
            )
            daily_norms.append(
                encoder.last_daily_tokens.norm(dim=-1).cpu().reshape(-1)
            )
            transformer_norms.append(
                transformer.last_transformer_output.norm(dim=-1).cpu().reshape(-1)
            )
            temporal_output_norms.append(
                transformer.last_temporal_output.norm(dim=-1).cpu().reshape(-1)
            )

    for name in encoder.MARKET_NAMES:
        attention = torch.cat(market_attentions[name])
        entropy = torch.cat(market_entropies[name])
        n_nodes = encoder.market_sizes[name]
        normalized_entropy = (
            entropy / np.log(n_nodes) if n_nodes > 1 else torch.ones_like(entropy)
        )
        # Recover per-(sample,time) max weights without retaining all batches in 3-D.
        attention_rows = attention.view(-1, n_nodes)
        print(
            f"  [{route_label} {name} attention] mean={attention.mean().item():.6f} "
            f"std={attention.std(unbiased=False).item():.6f} "
            f"min={attention.min().item():.6f} max={attention.max().item():.6f}"
        )
        print(
            f"  [{route_label} {name} attention] entropy={entropy.mean().item():.6f} "
            f"normalized_entropy={normalized_entropy.mean().item():.6f} "
            f"mean_max={attention_rows.max(dim=-1).values.mean().item():.6f} "
            f"effective_nodes={entropy.exp().mean().item():.6f}"
        )

    time_attention = torch.cat(temporal_attentions)
    time_steps = next(iter(loader))[0].shape[1]
    time_rows = time_attention.view(-1, time_steps)
    time_entropy = -(time_rows * (time_rows + 1e-12).log()).sum(dim=-1)
    print(
        f"  [{route_label} temporal pooling] mean={time_attention.mean().item():.6f} "
        f"std={time_attention.std(unbiased=False).item():.6f} "
        f"min={time_attention.min().item():.6f} max={time_attention.max().item():.6f} "
        f"entropy={time_entropy.mean().item():.6f} "
        f"mean_max={time_rows.max(dim=-1).values.mean().item():.6f}"
    )
    for name in encoder.MARKET_NAMES:
        print(
            f"  [{route_label} norm] {name} node encoding="
            f"{torch.cat(node_norms[name]).mean().item():.6f}; "
            f"g_{name}={torch.cat(market_norms[name]).mean().item():.6f}"
        )
        if encoder.use_dispersion:
            dispersion = torch.cat(dispersion_values[name], dim=0)
            dispersion_norm = dispersion.norm(dim=-1)
            level_norm = torch.cat(market_norms[name]).mean()
            dimension_std = dispersion.std(dim=0, unbiased=False)
            print(
                f"  [S0D {name} dispersion] mean={dispersion.mean().item():.6f} "
                f"std={dispersion.std(unbiased=False).item():.6f} "
                f"min={dispersion.min().item():.6f} "
                f"max={dispersion.max().item():.6f} "
                f"L2_norm={dispersion_norm.mean().item():.6f}"
            )
            print(
                f"  [S0D {name} dispersion activity] mean_dim_std="
                f"{dimension_std.mean().item():.6f} "
                f"min_dim_std={dimension_std.min().item():.6f} "
                f"max_dim_std={dimension_std.max().item():.6f}"
            )
            print(
                f"  [S0D {name} level/dispersion] level_norm={level_norm.item():.6f} "
                f"dispersion_norm={dispersion_norm.mean().item():.6f} "
                f"ratio={(dispersion_norm.mean() / (level_norm + 1e-12)).item():.6f}"
            )
    print(
        f"  [{route_label} norm] market concat="
        f"{torch.cat(market_concat_norms).mean().item():.6f}; "
        f"daily token E={torch.cat(daily_norms).mean().item():.6f}; "
        f"Transformer output={torch.cat(transformer_norms).mean().item():.6f}; "
        f"h_temporal={torch.cat(temporal_output_norms).mean().item():.6f}"
    )

    x_batch = next(iter(loader))[0][:16].to(device)
    with torch.no_grad():
        h_spatial = model._temp_weighted_spatial(x_batch)
        tokens_reference = branch.encode_market_tokens(x_batch)
        h_reference = branch.temporal_forward(tokens_reference)
        pred_reference = model._market_token_predict(h_spatial, h_reference)
        if encoder.use_dispersion:
            components = [
                f"{name}_{component}"
                for name in encoder.MARKET_NAMES
                for component in ("level", "dispersion")
            ] + ["all_dispersion"]
            for component in components:
                h_zero = branch(x_batch, zero_component=component)
                pred_zero = model._market_token_predict(h_spatial, h_zero)
                pred_diff = (pred_zero - pred_reference).abs()
                print(
                    f"  [S0D zero-{component.replace('_', '-')}] prediction "
                    f"mean_abs_diff={pred_diff.mean().item():.3e} "
                    f"max_abs_diff={pred_diff.max().item():.3e}"
                )
        else:
            for name in encoder.MARKET_NAMES:
                h_zero = branch(x_batch, zero_market=name)
                pred_zero = model._market_token_predict(h_spatial, h_zero)
                pred_diff = (pred_zero - pred_reference).abs()
                print(
                    f"  [S0 zero-{name}] prediction max_abs_diff="
                    f"{pred_diff.max().item():.3e} "
                    f"mean_abs_diff={pred_diff.mean().item():.3e}"
                )

        offsets = {
            "stock": (0, encoder.market_sizes["stock"]),
            "bond": (
                encoder.market_sizes["stock"],
                encoder.market_sizes["stock"] + encoder.market_sizes["bond"],
            ),
            "commodity": (
                encoder.market_sizes["stock"] + encoder.market_sizes["bond"],
                sum(encoder.market_sizes.values()),
            ),
        }
        for name, (start, end) in offsets.items():
            permuted = x_batch.clone()
            permutation = torch.randperm(end - start, device=device)
            permuted[:, :, start:end, :] = x_batch[:, :, start:end, :].index_select(
                2, permutation
            )
            tokens_permuted = branch.encode_market_tokens(permuted)
            h_permuted = branch.temporal_forward(tokens_permuted)
            pred_permuted = model._market_token_predict(h_spatial, h_permuted)
            print(
                f"  [{route_label} {name} permutation] "
                f"daily token max_abs_diff="
                f"{(tokens_permuted - tokens_reference).abs().max().item():.3e}; "
                f"h_temporal max_abs_diff="
                f"{(h_permuted - h_reference).abs().max().item():.3e}; "
                f"prediction max_abs_diff="
                f"{(pred_permuted - pred_reference).abs().max().item():.3e}"
            )

        tokens_batch = branch.encode_market_tokens(x_batch)
        h_batch = branch.temporal_forward(tokens_batch)
        tokens_single = branch.encode_market_tokens(x_batch[:1])
        h_single = branch.temporal_forward(tokens_single)
        print(
            f"  [{route_label} batch independence] "
            f"token max_abs_diff="
            f"{(tokens_batch[:1] - tokens_single).abs().max().item():.3e}; "
            f"h_temporal max_abs_diff="
            f"{(h_batch[:1] - h_single).abs().max().item():.3e}"
        )


def _aggregate_gradient_norm(parameters, gradients=None):
    if gradients is None:
        gradients = [parameter.grad for parameter in parameters]
    squared = [
        gradient.square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not squared:
        return 0.0
    return torch.stack(squared).sum().sqrt().item()


def _switching_parameter_groups(branch):
    return {
        "Regime Evidence Transformer": list(branch.regime_evidence.parameters()),
        "posterior heads": list(branch.regime_inference.posterior_heads.parameters()),
        "transition_logits": [branch.regime_inference.transition_logits],
        "regime_embeddings": [branch.regime_embeddings],
        "MarketAwareTemporalEncoder": list(branch.market_encoder.parameters()),
        "Forecast Transformer": list(branch.transformer.parameters()),
    }


def _snapshot_switching_parameter_groups(branch):
    return {
        name: [parameter.detach().cpu().clone() for parameter in parameters]
        for name, parameters in _switching_parameter_groups(branch).items()
    }


def _switching_transformer_diagnostics(model, loaders, device):
    """Diagnostics for S1 and the strict S1C switching null control."""
    branch = model.switching_transformer
    inference = branch.regime_inference
    label = "S1C" if branch.null_control else "S1"
    eps = inference.eps
    split_probabilities = {}
    test_parts = {
        name: []
        for name in ("p", "prior", "previous", "q", "E", "r_real", "r_effective")
    }
    input_stats = {
        name: {"entropy": [], "level": [], "dispersion": []}
        for name in branch.market_encoder.MARKET_NAMES
    }

    model.eval()
    with torch.no_grad():
        for split_name, loader in loaders.items():
            probability_batches = []
            for batch in loader:
                x_batch = batch[0].to(device)
                branch(x_batch)
                p = branch.last_regime_probabilities.cpu()
                probability_batches.append(p)
                if split_name == "test":
                    test_parts["p"].append(p)
                    test_parts["prior"].append(inference.last_priors.cpu())
                    test_parts["previous"].append(
                        inference.last_previous_probabilities.cpu()
                    )
                    test_parts["q"].append(inference.last_posterior_heads.cpu())
                    test_parts["E"].append(branch.last_market_tokens.cpu())
                    test_parts["r_real"].append(
                        branch.last_regime_intervention_real.cpu()
                    )
                    test_parts["r_effective"].append(
                        branch.last_regime_intervention_effective.cpu()
                    )
                    encoder = branch.market_encoder
                    for name in encoder.MARKET_NAMES:
                        attention = encoder.last_market_attentions[name]
                        entropy = -(
                            attention * (attention + 1e-12).log()
                        ).sum(dim=-1)
                        input_stats[name]["entropy"].append(entropy.cpu())
                        input_stats[name]["level"].append(
                            encoder.last_market_tokens[name].norm(dim=-1).cpu()
                        )
                        input_stats[name]["dispersion"].append(
                            encoder.last_market_dispersions[name].norm(dim=-1).cpu()
                        )
            probabilities = torch.cat(probability_batches)
            split_probabilities[split_name] = probabilities
            entropy = -(
                probabilities * (probabilities + eps).log()
            ).sum(dim=-1)
            hard = probabilities.argmax(dim=-1)
            mean_p = probabilities.mean(dim=(0, 1))
            print(f"  [{label} {split_name.upper()}] mean p={mean_p.numpy().round(4)}")
            print(
                f"  [{label} {split_name.upper()}] entropy={entropy.mean().item():.6f} "
                f"mean max p={probabilities.max(dim=-1).values.mean().item():.6f}"
            )
            print(f"  [{label} {split_name.upper()}] argmax occupancy")
            for state in range(branch.K):
                print(
                    f"    state {state} = {(hard == state).float().mean().item() * 100:.2f}%"
                )

    parts = {name: torch.cat(values) for name, values in test_parts.items()}
    p = parts["p"]
    prior = parts["prior"]
    previous = parts["previous"]
    q = parts["q"]
    entropy = -(p * (p + eps).log()).sum(dim=-1)
    transition_magnitude = (p[:, 1:] - p[:, :-1]).abs().sum(dim=-1)
    print(
        f"  [{label} p] mean={p.mean(dim=(0, 1)).numpy().round(4)} "
        f"std={p.std(unbiased=False).item():.6f} min={p.min().item():.6f} "
        f"max={p.max().item():.6f}"
    )
    print(
        f"  [{label} p] entropy mean={entropy.mean().item():.6f} "
        f"std={entropy.std(unbiased=False).item():.6f} "
        f"mean max probability={p.max(dim=-1).values.mean().item():.6f}"
    )
    print(
        f"  [{label} p] mean ||p_t-p_(t-1)||_1="
        f"{transition_magnitude.mean().item():.6f}"
    )
    for step in range(0, min(p.shape[1], 20), 2):
        print(f"  [{label} p t={step}] {p[:, step].mean(dim=0).numpy().round(4)}")

    transition = branch.transition_matrix().detach().cpu()
    row_entropy = -(transition * (transition + eps).log()).sum(dim=-1)
    diagonal = transition.diagonal()
    off_diagonal = transition[~torch.eye(branch.K, dtype=torch.bool)]
    print(f"  [{label} transition matrix]\n{transition.numpy().round(6)}")
    print(f"  [{label} transition] row sums={transition.sum(dim=-1).numpy().round(8)}")
    print(
        f"  [{label} transition] diagonal={diagonal.numpy().round(6)} "
        f"mean persistence={diagonal.mean().item():.6f} "
        f"off-diagonal mean={off_diagonal.mean().item():.6f}"
    )
    for state, value in enumerate(row_entropy):
        print(f"  [{label} transition] state {state} entropy={value.item():.6f}")
    print(f"  [{label} transition] mean entropy={row_entropy.mean().item():.6f}")

    prior_posterior_kl = (
        p * ((p + eps).log() - (prior + eps).log())
    ).sum(dim=-1)
    prior_posterior_l1 = (p - prior).abs().sum(dim=-1)
    print(
        f"  [{label} prior/posterior] mean KL={prior_posterior_kl.mean().item():.6f} "
        f"mean L1={prior_posterior_l1.mean().item():.6f}"
    )

    pairwise_l1 = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        value = (q[:, :, left] - q[:, :, right]).abs().sum(dim=-1).mean()
        pairwise_l1.append(value)
        print(f"  [{label} posterior heads] mean L1(q{left},q{right})={value.item():.6f}")
    print(
        f"  [{label} posterior heads] mean pairwise L1="
        f"{torch.stack(pairwise_l1).mean().item():.6f}"
    )
    for state in range(branch.K):
        q_entropy = -(
            q[:, :, state] * (q[:, :, state] + eps).log()
        ).sum(dim=-1).mean()
        print(f"  [{label} posterior head {state}] entropy={q_entropy.item():.6f}")

    hard = p.argmax(dim=-1)
    counts = torch.zeros(branch.K, branch.K, dtype=torch.long)
    for previous_state in range(branch.K):
        for next_state in range(branch.K):
            counts[previous_state, next_state] = (
                (hard[:, :-1] == previous_state)
                & (hard[:, 1:] == next_state)
            ).sum()
    empirical = counts.float() / counts.sum(dim=-1, keepdim=True).clamp(min=1)
    print(f"  [{label} empirical transition counts]\n{counts.numpy()}")
    print(f"  [{label} empirical transition probability]\n{empirical.numpy().round(6)}")

    embeddings = branch.regime_embeddings.detach().cpu()
    centered = branch.centered_regime_embeddings().detach().cpu()
    for state in range(branch.K):
        print(
            f"  [{label} embedding state {state}] raw norm="
            f"{embeddings[state].norm().item():.6f} centered norm="
            f"{centered[state].norm().item():.6f}"
        )
    token_norm = parts["E"].norm(dim=-1).mean()
    real_intervention_norm = parts["r_real"].norm(dim=-1).mean()
    effective_intervention_norm = parts["r_effective"].norm(dim=-1).mean()
    print(
        f"  [{label} intervention] mean ||E||={token_norm.item():.6f} "
        f"mean ||r_real||={real_intervention_norm.item():.6f} "
        f"ratio={(real_intervention_norm / (token_norm + eps)).item():.6f}"
    )
    if branch.null_control:
        print(
            f"  [S1C null intervention] mean ||r_real||="
            f"{real_intervention_norm.item():.6f} mean ||r_effective||="
            f"{effective_intervention_norm.item():.6f} "
            f"max_abs(r_effective)={parts['r_effective'].abs().max().item():.3e}"
        )
    uniform_intervention = centered.mean(dim=0)
    print(
        f"  [{label} uniform-p sanity] max_abs(r_uniform)="
        f"{uniform_intervention.abs().max().item():.3e}"
    )

    for name in branch.market_encoder.MARKET_NAMES:
        market_size = branch.market_encoder.market_sizes[name]
        market_entropy = torch.cat(input_stats[name]["entropy"])
        normalized = (
            market_entropy / np.log(market_size)
            if market_size > 1
            else torch.ones_like(market_entropy)
        )
        level_norm = torch.cat(input_stats[name]["level"]).mean()
        dispersion_norm = torch.cat(input_stats[name]["dispersion"]).mean()
        print(
            f"  [{label} input {name}] normalized attention entropy="
            f"{normalized.mean().item():.6f} effective nodes="
            f"{market_entropy.exp().mean().item():.6f} level norm="
            f"{level_norm.item():.6f} dispersion norm={dispersion_norm.item():.6f}"
        )
    print(f"  [{label} input] daily token E norm={token_norm.item():.6f}")

    x_batch, y_batch = next(iter(loaders["test"]))[:2]
    x_batch = x_batch[:16].to(device)
    y_batch = y_batch[:16].to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(x_batch)
    return_loss = _prediction_loss(prediction, y_batch, make_loss())
    switch_raw = inference._last_switch_loss
    weighted_switch = branch.switch_loss()
    total_loss = return_loss + weighted_switch
    prediction_groups = {
        "Regime Evidence Transformer": list(branch.regime_evidence.parameters()),
        "posterior head 0": list(inference.posterior_heads[0].parameters()),
        "posterior head 1": list(inference.posterior_heads[1].parameters()),
        "posterior head 2": list(inference.posterior_heads[2].parameters()),
        "transition_logits": [inference.transition_logits],
        "regime_embeddings": [branch.regime_embeddings],
    }
    prediction_parameters = [
        parameter
        for parameters in prediction_groups.values()
        for parameter in parameters
    ]
    prediction_gradients = torch.autograd.grad(
        return_loss,
        prediction_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    offset = 0
    for name, parameters in prediction_groups.items():
        group_gradients = prediction_gradients[offset:offset + len(parameters)]
        offset += len(parameters)
        print(
            f"  [{label} prediction-only grad] {name}="
            f"{_aggregate_gradient_norm(parameters, group_gradients):.3e}"
        )
    total_loss.backward()
    total_groups = {
        **prediction_groups,
        "MarketAwareTemporalEncoder": list(branch.market_encoder.parameters()),
        "Forecast Transformer": list(branch.transformer.parameters()),
    }
    for name, parameters in total_groups.items():
        print(
            f"  [{label} total-loss grad] {name}="
            f"{_aggregate_gradient_norm(parameters):.3e}"
        )
    if branch.null_control:
        would_be_weighted = branch.scheduled_beta * switch_raw
        print(f"  [S1C loss] L_return={return_loss.item():.6e}")
        print(f"  [S1C loss] L_switch_raw={switch_raw.item():.6e}")
        print(f"  [S1C loss] scheduled_beta={branch.scheduled_beta:.6e}")
        print(f"  [S1C loss] effective_beta={branch.effective_beta:.6e}")
        print(
            f"  [S1C loss] would_be_weighted_switch="
            f"{would_be_weighted.item():.6e}"
        )
        print(f"  [S1C loss] actual_weighted_switch={weighted_switch.item():.6e}")
        print(f"  [S1C loss] total={total_loss.item():.6e}")
        print("  [S1C loss ratio] actual_switch_to_return_ratio=0.000000e+00")
    else:
        print(f"  [S1 loss] L_return={return_loss.item():.6e}")
        print(f"  [S1 loss] L_switch_raw={switch_raw.item():.6e}")
        print(f"  [S1 loss] beta={inference.current_beta:.6e}")
        print(f"  [S1 loss] weighted_switch={weighted_switch.item():.6e}")
        print(f"  [S1 loss] total={total_loss.item():.6e}")
        print(
            f"  [S1 loss ratio] switch_to_return_ratio="
            f"{abs(weighted_switch.item()) / (return_loss.item() + eps):.6e}"
        )

    initial_groups = getattr(branch, "_initial_parameter_groups", None)
    if branch.null_control and initial_groups is not None:
        current_groups = _switching_parameter_groups(branch)
        print("  [S1C parameter drift]")
        for name, initial_parameters in initial_groups.items():
            current_parameters = current_groups[name]
            delta_sq = sum(
                (current.detach().cpu() - initial).square().sum()
                for current, initial in zip(current_parameters, initial_parameters)
            )
            initial_sq = sum(initial.square().sum() for initial in initial_parameters)
            relative_drift = (delta_sq.sqrt() / (initial_sq.sqrt() + eps)).item()
            print(f"    {name} = {relative_drift:.3e}")

    model.eval()
    with torch.no_grad():
        prediction_real = model(x_batch)
        temporal_real = branch.last_h_temporal.clone()
        if branch.null_control:
            forced_a = torch.tensor([1.0, 0.0, 0.0], device=device)
            forced_b = torch.tensor([0.0, 0.0, 1.0], device=device)
            prediction_a = model._switching_transformer_forward(
                x_batch, forced_probabilities=forced_a
            )
            temporal_a = branch.last_h_temporal.clone()
            prediction_b = model._switching_transformer_forward(
                x_batch, forced_probabilities=forced_b
            )
            temporal_b = branch.last_h_temporal.clone()
            print(
                f"  [S1C null-state invariance] h_temporal diff="
                f"{(temporal_a - temporal_b).abs().max().item():.3e} "
                f"prediction diff={(prediction_a - prediction_b).abs().max().item():.3e}"
            )
        else:
            for state in range(branch.K):
                forced = torch.zeros(branch.K, device=device)
                forced[state] = 1.0
                prediction_forced = model._switching_transformer_forward(
                    x_batch, forced_probabilities=forced
                )
                temporal_forced = branch.last_h_temporal
                temporal_diff = (temporal_forced - temporal_real).abs()
                prediction_diff = (prediction_forced - prediction_real).abs()
                print(
                    f"  [S1 forced state {state}] h_temporal mean/max abs diff="
                    f"{temporal_diff.mean().item():.3e}/{temporal_diff.max().item():.3e} "
                    f"prediction mean/max abs diff="
                    f"{prediction_diff.mean().item():.3e}/{prediction_diff.max().item():.3e}"
                )

        split = min(11, x_batch.shape[1])
        original_tokens = branch.encode_market_tokens(x_batch)
        original_p = branch.infer_regimes(original_tokens)
        perturbed = x_batch.clone()
        perturbed[:, split:] = torch.randn_like(perturbed[:, split:])
        perturbed_tokens = branch.encode_market_tokens(perturbed)
        perturbed_p = branch.infer_regimes(perturbed_tokens)
        print(
            f"  [{label} causality] max p diff through t=10="
            f"{(original_p[:, :split] - perturbed_p[:, :split]).abs().max().item():.3e}"
        )

        tokens_batch = branch.encode_market_tokens(x_batch)
        temporal_batch = branch.temporal_forward(tokens_batch)
        p_batch = branch.last_regime_probabilities.clone()
        tokens_single = branch.encode_market_tokens(x_batch[:1])
        temporal_single = branch.temporal_forward(tokens_single)
        p_single = branch.last_regime_probabilities.clone()
        print(
            f"  [{label} batch independence] E="
            f"{(tokens_batch[:1] - tokens_single).abs().max().item():.3e} "
            f"p={(p_batch[:1] - p_single).abs().max().item():.3e} "
            f"h_temporal={(temporal_batch[:1] - temporal_single).abs().max().item():.3e}"
        )

        tokens_reference = branch.encode_market_tokens(x_batch)
        temporal_reference = branch.temporal_forward(tokens_reference)
        p_reference = branch.last_regime_probabilities.clone()
        offsets = {
            "stock": (0, branch.market_encoder.market_sizes["stock"]),
            "bond": (
                branch.market_encoder.market_sizes["stock"],
                branch.market_encoder.market_sizes["stock"]
                + branch.market_encoder.market_sizes["bond"],
            ),
            "commodity": (
                branch.market_encoder.market_sizes["stock"]
                + branch.market_encoder.market_sizes["bond"],
                sum(branch.market_encoder.market_sizes.values()),
            ),
        }
        for name, (start, end) in offsets.items():
            permuted = x_batch.clone()
            order = torch.randperm(end - start, device=device)
            permuted[:, :, start:end] = x_batch[:, :, start:end].index_select(2, order)
            permuted_tokens = branch.encode_market_tokens(permuted)
            permuted_temporal = branch.temporal_forward(permuted_tokens)
            permuted_p = branch.last_regime_probabilities
            print(
                f"  [{label} {name} permutation] E="
                f"{(permuted_tokens - tokens_reference).abs().max().item():.3e} "
                f"p={(permuted_p - p_reference).abs().max().item():.3e} "
                f"h_temporal="
                f"{(permuted_temporal - temporal_reference).abs().max().item():.3e}"
            )


def _switching_filter_rpe_diagnostics(model, loaders, device):
    """Complete filtering, RPE, gradient, and invariance diagnostics for S2F."""
    branch = model.switching_filter_rpe
    inference = branch.regime_inference
    forecast = branch.transformer
    rpe = forecast.relative_position
    eps = inference.eps
    test_parts = {
        name: []
        for name in (
            "p", "prior", "evidence", "E", "H_reg", "H_forecast",
            "h_temporal", "base_bias", "regime_bias", "qk",
        )
    }
    dispersion_parts = {
        name: [] for name in branch.market_encoder.MARKET_NAMES
    }

    model.eval()
    with torch.no_grad():
        for split_name, loader in loaders.items():
            p_batches = []
            prior_batches = []
            for batch in loader:
                x_batch = batch[0].to(device)
                branch(x_batch)
                p_batch = branch.last_regime_probabilities.cpu()
                prior_batch = inference.last_priors.cpu()
                p_batches.append(p_batch)
                prior_batches.append(prior_batch)
                if split_name == "test":
                    test_parts["p"].append(p_batch)
                    test_parts["prior"].append(prior_batch)
                    test_parts["evidence"].append(
                        inference.last_observation_evidence.cpu()
                    )
                    test_parts["E"].append(branch.last_market_tokens.cpu())
                    test_parts["H_reg"].append(branch.last_regime_evidence.cpu())
                    test_parts["H_forecast"].append(
                        forecast.last_transformer_output.cpu()
                    )
                    test_parts["h_temporal"].append(
                        branch.last_h_temporal.cpu()
                    )
                    test_parts["base_bias"].append(rpe.last_base_bias.cpu())
                    test_parts["regime_bias"].append(rpe.last_regime_bias.cpu())
                    test_parts["qk"].append(
                        forecast.layers[0].attention.last_qk_logits.cpu()
                    )
                    for market_name in branch.market_encoder.MARKET_NAMES:
                        dispersion_parts[market_name].append(
                            branch.market_encoder.last_market_dispersions[
                                market_name
                            ].norm(dim=-1).cpu()
                        )

            probabilities = torch.cat(p_batches)
            priors = torch.cat(prior_batches)
            entropy = -(
                probabilities * (probabilities + eps).log()
            ).sum(dim=-1)
            prior_entropy = -(priors * (priors + eps).log()).sum(dim=-1)
            top2 = probabilities.topk(2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
            occupancy = probabilities.argmax(dim=-1)
            print(
                f"  [S2F {split_name.upper()}] mean p="
                f"{probabilities.mean(dim=(0, 1)).numpy().round(4)} "
                f"entropy={entropy.mean().item():.6f} "
                f"mean max p={probabilities.max(dim=-1).values.mean().item():.6f}"
            )
            print(
                f"  [S2F {split_name.upper()}] margin mean/median="
                f"{margin.mean().item():.6f}/{margin.median().item():.6f}"
            )
            print(
                f"  [S2F {split_name.upper()} prior] mean p="
                f"{priors.mean(dim=(0, 1)).numpy().round(4)} "
                f"entropy={prior_entropy.mean().item():.6f} "
                f"mean max p={priors.max(dim=-1).values.mean().item():.6f}"
            )
            print(f"  [S2F {split_name.upper()}] argmax occupancy")
            for state in range(branch.K):
                print(
                    f"    state {state} = "
                    f"{(occupancy == state).float().mean().item() * 100:.2f}%"
                )

    parts = {name: torch.cat(values) for name, values in test_parts.items()}
    p = parts["p"]
    prior = parts["prior"]
    evidence = parts["evidence"]
    probability_sum_error = (p.sum(dim=-1) - 1.0).abs().max()
    print(
        f"  [S2F probability legality] min={p.min().item():.6e} "
        f"max probability-sum error={probability_sum_error.item():.3e}"
    )
    evidence_range = evidence.max(dim=-1).values - evidence.min(dim=-1).values
    evidence_within_std = evidence.std(dim=-1, unbiased=False)
    print(
        f"  [S2F evidence] mean={evidence.mean().item():.6f} "
        f"std={evidence.std(unbiased=False).item():.6f} "
        f"min={evidence.min().item():.6f} max={evidence.max().item():.6f} "
        f"mean abs={evidence.abs().mean().item():.6f}"
    )
    print(
        f"  [S2F evidence] mean within-K range={evidence_range.mean().item():.6f} "
        f"mean within-K std={evidence_within_std.mean().item():.6f}"
    )

    posterior_entropy = -(p * (p + eps).log()).sum(dim=-1)
    prior_entropy = -(prior * (prior + eps).log()).sum(dim=-1)
    posterior_kl = (
        p * ((p + eps).log() - (prior + eps).log())
    ).sum(dim=-1)
    posterior_l1 = (p - prior).abs().sum(dim=-1)
    temporal_l1 = (p[:, 1:] - p[:, :-1]).abs().sum(dim=-1)
    top2 = p.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    print(
        f"  [S2F prior] entropy={prior_entropy.mean().item():.6f} "
        f"mean max p={prior.max(dim=-1).values.mean().item():.6f}"
    )
    print(
        f"  [S2F posterior] entropy={posterior_entropy.mean().item():.6f} "
        f"mean max p={p.max(dim=-1).values.mean().item():.6f} "
        f"margin mean/median={margin.mean().item():.6f}/{margin.median().item():.6f}"
    )
    print(
        f"  [S2F prior/posterior] mean KL={posterior_kl.mean().item():.6f} "
        f"mean L1={posterior_l1.mean().item():.6f}"
    )
    print(
        f"  [S2F dynamics] mean ||p_t-p_(t-1)||_1="
        f"{temporal_l1.mean().item():.6f}"
    )
    for step in range(0, min(p.shape[1], 20), 2):
        print(f"  [S2F p t={step}] {p[:, step].mean(dim=0).numpy().round(4)}")

    transition = branch.transition_matrix().detach().cpu()
    transition_entropy = -(
        transition * (transition + eps).log()
    ).sum(dim=-1)
    diagonal = transition.diagonal()
    print(f"  [S2F transition matrix]\n{transition.numpy().round(6)}")
    print(
        f"  [S2F transition] row sums="
        f"{transition.sum(dim=-1).numpy().round(8)} diagonal="
        f"{diagonal.numpy().round(6)} mean persistence={diagonal.mean().item():.6f}"
    )
    print(
        f"  [S2F transition] row entropy="
        f"{transition_entropy.numpy().round(6)} mean={transition_entropy.mean().item():.6f}"
    )
    initial_transition = getattr(branch, "_initial_transition_logits", None)
    absolute_drift_value = float("nan")
    if initial_transition is not None:
        drift = (
            inference.transition_logits.detach().cpu() - initial_transition
        ).norm() / (initial_transition.norm() + eps)
        absolute_drift = (
            inference.transition_logits.detach().cpu() - initial_transition
        ).norm()
        absolute_drift_value = absolute_drift.item()
        print(
            f"  [S2F transition] parameter drift absolute="
            f"{absolute_drift.item():.3e} relative={drift.item():.3e}"
        )

    base_bias = parts["base_bias"]
    regime_bias = parts["regime_bias"]
    qk = parts["qk"]
    centered_tables = rpe.centered_regime_rpe().detach().cpu()
    print(
        f"  [S2F RPE norm] base raw={rpe.base_rpe.detach().norm().item():.6f} "
        f"regime raw={rpe.regime_rpe.detach().norm().item():.6f} "
        f"centered regime={centered_tables.norm().item():.6f}"
    )
    mean_qk = qk.abs().mean()
    mean_base = base_bias.abs().mean()
    mean_regime = regime_bias.abs().mean()
    print(
        f"  [S2F attention scale] mean |QK/sqrt(d)|={mean_qk.item():.6e} "
        f"mean |base_rpe_bias|={mean_base.item():.6e} "
        f"mean |regime_rpe_bias|={mean_regime.item():.6e}"
    )
    print(
        f"  [S2F attention scale] base/QK="
        f"{(mean_base / (mean_qk + eps)).item():.6e} regime/QK="
        f"{(mean_regime / (mean_qk + eps)).item():.6e} regime/base="
        f"{(mean_regime / (mean_base + eps)).item():.6e}"
    )
    uniform = torch.full(
        (1, p.shape[1], branch.K), 1.0 / branch.K, device=device
    )
    _, uniform_regime_bias = rpe.components(uniform)
    print(
        f"  [S2F uniform-p RPE sanity] max_abs(regime_bias_uniform)="
        f"{uniform_regime_bias.abs().max().item():.3e}"
    )
    state_distances = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        distance = (centered_tables[left] - centered_tables[right]).abs().mean()
        state_distances.append(distance.item())
        print(
            f"  [S2F state RPE distance] state{left}/state{right}="
            f"{distance.item():.6e}"
        )
    table_center = rpe.max_len - 1
    for delta in (-19, -10, -5, -2, -1, 0, 1, 2, 5, 10, 19):
        if abs(delta) >= rpe.max_len:
            continue
        values = centered_tables[:, :, table_center + delta].mean(dim=-1)
        print(
            f"  [S2F lag RPE delta={delta:+d}] state means="
            f"{values.numpy().round(6)}"
        )

    for market_name in branch.market_encoder.MARKET_NAMES:
        dispersion = torch.cat(dispersion_parts[market_name]).reshape(-1)
        flat_p = p.reshape(-1, branch.K)
        count = max(1, int(0.1 * dispersion.numel()))
        order = dispersion.argsort()
        low_p = flat_p[order[:count]].mean(dim=0)
        high_p = flat_p[order[-count:]].mean(dim=0)
        print(
            f"  [S2F {market_name} dispersion sensitivity] high p="
            f"{high_p.numpy().round(4)} low p={low_p.numpy().round(4)} "
            f"L1={(high_p - low_p).abs().sum().item():.6f}"
        )

    print(
        f"  [S2F temporal norms] E={parts['E'].norm(dim=-1).mean().item():.6f} "
        f"H_reg={parts['H_reg'].norm(dim=-1).mean().item():.6f} "
        f"Forecast H={parts['H_forecast'].norm(dim=-1).mean().item():.6f} "
        f"h_temporal={parts['h_temporal'].norm(dim=-1).mean().item():.6f}"
    )

    x_batch, y_batch = next(iter(loaders["test"]))[:2]
    x_batch = x_batch[:16].to(device)
    y_batch = y_batch[:16].to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(x_batch)
    return_loss = _prediction_loss(prediction, y_batch, make_loss())
    switch_raw = inference._last_switch_loss
    weighted_switch = branch.switch_loss()
    total_loss = return_loss + weighted_switch
    gradient_groups = {
        "Regime Evidence Transformer": list(branch.regime_evidence.parameters()),
        "observation_head": list(inference.observation_head.parameters()),
        "transition_logits": [inference.transition_logits],
        "base_rpe": [rpe.base_rpe],
        "regime_rpe": [rpe.regime_rpe],
        "Forecast Transformer": [
            parameter
            for name, parameter in forecast.named_parameters()
            if not name.startswith("relative_position.")
        ],
    }
    flat_parameters = [
        parameter for parameters in gradient_groups.values()
        for parameter in parameters
    ]
    prediction_gradients = torch.autograd.grad(
        return_loss, flat_parameters, retain_graph=True, allow_unused=True
    )
    offset = 0
    prediction_gradient_norms = {}
    for name, parameters in gradient_groups.items():
        values = prediction_gradients[offset:offset + len(parameters)]
        offset += len(parameters)
        gradient_norm = _aggregate_gradient_norm(parameters, values)
        prediction_gradient_norms[name] = gradient_norm
        print(
            f"  [S2F prediction-only grad] {name}="
            f"{gradient_norm:.3e}"
        )
    total_loss.backward()
    for name, parameters in gradient_groups.items():
        print(
            f"  [S2F total-loss grad] {name}="
            f"{_aggregate_gradient_norm(parameters):.3e}"
        )
    print(f"  [S2F loss] L_return={return_loss.item():.6e}")
    print(f"  [S2F loss] L_switch_raw={switch_raw.item():.6e}")
    print(f"  [S2F loss] beta={inference.current_beta:.6e}")
    print(f"  [S2F loss] weighted_switch={weighted_switch.item():.6e}")
    print(f"  [S2F loss] total={total_loss.item():.6e}")
    print(
        f"  [S2F loss] switch_to_return_ratio="
        f"{abs(weighted_switch.item()) / (return_loss.item() + eps):.6e}"
    )

    model.eval()
    forced_prediction_diffs = []
    with torch.no_grad():
        prediction_real = model(x_batch)
        temporal_real = branch.last_h_temporal.clone()
        regime_bias_real = rpe.last_regime_bias.clone()
        for state in range(branch.K):
            forced = torch.zeros(branch.K, device=device)
            forced[state] = 1.0
            prediction_forced = model._switching_filter_rpe_forward(
                x_batch, forced_rpe_probabilities=forced
            )
            temporal_forced = branch.last_h_temporal
            regime_bias_forced = rpe.last_regime_bias
            bias_diff = (regime_bias_forced - regime_bias_real).abs()
            temporal_diff = (temporal_forced - temporal_real).abs()
            prediction_diff = (prediction_forced - prediction_real).abs()
            forced_prediction_diffs.append(prediction_diff.max().item())
            print(
                f"  [S2F forced state {state}] relative bias mean/max diff="
                f"{bias_diff.mean().item():.3e}/{bias_diff.max().item():.3e} "
                f"h_temporal mean/max diff="
                f"{temporal_diff.mean().item():.3e}/{temporal_diff.max().item():.3e} "
                f"prediction mean/max diff="
                f"{prediction_diff.mean().item():.3e}/{prediction_diff.max().item():.3e}"
            )

        split = min(11, x_batch.shape[1])
        original_tokens = branch.encode_market_tokens(x_batch)
        original_p = branch.infer_regimes(original_tokens)
        original_prior = inference.last_priors.clone()
        perturbed = x_batch.clone()
        perturbed[:, split:] = torch.randn_like(perturbed[:, split:])
        perturbed_tokens = branch.encode_market_tokens(perturbed)
        perturbed_p = branch.infer_regimes(perturbed_tokens)
        perturbed_prior = inference.last_priors
        print(
            f"  [S2F causality] max p diff through t=10="
            f"{(original_p[:, :split] - perturbed_p[:, :split]).abs().max().item():.3e} "
            f"max prior diff="
            f"{(original_prior[:, :split] - perturbed_prior[:, :split]).abs().max().item():.3e}"
        )

        tokens_batch = branch.encode_market_tokens(x_batch)
        temporal_batch = branch.temporal_forward(tokens_batch)
        p_batch = branch.last_regime_probabilities.clone()
        prior_batch = inference.last_priors.clone()
        tokens_single = branch.encode_market_tokens(x_batch[:1])
        temporal_single = branch.temporal_forward(tokens_single)
        p_single = branch.last_regime_probabilities.clone()
        prior_single = inference.last_priors.clone()
        print(
            f"  [S2F batch independence] E="
            f"{(tokens_batch[:1] - tokens_single).abs().max().item():.3e} "
            f"p={(p_batch[:1] - p_single).abs().max().item():.3e} "
            f"prior={(prior_batch[:1] - prior_single).abs().max().item():.3e} "
            f"h_temporal={(temporal_batch[:1] - temporal_single).abs().max().item():.3e}"
        )

        tokens_reference = branch.encode_market_tokens(x_batch)
        temporal_reference = branch.temporal_forward(tokens_reference)
        p_reference = branch.last_regime_probabilities.clone()
        offsets = {
            "stock": (0, branch.market_encoder.market_sizes["stock"]),
            "bond": (
                branch.market_encoder.market_sizes["stock"],
                branch.market_encoder.market_sizes["stock"]
                + branch.market_encoder.market_sizes["bond"],
            ),
            "commodity": (
                branch.market_encoder.market_sizes["stock"]
                + branch.market_encoder.market_sizes["bond"],
                sum(branch.market_encoder.market_sizes.values()),
            ),
        }
        for market_name, (start, end) in offsets.items():
            permuted = x_batch.clone()
            order = torch.randperm(end - start, device=device)
            permuted[:, :, start:end] = x_batch[:, :, start:end].index_select(
                2, order
            )
            permuted_tokens = branch.encode_market_tokens(permuted)
            permuted_temporal = branch.temporal_forward(permuted_tokens)
            permuted_p = branch.last_regime_probabilities
            print(
                f"  [S2F {market_name} permutation] E="
                f"{(permuted_tokens - tokens_reference).abs().max().item():.3e} "
                f"p={(permuted_p - p_reference).abs().max().item():.3e} "
                f"h_temporal="
                f"{(permuted_temporal - temporal_reference).abs().max().item():.3e}"
            )

    return {
        "posterior_entropy": posterior_entropy.mean().item(),
        "mean_margin": margin.mean().item(),
        "temporal_l1": temporal_l1.mean().item(),
        "transition_drift": absolute_drift_value,
        "prediction_gradient_norms": prediction_gradient_norms,
        "mean_state_rpe_distance": float(np.mean(state_distances)),
        "uniform_regime_bias_max": uniform_regime_bias.abs().max().item(),
        "max_forced_prediction_diff": max(forced_prediction_diffs),
    }


def _switching_latent_diagnostics(model, loaders, device):
    """Mechanism diagnostics for D0/D0B switching latent dynamics."""
    branch = model.switching_latent_transformer
    balanced = branch.balanced_readout
    latent_memory = branch.use_latent_memory
    dynamic_slope = branch.use_dynamic_slope
    active_memory = latent_memory and not branch.zero_init_memory_projection
    label = (
        "D0C" if dynamic_slope
        else ("D1" if branch.use_regime_relative_memory
        else (
            "D1A2" if active_memory
            else ("D1A" if latent_memory else ("D0B" if balanced else "D0"))
        ))
    )
    rpe_label = f"{label} LongMemory" if latent_memory else label
    filtering = branch.regime_filter
    memory = branch.long_memory
    eps = filtering.eps
    test_parts = {
        name: []
        for name in (
            "E", "H", "p", "prior", "evidence", "Z", "candidates",
            "h_temporal", "base_bias", "qk",
        )
    }
    balanced_parts = {
        name: [] for name in ("h_long", "h_micro", "balanced_concat")
    }
    memory_parts = {
        name: [] for name in (
            "M", "attention", "scores", "content_scores", "regime_bias",
            "q_norm", "k_norm", "v_norm", "base_preactivation",
            "memory_injection"
        )
    }
    slope_parts = {
        name: [] for name in (
            "delta_h", "contributions", "base", "weighted"
        )
    }
    dispersion_last = {
        name: [] for name in branch.market_encoder.MARKET_NAMES
    }

    model.eval()
    with torch.no_grad():
        for split_name, loader in loaders.items():
            p_batches = []
            prior_batches = []
            split_slope_batches = []
            split_h_batches = []
            for batch in loader:
                x_batch = batch[0].to(device)
                branch(x_batch)
                p_batch = branch.last_regime_probabilities.cpu()
                prior_batch = branch.last_regime_priors.cpu()
                p_batches.append(p_batch)
                prior_batches.append(prior_batch)
                if dynamic_slope:
                    split_slope_batches.append(
                        branch.last_long_memory_slopes[:, 1:].cpu()
                    )
                    split_h_batches.append(
                        branch.last_long_memory[:, 1:].cpu()
                    )
                if split_name == "test":
                    test_parts["E"].append(branch.last_market_tokens.cpu())
                    test_parts["H"].append(branch.last_long_memory.cpu())
                    test_parts["p"].append(p_batch)
                    test_parts["prior"].append(prior_batch)
                    test_parts["evidence"].append(
                        branch.last_regime_evidence.cpu()
                    )
                    test_parts["Z"].append(branch.last_latent_states.cpu())
                    test_parts["candidates"].append(
                        branch.last_latent_candidates.cpu()
                    )
                    test_parts["h_temporal"].append(
                        branch.last_h_temporal.cpu()
                    )
                    if balanced:
                        balanced_parts["h_long"].append(
                            branch.last_h_long.cpu()
                        )
                        balanced_parts["h_micro"].append(
                            branch.last_h_micro.cpu()
                        )
                        balanced_parts["balanced_concat"].append(
                            branch.last_balanced_concat.cpu()
                        )
                    if latent_memory:
                        memory_parts["M"].append(
                            branch.last_latent_memories.cpu()
                        )
                        memory_parts["attention"].append(
                            branch.last_latent_memory_attention.cpu()
                        )
                        memory_parts["scores"].append(
                            branch.last_latent_memory_scores.cpu()
                        )
                        memory_parts["content_scores"].append(
                            branch.last_latent_memory_content_scores.cpu()
                        )
                        memory_parts["regime_bias"].append(
                            branch.last_regime_memory_biases.cpu()
                        )
                        memory_parts["q_norm"].append(
                            branch.last_latent_memory_query_norms.cpu()
                        )
                        memory_parts["k_norm"].append(
                            branch.last_latent_memory_key_norms.cpu()
                        )
                        memory_parts["v_norm"].append(
                            branch.last_latent_memory_value_norms.cpu()
                        )
                        memory_parts["base_preactivation"].append(
                            branch.last_generator_base_preactivations.cpu()
                        )
                        memory_parts["memory_injection"].append(
                            branch.last_generator_memory_injections.cpu()
                        )
                    if dynamic_slope:
                        slope_parts["delta_h"].append(
                            branch.last_long_memory_slopes.cpu()
                        )
                        slope_parts["contributions"].append(
                            branch.last_slope_contributions.cpu()
                        )
                        slope_parts["base"].append(
                            branch.last_generator_base_preactivations.cpu()
                        )
                        slope_parts["weighted"].append(
                            branch.last_regime_weighted_slope.cpu()
                        )
                    test_parts["base_bias"].append(
                        memory.last_base_relative_bias.cpu()
                    )
                    test_parts["qk"].append(
                        memory.layers[0].attention.last_qk_logits.cpu()
                    )
                    for market_name in branch.market_encoder.MARKET_NAMES:
                        dispersion_last[market_name].append(
                            branch.market_encoder.last_market_dispersions[
                                market_name
                            ][:, -1].norm(dim=-1).cpu()
                        )

            probabilities = torch.cat(p_batches)
            priors = torch.cat(prior_batches)
            entropy = -(
                probabilities * (probabilities + eps).log()
            ).sum(dim=-1)
            margin_values = probabilities.topk(2, dim=-1).values
            margin = margin_values[..., 0] - margin_values[..., 1]
            occupancy = probabilities.argmax(dim=-1)
            print(
                f"  [{label} {split_name.upper()}] mean p="
                f"{probabilities.mean(dim=(0, 1)).numpy().round(4)} "
                f"entropy={entropy.mean().item():.6f} "
                f"mean max p={probabilities.max(dim=-1).values.mean().item():.6f} "
                f"margin={margin.mean().item():.6f}"
            )
            print(f"  [{label} {split_name.upper()}] argmax occupancy")
            for state in range(branch.K):
                print(
                    f"    state {state} = "
                    f"{(occupancy == state).float().mean().item() * 100:.2f}%"
                )
            prior_entropy = -(priors * (priors + eps).log()).sum(dim=-1)
            print(
                f"  [{label} {split_name.upper()} prior] entropy="
                f"{prior_entropy.mean().item():.6f} mean max p="
                f"{priors.max(dim=-1).values.mean().item():.6f}"
            )
            if dynamic_slope:
                split_slopes = torch.cat(split_slope_batches)
                split_h = torch.cat(split_h_batches)
                slope_norms = split_slopes.norm(dim=-1).reshape(-1)
                relative = (
                    split_slopes.norm(dim=-1)
                    / (split_h.norm(dim=-1) + eps)
                ).reshape(-1)
                print(
                    f"  [D0C {split_name.upper()} raw slope norm] mean/std/median/p90/max="
                    f"{slope_norms.mean().item():.6f}/"
                    f"{slope_norms.std(unbiased=False).item():.6f}/"
                    f"{slope_norms.median().item():.6f}/"
                    f"{torch.quantile(slope_norms, 0.9).item():.6f}/"
                    f"{slope_norms.max().item():.6f}"
                )
                print(
                    f"  [D0C {split_name.upper()} slope/H] mean/median/p90="
                    f"{relative.mean().item():.6e}/"
                    f"{relative.median().item():.6e}/"
                    f"{torch.quantile(relative, 0.9).item():.6e}"
                )

    parts = {name: torch.cat(values) for name, values in test_parts.items()}
    balanced_values = (
        {name: torch.cat(values) for name, values in balanced_parts.items()}
        if balanced else {}
    )
    memory_values = (
        {name: torch.cat(values) for name, values in memory_parts.items()}
        if latent_memory else {}
    )
    slope_values = (
        {name: torch.cat(values) for name, values in slope_parts.items()}
        if dynamic_slope else {}
    )
    p = parts["p"]
    prior = parts["prior"]
    Z = parts["Z"]
    candidates = parts["candidates"]
    probability_error = (p.sum(dim=-1) - 1.0).abs().max()
    print(
        f"  [{label} probability legality] min p={p.min().item():.6e} "
        f"max probability-sum error={probability_error.item():.3e}"
    )

    posterior_entropy = -(p * (p + eps).log()).sum(dim=-1)
    prior_entropy = -(prior * (prior + eps).log()).sum(dim=-1)
    posterior_kl = (
        p * ((p + eps).log() - (prior + eps).log())
    ).sum(dim=-1)
    posterior_l1 = (p - prior).abs().sum(dim=-1)
    temporal_l1 = (p[:, 1:] - p[:, :-1]).abs().sum(dim=-1)
    print(
        f"  [{label} prior/posterior] prior entropy={prior_entropy.mean().item():.6f} "
        f"posterior entropy={posterior_entropy.mean().item():.6f} "
        f"mean KL={posterior_kl.mean().item():.6f} "
        f"mean L1={posterior_l1.mean().item():.6f}"
    )
    print(
        f"  [{label} regime dynamics] mean ||p_t-p_(t-1)||_1="
        f"{temporal_l1.mean().item():.6f}"
    )
    for step in range(0, min(p.shape[1], 20), 2):
        print(f"  [{label} p t={step}] {p[:, step].mean(dim=0).numpy().round(4)}")

    transition = branch.transition_matrix().detach().cpu()
    transition_entropy = -(
        transition * (transition + eps).log()
    ).sum(dim=-1)
    diagonal = transition.diagonal()
    print(f"  [{label} transition matrix]\n{transition.numpy().round(6)}")
    print(
        f"  [{label} transition] row sums="
        f"{transition.sum(dim=-1).numpy().round(8)} diagonal="
        f"{diagonal.numpy().round(6)} mean persistence={diagonal.mean().item():.6f} "
        f"row entropy={transition_entropy.numpy().round(6)}"
    )
    initial_transition = getattr(branch, "_initial_transition_logits", None)
    transition_drift = float("nan")
    if initial_transition is not None:
        transition_drift = (
            filtering.transition_logits.detach().cpu() - initial_transition
        ).norm().item()
        print(
            f"  [{label} transition] absolute transition_logits drift="
            f"{transition_drift:.3e}"
        )

    delta = memory.last_relative_delta.cpu()
    causal_entries = delta >= 0
    qk_abs = parts["qk"].abs()[..., causal_entries].mean()
    base_abs = parts["base_bias"].abs()[..., causal_entries].mean()
    print(
        f"  [{rpe_label} Base RPE] norm={memory.base_rpe.detach().norm().item():.6f} "
        f"mean |QK/sqrt(d)|={qk_abs.item():.6e} "
        f"mean |base bias|={base_abs.item():.6e} "
        f"base/QK={(base_abs / (qk_abs + eps)).item():.6e}"
    )
    table_center = memory.max_len - 1
    base_table = memory.base_rpe.detach().cpu()
    for lag in (0, 1, 2, 5, 10, 19):
        if lag >= memory.max_len:
            continue
        head_values = base_table[:, table_center + lag]
        print(
            f"  [{rpe_label} Base RPE delta=+{lag}] heads="
            f"{head_values.numpy().round(6)} mean={head_values.mean().item():.6f} "
            f"std={head_values.std(unbiased=False).item():.6f}"
        )

    candidate_norms = candidates.norm(dim=-1)
    for state in range(branch.K):
        norms = candidate_norms[:, :, state]
        print(
            f"  [{label} candidate state {state}] norm mean="
            f"{norms.mean().item():.6f} std={norms.std(unbiased=False).item():.6f} "
            f"max={norms.max().item():.6f}"
        )
    pairwise_l1_values = []
    pairwise_cosine_values = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        left_values = candidates[:, :, left]
        right_values = candidates[:, :, right]
        l1 = (left_values - right_values).abs().mean()
        cosine = F.cosine_similarity(
            left_values, right_values, dim=-1
        ).mean()
        pairwise_l1_values.append(l1.item())
        pairwise_cosine_values.append(cosine.item())
        print(
            f"  [{label} candidate pair {left}/{right}] mean L1={l1.item():.6f} "
            f"cosine={cosine.item():.6f}"
        )

    z_norm = Z.norm(dim=-1)
    z_previous = torch.cat([torch.zeros_like(Z[:, :1]), Z[:, :-1]], dim=1)
    z_change = (Z - z_previous).norm(dim=-1)
    consecutive_z_cosine = F.cosine_similarity(
        Z[:, 1:], Z[:, :-1], dim=-1, eps=eps
    )
    print(
        f"  [{label} Z trajectory] mean norm={z_norm.mean().item():.6f} "
        f"max norm={z_norm.max().item():.6f} "
        f"mean ||Z_t-Z_(t-1)||_2={z_change.mean().item():.6f}"
    )
    print(
        f"  [{label} Z direction] consecutive cosine mean/std="
        f"{consecutive_z_cosine.mean().item():.6f}/"
        f"{consecutive_z_cosine.std(unbiased=False).item():.6f} "
        f"mean directional change="
        f"{(1.0 - consecutive_z_cosine).mean().item():.6f}"
    )
    for step in (0, 5, 10, 15, 19):
        if step < Z.shape[1]:
            print(
                f"  [{label} Z t={step}] mean norm="
                f"{z_norm[:, step].mean().item():.6f}"
            )
    weighted_contributions = (
        p * candidate_norms
    ).mean(dim=(0, 1))
    print(
        f"  [{label} weighted generator contributions] "
        f"{weighted_contributions.numpy().round(6)}"
    )

    slope_base_ratio_mean = float("nan")
    mean_slope_pairwise_l1 = float("nan")
    if dynamic_slope:
        delta_h = slope_values["delta_h"]
        slope_contributions_tensor = slope_values["contributions"]
        slope_base = slope_values["base"]
        weighted_slope = slope_values["weighted"]
        valid_delta = delta_h[:, 1:]
        h_values = parts["H"]
        h_cosine = F.cosine_similarity(
            h_values[:, 1:], h_values[:, :-1], dim=-1, eps=eps
        )
        delta_cosine = F.cosine_similarity(
            delta_h[:, 2:], delta_h[:, 1:-1], dim=-1, eps=eps
        )
        print(
            f"  [D0C dynamic slope] mean ||delta_H||="
            f"{valid_delta.norm(dim=-1).mean().item():.6f} "
            f"mean cosine(H_t,H_(t-1))={h_cosine.mean().item():.6f} "
            f"mean cosine(delta_H_t,delta_H_(t-1))="
            f"{delta_cosine.mean().item():.6f}"
        )
        slope_ratios = []
        for state in range(branch.K):
            state_slope_norm = slope_contributions_tensor[:, 1:, state].norm(
                dim=-1
            )
            state_base_norm = slope_base[:, 1:, state].norm(dim=-1)
            ratio = state_slope_norm.mean() / (state_base_norm.mean() + eps)
            slope_ratios.append(ratio.item())
            print(
                f"  [D0C slope state {state}] output norm mean/std/max="
                f"{state_slope_norm.mean().item():.6f}/"
                f"{state_slope_norm.std(unbiased=False).item():.6f}/"
                f"{state_slope_norm.max().item():.6f} slope/base={ratio.item():.6e}"
            )
        slope_base_ratio_mean = float(np.mean(slope_ratios))
        slope_pairwise_l1 = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            left_values = slope_contributions_tensor[:, 1:, left]
            right_values = slope_contributions_tensor[:, 1:, right]
            pair_l1 = (left_values - right_values).abs().mean()
            pair_cosine = F.cosine_similarity(
                left_values, right_values, dim=-1, eps=eps
            ).mean()
            slope_pairwise_l1.append(pair_l1.item())
            print(
                f"  [D0C slope pair {left}/{right}] mean L1="
                f"{pair_l1.item():.6f} cosine={pair_cosine.item():.6f}"
            )
        mean_slope_pairwise_l1 = float(np.mean(slope_pairwise_l1))
        weighted_slope_norm = weighted_slope[:, 1:].norm(dim=-1)
        print(
            "  [D0C regime-weighted slope contribution] norm mean/std="
            f"{weighted_slope_norm.mean().item():.6f}/"
            f"{weighted_slope_norm.std(unbiased=False).item():.6f}"
        )
        initial_slope = getattr(branch, "_initial_slope_parameters", {})
        for state, projection in enumerate(branch.slope_projections):
            initial_weight = initial_slope.get(f"slope_projections.{state}.weight")
            initial_norm = (
                initial_weight.norm().item()
                if initial_weight is not None else float("nan")
            )
            drift = (
                (projection.weight.detach().cpu() - initial_weight).norm().item()
                if initial_weight is not None else float("nan")
            )
            print(
                f"  [D0C slope projection {state}] initial norm="
                f"{initial_norm:.6e} best norm="
                f"{projection.weight.detach().norm().item():.6e} drift={drift:.6e}"
            )

    E_norm = parts["E"].norm(dim=-1).mean()
    H_norm = parts["H"].norm(dim=-1).mean()
    H_last_norm = parts["H"][:, -1].norm(dim=-1).mean()
    Z_last_norm = Z[:, -1].norm(dim=-1).mean()
    readout_input_norm = torch.cat(
        [parts["H"][:, -1], Z[:, -1]], dim=-1
    ).norm(dim=-1).mean()
    h_temporal_norm = parts["h_temporal"].norm(dim=-1).mean()
    print(
        f"  [{label} pre-balance scale] E={E_norm.item():.6f} "
        f"H={H_norm.item():.6f} Z={z_norm.mean().item():.6f} "
        f"H_T={H_last_norm.item():.6f} Z_T={Z_last_norm.item():.6f} "
        f"Z_T/H_T={(Z_last_norm / (H_last_norm + eps)).item():.6e} "
        f"readout input={readout_input_norm.item():.6f} "
        f"h_temporal={h_temporal_norm.item():.6f}"
    )
    readout_weight = branch.state_readout.weight.detach().cpu()
    split_dim = readout_weight.shape[1] // 2 if balanced else 128
    h_weight_norm = readout_weight[:, :split_dim].norm()
    z_weight_norm = readout_weight[:, split_dim:].norm()
    readout_weight_ratio = (z_weight_norm / (h_weight_norm + eps)).item()
    print(
        f"  [{label} readout weights] ||W_long||={h_weight_norm.item():.6f} "
        f"||W_micro||={z_weight_norm.item():.6f} "
        f"W_micro/W_long={readout_weight_ratio:.6f}"
    )
    post_balance_ratio = float("nan")
    if balanced:
        h_long_norm = balanced_values["h_long"].norm(dim=-1).mean()
        h_micro_norm = balanced_values["h_micro"].norm(dim=-1).mean()
        post_balance_ratio = (h_micro_norm / (h_long_norm + eps)).item()
        balanced_concat_norm = balanced_values["balanced_concat"].norm(
            dim=-1
        ).mean()
        print(
            f"  [{label} post-balance scale] ||h_long||={h_long_norm.item():.6f} "
            f"||h_micro||={h_micro_norm.item():.6f} "
            f"micro/long={post_balance_ratio:.6f} "
            f"balanced concat={balanced_concat_norm.item():.6f} "
            f"h_temporal={h_temporal_norm.item():.6f}"
        )

    memory_expected_lag = None
    memory_tv_to_uniform = float("nan")
    memory_injection_ratio = float("nan")
    regime_rpe_norm = float("nan")
    mean_regime_rpe_distance = float("nan")
    regime_bias_abs_mean = float("nan")
    if latent_memory:
        M = memory_values["M"]
        attention = memory_values["attention"]
        memory_scores_tensor = memory_values["scores"]
        memory_content_tensor = memory_values["content_scores"]
        memory_regime_bias_tensor = memory_values["regime_bias"]
        if branch.use_regime_relative_memory:
            raw_rpe = branch.regime_relative_lag_bias.detach().cpu()
            centered_rpe = (
                raw_rpe - raw_rpe.mean(dim=0, keepdim=True)
            )
            regime_rpe_norm = raw_rpe.norm().item()
            mean_regime_rpe_distance = float(np.mean([
                (centered_rpe[left] - centered_rpe[right]).norm().item()
                for left, right in ((0, 1), (0, 2), (1, 2))
            ]))
            uniform_p = torch.full((1, branch.K), 1.0 / branch.K)
            uniform_bias = branch.regime_memory_bias(
                uniform_p.to(device), min(5, raw_rpe.shape[-1])
            ).detach().cpu()
            initial_rpe = getattr(
                branch, "_initial_regime_relative_lag_bias", None
            )
            rpe_drift = (
                (raw_rpe - initial_rpe).norm().item()
                if initial_rpe is not None else float("nan")
            )
            print(
                f"  [D1 regime-relative latent RPE] shape={tuple(raw_rpe.shape)} "
                f"raw norm={raw_rpe.norm().item():.6e} "
                f"centered norm={centered_rpe.norm().item():.6e} "
                f"initial norm="
                f"{initial_rpe.norm().item() if initial_rpe is not None else float('nan'):.6e} "
                f"drift={rpe_drift:.6e} "
                f"uniform-p max |bias|={uniform_bias.abs().max().item():.3e}"
            )
            for state in range(branch.K):
                print(
                    f"  [D1 regime RPE state {state}] raw norm="
                    f"{raw_rpe[state].norm().item():.6e} centered norm="
                    f"{centered_rpe[state].norm().item():.6e}"
                )
            for left, right in ((0, 1), (0, 2), (1, 2)):
                pair_difference = centered_rpe[left] - centered_rpe[right]
                print(
                    f"  [D1 regime RPE pair {left}/{right}] mean abs/L2 distance="
                    f"{pair_difference.abs().mean().item():.6e}/"
                    f"{pair_difference.norm().item():.6e}"
                )
            for lag in (1, 2, 3, 5, 10, 15, 19):
                if lag <= centered_rpe.shape[-1]:
                    lag_values = centered_rpe[:, :, lag - 1]
                    print(
                        f"  [D1 regime RPE lag={lag}] centered heads="
                        f"{lag_values.numpy().round(6)} state head mean/std="
                        f"{lag_values.mean(dim=-1).numpy().round(6)}/"
                        f"{lag_values.std(dim=-1, unbiased=False).numpy().round(6)}"
                    )
        valid_probabilities = []
        raw_entropies = []
        normalized_entropies = []
        effective_counts = []
        max_weights = []
        uniform_max_weights = []
        tv_to_uniform = []
        score_abs_values = []
        score_raw_values = []
        score_range_values = []
        score_std_values = []
        content_abs_values = []
        content_raw_values = []
        content_range_values = []
        content_std_values = []
        regime_bias_abs_values = []
        regime_bias_raw_values = []
        regime_bias_range_values = []
        regime_bias_std_values = []
        expected_lags = []
        expected_lags_by_head = [
            [] for _ in range(branch.latent_memory_attention.n_heads)
        ]
        entropy_by_head = [
            [] for _ in range(branch.latent_memory_attention.n_heads)
        ]
        tv_by_head = [
            [] for _ in range(branch.latent_memory_attention.n_heads)
        ]
        short_masses = []
        medium_masses = []
        long_masses = []
        time_diagnostics = {}
        sum_errors = []
        for step in range(1, attention.shape[1]):
            weights = attention[:, step, :, :step]
            valid_probabilities.append(weights.reshape(-1))
            sum_errors.append((weights.sum(dim=-1) - 1.0).abs().reshape(-1))
            if step < 2:
                continue
            entropy_t = -(weights * (weights + eps).log()).sum(dim=-1)
            normalized_t = entropy_t / np.log(step)
            uniform_t = torch.full_like(weights, 1.0 / step)
            tv_t = 0.5 * (weights - uniform_t).abs().sum(dim=-1)
            scores_t = memory_scores_tensor[:, step, :, :step]
            content_t = memory_content_tensor[:, step, :, :step]
            regime_bias_t = memory_regime_bias_tensor[:, step, :, :step]
            score_range_t = scores_t.max(dim=-1).values - scores_t.min(dim=-1).values
            score_std_t = scores_t.std(dim=-1, unbiased=False)
            lags = torch.arange(
                step, 0, -1, dtype=weights.dtype
            ).view(1, 1, step)
            expected_t = (weights * lags).sum(dim=-1)
            max_t = weights.max(dim=-1).values
            raw_entropies.append(entropy_t.reshape(-1))
            normalized_entropies.append(normalized_t.reshape(-1))
            effective_counts.append(entropy_t.exp().reshape(-1))
            max_weights.append(max_t.reshape(-1))
            uniform_max_weights.append(torch.full_like(max_t, 1.0 / step).reshape(-1))
            tv_to_uniform.append(tv_t.reshape(-1))
            score_abs_values.append(scores_t.abs().reshape(-1))
            score_raw_values.append(scores_t.reshape(-1))
            score_range_values.append(score_range_t.reshape(-1))
            score_std_values.append(score_std_t.reshape(-1))
            content_abs_values.append(content_t.abs().reshape(-1))
            content_raw_values.append(content_t.reshape(-1))
            content_range_values.append(
                (content_t.max(dim=-1).values - content_t.min(dim=-1).values).reshape(-1)
            )
            content_std_values.append(
                content_t.std(dim=-1, unbiased=False).reshape(-1)
            )
            regime_bias_abs_values.append(regime_bias_t.abs().reshape(-1))
            regime_bias_raw_values.append(regime_bias_t.reshape(-1))
            regime_bias_range_values.append(
                (regime_bias_t.max(dim=-1).values - regime_bias_t.min(dim=-1).values).reshape(-1)
            )
            regime_bias_std_values.append(
                regime_bias_t.std(dim=-1, unbiased=False).reshape(-1)
            )
            expected_lags.append(expected_t.reshape(-1))
            for head in range(branch.latent_memory_attention.n_heads):
                expected_lags_by_head[head].append(expected_t[:, head].reshape(-1))
                entropy_by_head[head].append(normalized_t[:, head].reshape(-1))
                tv_by_head[head].append(tv_t[:, head].reshape(-1))
            short_masses.append(weights[..., lags.flatten() <= 2].sum(dim=-1).reshape(-1))
            medium_mask = (lags.flatten() >= 3) & (lags.flatten() <= 5)
            medium_masses.append(weights[..., medium_mask].sum(dim=-1).reshape(-1))
            long_masses.append(weights[..., lags.flatten() >= 6].sum(dim=-1).reshape(-1))
            if step in (5, 10, 15, 19):
                time_diagnostics[step] = (
                    normalized_t.mean().item(),
                    tv_t.mean().item(),
                    expected_t.mean().item(),
                    max_t.mean().item(),
                    score_range_t.mean().item(),
                )
        probability_values = torch.cat(valid_probabilities)
        sum_error_values = torch.cat(sum_errors)
        entropy_values = torch.cat(raw_entropies)
        normalized_values = torch.cat(normalized_entropies)
        effective_values = torch.cat(effective_counts)
        max_values = torch.cat(max_weights)
        uniform_max_values = torch.cat(uniform_max_weights)
        tv_values = torch.cat(tv_to_uniform)
        memory_tv_to_uniform = tv_values.mean().item()
        score_abs = torch.cat(score_abs_values)
        score_raw = torch.cat(score_raw_values)
        score_ranges = torch.cat(score_range_values)
        score_stds = torch.cat(score_std_values)
        content_abs = torch.cat(content_abs_values)
        content_raw = torch.cat(content_raw_values)
        content_ranges = torch.cat(content_range_values)
        content_stds = torch.cat(content_std_values)
        regime_bias_abs = torch.cat(regime_bias_abs_values)
        regime_bias_raw = torch.cat(regime_bias_raw_values)
        regime_bias_abs_mean = regime_bias_abs.mean().item()
        regime_bias_ranges = torch.cat(regime_bias_range_values)
        regime_bias_stds = torch.cat(regime_bias_std_values)
        memory_expected_lag = torch.stack([
            torch.zeros(attention.shape[0]),
            *[
                (
                    attention[:, step, :, :step]
                    * torch.arange(step, 0, -1, dtype=attention.dtype).view(1, 1, step)
                ).sum(dim=-1).mean(dim=-1)
                for step in range(1, attention.shape[1])
            ],
        ], dim=1)
        expected_values = torch.cat(expected_lags)
        print(
            f"  [{label} LatentMemory legality] min attention probability="
            f"{probability_values.min().item():.6e} max attention-sum error="
            f"{sum_error_values.max().item():.3e}"
        )
        print(
            f"  [{label} LatentMemory entropy] raw mean={entropy_values.mean().item():.6f} "
            f"normalized mean/std={normalized_values.mean().item():.6f}/"
            f"{normalized_values.std(unbiased=False).item():.6f} "
            f"p25/p50/p75={torch.quantile(normalized_values, torch.tensor([0.25, 0.5, 0.75])).numpy().round(6)} "
            f"effective history={effective_values.mean().item():.6f}"
        )
        print(
            f"  [{label} LatentMemory uniformity] TV mean/median/p90="
            f"{tv_values.mean().item():.6e}/{tv_values.median().item():.6e}/"
            f"{torch.quantile(tv_values, 0.9).item():.6e}"
        )
        print(
            f"  [{label} LatentMemory score] mean abs={score_abs.mean().item():.6e} "
            f"global std={score_raw.std(unbiased=False).item():.6e} "
            f"within-history range={score_ranges.mean().item():.6e} "
            f"within-history std={score_stds.mean().item():.6e}"
        )
        print(
            f"  [{label} LatentMemory content score] mean abs="
            f"{content_abs.mean().item():.6e} global mean/std/min/max="
            f"{content_raw.mean().item():.6e}/"
            f"{content_raw.std(unbiased=False).item():.6e}/"
            f"{content_raw.min().item():.6e}/"
            f"{content_raw.max().item():.6e} within-history range/std="
            f"{content_ranges.mean().item():.6e}/{content_stds.mean().item():.6e}"
        )
        print(
            f"  [{label} regime-memory bias] mean abs="
            f"{regime_bias_abs.mean().item():.6e} global mean/std/min/max="
            f"{regime_bias_raw.mean().item():.6e}/"
            f"{regime_bias_raw.std(unbiased=False).item():.6e}/"
            f"{regime_bias_raw.min().item():.6e}/"
            f"{regime_bias_raw.max().item():.6e} within-history range/std="
            f"{regime_bias_ranges.mean().item():.6e}/"
            f"{regime_bias_stds.mean().item():.6e} bias/content="
            f"{(regime_bias_abs.mean() / (content_abs.mean() + eps)).item():.6e}"
        )
        print(
            f"  [{label} LatentMemory QKV norm] Q="
            f"{memory_values['q_norm'][:, 1:].mean().item():.6f} K="
            f"{memory_values['k_norm'][:, 1:].mean().item():.6f} V="
            f"{memory_values['v_norm'][:, 1:].mean().item():.6f}"
        )
        print(
            f"  [{label} LatentMemory concentration] max weight mean/median="
            f"{max_values.mean().item():.6f}/{max_values.median().item():.6f} "
            f"matched-uniform mean max={uniform_max_values.mean().item():.6f} "
            f"expected lag mean/median/std={expected_values.mean().item():.6f}/"
            f"{expected_values.median().item():.6f}/"
            f"{expected_values.std(unbiased=False).item():.6f}"
        )
        print(
            f"  [{label} LatentMemory lag mass] short(1-2)={torch.cat(short_masses).mean().item():.6f} "
            f"medium(3-5)={torch.cat(medium_masses).mean().item():.6f} "
            f"long(6+)={torch.cat(long_masses).mean().item():.6f}"
        )
        for head in range(branch.latent_memory_attention.n_heads):
            print(
                f"  [{label} LatentMemory head {head}] normalized entropy="
                f"{torch.cat(entropy_by_head[head]).mean().item():.6f} "
                f"TV={torch.cat(tv_by_head[head]).mean().item():.6e} "
                f"expected lag={torch.cat(expected_lags_by_head[head]).mean().item():.6f}"
            )
        for step, (entropy_t, tv_t, lag_t, max_t, score_range_t) in time_diagnostics.items():
            print(
                f"  [{label} LatentMemory t={step}] normalized entropy={entropy_t:.6f} "
                f"TV={tv_t:.6e} expected lag={lag_t:.6f} max weight={max_t:.6f} "
                f"score range={score_range_t:.6e}"
            )
        if branch.use_regime_relative_memory:
            valid_lag = memory_expected_lag[:, 1:]
            valid_p = p[:, 1:]
            associated_state = valid_p.argmax(dim=-1)
            for state in range(branch.K):
                selected_lags = valid_lag[associated_state == state]
                print(
                    f"  [D1 state{state}-associated expected lag] count="
                    f"{selected_lags.numel()} mean="
                    f"{selected_lags.mean().item() if selected_lags.numel() else float('nan'):.6f}"
                )
            for state in range(branch.K):
                weights = valid_p[..., state]
                associated_lag = (
                    weights * valid_lag
                ).sum() / (weights.sum() + eps)
                centered_probability = weights - weights.mean()
                centered_lag = valid_lag - valid_lag.mean()
                correlation = (
                    centered_probability * centered_lag
                ).mean() / (
                    centered_probability.std(unbiased=False)
                    * centered_lag.std(unbiased=False) + eps
                )
                print(
                    f"  [D1 regime {state} memory horizon] soft expected lag="
                    f"{associated_lag.item():.6f} corr(p, expected lag)="
                    f"{correlation.item():.6f}"
                )
        m_norm = M.norm(dim=-1)
        print(
            f"  [{label} LatentMemory norm] mean/std/max={m_norm.mean().item():.6f}/"
            f"{m_norm.std(unbiased=False).item():.6f}/{m_norm.max().item():.6f}"
        )
        memory_change = (M[:, 2:] - M[:, 1:-1]).norm(dim=-1)
        memory_cosine = F.cosine_similarity(
            M[:, 2:], M[:, 1:-1], dim=-1, eps=eps
        )
        print(
            f"  [{label} LatentMemory temporal] mean L2 change="
            f"{memory_change.mean().item():.6f} consecutive cosine="
            f"{memory_cosine.mean().item():.6f}"
        )
        for step in (1, 5, 10, 15, 19):
            if step < M.shape[1]:
                print(
                    f"  [{label} LatentMemory norm t={step}] "
                    f"{m_norm[:, step].mean().item():.6f}"
                )
        base_norm = memory_values["base_preactivation"].norm(dim=-1)
        injection_norm = memory_values["memory_injection"].norm(dim=-1)
        memory_injection_ratio = (
            injection_norm.mean() / (base_norm.mean() + eps)
        ).item()
        for state in range(branch.K):
            print(
                f"  [{label} memory injection state {state}] base norm="
                f"{base_norm[:, :, state].mean().item():.6f} memory norm="
                f"{injection_norm[:, :, state].mean().item():.6f} ratio="
                f"{(injection_norm[:, :, state].mean() / (base_norm[:, :, state].mean() + eps)).item():.6e}"
            )
        initial_memory = getattr(branch, "_initial_latent_memory_parameters", {})
        for name, parameter in branch.named_parameters():
            if not (
                name.startswith("latent_memory_attention.")
                or name.startswith("memory_projections.")
            ):
                continue
            drift = float("nan")
            initial_norm = float("nan")
            if name in initial_memory:
                initial_norm = initial_memory[name].norm().item()
                drift = (
                    parameter.detach().cpu() - initial_memory[name]
                ).norm().item()
            print(
                f"  [{label} memory parameter] {name} initial norm="
                f"{initial_norm:.6e} best norm="
                f"{parameter.detach().norm().item():.6e} drift={drift:.6e}"
            )

    p_last = p[:, -1]
    z_last_sample_norm = Z[:, -1].norm(dim=-1)
    for market_name in branch.market_encoder.MARKET_NAMES:
        dispersion = torch.cat(dispersion_last[market_name])
        count = max(1, int(0.1 * dispersion.numel()))
        order = dispersion.argsort()
        low_indices = order[:count]
        high_indices = order[-count:]
        low_p = p_last[low_indices].mean(dim=0)
        high_p = p_last[high_indices].mean(dim=0)
        print(
            f"  [{label} {market_name} dispersion sensitivity] high p="
            f"{high_p.numpy().round(4)} low p={low_p.numpy().round(4)} "
            f"L1={(high_p - low_p).abs().sum().item():.6f} "
            f"high/low ||Z_T||="
            f"{z_last_sample_norm[high_indices].mean().item():.6f}/"
            f"{z_last_sample_norm[low_indices].mean().item():.6f}"
            + (
                f" high/low ||h_micro||="
                f"{balanced_values['h_micro'].norm(dim=-1)[high_indices].mean().item():.6f}/"
                f"{balanced_values['h_micro'].norm(dim=-1)[low_indices].mean().item():.6f}"
                if balanced else ""
            )
            + (
                f" high/low expected memory lag="
                f"{memory_expected_lag[high_indices, -1].mean().item():.6f}/"
                f"{memory_expected_lag[low_indices, -1].mean().item():.6f}"
                if latent_memory else ""
            )
            + (
                f" high/low ||delta_H_T||="
                f"{slope_values['delta_h'][:, -1].norm(dim=-1)[high_indices].mean().item():.6f}/"
                f"{slope_values['delta_h'][:, -1].norm(dim=-1)[low_indices].mean().item():.6f} "
                f"high/low ||weighted slope_T||="
                f"{slope_values['weighted'][:, -1].norm(dim=-1)[high_indices].mean().item():.6f}/"
                f"{slope_values['weighted'][:, -1].norm(dim=-1)[low_indices].mean().item():.6f}"
                if dynamic_slope else ""
            )
        )

    x_batch, y_batch = next(iter(loaders["test"]))[:2]
    x_batch = x_batch[:16].to(device)
    y_batch = y_batch[:16].to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(x_batch)
    return_loss = _prediction_loss(prediction, y_batch, make_loss())
    switch_raw = filtering._last_switch_loss
    weighted_switch = branch.switch_loss()
    total_loss = return_loss + weighted_switch
    gradient_groups = {
        "Market Encoder": list(branch.market_encoder.parameters()),
        "LongMemory Transformer": list(memory.layers.parameters()),
        "Base RPE": [memory.base_rpe],
        "regime evidence": list(filtering.regime_evidence.parameters()),
        "transition logits": [filtering.transition_logits],
        "G0": list(branch.latent_transition.generators[0].parameters()),
        "G1": list(branch.latent_transition.generators[1].parameters()),
        "G2": list(branch.latent_transition.generators[2].parameters()),
    }
    if balanced:
        gradient_groups.update({
            "long_memory_readout": list(branch.long_memory_readout.parameters())
            + list(branch.long_memory_norm.parameters()),
            "micro_state_readout": list(branch.micro_state_readout.parameters())
            + list(branch.micro_state_norm.parameters()),
        })
    if latent_memory:
        for projection_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            gradient_groups[f"LatentMemory {projection_name}"] = list(
                getattr(
                    branch.latent_memory_attention, projection_name
                ).parameters()
            )
        for state, projection in enumerate(branch.memory_projections):
            gradient_groups[f"memory_proj_{state}"] = list(
                projection.parameters()
            )
        if branch.use_regime_relative_memory:
            gradient_groups["Regime latent-memory RPE"] = [
                branch.regime_relative_lag_bias
            ]
    if dynamic_slope:
        for state, projection in enumerate(branch.slope_projections):
            gradient_groups[f"slope_projection_{state}"] = list(
                projection.parameters()
            )
    gradient_groups["state readout"] = list(branch.state_readout.parameters())
    flat_parameters = [
        parameter for values in gradient_groups.values() for parameter in values
    ]
    prediction_gradients = torch.autograd.grad(
        return_loss, flat_parameters, retain_graph=True, allow_unused=True
    )
    offset = 0
    prediction_gradient_norms = {}
    for name, parameters in gradient_groups.items():
        selected = prediction_gradients[offset:offset + len(parameters)]
        offset += len(parameters)
        norm = _aggregate_gradient_norm(parameters, selected)
        prediction_gradient_norms[name] = norm
        print(f"  [{label} prediction-only grad] {name}={norm:.3e}")
    generator_gradient_mean = float(np.mean([
        prediction_gradient_norms[name] for name in ("G0", "G1", "G2")
    ]))
    if dynamic_slope:
        slope_gradient_mean = float(np.mean([
            prediction_gradient_norms[f"slope_projection_{state}"]
            for state in range(branch.K)
        ]))
        print(
            "  [D0C prediction-only grad ratio] slope/generator="
            f"{slope_gradient_mean / (generator_gradient_mean + eps):.6e} "
            f"slope/LongMemory="
            f"{slope_gradient_mean / (prediction_gradient_norms['LongMemory Transformer'] + eps):.6e}"
        )
    if branch.use_regime_relative_memory:
        qk_gradient = float(np.mean([
            prediction_gradient_norms["LatentMemory q_proj"],
            prediction_gradient_norms["LatentMemory k_proj"],
        ]))
        regime_rpe_gradient = prediction_gradient_norms[
            "Regime latent-memory RPE"
        ]
        print(
            "  [D1 prediction-only grad comparison] regime-RPE/QK="
            f"{regime_rpe_gradient / (qk_gradient + eps):.6e} "
            f"regime-RPE/evidence="
            f"{regime_rpe_gradient / (prediction_gradient_norms['regime evidence'] + eps):.6e}"
        )
    long_memory_gradient = prediction_gradient_norms["LongMemory Transformer"]
    print(
        f"  [{label} prediction-only grad ratio] generator/LongMemory="
        f"{generator_gradient_mean / (long_memory_gradient + eps):.6e}"
    )
    if balanced:
        print(
            f"  [{label} prediction-only grad ratio] "
            "microReadout/longReadout="
            f"{prediction_gradient_norms['micro_state_readout'] / (prediction_gradient_norms['long_memory_readout'] + eps):.6e}"
        )
    total_loss.backward()
    total_gradient_norms = {}
    for name, parameters in gradient_groups.items():
        norm = _aggregate_gradient_norm(parameters)
        total_gradient_norms[name] = norm
        print(
            f"  [{label} total-loss grad] {name}={norm:.3e}"
        )
    total_generator_gradient_mean = float(np.mean([
        total_gradient_norms[name] for name in ("G0", "G1", "G2")
    ]))
    if dynamic_slope:
        total_slope_gradient_mean = float(np.mean([
            total_gradient_norms[f"slope_projection_{state}"]
            for state in range(branch.K)
        ]))
        print(
            "  [D0C total-loss grad ratio] slope/generator="
            f"{total_slope_gradient_mean / (total_generator_gradient_mean + eps):.6e} "
            f"slope/LongMemory="
            f"{total_slope_gradient_mean / (total_gradient_norms['LongMemory Transformer'] + eps):.6e}"
        )
    if branch.use_regime_relative_memory:
        total_qk_gradient = float(np.mean([
            total_gradient_norms["LatentMemory q_proj"],
            total_gradient_norms["LatentMemory k_proj"],
        ]))
        total_rpe_gradient = total_gradient_norms[
            "Regime latent-memory RPE"
        ]
        print(
            "  [D1 total-loss grad comparison] regime-RPE/QK="
            f"{total_rpe_gradient / (total_qk_gradient + eps):.6e} "
            f"regime-RPE/evidence="
            f"{total_rpe_gradient / (total_gradient_norms['regime evidence'] + eps):.6e}"
        )
    print(
        f"  [{label} total-loss grad ratio] generator/LongMemory="
        f"{total_generator_gradient_mean / (total_gradient_norms['LongMemory Transformer'] + eps):.6e}"
    )
    if balanced:
        print(
            f"  [{label} total-loss grad ratio] microReadout/longReadout="
            f"{total_gradient_norms['micro_state_readout'] / (total_gradient_norms['long_memory_readout'] + eps):.6e}"
        )
    print(f"  [{label} loss] L_return={return_loss.item():.6e}")
    print(f"  [{label} loss] L_switch_raw={switch_raw.item():.6e}")
    print(f"  [{label} loss] beta={filtering.current_beta:.6e}")
    print(f"  [{label} loss] weighted_switch={weighted_switch.item():.6e}")
    print(f"  [{label} loss] total={total_loss.item():.6e}")
    print(
        f"  [{label} loss] switch_to_return_ratio="
        f"{abs(weighted_switch.item()) / (return_loss.item() + eps):.6e}"
    )

    model.eval()
    forced_prediction_diffs = []
    forced_z_diffs = []
    zero_regime_rpe_prediction_mean_diff = float("nan")
    zero_slope_prediction_mean_diff = float("nan")
    flipped_slope_prediction_mean_diff = float("nan")
    slope_regime_interactions = []
    with torch.no_grad():
        prediction_real = model(x_batch)
        temporal_real = branch.last_h_temporal.clone()
        z_real = branch.last_z_last.clone()
        micro_real = branch.last_h_micro.clone() if balanced else None
        if latent_memory:
            query_last = branch.last_long_memory[:, -1]
            history_last = branch.last_latent_states[:, :-1]
            real_rpe_probability = branch.last_regime_probabilities[:, -1]
            real_bias = (
                branch.regime_memory_bias(
                    real_rpe_probability, history_last.shape[1]
                )
                if branch.use_regime_relative_memory else None
            )
            memory_original, attention_original = (
                branch.latent_memory_attention(
                    query_last, history_last, score_bias=real_bias
                )
            )
            order = torch.randperm(history_last.shape[1], device=device)
            memory_permuted, _ = branch.latent_memory_attention(
                query_last,
                history_last[:, order],
                score_bias=(
                    real_bias[:, :, order] if real_bias is not None else None
                ),
            )
            perturbed_history = history_last.clone()
            perturbed_history[:, 0] = perturbed_history[:, 0] + 1e-3
            memory_perturbed, attention_perturbed = (
                branch.latent_memory_attention(
                    query_last, perturbed_history, score_bias=real_bias
                )
            )
            print(
                f"  [{label} LatentMemory paired permutation] max M diff="
                f"{(memory_permuted - memory_original).abs().max().item():.3e}"
            )
            print(
                f"  [{label} LatentMemory content perturbation] M mean/max diff="
                f"{(memory_perturbed - memory_original).abs().mean().item():.3e}/"
                f"{(memory_perturbed - memory_original).abs().max().item():.3e} "
                f"attention mean/max diff="
                f"{(attention_perturbed - attention_original).abs().mean().item():.3e}/"
                f"{(attention_perturbed - attention_original).abs().max().item():.3e}"
            )
            if branch.use_regime_relative_memory:
                attention_module = branch.latent_memory_attention
                batch_size, history_length, _ = history_last.shape
                values = attention_module.v_proj(history_last).view(
                    batch_size, history_length, attention_module.n_heads,
                    attention_module.head_dim,
                ).transpose(1, 2)
                value_only_memory = torch.einsum(
                    "bhl,bhld->bhd",
                    attention_original,
                    values[:, :, order],
                ).reshape(batch_size, attention_module.memory_dim)
                value_only_memory = attention_module.out_proj(value_only_memory)
                print(
                    "  [D1 LatentMemory value-only permutation] M mean/max diff="
                    f"{(value_only_memory - memory_original).abs().mean().item():.3e}/"
                    f"{(value_only_memory - memory_original).abs().max().item():.3e}"
                )
                content_memory, content_attention = attention_module(
                    query_last, history_last,
                    score_bias=torch.zeros_like(real_bias),
                )
                rpe_memory, rpe_attention = attention_module(
                    query_last, history_last,
                    score_bias=real_bias,
                    use_content_scores=False,
                )
                uniform_probability = torch.full_like(
                    real_rpe_probability, 1.0 / branch.K
                )
                uniform_direct_bias = branch.regime_memory_bias(
                    uniform_probability, history_last.shape[1]
                )
                uniform_memory, uniform_attention = attention_module(
                    query_last, history_last,
                    score_bias=uniform_direct_bias,
                )
                print(
                    "  [D1 full/content-only/RPE-only attention] full-content="
                    f"{(attention_original - content_attention).abs().mean().item():.3e} "
                    f"full-RPE={ (attention_original - rpe_attention).abs().mean().item():.3e} "
                    f"M content/RPE diff={(content_memory - rpe_memory).abs().mean().item():.3e}"
                )
                print(
                    "  [D1 uniform-regime direct memory sanity] max |bias|="
                    f"{uniform_direct_bias.abs().max().item():.3e} "
                    f"attention/M vs content-only max diff="
                    f"{(uniform_attention-content_attention).abs().max().item():.3e}/"
                    f"{(uniform_memory-content_memory).abs().max().item():.3e}"
                )
                shift_history = history_last[:, :-1]
                shift_length = shift_history.shape[1]
                normal_shift_bias = branch.regime_memory_bias(
                    real_rpe_probability, shift_length
                )
                shifted_indices = torch.arange(
                    shift_length, 0, -1, device=device
                )
                shifted_bias = branch.regime_memory_bias(
                    real_rpe_probability,
                    shift_length,
                    lag_indices=shifted_indices,
                )
                normal_shift_memory, normal_shift_attention = attention_module(
                    query_last, shift_history, score_bias=normal_shift_bias
                )
                shifted_memory, shifted_attention = attention_module(
                    query_last, shift_history, score_bias=shifted_bias
                )
                print(
                    "  [D1 latent-memory lag-shift] attention mean/max diff="
                    f"{(shifted_attention - normal_shift_attention).abs().mean().item():.3e}/"
                    f"{(shifted_attention - normal_shift_attention).abs().max().item():.3e} "
                    f"M mean/max diff={(shifted_memory - normal_shift_memory).abs().mean().item():.3e}/"
                    f"{(shifted_memory - normal_shift_memory).abs().max().item():.3e}"
                )
                forced_rpe_memories = []
                for state in range(branch.K):
                    state_probability = torch.zeros(
                        branch.K, device=device
                    )
                    state_probability[state] = 1.0
                    state_bias = branch.regime_memory_bias(
                        state_probability.view(1, -1).expand(batch_size, -1),
                        history_length,
                    )
                    state_memory, state_attention = attention_module(
                        query_last, history_last, score_bias=state_bias
                    )
                    forced_rpe_memories.append(state_memory)
                    lags = torch.arange(
                        history_length, 0, -1, device=device,
                        dtype=state_attention.dtype,
                    ).view(1, 1, -1)
                    print(
                        f"  [D1 forced-RPE-only state {state}] expected lag="
                        f"{(state_attention * lags).sum(dim=-1).mean().item():.6f} "
                        f"attention/full mean diff="
                        f"{(state_attention - attention_original).abs().mean().item():.3e}"
                    )
                for left, right in ((0, 1), (0, 2), (1, 2)):
                    difference = (
                        forced_rpe_memories[left] - forced_rpe_memories[right]
                    ).abs()
                    cosine = F.cosine_similarity(
                        forced_rpe_memories[left],
                        forced_rpe_memories[right],
                        dim=-1,
                    ).mean()
                    print(
                        f"  [D1 forced-RPE M pair {left}/{right}] mean/max diff="
                        f"{difference.mean().item():.3e}/{difference.max().item():.3e} "
                        f"cosine={cosine.item():.6f}"
                    )
        for state in range(branch.K):
            forced = torch.zeros(branch.K, device=device)
            forced[state] = 1.0
            prediction_forced = model._switching_latent_transformer_forward(
                x_batch, forced_probabilities=forced
            )
            temporal_forced = branch.last_h_temporal
            z_forced = branch.last_z_last
            micro_forced = branch.last_h_micro if balanced else None
            z_diff = (z_forced - z_real).abs()
            temporal_diff = (temporal_forced - temporal_real).abs()
            prediction_diff = (prediction_forced - prediction_real).abs()
            forced_z_diffs.append(z_diff.max().item())
            forced_prediction_diffs.append(prediction_diff.max().item())
            print(
                f"  [{label} forced-full-regime state {state}] Z_T mean/max diff="
                f"{z_diff.mean().item():.3e}/{z_diff.max().item():.3e} "
                + (
                    f"h_micro mean/max diff="
                    f"{(micro_forced - micro_real).abs().mean().item():.3e}/"
                    f"{(micro_forced - micro_real).abs().max().item():.3e} "
                    if balanced else ""
                )
                + f"h_temporal mean/max diff="
                f"{temporal_diff.mean().item():.3e}/{temporal_diff.max().item():.3e} "
                f"prediction mean/max diff="
                f"{prediction_diff.mean().item():.3e}/{prediction_diff.max().item():.3e}"
            )
            if dynamic_slope:
                prediction_forced_zero_slope = (
                    model._switching_latent_transformer_forward(
                        x_batch,
                        forced_probabilities=forced,
                        slope_scale=0.0,
                    )
                )
                interaction = (
                    prediction_forced - prediction_forced_zero_slope
                ).abs()
                slope_regime_interactions.append(interaction.mean().item())
                print(
                    f"  [D0C slope-regime interaction state {state}] "
                    f"normalSlope-vs-zeroSlope prediction mean/max diff="
                    f"{interaction.mean().item():.3e}/"
                    f"{interaction.max().item():.3e}"
                )

        if branch.use_regime_relative_memory:
            for state in range(branch.K):
                forced_rpe = torch.zeros(branch.K, device=device)
                forced_rpe[state] = 1.0
                prediction_forced_rpe = model._switching_latent_transformer_forward(
                    x_batch, forced_rpe_probabilities=forced_rpe
                )
                z_forced_rpe = branch.last_z_last.clone()
                micro_forced_rpe = branch.last_h_micro.clone()
                temporal_forced_rpe = branch.last_h_temporal.clone()
                print(
                    f"  [D1 forced-RPE-only state {state}] Z_T mean/max diff="
                    f"{(z_forced_rpe-z_real).abs().mean().item():.3e}/"
                    f"{(z_forced_rpe-z_real).abs().max().item():.3e} "
                    f"h_micro={(micro_forced_rpe-micro_real).abs().mean().item():.3e}/"
                    f"{(micro_forced_rpe-micro_real).abs().max().item():.3e} "
                    f"h_temporal={(temporal_forced_rpe-temporal_real).abs().mean().item():.3e}/"
                    f"{(temporal_forced_rpe-temporal_real).abs().max().item():.3e} "
                    f"prediction={(prediction_forced_rpe-prediction_real).abs().mean().item():.3e}/"
                    f"{(prediction_forced_rpe-prediction_real).abs().max().item():.3e}"
                )
            uniform_rpe = torch.full(
                (branch.K,), 1.0 / branch.K, device=device
            )
            prediction_uniform_rpe = model._switching_latent_transformer_forward(
                x_batch, forced_rpe_probabilities=uniform_rpe
            )
            uniform_z = branch.last_z_last.clone()
            uniform_bias_max = branch.last_regime_memory_biases.abs().max().item()
            prediction_zero_rpe = model._switching_latent_transformer_forward(
                x_batch, zero_regime_memory_bias=True
            )
            zero_rpe_z = branch.last_z_last.clone()
            zero_rpe_micro = branch.last_h_micro.clone()
            zero_rpe_temporal = branch.last_h_temporal.clone()
            print(
                "  [D1 uniform-RPE] max |bias|="
                f"{uniform_bias_max:.3e} Z/prediction vs zero-RPE max diff="
                f"{(uniform_z-zero_rpe_z).abs().max().item():.3e}/"
                f"{(prediction_uniform_rpe-prediction_zero_rpe).abs().max().item():.3e}"
            )
            print(
                "  [D1 zero-regime-RPE trajectory] Z_T mean/max diff="
                f"{(zero_rpe_z-z_real).abs().mean().item():.3e}/"
                f"{(zero_rpe_z-z_real).abs().max().item():.3e} "
                f"h_micro={(zero_rpe_micro-micro_real).abs().mean().item():.3e}/"
                f"{(zero_rpe_micro-micro_real).abs().max().item():.3e} "
                f"h_temporal={(zero_rpe_temporal-temporal_real).abs().mean().item():.3e}/"
                f"{(zero_rpe_temporal-temporal_real).abs().max().item():.3e} "
                f"prediction={(prediction_zero_rpe-prediction_real).abs().mean().item():.3e}/"
                f"{(prediction_zero_rpe-prediction_real).abs().max().item():.3e}"
            )
            zero_regime_rpe_prediction_mean_diff = (
                prediction_zero_rpe - prediction_real
            ).abs().mean().item()

        if dynamic_slope:
            prediction_zero_slope = model._switching_latent_transformer_forward(
                x_batch, slope_scale=0.0
            )
            zero_slope_z = branch.last_z_last.clone()
            zero_slope_micro = branch.last_h_micro.clone()
            zero_slope_temporal = branch.last_h_temporal.clone()
            zero_slope_prediction_diff = (
                prediction_zero_slope - prediction_real
            ).abs()
            zero_slope_prediction_mean_diff = (
                zero_slope_prediction_diff.mean().item()
            )
            print(
                "  [D0C zero-slope trajectory] Z_T mean/max diff="
                f"{(zero_slope_z-z_real).abs().mean().item():.3e}/"
                f"{(zero_slope_z-z_real).abs().max().item():.3e} "
                f"h_micro={(zero_slope_micro-micro_real).abs().mean().item():.3e}/"
                f"{(zero_slope_micro-micro_real).abs().max().item():.3e} "
                f"h_temporal={(zero_slope_temporal-temporal_real).abs().mean().item():.3e}/"
                f"{(zero_slope_temporal-temporal_real).abs().max().item():.3e} "
                f"prediction={zero_slope_prediction_diff.mean().item():.3e}/"
                f"{zero_slope_prediction_diff.max().item():.3e}"
            )
            prediction_flipped_slope = model._switching_latent_transformer_forward(
                x_batch, slope_scale=-1.0
            )
            flipped_z = branch.last_z_last.clone()
            flipped_micro = branch.last_h_micro.clone()
            flipped_temporal = branch.last_h_temporal.clone()
            flipped_prediction_diff = (
                prediction_flipped_slope - prediction_real
            ).abs()
            flipped_slope_prediction_mean_diff = (
                flipped_prediction_diff.mean().item()
            )
            print(
                "  [D0C flipped-slope trajectory] Z_T mean/max diff="
                f"{(flipped_z-z_real).abs().mean().item():.3e}/"
                f"{(flipped_z-z_real).abs().max().item():.3e} "
                f"h_micro={(flipped_micro-micro_real).abs().mean().item():.3e}/"
                f"{(flipped_micro-micro_real).abs().max().item():.3e} "
                f"h_temporal={(flipped_temporal-temporal_real).abs().mean().item():.3e}/"
                f"{(flipped_temporal-temporal_real).abs().max().item():.3e} "
                f"prediction={flipped_prediction_diff.mean().item():.3e}/"
                f"{flipped_prediction_diff.max().item():.3e}"
            )
            for scale in (0.5, 1.5):
                scaled_prediction = model._switching_latent_transformer_forward(
                    x_batch, slope_scale=scale
                )
                scaled_difference = (scaled_prediction - prediction_real).abs()
                print(
                    f"  [D0C slope magnitude {scale:.1f}x] prediction mean/max diff="
                    f"{scaled_difference.mean().item():.3e}/"
                    f"{scaled_difference.max().item():.3e}"
                )

        prediction_zero_z = model._switching_latent_transformer_forward(
            x_batch, zero_readout_component="Z"
        )
        temporal_zero_z = branch.last_h_temporal.clone()
        prediction_zero_h = model._switching_latent_transformer_forward(
            x_batch, zero_readout_component="H"
        )
        temporal_zero_h = branch.last_h_temporal.clone()
        zero_z_prediction_diff = (prediction_zero_z - prediction_real).abs()
        zero_h_prediction_diff = (prediction_zero_h - prediction_real).abs()
        zero_z_temporal_diff = (temporal_zero_z - temporal_real).abs()
        zero_h_temporal_diff = (temporal_zero_h - temporal_real).abs()
        zero_label_z = "zero-balanced-micro" if balanced else "zero-Z"
        zero_label_h = "zero-balanced-long" if balanced else "zero-H_last"
        print(
            f"  [{label} {zero_label_z}] h_temporal mean/max diff="
            f"{zero_z_temporal_diff.mean().item():.3e}/"
            f"{zero_z_temporal_diff.max().item():.3e} "
            f"prediction mean/max diff={zero_z_prediction_diff.mean().item():.3e}/"
            f"{zero_z_prediction_diff.max().item():.3e}"
        )
        print(
            f"  [{label} {zero_label_h}] h_temporal mean/max diff="
            f"{zero_h_temporal_diff.mean().item():.3e}/"
            f"{zero_h_temporal_diff.max().item():.3e} "
            f"prediction mean/max diff={zero_h_prediction_diff.mean().item():.3e}/"
            f"{zero_h_prediction_diff.max().item():.3e}"
        )
        prediction_impact_ratio = (
            zero_z_prediction_diff.mean()
            / (zero_h_prediction_diff.mean() + eps)
        ).item()
        print(
            f"  [{label} zero-out impact] micro/long prediction-impact ratio="
            f"{prediction_impact_ratio:.6e}"
        )
        if balanced:
            print(
                "  [D0B vs D0 zero-out reference] D0 zero-Z mean=7.417e-05 "
                "D0 zero-H mean=1.112e-02 D0 ratio≈6.670e-03"
            )
        zero_memory_prediction_diff = None
        if latent_memory:
            prediction_zero_memory = model._switching_latent_transformer_forward(
                x_batch, zero_latent_memory=True
            )
            z_zero_memory = branch.last_z_last.clone()
            micro_zero_memory = branch.last_h_micro.clone()
            temporal_zero_memory = branch.last_h_temporal.clone()
            z_memory_diff = (z_zero_memory - z_real).abs()
            micro_memory_diff = (micro_zero_memory - micro_real).abs()
            temporal_memory_diff = (temporal_zero_memory - temporal_real).abs()
            zero_memory_prediction_diff = (
                prediction_zero_memory - prediction_real
            ).abs()
            print(
                f"  [{label} zero-memory trajectory] Z_T mean/max diff="
                f"{z_memory_diff.mean().item():.3e}/{z_memory_diff.max().item():.3e} "
                f"h_micro={micro_memory_diff.mean().item():.3e}/"
                f"{micro_memory_diff.max().item():.3e} h_temporal="
                f"{temporal_memory_diff.mean().item():.3e}/"
                f"{temporal_memory_diff.max().item():.3e} prediction="
                f"{zero_memory_prediction_diff.mean().item():.3e}/"
                f"{zero_memory_prediction_diff.max().item():.3e}"
            )

        split = min(11, x_batch.shape[1])
        branch(x_batch)
        original_E = branch.last_market_tokens.clone()
        original_H = branch.last_long_memory.clone()
        original_p = branch.last_regime_probabilities.clone()
        original_Z = branch.last_latent_states.clone()
        original_delta_h = (
            branch.last_long_memory_slopes.clone()
            if dynamic_slope else None
        )
        original_M = branch.last_latent_memories.clone() if latent_memory else None
        original_memory_bias = (
            branch.last_regime_memory_biases.clone()
            if branch.use_regime_relative_memory else None
        )
        if balanced:
            branch.readout(
                original_H[:, split - 1], original_Z[:, split - 1]
            )
            original_h_long_boundary = branch.last_h_long.clone()
            original_h_micro_boundary = branch.last_h_micro.clone()
            original_temporal_boundary = branch.last_h_temporal.clone()
        perturbed = x_batch.clone()
        perturbed[:, split:] = torch.randn_like(perturbed[:, split:])
        branch(perturbed)
        if balanced:
            branch.readout(
                branch.last_long_memory[:, split - 1],
                branch.last_latent_states[:, split - 1],
            )
            balanced_causal_text = (
                f" h_long@t10={(original_h_long_boundary - branch.last_h_long).abs().max().item():.3e}"
                f" h_micro@t10={(original_h_micro_boundary - branch.last_h_micro).abs().max().item():.3e}"
                f" h_temporal@t10={(original_temporal_boundary - branch.last_h_temporal).abs().max().item():.3e}"
            )
        else:
            balanced_causal_text = ""
        memory_causal_text = (
            f" M diff={(original_M[:, :split] - branch.last_latent_memories[:, :split]).abs().max().item():.3e}"
            if latent_memory else ""
        )
        regime_bias_causal_text = (
            f" RPE bias diff={(original_memory_bias[:, :split] - branch.last_regime_memory_biases[:, :split]).abs().max().item():.3e}"
            if branch.use_regime_relative_memory else ""
        )
        slope_causal_text = (
            f" delta_H diff={(original_delta_h[:, :split] - branch.last_long_memory_slopes[:, :split]).abs().max().item():.3e}"
            if dynamic_slope else ""
        )
        print(
            f"  [{label} causality] E diff through t=10="
            f"{(original_E[:, :split] - branch.last_market_tokens[:, :split]).abs().max().item():.3e} "
            f"H diff="
            f"{(original_H[:, :split] - branch.last_long_memory[:, :split]).abs().max().item():.3e} "
            f"p diff={(original_p[:, :split] - branch.last_regime_probabilities[:, :split]).abs().max().item():.3e} "
            f"Z diff={(original_Z[:, :split] - branch.last_latent_states[:, :split]).abs().max().item():.3e}"
            f"{memory_causal_text}"
            f"{regime_bias_causal_text}"
            f"{slope_causal_text}"
            f"{balanced_causal_text}"
        )

        temporal_batch = branch(x_batch)
        E_batch = branch.last_market_tokens.clone()
        H_batch = branch.last_long_memory.clone()
        p_batch = branch.last_regime_probabilities.clone()
        Z_batch = branch.last_latent_states.clone()
        delta_h_batch = (
            branch.last_long_memory_slopes.clone()
            if dynamic_slope else None
        )
        M_batch = branch.last_latent_memories.clone() if latent_memory else None
        bias_batch = (
            branch.last_regime_memory_biases.clone()
            if branch.use_regime_relative_memory else None
        )
        h_long_batch = branch.last_h_long.clone() if balanced else None
        h_micro_batch = branch.last_h_micro.clone() if balanced else None
        temporal_single = branch(x_batch[:1])
        print(
            f"  [{label} batch independence] E="
            f"{(E_batch[:1] - branch.last_market_tokens).abs().max().item():.3e} "
            f"H={(H_batch[:1] - branch.last_long_memory).abs().max().item():.3e} "
            f"p={(p_batch[:1] - branch.last_regime_probabilities).abs().max().item():.3e} "
            f"Z={(Z_batch[:1] - branch.last_latent_states).abs().max().item():.3e} "
            + (
                f"M={(M_batch[:1] - branch.last_latent_memories).abs().max().item():.3e} "
                if latent_memory else ""
            )
            + (
                f"RPE_bias={(bias_batch[:1] - branch.last_regime_memory_biases).abs().max().item():.3e} "
                if branch.use_regime_relative_memory else ""
            )
            + (
                f"delta_H={(delta_h_batch[:1] - branch.last_long_memory_slopes).abs().max().item():.3e} "
                if dynamic_slope else ""
            )
            + (
                f"h_long={(h_long_batch[:1] - branch.last_h_long).abs().max().item():.3e} "
                f"h_micro={(h_micro_batch[:1] - branch.last_h_micro).abs().max().item():.3e} "
                if balanced else ""
            )
            + f"h_temporal={(temporal_batch[:1] - temporal_single).abs().max().item():.3e}"
        )

        branch(x_batch)
        E_reference = branch.last_market_tokens.clone()
        H_reference = branch.last_long_memory.clone()
        p_reference = branch.last_regime_probabilities.clone()
        Z_reference = branch.last_latent_states.clone()
        delta_h_reference = (
            branch.last_long_memory_slopes.clone()
            if dynamic_slope else None
        )
        M_reference = branch.last_latent_memories.clone() if latent_memory else None
        bias_reference = (
            branch.last_regime_memory_biases.clone()
            if branch.use_regime_relative_memory else None
        )
        h_long_reference = branch.last_h_long.clone() if balanced else None
        h_micro_reference = branch.last_h_micro.clone() if balanced else None
        temporal_reference = branch.last_h_temporal.clone()
        offsets = {
            "stock": (0, branch.market_encoder.market_sizes["stock"]),
            "bond": (
                branch.market_encoder.market_sizes["stock"],
                branch.market_encoder.market_sizes["stock"]
                + branch.market_encoder.market_sizes["bond"],
            ),
            "commodity": (
                branch.market_encoder.market_sizes["stock"]
                + branch.market_encoder.market_sizes["bond"],
                sum(branch.market_encoder.market_sizes.values()),
            ),
        }
        for market_name, (start, end) in offsets.items():
            permuted = x_batch.clone()
            order = torch.randperm(end - start, device=device)
            permuted[:, :, start:end] = x_batch[:, :, start:end].index_select(
                2, order
            )
            temporal_permuted = branch(permuted)
            print(
                f"  [{label} {market_name} permutation] E="
                f"{(E_reference - branch.last_market_tokens).abs().max().item():.3e} "
                f"H={(H_reference - branch.last_long_memory).abs().max().item():.3e} "
                f"p={(p_reference - branch.last_regime_probabilities).abs().max().item():.3e} "
                f"Z={(Z_reference - branch.last_latent_states).abs().max().item():.3e} "
                + (
                    f"M={(M_reference - branch.last_latent_memories).abs().max().item():.3e} "
                    if latent_memory else ""
                )
                + (
                    f"RPE_bias={(bias_reference - branch.last_regime_memory_biases).abs().max().item():.3e} "
                    if branch.use_regime_relative_memory else ""
                )
                + (
                    f"delta_H={(delta_h_reference - branch.last_long_memory_slopes).abs().max().item():.3e} "
                    if dynamic_slope else ""
                )
                + (
                    f"h_long={(h_long_reference - branch.last_h_long).abs().max().item():.3e} "
                    f"h_micro={(h_micro_reference - branch.last_h_micro).abs().max().item():.3e} "
                    if balanced else ""
                )
                + f"h_temporal={(temporal_reference - temporal_permuted).abs().max().item():.3e}"
            )

    return {
        "posterior_entropy": posterior_entropy.mean().item(),
        "transition_drift": transition_drift,
        "mean_candidate_l1": float(np.mean(pairwise_l1_values)),
        "mean_candidate_cosine": float(np.mean(pairwise_cosine_values)),
        "mean_z_norm": z_norm.mean().item(),
        "max_z_norm": z_norm.max().item(),
        "max_forced_z_diff": max(forced_z_diffs),
        "max_forced_prediction_diff": max(forced_prediction_diffs),
        "zero_z_prediction_diff": zero_z_prediction_diff.max().item(),
        "zero_h_prediction_diff": zero_h_prediction_diff.max().item(),
        "zero_z_prediction_mean_diff": zero_z_prediction_diff.mean().item(),
        "zero_h_prediction_mean_diff": zero_h_prediction_diff.mean().item(),
        "readout_wz_wh_ratio": readout_weight_ratio,
        "post_balance_micro_long_ratio": post_balance_ratio,
        "micro_long_prediction_impact_ratio": prediction_impact_ratio,
        "mean_consecutive_z_cosine": consecutive_z_cosine.mean().item(),
        "mean_z_directional_change": (1.0 - consecutive_z_cosine).mean().item(),
        "zero_memory_prediction_mean_diff": (
            zero_memory_prediction_diff.mean().item()
            if zero_memory_prediction_diff is not None else float("nan")
        ),
        "memory_projection_norm": (
            float(np.mean([
                projection.weight.detach().norm().item()
                for projection in branch.memory_projections
            ]))
            if latent_memory else float("nan")
        ),
        "memory_tv_to_uniform": memory_tv_to_uniform,
        "memory_injection_ratio": memory_injection_ratio,
        "prediction_gradient_norms": prediction_gradient_norms,
        "regime_rpe_norm": regime_rpe_norm,
        "mean_regime_rpe_distance": mean_regime_rpe_distance,
        "regime_bias_abs_mean": regime_bias_abs_mean,
        "zero_regime_rpe_prediction_mean_diff": (
            zero_regime_rpe_prediction_mean_diff
        ),
        "slope_base_ratio_mean": slope_base_ratio_mean,
        "mean_slope_pairwise_l1": mean_slope_pairwise_l1,
        "zero_slope_prediction_mean_diff": zero_slope_prediction_mean_diff,
        "flipped_slope_prediction_mean_diff": (
            flipped_slope_prediction_mean_diff
        ),
        "slope_regime_interaction_std": (
            float(np.std(slope_regime_interactions))
            if slope_regime_interactions else float("nan")
        ),
    }


def _print_split_router_diagnostics(model, loaders, device, label):
    rd = model.regime_dynamic
    mean_probabilities = {}
    occupancies = {}
    model.eval()
    for split_name, loader in loaders.items():
        probability_batches = []
        with torch.no_grad():
            for x_batch, _, descriptor in loader:
                model(x_batch.to(device), market_descriptor=descriptor.to(device))
                probability_batches.append(rd.last_regime_p.cpu())
        p_raw = torch.cat(probability_batches).numpy()
        entropy = -(p_raw * np.log(p_raw + 1e-9)).sum(axis=-1)
        occupancy = p_raw.argmax(axis=-1)
        mean_probabilities[split_name] = p_raw.mean(axis=(0, 1))
        occupancies[split_name] = np.array([
            (occupancy == state).mean() for state in range(p_raw.shape[-1])
        ])
        prefix = f"{label} {split_name.upper()} router"
        print(f"  [{prefix}] mean p={np.round(mean_probabilities[split_name], 4)}")
        print(
            f"  [{prefix}] entropy={entropy.mean():.6f} "
            f"mean max p={p_raw.max(axis=-1).mean():.6f}"
        )
        print(f"  [{prefix}] argmax occupancy")
        for state in range(p_raw.shape[-1]):
            print(f"    state {state} = {occupancies[split_name][state] * 100:.2f}%")

        timeline = np.asarray(loader.dataset.market_descriptors)
        for index, descriptor_name in enumerate(("abs_return", "volatility", "slope")):
            values = timeline[:, index]
            p01, p50, p99 = np.percentile(values, (1, 50, 99))
            print(
                f"  [{label} {split_name.upper()} descriptor {descriptor_name}] "
                f"mean={values.mean():+.6f} std={values.std():.6f} "
                f"min={values.min():+.6f} p01={p01:+.6f} p50={p50:+.6f} "
                f"p99={p99:+.6f} max={values.max():+.6f}"
            )

    train_val_shift = np.abs(
        mean_probabilities["train"] - mean_probabilities["val"]
    ).sum()
    train_test_shift = np.abs(
        mean_probabilities["train"] - mean_probabilities["test"]
    ).sum()
    occupancy_shift = np.abs(
        occupancies["train"] - occupancies["test"]
    ).sum()
    print(f"  [{label} split shift] train-val mean-p L1 shift={train_val_shift:.6f}")
    print(f"  [{label} split shift] train-test mean-p L1 shift={train_test_shift:.6f}")
    print(
        f"  [{label} split shift] train-test occupancy L1 shift="
        f"{occupancy_shift:.6f}"
    )


def _semantic_router_diagnostics(model, test_loader, device, label, all_loaders=None):
    rd = model.regime_dynamic
    router = rd.regime_gen
    probability_batches, effective_probability_batches, descriptor_batches = [], [], []
    score_batches = {name: [] for name in ("context", "semantic", "transition", "final")}

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x_batch, _, descriptor = batch
            model(
                x_batch.to(device),
                market_descriptor=descriptor.to(device),
            )
            probability_batches.append(rd.last_regime_p.cpu())
            effective_probability_batches.append(rd.last_regime_p_use.cpu())
            descriptor_batches.append(descriptor.cpu())
            score_batches["context"].append(router.last_context_score.cpu())
            score_batches["semantic"].append(router.last_semantic_score.cpu())
            score_batches["transition"].append(router.last_transition_score.cpu())
            score_batches["final"].append(router.last_final_score.cpu())

    p = torch.cat(probability_batches).numpy()
    p_use = torch.cat(effective_probability_batches).numpy()
    descriptor = torch.cat(descriptor_batches).numpy()
    scores = {name: torch.cat(parts).numpy() for name, parts in score_batches.items()}
    for name in ("context", "semantic", "transition", "final"):
        print(
            f"  [{label} router score] {name}_score mean={scores[name].mean():+.6f} "
            f"std={scores[name].std():.6f} min={scores[name].min():+.6f} "
            f"max={scores[name].max():+.6f}"
        )
    for name in ("context", "semantic", "transition", "final"):
        print(
            f"  [{label} router score] mean absolute {name}_score="
            f"{np.abs(scores[name]).mean():.6f}"
        )
    if label in ("I", "J", "K", "L"):
        print(f"  [{label} context calibration] gamma={router.gamma:.1f}")
        within_ranges = {}
        for name in ("context", "semantic", "transition", "final"):
            within_range = scores[name].max(axis=-1) - scores[name].min(axis=-1)
            within_std = scores[name].std(axis=-1)
            within_ranges[name] = within_range.mean()
            print(
                f"  [{label} router score] {name} within-K range "
                f"mean={within_range.mean():.6f} std={within_range.std():.6f}"
            )
            print(
                f"  [{label} router score] {name} within-K std="
                f"{within_std.mean():.6f}"
            )
        eps = 1e-8
        print(
            f"  [{label} score dominance] context/semantic range ratio="
            f"{within_ranges['context'] / (within_ranges['semantic'] + eps):.6f}"
        )
        print(
            f"  [{label} score dominance] context/transition range ratio="
            f"{within_ranges['context'] / (within_ranges['transition'] + eps):.6f}"
        )
        print(
            f"  [{label} score dominance] semantic/transition range ratio="
            f"{within_ranges['semantic'] / (within_ranges['transition'] + eps):.6f}"
        )

    entropy = -(p * np.log(p + 1e-9)).sum(axis=-1)
    transition = np.abs(np.diff(p, axis=1)).sum(axis=-1)
    probability_label = (
        f"{label} p_raw" if label in ("I", "J", "K", "L") else f"{label} p"
    )
    print(
        f"  [{probability_label}] mean={np.round(p.mean(axis=(0, 1)), 4)} "
        f"std={p.std():.6f} min={p.min():.6f} max={p.max():.6f}"
    )
    print(f"  [{probability_label}] per-sample p std (B,T,K)={p.std(axis=(1, 2)).mean():.6f}")
    print(f"  [{probability_label}] mean max regime probability={p.max(axis=-1).mean():.6f}")
    print(f"  [{probability_label}] entropy mean={entropy.mean():.6f} std={entropy.std():.6f}")
    print(f"  [{probability_label}] mean ||p_t-p_(t-1)||_1={transition.mean():.6f}")
    for step in range(0, min(p.shape[1], 20), 2):
        print(f"  [{probability_label} t={step}] {np.round(p[:, step].mean(axis=0), 4)}")

    if label in ("I", "J", "K", "L"):
        entropy_use = -(p_use * np.log(p_use + 1e-9)).sum(axis=-1)
        print(
            f"  [{label} p_use] mean={np.round(p_use.mean(axis=(0, 1)), 4)} "
            f"std={p_use.std():.6f} min={p_use.min():.6f} max={p_use.max():.6f}"
        )
        print(f"  [{label} p_use] entropy mean={entropy_use.mean():.6f} std={entropy_use.std():.6f}")
        print(f"  [{label} p_use] mean max regime probability={p_use.max(axis=-1).mean():.6f}")
        raw_example = p[0, 0]
        use_example = p_use[0, 0]
        expected_use = (1.0 - rd.routing_strength) / p.shape[-1] + rd.routing_strength * raw_example
        print(f"  [{label} routing strength] rho={rd.routing_strength:.2f}")
        print(
            f"  [{label} routing strength] p_raw={np.round(raw_example, 6)} "
            f"sum={raw_example.sum():.8f}"
        )
        print(
            f"  [{label} routing strength] p_use={np.round(use_example, 6)} "
            f"sum={use_example.sum():.8f} formula_max_error="
            f"{np.abs(use_example - expected_use).max():.3e}"
        )

    p_last = p[:, -1]
    m_last = descriptor[:, -1]
    hard_state = p.argmax(axis=-1)
    last_hard_state = p_last.argmax(axis=-1)
    print(f"  [{label} all-time argmax occupancy]")
    for state in range(p.shape[-1]):
        print(f"    state {state} = {(hard_state == state).mean() * 100:.2f}%")
    print(f"  [{label} last-step argmax occupancy]")
    for state in range(p.shape[-1]):
        print(f"    state {state} = {(last_hard_state == state).mean() * 100:.2f}%")

    event_label = f"{label} p_raw" if label in ("I", "J", "K", "L") else label
    _event_test(event_label, "volatility", m_last[:, 1], p_last)
    _event_test(event_label, "abs_return", m_last[:, 0], p_last)
    if label in ("I", "J", "K", "L"):
        p_use_last = p_use[:, -1]
        _event_test(f"{label} p_use", "volatility", m_last[:, 1], p_use_last)
        _event_test(f"{label} p_use", "abs_return", m_last[:, 0], p_use_last)

    if label in ("I", "J", "K", "L") and all_loaders is not None:
        _print_split_router_diagnostics(model, all_loaders, device, label)

    centers = rd.state_centers.detach().cpu().numpy()
    for state, center in enumerate(centers):
        print(f"  [{label} center_{state}] {np.round(center, 4)}")
    print(
        f"  [{label} center distance] "
        f"d01={np.linalg.norm(centers[0] - centers[1]):.6f} "
        f"d02={np.linalg.norm(centers[0] - centers[2]):.6f} "
        f"d12={np.linalg.norm(centers[1] - centers[2]):.6f}"
    )
    semantic_means = []
    for state in range(p_last.shape[1]):
        weights = p_last[:, state]
        weighted = (weights[:, None] * m_last).sum(axis=0) / (weights.sum() + 1e-8)
        semantic_means.append(weighted)
        print(
            f"  [{label} state {state}] abs_ret={weighted[0]:+.6f} "
            f"vol={weighted[1]:+.6f} slope={weighted[2]:+.6f}"
        )
    semantic_means = np.stack(semantic_means)
    separation = np.linalg.norm(
        semantic_means[:, None] - semantic_means[None, :], axis=-1
    )
    if separation[np.triu_indices(3, 1)].max() < 1e-3:
        print(f"  [{label} semantics] semantic separation failed")

    # Gradient diagnostic uses the selected G/H/I variant's exact training loss.
    x_batch, y_batch, descriptor_batch = next(iter(test_loader))
    x_batch = x_batch[:16].to(device)
    y_batch = y_batch[:16].to(device)
    descriptor_batch = descriptor_batch[:16].to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(x_batch, market_descriptor=descriptor_batch)
    return_loss = _prediction_loss(prediction, y_batch, make_loss())
    auxiliary_loss = rd.dynamic_loss()
    diagnostic_loss = return_loss + auxiliary_loss
    router_prediction_parameters = [
        router.regime_prototypes,
        *router.context_encoder.parameters(),
        router.transition_matrix,
        rd.state_centers,
    ]
    prediction_router_gradients = torch.autograd.grad(
        return_loss,
        router_prediction_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    prediction_router_grad_norm = sum(
        gradient.square().sum()
        for gradient in prediction_router_gradients
        if gradient is not None
    ).sqrt().item()
    diagnostic_loss.backward()
    components = rd.last_loss_components
    weights = rd.last_loss_weights
    weighted_cluster = weights['cluster'] * components['cluster'].item()
    weighted_info_max = weights['info_max'] * components['info_max'].item()
    diversity_key = "orth" if rd.orthogonal_dynamic else "dynamic"
    weighted_dynamic = weights[diversity_key] * components[diversity_key].item()
    weighted_aux_total = weighted_cluster + weighted_info_max + weighted_dynamic
    return_magnitude = return_loss.item()
    eps = 1e-8
    print(f"  [{label} loss/raw] L_return={return_magnitude:.6e}")
    print(f"  [{label} loss/raw] L_cluster={components['cluster'].item():.6e}")
    print(f"  [{label} loss/raw] L_InfoMax={components['info_max'].item():.6e}")
    diversity_name = "L_orth" if label == "L" else "L_dynamic"
    weighted_diversity_name = "orth" if label == "L" else "dynamic"
    print(
        f"  [{label} loss/raw] {diversity_name}="
        f"{components['dynamic'].item():.6e}"
    )
    print(f"  [{label} loss/weighted] cluster={weighted_cluster:.6e}")
    print(f"  [{label} loss/weighted] InfoMax={weighted_info_max:.6e}")
    print(
        f"  [{label} loss/weighted] {weighted_diversity_name}="
        f"{weighted_dynamic:.6e}"
    )
    print(f"  [{label} loss/weighted] aux_total={weighted_aux_total:.6e}")
    print(f"  [{label} loss/weighted] total={diagnostic_loss.item():.6e}")
    print(
        f"  [{label} loss/ratio] cluster_to_return_ratio="
        f"{weighted_cluster / (return_magnitude + eps):.6e}"
    )
    print(
        f"  [{label} loss/ratio] aux_to_return_ratio="
        f"{abs(weighted_aux_total) / (return_magnitude + eps):.6e}"
    )
    print(
        f"  [{label} grad] prediction-path Router aggregate="
        f"{prediction_router_grad_norm:.3e}"
    )

    if label == "L":
        adapter_outputs = rd.last_zs
        flattened = torch.nn.functional.normalize(
            adapter_outputs.reshape(adapter_outputs.shape[0], -1), dim=-1
        )
        cosine_matrix = flattened @ flattened.T
        pairwise_values = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            value = cosine_matrix[left, right].item()
            pairwise_values.append(value)
            print(f"  [L adapter cosine] cos(z{left + 1},z{right + 1})={value:+.6f}")
        pairwise_values = np.asarray(pairwise_values)
        print(
            f"  [L adapter cosine] mean pairwise cosine="
            f"{pairwise_values.mean():+.6f}"
        )
        print(
            f"  [L adapter cosine] mean pairwise cosine^2="
            f"{np.square(pairwise_values).mean():.6f}"
        )
        adapter_mean_norm = adapter_outputs.norm(dim=-1).mean().item()
        uniform_mixed_norm = adapter_outputs.mean(dim=0).norm(dim=-1).mean().item()
        cancellation_ratio = uniform_mixed_norm / (adapter_mean_norm + 1e-8)
        print(f"  [L adapter cancellation] adapter mean norm={adapter_mean_norm:.6e}")
        print(f"  [L adapter cancellation] uniform mixed norm={uniform_mixed_norm:.6e}")
        print(f"  [L adapter cancellation] cancellation ratio={cancellation_ratio:.6f}")
    gradients = [
        ("prototypes", router.regime_prototypes),
        ("context_encoder", router.context_encoder[0].weight),
        ("transition_matrix", router.transition_matrix),
        ("state_centers", rd.state_centers),
        ("adapter1", rd.adapters.adapters[0][0].weight),
        ("adapter2", rd.adapters.adapters[1][0].weight),
        ("adapter3", rd.adapters.adapters[2][0].weight),
    ]
    for name, parameter in gradients:
        norm = parameter.grad.norm().item() if parameter.grad is not None else float("nan")
        print(f"  [{label} grad] {name}={norm:.3e}")

    if label in ("I", "J", "K", "L"):
        print(f"  [{label} temporal norm] E={rd.last_E.norm(dim=-1).mean().item():.6e}")
        for state in range(rd.last_zs.shape[0]):
            value = rd.last_zs[state].norm(dim=-1).mean().item()
            print(f"  [{label} temporal norm] adapter{state + 1}(E)={value:.6e}")
        print(f"  [{label} temporal norm] z_mixed={rd.last_z.norm(dim=-1).mean().item():.6e}")
        print(
            f"  [{label} temporal norm] transformer_output="
            f"{rd.last_transformer_output.norm(dim=-1).mean().item():.6e}"
        )
        print(
            f"  [{label} temporal norm] h_temporal="
            f"{rd.last_h_temporal.norm(dim=-1).mean().item():.6e}"
        )

    # Forced-state comparison uses the complete temporal and prediction paths.
    model.eval()
    with torch.no_grad():
        rd.forced_regime = None
        rd.forced_regime_use = None
        prediction_real = model(x_batch, market_descriptor=descriptor_batch)
        temporal_real = rd.last_h_temporal.clone()
        for state in range(3):
            forced = torch.zeros(3, device=device)
            forced[state] = 1.0
            if label in ("I", "J", "K", "L"):
                rd.forced_regime_use = forced
            else:
                rd.forced_regime = forced
            prediction_forced = model(x_batch, market_descriptor=descriptor_batch)
            temporal_forced = rd.last_h_temporal
            forced_label = (
                "forced downstream p_use"
                if label in ("I", "J", "K", "L")
                else "forced state"
            )
            print(
                f"  [{label} {forced_label} {state}] max_abs_diff h_temporal="
                f"{(temporal_forced - temporal_real).abs().max().item():.6e} "
                f"prediction={(prediction_forced - prediction_real).abs().max().item():.6e}"
            )
        rd.forced_regime = None
        rd.forced_regime_use = None

        if label in ("I", "J", "K", "L"):
            for state in range(3):
                forced = torch.zeros(3, device=device)
                forced[state] = 1.0
                rd.forced_regime = forced
                prediction_forced = model(x_batch, market_descriptor=descriptor_batch)
                temporal_forced = rd.last_h_temporal
                effective = rd.last_regime_p_use[0, 0].cpu().numpy()
                print(
                    f"  [{label} forced raw-state-smoothed {state}] p_use="
                    f"{np.round(effective, 4)} max_abs_diff h_temporal="
                    f"{(temporal_forced - temporal_real).abs().max().item():.6e} "
                    f"prediction={(prediction_forced - prediction_real).abs().max().item():.6e}"
                )
            rd.forced_regime = None

        # Both neural input and descriptor are perturbed only at t >= 10.
        split = min(10, x_batch.size(1))
        x_perturbed = x_batch.clone()
        descriptor_perturbed = descriptor_batch.clone()
        x_perturbed[:, split:] = torch.randn_like(x_perturbed[:, split:])
        descriptor_perturbed[:, split:] = torch.randn_like(
            descriptor_perturbed[:, split:]
        )
        encoded = rd.proj(x_batch.reshape(x_batch.size(0), x_batch.size(1), -1))
        encoded_perturbed = rd.proj(
            x_perturbed.reshape(x_perturbed.size(0), x_perturbed.size(1), -1)
        )
        p_original = router(encoded, descriptor_batch, rd.state_centers)
        p_perturbed = router(
            encoded_perturbed, descriptor_perturbed, rd.state_centers
        )
        causal_diff = (
            p_original[:, :split] - p_perturbed[:, :split]
        ).abs().max().item()
        uniform_original = torch.full_like(p_original, 1.0 / p_original.shape[-1])
        uniform_perturbed = torch.full_like(p_perturbed, 1.0 / p_perturbed.shape[-1])
        p_use_original = (
            (1.0 - rd.routing_strength) * uniform_original
            + rd.routing_strength * p_original
        )
        p_use_perturbed = (
            (1.0 - rd.routing_strength) * uniform_perturbed
            + rd.routing_strength * p_perturbed
        )
        causal_use_diff = (
            p_use_original[:, :split] - p_use_perturbed[:, :split]
        ).abs().max().item()
    print(f"  [{label} causality] max p_raw diff before t=10={causal_diff:.3e}")
    if label in ("I", "J", "K", "L"):
        print(f"  [{label} causality] max p_use diff before t=10={causal_use_diff:.3e}")
    print("  [RPE delta] implementation uses query index i minus key index j")


def print_diagnostics(model, variant, loaders, device):
    test_loader = loaders["test"]
    batch = next(iter(test_loader))
    x_batch = batch[0][:16].to(device)
    descriptor = batch[2][:16].to(device) if len(batch) == 3 else None
    model.eval()
    with torch.no_grad():
        model(x_batch, market_descriptor=descriptor)

    alpha = model.last_alpha
    print(
        f"  Temporal weights: mean={alpha.mean().item():.4f} "
        f"std={alpha.std().item():.4f} min={alpha.min().item():.4f} "
        f"max={alpha.max().item():.4f}"
    )
    if variant in ("switching_transformer", "switching_null_control"):
        return _switching_transformer_diagnostics(model, loaders, device)
    elif variant == "switching_filter_rpe":
        return _switching_filter_rpe_diagnostics(model, loaders, device)
    elif variant in (
        "switching_latent_transformer",
        "switching_latent_balanced_readout",
        "switching_latent_dynamic_slope",
        "switching_latent_memory",
        "switching_active_latent_memory",
        "switching_regime_relative_latent_memory",
    ):
        return _switching_latent_diagnostics(model, loaders, device)
    elif variant in ("market_token_transformer", "market_dispersion_transformer"):
        _market_token_diagnostics(model, test_loader, device)
    elif variant in (
        "semantic_router", "loss_rebalance", "routing_strength", "context_calibrated",
        "context_balance", "adapter_orthogonal"
    ):
        labels = {
            "semantic_router": "G",
            "loss_rebalance": "H",
            "routing_strength": "I",
            "context_calibrated": "J",
            "context_balance": "K",
            "adapter_orthogonal": "L",
        }
        label = labels[variant]
        _semantic_router_diagnostics(
            model,
            test_loader,
            device,
            label,
            all_loaders=(
                loaders
                if variant in (
                    "routing_strength", "context_calibrated", "context_balance",
                    "adapter_orthogonal"
                )
                else None
            ),
        )
    elif variant.startswith("regime_dynamic"):
        probabilities = model.regime_dynamic.last_regime_p
        mean_probability = probabilities.mean(dim=(0, 1)).cpu().numpy()
        entropy = -(probabilities * (probabilities + 1e-9).log()).sum(dim=-1).mean()
        print(
            f"  Regime probabilities: {np.round(mean_probability, 4)}; "
            f"entropy={entropy.item():.4f}"
        )
        if variant == "regime_dynamic_semantic":
            centers = model.regime_dynamic.state_centers.detach().cpu().numpy()
            print(f"  Semantic state centers:\n{np.round(centers, 4)}")


def _active_memory_checkpoint_diagnostic(model, loader, device, stage):
    """Small fixed-batch probe; it never updates parameters or optimizer state."""
    branch = model.switching_latent_transformer
    label = "D1" if branch.use_regime_relative_memory else "D1A2"
    was_training = model.training
    model.eval()
    batch = next(iter(loader))
    x_batch, y_batch = batch[:2]
    x_batch = x_batch[:16].to(device)
    y_batch = y_batch[:16].to(device)
    prediction = model(x_batch)
    return_loss = _prediction_loss(prediction, y_batch, make_loss())
    projection_modules = {
        name: getattr(branch.latent_memory_attention, name)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj")
    }
    projection_modules.update({
        f"memory_proj_{state}": projection
        for state, projection in enumerate(branch.memory_projections)
    })
    parameters = [
        parameter
        for module in projection_modules.values()
        for parameter in module.parameters()
    ]
    if branch.use_regime_relative_memory:
        parameters.append(branch.regime_relative_lag_bias)
    gradients = torch.autograd.grad(
        return_loss, parameters, allow_unused=True
    )
    offset = 0
    gradient_norms = {}
    for name, module in projection_modules.items():
        module_parameters = list(module.parameters())
        selected = gradients[offset:offset + len(module_parameters)]
        offset += len(module_parameters)
        gradient_norms[name] = _aggregate_gradient_norm(
            module_parameters, selected
        )
    if branch.use_regime_relative_memory:
        rpe_gradient = gradients[-1]
        gradient_norms["regime_rpe"] = (
            rpe_gradient.norm().item() if rpe_gradient is not None else 0.0
        )

    base_norm = branch.last_generator_base_preactivations.norm(dim=-1)
    injection_norm = branch.last_generator_memory_injections.norm(dim=-1)
    injection_ratios = [
        (
            injection_norm[:, :, state].mean()
            / (base_norm[:, :, state].mean() + 1e-8)
        ).item()
        for state in range(branch.K)
    ]
    for state in range(branch.K):
        print(
            f"  [{label} {stage} memory injection state {state}] base norm="
            f"{base_norm[:, :, state].mean().item():.6e} memory norm="
            f"{injection_norm[:, :, state].mean().item():.6e} "
            f"memory/base={injection_ratios[state]:.6e}"
        )
    attention = branch.last_latent_memory_attention
    regime_bias_snapshot = (
        branch.last_regime_memory_biases.clone()
        if branch.use_regime_relative_memory else None
    )
    content_snapshot = (
        branch.last_latent_memory_content_scores.clone()
        if branch.use_regime_relative_memory else None
    )
    normalized_entropy = []
    for step in range(2, attention.shape[1]):
        weights = attention[:, step, :, :step]
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)
        normalized_entropy.append(entropy / np.log(step))
    normalized_entropy_value = torch.cat([
        value.reshape(-1) for value in normalized_entropy
    ]).mean().item()
    parameter_norms = {
        name: float(sum(
            parameter.detach().square().sum().item()
            for parameter in module.parameters()
        ) ** 0.5)
        for name, module in projection_modules.items()
    }
    with torch.no_grad():
        real_prediction = model(x_batch)
        zero_prediction = model._switching_latent_transformer_forward(
            x_batch, zero_latent_memory=True
        )
        zero_memory_diff = (
            zero_prediction - real_prediction
        ).abs().mean().item()
    print(
        f"  [{label} {stage} memory] injection/base="
        f"{np.round(injection_ratios, 8)} normalized entropy="
        f"{normalized_entropy_value:.6f} memory_proj mean norm="
        f"{np.mean([parameter_norms[f'memory_proj_{state}'] for state in range(branch.K)]):.6e} "
        f"zero-memory prediction mean diff={zero_memory_diff:.6e}"
    )
    print(
        f"  [{label} {stage} prediction gradient] "
        + " ".join(
            f"{name}={gradient_norms[name]:.3e}"
            for name in projection_modules
        )
    )
    print(
        f"  [{label} {stage} parameter norm] "
        + " ".join(
            f"{name}={parameter_norms[name]:.6e}"
            for name in projection_modules
        )
    )
    if branch.use_regime_relative_memory:
        regime_bias = regime_bias_snapshot
        content = content_snapshot
        valid_bias = []
        valid_content = []
        tv_values = []
        for step in range(2, attention.shape[1]):
            valid_bias.append(regime_bias[:, step, :, :step].reshape(-1))
            valid_content.append(content[:, step, :, :step].reshape(-1))
            weights = attention[:, step, :, :step]
            tv_values.append(
                (0.5 * (weights - 1.0 / step).abs().sum(dim=-1)).reshape(-1)
            )
        bias_abs = torch.cat(valid_bias).abs().mean().item()
        content_abs = torch.cat(valid_content).abs().mean().item()
        tv_mean = torch.cat(tv_values).mean().item()
        print(
            f"  [D1 {stage} regime RPE] norm="
            f"{branch.regime_relative_lag_bias.detach().norm().item():.6e} "
            f"prediction grad={gradient_norms['regime_rpe']:.3e} "
            f"mean |bias|={bias_abs:.6e} bias/content="
            f"{bias_abs / (content_abs + 1e-8):.6e} "
            f"attention TV-to-uniform={tv_mean:.6e}"
        )
    if was_training:
        model.train()
    return {
        "injection_ratios": injection_ratios,
        "normalized_entropy": normalized_entropy_value,
        "gradient_norms": gradient_norms,
        "parameter_norms": parameter_norms,
        "zero_memory_prediction_mean_diff": zero_memory_diff,
    }


def run_variant(name, variant, args, device, data):
    set_seed(args.seed)
    n_stock = data["market_indices"]["stock"][1] - data["market_indices"]["stock"][0]
    n_bond = data["market_indices"]["bond"][1] - data["market_indices"]["bond"][0]
    model_kwargs = dict(
        n_stock=n_stock,
        n_bond=n_bond,
        feat_dim=FEATURE_DIM,
    )
    # Keep construction on CPU so S1/S1C can be checked tensor-for-tensor
    # before either model is moved to the selected accelerator.
    model = HeteroMixHopCMGM(
        data["n_nodes"], data["n_commodities"],
        variant=variant, **model_kwargs,
    )
    if variant == "switching_null_control":
        with torch.random.fork_rng():
            torch.manual_seed(args.seed)
            s1_reference = HeteroMixHopCMGM(
                data["n_nodes"], data["n_commodities"],
                variant="switching_transformer", **model_kwargs,
            )
            s1_state = s1_reference.state_dict()
            s1c_state = model.state_dict()
            assert s1_state.keys() == s1c_state.keys()
            for parameter_name in s1_state:
                torch.testing.assert_close(
                    s1_state[parameter_name], s1c_state[parameter_name],
                    rtol=0.0, atol=0.0,
                )

            # Both variants execute the complete regime side branch in train
            # mode and therefore consume its dropout/RNG in the same order.
            sanity_x = torch.randn(
                2, args.seq_len, data["n_nodes"], FEATURE_DIM
            )
            s1_reference.train()
            model.train()
            torch.manual_seed(91017)
            s1_reference.switching_transformer(sanity_x)
            torch.manual_seed(91017)
            model.switching_transformer(sanity_x)
            for left, right in (
                (
                    s1_reference.switching_transformer.last_regime_evidence,
                    model.switching_transformer.last_regime_evidence,
                ),
                (
                    s1_reference.switching_transformer.last_regime_probabilities,
                    model.switching_transformer.last_regime_probabilities,
                ),
                (
                    s1_reference.switching_transformer.last_regime_intervention_real,
                    model.switching_transformer.last_regime_intervention_real,
                ),
            ):
                torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        print("  [S1/S1C shared initialization] PASS")
        print("  [S1/S1C forward RNG sanity] H_reg/p/r_real PASS")
    if variant in (
        "switching_latent_balanced_readout",
        "switching_latent_dynamic_slope",
        "switching_latent_memory",
        "switching_active_latent_memory",
        "switching_regime_relative_latent_memory",
    ):
        with torch.random.fork_rng():
            torch.manual_seed(args.seed)
            reference_variant = (
                "switching_latent_balanced_readout"
                if variant == "switching_latent_dynamic_slope"
                else ("switching_active_latent_memory"
                if variant == "switching_regime_relative_latent_memory"
                else (
                    "switching_latent_memory"
                    if variant == "switching_active_latent_memory"
                    else (
                        "switching_latent_balanced_readout"
                        if variant == "switching_latent_memory"
                        else "switching_latent_transformer"
                    )
                ))
            )
            d0_reference = HeteroMixHopCMGM(
                data["n_nodes"], data["n_commodities"],
                variant=reference_variant, **model_kwargs,
            )
            shared_module_names = [
                "market_encoder", "long_memory", "regime_filter",
                "latent_transition",
            ]
            if variant in (
                "switching_latent_dynamic_slope",
                "switching_latent_memory",
                "switching_active_latent_memory",
                "switching_regime_relative_latent_memory",
            ):
                shared_module_names.extend([
                    "long_memory_readout", "micro_state_readout",
                    "long_memory_norm", "micro_state_norm", "state_readout",
                ])
            if variant in (
                "switching_active_latent_memory",
                "switching_regime_relative_latent_memory",
            ):
                shared_module_names.append("latent_memory_attention")
            for module_name in shared_module_names:
                reference_state = getattr(
                    d0_reference.switching_latent_transformer, module_name
                ).state_dict()
                balanced_state = getattr(
                    model.switching_latent_transformer, module_name
                ).state_dict()
                assert reference_state.keys() == balanced_state.keys()
                for parameter_name in reference_state:
                    torch.testing.assert_close(
                        reference_state[parameter_name],
                        balanced_state[parameter_name],
                        rtol=0.0, atol=0.0,
                    )
            if variant == "switching_latent_dynamic_slope":
                reference_state = d0_reference.state_dict()
                dynamic_state = model.state_dict()
                shared_keys = [
                    key for key in dynamic_state
                    if "slope_projections." not in key
                ]
                assert set(shared_keys) == set(reference_state)
                for parameter_name in shared_keys:
                    torch.testing.assert_close(
                        reference_state[parameter_name],
                        dynamic_state[parameter_name],
                        rtol=0.0, atol=0.0,
                    )
                sanity_x = torch.randn(
                    2, args.seq_len, data["n_nodes"], FEATURE_DIM
                )
                d0_reference.eval()
                model.eval()
                with torch.no_grad():
                    reference_prediction = d0_reference(sanity_x)
                    zero_slope_prediction = (
                        model._switching_latent_transformer_forward(
                            sanity_x, slope_scale=0.0
                        )
                    )
                reference_branch = d0_reference.switching_latent_transformer
                dynamic_branch = model.switching_latent_transformer
                for attribute in (
                    "last_latent_candidates", "last_latent_states",
                    "last_h_temporal",
                ):
                    torch.testing.assert_close(
                        getattr(reference_branch, attribute),
                        getattr(dynamic_branch, attribute),
                        rtol=0.0, atol=1e-7,
                    )
                torch.testing.assert_close(
                    reference_prediction, zero_slope_prediction,
                    rtol=0.0, atol=1e-7,
                )
                print(
                    "  [D0C zero-slope initialization equivalence] "
                    "candidates/Z/h_temporal/prediction PASS"
                )
            if variant in (
                "switching_latent_memory",
                "switching_active_latent_memory",
                "switching_regime_relative_latent_memory",
            ):
                sanity_x = torch.randn(
                    2, args.seq_len, data["n_nodes"], FEATURE_DIM
                )
                memory_branch = model.switching_latent_transformer
                if variant == "switching_latent_memory":
                    d0_reference.eval()
                    model.eval()
                    with torch.no_grad():
                        reference_prediction = d0_reference(sanity_x)
                        memory_prediction = model(sanity_x)
                    reference_branch = d0_reference.switching_latent_transformer
                    z_diff = (
                        reference_branch.last_latent_states
                        - memory_branch.last_latent_states
                    ).abs().max().item()
                    temporal_diff = (
                        reference_branch.last_h_temporal
                        - memory_branch.last_h_temporal
                    ).abs().max().item()
                    prediction_diff = (
                        reference_prediction - memory_prediction
                    ).abs().max().item()
                    torch.testing.assert_close(
                        reference_branch.last_latent_states,
                        memory_branch.last_latent_states,
                        rtol=0.0, atol=1e-7,
                    )
                    torch.testing.assert_close(
                        reference_prediction, memory_prediction,
                        rtol=0.0, atol=1e-7,
                    )
                    print(
                        "  [D1A zero-init equivalence] max Z diff="
                        f"{z_diff:.3e} h_temporal={temporal_diff:.3e} "
                        f"prediction={prediction_diff:.3e}"
                    )
                elif variant == "switching_active_latent_memory":
                    reference_branch = d0_reference.switching_latent_transformer
                    zero_norms = [
                        projection.weight.norm().item()
                        for projection in reference_branch.memory_projections
                    ]
                    active_norms = [
                        projection.weight.norm().item()
                        for projection in memory_branch.memory_projections
                    ]
                    assert max(zero_norms) == 0.0
                    assert min(active_norms) > 0.0
                    print(
                        "  [D1A/D1A2 memory_proj initial norm] D1A="
                        f"{np.round(zero_norms, 6)} D1A2={np.round(active_norms, 6)}"
                    )
                else:
                    reference_branch = d0_reference.switching_latent_transformer
                    model.eval()
                    d0_reference.eval()
                    with torch.no_grad():
                        reference_prediction = d0_reference(sanity_x)
                        d1_prediction = model(sanity_x)
                    d1_branch = model.switching_latent_transformer
                    assert d1_branch.regime_relative_lag_bias.norm().item() == 0.0
                    for attribute in (
                        "last_latent_memories", "last_latent_states",
                        "last_h_temporal",
                    ):
                        torch.testing.assert_close(
                            getattr(reference_branch, attribute),
                            getattr(d1_branch, attribute),
                            rtol=0.0, atol=1e-7,
                        )
                    torch.testing.assert_close(
                        reference_prediction, d1_prediction,
                        rtol=0.0, atol=1e-7,
                    )
                    print(
                        "  [D1 zero-init equivalence] M/Z/h_temporal/prediction PASS"
                    )
                    assert d1_branch.regime_relative_lag_bias.shape == (
                        3, 4, args.seq_len - 1
                    )
                    uniform = torch.full((2, 3), 1.0 / 3.0)
                    uniform_bias = d1_branch.regime_memory_bias(
                        uniform, args.seq_len - 1
                    )
                    torch.testing.assert_close(
                        uniform_bias, torch.zeros_like(uniform_bias),
                        rtol=0.0, atol=1e-7,
                    )
                    with torch.no_grad():
                        pattern = torch.arange(
                            args.seq_len - 1, dtype=torch.float32
                        )
                        d1_branch.regime_relative_lag_bias[0, 0].copy_(pattern)
                    state_zero = torch.tensor([[1.0, 0.0, 0.0]])
                    lag_bias = d1_branch.regime_memory_bias(state_zero, 3)
                    expected = (
                        pattern - pattern / 3.0
                    )[[2, 1, 0]]
                    torch.testing.assert_close(
                        lag_bias[0, 0], expected, rtol=0.0, atol=1e-7
                    )
                    with torch.no_grad():
                        d1_branch.regime_relative_lag_bias.zero_()
                    print(
                        "  [D1 RPE sanity] shape=(3,4,T-1), lag=t-j, "
                        "lag1@index0, uniform-p zero bias PASS"
                    )
                query = torch.randn(3, 128)
                history = torch.randn(3, 9, 64)
                original_memory, _ = memory_branch.latent_memory_attention(
                    query, history
                )
                order = torch.randperm(history.shape[1])
                permuted_memory, _ = memory_branch.latent_memory_attention(
                    query, history[:, order]
                )
                permutation_diff = (
                    original_memory - permuted_memory
                ).abs().max().item()
                torch.testing.assert_close(
                    original_memory, permuted_memory, atol=1e-6, rtol=1e-6
                )
                print(
                    f"  [{'D1' if variant == 'switching_regime_relative_latent_memory' else ('D1A2' if variant == 'switching_active_latent_memory' else 'D1A')} "
                    "LatentMemory history permutation] max M diff="
                    f"{permutation_diff:.3e} PASS"
                )
        comparison = (
            "D0B/D0C" if variant == "switching_latent_dynamic_slope"
            else ("D1A2/D1" if variant == "switching_regime_relative_latent_memory"
            else (
                "D1A/D1A2" if variant == "switching_active_latent_memory"
                else ("D0B/D1A" if variant == "switching_latent_memory" else "D0/D0B")
            ))
        )
        print(f"  [{comparison} shared initialization] PASS")

    model = model.to(device)
    if variant == "switching_null_control":
        model.switching_transformer._initial_parameter_groups = (
            _snapshot_switching_parameter_groups(model.switching_transformer)
        )
    if variant == "switching_filter_rpe":
        model.switching_filter_rpe._initial_transition_logits = (
            model.switching_filter_rpe.regime_inference.transition_logits
            .detach().cpu().clone()
        )
    if variant in (
        "switching_latent_transformer",
        "switching_latent_balanced_readout",
        "switching_latent_dynamic_slope",
        "switching_latent_memory",
        "switching_active_latent_memory",
        "switching_regime_relative_latent_memory",
    ):
        model.switching_latent_transformer._initial_transition_logits = (
            model.switching_latent_transformer.regime_filter.transition_logits
            .detach().cpu().clone()
        )
        if variant == "switching_latent_dynamic_slope":
            branch = model.switching_latent_transformer
            branch._initial_slope_parameters = {
                name: parameter.detach().cpu().clone()
                for name, parameter in branch.named_parameters()
                if name.startswith("slope_projections.")
            }
        if variant in (
            "switching_latent_memory",
            "switching_active_latent_memory",
            "switching_regime_relative_latent_memory",
        ):
            branch = model.switching_latent_transformer
            branch._initial_latent_memory_parameters = {
                name: parameter.detach().cpu().clone()
                for name, parameter in branch.named_parameters()
                if name.startswith("latent_memory_attention.")
                or name.startswith("memory_projections.")
            }
            if branch.use_regime_relative_memory:
                branch._initial_regime_relative_lag_bias = (
                    branch.regime_relative_lag_bias.detach().cpu().clone()
                )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"\n{name} ({variant}) — {parameter_count:,} parameters")
    if variant in ("market_token_transformer", "market_dispersion_transformer"):
        flat_projection_params = data["n_nodes"] * FEATURE_DIM * 128 + 128
        market_encoder_params = sum(
            parameter.numel()
            for parameter in model.market_token_transformer.market_encoder.parameters()
        )
        temporal_branch_params = sum(
            parameter.numel()
            for parameter in model.market_token_transformer.parameters()
        )
        print(
            f"  Flat temporal input projection params={flat_projection_params:,}; "
            f"market-aware input encoder params={market_encoder_params:,}; "
            f"difference={market_encoder_params - flat_projection_params:+,}"
        )
        print(
            f"  {'S0D' if variant == 'market_dispersion_transformer' else 'S0'} "
            f"temporal branch params={temporal_branch_params:,}; "
            f"total params={parameter_count:,}"
        )
        if variant == "market_dispersion_transformer":
            projection_increase = 3 * 32 * 128
            s0_parameter_count = parameter_count - projection_increase
            print(
                f"  S0 total params={s0_parameter_count:,}; "
                f"S0D total params={parameter_count:,}; "
                f"parameter increase={projection_increase:+,} "
                "(daily projection 96→192 input width)"
            )
    elif variant in ("switching_transformer", "switching_null_control"):
        branch = model.switching_transformer
        evidence_params = sum(
            parameter.numel() for parameter in branch.regime_evidence.parameters()
        )
        posterior_params = sum(
            parameter.numel()
            for parameter in branch.regime_inference.posterior_heads.parameters()
        )
        transition_params = branch.regime_inference.transition_logits.numel()
        embedding_params = branch.regime_embeddings.numel()
        increase = (
            evidence_params + posterior_params + transition_params + embedding_params
        )
        route_label = "S1C" if branch.null_control else "S1"
        print(
            f"  S0D total params={parameter_count - increase:,}; "
            f"{route_label} total params={parameter_count:,}; "
            f"{route_label}-S0D increase={increase:+,}"
        )
        if branch.null_control:
            print(
                f"  S1 params={parameter_count:,}; S1C params={parameter_count:,}; "
                "difference=0"
            )
        print(
            f"  {route_label} added params: evidence Transformer={evidence_params:,}; "
            f"posterior heads={posterior_params:,}; transition={transition_params:,}; "
            f"regime embeddings={embedding_params:,}"
        )
    elif variant == "switching_filter_rpe":
        branch = model.switching_filter_rpe
        evidence_params = sum(
            parameter.numel() for parameter in branch.regime_evidence.parameters()
        )
        observation_params = sum(
            parameter.numel()
            for parameter in branch.regime_inference.observation_head.parameters()
        )
        transition_params = branch.regime_inference.transition_logits.numel()
        base_rpe_params = branch.transformer.relative_position.base_rpe.numel()
        regime_rpe_params = branch.transformer.relative_position.regime_rpe.numel()
        s2f_increase = (
            evidence_params + observation_params + transition_params
            + base_rpe_params + regime_rpe_params
        )
        s0d_params = parameter_count - s2f_increase
        s1_posterior_params = 3 * (128 * 3 + 3)
        s1_embedding_params = 3 * 128
        s1_params = (
            s0d_params + evidence_params + s1_posterior_params
            + transition_params + s1_embedding_params
        )
        print(
            f"  S0D params={s0d_params:,}; S1 params={s1_params:,}; "
            f"S2F params={parameter_count:,}"
        )
        print(
            f"  S2F added params: evidence Transformer={evidence_params:,}; "
            f"observation head={observation_params:,}; "
            f"transition={transition_params:,}; base RPE={base_rpe_params:,}; "
            f"regime RPE={regime_rpe_params:,}"
        )
        print(
            f"  S2F-S1 net change={parameter_count - s1_params:+,}; "
            f"removed posterior heads={s1_posterior_params:,} and "
            f"regime embeddings={s1_embedding_params:,}"
        )
    elif variant in (
        "switching_latent_transformer",
        "switching_latent_balanced_readout",
        "switching_latent_dynamic_slope",
        "switching_latent_memory",
        "switching_active_latent_memory",
        "switching_regime_relative_latent_memory",
    ):
        branch = model.switching_latent_transformer
        market_encoder_params = sum(
            parameter.numel() for parameter in branch.market_encoder.parameters()
        )
        long_memory_params = sum(
            parameter.numel() for parameter in branch.long_memory.parameters()
        )
        base_rpe_params = branch.long_memory.base_rpe.numel()
        long_memory_without_rpe = long_memory_params - base_rpe_params
        regime_evidence_params = sum(
            parameter.numel()
            for parameter in branch.regime_filter.regime_evidence.parameters()
        )
        transition_params = branch.regime_filter.transition_logits.numel()
        generator_params = [
            sum(parameter.numel() for parameter in generator.parameters())
            for generator in branch.latent_transition.generators
        ]
        readout_params = sum(
            parameter.numel() for parameter in branch.state_readout.parameters()
        )
        d0_temporal_params = sum(
            parameter.numel() for parameter in branch.parameters()
        )
        # S0D uses the same encoder plus an ordinary two-layer Transformer
        # with attention pooling and a 128->64 output projection.
        s0d_temporal_params = market_encoder_params + 273_345
        s0d_total_params = parameter_count - d0_temporal_params + s0d_temporal_params
        if branch.balanced_readout:
            long_readout_params = sum(
                parameter.numel()
                for parameter in branch.long_memory_readout.parameters()
            )
            micro_readout_params = sum(
                parameter.numel()
                for parameter in branch.micro_state_readout.parameters()
            )
            long_norm_params = sum(
                parameter.numel()
                for parameter in branch.long_memory_norm.parameters()
            )
            micro_norm_params = sum(
                parameter.numel()
                for parameter in branch.micro_state_norm.parameters()
            )
            d0_readout_params = 192 * 64 + 64
            d0b_readout_params = (
                long_readout_params + micro_readout_params
                + long_norm_params + micro_norm_params + readout_params
            )
            latent_memory_params = 0
            if branch.use_latent_memory:
                memory_label = (
                    "D1" if branch.use_regime_relative_memory
                    else ("D1A" if branch.zero_init_memory_projection else "D1A2")
                )
                attention_params = sum(
                    parameter.numel()
                    for parameter in branch.latent_memory_attention.parameters()
                )
                memory_projection_params = [
                    sum(parameter.numel() for parameter in projection.parameters())
                    for projection in branch.memory_projections
                ]
                latent_memory_params = attention_params + sum(
                    memory_projection_params
                )
                regime_rpe_params = (
                    branch.regime_relative_lag_bias.numel()
                    if branch.use_regime_relative_memory else 0
                )
                latent_memory_params += regime_rpe_params
                d0b_total_params = parameter_count - latent_memory_params
                print(
                    f"  D0B params={d0b_total_params:,}; "
                    f"{memory_label} params={parameter_count:,}; "
                    f"increase={latent_memory_params:+,}"
                )
                if not branch.zero_init_memory_projection:
                    print(
                        f"  D1A params={parameter_count:,}; "
                        f"D1A2 params={parameter_count:,}; difference=0"
                    )
                print(
                    f"  {memory_label} added params: LatentMemory q/k/v/out="
                    f"{attention_params:,}; memory_proj_0/1/2="
                    f"{memory_projection_params}"
                )
                if branch.use_regime_relative_memory:
                    print(
                        f"  D1A2 params={parameter_count - regime_rpe_params:,}; "
                        f"D1 params={parameter_count:,}; "
                        f"difference={regime_rpe_params:+,}; "
                        f"regime-relative latent-memory RPE={regime_rpe_params:,}"
                    )
            else:
                if branch.use_dynamic_slope:
                    slope_projection_params = [
                        sum(parameter.numel() for parameter in projection.parameters())
                        for projection in branch.slope_projections
                    ]
                    slope_params = sum(slope_projection_params)
                    print(
                        f"  D0B params={parameter_count - slope_params:,}; "
                        f"D0C params={parameter_count:,}; difference={slope_params:+,}; "
                        f"slope_projection_0/1/2={slope_projection_params}"
                    )
                else:
                    d0_total_params = (
                        parameter_count - d0b_readout_params + d0_readout_params
                    )
                    print(
                        f"  D0 params={d0_total_params:,}; D0B params={parameter_count:,}; "
                        f"difference={parameter_count - d0_total_params:+,}"
                    )
            print(
                f"  D0B readout params: long projection={long_readout_params:,}; "
                f"micro projection={micro_readout_params:,}; "
                f"long LayerNorm={long_norm_params:,}; "
                f"micro LayerNorm={micro_norm_params:,}; "
                f"final state_readout={readout_params:,}"
            )
        else:
            print(
                f"  S0D total params={s0d_total_params:,}; "
                f"D0 total params={parameter_count:,}; "
                f"D0-S0D increase={parameter_count - s0d_total_params:+,}"
            )
        print(
            f"  {('D1' if branch.use_regime_relative_memory else ('D1A' if branch.zero_init_memory_projection else 'D1A2')) if branch.use_latent_memory else ('D0C' if branch.use_dynamic_slope else ('D0B' if branch.balanced_readout else 'D0'))} temporal params: "
            f"Market Encoder={market_encoder_params:,}; "
            f"LongMemory Transformer={long_memory_without_rpe:,}; "
            f"Base RPE={base_rpe_params:,}; regime evidence="
            f"{regime_evidence_params:,}; transition={transition_params:,}; "
            f"G0/G1/G2={generator_params}; state_readout={readout_params:,}"
        )

    started = time.time()
    loaders = (
        data["semantic_loaders"]
        if variant in (
            "semantic_router", "loss_rebalance", "routing_strength",
            "context_calibrated", "context_balance", "adapter_orthogonal"
        )
        else data["loaders"]
    )
    empty_edges = torch.empty(2, 0, dtype=torch.long)
    empty_weights = torch.zeros(0)
    epoch_diagnostic = None
    if variant in (
        "switching_active_latent_memory",
        "switching_regime_relative_latent_memory",
    ):
        # Snapshot one diagnostic batch without advancing the training
        # sampler RNG; repeated probes then reuse this immutable batch.
        with torch.random.fork_rng():
            diagnostic_batch = next(iter(loaders["train"]))
        diagnostic_source = [diagnostic_batch]
        _active_memory_checkpoint_diagnostic(
            model, diagnostic_source, device, "initial"
        )
        epoch_diagnostic = lambda active_model, stage: (
            _active_memory_checkpoint_diagnostic(
                active_model, diagnostic_source, device, stage
            )
        )
    checkpoint_path = _checkpoint_path_for_variant(
        variant, args.checkpoint_dir
    )
    if checkpoint_path is not None:
        print(f"  Best checkpoint path={checkpoint_path.resolve()}")
    history = train(
        model,
        loaders["train"],
        loaders["val"],
        empty_edges,
        empty_weights,
        device,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=(
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        epoch_diagnostic=epoch_diagnostic,
    )
    normalized, original, target = evaluate_primary_horizon(
        model, loaders["test"], data, device
    )
    diagnostics = print_diagnostics(model, variant, loaders, device)

    zero_metrics = compute_metrics(np.zeros_like(target), target)
    versus_zero = (normalized["MAE"] - zero_metrics["MAE"]) / zero_metrics["MAE"] * 100
    elapsed = time.time() - started
    hit_ratio = normalized.get("Hit_Ratio", float("nan"))
    print(
        f"  MAE={normalized['MAE']:.6f} RMSE={normalized['RMSE']:.6f} "
        f"Hit={hit_ratio * 100:.1f}% vs-zero={versus_zero:+.2f}% ({elapsed:.0f}s)"
    )
    return {
        "variant": name,
        "params": parameter_count,
        "time": elapsed,
        "MAE": normalized["MAE"],
        "RMSE": normalized["RMSE"],
        "Hit_Ratio": hit_ratio,
        "vs_zero_pct": versus_zero,
        "mn": normalized,
        "mo": original,
        "diagnostics": diagnostics,
    }


def select_variants(requested):
    if not requested:
        return VARIANTS
    names = {item.strip() for item in requested.split(",") if item.strip()}
    selected = [item for item in VARIANTS if item[0] in names or item[1] in names]
    matched = {value for item in selected for value in item}
    unknown = names - matched
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(sorted(unknown))}")
    return selected


def _print_s1c_control_comparison(results):
    s1c = next(
        (result for result in results if result["variant"] == "S1C-NullSwitchControl"),
        None,
    )
    if s1c is None:
        return

    s0d = {"MAE": 0.023073, "RMSE": 0.029565, "Hit_Ratio": 0.511}
    s1 = {"MAE": 0.022228, "RMSE": 0.028661, "Hit_Ratio": 0.503}
    delta_s1 = s1c["MAE"] - s1["MAE"]
    delta_s0d = s1c["MAE"] - s0d["MAE"]
    print("\nS1C causal-control comparison")
    print(
        f"  S0D: MAE={s0d['MAE']:.6f} RMSE={s0d['RMSE']:.6f} "
        f"Hit={s0d['Hit_Ratio'] * 100:.1f}%"
    )
    print(
        f"  S1:  MAE={s1['MAE']:.6f} RMSE={s1['RMSE']:.6f} "
        f"Hit={s1['Hit_Ratio'] * 100:.1f}%"
    )
    print(
        f"  S1C: MAE={s1c['MAE']:.6f} RMSE={s1c['RMSE']:.6f} "
        f"Hit={s1c['Hit_Ratio'] * 100:.1f}%"
    )
    print(f"  S1C-S1 delta MAE={delta_s1:+.6f}")
    print(f"  S1C-S0D delta MAE={delta_s0d:+.6f}")

    tolerance = 2e-4
    if abs(delta_s1) <= tolerance:
        case = "Case A"
        conclusion = (
            "S1 gain cannot be reliably attributed to switching; the null "
            "control retains essentially the same prediction performance."
        )
    elif abs(delta_s0d) <= tolerance:
        case = "Case B"
        conclusion = (
            "S1C returns to the S0D band; switching/KL may have changed the "
            "optimization trajectory, without establishing regime semantics."
        )
    elif s1c["MAE"] < min(s0d["MAE"], s1["MAE"]) - tolerance:
        case = "Case C"
        conclusion = (
            "The null control outperforms both references; seed-level "
            "stochastic variation must be checked before structural claims."
        )
    elif s1c["MAE"] > s0d["MAE"] + tolerance:
        case = "Case D"
        conclusion = (
            "S1C is worse than S0D; verify shared initialization, RNG order, "
            "forecast initialization, and training settings first."
        )
    else:
        case = "Between reference bands"
        conclusion = (
            "S1C lies between the predefined S1 and S0D equivalence bands; "
            "the control is inconclusive under the stated single-run rule."
        )
    print(f"  [{case}] {conclusion}")


def _print_s2f_conclusion(results):
    result = next(
        (item for item in results if item["variant"] == "S2F-SwitchingFilterRPE"),
        None,
    )
    if result is None or not result.get("diagnostics"):
        return
    diagnostics = result["diagnostics"]
    gradients = diagnostics["prediction_gradient_norms"]
    required_gradient_names = (
        "Regime Evidence Transformer",
        "observation_head",
        "transition_logits",
        "regime_rpe",
    )
    connected = all(gradients[name] > 0.0 for name in required_gradient_names)
    near_uniform = (
        abs(diagnostics["posterior_entropy"] - np.log(3.0)) < 1e-3
        and diagnostics["mean_margin"] < 1e-3
    )
    dynamic = (
        diagnostics["temporal_l1"] > 1e-3
        and diagnostics["transition_drift"] > 0.0
    )
    distinct_rpe = (
        diagnostics["mean_state_rpe_distance"] > 1e-8
        and diagnostics["max_forced_prediction_diff"] > 0.0
        and diagnostics["uniform_regime_bias_max"] < 1e-6
    )
    extreme = (
        diagnostics["posterior_entropy"] < 0.1
        or diagnostics["temporal_l1"] > 0.75
    )
    improves = result["MAE"] < 0.023073
    near_s0d = abs(result["MAE"] - 0.023073) <= 2e-4

    if extreme:
        case = "Case E"
        conclusion = (
            "The posterior is extremely sharp or changes unusually fast; "
            "report prior/evidence/posterior and loss scale without retuning."
        )
    elif improves and near_uniform:
        case = "Case B"
        conclusion = (
            "Prediction improves while filtering remains effectively uniform; "
            "the gain must not be attributed to regime conditioning."
        )
    elif improves and dynamic and connected and distinct_rpe:
        case = "Case A"
        conclusion = (
            "Filtering is dynamic, prediction-connected, and RPE-distinct while "
            "outperforming S0D."
        )
    elif dynamic and connected and distinct_rpe and near_s0d:
        case = "Case C"
        conclusion = (
            "The switching mechanism is identifiable and prediction-connected, "
            "but its forecast accuracy remains near S0D."
        )
    elif near_uniform and connected:
        case = "Case D"
        conclusion = (
            "The regime path receives prediction gradients but the posterior "
            "still collapses near uniform; observation sufficiency is the next question."
        )
    else:
        case = "Mixed"
        conclusion = (
            "The run does not cleanly satisfy one predefined case; interpret the "
            "printed filtering, gradient, RPE, and prediction diagnostics jointly."
        )
    print(f"\nS2F structural conclusion: [{case}] {conclusion}")


def _print_d0_conclusion(results):
    result = next(
        (
            item for item in results
            if item["variant"] == "D0-SwitchingLatentTransformer"
        ),
        None,
    )
    if result is None or not result.get("diagnostics"):
        return
    diagnostics = result["diagnostics"]
    gradients = diagnostics["prediction_gradient_norms"]
    generator_gradients = all(gradients[name] > 0.0 for name in ("G0", "G1", "G2"))
    regime_connected = (
        gradients["regime evidence"] > 0.0
        and gradients["transition logits"] > 0.0
    )
    generators_differ = (
        diagnostics["mean_candidate_l1"] > 1e-6
        and diagnostics["max_forced_z_diff"] > 1e-6
    )
    z_used = (
        diagnostics["zero_z_prediction_diff"] > 1e-6
        and diagnostics["readout_wz_wh_ratio"] > 1e-3
        and generator_gradients
    )
    near_uniform = abs(diagnostics["posterior_entropy"] - np.log(3.0)) < 1e-3
    improves = result["MAE"] < 0.023073
    vanishing_or_exploding = (
        diagnostics["mean_z_norm"] < 1e-6
        or diagnostics["max_z_norm"] > 100.0
    )

    if vanishing_or_exploding:
        case = "Case E"
        conclusion = (
            "The deterministic micro-state trajectory vanishes or explodes; "
            "report norms, gradients, and readout scales without adding controls."
        )
    elif not generators_differ:
        case = "Case D"
        conclusion = (
            "The state-specific generators collapse to effectively identical "
            "micro dynamics; no diversity regularizer is added in D0."
        )
    elif improves and not z_used:
        case = "Case B"
        conclusion = (
            "Prediction improves while the micro state is effectively ignored; "
            "the gain cannot be attributed to switching latent dynamics."
        )
    elif improves and generators_differ and z_used and regime_connected:
        case = "Case A"
        conclusion = (
            "Distinct regime-conditioned micro dynamics are prediction-connected "
            "and D0 improves over S0D."
        )
    elif near_uniform and generators_differ and z_used and regime_connected:
        case = "Case C"
        conclusion = (
            "The posterior is near uniform, but distinct generators and an active "
            "micro-state path remain; entropy alone is not treated as failure."
        )
    else:
        case = "Mixed"
        conclusion = (
            "The run does not cleanly match one predefined case; interpret memory, "
            "filtering, generator, micro-state, and prediction diagnostics jointly."
        )
    print(f"\nD0 mechanism conclusion: [{case}] {conclusion}")


def _print_d0b_conclusion(results):
    result = next(
        (
            item for item in results
            if item["variant"] == "D0B-BalancedLatentReadout"
        ),
        None,
    )
    if result is None or not result.get("diagnostics"):
        return
    diagnostics = result["diagnostics"]
    balanced = 0.2 <= diagnostics["post_balance_micro_long_ratio"] <= 5.0
    micro_visible = (
        diagnostics["zero_z_prediction_mean_diff"] > 7.417e-05
        and diagnostics["max_forced_prediction_diff"] > 9.038e-05
    )
    final_weights_ignore_micro = (
        diagnostics["readout_wz_wh_ratio"] < 0.1
        and diagnostics["zero_z_prediction_mean_diff"] <= 7.417e-05
    )
    improves = result["MAE"] < 0.023090

    if not balanced:
        case = "Implementation check"
        conclusion = (
            "Post-normalization long/micro representations are still badly "
            "imbalanced; inspect the D0B readout before attributing performance."
        )
    elif improves and micro_visible:
        case = "Case A"
        conclusion = (
            "Balanced latent readout makes the micro state more prediction-visible "
            "and improves over D0."
        )
    elif micro_visible and not improves:
        case = "Case B" if result["MAE"] <= 0.023090 * 1.01 else "Case C"
        conclusion = (
            "Micro-state influence increased without an accuracy gain."
            if case == "Case B" else
            "Micro-state influence increased while accuracy worsened, suggesting "
            "the current micro dynamics may add noise."
        )
    elif final_weights_ignore_micro:
        case = "Case D"
        conclusion = (
            "Inputs are scale-balanced, but the learned final readout still "
            "suppresses the micro-state path."
        )
    elif improves:
        case = "Case E"
        conclusion = (
            "Accuracy improved without stronger micro-state diagnostics; do not "
            "attribute the gain to the balancing mechanism."
        )
    else:
        case = "Mixed"
        conclusion = (
            "The run does not cleanly match Case A-E; interpret scale, zero-out, "
            "forced-state, gradient, and forecast diagnostics jointly."
        )
    print(f"\nD0B mechanism conclusion: [{case}] {conclusion}")


def _print_d0c_conclusion(results):
    result = next(
        (
            item for item in results
            if item["variant"] == "D0C-DynamicSlopeTransition"
        ),
        None,
    )
    if result is None or not result.get("diagnostics"):
        return
    diagnostics = result["diagnostics"]
    gradients = diagnostics["prediction_gradient_norms"]
    slope_gradient = float(np.mean([
        gradients[f"slope_projection_{state}"] for state in range(3)
    ]))
    slope_active = (
        slope_gradient > 1e-12
        and diagnostics["slope_base_ratio_mean"] > 1e-6
        and diagnostics["zero_slope_prediction_mean_diff"] > 1e-8
    )
    regime_specific = (
        diagnostics["mean_slope_pairwise_l1"] > 1e-6
        and diagnostics["slope_regime_interaction_std"] > 1e-8
    )
    improves = result["MAE"] < 0.021995
    near_d0b = abs(result["MAE"] - 0.021995) <= 0.021995 * 0.01

    if slope_active and regime_specific and improves:
        case = "Case A"
        conclusion = (
            "The active, regime-specific latent velocity path improves over D0B."
        )
    elif slope_active and near_d0b:
        case = "Case B"
        conclusion = (
            "Slope changes the latent dynamics but adds limited incremental "
            "forecast information."
        )
    elif slope_active and not improves:
        case = "Case C"
        conclusion = (
            "Explicit latent velocity is active but worsens accuracy, consistent "
            "with redundant or noisy information already represented in H_t."
        )
    elif not slope_active and improves:
        case = "Case E"
        conclusion = (
            "Accuracy improves while the slope mechanism is inactive; do not "
            "attribute the gain to DynamicSlopeTransition."
        )
    else:
        case = "Case D"
        conclusion = (
            "The model effectively ignores explicit local long-memory velocity."
        )
    print(f"\nD0C mechanism conclusion: [{case}] {conclusion}")


def _print_d1a_conclusion(results):
    result = next(
        (item for item in results if item["variant"] == "D1A-LatentMemory"),
        None,
    )
    if result is None or not result.get("diagnostics"):
        return
    diagnostics = result["diagnostics"]
    gradients = diagnostics["prediction_gradient_norms"]
    attention_gradient = float(np.mean([
        gradients[f"LatentMemory {name}"]
        for name in ("q_proj", "k_proj", "v_proj", "out_proj")
    ]))
    memory_active = (
        diagnostics["memory_projection_norm"] > 1e-8
        and attention_gradient > 1e-12
        and diagnostics["zero_memory_prediction_mean_diff"] > 1e-8
    )
    improves = result["MAE"] < 0.021995
    near_baseline = abs(result["MAE"] - 0.021995) <= 0.021995 * 0.01

    if memory_active and improves:
        case = "Case A"
        conclusion = "Multi-lag latent memory is active and improves over D0B."
    elif memory_active and near_baseline:
        case = "Case B"
        conclusion = (
            "Latent memory changes the representation but adds no clear forecast gain."
        )
    elif memory_active:
        case = "Case C"
        conclusion = (
            "Latent memory is used but worsens accuracy; extra Z history may add noise."
        )
    elif improves:
        case = "Case E"
        conclusion = (
            "Accuracy improves while the memory path remains ineffective; do not "
            "attribute the gain to latent memory."
        )
    else:
        case = "Case D"
        conclusion = (
            "The optimizer keeps the zero-initialized latent-memory path near its "
            "D0B null state."
        )
    print(f"\nD1A mechanism conclusion: [{case}] {conclusion}")


def _print_d1a2_conclusion(results):
    result = next(
        (
            item for item in results
            if item["variant"] == "D1A2-ActiveLatentMemory"
        ),
        None,
    )
    if result is None or not result.get("diagnostics"):
        return
    diagnostics = result["diagnostics"]
    gradients = diagnostics["prediction_gradient_norms"]
    qk_gradient = float(np.mean([
        gradients["LatentMemory q_proj"],
        gradients["LatentMemory k_proj"],
    ]))
    active = (
        qk_gradient > 1e-12
        and diagnostics["memory_injection_ratio"] > 1e-8
        and diagnostics["zero_memory_prediction_mean_diff"] > 1e-8
    )
    non_uniform = diagnostics["memory_tv_to_uniform"] > 1e-4
    improves = result["MAE"] < 0.021995
    near_baseline = abs(result["MAE"] - 0.021995) <= 0.021995 * 0.01

    if active and non_uniform and improves:
        case = "Case A"
        conclusion = (
            "Active content memory is non-uniform and improves over D0B."
        )
    elif active and not non_uniform:
        case = "Case B"
        conclusion = (
            "The memory path is active but retrieval remains near uniform; "
            "historical Z content may lack temporal discrimination."
        )
    elif active and near_baseline:
        case = "Case C"
        conclusion = (
            "Memory changes the latent dynamics without a clear forecast gain."
        )
    elif not active:
        case = "Case D"
        conclusion = (
            "Memory remains inactive despite default initialization; zero init "
            "was not the only cause."
        )
    elif active and not improves:
        case = "Case E"
        conclusion = (
            "Memory is active but accuracy worsens; extra latent history may add noise."
        )
    else:
        case = "Mixed"
        conclusion = (
            "Interpret gradients, injection, uniformity, intervention, and metrics jointly."
        )
    print(f"\nD1A2 mechanism conclusion: [{case}] {conclusion}")


def _print_d1_conclusion(results):
    result = next(
        (
            item for item in results
            if item["variant"] == "D1-RegimeRelativeLatentMemory"
        ),
        None,
    )
    if result is None or not result.get("diagnostics"):
        return
    diagnostics = result["diagnostics"]
    rpe_gradient = diagnostics["prediction_gradient_norms"].get(
        "Regime latent-memory RPE", 0.0
    )
    rpe_learned = (
        diagnostics["regime_rpe_norm"] > 1e-8
        and diagnostics["mean_regime_rpe_distance"] > 1e-8
        and diagnostics["regime_bias_abs_mean"] > 1e-10
        and rpe_gradient > 0.0
    )
    attention_non_uniform = diagnostics["memory_tv_to_uniform"] > 1e-4
    trajectory_effect = (
        diagnostics["zero_regime_rpe_prediction_mean_diff"] > 1e-8
    )
    improves = result["MAE"] < 0.022101

    if rpe_learned and attention_non_uniform and trajectory_effect and improves:
        case = "Case A"
        conclusion = (
            "Regime-relative latent-memory RPE is learned, changes retrieval and "
            "the latent trajectory, and improves over D1A2."
        )
    elif rpe_learned and trajectory_effect and not improves:
        case = "Case B"
        conclusion = (
            "The RPE forms an active regime-specific memory horizon, but it does "
            "not improve forecast accuracy."
        )
    elif rpe_learned and diagnostics["mean_regime_rpe_distance"] <= 1e-6:
        case = "Case C"
        conclusion = (
            "RPE is active, but the learned state kernels remain effectively similar."
        )
    elif not rpe_learned:
        case = "Case D"
        conclusion = (
            "The optimizer keeps regime-relative memory RPE near its zero null; "
            "do not add manual scaling in this experiment."
        )
    else:
        case = "Case E"
        conclusion = (
            "RPE materially changes retrieval but accuracy worsens; the learned "
            "memory-horizon bias may be harmful for this forecasting task."
        )
    print(f"\nD1 mechanism conclusion: [{case}] {conclusion}")


def main():
    args = parse_args()
    variants = select_variants(args.variants)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(
        f"Device={device} target={TARGET_TYPE} horizon={TARGET_HORIZON} "
        f"variants={[variant[0] for variant in variants]}"
    )
    data = build_data(args)
    descriptor_stats = data["descriptor_stats"]
    print(
        "Descriptor timeline: [abs_return, volatility_5, slope], "
        f"train mean={np.round(descriptor_stats['mean'], 6)}, "
        f"train std={np.round(descriptor_stats['std'], 6)}"
    )
    results = [run_variant(name, variant, args, device, data) for name, variant in variants]

    print("\nRetained ablation summary")
    print(
        f"{'Variant':<20} {'Params':>12} {'Train Time':>12} {'MAE':>10} "
        f"{'RMSE':>10} {'Hit%':>8} {'vs Zero':>10}"
    )
    for result in results:
        print(
            f"{result['variant']:<20} {result['params']:>12,d} "
            f"{result['time']:>11.0f}s "
            f"{result['MAE']:>10.6f} {result['RMSE']:>10.6f} "
            f"{result['Hit_Ratio'] * 100:>7.1f}% {result['vs_zero_pct']:>+9.2f}%"
        )

    _print_s1c_control_comparison(results)
    _print_s2f_conclusion(results)
    _print_d0_conclusion(results)
    _print_d0b_conclusion(results)
    _print_d0c_conclusion(results)
    _print_d1a_conclusion(results)
    _print_d1a2_conclusion(results)
    _print_d1_conclusion(results)

    ExperimentLogger().log_run(
        {
            "version": "retained-ablation-v1",
            "epochs": args.epochs,
            "seed": args.seed,
            "target": TARGET_TYPE,
            "horizon": TARGET_HORIZON,
        },
        [(result["variant"], result["time"], result["mn"], result["mo"]) for result in results],
    )
    return results


if __name__ == "__main__":
    main()
