"""Run retained CMGM ablations plus the independent S0 temporal baseline.

Usage:
    python -m cmgm.scripts.main_ablation
    python -m cmgm.scripts.main_ablation --variants +TempWeighted,F-RegimeDynamic
"""

import argparse
import time

import numpy as np
import torch
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
    ("F-RegimeDynamic", "regime_dynamic_transformer"),
    ("F2-RegimeSemantic", "regime_dynamic_semantic"),
    ("G-SemanticRouter", "semantic_router"),
    ("H-LossRebalance", "loss_rebalance"),
    ("I-RoutingStrength", "routing_strength"),
    ("J-ContextCalibrated", "context_calibrated"),
    ("K-ContextBalance", "context_balance"),
    ("L-AdapterOrthogonal", "adapter_orthogonal"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "HeteroMixHop A/B/C/S0/S0D/S1 and retained F/F2/G/H/I/J/K/L ablations"
        )
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--variants",
        help=(
            "Comma-separated display or internal names; defaults to "
            "A/B/C/S0/S0D/S1/F/F2/G/H/I/J/K/L"
        ),
    )
    return parser.parse_args()


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


def _switching_transformer_diagnostics(model, loaders, device):
    """Diagnostics for S1 switching inference and its S0D observation path."""
    branch = model.switching_transformer
    inference = branch.regime_inference
    eps = inference.eps
    split_probabilities = {}
    test_parts = {
        name: [] for name in ("p", "prior", "previous", "q", "E", "r")
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
                    test_parts["r"].append(branch.last_regime_intervention.cpu())
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
            print(f"  [S1 {split_name.upper()}] mean p={mean_p.numpy().round(4)}")
            print(
                f"  [S1 {split_name.upper()}] entropy={entropy.mean().item():.6f} "
                f"mean max p={probabilities.max(dim=-1).values.mean().item():.6f}"
            )
            print(f"  [S1 {split_name.upper()}] argmax occupancy")
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
        f"  [S1 p] mean={p.mean(dim=(0, 1)).numpy().round(4)} "
        f"std={p.std(unbiased=False).item():.6f} min={p.min().item():.6f} "
        f"max={p.max().item():.6f}"
    )
    print(
        f"  [S1 p] entropy mean={entropy.mean().item():.6f} "
        f"std={entropy.std(unbiased=False).item():.6f} "
        f"mean max probability={p.max(dim=-1).values.mean().item():.6f}"
    )
    print(
        f"  [S1 p] mean ||p_t-p_(t-1)||_1="
        f"{transition_magnitude.mean().item():.6f}"
    )
    for step in range(0, min(p.shape[1], 20), 2):
        print(f"  [S1 p t={step}] {p[:, step].mean(dim=0).numpy().round(4)}")

    transition = branch.transition_matrix().detach().cpu()
    row_entropy = -(transition * (transition + eps).log()).sum(dim=-1)
    diagonal = transition.diagonal()
    off_diagonal = transition[~torch.eye(branch.K, dtype=torch.bool)]
    print(f"  [S1 transition matrix]\n{transition.numpy().round(6)}")
    print(f"  [S1 transition] row sums={transition.sum(dim=-1).numpy().round(8)}")
    print(
        f"  [S1 transition] diagonal={diagonal.numpy().round(6)} "
        f"mean persistence={diagonal.mean().item():.6f} "
        f"off-diagonal mean={off_diagonal.mean().item():.6f}"
    )
    for state, value in enumerate(row_entropy):
        print(f"  [S1 transition] state {state} entropy={value.item():.6f}")
    print(f"  [S1 transition] mean entropy={row_entropy.mean().item():.6f}")

    prior_posterior_kl = (
        p * ((p + eps).log() - (prior + eps).log())
    ).sum(dim=-1)
    prior_posterior_l1 = (p - prior).abs().sum(dim=-1)
    print(
        f"  [S1 prior/posterior] mean KL={prior_posterior_kl.mean().item():.6f} "
        f"mean L1={prior_posterior_l1.mean().item():.6f}"
    )

    pairwise_l1 = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        value = (q[:, :, left] - q[:, :, right]).abs().sum(dim=-1).mean()
        pairwise_l1.append(value)
        print(f"  [S1 posterior heads] mean L1(q{left},q{right})={value.item():.6f}")
    print(
        f"  [S1 posterior heads] mean pairwise L1="
        f"{torch.stack(pairwise_l1).mean().item():.6f}"
    )
    for state in range(branch.K):
        q_entropy = -(
            q[:, :, state] * (q[:, :, state] + eps).log()
        ).sum(dim=-1).mean()
        print(f"  [S1 posterior head {state}] entropy={q_entropy.item():.6f}")

    hard = p.argmax(dim=-1)
    counts = torch.zeros(branch.K, branch.K, dtype=torch.long)
    for previous_state in range(branch.K):
        for next_state in range(branch.K):
            counts[previous_state, next_state] = (
                (hard[:, :-1] == previous_state)
                & (hard[:, 1:] == next_state)
            ).sum()
    empirical = counts.float() / counts.sum(dim=-1, keepdim=True).clamp(min=1)
    print(f"  [S1 empirical transition counts]\n{counts.numpy()}")
    print(f"  [S1 empirical transition probability]\n{empirical.numpy().round(6)}")

    embeddings = branch.regime_embeddings.detach().cpu()
    centered = branch.centered_regime_embeddings().detach().cpu()
    for state in range(branch.K):
        print(
            f"  [S1 embedding state {state}] raw norm="
            f"{embeddings[state].norm().item():.6f} centered norm="
            f"{centered[state].norm().item():.6f}"
        )
    token_norm = parts["E"].norm(dim=-1).mean()
    intervention_norm = parts["r"].norm(dim=-1).mean()
    print(
        f"  [S1 intervention] mean ||E||={token_norm.item():.6f} "
        f"mean ||r||={intervention_norm.item():.6f} "
        f"ratio={(intervention_norm / (token_norm + eps)).item():.6f}"
    )
    uniform_intervention = centered.mean(dim=0)
    print(
        f"  [S1 uniform-p sanity] max_abs(r_uniform)="
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
            f"  [S1 input {name}] normalized attention entropy="
            f"{normalized.mean().item():.6f} effective nodes="
            f"{market_entropy.exp().mean().item():.6f} level norm="
            f"{level_norm.item():.6f} dispersion norm={dispersion_norm.item():.6f}"
        )
    print(f"  [S1 input] daily token E norm={token_norm.item():.6f}")

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
            f"  [S1 prediction-only grad] {name}="
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
            f"  [S1 total-loss grad] {name}="
            f"{_aggregate_gradient_norm(parameters):.3e}"
        )
    print(f"  [S1 loss] L_return={return_loss.item():.6e}")
    print(f"  [S1 loss] L_switch_raw={switch_raw.item():.6e}")
    print(f"  [S1 loss] beta={inference.current_beta:.6e}")
    print(f"  [S1 loss] weighted_switch={weighted_switch.item():.6e}")
    print(f"  [S1 loss] total={total_loss.item():.6e}")
    print(
        f"  [S1 loss ratio] switch_to_return_ratio="
        f"{abs(weighted_switch.item()) / (return_loss.item() + eps):.6e}"
    )

    model.eval()
    with torch.no_grad():
        prediction_real = model(x_batch)
        temporal_real = branch.last_h_temporal.clone()
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
            f"  [S1 causality] max p diff through t=10="
            f"{(original_p[:, :split] - perturbed_p[:, :split]).abs().max().item():.3e}"
        )

        tokens_batch = branch.encode_market_tokens(x_batch)
        temporal_batch = branch.temporal_forward(tokens_batch)
        p_batch = branch.last_regime_probabilities.clone()
        tokens_single = branch.encode_market_tokens(x_batch[:1])
        temporal_single = branch.temporal_forward(tokens_single)
        p_single = branch.last_regime_probabilities.clone()
        print(
            f"  [S1 batch independence] E="
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
                f"  [S1 {name} permutation] E="
                f"{(permuted_tokens - tokens_reference).abs().max().item():.3e} "
                f"p={(permuted_p - p_reference).abs().max().item():.3e} "
                f"h_temporal="
                f"{(permuted_temporal - temporal_reference).abs().max().item():.3e}"
            )


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
    if variant == "switching_transformer":
        _switching_transformer_diagnostics(model, loaders, device)
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


def run_variant(name, variant, args, device, data):
    set_seed(args.seed)
    n_stock = data["market_indices"]["stock"][1] - data["market_indices"]["stock"][0]
    n_bond = data["market_indices"]["bond"][1] - data["market_indices"]["bond"][0]
    model = HeteroMixHopCMGM(
        data["n_nodes"],
        data["n_commodities"],
        n_stock=n_stock,
        n_bond=n_bond,
        variant=variant,
        feat_dim=FEATURE_DIM,
    ).to(device)
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
    elif variant == "switching_transformer":
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
        print(
            f"  S0D total params={parameter_count - increase:,}; "
            f"S1 total params={parameter_count:,}; S1-S0D increase={increase:+,}"
        )
        print(
            f"  S1 added params: evidence Transformer={evidence_params:,}; "
            f"posterior heads={posterior_params:,}; transition={transition_params:,}; "
            f"regime embeddings={embedding_params:,}"
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
    train(
        model,
        loaders["train"],
        loaders["val"],
        empty_edges,
        empty_weights,
        device,
        num_epochs=args.epochs,
        patience=args.patience,
    )
    normalized, original, target = evaluate_primary_horizon(
        model, loaders["test"], data, device
    )
    print_diagnostics(model, variant, loaders, device)

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
