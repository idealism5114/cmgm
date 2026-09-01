"""Run the retained A/F/F2/G/H/I/J ablations and K-ContextBalance.

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
    ("F-RegimeDynamic", "regime_dynamic_transformer"),
    ("F2-RegimeSemantic", "regime_dynamic_semantic"),
    ("G-SemanticRouter", "semantic_router"),
    ("H-LossRebalance", "loss_rebalance"),
    ("I-RoutingStrength", "routing_strength"),
    ("J-ContextCalibrated", "context_calibrated"),
    ("K-ContextBalance", "context_balance"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="HeteroMixHop A/F/F2/G/H/I/J/K ablations")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--variants",
        help="Comma-separated display or internal names; defaults to A/F/F2/G/H/I/J/K",
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
    if label in ("I", "J", "K"):
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
        f"{label} p_raw" if label in ("I", "J", "K") else f"{label} p"
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

    if label in ("I", "J", "K"):
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

    event_label = f"{label} p_raw" if label in ("I", "J", "K") else label
    _event_test(event_label, "volatility", m_last[:, 1], p_last)
    _event_test(event_label, "abs_return", m_last[:, 0], p_last)
    if label in ("I", "J", "K"):
        p_use_last = p_use[:, -1]
        _event_test(f"{label} p_use", "volatility", m_last[:, 1], p_use_last)
        _event_test(f"{label} p_use", "abs_return", m_last[:, 0], p_use_last)

    if label in ("I", "J", "K") and all_loaders is not None:
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
    weighted_dynamic = weights['dynamic'] * components['dynamic'].item()
    weighted_aux_total = weighted_cluster + weighted_info_max + weighted_dynamic
    return_magnitude = return_loss.item()
    eps = 1e-8
    print(f"  [{label} loss/raw] L_return={return_magnitude:.6e}")
    print(f"  [{label} loss/raw] L_cluster={components['cluster'].item():.6e}")
    print(f"  [{label} loss/raw] L_InfoMax={components['info_max'].item():.6e}")
    print(f"  [{label} loss/raw] L_dynamic={components['dynamic'].item():.6e}")
    print(f"  [{label} loss/weighted] cluster={weighted_cluster:.6e}")
    print(f"  [{label} loss/weighted] InfoMax={weighted_info_max:.6e}")
    print(f"  [{label} loss/weighted] dynamic={weighted_dynamic:.6e}")
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

    if label in ("I", "J", "K"):
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
            if label in ("I", "J", "K"):
                rd.forced_regime_use = forced
            else:
                rd.forced_regime = forced
            prediction_forced = model(x_batch, market_descriptor=descriptor_batch)
            temporal_forced = rd.last_h_temporal
            forced_label = (
                "forced downstream p_use"
                if label in ("I", "J", "K")
                else "forced state"
            )
            print(
                f"  [{label} {forced_label} {state}] max_abs_diff h_temporal="
                f"{(temporal_forced - temporal_real).abs().max().item():.6e} "
                f"prediction={(prediction_forced - prediction_real).abs().max().item():.6e}"
            )
        rd.forced_regime = None
        rd.forced_regime_use = None

        if label in ("I", "J", "K"):
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
    if label in ("I", "J", "K"):
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
    if variant in (
        "semantic_router", "loss_rebalance", "routing_strength", "context_calibrated",
        "context_balance"
    ):
        labels = {
            "semantic_router": "G",
            "loss_rebalance": "H",
            "routing_strength": "I",
            "context_calibrated": "J",
            "context_balance": "K",
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
                    "routing_strength", "context_calibrated", "context_balance"
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

    started = time.time()
    loaders = (
        data["semantic_loaders"]
        if variant in (
            "semantic_router", "loss_rebalance", "routing_strength",
            "context_calibrated", "context_balance"
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
