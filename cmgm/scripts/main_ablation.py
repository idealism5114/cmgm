"""Run the three retained HeteroMixHop ablation variants.

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
from cmgm.data.data_loader import MarketSequenceDataset, create_data_loaders, set_seed
from cmgm.data.feature_builder import build_feature_matrix
from cmgm.experiment_logger import ExperimentLogger
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.evaluate import compute_metrics, inverse_transform_predictions
from cmgm.training.train import train


VARIANTS = [
    ("+TempWeighted", "temporal_weighted_graph"),
    ("F-RegimeDynamic", "regime_dynamic_transformer"),
    ("F2-RegimeSemantic", "regime_dynamic_semantic"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Retained HeteroMixHop ablations")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--variants",
        help="Comma-separated display or internal names; defaults to all retained variants",
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
            ("train", "val", "test"), norm_splits, feature_splits, raw_splits
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
    return data


def evaluate_primary_horizon(model, loader, data, device):
    model.eval()
    predictions, targets = [], []
    horizon_index = MULTI_HORIZONS.index(TARGET_HORIZON)
    with torch.no_grad():
        for x_batch, y_batch in loader:
            pred = model(x_batch.to(device)).cpu().numpy()
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


def print_diagnostics(model, variant, test_loader, device):
    x_batch = next(iter(test_loader))[0][:16].to(device)
    model.eval()
    with torch.no_grad():
        model(x_batch)

    alpha = model.last_alpha
    print(
        f"  Temporal weights: mean={alpha.mean().item():.4f} "
        f"std={alpha.std().item():.4f} min={alpha.min().item():.4f} "
        f"max={alpha.max().item():.4f}"
    )
    if variant.startswith("regime_dynamic"):
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
    empty_edges = torch.empty(2, 0, dtype=torch.long)
    empty_weights = torch.zeros(0)
    train(
        model,
        data["loaders"]["train"],
        data["loaders"]["val"],
        empty_edges,
        empty_weights,
        device,
        num_epochs=args.epochs,
        patience=args.patience,
    )
    normalized, original, target = evaluate_primary_horizon(
        model, data["loaders"]["test"], data, device
    )
    print_diagnostics(model, variant, data["loaders"]["test"], device)

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
    results = [run_variant(name, variant, args, device, data) for name, variant in variants]

    print("\nRetained ablation summary")
    print(f"{'Variant':<20} {'Params':>12} {'MAE':>10} {'RMSE':>10} {'Hit%':>8} {'vs Zero':>10}")
    for result in results:
        print(
            f"{result['variant']:<20} {result['params']:>12,d} "
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
