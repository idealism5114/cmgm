#!/usr/bin/env python3
"""
Multi-Seed Ensemble for CMGM.

Trains N independent CMGM models with different random seeds,
then averages their predictions to produce the final forecast.

Ensembling reduces prediction variance and typically improves
MSE by 3-5% over a single seed.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.ensemble
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.ensemble --seeds 42 43 44 45 46
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.ensemble --method pearson --tag ensemble-v1
"""

import argparse
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmgm.config import (
    NUM_EPOCHS, BATCH_SIZE, SEQ_LEN, PATIENCE,
)
from cmgm.data.data_loader import set_seed, create_data_loaders
from cmgm.graph.graph_builder import build_graph
from cmgm.models.model import CMGM
from cmgm.training.train import train
from cmgm.training.evaluate import (
    compute_metrics, inverse_transform_predictions,
)
from cmgm.experiment_logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description='CMGM Multi-Seed Ensemble'
    )
    parser.add_argument('--method', type=str, default='volatility_adjusted',
                        choices=['pearson', 'volatility_adjusted',
                                 'skewness_kurtosis_adjusted', 'dynamic',
                                 'dcc_garch'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 43, 44, 45, 46],
                        help='Random seeds for ensemble members')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )

    # ── Shared data & graph (same for all seeds) ──
    print("=" * 60)
    print(f"CMGM Ensemble — {len(args.seeds)} seeds: {args.seeds}")
    print("=" * 60)
    print(f"Method: {args.method}  |  Device: {device}")
    print(f"Epochs: {args.epochs}  |  Patience: {args.patience}\n")

    # Load data once (data shape and split are seed-independent)
    set_seed(42)  # temporary — will be reset per-member
    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    graph = build_graph(data['train_returns'], data['market_indices'],
                        method=args.method)
    ei, ew = graph['edge_index'], graph['edge_weight']
    cs, ce = data['market_indices']['commodity']
    scaler = data['scaler']

    test_loader = data['test_loader']

    # ── Train each seed ──
    all_preds = []   # list of (N_test, N_commodities) per seed
    all_histories = []

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'─' * 60}")
        print(f"[{seed_idx + 1}/{len(args.seeds)}] Training with seed={seed}")
        print(f"{'─' * 60}")

        set_seed(seed)
        model = CMGM(num_nodes=data['n_nodes'],
                     n_commodities=data['n_commodities']).to(device)

        t0 = time.time()
        history = train(
            model=model,
            train_loader=data['train_loader'],
            val_loader=data['val_loader'],
            edge_index=ei,
            edge_weight=ew,
            device=device,
            num_epochs=args.epochs,
            patience=args.patience,
        )
        train_time = time.time() - t0

        # Evaluate on test set
        model.eval()
        preds_list = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                pred = model(X_batch.to(device), ei.to(device), ew.to(device))
                preds_list.append(pred.cpu().numpy())

        preds_seed = np.concatenate(preds_list, axis=0)

        # Single-seed metrics (for reference)
        targets = np.concatenate(
            [y.numpy() for _, y in test_loader], axis=0)
        single_metrics = compute_metrics(preds_seed, targets)
        print(f"  Seed {seed}: Test MSE = {single_metrics['MSE']:.6f}  "
              f"(trained {history.get('train_time', 0):.1f}s)")

        all_preds.append(preds_seed)
        all_histories.append(history)

    # ── Ensemble: average predictions ──
    ensemble_preds = np.mean(all_preds, axis=0)  # (N_test, N_commodities)
    targets = np.concatenate(
        [y.numpy() for _, y in test_loader], axis=0)

    # Ensemble metrics (normalized)
    ensemble_metrics_norm = compute_metrics(ensemble_preds, targets)

    # Original price space
    preds_orig, targets_orig = inverse_transform_predictions(
        ensemble_preds, targets, scaler, cs, ce)
    ensemble_metrics_orig = compute_metrics(preds_orig, targets_orig)

    # ── Print results ──
    print(f"\n{'=' * 70}")
    print(f"ENSEMBLE RESULTS ({len(args.seeds)} seeds)")
    print(f"{'=' * 70}")
    print(f"{'Seed':<8s} {'Test MSE':>10s} {'Test MAE':>10s} {'Test RMSE':>10s}")
    print("-" * 70)
    for seed_idx, seed in enumerate(args.seeds):
        targs = np.concatenate([y.numpy() for _, y in test_loader], axis=0)
        m = compute_metrics(all_preds[seed_idx], targs)
        print(f"{seed:<8d} {m['MSE']:>10.6f} {m['MAE']:>10.6f} {m['RMSE']:>10.6f}")
    print("-" * 70)
    print(f"{'ENSEMBLE':<8s} {ensemble_metrics_norm['MSE']:>10.6f} "
          f"{ensemble_metrics_norm['MAE']:>10.6f} "
          f"{ensemble_metrics_norm['RMSE']:>10.6f}")
    print(f"{'Improvement':>18s} vs individual seeds → see above")
    print("=" * 70)

    print(f"\n{'=' * 70}")
    print("ENSEMBLE — Original Price Space")
    print(f"{'=' * 70}")
    print(f"MSE: {ensemble_metrics_orig['MSE']:.2f}")
    print(f"MAE: {ensemble_metrics_orig['MAE']:.2f}")
    print(f"RMSE: {ensemble_metrics_orig['RMSE']:.2f}")
    print("=" * 70)

    # ── Log ──
    config = {
        'method': args.method,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'tag': args.tag,
        'version': 'ensemble',
        'seeds': args.seeds,
        'n_seeds': len(args.seeds),
    }
    # Log each individual seed
    for seed_idx, seed in enumerate(args.seeds):
        targs = np.concatenate([y.numpy() for _, y in test_loader], axis=0)
        m = compute_metrics(all_preds[seed_idx], targs)
        po, to = inverse_transform_predictions(
            all_preds[seed_idx], targs, scaler, cs, ce)
        mo = compute_metrics(po, to)
        ExperimentLogger().log_run(config, [(
            f'CMGM_seed{seed}',
            all_histories[seed_idx].get('train_time', 0),
            m, mo,
        )])

    # Log ensemble
    ExperimentLogger().log_run(config, [(
        f'CMGM_ensemble_{len(args.seeds)}seeds',
        sum(h.get('train_time', 0) for h in all_histories),
        ensemble_metrics_norm,
        ensemble_metrics_orig,
    )])

    print(f"\n✅ Ensemble complete. Best individual MSE vs Ensemble MSE:")
    best_individual = min(
        compute_metrics(all_preds[i], np.concatenate([y.numpy() for _, y in test_loader], axis=0))['MSE']
        for i in range(len(args.seeds))
    )
    improvement = (best_individual - ensemble_metrics_norm['MSE']) / best_individual * 100
    print(f"   Best single: {best_individual:.6f}")
    print(f"   Ensemble:    {ensemble_metrics_norm['MSE']:.6f}")
    print(f"   Improvement: {improvement:+.2f}%")


if __name__ == '__main__':
    main()
