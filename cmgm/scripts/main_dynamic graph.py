#!/usr/bin/env python3
"""
CMGM v2 — Dynamic Graph Version.

Each sample uses a unique graph computed from the most recent N days of returns,
ensuring no data leakage. Graph evolves over time to reflect changing market states.

Usage:
  CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_v2
  CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_v2 --graph-window 90
  CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_v2 --method dynamic
  CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_v2 --baselines
"""

import argparse
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmgm.config import (
    CORRELATION_METHOD, NUM_EPOCHS, BATCH_SIZE,
    SEQ_LEN, RANDOM_SEED, PATIENCE,
)
from cmgm.data.data_loader import set_seed, create_data_loaders, compute_returns
from cmgm.graph.graph_builder import build_graph
from cmgm.training.train import train
from cmgm.training.evaluate import (
    evaluate, evaluate_per_commodity,
    compute_metrics, inverse_transform_predictions,
)
from cmgm.models.model_v2 import CMGM_Dynamic
from cmgm.data.dynamic_data import create_dynamic_data_loaders
from cmgm.baselines.deep_learning import train_lstm, train_bilstm
from cmgm.experiment_logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description='CMGM v2 — Dynamic Graph Version'
    )
    parser.add_argument('--method', type=str, default=CORRELATION_METHOD,
                        choices=['pearson', 'volatility_adjusted',
                                 'skewness_kurtosis_adjusted', 'dynamic'],
                        help='Correlation strategy')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN)
    parser.add_argument('--graph-window', type=int, default=60,
                        help='Rolling window for dynamic graph (trading days)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--no-per-commodity', action='store_true')
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--tag', type=str, default='')
    return parser.parse_args()


def run_dynamic(args):
    """Run CMGM with dynamic graphs."""
    set_seed(args.seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"\n[Config] Device: {device}")
    print(f"[Config] Method: {args.method}, Graph window: {args.graph_window}")
    print(f"[Config] Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"[Config] Seed: {args.seed}")

    # ========== Step 1: Load & normalize data (reuse v1 loader) ==========
    # We still use create_data_loaders for the core data pipeline,
    # then extract the normalized prices for our dynamic graph dataset.

    print(f"\n{'=' * 60}")
    print("STEP 1: Data Loading & Normalization")
    print(f"{'=' * 60}")

    data_v1 = create_data_loaders(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    n_nodes = data_v1['n_nodes']
    n_commodities = data_v1['n_commodities']
    market_indices = data_v1['market_indices']

    # ========== Step 2: Compute FULL return series for dynamic graph ==========

    print(f"\n{'=' * 60}")
    print("STEP 2: Computing full return series for dynamic graphs")
    print(f"{'=' * 60}")

    raw_train = data_v1['raw_prices_train']
    raw_val = data_v1['raw_prices_val']
    raw_test = data_v1['raw_prices_test']
    all_raw = np.concatenate([raw_train, raw_val, raw_test], axis=0)
    all_returns = compute_returns(all_raw)
    print(f"  Full prices: {all_raw.shape}, Full returns: {all_returns.shape}")

    # ========== Step 3: Build dynamic graph datasets ==========

    print(f"\n{'=' * 60}")
    print("STEP 3: Building dynamic graph datasets")
    print(f"{'=' * 60}")

    # Normalized prices from v1
    train_norm = data_v1['train_loader'].dataset.prices  # (T_train, N)
    val_norm = data_v1['val_loader'].dataset.prices
    test_norm = data_v1['test_loader'].dataset.prices

    prices_dict = {'train': train_norm, 'val': val_norm, 'test': test_norm}

    dynamic_data = create_dynamic_data_loaders(
        prices_dict, all_returns, market_indices,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        graph_window=args.graph_window,
        method=args.method,
    )

    # ========== Step 4: Model initialization ==========

    print(f"\n{'=' * 60}")
    print("STEP 4: Model Initialization")
    print(f"{'=' * 60}")

    model = CMGM_Dynamic(n_nodes, n_commodities)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: CMGM_Dynamic")
    print(f"Total parameters: {total_params:,}")
    print(f"Architecture (Parallel + Dynamic Graph):")
    print(f"  ┌─ GCN Branch:  3× Mean+Concat → mean pool → Linear({n_nodes*10}→64)")
    print(f"  ├─ LSTM Branch: LSTM({n_nodes}→64)  (raw prices)")
    print(f"  └─ Fusion:      Concat(64+64) → FC(128→64→{n_commodities})")

    # ========== Step 5: Training ==========

    print(f"\n{'=' * 60}")
    print("STEP 5: Training")
    print(f"{'=' * 60}")

    history = train(
        model=model,
        train_loader=dynamic_data['train_loader'],
        val_loader=dynamic_data['val_loader'],
        edge_index=torch.zeros(2, 0, dtype=torch.long),   # dummy, not used
        edge_weight=torch.zeros(0),
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
    )

    # ========== Step 6: Evaluation ==========

    print(f"\n{'=' * 60}")
    print("STEP 6: Evaluation")
    print(f"{'=' * 60}")

    # We need to evaluate with the dynamic test loader
    eval_results = evaluate(
        model=model,
        test_loader=dynamic_data['test_loader'],
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_weight=torch.zeros(0),
        scaler=data_v1['scaler'],
        market_indices=data_v1['market_indices'],
        device=device,
        compute_ci=True,
        model_name='CMGM_Dynamic',
    )

    # ========== Summary ==========

    print(f"\n{'=' * 60}")
    print("CMGM v2 Dynamic Graph — Complete")
    print(f"{'=' * 60}")
    print(f"Method:          {args.method}")
    print(f"Graph window:     {args.graph_window} days")
    print(f"Best epoch:      {history.get('best_epoch', 'N/A')}")
    print(f"Test MSE (norm): {eval_results['metrics_norm']['MSE']:.6f}")
    print(f"Test MAE (orig): {eval_results['metrics_orig']['MAE']:.2f}")
    print(f"{'=' * 60}")

    # Log experiment
    logger = ExperimentLogger()
    results = [(
        'CMGM_Dynamic',
        history.get('train_time', 0),
        eval_results['metrics_norm'],
        eval_results['metrics_orig'],
    )]
    config = {
        'method': args.method,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'graph_window': args.graph_window,
        'seed': args.seed,
        'tag': args.tag,
        'version': 'v2_dynamic',
    }
    logger.log_run(config, results)

    return eval_results


def main():
    args = parse_args()

    if args.baselines:
        print("Baseline comparison not yet implemented for v2.")
        print("Running CMGM_Dynamic only...")

    return run_dynamic(args)


if __name__ == '__main__':
    main()
