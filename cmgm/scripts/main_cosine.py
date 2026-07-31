#!/usr/bin/env python3
"""
CMGM + Cosine Annealing Warm Restarts training.

Replaces the original ReduceLROnPlateau scheduler with
CosineAnnealingWarmRestarts(T_0=30, T_mult=2).

Works with any model variant (original CMGM, CMGM_Feature).

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_cosine
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_cosine --T_0 20 --T_mult 2
"""

import argparse
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmgm.config import (
    NUM_EPOCHS, BATCH_SIZE, SEQ_LEN, RANDOM_SEED, PATIENCE,
)
from cmgm.data.data_loader import set_seed, create_data_loaders
from cmgm.graph.graph_builder import build_graph
from cmgm.models.model import CMGM
from cmgm.training.train_cosine import train as cosine_train
from cmgm.training.evaluate import evaluate, evaluate_per_commodity
from cmgm.experiment_logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description='CMGM with Cosine Annealing Warm Restarts'
    )
    parser.add_argument('--method', type=str, default='volatility_adjusted',
                        choices=['pearson', 'volatility_adjusted',
                                 'skewness_kurtosis_adjusted', 'dynamic',
                                 'dcc_garch'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--tag', type=str, default='')
    # Cosine annealing params
    parser.add_argument('--T_0', type=int, default=30,
                        help='Cosine restart period (epochs)')
    parser.add_argument('--T_mult', type=int, default=2,
                        help='Restart period multiplier')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help='Minimum learning rate')
    # Gradient clipping
    parser.add_argument('--max_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='LR warmup epochs')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"\n[Config] Device: {device}")
    print(f"[Config] Method: {args.method}")
    print(f"[Config] Cosine(T_0={args.T_0}, T_mult={args.T_mult}, eta_min={args.eta_min})")

    # ── Data ──
    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    train_loader = data['train_loader']
    val_loader   = data['val_loader']
    test_loader  = data['test_loader']
    scaler = data['scaler']
    market_indices = data['market_indices']
    n_nodes = data['n_nodes']
    n_commodities = data['n_commodities']

    # ── Graph ──
    graph = build_graph(data['train_returns'], market_indices, method=args.method)
    edge_index = graph['edge_index']
    edge_weight = graph['edge_weight']

    # ── Model ──
    print(f"\n{'=' * 60}")
    print("Model Initialization")
    print(f"{'=' * 60}")
    model = CMGM(num_nodes=n_nodes, n_commodities=n_commodities)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    print(f"Total params: {total_params:,}")
    print(f"Scheduler: CosineAnnealingWarmRestarts(T_0={args.T_0}, T_mult={args.T_mult})")

    # ── Training (cosine annealing + gradient clipping) ──
    history = cosine_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        T_0=args.T_0,
        T_mult=args.T_mult,
        eta_min=args.eta_min,
        max_norm=args.max_norm,
        warmup_epochs=args.warmup_epochs,
    )

    # ── Evaluation ──
    eval_results = evaluate(
        model=model,
        test_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        scaler=scaler,
        market_indices=market_indices,
        device=device,
        compute_ci=True,
        model_name=f'CMGM_Cosine(T₀={args.T_0})',
    )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("CMGM + Cosine Annealing Complete")
    print(f"{'=' * 60}")
    print(f"Method:          {args.method}")
    print(f"Best epoch:      {history.get('best_epoch', 'N/A')}")
    print(f"Best val loss:   {min(history['val_loss']):.6f}")
    print(f"Test MSE (norm): {eval_results['metrics_norm']['MSE']:.6f}")
    print(f"Test MAE (norm): {eval_results['metrics_norm']['MAE']:.6f}")
    print(f"{'=' * 60}")

    # ── Log ──
    logger = ExperimentLogger()
    results = [('CMGM_Cosine', history.get('train_time', 0),
                 eval_results['metrics_norm'], eval_results['metrics_orig'])]
    config = {
        'method': args.method, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'seed': args.seed, 'tag': args.tag,
        'version': 'cosine',
        'T_0': args.T_0, 'T_mult': args.T_mult,
    }
    logger.log_run(config, results)

    return eval_results


if __name__ == '__main__':
    main()
