#!/usr/bin/env python3
"""
CMGM_LearnableGraph — Edge weights are learnable parameters.

Uses the same ReduceFirst architecture, but the graph edge weights
are nn.Parameter instances that get updated via gradient descent.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_learnable
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_learnable --baselines
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
from cmgm.models.learnable_graph_model import CMGM_LearnableGraph
from cmgm.training.train import train
from cmgm.training.evaluate import (
    evaluate, evaluate_per_commodity,
    compute_metrics, inverse_transform_predictions,
)
from cmgm.baselines.traditional import (
    train_linear_regression, train_svr, prepare_sklearn_data,
)
from cmgm.baselines.deep_learning import train_lstm, train_bilstm
from cmgm.baselines.graph import train_gcn_only, train_gcn_gat
from cmgm.experiment_logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description='CMGM_LearnableGraph — Learnable edge weights'
    )
    parser.add_argument('--method', type=str, default='pearson',
                        choices=['pearson', 'volatility_adjusted',
                                 'skewness_kurtosis_adjusted', 'dynamic',
                                 'dcc_garch'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--no-per-commodity', action='store_true')
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--abs-init', action='store_true', default=True,
                        help='Initialize weights from |correlation| (default: True)')
    return parser.parse_args()


# =========================================================================
# Evaluation helpers
# =========================================================================

def evaluate_torch(model, loader, device, edge_index=None, edge_weight=None):
    """Evaluate — passes graph args for models that need them."""
    model.eval()
    all_p, all_t = [], []
    has_graph = (edge_index is not None and edge_weight is not None)
    with torch.no_grad():
        for batch in loader:
            X_batch, y_batch = batch[0], batch[1]
            X_batch = X_batch.to(device)
            if has_graph:
                pred = model(X_batch, edge_index.to(device), edge_weight.to(device))
            else:
                pred = model(X_batch)
            all_p.append(pred.cpu().numpy())
            all_t.append(y_batch.numpy())
    return np.concatenate(all_p), np.concatenate(all_t)


def evaluate_sklearn(model, loader, scaler, market_indices):
    from cmgm.baselines.traditional import prepare_sklearn_data
    X_te, y_te = prepare_sklearn_data(loader)
    preds = model.predict(X_te)
    mn = compute_metrics(preds, y_te)
    cs, ce = market_indices['commodity']
    po, to = inverse_transform_predictions(preds, y_te, scaler, cs, ce)
    mo = compute_metrics(po, to)
    return mn, mo


# =========================================================================
# Single run
# =========================================================================

def run_single(args):
    set_seed(args.seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"\n[Config] Device: {device}")
    print(f"[Config] Method: {args.method}")
    print(f"[Config] Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"[Config] Learnable edge weights: initialized from {'|correlation|' if args.abs_init else 'correlation'}")

    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    scaler = data['scaler']
    market_indices = data['market_indices']
    n_nodes = data['n_nodes']
    n_commodities = data['n_commodities']

    # ── Graph (needed only for structure + initial weights) ──
    graph = build_graph(data['train_returns'], market_indices, method=args.method)
    ei = graph['edge_index']
    init_w = graph['edge_weight']

    # Initialize weights: |correlation| so all weights are positive
    if args.abs_init:
        init_w = init_w.abs()
    print(f"[Learnable] Initial edge weights: "
          f"range=[{init_w.min():.4f}, {init_w.max():.4f}], "
          f"shape=({len(init_w)},)")

    # ── Model with learnable edge weights ──
    print(f"\n{'=' * 60}")
    print("Model Initialization")
    print(f"{'=' * 60}")
    model = CMGM_LearnableGraph(
        num_nodes=n_nodes,
        n_commodities=n_commodities,
        edge_index=ei,
        init_edge_weight=init_w,
    )
    total_params = sum(p.numel() for p in model.parameters())
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"  - Neural net: {total_params - len(init_w):,}")
    print(f"  - Learnable edge weights: {len(init_w):,}")
    print(f"Architecture: GCN(weighted_mean) → reduce(2840→64) → mean pool || LSTM")
    print(f"Note: passed edge_index/edge_weight are ignored; model uses internal graph.")

    # ── Training ──
    # Pass the graph tensors (they're ignored by model but expected by train.py)
    dummy_ei = torch.empty(2, 0, dtype=torch.long)
    dummy_ew = torch.zeros(0)

    history = train(model, data['train_loader'], data['val_loader'],
                    dummy_ei, dummy_ew, device,
                    num_epochs=args.epochs, patience=args.patience)

    # ── Evaluate ──
    # Override evaluate() to use our custom evaluate_torch that doesn't pass graph
    preds_norm, targets_norm = evaluate_torch(model, data['test_loader'], device)

    print(f"\n{'=' * 60}")
    print("Evaluation")
    print(f"{'=' * 60}")
    print(f"\nPredictions shape: {preds_norm.shape}")

    metrics_norm = compute_metrics(preds_norm, targets_norm)
    print(f"\n{'─' * 50}")
    print(f"  Normalized [0,1] Metrics")
    print(f"{'─' * 50}")
    for name, value in metrics_norm.items():
        print(f"  {name:<18s} {value:.6f}")

    cs, ce = market_indices['commodity']
    po, to = inverse_transform_predictions(preds_norm, targets_norm, scaler, cs, ce)
    metrics_orig = compute_metrics(po, to)
    print(f"\n{'─' * 50}")
    print(f"  Original Price Space Metrics")
    print(f"{'─' * 50}")
    for name, value in metrics_orig.items():
        print(f"  {name:<18s} {value:.2f}")

    # Check learned weights
    learned_w = model.learnable_edge_weight.detach().cpu()
    print(f"\n{'─' * 50}")
    print(f"  Learned Edge Weights")
    print(f"{'─' * 50}")
    print(f"  Range:     [{learned_w.min():.4f}, {learned_w.max():.4f}]")
    print(f"  Mean:      {learned_w.mean():.4f}")
    print(f"  Std:       {learned_w.std():.4f}")
    print(f"  Initial:   [{init_w.min():.4f}, {init_w.max():.4f}]")
    print(f"  Changed:   {(learned_w - init_w).abs().max():.4f} (max abs diff)")

    if not args.no_per_commodity:
        residuals = targets_norm - preds_norm
        cs, ce = market_indices['commodity']
        names = data['feature_names'][cs:ce]
        print(f"\n{'─' * 55}")
        print(f"  Per-Commodity Metrics (Normalized)")
        print(f"{'─' * 55}")
        print(f"  {'Commodity':<20s} {'MAE':<10s} {'MSE':<10s}")
        print(f"{'─' * 55}")
        for i, cname in enumerate(names):
            mae = float(np.mean(np.abs(residuals[:, i])))
            mse = float(np.mean(residuals[:, i] ** 2))
            print(f"  {cname:<20s} {mae:<10.6f} {mse:<10.6f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("CMGM_LearnableGraph Complete")
    print(f"{'=' * 60}")
    print(f"Method:          {args.method}")
    print(f"Best epoch:      {history.get('best_epoch', 'N/A')}")
    print(f"Best val loss:   {min(history['val_loss']):.6f}")
    print(f"Test MSE (norm): {metrics_norm['MSE']:.6f}")
    print(f"Test MAE (norm): {metrics_norm['MAE']:.6f}")
    print(f"Test MAE (orig): {metrics_orig['MAE']:.2f}")
    print(f"{'=' * 60}")

    logger = ExperimentLogger()
    results_log = [(
        'CMGM_LearnableGraph',
        history.get('train_time', 0),
        metrics_norm,
        metrics_orig,
    )]
    log_config = {
        'method': args.method, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'seed': args.seed, 'tag': args.tag,
        'version': 'learnable-graph-v1',
        'abs_init': args.abs_init,
    }
    logger.log_run(log_config, results_log)
    return metrics_norm, metrics_orig


# =========================================================================
# Comparison mode
# =========================================================================

def run_comparison(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"Device: {device}  |  Epochs: {args.epochs}  |  Seed: {args.seed}\n")

    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    cs, ce = data['market_indices']['commodity']
    results = []

    graph = build_graph(data['train_returns'], data['market_indices'],
                        method=args.method)
    ei, ew = graph['edge_index'], graph['edge_weight']

    # Standard baselines (price prediction, fixed graph)
    # 1/8: PCA+Ridge
    print("\n1/8: PCA+Ridge")
    t0 = time.time()
    res = train_linear_regression(data['train_loader'], data['val_loader'], data['n_commodities'])
    X_te, y_te = prepare_sklearn_data(data['test_loader'])
    preds = res['model'].predict(X_te)
    mn = compute_metrics(preds, y_te)
    po, to = inverse_transform_predictions(preds, y_te, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('PCA+Ridge', time.time() - t0, mn, mo))

    # 2/8: LSTM
    print("2/8: LSTM")
    t0 = time.time()
    res = train_lstm(data['train_loader'], data['val_loader'],
                     data['n_nodes'], data['n_commodities'], device, num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'], device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('LSTM', time.time() - t0, mn, mo))

    # 3/8: GCN-Only
    print("3/8: GCN-Only")
    t0 = time.time()
    res = train_gcn_only(data['train_loader'], data['val_loader'],
                         ei, ew, data['n_nodes'], data['n_commodities'], device, num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'], device,
                                   edge_index=ei, edge_weight=ew)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('GCN-Only', time.time() - t0, mn, mo))

    # 4/8: CMGM (original)
    print("4/8: CMGM (original)")
    t0 = time.time()
    from cmgm.models.model import CMGM
    model_cmgm = CMGM(data['n_nodes'], data['n_commodities'])
    train(model_cmgm, data['train_loader'], data['val_loader'],
          ei, ew, device, num_epochs=args.epochs, patience=args.patience)
    preds, targs = evaluate_torch(model_cmgm, data['test_loader'], device,
                                   edge_index=ei, edge_weight=ew)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('CMGM', time.time() - t0, mn, mo))

    # 5/8: CMGM_LearnableGraph (ignores passed graph)
    print("5/8: CMGM_LearnableGraph")
    t0 = time.time()
    init_w = ew.abs() if args.abs_init else ew
    model_lg = CMGM_LearnableGraph(
        num_nodes=data['n_nodes'],
        n_commodities=data['n_commodities'],
        edge_index=ei,
        init_edge_weight=init_w,
    )
    dummy_ei = torch.empty(2, 0, dtype=torch.long)
    dummy_ew = torch.zeros(0)
    train(model_lg, data['train_loader'], data['val_loader'],
          dummy_ei, dummy_ew, device, num_epochs=args.epochs, patience=args.patience)
    preds, targs = evaluate_torch(model_lg, data['test_loader'], device,
                                   edge_index=ei, edge_weight=ew)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('CMGM-Learn', time.time() - t0, mn, mo))

    # Table
    print("\n" + "=" * 100)
    print("COMPARISON — Normalized [0,1] Space")
    print("=" * 100)
    print(f"{'Model':<18s} {'Time':>8s} {'MAE':>10s} {'MSE':>10s} {'RMSE':>10s} "
          f"{'ResMean':>10s} {'ResStd':>10s} {'Skew':>10s}")
    print("-" * 100)
    for name, t, mn, mo in results:
        print(f"{name:<18s} {t:>7.1f}s {mn['MAE']:>10.6f} {mn['MSE']:>10.6f} "
              f"{mn['RMSE']:>10.6f} {mn['Residual_Mean']:>10.6f} "
              f"{mn['Residual_Std']:>10.6f} {mn['Skewness']:>10.6f}")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("COMPARISON — Original Price Space")
    print("=" * 100)
    print(f"{'Model':<18s} {'MAE':>14s} {'MSE':>18s} {'RMSE':>14s}")
    print("-" * 100)
    for name, t, mn, mo in results:
        print(f"{name:<18s} {mo['MAE']:>14.2f} {mo['MSE']:>18.2f} {mo['RMSE']:>14.2f}")
    print("=" * 100)

    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.baselines:
        return run_comparison(args)
    return run_single(args)


if __name__ == '__main__':
    main()
