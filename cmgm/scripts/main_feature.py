#!/usr/bin/env python3
"""
CMGM_Feature — 21-dim features from RAW prices, GCN || LSTM both use all features.

Key differences from main.py:
  - Features computed from RAW prices (returns, RSI, BB, etc. are meaningful)
  - ALL 21 features standardized for cross-asset comparability
  - Both GCN and LSTM branches use the full F-dim feature matrix
  - ALL baselines also use 21-dim features

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_feature
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_feature --baselines
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
from cmgm.data.data_loader import set_seed, create_data_loaders
from cmgm.data.feature_builder import NUM_FEATURES
from cmgm.data.feature_dataset import create_feature_loaders
from cmgm.graph.graph_builder import build_graph
from cmgm.models.feature_model import CMGM_Feature
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
        description='CMGM_Feature — 21-dim features from RAW prices, all models'
    )
    parser.add_argument('--method', type=str, default=CORRELATION_METHOD,
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
    return parser.parse_args()


# =========================================================================
# Evaluation helpers
# =========================================================================

def evaluate_torch(model, loader, edge_index, edge_weight, device, has_graph=True):
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                X_batch, y_batch, _, _ = batch
            else:
                X_batch, y_batch = batch
            X_batch = X_batch.to(device)
            if has_graph:
                pred = model(X_batch, edge_index.to(device), edge_weight.to(device))
            else:
                pred = model(X_batch)
            all_p.append(pred.cpu().numpy())
            all_t.append(y_batch.numpy())
    return np.concatenate(all_p), np.concatenate(all_t)


def evaluate_sklearn(model, loader, scaler, market_indices):
    X_te, y_te = prepare_sklearn_data(loader)
    preds = model.predict(X_te)
    mn = compute_metrics(preds, y_te)
    cs, ce = market_indices['commodity']
    po, to = inverse_transform_predictions(preds, y_te, scaler, cs, ce)
    mo = compute_metrics(po, to)
    return mn, mo


# =========================================================================
# Comparison mode — ALL models use 21-dim features
# =========================================================================

def run_comparison(args):
    """7 models, all using 21-dim features from RAW prices."""
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"Device: {device}  |  Epochs: {args.epochs}  |  Seed: {args.seed}\n")

    # ── Original data (for graph + scaler + raw prices) ──
    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    cs, ce = data['market_indices']['commodity']

    # ── Feature loaders from RAW prices ──
    raw_full = np.concatenate([
        data['raw_prices_train'],
        data['raw_prices_val'],
        data['raw_prices_test'],
    ], axis=0)
    norm_full = data['scaler'].transform(raw_full)

    feat = create_feature_loaders(
        raw_prices=raw_full,
        norm_prices=norm_full,
        market_indices=data['market_indices'],
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    F = feat['feature_dim']
    assert F == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {F}"

    train_loader = feat['feature_train_loader']
    val_loader   = feat['feature_val_loader']
    test_loader  = feat['feature_test_loader']

    # ── Graph (still from training returns) ──
    graph = build_graph(data['train_returns'], data['market_indices'],
                        method=args.method)
    ei, ew = graph['edge_index'], graph['edge_weight']

    results = []

    # 1/6: PCA+Ridge
    print("\n" + "=" * 60)
    print("1/6: PCA+Ridge (21-dim features)")
    print("=" * 60)
    t0 = time.time()
    res = train_linear_regression(train_loader, val_loader, data['n_commodities'])
    mn, mo = evaluate_sklearn(res['model'], test_loader, data['scaler'],
                               data['market_indices'])
    results.append(('PCA+Ridge', time.time() - t0, mn, mo))

    # 2/6: LSTM (21-dim features)
    print("\n" + "=" * 60)
    print("2/6: LSTM (21-dim features)")
    print("=" * 60)
    t0 = time.time()
    res = train_lstm(train_loader, val_loader, data['n_nodes'],
                      data['n_commodities'], device, feat_dim=F,
                      num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], test_loader, ei, ew, device,
                                   has_graph=False)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('LSTM', time.time() - t0, mn, mo))

    # 3/6: BiLSTM (21-dim features)
    print("\n" + "=" * 60)
    print("3/6: BiLSTM (21-dim features)")
    print("=" * 60)
    t0 = time.time()
    res = train_bilstm(train_loader, val_loader, data['n_nodes'],
                        data['n_commodities'], device, feat_dim=F,
                        num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], test_loader, ei, ew, device,
                                   has_graph=False)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('BiLSTM', time.time() - t0, mn, mo))

    # 4/6: GCN-Only (21-dim features)
    print("\n" + "=" * 60)
    print("4/6: GCN-Only (21-dim features)")
    print("=" * 60)
    t0 = time.time()
    res = train_gcn_only(train_loader, val_loader, ei, ew,
                          data['n_nodes'], data['n_commodities'],
                          device, in_dim=F, num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], test_loader, ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('GCN-Only', time.time() - t0, mn, mo))

    # 5/6: GCN+GAT (21-dim features)
    print("\n" + "=" * 60)
    print("5/6: GCN+GAT (21-dim features)")
    print("=" * 60)
    t0 = time.time()
    res = train_gcn_gat(train_loader, val_loader, ei, ew,
                         data['n_nodes'], data['n_commodities'],
                         device, in_dim=F, num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], test_loader, ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('GCN+GAT', time.time() - t0, mn, mo))

    # 6/6: CMGM_Feature (GCN+LSTM both use all features)
    print("\n" + "=" * 60)
    print("6/6: CMGM_Feature (GCN+LSTM both use 21-dim)")
    print("=" * 60)
    t0 = time.time()
    model_feat = CMGM_Feature(data['n_nodes'], data['n_commodities'], feat_dim=F)
    train(model_feat, train_loader, val_loader,
          ei, ew, device, num_epochs=args.epochs, patience=args.patience)
    preds, targs = evaluate_torch(model_feat, test_loader, ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('CMGM-Feat', time.time() - t0, mn, mo))

    # ── Results ──
    print("\n" + "=" * 100)
    print("COMPARISON — Normalized [0,1] Space (21-dim features from RAW prices)")
    print("=" * 100)
    print(f"{'Model':<16s} {'Time':>8s} {'MAE':>10s} {'MSE':>10s} {'RMSE':>10s} "
          f"{'ResMean':>10s} {'ResStd':>10s} {'Skew':>10s}")
    print("-" * 100)
    for name, t, mn, mo in results:
        print(f"{name:<16s} {t:>7.1f}s {mn['MAE']:>10.6f} {mn['MSE']:>10.6f} "
              f"{mn['RMSE']:>10.6f} {mn['Residual_Mean']:>10.6f} "
              f"{mn['Residual_Std']:>10.6f} {mn['Skewness']:>10.6f}")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("COMPARISON — Original Price Space")
    print("=" * 100)
    print(f"{'Model':<16s} {'MAE':>14s} {'MSE':>18s} {'RMSE':>14s}")
    print("-" * 100)
    for name, t, mn, mo in results:
        print(f"{name:<16s} {mo['MAE']:>14.2f} {mo['MSE']:>18.2f} {mo['RMSE']:>14.2f}")
    print("=" * 100)

    log_config = {
        'method': args.method, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'seed': args.seed, 'tag': args.tag,
        'version': 'feature-v2-raw',
        'feature_dim': F,
        'features_from': 'raw_prices',
        'lstm_uses_all_features': True,
    }
    ExperimentLogger().log_run(log_config, results)
    return results


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
    print(f"[Config] Features from RAW prices, dim={NUM_FEATURES}")

    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    scaler = data['scaler']
    market_indices = data['market_indices']
    n_nodes = data['n_nodes']
    n_commodities = data['n_commodities']

    raw_full = np.concatenate([
        data['raw_prices_train'],
        data['raw_prices_val'],
        data['raw_prices_test'],
    ], axis=0)
    norm_full = scaler.transform(raw_full)

    feat = create_feature_loaders(
        raw_prices=raw_full,
        norm_prices=norm_full,
        market_indices=market_indices,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    F = feat['feature_dim']

    graph = build_graph(data['train_returns'], market_indices, method=args.method)
    ei, ew = graph['edge_index'], graph['edge_weight']

    print(f"\n{'=' * 60}")
    print("Model Initialization")
    print(f"{'=' * 60}")
    model = CMGM_Feature(num_nodes=n_nodes, n_commodities=n_commodities, feat_dim=F)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Feature dim: {F} (from RAW prices)")
    print(f"GCN: {F}→10×3 → mean pool → Linear(2840→64)")
    print(f"LSTM: LSTM({n_nodes*F}→64) — ALL features")
    print(f"Feat list: {', '.join(feat['feature_names'])}")

    history = train(model, feat['feature_train_loader'], feat['feature_val_loader'],
                    ei, ew, device, num_epochs=args.epochs, patience=args.patience)

    eval_results = evaluate(model, feat['feature_test_loader'], ei, ew, scaler,
                            market_indices, device, compute_ci=True,
                            model_name='CMGM_Feature')

    if not args.no_per_commodity:
        evaluate_per_commodity(
            model, feat['feature_test_loader'], ei, ew, scaler,
            market_indices, device, data['feature_names'],
        )

    print(f"\n{'=' * 60}")
    print("CMGM_Feature Complete")
    print(f"{'=' * 60}")
    print(f"Features:        {F}-dim (from RAW prices)")
    print(f"LSTM input:      ALL {F} features (N*F = {n_nodes*F})")
    print(f"Best epoch:      {history.get('best_epoch', 'N/A')}")
    print(f"Best val loss:   {min(history['val_loss']):.6f}")
    print(f"Test MSE (norm): {eval_results['metrics_norm']['MSE']:.6f}")
    print(f"Test MAE (norm): {eval_results['metrics_norm']['MAE']:.6f}")
    print(f"Test MAE (orig): {eval_results['metrics_orig']['MAE']:.2f}")
    print(f"{'=' * 60}")

    logger = ExperimentLogger()
    results_log = [(
        'CMGM_Feature',
        history.get('train_time', 0),
        eval_results['metrics_norm'],
        eval_results['metrics_orig'],
    )]
    log_config = {
        'method': args.method, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'seed': args.seed, 'tag': args.tag,
        'version': 'feature-v2-raw',
        'num_features': F,
    }
    logger.log_run(log_config, results_log)
    return eval_results


# =========================================================================
# Entry point
# =========================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.baselines:
        return run_comparison(args)
    return run_single(args)


if __name__ == '__main__':
    main()
