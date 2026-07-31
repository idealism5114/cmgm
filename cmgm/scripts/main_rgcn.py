#!/usr/bin/env python3
"""
RGCN-CMGM: Cross-Market Graph Modeling with Relational Graph Convolution.

Replaces the homogeneous GCN (Mean+Concat) with RGCNConv using 9
relation types defined by source/target market categories.

Architecture:
    RGCN branch (9 relation types, 3 layers) ──┐
    LSTM branch (raw prices)                ──┤── Concat → FC(128→64→24)

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_rgcn                   # RGCN-CMGM only
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_rgcn --baselines       # + baseline comparison
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_rgcn --method dcc_garch
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
from cmgm.graph.graph_builder import build_graph
from cmgm.graph.rgcn_graph_builder import build_rgcn_graph
from cmgm.models.rgcn_model import CMGM_RGCN
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
        description='RGCN-CMGM: Relational Graph Convolution for Cross-Market Modeling'
    )
    parser.add_argument('--method', type=str, default='volatility_adjusted',
                        choices=['pearson', 'volatility_adjusted',
                                 'skewness_kurtosis_adjusted', 'dynamic',
                                 'dcc_garch'],
                        help='Correlation strategy (default: volatility_adjusted)')
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


# ── Shared helpers ──────────────────────────────────────────────────────────

def evaluate_sklearn(model, test_loader, scaler, market_indices):
    """Evaluate sklearn model (Ridge, SVR)."""
    X_test, y_test = prepare_sklearn_data(test_loader)
    preds = model.predict(X_test)
    mn = compute_metrics(preds, y_test)
    cs, ce = market_indices['commodity']
    po, to = inverse_transform_predictions(preds, y_test, scaler, cs, ce)
    mo = compute_metrics(po, to)
    return mn, mo


def evaluate_torch(model, test_loader, edge_index, edge_third, device, has_graph=True):
    """Evaluate a torch model (for baselines with standard interface)."""
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            if has_graph:
                pred = model(X_batch, edge_index.to(device), edge_third.to(device))
            else:
                pred = model(X_batch)
            all_p.append(pred.cpu().numpy())
            all_t.append(y_batch.numpy())
    return np.concatenate(all_p), np.concatenate(all_t)


# ── Baseline comparison ─────────────────────────────────────────────────────

def run_comparison(args):
    """Run 7 baselines + RGCN-CMGM comparison."""
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"Device: {device}  |  Epochs: {args.epochs}  |  Seed: {args.seed}\n")

    # ── Data ──
    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    cs, ce = data['market_indices']['commodity']
    results = []

    # ── Standard graph (for baselines 5-6: GCN, GCN+GAT) ──
    std_graph = build_graph(data['train_returns'], data['market_indices'],
                            method=args.method)
    ei, ew = std_graph['edge_index'], std_graph['edge_weight']

    # ── Relational graph (for RGCN-CMGM) ──
    rgcn_graph = build_rgcn_graph(data['train_returns'], data['market_indices'],
                                   method=args.method)
    rgcn_ei = rgcn_graph['edge_index']
    rgcn_et = rgcn_graph['edge_type']
    rgcn_ew = rgcn_graph['edge_weight']

    # 1/8: PCA+Ridge
    print("\n" + "=" * 60)
    print("1/8: PCA+Ridge Regression")
    print("=" * 60)
    t0 = time.time()
    res = train_linear_regression(data['train_loader'], data['val_loader'],
                                   data['n_commodities'])
    mn, mo = evaluate_sklearn(res['model'], data['test_loader'],
                               data['scaler'], data['market_indices'])
    results.append(('PCA+Ridge', time.time() - t0, mn, mo))

    # 2/8: PCA+LinearSVR
    print("\n" + "=" * 60)
    print("2/8: PCA+LinearSVR")
    print("=" * 60)
    t0 = time.time()
    res = train_svr(data['train_loader'], data['val_loader'],
                     data['n_commodities'])
    mn, mo = evaluate_sklearn(res['model'], data['test_loader'],
                               data['scaler'], data['market_indices'])
    results.append(('PCA+LinearSVR', time.time() - t0, mn, mo))

    # 3/8: LSTM
    print("\n" + "=" * 60)
    print("3/8: LSTM")
    print("=" * 60)
    t0 = time.time()
    res = train_lstm(data['train_loader'], data['val_loader'],
                      data['n_nodes'], data['n_commodities'], device,
                      num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'],
                                   ei, ew, device, has_graph=False)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('LSTM', time.time() - t0, mn, mo))

    # 4/8: BiLSTM
    print("\n" + "=" * 60)
    print("4/8: BiLSTM")
    print("=" * 60)
    t0 = time.time()
    res = train_bilstm(data['train_loader'], data['val_loader'],
                        data['n_nodes'], data['n_commodities'], device,
                        num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'],
                                   ei, ew, device, has_graph=False)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('BiLSTM', time.time() - t0, mn, mo))

    # 5/8: GCN-Only
    print("\n" + "=" * 60)
    print("5/8: GCN-Only")
    print("=" * 60)
    t0 = time.time()
    res = train_gcn_only(data['train_loader'], data['val_loader'],
                          ei, ew, data['n_nodes'], data['n_commodities'],
                          device, num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'],
                                   ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('GCN-Only', time.time() - t0, mn, mo))

    # 6/8: GCN+GAT
    print("\n" + "=" * 60)
    print("6/8: GCN+GAT")
    print("=" * 60)
    t0 = time.time()
    res = train_gcn_gat(data['train_loader'], data['val_loader'],
                         ei, ew, data['n_nodes'], data['n_commodities'],
                         device, num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'],
                                   ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('GCN+GAT', time.time() - t0, mn, mo))

    # 7/8: CMGM (original — standard graph)
    print("\n" + "=" * 60)
    print("7/8: CMGM (original)")
    print("=" * 60)
    t0 = time.time()
    from cmgm.models.model import CMGM
    model_cmgm = CMGM(data['n_nodes'], data['n_commodities'])
    train(model_cmgm, data['train_loader'], data['val_loader'],
           ei, ew, device, num_epochs=args.epochs, patience=args.patience)
    preds, targs = evaluate_torch(model_cmgm, data['test_loader'], ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('CMGM', time.time() - t0, mn, mo))

    # 8/8: RGCN-CMGM
    print("\n" + "=" * 60)
    print("8/8: RGCN-CMGM (Relational Graph)")
    print("=" * 60)
    t0 = time.time()
    model = CMGM_RGCN(data['n_nodes'], data['n_commodities'])
    train(model, data['train_loader'], data['val_loader'],
           rgcn_ei, rgcn_et,    # edge_index, edge_type (as third arg)
           device, num_epochs=args.epochs, patience=args.patience)
    preds, targs = evaluate_torch(model, data['test_loader'],
                                   rgcn_ei, rgcn_et, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('RGCN-CMGM', time.time() - t0, mn, mo))

    # ── Results table ──
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON — Normalized [0,1] Space")
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
    print("BASELINE COMPARISON — Original Price Space")
    print("=" * 100)
    print(f"{'Model':<16s} {'MAE':>14s} {'MSE':>18s} {'RMSE':>14s}")
    print("-" * 100)
    for name, t, mn, mo in results:
        print(f"{name:<16s} {mo['MAE']:>14.2f} {mo['MSE']:>18.2f} {mo['RMSE']:>14.2f}")
    print("=" * 100)

    # ── Log ──
    config = {
        'method': args.method,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'seed': args.seed,
        'tag': args.tag,
        'version': 'rgcn',
    }
    ExperimentLogger().log_run(config, results)
    return results


# ── Single RGCN-CMGM run ────────────────────────────────────────────────────

def run_single(args):
    """Run RGCN-CMGM only."""
    set_seed(args.seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"\n[Config] Device: {device}")
    print(f"[Config] Method: {args.method}")
    print(f"[Config] Epochs: {args.epochs}, Batch size: {args.batch_size}")

    # ── Data ──
    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    scaler = data['scaler']
    market_indices = data['market_indices']
    n_nodes = data['n_nodes']
    n_commodities = data['n_commodities']

    # ── Relational graph ──
    rgcn_graph = build_rgcn_graph(
        returns=data['train_returns'],
        market_indices=market_indices,
        method=args.method,
    )
    edge_index = rgcn_graph['edge_index']
    edge_type = rgcn_graph['edge_type']
    edge_weight = rgcn_graph['edge_weight']

    # ── Model ──
    print(f"\n{'=' * 60}")
    print("Model Initialization")
    print(f"{'=' * 60}")
    model = CMGM_RGCN(num_nodes=n_nodes, n_commodities=n_commodities)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Relation types: 9 (Stock, Future, Bond × directional pairs)")
    print(f"Architecture:")
    print(f"  ┌─ RGCN Branch:  3× RGCNConv(9 rel) → mean pool → Linear(2840→64)")
    print(f"  ├─ LSTM Branch:  LSTM({n_nodes}→64)")
    print(f"  └─ Fusion:       Concat(64+64) → FC(128→64→{n_commodities})")

    # ── Training ──
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        edge_index=edge_index,
        edge_weight=edge_type,       # ← edge_type is passed here (third arg)
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
    )

    # ── Evaluation ──
    eval_results = evaluate(
        model=model,
        test_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_type,       # ← edge_type is passed here (third arg)
        scaler=scaler,
        market_indices=market_indices,
        device=device,
        compute_ci=True,
        model_name='RGCN-CMGM',
    )

    # ── Per-commodity ──
    if not args.no_per_commodity:
        evaluate_per_commodity(
            model=model,
            test_loader=test_loader,
            edge_index=edge_index,
            edge_weight=edge_type,   # ← edge_type (third arg)
            scaler=scaler,
            market_indices=market_indices,
            device=device,
            feature_names=data['feature_names'],
        )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("RGCN-CMGM Pipeline Complete")
    print(f"{'=' * 60}")
    print(f"Method:          {args.method}")
    print(f"Best epoch:      {history.get('best_epoch', 'N/A')}")
    print(f"Best val loss:   {min(history['val_loss']):.6f}")
    print(f"Test MAE (norm): {eval_results['metrics_norm']['MAE']:.6f}")
    print(f"Test RMSE (norm):{eval_results['metrics_norm']['RMSE']:.6f}")
    print(f"Test MAE (orig): {eval_results['metrics_orig']['MAE']:.2f}")
    print(f"{'=' * 60}")

    # ── Log ──
    logger = ExperimentLogger()
    results = [(
        'RGCN-CMGM',
        history.get('train_time', 0),
        eval_results['metrics_norm'],
        eval_results['metrics_orig'],
    )]
    config = {
        'method': args.method,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'seed': args.seed,
        'tag': args.tag,
        'version': 'rgcn',
    }
    logger.log_run(config, results)

    return eval_results


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.baselines:
        return run_comparison(args)
    else:
        return run_single(args)


if __name__ == '__main__':
    main()
