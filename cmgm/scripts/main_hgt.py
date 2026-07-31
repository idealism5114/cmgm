#!/usr/bin/env python3
"""
HGT-CMGM: Cross-Market Graph Modeling with Heterogeneous Graph Transformer.

Replaces homogeneous GCN/RGCN with HGTConv operating on a HeteroData
graph (3 node types × 9 edge types).  The rest of the pipeline
(volatility_adjusted correlation, LSTM, fusion, training) is identical.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_hgt
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_hgt --baselines
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
from cmgm.graph.hetero_graph_builder import build_heterodata
from cmgm.models.hgt_model import CMGM_HGT
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
        description='HGT-CMGM: Heterogeneous Graph Transformer for Cross-Market Modeling'
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
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='HGT hidden dimension (default: 64)')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='HGT attention heads (default: 4)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='HGT layers (default: 2)')
    return parser.parse_args()


# ── Shared helpers ──

def evaluate_sklearn(model, test_loader, scaler, market_indices):
    X_test, y_test = prepare_sklearn_data(test_loader)
    preds = model.predict(X_test)
    mn = compute_metrics(preds, y_test)
    cs, ce = market_indices['commodity']
    po, to = inverse_transform_predictions(preds, y_test, scaler, cs, ce)
    mo = compute_metrics(po, to)
    return mn, mo


def evaluate_torch(model, test_loader, ei_or_hd, ew_or_dummy, device, has_graph=True):
    """Evaluate a PyTorch model (supports both tensor-based and HeteroData-based models)."""
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                X_batch, y_batch, _, _ = batch
            else:
                X_batch, y_batch = batch
            X_batch = X_batch.to(device)
            if has_graph:
                # HeteroData.to(device) works; Tensor.to(device) also works
                cur_hd = ei_or_hd.to(device) if hasattr(ei_or_hd, 'to') else ei_or_hd
                pred = model(X_batch, cur_hd, ew_or_dummy)
            else:
                pred = model(X_batch)
            all_p.append(pred.cpu().numpy())
            all_t.append(y_batch.numpy())
    return np.concatenate(all_p), np.concatenate(all_t)


# ── Baseline comparison ──

def run_comparison(args):
    """Run 7 baselines + HGT-CMGM comparison."""
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"Device: {device}  |  Epochs: {args.epochs}  |  Seed: {args.seed}\n")

    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    cs, ce = data['market_indices']['commodity']
    results = []

    # Standard graph (for baselines 5-6: GCN, GCN+GAT)
    std_graph = build_graph(data['train_returns'], data['market_indices'],
                            method=args.method)
    ei, ew = std_graph['edge_index'], std_graph['edge_weight']

    # Heterogeneous graph (for HGT-CMGM)
    hetero_data = build_heterodata(data['train_returns'], data['market_indices'],
                                    method=args.method)
    dummy_ew = torch.zeros(0)

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

    # 7/8: CMGM (original)
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

    # 8/8: HGT-CMGM (21-dim features)
    print("\n" + "=" * 60)
    print("8/8: HGT-CMGM (Heterogeneous Graph Transformer + 21-dim features)")
    print("=" * 60)
    t0 = time.time()

    # Feature loaders for HGT-CMGM
    raw_full = np.concatenate([
        data['raw_prices_train'],
        data['raw_prices_val'],
        data['raw_prices_test'],
    ], axis=0)
    full_norm = data['scaler'].transform(raw_full)
    feat = create_feature_loaders(full_norm, data['market_indices'],
                                   batch_size=args.batch_size, seq_len=args.seq_len)
    F = feat['feature_dim']

    model = CMGM_HGT(data['n_nodes'], data['n_commodities'],
                      hidden_dim=args.hidden_dim,
                      num_heads=args.num_heads,
                      num_layers=args.num_layers,
                      in_dim=F)
    train(model, feat['feature_train_loader'], feat['feature_val_loader'],
           hetero_data, dummy_ew,
           device, num_epochs=args.epochs, patience=args.patience)
    preds, targs = evaluate_torch(model, feat['feature_test_loader'],
                                   hetero_data, dummy_ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('HGT-CMGM', time.time() - t0, mn, mo))

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
        'version': 'hgt',
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
    }
    ExperimentLogger().log_run(config, results)
    return results


# ── Single HGT-CMGM run ──

def run_single(args):
    """Run HGT-CMGM only."""
    set_seed(args.seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"\n[Config] Device: {device}")
    print(f"[Config] Method: {args.method}")
    print(f"[Config] Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"[Config] HGT: hidden={args.hidden_dim}, heads={args.num_heads}, "
          f"layers={args.num_layers}")

    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    scaler = data['scaler']
    market_indices = data['market_indices']
    n_nodes = data['n_nodes']
    n_commodities = data['n_commodities']

    # ── Feature loaders (21-dim per-node features) ──
    raw_full = np.concatenate([
        data['raw_prices_train'],
        data['raw_prices_val'],
        data['raw_prices_test'],
    ], axis=0)
    full_norm = scaler.transform(raw_full)                   # (T_total, N)

    feat_loaders = create_feature_loaders(
        prices=full_norm,
        market_indices=market_indices,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    F = feat_loaders['feature_dim']

    # ── Heterogeneous graph ──
    hetero_data = build_heterodata(
        returns=data['train_returns'],
        market_indices=market_indices,
        method=args.method,
    )
    dummy_ew = torch.zeros(0)

    # ── Model ──
    print(f"\n{'=' * 60}")
    print("Model Initialization")
    print(f"{'=' * 60}")
    model = CMGM_HGT(num_nodes=n_nodes, n_commodities=n_commodities,
                      hidden_dim=args.hidden_dim,
                      num_heads=args.num_heads,
                      num_layers=args.num_layers,
                      in_dim=F)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Feature dim:      {F}")
    print(f"HGT: {args.num_layers} layers × {args.num_heads} heads, "
          f"hidden={args.hidden_dim}")
    print(f"Edge types: 9 (3 intra-market corr + 6 cross-market)")
    print(f"Node types: stock={hetero_data['stock'].num_nodes}, "
          f"bond={hetero_data['bond'].num_nodes}, "
          f"future={hetero_data['future'].num_nodes}")
    print(f"Architecture:")
    print(f"  ┌─ HGT Branch:  Input({F}→{args.hidden_dim}) → {args.num_layers}× HGTConv → "
          f"type-pool → Linear(192→64)")
    print(f"  ├─ LSTM Branch: LSTM({n_nodes}→64) — price only")
    print(f"  └─ Fusion:      Concat(64+64) → FC(128→64→{n_commodities})")

    # ── Training ──
    history = train(
        model=model,
        train_loader=feat_loaders['feature_train_loader'],
        val_loader=feat_loaders['feature_val_loader'],
        edge_index=hetero_data,     # ← HeteroData (supports .to(device))
        edge_weight=dummy_ew,        # ← dummy, ignored by model
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
    )

    # ── Evaluation ──
    eval_results = evaluate(
        model=model,
        test_loader=feat_loaders['feature_test_loader'],
        edge_index=hetero_data,     # ← HeteroData
        edge_weight=dummy_ew,
        scaler=scaler,
        market_indices=market_indices,
        device=device,
        compute_ci=True,
        model_name='HGT-CMGM',
    )

    # ── Per-commodity ──
    if not args.no_per_commodity:
        evaluate_per_commodity(
            model=model,
            test_loader=feat_loaders['feature_test_loader'],
            edge_index=hetero_data,
            edge_weight=dummy_ew,
            scaler=scaler,
            market_indices=market_indices,
            device=device,
            feature_names=data['feature_names'],
        )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("HGT-CMGM Pipeline Complete")
    print(f"{'=' * 60}")
    print(f"Method:          {args.method}")
    print(f"Features:        {F}-dim ({', '.join(feat_loaders['feature_names'])})")
    print(f"Best epoch:      {history.get('best_epoch', 'N/A')}")
    print(f"Best val loss:   {min(history['val_loss']):.6f}")
    print(f"Test MAE (norm): {eval_results['metrics_norm']['MAE']:.6f}")
    print(f"Test RMSE (norm):{eval_results['metrics_norm']['RMSE']:.6f}")
    print(f"Test MAE (orig): {eval_results['metrics_orig']['MAE']:.2f}")
    print(f"{'=' * 60}")

    # ── Log ──
    logger = ExperimentLogger()
    results_log = [(
        'HGT-CMGM',
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
        'version': 'hgt-feature',
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'num_features': F,
    }
    logger.log_run(config, results_log)

    return eval_results


# ── Entry point ──

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.baselines:
        return run_comparison(args)
    else:
        return run_single(args)


if __name__ == '__main__':
    main()
