#!/usr/bin/env python3
"""
AdaptiveCMGM — Learnable graph structure via MTGNN-style formulation.

The graph adjacency matrix is learned end-to-end as part of model training,
replacing pre-computed correlation-based graphs. The GCN branch uses a
dense adjacency matrix A instead of sparse edge_index.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_adaptive
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_adaptive --baselines
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_adaptive --embed-dim 20 --top-k 10
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
from cmgm.models.adaptive_graph_model import AdaptiveCMGM
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
        description='AdaptiveCMGM — Learnable graph + Dense GCN || LSTM'
    )
    # Correlation method (used only for baselines, not for adaptive model)
    parser.add_argument('--method', type=str, default=CORRELATION_METHOD,
                        choices=['pearson', 'volatility_adjusted',
                                 'skewness_kurtosis_adjusted', 'dynamic'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--no-per-commodity', action='store_true')
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--tag', type=str, default='')

    # Adaptive graph hyperparameters
    parser.add_argument('--embed-dim', type=int, default=10,
                        help='Node embedding dimension for graph learner')
    parser.add_argument('--alpha', type=float, default=3.0,
                        help='Saturation rate for tanh in graph learner')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top-k neighbors per node in learned graph')
    return parser.parse_args()


# =========================================================================
# Evaluation helpers
# =========================================================================

def evaluate_torch(model, loader, edge_index, edge_weight, device, has_graph=True):
    model.eval()
    all_p, all_t = [], []
    has_internal = hasattr(model, 'graph_learner')
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                X_batch, y_batch, _, _ = batch
            else:
                X_batch, y_batch = batch
            X_batch = X_batch.to(device)
            if has_internal:
                pred = model(X_batch)
            elif has_graph:
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
# Single run
# =========================================================================

def run_single(args):
    set_seed(args.seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"\n[Config] Device: {device}")
    print(f"[Config] Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"[Config] Adaptive graph: embed_dim={args.embed_dim}, "
          f"alpha={args.alpha}, top_k={args.top_k}")

    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    scaler = data['scaler']
    market_indices = data['market_indices']
    n_nodes = data['n_nodes']
    n_commodities = data['n_commodities']

    # ── Model (no graph building needed — graph is learned) ──
    print(f"\n{'=' * 60}")
    print("Model Initialization")
    print(f"{'=' * 60}")
    model = AdaptiveCMGM(
        num_nodes=n_nodes,
        n_commodities=n_commodities,
        embed_dim=args.embed_dim,
        alpha=args.alpha,
        top_k=args.top_k,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Graph learner:    E=({n_nodes}×{args.embed_dim}), "
          f"top_k={args.top_k}, α={args.alpha}")
    print(f"Architecture:")
    print(f"  ┌─ Graph: AdaptiveGraphLearner → A (N×N)")
    print(f"  ├─ GCN:  DenseMeanConcat(×3) → reduce(2840→64) per step → mean pool")
    print(f"  ├─ LSTM: LSTM({n_nodes}→64)")
    print(f"  └─ Fusion: Concat(64+64) → FC(128→64→{n_commodities})")

    # Dummy tensors (ignored by model — it has internal graph_learner)
    dummy_ei = torch.empty(2, 0, dtype=torch.long)
    dummy_ew = torch.zeros(0)

    history = train(model, data['train_loader'], data['val_loader'],
                    dummy_ei, dummy_ew, device,
                    num_epochs=args.epochs, patience=args.patience)

    eval_results = evaluate(model, data['test_loader'], dummy_ei, dummy_ew,
                            scaler, market_indices, device, compute_ci=True,
                            model_name='AdaptiveCMGM')

    if not args.no_per_commodity:
        evaluate_per_commodity(model, data['test_loader'], dummy_ei, dummy_ew,
                               scaler, market_indices, device,
                               data['feature_names'])

    # Inspect learned graph
    A = model.graph_learner().detach().cpu()
    nz = (A > 0).sum().item()
    print(f"\n{'─' * 50}")
    print(f"  Learned Graph Statistics")
    print(f"{'─' * 50}")
    print(f"  Sparsity:     {nz}/{A.numel()} edges ({nz/A.numel()*100:.1f}%)")
    print(f"  Row non-zeros: {A.gt(0).sum(dim=1).tolist()[:10]}...")
    print(f"  Weight range: [{A[A>0].min():.4f}, {A.max():.4f}]")
    print(f"  Symmetric:    {(A - A.T).abs().max():.6f} diff (0 = fully symmetric)")

    # Summary
    print(f"\n{'=' * 60}")
    print("AdaptiveCMGM Complete")
    print(f"{'=' * 60}")
    print(f"Best epoch:      {history.get('best_epoch', 'N/A')}")
    print(f"Best val loss:   {min(history['val_loss']):.6f}")
    print(f"Test MSE (norm): {eval_results['metrics_norm']['MSE']:.6f}")
    print(f"Test MAE (norm): {eval_results['metrics_norm']['MAE']:.6f}")
    print(f"Test MAE (orig): {eval_results['metrics_orig']['MAE']:.2f}")
    print(f"Learned graph:   {nz} edges ({nz/A.numel()*100:.1f}% dense)")
    print(f"{'=' * 60}")

    logger = ExperimentLogger()
    results_log = [(
        'AdaptiveCMGM',
        history.get('train_time', 0),
        eval_results['metrics_norm'],
        eval_results['metrics_orig'],
    )]
    log_config = {
        'method': 'adaptive', 'epochs': args.epochs,
        'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'seed': args.seed, 'tag': args.tag,
        'version': 'adaptive-graph-v1',
        'embed_dim': args.embed_dim, 'alpha': args.alpha, 'top_k': args.top_k,
    }
    logger.log_run(log_config, results_log)
    return eval_results


# =========================================================================
# Comparison mode
# =========================================================================

def run_comparison(args):
    """Baselines (fixed graph) + AdaptiveCMGM."""
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

    # 1/7: PCA+Ridge
    print("\n1/7: PCA+Ridge")
    t0 = time.time()
    res = train_linear_regression(data['train_loader'], data['val_loader'],
                                   data['n_commodities'])
    X_te, y_te = prepare_sklearn_data(data['test_loader'])
    preds = res['model'].predict(X_te)
    mn = compute_metrics(preds, y_te)
    po, to = inverse_transform_predictions(preds, y_te, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('PCA+Ridge', time.time() - t0, mn, mo))

    # 2/7: LSTM
    print("2/7: LSTM")
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

    # 3/7: BiLSTM
    print("3/7: BiLSTM")
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

    # 4/7: GCN-Only
    print("4/7: GCN-Only")
    t0 = time.time()
    res = train_gcn_only(data['train_loader'], data['val_loader'],
                         ei, ew, data['n_nodes'], data['n_commodities'],
                         device, num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'], ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('GCN-Only', time.time() - t0, mn, mo))

    # 5/7: CMGM (original)
    print("5/7: CMGM (original)")
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

    # 6/7: AdaptiveCMGM (learned graph)
    print("6/7: AdaptiveCMGM (learned graph)")
    t0 = time.time()
    model_adp = AdaptiveCMGM(data['n_nodes'], data['n_commodities'],
                              embed_dim=args.embed_dim,
                              alpha=args.alpha, top_k=args.top_k)
    dummy_ei = torch.empty(2, 0, dtype=torch.long)
    dummy_ew = torch.zeros(0)
    train(model_adp, data['train_loader'], data['val_loader'],
          dummy_ei, dummy_ew, device,
          num_epochs=args.epochs, patience=args.patience)
    preds, targs = evaluate_torch(model_adp, data['test_loader'],
                                   dummy_ei, dummy_ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('AdaptCMGM', time.time() - t0, mn, mo))

    # ── Table ──
    print("\n" + "=" * 100)
    print("COMPARISON — Normalized [0,1] Space")
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
        'method': args.method,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'seq_len': args.seq_len, 'seed': args.seed, 'tag': args.tag,
        'version': 'adaptive-compare-v1',
    }
    ExperimentLogger().log_run(log_config, results)
    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.baselines:
        return run_comparison(args)
    return run_single(args)


if __name__ == '__main__':
    main()
