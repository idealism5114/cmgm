#!/usr/bin/env python3
"""
CMGM: Cross-Market Graph Modeling for Financial Market Forecasting

Reproduction of:
  Ali et al. (2025), "CMGM: A novel cross-market assets and multi-market
  modeling graph neural networks for financial market forecasting leveraging
  market states dependencies", Alexandria Engineering Journal.

Usage:
  python -m cmgm.scripts.main                                                # CMGM original (concat)
  python -m cmgm.scripts.main --fusion-mode gate                             # Gated fusion
  python -m cmgm.scripts.main --fusion-mode mixhop                           # MixHop GCN + gated fusion
  python -m cmgm.scripts.main --baselines                                    # All models comparison
  python -m cmgm.scripts.main --epochs 200 --batch-size 64                   # Custom epochs
  python -m cmgm.scripts.main --fusion-mode mixhop --baselines --epochs 200  # MixHop comparison
"""

import argparse
import os
import sys
import time
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmgm.config import (
    CORRELATION_METHOD, NUM_EPOCHS, BATCH_SIZE,
    SEQ_LEN, RANDOM_SEED, PATIENCE,
)
from cmgm.data.data_loader import set_seed, create_data_loaders
from cmgm.graph.graph_builder import build_graph
from cmgm.models.model import CMGM
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
        description='CMGM: Cross-Market Graph Modeling for Financial Market Forecasting'
    )
    parser.add_argument('--method', type=str, default=CORRELATION_METHOD,
                        choices=['pearson', 'volatility_adjusted',
                                 'skewness_kurtosis_adjusted', 'dynamic',
                                 'dcc_garch'],
                        help='Correlation strategy (Section 3.2)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN,
                        help='Sequence length (default: 20 per paper)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to save model checkpoint')
    parser.add_argument('--no-per-commodity', action='store_true',
                        help='Skip per-commodity evaluation')
    parser.add_argument('--baselines', action='store_true',
                        help='Run all baselines + CMGM comparison')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help='Early stopping patience')
    parser.add_argument('--tag', type=str, default='',
                        help='Optional tag/label for this experiment run')
    parser.add_argument('--fusion-mode', type=str, default='concat',
                        choices=['concat', 'gate', 'mixhop'],
                        help='Fusion mode: concat (original), gate (gated), mixhop (MixHop+GCN)')
    return parser.parse_args()


def evaluate_sklearn(model, test_loader, scaler, market_indices):
    """Evaluate sklearn model (Ridge, SVR)."""
    X_test, y_test = prepare_sklearn_data(test_loader)
    preds = model.predict(X_test)
    mn = compute_metrics(preds, y_test)
    cs, ce = market_indices['commodity']
    po, to = inverse_transform_predictions(preds, y_test, scaler, cs, ce)
    mo = compute_metrics(po, to)
    return mn, mo


def evaluate_torch(model, test_loader, edge_index, edge_weight, device, has_graph=True):
    """Evaluate PyTorch model, return raw predictions and targets."""
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            if has_graph:
                ei = edge_index.to(device)
                ew = edge_weight.to(device)
                pred = model(X_batch, ei, ew)
            else:
                pred = model(X_batch)
            all_p.append(pred.cpu().numpy())
            all_t.append(y_batch.numpy())
    return np.concatenate(all_p), np.concatenate(all_t)


def run_baseline_comparison(args):
    """Run all 7 models and print comparison table."""
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"Device: {device}  |  Epochs: {args.epochs}  |  Seed: {args.seed}\n")

    # --- Data & Graph ---
    print("=" * 60)
    print("Loading data & building graph...")
    print("=" * 60)
    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    graph = build_graph(data['train_returns'], data['market_indices'], method=args.method)
    ei, ew = graph['edge_index'], graph['edge_weight']
    cs, ce = data['market_indices']['commodity']
    results = []

    def fresh_seed():
        set_seed(args.seed)

    # 1/7: PCA+Ridge
    print("\n" + "=" * 60)
    print("1/7: PCA+Ridge Regression")
    print("=" * 60)
    t0 = time.time()
    fresh_seed()
    fresh_seed()
    res = train_linear_regression(data['train_loader'], data['val_loader'],
                                  data['n_commodities'])
    mn, mo = evaluate_sklearn(res['model'], data['test_loader'],
                              data['scaler'], data['market_indices'])
    results.append(('PCA+Ridge', time.time() - t0, mn, mo))

    # 2/7: PCA+LinearSVR
    print("\n" + "=" * 60)
    print("2/7: PCA+LinearSVR")
    print("=" * 60)
    t0 = time.time()
    fresh_seed()
    res = train_svr(data['train_loader'], data['val_loader'],
                    data['n_commodities'])
    mn, mo = evaluate_sklearn(res['model'], data['test_loader'],
                              data['scaler'], data['market_indices'])
    results.append(('PCA+LinearSVR', time.time() - t0, mn, mo))

    # 3/7: LSTM
    print("\n" + "=" * 60)
    print("3/7: LSTM")
    print("=" * 60)
    t0 = time.time()
    fresh_seed()
    res = train_lstm(data['train_loader'], data['val_loader'],
                     data['n_nodes'], data['n_commodities'], device,
                     num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'],
                                  ei, ew, device, has_graph=False)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('LSTM', time.time() - t0, mn, mo))

    # 4/7: BiLSTM
    print("\n" + "=" * 60)
    print("4/7: BiLSTM")
    print("=" * 60)
    t0 = time.time()
    fresh_seed()
    res = train_bilstm(data['train_loader'], data['val_loader'],
                       data['n_nodes'], data['n_commodities'], device,
                       num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'],
                                  ei, ew, device, has_graph=False)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('BiLSTM', time.time() - t0, mn, mo))

    # 5/7: GCN-Only
    print("\n" + "=" * 60)
    print("5/7: GCN-Only")
    print("=" * 60)
    t0 = time.time()
    fresh_seed()
    res = train_gcn_only(data['train_loader'], data['val_loader'],
                         ei, ew, data['n_nodes'], data['n_commodities'],
                         device, num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'],
                                  ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('GCN-Only', time.time() - t0, mn, mo))

    # 6/7: GCN+GAT
    print("\n" + "=" * 60)
    print("6/7: GCN+GAT")
    print("=" * 60)
    t0 = time.time()
    fresh_seed()
    res = train_gcn_gat(data['train_loader'], data['val_loader'],
                        ei, ew, data['n_nodes'], data['n_commodities'],
                        device, num_epochs=args.epochs)
    preds, targs = evaluate_torch(res['model'], data['test_loader'],
                                  ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('GCN+GAT', time.time() - t0, mn, mo))

    # 7/7: CMGM
    print("\n" + "=" * 60)
    print("7/7: CMGM (full)")
    print("=" * 60)
    t0 = time.time()
    fresh_seed()
    model = CMGM(data['n_nodes'], data['n_commodities'], fusion_mode=args.fusion_mode)
    train(model, data['train_loader'], data['val_loader'],
          ei, ew, device, num_epochs=args.epochs, patience=args.patience)
    preds, targs = evaluate_torch(model, data['test_loader'], ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('CMGM', time.time() - t0, mn, mo))

    # --- Results Table ---
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON — Normalized [0,1] Space")
    print("=" * 100)
    print(f"{'Model':<16s} {'Time':>8s} {'MAE':>10s} {'MSE':>10s} {'RMSE':>10s} {'ResMean':>10s} {'ResStd':>10s} {'Skew':>10s}")
    print("-" * 100)
    for name, t, mn, mo in results:
        print(f"{name:<16s} {t:>7.1f}s {mn['MAE']:>10.6f} {mn['MSE']:>10.6f} {mn['RMSE']:>10.6f} {mn['Residual_Mean']:>10.6f} {mn['Residual_Std']:>10.6f} {mn['Skewness']:>10.6f}")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("BASELINE COMPARISON — Original Price Space")
    print("=" * 100)
    print(f"{'Model':<16s} {'MAE':>14s} {'MSE':>18s} {'RMSE':>14s}")
    print("-" * 100)
    for name, t, mn, mo in results:
        print(f"{name:<16s} {mo['MAE']:>14.2f} {mo['MSE']:>18.2f} {mo['RMSE']:>14.2f}")
    print("=" * 100)

    # --- Log experiment ---
    logger = ExperimentLogger()
    config = {
        'method': args.method,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'seed': args.seed,
        'tag': args.tag,
    }
    logger.log_run(config, results)

    return results


def main():
    args = parse_args()

    set_seed(args.seed)

    # --- Baseline comparison mode ---
    if args.baselines:
        return run_baseline_comparison(args)

    # =========================================================================
    # Reproducibility (Section 4.5)
    # =========================================================================
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"\n[Config] Device: {device}")
    print(f"[Config] Method: {args.method}")
    print(f"[Config] Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"[Config] Sequence length: {args.seq_len}")
    print(f"[Config] Seed: {args.seed}")

    # =========================================================================
    # Step 1: Data Loading (Section 4.1)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("STEP 1: Data Loading (Section 4.1)")
    print(f"{'=' * 60}")

    data = create_data_loaders(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    scaler = data['scaler']
    market_indices = data['market_indices']
    n_nodes = data['n_nodes']
    n_commodities = data['n_commodities']
    train_returns = data['train_returns']
    feature_names = data['feature_names']

    # =========================================================================
    # Step 2: Graph Construction (Section 3.2)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("STEP 2: Graph Construction (Section 3.2)")
    print(f"{'=' * 60}")

    graph_data = build_graph(
        returns=train_returns,
        market_indices=market_indices,
        method=args.method,
    )

    edge_index = graph_data['edge_index']
    edge_weight = graph_data['edge_weight']

    # =========================================================================
    # Step 3: Model Initialization (Section 3)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("STEP 3: Model Initialization")
    print(f"{'=' * 60}")

    model = CMGM(num_nodes=n_nodes, n_commodities=n_commodities,
                  fusion_mode=args.fusion_mode)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Architecture (Parallel):")
    print(f"  ┌─ GCN Branch:  3× Mean+Concat → mean pool → Linear(2840→64)")
    print(f"  ├─ LSTM Branch: LSTM({n_nodes}→64)  (raw prices)")
    print(f"  └─ Fusion:      Concat(64+64) → FC(128→64→{n_commodities})")

    # =========================================================================
    # Step 4: Training (Section 3.4)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("STEP 4: Training (Section 3.4)")
    print(f"{'=' * 60}")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint,
    )

    # =========================================================================
    # Step 5: Evaluation (Section 4.4)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("STEP 5: Evaluation (Section 4.4)")
    print(f"{'=' * 60}")

    eval_results = evaluate(
        model=model,
        test_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        scaler=scaler,
        market_indices=market_indices,
        device=device,
        compute_ci=True,
        model_name='CMGM',
    )

    # =========================================================================
    # Step 6: Per-Commodity Evaluation (optional)
    # =========================================================================
    if not args.no_per_commodity:
        print(f"\n{'=' * 60}")
        print("STEP 6: Per-Commodity Evaluation")
        print(f"{'=' * 60}")

        per_commodity = evaluate_per_commodity(
            model=model,
            test_loader=test_loader,
            edge_index=edge_index,
            edge_weight=edge_weight,
            scaler=scaler,
            market_indices=market_indices,
            device=device,
            feature_names=feature_names,
        )

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("CMGM Pipeline Complete")
    print(f"{'=' * 60}")
    print(f"Method:          {args.method}")
    print(f"Best epoch:      {history.get('best_epoch', 'N/A')}")
    print(f"Best val loss:   {min(history['val_loss']):.6f}")
    print(f"Test MAE (norm): {eval_results['metrics_norm']['MAE']:.6f}")
    print(f"Test RMSE (norm): {eval_results['metrics_norm']['RMSE']:.6f}")
    print(f"Test MAE (orig): {eval_results['metrics_orig']['MAE']:.2f}")
    print(f"{'=' * 60}")

    # --- Log experiment ---
    logger = ExperimentLogger()
    results = [(
        'CMGM',
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
    }
    logger.log_run(config, results)

    return eval_results


if __name__ == '__main__':
    main()
