#!/usr/bin/env python3
"""
Unified baseline runner for CMGM comparison.

Runs all baselines and reports metrics for comparison with CMGM.

Usage:
  python -m cmgm.baselines.run                        # All baselines
  python -m cmgm.baselines.run --lr --lstm            # Selected baselines
  python -m cmgm.baselines.run --epochs 50 --method dynamic
"""

import argparse
import sys
import os
import torch
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))

from cmgm.config import CORRELATION_METHOD, NUM_EPOCHS, BATCH_SIZE, SEQ_LEN, RANDOM_SEED
from cmgm.data.data_loader import set_seed, create_data_loaders
from cmgm.graph.graph_builder import build_graph
from cmgm.models.model import CMGM
from cmgm.training.train import train
from cmgm.training.evaluate import evaluate, evaluate_per_commodity

from .traditional import train_linear_regression, train_svr, prepare_sklearn_data
from .deep_learning import train_lstm, train_bilstm
from .graph import train_gcn_only, train_gcn_gat


def parse_args():
    parser = argparse.ArgumentParser(description='Run CMGM baselines')
    parser.add_argument('--method', default=CORRELATION_METHOD,
                        choices=['pearson', 'volatility_adjusted',
                                 'skewness_kurtosis_adjusted', 'dynamic'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', action='store_true', help='Linear Regression')
    parser.add_argument('--svr', action='store_true', help='SVR')
    parser.add_argument('--lstm', action='store_true', help='LSTM')
    parser.add_argument('--bilstm', action='store_true', help='BiLSTM')
    parser.add_argument('--gcn', action='store_true', help='GCN-only')
    parser.add_argument('--gcn-gat', action='store_true', help='GCN+GAT')
    parser.add_argument('--cmgm', action='store_true', help='Full CMGM')
    parser.add_argument('--all', action='store_true', default=True,
                        help='Run all baselines (default)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"Device: {device}  |  Seed: {args.seed}  |  Method: {args.method}")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  SeqLen: {SEQ_LEN}")

    # =========================================================================
    # Data & Graph (shared across all models)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Loading data and building graph...")
    print(f"{'=' * 60}")
    data = create_data_loaders(batch_size=args.batch_size, seq_len=SEQ_LEN)
    graph = build_graph(data['train_returns'], data['market_indices'], method=args.method)
    device_cpu = torch.device('cpu')

    # =========================================================================
    # Results table
    # =========================================================================
    results_table = []

    def run_and_record(name, eval_fn, *eval_args, **eval_kwargs):
        """Run evaluation and record results."""
        t0 = time.time()
        result = eval_fn(*eval_args, **eval_kwargs)
        elapsed = time.time() - t0
        m_norm = result.get('metrics_norm', result.get('metrics', {}))
        m_orig = result.get('metrics_orig', result.get('metrics', {}))
        results_table.append({
            'Model': name,
            'Time': f"{elapsed:.1f}s",
            'MAE_norm': m_norm.get('MAE', float('nan')),
            'MSE_norm': m_norm.get('MSE', float('nan')),
            'RMSE_norm': m_norm.get('RMSE', float('nan')),
            'MAE_orig': m_orig.get('MAE', float('nan')),
            'MSE_orig': m_orig.get('MSE', float('nan')),
        })
        return result

    # =========================================================================
    # 1. Linear Regression
    # =========================================================================
    if args.all or args.lr:
        lr_result = train_linear_regression(
            data['train_loader'], data['val_loader'], data['n_commodities']
        )

        # Pipeline handles PCA + scaling internally
        X_test, y_test = prepare_sklearn_data(data['test_loader'])
        preds = lr_result['model'].predict(X_test)

        from ..evaluate import compute_metrics, inverse_transform_predictions
        preds_norm = preds
        targets_norm = y_test
        metrics_norm = compute_metrics(preds_norm, targets_norm)
        cs, ce = data['market_indices']['commodity']
        preds_orig, targets_orig = inverse_transform_predictions(
            preds_norm, targets_norm, data['scaler'], cs, ce
        )
        metrics_orig = compute_metrics(preds_orig, targets_orig)

        print(f"\n{'─' * 40}")
        print(f"  LR Normalized: MAE={metrics_norm['MAE']:.6f}, MSE={metrics_norm['MSE']:.6f}")
        print(f"  LR Original:   MAE={metrics_orig['MAE']:.2f}, MSE={metrics_orig['MSE']:.2f}")
        print(f"{'─' * 40}")
        results_table.append({
            'Model': 'Linear Regression', 'Time': f"{lr_result['train_time']:.1f}s",
            'MAE_norm': metrics_norm['MAE'], 'MSE_norm': metrics_norm['MSE'],
            'RMSE_norm': metrics_norm['RMSE'],
            'MAE_orig': metrics_orig['MAE'], 'MSE_orig': metrics_orig['MSE'],
        })

    # =========================================================================
    # 2. SVR
    # =========================================================================
    if args.all or args.svr:
        svr_result = train_svr(data['train_loader'], data['val_loader'], data['n_commodities'])

        X_test, y_test = prepare_sklearn_data(data['test_loader'])
        preds = svr_result['model'].predict(X_test)

        metrics_norm = compute_metrics(preds, y_test)
        cs, ce = data['market_indices']['commodity']
        preds_orig, targets_orig = inverse_transform_predictions(
            preds, y_test, data['scaler'], cs, ce
        )
        metrics_orig = compute_metrics(preds_orig, targets_orig)

        print(f"\n{'─' * 40}")
        print(f"  SVR Normalized: MAE={metrics_norm['MAE']:.6f}, MSE={metrics_norm['MSE']:.6f}")
        print(f"  SVR Original:   MAE={metrics_orig['MAE']:.2f}, MSE={metrics_orig['MSE']:.2f}")
        print(f"{'─' * 40}")
        results_table.append({
            'Model': 'SVR', 'Time': f"{svr_result['train_time']:.1f}s",
            'MAE_norm': metrics_norm['MAE'], 'MSE_norm': metrics_norm['MSE'],
            'RMSE_norm': metrics_norm['RMSE'],
            'MAE_orig': metrics_orig['MAE'], 'MSE_orig': metrics_orig['MSE'],
        })

    # =========================================================================
    # 3. LSTM
    # =========================================================================
    if args.all or args.lstm:
        lstm_result = train_lstm(
            data['train_loader'], data['val_loader'],
            data['n_nodes'], data['n_commodities'], device,
            num_epochs=args.epochs,
        )

        from ..evaluate import predict_no_graph
        preds_norm, targets_norm = predict_no_graph(
            lstm_result['model'], data['test_loader'], device
        )
        metrics_norm = compute_metrics(preds_norm, targets_norm)
        cs, ce = data['market_indices']['commodity']
        preds_orig, targets_orig = inverse_transform_predictions(
            preds_norm, targets_norm, data['scaler'], cs, ce
        )
        metrics_orig = compute_metrics(preds_orig, targets_orig)

        print(f"\n{'─' * 40}")
        print(f"  LSTM Normalized: MAE={metrics_norm['MAE']:.6f}, MSE={metrics_norm['MSE']:.6f}")
        print(f"  LSTM Original:   MAE={metrics_orig['MAE']:.2f}, MSE={metrics_orig['MSE']:.2f}")
        print(f"{'─' * 40}")
        results_table.append({
            'Model': 'LSTM', 'Time': f"{lstm_result['train_time']:.1f}s",
            'MAE_norm': metrics_norm['MAE'], 'MSE_norm': metrics_norm['MSE'],
            'RMSE_norm': metrics_norm['RMSE'],
            'MAE_orig': metrics_orig['MAE'], 'MSE_orig': metrics_orig['MSE'],
        })

    # =========================================================================
    # 4. BiLSTM
    # =========================================================================
    if args.all or args.bilstm:
        bilstm_result = train_bilstm(
            data['train_loader'], data['val_loader'],
            data['n_nodes'], data['n_commodities'], device,
            num_epochs=args.epochs,
        )

        preds_norm, targets_norm = predict_no_graph(
            bilstm_result['model'], data['test_loader'], device
        )
        metrics_norm = compute_metrics(preds_norm, targets_norm)
        cs, ce = data['market_indices']['commodity']
        preds_orig, targets_orig = inverse_transform_predictions(
            preds_norm, targets_norm, data['scaler'], cs, ce
        )
        metrics_orig = compute_metrics(preds_orig, targets_orig)

        print(f"\n{'─' * 40}")
        print(f"  BiLSTM Normalized: MAE={metrics_norm['MAE']:.6f}, MSE={metrics_norm['MSE']:.6f}")
        print(f"  BiLSTM Original:   MAE={metrics_orig['MAE']:.2f}, MSE={metrics_orig['MSE']:.2f}")
        print(f"{'─' * 40}")
        results_table.append({
            'Model': 'BiLSTM', 'Time': f"{bilstm_result['train_time']:.1f}s",
            'MAE_norm': metrics_norm['MAE'], 'MSE_norm': metrics_norm['MSE'],
            'RMSE_norm': metrics_norm['RMSE'],
            'MAE_orig': metrics_orig['MAE'], 'MSE_orig': metrics_orig['MSE'],
        })

    # =========================================================================
    # 5. GCN-only
    # =========================================================================
    if args.all or args.gcn:
        gcn_result = train_gcn_only(
            data['train_loader'], data['val_loader'],
            graph['edge_index'], graph['edge_weight'],
            data['n_nodes'], data['n_commodities'], device,
            num_epochs=args.epochs,
        )

        preds_norm, targets_norm = evaluate.predict(
            gcn_result['model'], data['test_loader'],
            graph['edge_index'], graph['edge_weight'], device
        )
        metrics_norm = compute_metrics(preds_norm, targets_norm)
        cs, ce = data['market_indices']['commodity']
        preds_orig, targets_orig = inverse_transform_predictions(
            preds_norm, targets_norm, data['scaler'], cs, ce
        )
        metrics_orig = compute_metrics(preds_orig, targets_orig)

        print(f"\n{'─' * 40}")
        print(f"  GCN Normalized: MAE={metrics_norm['MAE']:.6f}, MSE={metrics_norm['MSE']:.6f}")
        print(f"  GCN Original:   MAE={metrics_orig['MAE']:.2f}, MSE={metrics_orig['MSE']:.2f}")
        print(f"{'─' * 40}")
        results_table.append({
            'Model': 'GCN-Only', 'Time': f"{gcn_result['train_time']:.1f}s",
            'MAE_norm': metrics_norm['MAE'], 'MSE_norm': metrics_norm['MSE'],
            'RMSE_norm': metrics_norm['RMSE'],
            'MAE_orig': metrics_orig['MAE'], 'MSE_orig': metrics_orig['MSE'],
        })

    # =========================================================================
    # 6. GCN+GAT
    # =========================================================================
    if args.all or args.gcn_gat:
        gcn_gat_result = train_gcn_gat(
            data['train_loader'], data['val_loader'],
            graph['edge_index'], graph['edge_weight'],
            data['n_nodes'], data['n_commodities'], device,
            num_epochs=args.epochs,
        )

        preds_norm, targets_norm = evaluate.predict(
            gcn_gat_result['model'], data['test_loader'],
            graph['edge_index'], graph['edge_weight'], device
        )
        metrics_norm = compute_metrics(preds_norm, targets_norm)
        cs, ce = data['market_indices']['commodity']
        preds_orig, targets_orig = inverse_transform_predictions(
            preds_norm, targets_norm, data['scaler'], cs, ce
        )
        metrics_orig = compute_metrics(preds_orig, targets_orig)

        print(f"\n{'─' * 40}")
        print(f"  GCN+GAT Normalized: MAE={metrics_norm['MAE']:.6f}, MSE={metrics_norm['MSE']:.6f}")
        print(f"  GCN+GAT Original:   MAE={metrics_orig['MAE']:.2f}, MSE={metrics_orig['MSE']:.2f}")
        print(f"{'─' * 40}")
        results_table.append({
            'Model': 'GCN+GAT', 'Time': f"{gcn_gat_result['train_time']:.1f}s",
            'MAE_norm': metrics_norm['MAE'], 'MSE_norm': metrics_norm['MSE'],
            'RMSE_norm': metrics_norm['RMSE'],
            'MAE_orig': metrics_orig['MAE'], 'MSE_orig': metrics_orig['MSE'],
        })

    # =========================================================================
    # 7. Full CMGM
    # =========================================================================
    if args.all or args.cmgm:
        print(f"\n{'=' * 60}")
        print("Full CMGM")
        print(f"{'=' * 60}")
        model = CMGM(data['n_nodes'], data['n_commodities'])
        history = train(
            model, data['train_loader'], data['val_loader'],
            graph['edge_index'], graph['edge_weight'], device,
            num_epochs=args.epochs,
        )

        eval_results = evaluate(
            model, data['test_loader'], graph['edge_index'], graph['edge_weight'],
            data['scaler'], data['market_indices'], device, compute_ci=False,
            model_name='CMGM',
        )

        results_table.append({
            'Model': 'CMGM', 'Time': f"{history.get('train_time', 0):.1f}s",
            'MAE_norm': eval_results['metrics_norm']['MAE'],
            'MSE_norm': eval_results['metrics_norm']['MSE'],
            'RMSE_norm': eval_results['metrics_norm']['RMSE'],
            'MAE_orig': eval_results['metrics_orig']['MAE'],
            'MSE_orig': eval_results['metrics_orig']['MSE'],
        })

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("BASELINE COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<20s} {'Time':<8s} {'MAE(norm)':<12s} {'MSE(norm)':<12s} "
          f"{'RMSE(norm)':<12s} {'MAE(orig)':<12s}")
    print(f"{'─' * 70}")
    for r in results_table:
        print(f"{r['Model']:<20s} {r['Time']:<8s} "
              f"{r['MAE_norm']:<12.6f} {r['MSE_norm']:<12.6f} "
              f"{r['RMSE_norm']:<12.6f} {r['MAE_orig']:<12.2f}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
