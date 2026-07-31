#!/usr/bin/env python3
"""
Run full baseline comparison: LR, SVR, LSTM, BiLSTM, GCN, GCN+GAT, CMGM.

Outputs a clean comparison table.
"""

import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmgm.data.data_loader import set_seed, create_data_loaders
from cmgm.graph.graph_builder import build_graph
from cmgm.models.model import CMGM
from cmgm.training.train import train
from cmgm.training.evaluate import (
    evaluate, compute_metrics, inverse_transform_predictions,
    predict, predict_no_graph,
)
from cmgm.baselines.traditional import (
    train_linear_regression, train_svr, prepare_sklearn_data,
)
from cmgm.baselines.deep_learning import train_lstm, train_bilstm
from cmgm.baselines.graph import train_gcn_only, train_gcn_gat

# Config
N_EPOCHS = 50
BATCH_SIZE = 64
CORR_METHOD = 'pearson'
SEED = 42

def evaluate_sklearn(model, test_loader, scaler, market_indices):
    """Evaluate sklearn model."""
    X_test, y_test = prepare_sklearn_data(test_loader)
    preds = model.predict(X_test)
    mn = compute_metrics(preds, y_test)
    cs, ce = market_indices['commodity']
    po, to = inverse_transform_predictions(preds, y_test, scaler, cs, ce)
    mo = compute_metrics(po, to)
    return mn, mo

def evaluate_torch(model, test_loader, edge_index, edge_weight, device, has_graph=True):
    """Evaluate PyTorch model."""
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
    preds = np.concatenate(all_p)
    targets = np.concatenate(all_t)
    return preds, targets

def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Epochs: {N_EPOCHS}  |  Seed: {SEED}\n")

    # ========== Data ==========
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    data = create_data_loaders(batch_size=BATCH_SIZE, seq_len=20)
    graph = build_graph(data['train_returns'], data['market_indices'], method=CORR_METHOD)
    ei, ew = graph['edge_index'], graph['edge_weight']

    results = []

    # ========== 1. Ridge ==========
    print("\n" + "=" * 60)
    print("1/7: PCA+Ridge Regression")
    print("=" * 60)
    t0 = time.time()
    res = train_linear_regression(data['train_loader'], data['val_loader'], data['n_commodities'])
    t1 = time.time()
    mn, mo = evaluate_sklearn(res['model'], data['test_loader'], data['scaler'], data['market_indices'])
    results.append(('PCA+Ridge', t1-t0, mn, mo))

    # ========== 2. LinearSVR ==========
    print("\n" + "=" * 60)
    print("2/7: PCA+LinearSVR")
    print("=" * 60)
    t0 = time.time()
    res = train_svr(data['train_loader'], data['val_loader'], data['n_commodities'])
    t1 = time.time()
    mn, mo = evaluate_sklearn(res['model'], data['test_loader'], data['scaler'], data['market_indices'])
    results.append(('PCA+LinearSVR', t1-t0, mn, mo))

    # ========== 3. LSTM ==========
    print("\n" + "=" * 60)
    print("3/7: LSTM")
    print("=" * 60)
    t0 = time.time()
    res = train_lstm(data['train_loader'], data['val_loader'],
                     data['n_nodes'], data['n_commodities'], device, num_epochs=N_EPOCHS)
    t1 = time.time()
    preds, targs = evaluate_torch(res['model'], data['test_loader'], ei, ew, device, has_graph=False)
    mn = compute_metrics(preds, targs)
    cs, ce = data['market_indices']['commodity']
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('LSTM', t1-t0, mn, mo))

    # ========== 4. BiLSTM ==========
    print("\n" + "=" * 60)
    print("4/7: BiLSTM")
    print("=" * 60)
    t0 = time.time()
    res = train_bilstm(data['train_loader'], data['val_loader'],
                       data['n_nodes'], data['n_commodities'], device, num_epochs=N_EPOCHS)
    t1 = time.time()
    preds, targs = evaluate_torch(res['model'], data['test_loader'], ei, ew, device, has_graph=False)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('BiLSTM', t1-t0, mn, mo))

    # ========== 5. GCN-only ==========
    print("\n" + "=" * 60)
    print("5/7: GCN-Only")
    print("=" * 60)
    t0 = time.time()
    res = train_gcn_only(data['train_loader'], data['val_loader'],
                         ei, ew, data['n_nodes'], data['n_commodities'], device, num_epochs=N_EPOCHS)
    t1 = time.time()
    preds, targs = evaluate_torch(res['model'], data['test_loader'], ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('GCN-Only', t1-t0, mn, mo))

    # ========== 6. GCN+GAT ==========
    print("\n" + "=" * 60)
    print("6/7: GCN+GAT")
    print("=" * 60)
    t0 = time.time()
    res = train_gcn_gat(data['train_loader'], data['val_loader'],
                        ei, ew, data['n_nodes'], data['n_commodities'], device, num_epochs=N_EPOCHS)
    t1 = time.time()
    preds, targs = evaluate_torch(res['model'], data['test_loader'], ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('GCN+GAT', t1-t0, mn, mo))

    # ========== 7. CMGM ==========
    print("\n" + "=" * 60)
    print("7/7: CMGM (full)")
    print("=" * 60)
    t0 = time.time()
    model = CMGM(data['n_nodes'], data['n_commodities'])
    history = train(model, data['train_loader'], data['val_loader'],
                    ei, ew, device, num_epochs=N_EPOCHS)
    t1 = time.time()
    preds, targs = evaluate_torch(model, data['test_loader'], ei, ew, device)
    mn = compute_metrics(preds, targs)
    po, to = inverse_transform_predictions(preds, targs, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('CMGM', t1-t0, mn, mo))

    # ========== Results Table ==========
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON — Normalized [0,1] Space (matches paper)")
    print("=" * 80)
    print(f"{'Model':<16s} {'Time':>8s} {'MAE':>10s} {'MSE':>10s} {'RMSE':>10s} {'ResMean':>10s} {'ResStd':>10s} {'Skew':>10s}")
    print("-" * 80)
    for name, t, mn, mo in results:
        print(f"{name:<16s} {t:>7.1f}s {mn['MAE']:>10.6f} {mn['MSE']:>10.6f} {mn['RMSE']:>10.6f} {mn['Residual_Mean']:>10.6f} {mn['Residual_Std']:>10.6f} {mn['Skewness']:>10.6f}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON — Original Price Space")
    print("=" * 80)
    print(f"{'Model':<16s} {'MAE':>14s} {'MSE':>18s} {'RMSE':>14s}")
    print("-" * 80)
    for name, t, mn, mo in results:
        print(f"{name:<16s} {mo['MAE']:>14.2f} {mo['MSE']:>18.2f} {mo['RMSE']:>14.2f}")
    print("=" * 80)

if __name__ == '__main__':
    main()
