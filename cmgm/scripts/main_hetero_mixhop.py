"""
HeteroMixHopCMGM — Heterogeneous per-type projection + MixHop + gated fusion.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_hetero_mixhop
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_hetero_mixhop --baselines
"""

import argparse, os, sys, time, torch, numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmgm.config import NUM_EPOCHS, BATCH_SIZE, SEQ_LEN, RANDOM_SEED, PATIENCE, FEATURE_DIM
from cmgm.data.data_loader import set_seed, create_data_loaders, compute_features, MarketSequenceDataset
from cmgm.graph.graph_builder import build_graph
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import train
from cmgm.training.evaluate import (
    evaluate, compute_metrics, inverse_transform_predictions,
)
from cmgm.baselines.traditional import train_linear_regression, train_svr, prepare_sklearn_data
from cmgm.baselines.deep_learning import train_lstm, train_bilstm
from cmgm.baselines.graph import train_gcn_only, train_gcn_gat
from cmgm.experiment_logger import ExperimentLogger


def parse_args():
    p = argparse.ArgumentParser(description='HeteroMixHopCMGM')
    p.add_argument('--method', type=str, default='pearson')
    p.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--seq-len', type=int, default=SEQ_LEN)
    p.add_argument('--no-cuda', action='store_true')
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    p.add_argument('--patience', type=int, default=PATIENCE)
    p.add_argument('--baselines', action='store_true')
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()


def evaluate_torch(model, loader, device, ei=None, ew=None, has_graph=False):
    model.eval()
    all_p, all_t = [], []
    has_internal = hasattr(model, 'graph_learner')
    with torch.no_grad():
        for batch in loader:
            X_batch, y_batch = batch[0], batch[1]
            X_batch = X_batch.to(device)
            if has_internal:
                pred = model(X_batch)
            elif has_graph:
                pred = model(X_batch, ei.to(device), ew.to(device))
            else:
                pred = model(X_batch)
            all_p.append(pred.cpu().numpy())
            all_t.append(y_batch.numpy())
    return np.concatenate(all_p), np.concatenate(all_t)


def run_single(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"[Config] Device: {device}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}")
    print(f"[Config] Feature dim: {FEATURE_DIM}")

    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    n_stock = data['market_indices']['stock'][1] - data['market_indices']['stock'][0]
    n_bond  = data['market_indices']['bond'][1] - data['market_indices']['bond'][0]

    # ── Compute 7-dim features from RAW prices ──
    raw_full = np.concatenate([
        data['raw_prices_train'], data['raw_prices_val'], data['raw_prices_test'],
    ], axis=0)
    feat_raw = compute_features(raw_full)                                 # (T, N, 7) — raw scale

    # Unified MinMaxScaler on features (fit on train only)
    from sklearn.preprocessing import MinMaxScaler
    train_sz = data['raw_prices_train'].shape[0]
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(feat_raw[:train_sz].reshape(-1, 7))
    feat_tensor = feat_scaler.transform(feat_raw.reshape(-1, 7)).reshape(feat_raw.shape)
    print(f"[Features] MinMaxScaler on {train_sz} train samples → "
          f"range: [{feat_tensor.min():.4f}, {feat_tensor.max():.4f}]")

    # Normalize prices for target y (unchanged)
    full_norm = data['scaler'].transform(raw_full)

    T = feat_raw.shape[0]
    tr, va = int(T*0.7), int(T*0.7) + int(T*0.15)
    feat_splits = [feat_tensor[:tr], feat_tensor[tr:va], feat_tensor[va:]]
    norm_splits = [full_norm[:tr], full_norm[tr:va], full_norm[va:]]

    dss = {k: MarketSequenceDataset(n, data['market_indices'], args.seq_len, feature_matrix=f)
           for k, n, f in zip(['train','val','test'], norm_splits, feat_splits)}
    loaders = {k: DataLoader(dss[k], batch_size=args.batch_size, shuffle=False,
                             drop_last=(k=='train')) for k in ['train','val','test']}

    model = HeteroMixHopCMGM(data['n_nodes'], data['n_commodities'],
                              n_stock=n_stock, n_bond=n_bond).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Types: stock={n_stock}, bond={n_bond}, future={data['n_nodes']-n_stock-n_bond}")
    print(f"Features: 7-dim (price, return, ma5/20, volatility, rsi_14, macd)")

    dummy = torch.empty(2, 0, dtype=torch.long), torch.zeros(0)
    history = train(model, loaders['train'], loaders['val'],
                    dummy[0], dummy[1], device, num_epochs=args.epochs, patience=args.patience)
    results = evaluate(model, loaders['test'], dummy[0], dummy[1],
                       data['scaler'], data['market_indices'], device,
                       compute_ci=True, model_name='HeteroMixHop')

    final = model.get_gate_stats(next(iter(loaders['train']))[0][:16].to(device))
    print(f"\nBest val: {min(history['val_loss']):.6f}")
    print(f"Test MSE: {results['metrics_norm']['MSE']:.6f}")
    print(f"Gate:     mean={final['gate_mean']:.3f}  mixhop_diff={final.get('mixhop_diff',0):.2f}")

    ExperimentLogger().log_run({'version': 'hetero-mixhop-v2-feat7'}, [(
        'HeteroMixHop', history.get('train_time', 0),
        results['metrics_norm'], results['metrics_orig'],
    )])
    return results


def run_comparison(args):
    """7 models — each uses its own training function, all predict normalized price."""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}  |  Epochs: {args.epochs}  |  Seed: {args.seed}\n")

    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    graph = build_graph(data['train_returns'], data['market_indices'], method=args.method)
    ei, ew = graph['edge_index'], graph['edge_weight']
    cs, ce = data['market_indices']['commodity']
    n_stock = data['market_indices']['stock'][1] - data['market_indices']['stock'][0]
    n_bond  = data['market_indices']['bond'][1] - data['market_indices']['bond'][0]
    results = []

    def eval_torch(model, loader, has_graph=True):
        model.eval()
        all_p, all_t = [], []
        is_internal = hasattr(model, 'graph_learner')
        with torch.no_grad():
            for batch in loader:
                X_batch, y_batch = batch[0], batch[1]
                X_batch = X_batch.to(device)
                if is_internal:
                    pred = model(X_batch)
                elif has_graph:
                    pred = model(X_batch, ei.to(device), ew.to(device))
                else:
                    pred = model(X_batch)
                all_p.append(pred.cpu().numpy())
                all_t.append(y_batch.numpy())
        p, t = np.concatenate(all_p), np.concatenate(all_t)
        mn = compute_metrics(p, t)
        po, to = inverse_transform_predictions(p, t, data['scaler'], cs, ce)
        return mn, compute_metrics(po, to)

    from cmgm.models.model import MixHopPropagation
    from cmgm.models.feature_model import CMGM_Feature
    from cmgm.baselines.traditional import train_svr

    def fresh_seed():
        set_seed(args.seed)

    # ── 7-dim features from RAW prices ──
    raw_full = np.concatenate([
        data['raw_prices_train'], data['raw_prices_val'], data['raw_prices_test'],
    ], axis=0)
    feat_raw = compute_features(raw_full)                                 # (T, N, 7) — raw scale

    # Unified MinMaxScaler on features (fit on train only)
    from sklearn.preprocessing import MinMaxScaler
    train_sz = data['raw_prices_train'].shape[0]
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(feat_raw[:train_sz].reshape(-1, 7))
    feat_tensor = feat_scaler.transform(feat_raw.reshape(-1, 7)).reshape(feat_raw.shape)
    print(f"[Features] MinMaxScaler on {train_sz} train samples → "
          f"range: [{feat_tensor.min():.4f}, {feat_tensor.max():.4f}]")

    # Normalize prices for target y (unchanged)
    full_norm = data['scaler'].transform(raw_full)
    T = full_norm.shape[0]
    tr, va = int(T*0.7), int(T*0.7) + int(T*0.15)
    feat_splits = [feat_tensor[:tr], feat_tensor[tr:va], feat_tensor[va:]]
    norm_splits = [full_norm[:tr], full_norm[tr:va], full_norm[va:]]
    dss = {k: MarketSequenceDataset(n, data['market_indices'], args.seq_len, feature_matrix=f)
           for k, n, f in zip(['train','val','test'], norm_splits, feat_splits)}
    fl = {k: DataLoader(dss[k], batch_size=args.batch_size, shuffle=False,
                        drop_last=(k=='train')) for k in ['train','val','test']}

    results = []

    def eval_torch_7d(model, loader, has_graph=True):
        model.eval()
        all_p, all_t = [], []
        is_internal = hasattr(model, 'graph_learner')
        with torch.no_grad():
            for batch in loader:
                X_batch, y_batch = batch[0], batch[1]
                X_batch = X_batch.to(device)
                if is_internal:
                    pred = model(X_batch)
                elif has_graph:
                    pred = model(X_batch, ei.to(device), ew.to(device))
                else:
                    pred = model(X_batch)
                all_p.append(pred.cpu().numpy())
                all_t.append(y_batch.numpy())
        p, t = np.concatenate(all_p), np.concatenate(all_t)
        mn = compute_metrics(p, t)
        po, to = inverse_transform_predictions(p, t, data['scaler'], cs, ce)
        return mn, compute_metrics(po, to)

    # 1/8: PCA+Ridge (7-dim, sklearn handles any dim)
    print("\n1/8: PCA+Ridge"); fresh_seed(); t0 = time.time()
    m = train_linear_regression(fl['train'], fl['val'], data['n_commodities'])['model']
    X_te, y_te = prepare_sklearn_data(fl['test'])
    mn = compute_metrics(m.predict(X_te), y_te)
    po, to = inverse_transform_predictions(m.predict(X_te), y_te, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('PCA+Ridge', time.time() - t0, mn, mo))

    # 2/8: PCA+SVR (7-dim)
    print("2/8: PCA+SVR"); fresh_seed(); t0 = time.time()
    m = train_svr(fl['train'], fl['val'], data['n_commodities'])['model']
    X_te, y_te = prepare_sklearn_data(fl['test'])
    mn = compute_metrics(m.predict(X_te), y_te)
    po, to = inverse_transform_predictions(m.predict(X_te), y_te, data['scaler'], cs, ce)
    mo = compute_metrics(po, to)
    results.append(('PCA+SVR', time.time() - t0, mn, mo))

    # 3/8: LSTM (7-dim)
    print("3/8: LSTM"); fresh_seed(); t0 = time.time()
    m = train_lstm(fl['train'], fl['val'], data['n_nodes'],
                    data['n_commodities'], device, feat_dim=FEATURE_DIM,
                    num_epochs=args.epochs)['model']
    mn, mo = eval_torch_7d(m, fl['test'], has_graph=False)
    results.append(('LSTM', time.time() - t0, mn, mo))

    # 4/8: BiLSTM (7-dim)
    print("4/8: BiLSTM"); fresh_seed(); t0 = time.time()
    m = train_bilstm(fl['train'], fl['val'], data['n_nodes'],
                      data['n_commodities'], device, feat_dim=FEATURE_DIM,
                      num_epochs=args.epochs)['model']
    mn, mo = eval_torch_7d(m, fl['test'], has_graph=False)
    results.append(('BiLSTM', time.time() - t0, mn, mo))

    # 5/8: GCN-Only (7-dim)
    print("5/8: GCN-Only"); fresh_seed(); t0 = time.time()
    m = train_gcn_only(fl['train'], fl['val'], ei, ew,
                        data['n_nodes'], data['n_commodities'], device,
                        in_dim=FEATURE_DIM, num_epochs=args.epochs)['model']
    mn, mo = eval_torch_7d(m, fl['test'])
    results.append(('GCN-Only', time.time() - t0, mn, mo))

    # 6/8: GCN+GAT (7-dim)
    print("6/8: GCN+GAT"); fresh_seed(); t0 = time.time()
    m = train_gcn_gat(fl['train'], fl['val'], ei, ew,
                       data['n_nodes'], data['n_commodities'], device,
                       in_dim=FEATURE_DIM, num_epochs=args.epochs)['model']
    mn, mo = eval_torch_7d(m, fl['test'])
    results.append(('GCN+GAT', time.time() - t0, mn, mo))

    # 7/8: CMGM_Feature (7-dim, GCN+LSTM both use all features)
    print("7/8: CMGM-Feat"); fresh_seed(); t0 = time.time()
    m = CMGM_Feature(data['n_nodes'], data['n_commodities'], feat_dim=FEATURE_DIM).to(device)
    train(m, fl['train'], fl['val'], ei, ew, device,
          num_epochs=args.epochs, patience=args.patience)
    mn, mo = eval_torch_7d(m, fl['test'])
    results.append(('CMGM-Feat', time.time() - t0, mn, mo))

    # 8/8: HeteroMixHop (7-dim, learnable graph)
    print("8/8: HeteroMixHop"); fresh_seed(); t0 = time.time()
    d = torch.empty(2, 0, dtype=torch.long), torch.zeros(0)
    m = HeteroMixHopCMGM(data['n_nodes'], data['n_commodities'],
                          n_stock=n_stock, n_bond=n_bond).to(device)
    train(m, fl['train'], fl['val'], d[0], d[1], device,
          num_epochs=args.epochs, patience=args.patience)
    mn, mo = eval_torch_7d(m, fl['test'])
    results.append(('HeteroMix', time.time() - t0, mn, mo))

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

    ExperimentLogger().log_run({'version': 'hetero-mixhop-compare-v1'}, results)
    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.baselines:
        return run_comparison(args)
    return run_single(args)


if __name__ == '__main__':
    main()
