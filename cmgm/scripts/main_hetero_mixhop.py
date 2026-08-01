"""
HeteroMixHopCMGM — Heterogeneous per-type projection + MixHop + gated fusion.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_hetero_mixhop
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_hetero_mixhop --baselines
"""

import argparse, os, sys, time, torch, numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmgm.config import (
    NUM_EPOCHS, BATCH_SIZE, SEQ_LEN, RANDOM_SEED, PATIENCE,
    FEATURE_DIM, TARGET_TYPE, FEAT_ZSCORE_EPS,
)
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
    print(f"[Config] Feature dim: {FEATURE_DIM}  |  Target: {TARGET_TYPE}  |  Norm: per-asset z-score")

    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    n_stock = data['market_indices']['stock'][1] - data['market_indices']['stock'][0]
    n_bond  = data['market_indices']['bond'][1] - data['market_indices']['bond'][0]

    # ── Compute 7-dim features from RAW prices ──
    raw_full = np.concatenate([
        data['raw_prices_train'], data['raw_prices_val'], data['raw_prices_test'],
    ], axis=0)
    feat_raw = compute_features(raw_full)                                 # (T, N, 7) — raw scale

    # Per-asset per-channel z-score on features (fit on train only)
    train_sz = data['raw_prices_train'].shape[0]
    feat_train = feat_raw[:train_sz]                                      # (T_train, N, 7)
    feat_mean = feat_train.mean(axis=(0,), keepdims=True)                 # (1, N, 7)
    feat_std  = feat_train.std(axis=(0,), keepdims=True)                  # (1, N, 7)
    feat_std  = np.maximum(feat_std, FEAT_ZSCORE_EPS)
    feat_tensor = (feat_raw - feat_mean) / feat_std
    print(f"[Features] Per-asset per-channel z-score on {train_sz} train samples → "
          f"range: [{feat_tensor.min():.4f}, {feat_tensor.max():.4f}]")

    # Z-score prices from norm_stats (fit on train only)
    norm_mean = data['norm_stats']['mean']                                # (N,)
    norm_std  = data['norm_stats']['std']                                 # (N,)
    full_norm = (raw_full - norm_mean) / norm_std

    T = feat_raw.shape[0]
    tr, va = int(T*0.7), int(T*0.7) + int(T*0.15)
    feat_splits = [feat_tensor[:tr], feat_tensor[tr:va], feat_tensor[va:]]
    norm_splits = [full_norm[:tr], full_norm[tr:va], full_norm[va:]]
    raw_splits  = [data['raw_prices_train'],
                   data['raw_prices_val'],
                   data['raw_prices_test']]

    dss = {k: MarketSequenceDataset(
               n, data['market_indices'], args.seq_len,
               feature_matrix=f, raw_prices=r, target_type=TARGET_TYPE,
           )
           for k, n, f, r in zip(['train','val','test'],
                                  norm_splits, feat_splits, raw_splits)}
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
                       data['norm_stats'], data['raw_prices_test'],
                       data['market_indices'], device,
                       target_type=TARGET_TYPE,
                       compute_ci=True, model_name='HeteroMixHop')

    final = model.get_gate_stats(next(iter(loaders['train']))[0][:16].to(device))
    print(f"\nBest val: {min(history['val_loss']):.6f}")
    print(f"Test MSE: {results['metrics_norm']['MSE']:.6f}")
    print(f"Gate:     mean={final['gate_mean']:.3f}  mixhop_diff={final.get('mixhop_diff',0):.2f}")

    ExperimentLogger().log_run({'version': 'hetero-mixhop-return-v1'}, [(
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
        po, to = inverse_transform_predictions(
            p, t, data['norm_stats'], data['raw_prices_test'],
            data['market_indices'], target_type=TARGET_TYPE,
        )
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

    # Per-asset per-channel z-score on features (fit on train only)
    train_sz = data['raw_prices_train'].shape[0]
    feat_train = feat_raw[:train_sz]                                      # (T_train, N, 7)
    feat_mean = feat_train.mean(axis=(0,), keepdims=True)                 # (1, N, 7)
    feat_std  = feat_train.std(axis=(0,), keepdims=True)                  # (1, N, 7)
    feat_std  = np.maximum(feat_std, FEAT_ZSCORE_EPS)
    feat_tensor = (feat_raw - feat_mean) / feat_std
    print(f"[Features] Per-asset per-channel z-score on {train_sz} train samples → "
          f"range: [{feat_tensor.min():.4f}, {feat_tensor.max():.4f}]")

    # Z-score prices from norm_stats
    norm_mean = data['norm_stats']['mean']                                # (N,)
    norm_std  = data['norm_stats']['std']                                 # (N,)
    full_norm = (raw_full - norm_mean) / norm_std
    T = full_norm.shape[0]
    tr, va = int(T*0.7), int(T*0.7) + int(T*0.15)
    feat_splits = [feat_tensor[:tr], feat_tensor[tr:va], feat_tensor[va:]]
    norm_splits = [full_norm[:tr], full_norm[tr:va], full_norm[va:]]
    raw_splits  = [data['raw_prices_train'],
                   data['raw_prices_val'],
                   data['raw_prices_test']]
    dss = {k: MarketSequenceDataset(
               n, data['market_indices'], args.seq_len,
               feature_matrix=f, raw_prices=r, target_type=TARGET_TYPE,
           )
           for k, n, f, r in zip(['train','val','test'],
                                  norm_splits, feat_splits, raw_splits)}
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
        po, to = inverse_transform_predictions(
            p, t, data['norm_stats'], data['raw_prices_test'],
            data['market_indices'], target_type=TARGET_TYPE,
        )
        return mn, compute_metrics(po, to)

    # 0/9: Zero baseline — always predict 0 (unconditional mean of returns ≈ 0)
    print("\n0/9: Zero (predict 0)"); fresh_seed(); t0 = time.time()
    X_te, y_te = prepare_sklearn_data(fl['test'])
    zero_preds = np.zeros_like(y_te)
    mn = compute_metrics(zero_preds, y_te)
    po, to = inverse_transform_predictions(
        zero_preds, y_te, data['norm_stats'], data['raw_prices_test'],
        data['market_indices'], target_type=TARGET_TYPE,
    )
    mo = compute_metrics(po, to)
    results.append(('Zero (always 0)', time.time() - t0, mn, mo))

    # 1/9: PCA+Ridge (7-dim, sklearn handles any dim)
    print("\n1/9: PCA+Ridge"); fresh_seed(); t0 = time.time()
    m = train_linear_regression(fl['train'], fl['val'], data['n_commodities'])['model']
    X_te, y_te = prepare_sklearn_data(fl['test'])
    mn = compute_metrics(m.predict(X_te), y_te)
    po, to = inverse_transform_predictions(
        m.predict(X_te), y_te, data['norm_stats'], data['raw_prices_test'],
        data['market_indices'], target_type=TARGET_TYPE,
    )
    mo = compute_metrics(po, to)
    results.append(('PCA+Ridge', time.time() - t0, mn, mo))

    # 2/9: PCA+SVR (7-dim)
    print("\n2/9: PCA+SVR"); fresh_seed(); t0 = time.time()
    m = train_svr(fl['train'], fl['val'], data['n_commodities'])['model']
    X_te, y_te = prepare_sklearn_data(fl['test'])
    mn = compute_metrics(m.predict(X_te), y_te)
    po, to = inverse_transform_predictions(
        m.predict(X_te), y_te, data['norm_stats'], data['raw_prices_test'],
        data['market_indices'], target_type=TARGET_TYPE,
    )
    mo = compute_metrics(po, to)
    results.append(('PCA+SVR', time.time() - t0, mn, mo))

    # 3/9: LSTM (7-dim)
    print("\n3/9: LSTM"); fresh_seed(); t0 = time.time()
    m = train_lstm(fl['train'], fl['val'], data['n_nodes'],
                    data['n_commodities'], device, feat_dim=FEATURE_DIM,
                    num_epochs=args.epochs)['model']
    mn, mo = eval_torch_7d(m, fl['test'], has_graph=False)
    results.append(('LSTM', time.time() - t0, mn, mo))

    # 4/9: BiLSTM (7-dim)
    print("\n4/9: BiLSTM"); fresh_seed(); t0 = time.time()
    m = train_bilstm(fl['train'], fl['val'], data['n_nodes'],
                      data['n_commodities'], device, feat_dim=FEATURE_DIM,
                      num_epochs=args.epochs)['model']
    mn, mo = eval_torch_7d(m, fl['test'], has_graph=False)
    results.append(('BiLSTM', time.time() - t0, mn, mo))

    # 5/9: GCN-Only (7-dim)
    print("\n5/9: GCN-Only"); fresh_seed(); t0 = time.time()
    m = train_gcn_only(fl['train'], fl['val'], ei, ew,
                        data['n_nodes'], data['n_commodities'], device,
                        in_dim=FEATURE_DIM, num_epochs=args.epochs)['model']
    mn, mo = eval_torch_7d(m, fl['test'])
    results.append(('GCN-Only', time.time() - t0, mn, mo))

    # 6/9: GCN+GAT (7-dim)
    print("\n6/9: GCN+GAT"); fresh_seed(); t0 = time.time()
    m = train_gcn_gat(fl['train'], fl['val'], ei, ew,
                       data['n_nodes'], data['n_commodities'], device,
                       in_dim=FEATURE_DIM, num_epochs=args.epochs)['model']
    mn, mo = eval_torch_7d(m, fl['test'])
    results.append(('GCN+GAT', time.time() - t0, mn, mo))

    # 7/9: CMGM_Feature (7-dim, GCN+LSTM both use all features)
    print("\n7/9: CMGM-Feat"); fresh_seed(); t0 = time.time()
    m = CMGM_Feature(data['n_nodes'], data['n_commodities'], feat_dim=FEATURE_DIM).to(device)
    train(m, fl['train'], fl['val'], ei, ew, device,
          num_epochs=args.epochs, patience=args.patience)
    mn, mo = eval_torch_7d(m, fl['test'])
    results.append(('CMGM-Feat', time.time() - t0, mn, mo))

    # 8/9: HeteroMixHop (7-dim, learnable graph)
    print("\n8/9: HeteroMixHop"); fresh_seed(); t0 = time.time()
    d = torch.empty(2, 0, dtype=torch.long), torch.zeros(0)
    m = HeteroMixHopCMGM(data['n_nodes'], data['n_commodities'],
                          n_stock=n_stock, n_bond=n_bond).to(device)
    train(m, fl['train'], fl['val'], d[0], d[1], device,
          num_epochs=args.epochs, patience=args.patience)
    mn, mo = eval_torch_7d(m, fl['test'])
    results.append(('HeteroMix', time.time() - t0, mn, mo))

    # ── Table ──
    print("\n" + "=" * 115)
    print("COMPARISON — Return Space")
    print("=" * 115)
    print(f"{'Model':<16s} {'Time':>7s} {'MAE':>9s} {'RMSE':>9s} "
          f"{'ResMean':>9s} {'ResStd':>9s} {'Hit%':>7s}")
    print("-" * 115)
    for name, t, mn, mo in results:
        hit = mn.get('Hit_Ratio', float('nan'))
        hit_str = f"{hit*100:>6.1f}" if not np.isnan(hit) else "    nan"
        print(f"{name:<16s} {t:>6.1f}s {mn['MAE']:>9.6f} {mn['RMSE']:>9.6f} "
              f"{mn['Residual_Mean']:>9.6f} {mn['Residual_Std']:>9.6f} "
              f"{hit_str}")
    print("=" * 115)

    print("\n" + "=" * 100)
    print("COMPARISON — Original Price Space")
    print("=" * 100)
    print(f"{'Model':<16s} {'MAE':>14s} {'MSE':>18s} {'RMSE':>14s}")
    print("-" * 100)
    for name, t, mn, mo in results:
        print(f"{name:<16s} {mo['MAE']:>14.2f} {mo['MSE']:>18.2f} {mo['RMSE']:>14.2f}")
    print("=" * 100)

    ExperimentLogger().log_run({'version': 'hetero-mixhop-compare-return-v1'}, results)
    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.baselines:
        return run_comparison(args)
    return run_single(args)


if __name__ == '__main__':
    main()
