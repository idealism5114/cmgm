"""
Ablation study for HeteroMixHopCMGM.

Runs 9 variants (full + 8 ablations) with identical config and reports
MAE / RMSE / Hit% / vs-Zero on the primary horizon (5-day return).

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_ablation --epochs 200
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.scripts.main_ablation --epochs 200 --variants full,no_gate,no_mixhop
"""

import argparse, os, sys, time, numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmgm.config import (
    NUM_EPOCHS, BATCH_SIZE, SEQ_LEN, RANDOM_SEED, PATIENCE,
    FEATURE_DIM, TARGET_TYPE, TARGET_HORIZON, MULTI_HORIZONS, FEAT_ZSCORE_EPS,
)
from cmgm.data.data_loader import set_seed, create_data_loaders, compute_features, MarketSequenceDataset
from cmgm.data.feature_builder import build_feature_matrix
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import train
from cmgm.training.evaluate import compute_metrics, inverse_transform_predictions
from cmgm.graph.graph_builder import build_graph
from cmgm.experiment_logger import ExperimentLogger

from torch_geometric.utils import to_dense_adj


def build_ewma_graph(returns: np.ndarray, lam: float, top_k: int = 10) -> torch.Tensor:
    """
    Exponentially-weighted (EWMA) correlation graph.

    Weight w_t = (1−λ)·λ^(T−1−t): recent days matter more.  λ controls
    the time memory of the graph — small λ → short-memory (fast linkages),
    large λ → long-memory (stable structure).

    Returns: dense adjacency (N, N) — top-k, symmetrized, self-loops,
    row-normalized (suitable for EdgeAttnMixHop hard-mask mode).
    """
    T, N = returns.shape
    w = (1.0 - lam) * lam ** np.arange(T - 1, -1, -1)      # (T,)
    w /= w.sum()
    mu = (returns * w[:, None]).sum(axis=0)                 # (N,)
    centered = returns - mu
    cov = (centered * w[:, None]).T @ centered              # (N, N)
    sigma = np.sqrt(np.diag(cov))
    sigma[sigma < 1e-12] = 1e-12
    corr = cov / np.outer(sigma, sigma)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Top-k per row (keep signed correlations)
    A = np.zeros_like(corr)
    idx = np.argsort(np.abs(corr), axis=1)[:, -top_k:]
    for i in range(N):
        A[i, idx[i]] = corr[i, idx[i]]
    # Symmetrize + self-loops + row-normalize
    A = np.maximum(A, A.T)
    A = A + np.eye(N)
    A = A / A.sum(axis=1, keepdims=True).clip(min=1e-8)
    return torch.from_numpy(A.astype(np.float32))

# All variants: (display name, model variant, feat_dim, model kwargs)
ALL_VARIANTS = [
    ("Full",                "full",             FEATURE_DIM, {}),
    ("+EdgeAttn",           "edge_attn",        FEATURE_DIM, {}),
    ("+EdgeAttnStatic",     "edge_attn_static", FEATURE_DIM, {}),
    # ── Attention hyperparameter scan (heads / dropout / prior strength) ──
    ("+AttnH8",             "edge_attn",        FEATURE_DIM, {'attn_heads': 8}),
    ("+AttnH2",             "edge_attn",        FEATURE_DIM, {'attn_heads': 2}),
    ("+AttnDrop05",         "edge_attn",        FEATURE_DIM, {'attn_dropout': 0.05}),
    ("+AttnPrior2",         "edge_attn",        FEATURE_DIM, {'attn_prior_scale': 2.0}),
    ("+AttnPrior05",        "edge_attn",        FEATURE_DIM, {'attn_prior_scale': 0.5}),
    # Cross-validation of the two best directions (heads=8 × prior=0.5)
    ("+AttnH8P05",          "edge_attn",        FEATURE_DIM,
     {'attn_heads': 8, 'attn_prior_scale': 0.5}),
    ("+AttnH8Drop05",       "edge_attn",        FEATURE_DIM,
     {'attn_heads': 8, 'attn_dropout': 0.05}),
    # ── Temporal branch upgrade: per-type compression + temporal attention ──
    ("+TemporalAttn",       "temporal_attn",    FEATURE_DIM, {}),
    # Multi-scale temporal pooling: last + full + 10-step + 5-step means
    ("+MultiScaleT",        "multiscale_time",  FEATURE_DIM, {}),
    # Diff input: concat first-order differences to LSTM input
    ("+DiffInput",          "diff_input",       FEATURE_DIM, {}),
    # Horizon-aligned output: per-horizon context window + per-horizon head
    ("+HorizonAlign",       "horizon_align",    FEATURE_DIM, {}),
    # Multi-scale TCN replaces the LSTM temporal branch
    ("+TCNTemporal",        "tcn_temporal",     FEATURE_DIM, {}),
    # PatchTST-style temporal branch (patch_len=5 aligned with 5d horizon)
    ("+PatchTST",           "patch_temporal",   FEATURE_DIM, {}),
    # Multi-scale attention pooling (attention in window instead of mean)
    ("+AttnPool",           "attn_pool",        FEATURE_DIM, {}),
    # Multi-scale graph: dual spatial branches (EWMA short λ=0.9 / long λ=0.99)
    ("+MultiScaleGraph",    "multiscale_graph", FEATURE_DIM, {}),
    # ── Component ablations ──
    ("-TypeProj",           "no_type_proj",     FEATURE_DIM, {}),
    ("-LearnGraph",         "no_learn_graph",   FEATURE_DIM, {}),
    ("-MixHop",             "no_mixhop",        FEATURE_DIM, {}),
    ("-Gate",               "no_gate",          FEATURE_DIM, {}),
    ("-GCN",                "lstm_only",        FEATURE_DIM, {}),
    ("-LSTM",               "gcn_only",         FEATURE_DIM, {}),
    ("-MultiHorizon",       "single_horizon",   FEATURE_DIM, {}),
    ("-Feat21",             "feat7",            7,           {}),
]


def parse_args():
    p = argparse.ArgumentParser(description='HeteroMixHopCMGM ablation study')
    p.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--seq-len', type=int, default=SEQ_LEN)
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    p.add_argument('--patience', type=int, default=PATIENCE)
    p.add_argument('--no-cuda', action='store_true')
    p.add_argument('--variants', type=str, default=None,
                   help='Comma-separated display names, e.g. "Full,no_gate"')
    return p.parse_args()


def build_data(args, feat_dim):
    """Build train/val/test loaders for a given feature dim."""
    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)

    raw_full = np.concatenate([
        data['raw_prices_train'], data['raw_prices_val'], data['raw_prices_test'],
    ], axis=0)

    # ── Features: 21-dim (build_feature_matrix) or 7-dim (compute_features) ──
    if feat_dim == 21:
        feat_raw, _ = build_feature_matrix(raw_full)          # (T, N, 21)
    else:
        feat_raw = compute_features(raw_full)                 # (T, N, 7)

    train_sz = data['raw_prices_train'].shape[0]
    feat_train = feat_raw[:train_sz]
    feat_mean = feat_train.mean(axis=(0,), keepdims=True)
    feat_std  = np.maximum(feat_train.std(axis=(0,), keepdims=True), FEAT_ZSCORE_EPS)
    feat_tensor = (feat_raw - feat_mean) / feat_std

    # ── Z-scored prices (for X base) ──
    norm_mean = data['norm_stats']['mean']
    norm_std  = data['norm_stats']['std']
    full_norm = (raw_full - norm_mean) / norm_std

    T = feat_raw.shape[0]
    tr, va = int(T * 0.7), int(T * 0.7) + int(T * 0.15)
    feat_splits = [feat_tensor[:tr], feat_tensor[tr:va], feat_tensor[va:]]
    norm_splits = [full_norm[:tr], full_norm[tr:va], full_norm[va:]]
    raw_splits  = [data['raw_prices_train'],
                   data['raw_prices_val'],
                   data['raw_prices_test']]

    # Multi-horizon for return targets; single primary horizon otherwise
    use_multi = (TARGET_TYPE == "return")
    hrz = MULTI_HORIZONS if use_multi else [TARGET_HORIZON]

    dss = {k: MarketSequenceDataset(
               n, data['market_indices'], args.seq_len,
               feature_matrix=f, raw_prices=r, target_type=TARGET_TYPE,
               horizons=hrz,
           )
           for k, n, f, r in zip(['train','val','test'],
                                  norm_splits, feat_splits, raw_splits)}
    loaders = {k: DataLoader(dss[k], batch_size=args.batch_size, shuffle=False,
                             drop_last=(k=='train')) for k in ['train','val','test']}

    data['loaders'] = loaders
    data['feat_splits'] = feat_splits
    data['norm_splits'] = norm_splits
    data['raw_splits'] = raw_splits
    return data


def build_single_horizon_data(args, data, feat_splits, norm_splits, raw_splits):
    """Loaders with only the primary horizon (for single_horizon variant)."""
    dss = {k: MarketSequenceDataset(
               n, data['market_indices'], args.seq_len,
               feature_matrix=f, raw_prices=r, target_type=TARGET_TYPE,
               horizons=[TARGET_HORIZON],
           )
           for k, n, f, r in zip(['train','val','test'],
                                  norm_splits, feat_splits, raw_splits)}
    return {k: DataLoader(dss[k], batch_size=args.batch_size, shuffle=False,
                          drop_last=(k=='train')) for k in ['train','val','test']}


def evaluate_primary_horizon(model, loader, data, device):
    """Evaluate on primary horizon (5d) — handles multi and single output."""
    h_idx = MULTI_HORIZONS.index(TARGET_HORIZON)
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in loader:
            X_batch, y_batch = batch[0], batch[1]
            X_batch = X_batch.to(device)
            pred = model(X_batch)                    # (B, H, Nc) or (B, Nc)
            pred_np = pred.cpu().numpy()
            y_np = y_batch.numpy()
            if pred_np.ndim == 3:
                pred_np = pred_np[:, h_idx, :]
            if y_np.ndim == 3:
                y_np = y_np[:, h_idx, :]
            all_p.append(pred_np)
            all_t.append(y_np)
    p, t = np.concatenate(all_p), np.concatenate(all_t)
    mn = compute_metrics(p, t)
    po, to = inverse_transform_predictions(
        p, t, data['norm_stats'], data['raw_prices_test'],
        data['market_indices'], target_type=TARGET_TYPE,
    )
    return mn, compute_metrics(po, to)


def run_variant(name, variant, feat_dim, kwargs, args, device, data21, data7):
    set_seed(args.seed)
    print(f"\n{'=' * 90}")
    print(f"  ABLATION: {name}  (variant={variant}, feat_dim={feat_dim}, {kwargs})")
    print(f"{'=' * 90}")
    t0 = time.time()

    # ── Data ──
    feat_splits = data21['feat_splits']
    norm_splits = data21['norm_splits']
    raw_splits  = data21['raw_splits']
    if feat_dim == 7:
        data = data7
        feat_splits = data7['feat_splits']
        norm_splits = data7['norm_splits']
        raw_splits  = data7['raw_splits']
    else:
        data = data21

    if variant == "single_horizon":
        loaders = build_single_horizon_data(args, data, feat_splits, norm_splits, raw_splits)
    else:
        loaders = data['loaders'] if feat_dim == 21 else data7['loaders']

    # ── Model ──
    n_stock = data['market_indices']['stock'][1] - data['market_indices']['stock'][0]
    n_bond  = data['market_indices']['bond'][1] - data['market_indices']['bond'][0]
    model = HeteroMixHopCMGM(data['n_nodes'], data['n_commodities'],
                             n_stock=n_stock, n_bond=n_bond,
                             variant=variant, feat_dim=feat_dim, **kwargs).to(device)

    # ── Static graph for variants without adaptive learner ──
    if variant in ("no_learn_graph", "edge_attn_static"):
        graph = build_graph(data['train_returns'], data['market_indices'], method='pearson')
        ei, ew = graph['edge_index'], graph['edge_weight']
        A = to_dense_adj(ei, edge_attr=ew)[0].to(device)   # (N, N)
        model.static_A.copy_(A)
        print(f"  [Static graph] Pearson top-10 dense adjacency: {tuple(A.shape)}")

    # ── Dual EWMA graphs for multi-scale graph variant ──
    if variant == "multiscale_graph":
        A_short = build_ewma_graph(data['train_returns'], lam=0.9).to(device)
        A_long  = build_ewma_graph(data['train_returns'], lam=0.99).to(device)
        model.static_A_short.copy_(A_short)
        model.static_A_long.copy_(A_long)
        print(f"  [EWMA graphs] short λ=0.9: {tuple(A_short.shape)}  "
              f"long λ=0.99: {tuple(A_long.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # ── Train ──
    dummy = torch.empty(2, 0, dtype=torch.long), torch.zeros(0)
    history = train(model, loaders['train'], loaders['val'],
                    dummy[0], dummy[1], device,
                    num_epochs=args.epochs, patience=args.patience)

    # ── Evaluate on primary horizon ──
    mn, mo = evaluate_primary_horizon(model, loaders['test'], data, device)

    # ── Zero baseline (always 0 for returns, mean vol otherwise) ──
    # NOTE: must use PRIMARY horizon only — y from multi-horizon loaders is (N, H, Nc)
    from cmgm.baselines.traditional import prepare_sklearn_data
    X_te, y_te = prepare_sklearn_data(loaders['test'])
    if y_te.ndim == 3:
        h_idx = MULTI_HORIZONS.index(TARGET_HORIZON)
        y_te = y_te[:, h_idx, :]
    if TARGET_TYPE == "volatility":
        zero_preds = np.tile(y_te.mean(axis=0, keepdims=True), (len(y_te), 1))
    else:
        zero_preds = np.zeros_like(y_te)
    mn_zero = compute_metrics(zero_preds, y_te)
    vs_zero = (mn['MAE'] - mn_zero['MAE']) / mn_zero['MAE'] * 100

    hit = mn.get('Hit_Ratio', float('nan'))
    result = {
        'variant': name,
        'params': n_params,
        'time': time.time() - t0,
        'MAE': mn['MAE'],
        'RMSE': mn['RMSE'],
        'MSE': mn['MSE'],
        'Hit_Ratio': hit,
        'vs_zero_pct': vs_zero,
        'mn': mn,     # full metrics dict for logger
        'mo': mo,     # full original-space metrics dict for logger
    }
    print(f"\n  ── Result ──")
    print(f"  MAE: {mn['MAE']:.6f}  RMSE: {mn['RMSE']:.6f}  "
          f"Hit%: {hit*100:.1f}  vs Zero: {vs_zero:+.2f}%  [{time.time()-t0:.0f}s]")
    return result


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}  |  Epochs: {args.epochs}  |  Seed: {args.seed}")
    print(f"Target: {TARGET_TYPE}  |  Primary horizon: {TARGET_HORIZON}d  |  "
          f"Multi-horizons: {MULTI_HORIZONS}")

    # Filter variants if requested
    variants = ALL_VARIANTS
    if args.variants:
        requested = [v.strip() for v in args.variants.split(',')]
        variants = [v for v in ALL_VARIANTS if v[1] in requested or v[0] in requested]
        print(f"Filtered to {len(variants)} variants: {[v[0] for v in variants]}")

    # ── Build data once (21-dim and 7-dim) ──
    print("\n[Data] Building 21-dim feature pipeline...")
    data21 = build_data(args, 21)
    print("\n[Data] Building 7-dim feature pipeline...")
    data7 = build_data(args, 7)

    # ── Run all variants ──
    results = []
    for name, variant, feat_dim, kwargs in variants:
        try:
            r = run_variant(name, variant, feat_dim, kwargs, args, device, data21, data7)
            results.append(r)
        except Exception as e:
            print(f"\n[FAILED] {name}: {e}")
            import traceback; traceback.print_exc()
            results.append({'variant': name, 'MAE': float('nan'),
                            'RMSE': float('nan'), 'MSE': float('nan'),
                            'Hit_Ratio': float('nan'), 'vs_zero_pct': float('nan'),
                            'params': 0, 'time': 0,
                            'mn': {'MAE': float('nan'), 'MSE': float('nan'),
                                   'RMSE': float('nan'), 'Residual_Mean': float('nan'),
                                   'Residual_Std': float('nan'), 'Skewness': float('nan')},
                            'mo': {'MAE': float('nan'), 'MSE': float('nan'),
                                   'RMSE': float('nan')}})

    # ── Summary table ──
    print("\n" + "=" * 100)
    print("ABLATION STUDY — 5-day Return Space")
    print("=" * 100)
    print(f"{'Variant':<18s} {'Params':>10s} {'Time':>7s} {'MAE':>10s} "
          f"{'RMSE':>10s} {'Hit%':>7s} {'vs Zero':>10s}")
    print("-" * 100)
    for r in results:
        hit = r['Hit_Ratio']
        hit_str = f"{hit*100:>6.1f}" if not np.isnan(hit) else "    nan"
        vs = r['vs_zero_pct']
        vs_str = f"{vs:>+9.2f}%" if not np.isnan(vs) else "      nan"
        print(f"{r['variant']:<18s} {r['params']:>10,d} {r['time']:>6.0f}s "
              f"{r['MAE']:>10.6f} {r['RMSE']:>10.6f} {hit_str} {vs_str}")
    print("=" * 100)

    # ── Log ──
    ExperimentLogger().log_run(
        {'version': 'ablation-v1', 'epochs': args.epochs, 'seed': args.seed,
         'target': TARGET_TYPE, 'horizon': TARGET_HORIZON},
        [(r['variant'], r['time'], r['mn'], r['mo']) for r in results],
    )
    return results


if __name__ == '__main__':
    main()
