"""
Paper-level analysis of CMGM baseline comparison.

Produces:
  - Per-commodity error breakdown (Table 1)
  - Statistical significance tests — Diebold-Mariano (Table 2)
  - Residual diagnostics — histogram, Q-Q, ACF (Figures 1-3)
  - Directional accuracy (Table 3)
  - Error CDF comparison (Figure 4)
  - Error over time (Figure 5)
  - HeteroMixHop gate analysis (Figure 6)
  - Scatter: predicted vs actual (Figure 7)
"""

import argparse, os, sys, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, jarque_bera, shapiro
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from cmgm.config import (
    NUM_EPOCHS, BATCH_SIZE, SEQ_LEN, RANDOM_SEED, PATIENCE,
    FEATURE_DIM, TRAIN_RATIO, TEST_RATIO, VAL_RATIO,
)
from cmgm.data.data_loader import (
    set_seed, create_data_loaders, compute_features, MarketSequenceDataset,
)
from cmgm.graph.graph_builder import build_graph
from cmgm.training.train import train
from cmgm.training.evaluate import compute_metrics, inverse_transform_predictions
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.feature_model import CMGM_Feature
from cmgm.models.model import MixHopPropagation
from cmgm.baselines.traditional import train_linear_regression, train_svr, prepare_sklearn_data
from cmgm.baselines.deep_learning import train_lstm, train_bilstm
from cmgm.baselines.graph import train_gcn_only, train_gcn_gat

OUTPUT = Path('experiments/paper_analysis')
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'figure.figsize': (8, 5),
    'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 12, 'legend.fontsize': 10,
})
COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f']
MODEL_NAMES = ['PCA+Ridge','PCA+SVR','LSTM','BiLSTM','GCN-Only','GCN+GAT','CMGM-Feat','HeteroMix']


# =========================================================================
# 1. Train all models and collect per-sample predictions
# =========================================================================

def collect_predictions(args):
    print("=" * 80)
    print("COLLECTING PREDICTIONS — 8 models")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    set_seed(args.seed)

    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    graph = build_graph(data['train_returns'], data['market_indices'], method=args.method)
    ei, ew = graph['edge_index'], graph['edge_weight']
    cs, ce = data['market_indices']['commodity']
    n_stock = data['market_indices']['stock'][1] - data['market_indices']['stock'][0]
    n_bond  = data['market_indices']['bond'][1] - data['market_indices']['bond'][0]

    # ── 7-dim features from RAW prices ──
    raw_full = np.concatenate([
        data['raw_prices_train'], data['raw_prices_val'], data['raw_prices_test'],
    ], axis=0)
    feat_raw = compute_features(raw_full)
    train_sz = data['raw_prices_train'].shape[0]
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(feat_raw[:train_sz].reshape(-1, 7))
    feat_tensor = feat_scaler.transform(feat_raw.reshape(-1, 7)).reshape(feat_raw.shape)
    full_norm = data['scaler'].transform(raw_full)

    T = feat_raw.shape[0]
    tr, va = int(T * 0.7), int(T * 0.7) + int(T * 0.15)
    feat_splits = [feat_tensor[:tr], feat_tensor[tr:va], feat_tensor[va:]]
    norm_splits = [full_norm[:tr], full_norm[tr:va], full_norm[va:]]
    dss = {k: MarketSequenceDataset(n, data['market_indices'], args.seq_len, feature_matrix=f)
           for k, n, f in zip(['train','val','test'], norm_splits, feat_splits)}
    fl = {k: DataLoader(dss[k], batch_size=args.batch_size, shuffle=False,
                         drop_last=(k=='train')) for k in ['train','val','test']}

    def eval(model, loader, has_graph=True):
        model.eval()
        all_p, all_t = [], []
        is_internal = hasattr(model, 'graph_learner')
        with torch.no_grad():
            for batch in loader:
                Xb, yb = batch[0].to(device), batch[1]
                if is_internal:
                    pred = model(Xb)
                elif has_graph:
                    pred = model(Xb, ei.to(device), ew.to(device))
                else:
                    pred = model(Xb)
                all_p.append(pred.cpu().numpy())
                all_t.append(yb.numpy())
        p = np.concatenate(all_p)
        t = np.concatenate(all_t)
        po, to = inverse_transform_predictions(p, t, data['scaler'], cs, ce)
        return p, t, po, to

    all_preds = {}  # model_name -> {norm_pred, norm_true, orig_pred, orig_true}

    def fresh_seed():
        set_seed(args.seed)

    # 1/8 PCA+Ridge
    print("\n1/8: PCA+Ridge"); fresh_seed()
    m = train_linear_regression(fl['train'], fl['val'], data['n_commodities'])['model']
    X_te, y_te = prepare_sklearn_data(fl['test'])
    p_norm = m.predict(X_te)
    t_norm = y_te
    po, to = inverse_transform_predictions(p_norm, t_norm, data['scaler'], cs, ce)
    all_preds['PCA+Ridge'] = {'norm_pred': p_norm, 'norm_true': t_norm, 'orig_pred': po, 'orig_true': to}

    # 2/8 PCA+SVR
    print("2/8: PCA+SVR"); fresh_seed()
    m = train_svr(fl['train'], fl['val'], data['n_commodities'])['model']
    X_te, y_te = prepare_sklearn_data(fl['test'])
    p_norm = m.predict(X_te)
    t_norm = y_te
    po, to = inverse_transform_predictions(p_norm, t_norm, data['scaler'], cs, ce)
    all_preds['PCA+SVR'] = {'norm_pred': p_norm, 'norm_true': t_norm, 'orig_pred': po, 'orig_true': to}

    # 3/8 LSTM
    print("3/8: LSTM"); fresh_seed()
    m = train_lstm(fl['train'], fl['val'], data['n_nodes'],
                    data['n_commodities'], device, feat_dim=FEATURE_DIM,
                    num_epochs=args.epochs)['model']
    p_norm, t_norm, po, to = eval(m, fl['test'], has_graph=False)
    all_preds['LSTM'] = {'norm_pred': p_norm, 'norm_true': t_norm, 'orig_pred': po, 'orig_true': to}

    # 4/8 BiLSTM
    print("4/8: BiLSTM"); fresh_seed()
    m = train_bilstm(fl['train'], fl['val'], data['n_nodes'],
                      data['n_commodities'], device, feat_dim=FEATURE_DIM,
                      num_epochs=args.epochs)['model']
    p_norm, t_norm, po, to = eval(m, fl['test'], has_graph=False)
    all_preds['BiLSTM'] = {'norm_pred': p_norm, 'norm_true': t_norm, 'orig_pred': po, 'orig_true': to}

    # 5/8 GCN-Only
    print("5/8: GCN-Only"); fresh_seed()
    m = train_gcn_only(fl['train'], fl['val'], ei, ew,
                        data['n_nodes'], data['n_commodities'], device,
                        in_dim=FEATURE_DIM, num_epochs=args.epochs)['model']
    p_norm, t_norm, po, to = eval(m, fl['test'])
    all_preds['GCN-Only'] = {'norm_pred': p_norm, 'norm_true': t_norm, 'orig_pred': po, 'orig_true': to}

    # 6/8 GCN+GAT
    print("6/8: GCN+GAT"); fresh_seed()
    m = train_gcn_gat(fl['train'], fl['val'], ei, ew,
                       data['n_nodes'], data['n_commodities'], device,
                       in_dim=FEATURE_DIM, num_epochs=args.epochs)['model']
    p_norm, t_norm, po, to = eval(m, fl['test'])
    all_preds['GCN+GAT'] = {'norm_pred': p_norm, 'norm_true': t_norm, 'orig_pred': po, 'orig_true': to}

    # 7/8 CMGM-Feat
    print("7/8: CMGM-Feat"); fresh_seed()
    m = CMGM_Feature(data['n_nodes'], data['n_commodities'], feat_dim=FEATURE_DIM).to(device)
    train(m, fl['train'], fl['val'], ei, ew, device,
          num_epochs=args.epochs, patience=args.patience)
    p_norm, t_norm, po, to = eval(m, fl['test'])
    all_preds['CMGM-Feat'] = {'norm_pred': p_norm, 'norm_true': t_norm, 'orig_pred': po, 'orig_true': to}

    # 8/8 HeteroMixHop
    print("8/8: HeteroMixHop"); fresh_seed()
    d = torch.empty(2, 0, dtype=torch.long), torch.zeros(0)
    m = HeteroMixHopCMGM(data['n_nodes'], data['n_commodities'],
                          n_stock=n_stock, n_bond=n_bond).to(device)
    train(m, fl['train'], fl['val'], d[0], d[1], device,
          num_epochs=args.epochs, patience=args.patience)
    p_norm, t_norm, po, to = eval(m, fl['test'], has_graph=False)

    # Extra: collect gate stats per sample
    gate_samples = []
    m.eval()
    with torch.no_grad():
        for batch in fl['test']:
            Xb = batch[0].to(device)
            A = m.graph_learner()
            x_gcn = Xb.mean(dim=0).permute(1, 0, 2)
            x_proj = m.type_proj(x_gcn, m.n_stock, m.n_bond)
            x_proj = x_proj.mean(dim=1)
            h1 = torch.relu(m.mixhop1(x_proj, A))
            h2 = m.mixhop2(h1, A)
            h = m.gcn_norm(h2)
            B = Xb.size(0)
            gcn_out = m.type_pool(h).unsqueeze(0).expand(B, -1)
            x_seq = Xb.reshape(B, Xb.size(1), -1)
            lstm_out, (hn, _) = m.temporal(x_seq)
            lstm_out = hn[-1]
            combined = torch.cat([gcn_out, lstm_out], dim=-1)
            gate = torch.sigmoid(m.gate_fc(combined))
            gate_samples.append(gate.cpu().numpy())
    all_preds['HeteroMix'] = {
        'norm_pred': p_norm, 'norm_true': t_norm, 'orig_pred': po, 'orig_true': to,
        'gate': np.concatenate(gate_samples),
    }

    np.savez(OUTPUT / 'all_predictions.npz',
             **{k: {sk: sv for sk, sv in v.items()}
                for k, v in all_preds.items()})

    # Commodity names
    feat_names = data.get('feature_names', [])
    commodity_names = feat_names[data['market_indices']['commodity'][0]:
                                  data['market_indices']['commodity'][1]]
    print(f"\nPredictions saved. Commodities: {list(commodity_names)}")
    return all_preds, commodity_names


# =========================================================================
# 2. Analysis functions
# =========================================================================

def per_commodity_metrics(all_preds, commodity_names):
    """Table 1: Error metrics for each commodity across all models."""
    print("\n" + "=" * 80)
    print("TABLE 1 — Per-Commodity RMSE (Original Price)")
    print("=" * 80)

    rows = []
    for i, cname in enumerate(commodity_names):
        row = {'Commodity': cname}
        for mname in MODEL_NAMES:
            err = all_preds[mname]['orig_true'][:, i] - all_preds[mname]['orig_pred'][:, i]
            row[f'{mname}_RMSE'] = np.sqrt(np.mean(err ** 2))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / 'table1_per_commodity_rmse.csv', index=False)
    print(df.round(1).to_string(index=False))
    return df


def per_commodity_summary_table(all_preds, commodity_names):
    """Compact table: each commodity's best model and improvement vs worst."""
    print("\n" + "=" * 80)
    print("TABLE 2 — Best/Worse per Commodity")
    print("=" * 80)

    rows = []
    for i, cname in enumerate(commodity_names):
        rmse_dict = {}
        for mname in MODEL_NAMES:
            err = all_preds[mname]['orig_true'][:, i] - all_preds[mname]['orig_pred'][:, i]
            rmse_dict[mname] = np.sqrt(np.mean(err ** 2))
        best = min(rmse_dict, key=rmse_dict.get)
        worst = max(rmse_dict, key=rmse_dict.get)
        rows.append({
            'Commodity': cname,
            'Best_Model': best,
            'Best_RMSE': round(rmse_dict[best], 1),
            'Worst_Model': worst,
            'Worst_RMSE': round(rmse_dict[worst], 1),
            'Improvement_%': round((1 - rmse_dict[best]/rmse_dict[worst]) * 100, 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / 'table2_best_worst_per_commodity.csv', index=False)
    print(df.to_string(index=False))
    return df


def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.
    H0: Both models have equal forecast accuracy.
    e1, e2: (T, N) error matrices (true - pred)
    Returns: DM statistic and p-value for each commodity
    """
    T = e1.shape[0]
    d = np.abs(e1) - np.abs(e2)  # loss differential (MAE)
    d_bar = d.mean(axis=0)
    if T <= 1:
        return np.zeros(e1.shape[1]), np.ones(e1.shape[1])
    # Newey-West type variance
    gamma = []
    for lag in range(min(h + 1, T)):
        d_lag = d[lag:] - d_bar
        d_lead = d[:T - lag] - d_bar
        cov = (d_lag * d_lead).mean(axis=0)
        gamma.append(cov)
    var_d = gamma[0] + 2 * sum(gamma[1:]) if len(gamma) > 1 else gamma[0]
    var_d = np.clip(var_d, 1e-10, None)
    dm_stat = d_bar / np.sqrt(var_d / T)
    p_val = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_val


def statistical_significance_test(all_preds, commodity_names):
    """Table 3: Diebold-Mariano test — HeteroMix vs each baseline."""
    print("\n" + "=" * 80)
    print("TABLE 3 — Diebold-Mariano Test (HeteroMix vs Baselines)")
    print("=" * 80)
    print("H0: Equal predictive accuracy. DM > 1.96 → HeteroMix better at 5%")

    baseline_names = [n for n in MODEL_NAMES if n != 'HeteroMix']
    e_ours = (all_preds['HeteroMix']['norm_true'] - all_preds['HeteroMix']['norm_pred'])

    rows = []
    for bname in baseline_names:
        e_base = all_preds[bname]['norm_true'] - all_preds[bname]['norm_pred']
        dm_stats, p_vals = diebold_mariano_test(e_ours, e_base, h=1)
        n_sig = (dm_stats > 1.96).sum()
        n_total = len(dm_stats)
        avg_dm = dm_stats.mean()
        rows.append({
            'Baseline': bname,
            'Avg_DM': f'{avg_dm:.3f}',
            'DM>1.96': f'{n_sig}/{n_total}',
            'Ratio_%': f'{n_sig/n_total*100:.0f}%',
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / 'table3_diebold_mariano.csv', index=False)
    print(df.to_string(index=False))
    return df


def directional_accuracy(all_preds, commodity_names):
    """Table 4: % of correct direction predictions."""
    print("\n" + "=" * 80)
    print("TABLE 4 — Directional Accuracy (%)")
    print("=" * 80)

    rows = []
    for mname in MODEL_NAMES:
        t = all_preds[mname]['orig_true']
        p = all_preds[mname]['orig_pred']
        # Direction: actual movement (today vs yesterday)
        actual_dir = np.sign(np.diff(t, axis=0))
        pred_dir = np.sign(p[1:] - t[:-1])  # pred_t+1 vs actual_t
        correct = (actual_dir == pred_dir) & (actual_dir != 0)
        valid = actual_dir != 0
        acc = correct.sum() / valid.sum() * 100 if valid.sum() > 0 else 0
        rows.append({'Model': mname, 'Directional_Accuracy_%': f'{acc:.2f}'})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / 'table4_directional_accuracy.csv', index=False)
    print(df.to_string(index=False))
    return df


def residual_diagnostics(all_preds, commodity_names):
    """Figures: residual distribution, Q-Q plot, ACF."""
    print("\nResidual diagnostics...")

    # Pick HeteroMix and one baseline (PCA+Ridge) for comparison
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    models_plot = ['PCA+Ridge', 'HeteroMix']
    titles = ['PCA+Ridge (Baseline)', 'HeteroMixHop (Ours)']

    for idx, (mname, axrow) in enumerate(zip(models_plot, axes)):
        resid = (all_preds[mname]['orig_true'] - all_preds[mname]['orig_pred']).ravel()

        # Histogram
        axrow[0].hist(resid, bins=80, density=True, alpha=0.7, color=COLORS[0])
        x_range = np.linspace(resid.min(), resid.max(), 100)
        mu, std = resid.mean(), resid.std()
        axrow[0].plot(x_range, norm.pdf(x_range, mu, std), 'r-', lw=2, label=f'N({mu:.0f},{std:.0f})')
        axrow[0].set_xlabel('Residual')
        axrow[0].set_ylabel('Density')
        axrow[0].set_title(f'{titles[idx]} — Residual Distribution')
        axrow[0].legend(fontsize=8)

        # Q-Q plot
        stats.probplot(resid[::100], dist="norm", plot=axrow[1])
        axrow[1].set_title(f'{titles[idx]} — Q-Q Plot')

        # ACF
        sample_resid = resid[:500:5] if len(resid) > 500 else resid
        plot_acf(sample_resid, lags=40, ax=axrow[2], alpha=0.05)
        axrow[2].set_title(f'{titles[idx]} — Residual ACF')

    plt.tight_layout()
    fig.savefig(OUTPUT / 'fig1_residual_diagnostics.png', bbox_inches='tight')
    plt.close()
    print("  -> fig1_residual_diagnostics.png")

    # JB test
    for mname in models_plot:
        resid = (all_preds[mname]['orig_true'] - all_preds[mname]['orig_pred']).ravel()
        jb_stat, jb_p = jarque_bera(resid[::10])
        print(f"  {mname}: Jarque-Bera stat={jb_stat:.1f}, p={jb_p:.2e} (normality {'rejected' if jb_p<0.05 else 'not rejected'})")


def error_cdf_comparison(all_preds):
    """Figure: CDF of absolute errors for all models."""
    print("\nError CDF comparison...")
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, mname in enumerate(MODEL_NAMES):
        abs_err = np.abs(all_preds[mname]['orig_true'] - all_preds[mname]['orig_pred']).ravel()
        sorted_err = np.sort(abs_err)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(sorted_err, cdf, color=COLORS[i], lw=1.5, label=f'{mname}')

    ax.set_xlabel('Absolute Error (Original Price)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution of Absolute Errors')
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(20000, np.percentile(abs_err, 99)))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT / 'fig2_error_cdf.png', bbox_inches='tight')
    plt.close()
    print("  -> fig2_error_cdf.png")


def gate_analysis(all_preds):
    """Figure: HeteroMixHop gate distribution."""
    if 'HeteroMix' not in all_preds or 'gate' not in all_preds['HeteroMix']:
        print("  No gate data available, skipping.")
        return

    print("\nGate analysis...")
    gate = all_preds['HeteroMix']['gate']  # (n_samples, 64) -> average
    gate_mean = gate.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(gate_mean, bins=50, alpha=0.7, color='#1f77b4', edgecolor='white')
    axes[0].axvline(gate_mean.mean(), color='r', ls='--', lw=2,
                    label=f'Mean={gate_mean.mean():.3f}')
    axes[0].set_xlabel('Gate Value (0=LSTM, 1=GCN)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Gating Weight Distribution')
    axes[0].legend()

    # Gate vs time
    axes[1].plot(gate_mean[:200], color='#1f77b4', lw=0.8)
    axes[1].axhline(0.5, color='gray', ls='--', alpha=0.5)
    axes[1].set_xlabel('Test Sample (first 200)')
    axes[1].set_ylabel('Gate Value')
    axes[1].set_title('Gating Weight Over Time')
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(OUTPUT / 'fig3_gate_analysis.png', bbox_inches='tight')
    plt.close()
    print(f"  Gate mean={gate_mean.mean():.3f}, std={gate_mean.std():.3f}")
    print("  -> fig3_gate_analysis.png")


def error_over_time(all_preds, commodity_names):
    """Figure: Rolling RMSE over test period for top models."""
    print("\nError over time...")
    fig, ax = plt.subplots(figsize=(12, 5))

    best_models = ['PCA+Ridge', 'GCN+GAT', 'CMGM-Feat', 'HeteroMix']
    window = 20

    for mname in best_models:
        err = (all_preds[mname]['orig_true'] - all_preds[mname]['orig_pred'])
        se = err ** 2  # (T, N)
        rolling_mse = pd.DataFrame(se).rolling(window, min_periods=1).mean().values
        rolling_rmse = np.sqrt(rolling_mse.mean(axis=1))
        ax.plot(rolling_rmse, label=f'{mname}', lw=1.5)

    ax.set_xlabel('Test Sample')
    ax.set_ylabel(f'Rolling RMSE (window={window})')
    ax.set_title('Prediction Error Over Test Period')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT / 'fig4_error_over_time.png', bbox_inches='tight')
    plt.close()
    print("  -> fig4_error_over_time.png")


def scatter_pred_vs_actual(all_preds, commodity_names, n_examples=4):
    """Figure: Predicted vs actual scatter for selected commodities."""
    print("\nScatter plots...")
    # Pick the most and least volatile commodities
    vol = all_preds['HeteroMix']['orig_true'].std(axis=0)
    indices = [vol.argmax(), vol.argmin(), 0, min(5, len(commodity_names)-1)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    models_plot = ['PCA+Ridge', 'CMGM-Feat', 'HeteroMix']
    colors_plot = ['#d62728', '#e377c2', '#1f77b4']

    for idx, ax in enumerate(axes.ravel()):
        ci = indices[idx]
        cname = commodity_names[ci]
        for mi, mname in enumerate(models_plot):
            t = all_preds[mname]['orig_true'][:, ci]
            p = all_preds[mname]['orig_pred'][:, ci]
            ax.scatter(t, p, s=3, alpha=0.4, color=colors_plot[mi], label=mname)
        ax.plot([t.min(), t.max()], [t.min(), t.max()], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{cname}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.tight_layout()
    fig.savefig(OUTPUT / 'fig5_scatter_pred_vs_actual.png', bbox_inches='tight')
    plt.close()
    print("  -> fig5_scatter_pred_vs_actual.png")


def summary_mse_table(all_preds):
    """Final aggregate metrics table."""
    print("\n" + "=" * 80)
    print("TABLE 5 — Aggregate Metrics Summary")
    print("=" * 80)
    rows = []
    for mname in MODEL_NAMES:
        t = all_preds[mname]['orig_true']
        p = all_preds[mname]['orig_pred']
        err = t - p
        mae = np.mean(np.abs(err))
        mse = np.mean(err ** 2)
        rmse = np.sqrt(mse)
        # Directional accuracy
        actual_dir = np.sign(np.diff(t, axis=0))
        pred_dir = np.sign(p[1:] - t[:-1])
        correct = (actual_dir == pred_dir) & (actual_dir != 0)
        valid = actual_dir != 0
        dir_acc = correct.sum() / valid.sum() * 100 if valid.sum() > 0 else 0
        rows.append({
            'Model': mname,
            'MAE': f'{mae:.1f}',
            'MSE': f'{mse:.1f}',
            'RMSE': f'{rmse:.1f}',
            'DirAcc%': f'{dir_acc:.2f}',
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / 'table5_aggregate_metrics.csv', index=False)
    print(df.to_string(index=False))
    return df


# =========================================================================
# Main
# =========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='pearson')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seq-len', type=int, default=20)
    p.add_argument('--no-cuda', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=10)
    args = p.parse_args()

    # Step 1: Train all models & collect predictions
    all_preds, commodity_names = collect_predictions(args)

    # Step 2: Analyses
    summary_mse_table(all_preds)

    per_commodity_metrics(all_preds, commodity_names)
    per_commodity_summary_table(all_preds, commodity_names)

    statistical_significance_test(all_preds, commodity_names)
    directional_accuracy(all_preds, commodity_names)

    # Step 3: Figures
    residual_diagnostics(all_preds, commodity_names)
    error_cdf_comparison(all_preds)
    gate_analysis(all_preds)
    error_over_time(all_preds, commodity_names)
    scatter_pred_vs_actual(all_preds, commodity_names)

    print(f"\n{'=' * 80}")
    print(f"ALL OUTPUTS SAVED TO: {OUTPUT.resolve()}")
    print(f"{'=' * 80}")
    print(f"Files:")
    for f in sorted(OUTPUT.iterdir()):
        print(f"  {f.name}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
