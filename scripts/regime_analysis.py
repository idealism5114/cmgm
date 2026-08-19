"""
Regime analysis: does the stock-bond-futures cross-market relationship
change systematically with market state, and does the temporal structure
of 5-day futures returns differ across states?

Pipeline:
  1. Load the three markets (reuses cmgm.data.data_loader loaders)
  2. Build regime features (rolling vol / return / cross-market correlations)
  3. Assign regimes: Gaussian HMM (primary) + KMeans (cross-check)
  4. Per-regime statistics:
       - daily & 5-day return ACF
       - momentum (rolling mean return)
       - volatility
       - cross-market lead-lag (stock→futures, bond→futures, lags −5..+5)
       - 5-day return predictability (in-regime OLS vs global OLS)
       - optimal look-back window (AR windows 5/10/20/40)
  5. Print tables + save CSVs to experiments/regime_analysis/

Run:  python scripts/regime_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from hmmlearn.hmm import GaussianHMM

from cmgm.data.data_loader import (
    load_stock_prices, load_bond_prices, load_commodity_prices, align_markets,
)
from cmgm.config import STOCK_FILE, BOND_FILE, COMMODITY_FILE

OUTPUT = 'experiments/regime_analysis'
os.makedirs(OUTPUT, exist_ok=True)

N_REGIMES = 3
ROLL = 20          # rolling window for regime features
MAX_LAG = 5        # lead-lag analysis range


def load_data():
    """Load and align the three markets; return aligned close prices."""
    stocks = load_stock_prices(STOCK_FILE)
    bonds = load_bond_prices(BOND_FILE)
    futures = load_commodity_prices(COMMODITY_FILE)
    prices, indices = align_markets(stocks, bonds, futures)
    return prices, indices


def market_index_returns(prices: pd.DataFrame, indices: dict):
    """Equal-weighted daily returns of each market index."""
    rets = {}
    for name, (s, e) in indices.items():
        p = prices.iloc[:, s:e]
        r = p.pct_change().fillna(0.0).mean(axis=1)   # equal-weight index
        rets[name] = r.values
    return rets


def build_regime_features(rets: dict):
    """Rolling features that describe the market state."""
    stock = rets['stock']
    bond = rets['bond']
    fut = rets['commodity']
    T = len(fut)
    df = pd.DataFrame({'stock': stock, 'bond': bond, 'commodity': fut})

    feat = pd.DataFrame(index=df.index)
    feat['fut_ret20'] = df['commodity'].rolling(ROLL).mean()
    feat['fut_vol20'] = df['commodity'].rolling(ROLL).std()
    feat['corr_sf20'] = df['stock'].rolling(ROLL).corr(df['commodity'])
    feat['corr_bf20'] = df['bond'].rolling(ROLL).corr(df['commodity'])
    feat['fut_abs_ret'] = df['commodity'].abs().rolling(ROLL).mean()

    feat = feat.dropna()
    X = StandardScaler().fit_transform(feat.values)
    return X, feat.index


def assign_regimes(X):
    """HMM (primary) + KMeans (cross-check) regime assignment."""
    hmm = GaussianHMM(n_components=N_REGIMES, covariance_type='full',
                      n_iter=200, random_state=42, tol=1e-4)
    hmm.fit(X)
    states_hmm = hmm.predict(X)

    km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10)
    states_km = km.fit_predict(X)

    return states_hmm, states_km, hmm


def acf(x, max_lag=10):
    """Sample autocorrelation of a series at lags 1..max_lag."""
    x = x - x.mean()
    denom = (x ** 2).sum()
    if denom < 1e-12:
        return np.zeros(max_lag)
    return np.array([(x[:-l] * x[l:]).sum() / denom for l in range(1, max_lag + 1)])


def lead_lag(stock_ret, fut_ret, bond_ret=None, max_lag=MAX_LAG):
    """Cross-correlation of market returns at lags −max_lag..+max_lag.
    Positive lag = predictor leads futures: corr(x[t−lag], y[t])."""
    def xcorr(x, y, L):
        out = {}
        n = len(y)
        for lag in range(-L, L + 1):
            if lag < 0:
                # x lags y: corr(x[t+|lag|], y[t])
                a, b = x[abs(lag):], y[:n - abs(lag)]
            elif lag == 0:
                a, b = x, y
            else:
                # x leads y: corr(x[t−lag], y[t])
                a, b = x[:n - lag], y[lag:]
            out[lag] = np.corrcoef(a, b)[0, 1] if len(a) > 5 else np.nan
        return out

    res = {'stock': xcorr(stock_ret, fut_ret, max_lag)}
    if bond_ret is not None:
        res['bond'] = xcorr(bond_ret, fut_ret, max_lag)
    return res


def predictability(rets, states, horizon=5):
    """Compare in-regime OLS vs global OLS for 5-day return prediction.

    Features: past `horizon` daily commodity returns (ending at day t).
    Target:   horizon-day forward return of the commodity index.
    """
    fut = rets['commodity']
    T = len(fut)
    # forward return: (cumulative product of daily returns over horizon) − 1
    fwd = np.full(T, np.nan)
    for t in range(T - horizon):
        fwd[t] = np.prod(1 + fut[t + 1:t + 1 + horizon]) - 1.0

    # valid samples: t in [horizon−1, T−horizon)
    t0, t1 = horizon - 1, T - horizon
    idx = np.arange(t0, t1)
    X = np.stack([fut[idx - i] for i in range(horizon - 1, -1, -1)], axis=-1)  # (N, h)
    y = fwd[idx]
    st = states[idx]
    valid = ~np.isnan(y)
    Xv, yv, sv = X[valid], y[valid], st[valid]

    # global model
    global_mse = {}
    for s in range(N_REGIMES):
        mask = sv == s
        if mask.sum() < 30:
            global_mse[s] = np.nan
            continue
        m = LinearRegression().fit(Xv, yv)      # trained on ALL regimes
        pred = m.predict(Xv[mask])
        global_mse[s] = np.mean((yv[mask] - pred) ** 2)

    # in-regime model
    inreg_mse = {}
    for s in range(N_REGIMES):
        mask = sv == s
        if mask.sum() < 30:
            inreg_mse[s] = np.nan
            continue
        m = LinearRegression().fit(Xv[mask], yv[mask])
        pred = m.predict(Xv[mask])
        inreg_mse[s] = np.mean((yv[mask] - pred) ** 2)

    return global_mse, inreg_mse


def optimal_lookback(rets, states, horizons=(5, 10, 20, 40)):
    """Per-regime: AR-type prediction of 5-day forward return using the
    mean of the past `w` daily returns as feature.  Report MSE."""
    fut = rets['commodity']
    T = len(fut)
    fwd = np.full(T, np.nan)
    for t in range(T - 5):
        fwd[t] = np.prod(1 + fut[t + 1:t + 6]) - 1.0

    out = {}
    for w in horizons:
        feat = np.array([fut[t - w:t].mean() for t in range(w, T - 5)])
        y = fwd[w:T - 5]
        st = states[w:T - 5]
        valid = ~np.isnan(y)
        feat, y, st = feat[valid], y[valid], st[valid]
        per = {}
        for s in range(N_REGIMES):
            mask = st == s
            if mask.sum() < 30:
                per[s] = np.nan
                continue
            m = LinearRegression().fit(feat[mask].reshape(-1, 1), y[mask])
            pred = m.predict(feat[mask].reshape(-1, 1))
            per[s] = np.mean((y[mask] - pred) ** 2)
        out[w] = per
    return out


def main():
    print("=" * 80)
    print("REGIME ANALYSIS — cross-market structure vs market state")
    print("=" * 80)

    # ── 1. Data ──
    prices, indices = load_data()
    rets = market_index_returns(prices, indices)
    T = len(rets['commodity'])
    dates = prices.index
    print(f"\n[Data] {T} aligned days ({dates[0].date()} → {dates[-1].date()})")

    # ── 2. Regime features + assignment ──
    X, feat_idx = build_regime_features(rets)
    print(f"[Regime] features: {X.shape[1]} rolling indicators, {len(X)} valid days")
    states_hmm, states_km, hmm = assign_regimes(X)

    # Align to full-length state array (NaN before first valid feature)
    full_states = np.full(T, -1)
    full_states[feat_idx] = states_hmm

    # Map HMM state ids to vol-sorted labels (regime 0 = calmest)
    vol_by_state = {}
    for s in range(N_REGIMES):
        mask = full_states == s
        vol_by_state[s] = rets['commodity'][mask].std()
    order = sorted(range(N_REGIMES), key=lambda s: vol_by_state[s])
    label_map = {old: new for new, old in enumerate(order)}
    full_states = np.array([label_map[s] if s >= 0 else -1 for s in full_states])
    states = full_states

    print(f"\n[HMM] transition matrix:\n{np.round(hmm.transmat_, 3)}")

    # ── 3. Per-regime overview ──
    print("\n" + "=" * 80)
    print("PER-REGIME OVERVIEW (HMM, 3 regimes sorted by volatility)")
    print("=" * 80)
    names = ['Calm', 'Normal', 'Turbulent']
    for s in range(N_REGIMES):
        mask = states == s
        n = mask.sum()
        if n == 0:
            continue
        r = rets['commodity'][mask]
        print(f"\n  Regime {s} [{names[s]}] — {n} days ({n / T * 100:.1f}%), "
              f"period: {dates[mask][0].date()} → {dates[mask][-1].date()}")
        print(f"    daily ret: mean={r.mean():+.4f}  std={r.std():.4f}  "
              f"skew={pd.Series(r).skew():+.2f}")
        print(f"    momentum (5d roll mean): {pd.Series(r).rolling(5).mean().mean():+.4f}")

    # ── 4. ACF ──
    print("\n" + "=" * 80)
    print("ACF — commodity index returns (lag 1..10)")
    print("=" * 80)
    print(f"{'Regime':<10s}" + "".join([f"{'lag' + str(l):>9s}" for l in range(1, 11)]))
    for s in range(N_REGIMES):
        mask = states == s
        r = rets['commodity'][mask]
        a = acf(r, 10)
        print(f"{names[s]:<10s}" + "".join([f"{v:>9.3f}" for v in a]))

    # 5-day forward returns ACF (overlapping windows, same as prediction target)
    print("\nACF — 5-day forward returns (overlapping, lag 1..10)")
    fut = rets['commodity']
    fwd5 = np.array([np.prod(1 + fut[t + 1:t + 6]) - 1 for t in range(T - 5)])
    print(f"{'Regime':<10s}" + "".join([f"{'lag' + str(l):>9s}" for l in range(1, 11)]))
    for s in range(N_REGIMES):
        mask = states[:T - 5] == s
        a = acf(fwd5[mask], 10)
        print(f"{names[s]:<10s}" + "".join([f"{v:>9.3f}" for v in a]))

    # ── 5. Lead-lag ──
    print("\n" + "=" * 80)
    print("CROSS-MARKET LEAD-LAG (corr of market return at t−lag with futures at t)")
    print("=" * 80)
    ll = lead_lag(rets['stock'], rets['commodity'], rets['bond'], MAX_LAG)
    print(f"{'Regime':<10s} {'market':<8s}" + "".join(
        [f"{'lag' + str(l):>7s}" for l in range(-MAX_LAG, MAX_LAG + 1)]))
    for s in range(N_REGIMES):
        mask = states == s
        for mk in ['stock', 'bond']:
            x = rets[mk][mask]
            y = rets['commodity'][mask]
            vals = []
            for lag in range(-MAX_LAG, MAX_LAG + 1):
                if lag < 0:
                    a, b = x[abs(lag):], y[:len(y) - abs(lag)]
                elif lag == 0:
                    a, b = x, y
                else:
                    a, b = x[:len(x) - lag], y[lag:]
                vals.append(np.corrcoef(a, b)[0, 1] if len(a) > 5 else np.nan)
            print(f"{names[s]:<10s} {mk:<8s}" + "".join([f"{v:>7.3f}" for v in vals]))

    # ── 6. Predictability ──
    print("\n" + "=" * 80)
    print("5-DAY RETURN PREDICTABILITY — in-regime OLS vs global OLS (MSE)")
    print("=" * 80)
    global_mse, inreg_mse = predictability(rets, states, horizon=5)
    print(f"{'Regime':<10s} {'global-MSE':>12s} {'in-regime-MSE':>15s}  improvement")
    for s in range(N_REGIMES):
        g, i = global_mse[s], inreg_mse[s]
        imp = (g - i) / g * 100 if g and not np.isnan(g) else np.nan
        print(f"{names[s]:<10s} {g:>12.6f} {i:>15.6f}  {imp:+.1f}%")

    # ── 7. Optimal look-back ──
    print("\n" + "=" * 80)
    print("OPTIMAL LOOK-BACK — AR(window) → 5-day forward return (MSE)")
    print("=" * 80)
    lb = optimal_lookback(rets, states)
    print(f"{'Regime':<10s}" + "".join([f"{'w=' + str(w):>10s}" for w in lb.keys()]))
    for s in range(N_REGIMES):
        row = [f"{lb[w][s]:>10.6f}" if not np.isnan(lb[w][s]) else f"{'nan':>10s}"
               for w in lb.keys()]
        print(f"{names[s]:<10s}" + "".join(row))

    # ── 8. Save ──
    pd.DataFrame({'date': dates, 'regime': states}).to_csv(
        os.path.join(OUTPUT, 'regimes_hmm.csv'), index=False)
    np.save(os.path.join(OUTPUT, 'states_hmm.npy'), states)
    print(f"\n[Saved] regimes to {OUTPUT}/regimes_hmm.csv")

    # ── 9. KMeans agreement ──
    # HMM and KMeans labels are arbitrary — compare adjusted Rand index
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(states_hmm, states_km)
    print(f"[Cross-check] KMeans vs HMM adjusted Rand index: {ari:.3f}"
          f"  ({'consistent' if ari > 0.5 else 'inconsistent'})")


if __name__ == '__main__':
    main()
