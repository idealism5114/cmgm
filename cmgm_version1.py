
"""
CMGM: Cross-Market Graph Model
复现自: Ali et al. (2025) "CMGM: A novel cross-market assets and multi-market
       modeling graph neural networks for financial market forecasting"

架构:
  1. 市场内子图 (Intra-market Sub-graphs): 皮尔逊相关系数构建边
  2. 跨市场超图 (Cross-market Super-graph): 市场间资产相关性边
  3. GCN 层: 提取资产间空间依赖关系
  4. LSTM 层: 提取时间序列依赖关系
  5. 全连接输出: 预测下一日收盘价
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import time
import warnings
from pathlib import Path
import os

warnings.filterwarnings("ignore")

# ================================================================
# 全局配置
# ================================================================
SEQUENCE_LENGTH = 20       # 历史窗口长度
PRED_HORIZON = 1           # create_sequences 用, 保持 1 即可 (单步预测)
TARGET_HORIZON = 1         # 预测未来 N 天累计收益率
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
CORRELATION_THRESHOLD = 0.5    # 相关性阈值 (降低使跨市场边有效连接)
CORRELATION_TOP_K = 10         # 每个节点最多保留K条边（稀疏化）
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-4
MIN_LR = 1e-6                  # 学习率下限
WARMUP_EPOCHS = 10             # 预热轮数: LR 从 0 线性增加到 LEARNING_RATE
STOCK_MISSING_THRESHOLD = 0.05 # 股票缺失率阈值 (只保留缺失率低于此值的股票)
WEIGHT_DECAY = 1e-2            # L2正则化防过拟合
HIDDEN_DIM = 4        # GCN 输出特征维数
GCN_LAYERS = 3         # GCN 卷积层数
LSTM_HIDDEN = 32       # LSTM 隐藏层单元数
DROPOUT = 0.5
EARLY_STOP_PATIENCE = 25  # 早停: val_loss 连续 25 轮不下降则停止训练
MODEL_SAVE_PATH = "cmgm_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# 沪深300成分股: 动态从数据文件中按缺失率过滤 (见 STOCK_MISSING_THRESHOLD)

# ================================================================
# 1. 数据加载与预处理
# ================================================================


def load_all_markets():
    """加载三个市场数据: 收盘价 + OHLCV (期货/债券)

    Returns:
        stock_close:  DataFrame  (T, n_stock)
        fut_close:    DataFrame  (T, n_fut)
        bond_close:   DataFrame  (T, n_bond)
        fut_ohlcv:    dict of DataFrames {col: (T, n_fut)} for open/high/low/close/volume/hold
        bond_ohlcv:   dict of DataFrames {col: (T, n_bond)} for open/high/low/close/volume/hold
    """
    base = Path(__file__).resolve().parent / "Data"

    # ---- 市场1: 股票 (沪深300, 动态过滤) ----
    df_stock = pd.read_csv(base / "hs300_data/hs300_close.csv", index_col=0)
    df_stock.index = pd.to_datetime(df_stock.index)
    df_stock.index.name = "date"

    # ---- 市场2: 商品期货 ----
    df_fut = pd.read_csv(base / "futures_data/全部品种_合并.csv")
    df_fut["date"] = pd.to_datetime(df_fut["date"])
    fut_close = df_fut.pivot_table(
        index="date", columns="品种", values="close", aggfunc="mean"
    )
    fut_close.index.name = "date"

    # ---- 市场3: 债市 ----
    df_bond = pd.read_csv(base / "bond_data/全部债券_合并.csv")
    df_bond["date"] = pd.to_datetime(df_bond["date"])
    bond_close = df_bond.pivot_table(
        index="date", columns="品种", values="close", aggfunc="mean"
    )
    bond_close.index.name = "date"

    # ---- 对齐日期 (取三者交集) ----
    common_dates = (
        df_stock.index
        .intersection(fut_close.index)
        .intersection(bond_close.index)
    ).sort_values()

    stock_all = df_stock.loc[common_dates].ffill().bfill()
    fut_close = fut_close.loc[common_dates].ffill().bfill()
    bond_close = bond_close.loc[common_dates].ffill().bfill()

    # ---- 动态过滤股票: 按缺失率剔除 ----
    missing_ratio = stock_all.isnull().mean(axis=0)
    keep_cols = missing_ratio[missing_ratio < STOCK_MISSING_THRESHOLD].index.tolist()
    n_dropped = stock_all.shape[1] - len(keep_cols)
    stock_close = stock_all[keep_cols].dropna(axis=0, how="any")
    print(f"  股票: {len(keep_cols)}/{stock_all.shape[1]} 只保留 (缺失率<{STOCK_MISSING_THRESHOLD}), "
          f"剔除 {n_dropped} 只")

    # 重新对齐 (dropna 可能减少日期)
    common_dates = (
        stock_close.index
        .intersection(fut_close.index)
        .intersection(bond_close.index)
    ).sort_values()
    stock_close = stock_close.loc[common_dates]
    fut_close = fut_close.loc[common_dates]
    bond_close = bond_close.loc[common_dates]

    # ---- 提取期货 OHLCV (对齐最终日期与品种) ----
    fut_ref_cols = fut_close.columns  # 品种列名集合
    fut_pivot_raw = {
        col: df_fut.pivot_table(index="date", columns="品种", values=col, aggfunc="mean")
        for col in ["open", "high", "low", "close", "volume", "hold"] if col in df_fut.columns
    }
    fut_ohlcv = {
        col: piv.reindex(index=common_dates, columns=fut_ref_cols).ffill().bfill().values.astype(np.float64)
        for col, piv in fut_pivot_raw.items()
    }

    # ---- 提取债券 OHLCV (对齐最终日期与品种) ----
    bond_ref_cols = bond_close.columns
    bond_pivot_raw = {
        col: df_bond.pivot_table(index="date", columns="品种", values=col, aggfunc="mean")
        for col in ["open", "high", "low", "close", "volume", "hold"] if col in df_bond.columns
    }
    bond_ohlcv = {
        col: piv.reindex(index=common_dates, columns=bond_ref_cols).ffill().bfill().values.astype(np.float64)
        for col, piv in bond_pivot_raw.items()
    }

    print(f"\n数据对齐完成，日期范围: "
          f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  股票: {stock_close.shape[1]} assets × {stock_close.shape[0]} 天")
    print(f"  期货: {fut_close.shape[1]} assets × {fut_close.shape[0]} 天")
    print(f"  债券: {bond_close.shape[1]} assets × {bond_close.shape[0]} 天")

    return stock_close, fut_close, bond_close, fut_ohlcv, bond_ohlcv


def ema_span(data, span):
    """对每列计算EMA, 返回 (T, n)"""
    T, n = data.shape
    result = np.empty_like(data)
    for j in range(n):
        result[:, j] = pd.Series(data[:, j]).ewm(span=span, adjust=False).mean().values
    return result


def rolling_std(data, window):
    """对每列计算滚动标准差, min_periods=1, 返回 (T, n)"""
    T, n = data.shape
    result = np.empty_like(data)
    for j in range(n):
        result[:, j] = pd.Series(data[:, j]).rolling(window, min_periods=1).std().fillna(0).values
    return result


def rolling_mean(data, window):
    """对每列计算滚动均值, min_periods=1, 返回 (T, n)"""
    T, n = data.shape
    result = np.empty_like(data)
    for j in range(n):
        s = pd.Series(data[:, j])
        result[:, j] = s.rolling(window, min_periods=1).mean().fillna(s.iloc[0]).values
    return result


def compute_close_features(df_close):
    """从收盘价计算扩展特征集 (14维)

    特征列表:
      [0]  log_return:     对数收益率
      [1]  vol_5d:         5日滚动波动率
      [2]  vol_10d:        10日滚动波动率
      [3]  vol_21d:        21日滚动波动率 (月)
      [4]  ma_ratio_5:     (价格-MA5)/MA5
      [5]  ma_ratio_10:    (价格-MA10)/MA10
      [6]  ma_ratio_20:    (价格-MA20)/MA20
      [7]  ma_ratio_60:    (价格-MA60)/MA60
      [8]  ema_ratio_12:  (价格-EMA12)/EMA12
      [9]  ema_ratio_26:  (价格-EMA26)/EMA26
      [10] macd:           (EMA12-EMA26)/价格
      [11] rsi_14:         14日RSI
      [12] cs_rank:        市场内横截面收益率排名 (0~1)
      [13] cs_zscore:      市场内横截面收益率z-score
    """
    prices = df_close.values.astype(np.float64)
    prices = np.clip(prices, 1e-8, None)
    T, n = prices.shape

    # 1. 对数收益率
    log_ret = np.log(prices[1:] / prices[:-1])
    log_ret = np.vstack([np.zeros((1, n)), log_ret])  # (T, n)

    # 2-4. 多窗口滚动波动率
    vol_5d = rolling_std(log_ret, 5)
    vol_10d = rolling_std(log_ret, 10)
    vol_21d = rolling_std(log_ret, 21)

    # 5-8. MA ratio: (price - MA) / MA (多个窗口)
    ma5 = rolling_mean(prices, 5)
    ma10 = rolling_mean(prices, 10)
    ma20 = rolling_mean(prices, 20)
    ma60 = rolling_mean(prices, 60)
    ma_ratio_5 = (prices - ma5) / np.clip(ma5, 1e-8, None)
    ma_ratio_10 = (prices - ma10) / np.clip(ma10, 1e-8, None)
    ma_ratio_20 = (prices - ma20) / np.clip(ma20, 1e-8, None)
    ma_ratio_60 = (prices - ma60) / np.clip(ma60, 1e-8, None)

    # 9-10. EMA ratio: (price - EMA) / EMA
    ema12 = ema_span(prices, 12)
    ema26 = ema_span(prices, 26)
    ema_ratio_12 = (prices - ema12) / np.clip(ema12, 1e-8, None)
    ema_ratio_26 = (prices - ema26) / np.clip(ema26, 1e-8, None)

    # 11. MACD: (EMA12 - EMA26) / price
    macd = (ema12 - ema26) / np.clip(prices, 1e-8, None)

    # 12. RSI(14)
    rsi_14 = np.zeros_like(log_ret)
    for j in range(n):
        delta = np.diff(prices[:, j])
        delta = np.insert(delta, 0, 0)
        gain = np.clip(delta, 0, None)
        loss = np.clip(-delta, 0, None)
        avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().fillna(0).values
        avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().fillna(1e-8).values
        rsi_14[:, j] = 100 - 100 / (1 + avg_gain / avg_loss)

    # 13-14. 市场内横截面特征
    cs_rank = np.zeros_like(log_ret)
    cs_zscore = np.zeros_like(log_ret)
    for t in range(T):
        ret_t = log_ret[t, :]
        if n > 1:
            cs_rank[t, :] = np.argsort(np.argsort(ret_t)) / (n - 1)
            mu, std = np.mean(ret_t), np.std(ret_t) + 1e-8
            cs_zscore[t, :] = (ret_t - mu) / std

    feats = np.stack([
        log_ret, vol_5d, vol_10d, vol_21d,
        ma_ratio_5, ma_ratio_10, ma_ratio_20, ma_ratio_60,
        ema_ratio_12, ema_ratio_26,
        macd, rsi_14,
        cs_rank, cs_zscore,
    ], axis=-1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def compute_volume_features_from_ohlcv(ohlcv_dict, n_assets):
    """从OHLCV数据计算成交量类特征 (3维)

    ohlcv_dict 包含键 'open'/'high'/'low'/'close'/'volume'/'hold',
    每个值为 (T, n_assets) numpy 数组。

    特征:
      [0] vol_change:   对数成交量变化率
      [1] oi_change:    对数持仓量变化率
      [2] atr_ratio:    ATR(14)/close

    返回: (T, n_assets, 3)
    """
    T = len(ohlcv_dict["close"])
    eps = 1e-8

    # 成交量变化
    volume = np.clip(ohlcv_dict["volume"], eps, None)
    vol_change = np.log(volume[1:] / volume[:-1])
    vol_change = np.vstack([np.zeros((1, n_assets)), vol_change])

    # 持仓量变化
    hold = np.clip(ohlcv_dict["hold"], eps, None)
    oi_change = np.log(hold[1:] / hold[:-1])
    oi_change = np.vstack([np.zeros((1, n_assets)), oi_change])

    # ATR(14) / close
    high = ohlcv_dict["high"].astype(np.float64)
    low = ohlcv_dict["low"].astype(np.float64)
    close = np.clip(ohlcv_dict["close"].astype(np.float64), eps, None)

    prev_close = np.vstack([close[0:1], close[:-1]])
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )
    atr = np.zeros_like(tr)
    for j in range(n_assets):
        atr[:, j] = pd.Series(tr[:, j]).rolling(14, min_periods=1).mean().fillna(0).values
    atr_ratio = atr / close

    feats = np.stack([vol_change, oi_change, atr_ratio], axis=-1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def compute_macro_features(T, n_assets, dates=None):
    """宏观特征 (2维): 利率变化 + VIX 收益率

    尝试从 Data/macro_data/macro.csv 加载真实数据,
    若文件不存在则返回全零占位。

    CSV 格式要求:
        date, interest_rate, vix
    其中 interest_rate 为日频利率 (如 SHIBOR 1W),
    vix 为波动率指数 (如 iVIX / CBOE VIX)。

    Args:
        T: 时间步数
        n_assets: 资产数量 (用于 broadcast)
        dates: DatetimeIndex, 用于对齐日期索引

    Returns:
        (T, n_assets, 2) — [interest_change, vix_return]
    """
    macro_path = Path(__file__).resolve().parent / "Data" / "macro_data" / "macro.csv"
    base = Path(__file__).resolve().parent / "Data"

    if macro_path.exists():
        try:
            df_macro = pd.read_csv(macro_path, index_col=0)
            df_macro.index = pd.to_datetime(df_macro.index).sort_values()

            if dates is not None:
                df_macro = df_macro.reindex(dates).ffill().bfill().fillna(0)
            else:
                df_macro = df_macro.fillna(0)

            rate = df_macro["interest_rate"].values.astype(np.float64)
            vix = df_macro["vix"].values.astype(np.float64)

            # 转换为日变化率
            rate_change = np.diff(rate, prepend=rate[0])
            vix_return = np.diff(np.log(np.clip(vix, 1e-8, None)),
                                 prepend=0)

            feat = np.stack([rate_change, vix_return], axis=-1)  # (T, 2)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            return np.broadcast_to(feat[np.newaxis, :, :],
                                   (n_assets, T, 2)).transpose(1, 0, 2)
        except Exception as e:
            print(f"  [宏观] 数据加载失败: {e}，使用占位零")

    # ---- 尝试从 bond 数据中提取国债收益率 proxy ----
    bond_summary = base / "bond_data" / "数据汇总.csv"
    if bond_summary.exists() and dates is not None:
        try:
            df_bond = pd.read_csv(bond_summary, index_col=0)
            df_bond.index = pd.to_datetime(df_bond.index)
            # 用债券指数的滚动收益率作为利率 proxy
            if "close" in df_bond.columns:
                bond_px = df_bond["close"].reindex(dates).ffill().bfill().values.astype(np.float64)
                rate_proxy = np.diff(np.log(np.clip(bond_px, 1e-8, None)), prepend=0)
                # VIX 仍为零
                feat = np.stack([rate_proxy, np.zeros(T)], axis=-1)
                feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                return np.broadcast_to(feat[np.newaxis, :, :],
                                       (n_assets, T, 2)).transpose(1, 0, 2)
        except Exception:
            pass

    # 无数据源: 全零占位
    return np.zeros((T, n_assets, 2))


def compute_garch_volatility(logret):
    """GARCH(1,1) 条件波动率

    对每个资产独立拟合 GARCH(1,1) 模型，提取条件波动率序列。
    使用 Zero Mean 设定（金融收益率均值接近 0），加速拟合收敛。

    Args:
        logret: (T, n_assets) 对数收益率

    Returns:
        (T, n_assets) 条件波动率（年化, 已做 nan 填充）
    """
    T, n = logret.shape
    vol = np.zeros_like(logret)

    try:
        from arch import arch_model
        for j in range(n):
            am = arch_model(logret[:, j], vol="Garch", p=1, q=1,
                            dist="normal", mean="Zero")
            res = am.fit(disp="off", show_warning=False)
            vol[:, j] = np.sqrt(res.conditional_volatility)
    except Exception:
        # Fallback: EWMA 波动率 (lambda ~ 0.94 → span ≈ 60)
        for j in range(n):
            vol[:, j] = pd.Series(logret[:, j]).ewm(span=60).std().fillna(0).values

    return np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)


def compute_global_cross_sectional(all_logret):
    """全市场横截面特征: 跨所有资产的收益率排名和z-score

    Args:
        all_logret: (T, n_total) 所有资产的对数收益率
    Returns:
        (T, n_total, 2) — [global_rank, global_zscore]
    """
    T, n_total = all_logret.shape
    rank = np.zeros_like(all_logret)
    zscore = np.zeros_like(all_logret)
    for t in range(T):
        ret_t = all_logret[t, :]
        if n_total > 1:
            rank[t, :] = np.argsort(np.argsort(ret_t)) / (n_total - 1)
            mu, std = np.mean(ret_t), np.std(ret_t) + 1e-8
            zscore[t, :] = (ret_t - mu) / std
    return np.stack([rank, zscore], axis=-1)


def create_sequences(data, targets, seq_len):
    """滑窗构造 (样本数, seq_len, n_nodes, n_feat) 格式"""
    X, y = [], []
    for i in range(len(data) - seq_len - PRED_HORIZON + 1):
        X.append(data[i: i + seq_len])
        y.append(targets[i + seq_len: i + seq_len + PRED_HORIZON].flatten())
    return np.array(X), np.array(y)


# ================================================================
# 2. 图构建 
# ================================================================


def pearson_correlation(data):
    """皮尔逊相关系数矩阵 (论文 Eq. 15)"""
    return np.corrcoef(data.T)


def build_intra_market_adj(corr_matrix, threshold=CORRELATION_THRESHOLD,
                           top_k=CORRELATION_TOP_K):
    """市场内邻接矩阵 (向量化): 阈值过滤 + Top-K 稀疏化"""
    n = len(corr_matrix)
    if n <= 1:
        return np.zeros_like(corr_matrix)

    corr_no_diag = corr_matrix.copy()
    np.fill_diagonal(corr_no_diag, 0)

    # 向量化 top-k: 用 argpartition 取每行绝对值最大的 top_k 个
    actual_k = min(top_k, n - 1)
    abs_corr = np.abs(corr_no_diag)
    top_k_idx = np.argpartition(abs_corr, -actual_k, axis=1)[:, -actual_k:]  # (n, actual_k)

    # 批量取值并过滤
    rows = np.repeat(np.arange(n), actual_k)
    cols = top_k_idx.ravel()
    vals = corr_matrix[rows, cols]
    mask = vals > threshold

    adj = np.zeros_like(corr_matrix)
    adj[rows[mask], cols[mask]] = vals[mask]
    return adj


def build_cross_market_adj(market_a, market_b, threshold=CORRELATION_THRESHOLD,
                           top_k=CORRELATION_TOP_K):
    """跨市场邻接矩阵 (向量化): 快速计算全部配对相关性 + Top-K"""
    n_a, n_b = market_a.shape[1], market_b.shape[1]
    T = market_a.shape[0]

    # 向量化计算全部配对皮尔逊相关系数 (矩阵乘法)
    a_std = market_a.std(axis=0, ddof=1)
    b_std = market_b.std(axis=0, ddof=1)
    a_norm = (market_a - market_a.mean(axis=0)) / (a_std + 1e-8)
    b_norm = (market_b - market_b.mean(axis=0)) / (b_std + 1e-8)
    cross_corr = (a_norm.T @ b_norm) / (T - 1)  # (n_a, n_b)

    # 向量化 top-k (限制 k <= n_b)
    actual_k = min(top_k, n_b)
    abs_corr = np.abs(cross_corr)
    top_k_idx = np.argpartition(abs_corr, -actual_k, axis=1)[:, -actual_k:]  # (n_a, actual_k)

    rows = np.repeat(np.arange(n_a), actual_k)
    cols = top_k_idx.ravel()
    vals = cross_corr[rows, cols]
    mask = vals > threshold

    adj = np.zeros((n_a, n_b))
    adj[rows[mask], cols[mask]] = vals[mask]
    return adj


def build_super_graph(stock_data, fut_data, bond_data):
    """构建超图: 股票 + 期货 + 债券"""
    n_s = stock_data.shape[1]   # 股票数
    n_f = fut_data.shape[1]     # 期货数
    n_b = bond_data.shape[1]    # 债券数
    n_total = n_s + n_f + n_b

    # 市场内子图
    stock_corr = pearson_correlation(stock_data)
    fut_corr = pearson_correlation(fut_data)
    bond_corr = pearson_correlation(bond_data)

    adj_stock = build_intra_market_adj(stock_corr)
    adj_fut = build_intra_market_adj(fut_corr)
    adj_bond = build_intra_market_adj(bond_corr)

    # 跨市场边
    cross_sf = build_cross_market_adj(stock_data, fut_data)    # 股票↔期货
    cross_sb = build_cross_market_adj(stock_data, bond_data)   # 股票↔债券
    cross_fb = build_cross_market_adj(fut_data, bond_data)     # 期货↔债券

    # 组装超图
    super_adj = np.zeros((n_total, n_total))

    # 对角块: 市场内
    super_adj[:n_s, :n_s] = adj_stock
    super_adj[n_s: n_s + n_f, n_s: n_s + n_f] = adj_fut
    super_adj[n_s + n_f:, n_s + n_f:] = adj_bond

    # 跨市场块 (对称)
    super_adj[:n_s, n_s: n_s + n_f] = cross_sf
    super_adj[n_s: n_s + n_f, :n_s] = cross_sf.T
    super_adj[:n_s, n_s + n_f:] = cross_sb
    super_adj[n_s + n_f:, :n_s] = cross_sb.T
    super_adj[n_s: n_s + n_f, n_s + n_f:] = cross_fb
    super_adj[n_s + n_f:, n_s: n_s + n_f] = cross_fb.T

    # 加自环 + 列归一化 (mean aggregation: A D^{-1})
    super_adj_with_self = super_adj + np.eye(n_total)
    degrees = np.sum(np.abs(super_adj_with_self), axis=1)
    degrees = np.clip(degrees, 1e-8, None)
    d_inv = np.diag(1.0 / degrees)
    super_adj_norm = super_adj_with_self @ d_inv

    # 统计 (每轮只打第一次, 减少日志)
    if not hasattr(build_super_graph, "_counter"):
        build_super_graph._counter = 0
    build_super_graph._counter += 1
    if build_super_graph._counter == 1:
        print(f"\n超图构建完成 ({n_total} 个节点)")
        print(f"  股票市场内边:       {np.count_nonzero(adj_stock):4d}")
        print(f"  期货市场内边:       {np.count_nonzero(adj_fut):4d}")
        print(f"  债券市场内边:       {np.count_nonzero(adj_bond):4d}")
        print(f"  跨市场边 (股票↔期货): {np.count_nonzero(cross_sf) * 2:4d}")
        print(f"  跨市场边 (股票↔债券): {np.count_nonzero(cross_sb) * 2:4d}")
        print(f"  跨市场边 (期货↔债券): {np.count_nonzero(cross_fb) * 2:4d}")

    return torch.tensor(super_adj_norm, dtype=torch.float32)


# ================================================================
# 3. CMGM 模型 (论文 Section 3.5-3.6)
# ================================================================


class GCNLayer(nn.Module):
    """GCN 层: mean 聚合 + concat 组合

    - Mean aggregation: A D^{-1} X (邻域特征平均)
    - Concat combination: [W_self * x || W_neigh * agg_neighbors]
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.neigh_dim = out_features // 2
        self.self_dim = out_features - self.neigh_dim
        self.linear_self = nn.Linear(in_features, self.self_dim)
        self.linear_neigh = nn.Linear(in_features, self.neigh_dim)

    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x:           (N, in_features)
            edge_index:  (2, E) — [0]=src, [1]=dst
            edge_weight: (E,)   — 列归一化权重 (mean: 1/d_j)
        Returns:
            out: (N, out_features)
        """
        src, dst = edge_index[0], edge_index[1]
        weights = edge_weight.unsqueeze(-1)  # (E, 1)

        # Mean aggregation: message from src → dst weighted by 1/d_dst
        weighted_x = x[src] * weights
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, dst, weighted_x)

        # Concat combination
        self_feat = self.linear_self(x)
        neigh_feat = self.linear_neigh(aggr)
        return torch.cat([self_feat, neigh_feat], dim=-1)


class CMGM(nn.Module):
    """Cross-Market Graph Model — GCN→LSTM 串联架构

    GCN 编码每个时间步的空间结构, 输出序列送入 LSTM 捕捉时间依赖,
    最后 LSTM 末位隐状态 → 全连接输出。

    Args:
        n_nodes:     超图总节点数
        in_features: 每个节点的特征维数
        hidden_dim:  GCN 输出特征维数
        lstm_hidden: LSTM 隐藏层单元数
        pred_dim:    预测目标维数 (= 目标市场节点数)
        gcn_layers:  GCN 卷积层数
        dropout:     Dropout 比率
    """

    def __init__(
        self,
        n_nodes,
        in_features=2,
        hidden_dim=10,
        lstm_hidden=64,
        pred_dim=15,
        gcn_layers=3,
        dropout=0.3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim

        # ---- GCN 编码器: 每个时间步图卷积 ----
        self.gcn_convs = nn.ModuleList()
        self.gcn_convs.append(GCNLayer(in_features, hidden_dim))
        for _ in range(gcn_layers - 1):
            self.gcn_convs.append(GCNLayer(hidden_dim, hidden_dim))

        self.gcn_bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(gcn_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # ---- LSTM 时序编码: GCN 输出序列 → 时序 ----
        self.lstm = nn.LSTM(
            input_size=n_nodes * hidden_dim,   # GCN 每个时间步的输出
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # ---- 输出层 ----
        self.fc = nn.Linear(lstm_hidden, pred_dim)

    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x:           (batch, seq_len, n_nodes, in_features)
            edge_index:  (2, num_edges)
            edge_weight: (num_edges,)
        Returns:
            out: (batch, pred_dim)
        """
        batch_size, seq_len, n_nodes, n_feat = x.shape

        # GCN 编码每个时间步
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :].reshape(-1, n_feat)  # (batch * n_nodes, n_feat)

            for i, conv in enumerate(self.gcn_convs):
                x_t = conv(x_t, edge_index, edge_weight)
                x_t = self.gcn_bns[i](x_t)
                x_t = F.relu(x_t)
                x_t = self.dropout(x_t)

            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)
            x_t = x_t.reshape(batch_size, n_nodes * self.hidden_dim)
            gcn_outputs.append(x_t)

        # 堆叠 GCN 输出 → LSTM
        gcn_seq = torch.stack(gcn_outputs, dim=1)  # (batch, seq_len, n_nodes * hidden_dim)
        lstm_out, _ = self.lstm(gcn_seq)            # (batch, seq_len, lstm_hidden)
        lstm_last = lstm_out[:, -1, :]              # (batch, lstm_hidden)

        return self.fc(lstm_last)


class LSTMBaseline(nn.Module):
    """LSTM 基线: 无 GCN, 只用期货自身特征, 无跨市场信息"""
    def __init__(self, input_size, hidden_size=128, output_size=15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTM(nn.Module):
    """双向 LSTM 基线: 与 LSTM 基线相同结构, 但使用双向"""
    def __init__(self, input_size, hidden_size=128, output_size=15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,
                            dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ================================================================
# 4. 稠密邻接矩阵 → PyG 稀疏边格式
# ================================================================


def adj_to_edge_index(adj_matrix):
    """转为 PyG 的 (edge_index, edge_weight)"""
    n = adj_matrix.shape[0]
    edge_list = []
    weights = []
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] != 0:
                edge_list.append([i, j])
                weights.append(adj_matrix[i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def build_batch_graph_cache(logret, n_stock, n_fut, n_bond, seq_len, batch_size, n_samples=None):
    """预计算: 每个 batch 位置一个滚动窗口图

    使用每个 batch 最后一个样本的时间窗口内的对数收益率构建图,
    使图随训练推进而时变。

    Args:
        logret: (T, n_total) 未标准化的对数收益率
        seq_len: 序列长度
        batch_size: 批大小
        n_samples: 实际送入 DataLoader 的样本数 (来自 create_sequences)。
                   若为 None 则用 len(logret) - seq_len + 1,
                   否则用此值以确保与 DataLoader 的 batch 划分完全一致。

    Returns:
        list of (edge_index, edge_weight) for each batch
    """
    if n_samples is None:
        n_samples = len(logret) - seq_len + 1
    n_batches = (n_samples + batch_size - 1) // batch_size
    n_total = n_stock + n_fut + n_bond
    cache = []
    for b in range(n_batches):
        last_idx = min((b + 1) * batch_size, n_samples) - 1
        window = logret[last_idx:last_idx + seq_len]  # (seq_len, n_total)
        adj = build_super_graph(
            window[:, :n_stock],
            window[:, n_stock:n_stock + n_fut],
            window[:, n_stock + n_fut:],
        )
        ei, ew = adj_to_edge_index(adj)

        # ---- 关键: 按 batch 展开节点索引偏移 ----
        # GCN  forward 中 x 展开为 (batch * n_nodes, n_feat),
        # 所以每个样本的节点占据连续的 n_total 个索引:
        #   样本 0: [0, n_total)
        #   样本 1: [n_total, 2*n_total)
        #   ...
        # 将 edge_index 偏移后拼接, 使 GCN 正确区分不同样本的节点。
        actual_bs = min(batch_size, n_samples - b * batch_size)
        offsets = torch.arange(actual_bs, dtype=torch.long).view(-1, 1) * n_total  # (B, 1)
        offsets = offsets.view(-1, 1, 1)  # (B, 1, 1) for broadcasting with (1, 2, E)
        ei_exp = ei.unsqueeze(0) + offsets  # (B, 2, E)
        ei_exp = ei_exp.permute(1, 0, 2).contiguous().reshape(2, -1)  # (2, B*E)
        ew_exp = ew.repeat(actual_bs)
        cache.append((ei_exp, ew_exp))
    return cache


# ================================================================
# 5. 训练与评估
# ================================================================


def train_epoch(model, loader, graph_cache, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch_idx, (x_batch, y_batch) in enumerate(loader):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        edge_index, edge_weight = graph_cache[batch_idx]
        edge_index = edge_index.to(DEVICE)
        edge_weight = edge_weight.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x_batch, edge_index, edge_weight)
        loss = criterion(pred, y_batch)

        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn(f"Batch loss is NaN/Inf (loss={loss.item():.4f}), skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, graph_cache, scaler_target, criterion):
    """评估函数: 返回收益率空间指标 + 方向准确率"""
    model.eval()
    total_loss = 0.0
    all_preds, all_actual = [], []
    n_batches = 0
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            edge_index, edge_weight = graph_cache[batch_idx]
            edge_index = edge_index.to(DEVICE)
            edge_weight = edge_weight.to(DEVICE)

            pred = model(x_batch, edge_index, edge_weight)

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                continue

            loss = criterion(pred, y_batch)
            total_loss += loss.item()

            all_preds.append(pred.cpu().numpy())
            all_actual.append(y_batch.cpu().numpy())
            n_batches += 1

    if n_batches == 0:
        return (float("inf"), float("inf"), float("inf"), float("inf"),
                float("inf"), 0.0, np.zeros(1), np.zeros(1))

    preds = np.concatenate(all_preds)
    actual = np.concatenate(all_actual)

    # 反标准化为原始收益率
    preds_inv = scaler_target.inverse_transform(preds)
    actual_inv = scaler_target.inverse_transform(actual)

    # 收益率空间计算误差
    mae = mean_absolute_error(actual_inv, preds_inv)
    mse = mean_squared_error(actual_inv, preds_inv)
    rmse = np.sqrt(mse)

    # 方向准确率 (sign accuracy)
    pred_sign = np.sign(preds_inv)
    actual_sign = np.sign(actual_inv)
    nonzero_mask = actual_sign != 0
    if nonzero_mask.sum() > 0:
        dir_acc = np.mean(pred_sign[nonzero_mask] == actual_sign[nonzero_mask])
    else:
        dir_acc = 0.5
    return total_loss / n_batches, mae, mse, rmse, dir_acc, actual_inv, preds_inv


# ================================================================
# 6. 主流程
# ================================================================


def main():
    print("=" * 60)
    print("CMGM: Cross-Market Graph Model")
    print("=" * 60)

    # 设置随机种子（保证结果可复现）
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---- 数据加载 ----
    stock_close, fut_close, bond_close, fut_ohlcv, bond_ohlcv = load_all_markets()

    fut_arr = fut_close.values
    T = fut_arr.shape[0]

    n_stock = stock_close.shape[1]
    n_fut = fut_close.shape[1]
    n_bond = bond_close.shape[1]
    n_total = n_stock + n_fut + n_bond

    # ---- 特征工程 ----
    # 1) 收盘价衍生特征 (14维): 所有市场统一
    stock_feat = compute_close_features(stock_close)  # (T, n_stock, 14)
    fut_feat   = compute_close_features(fut_close)    # (T, n_fut, 14)
    bond_feat  = compute_close_features(bond_close)   # (T, n_bond, 14)

    # 2) 成交量衍生特征 (3维): 仅期货/债券有OHLCV, 股票补0
    n_vol_feat = 3
    stock_vol_feat = np.zeros((T, n_stock, n_vol_feat))
    fut_vol_feat = compute_volume_features_from_ohlcv(fut_ohlcv, n_fut)
    bond_vol_feat = compute_volume_features_from_ohlcv(bond_ohlcv, n_bond)

    # 3) 合并收盘价特征 + 成交量特征
    close_feats = np.concatenate([stock_feat, fut_feat, bond_feat], axis=1)   # (T, n_total, 14)
    vol_feats   = np.concatenate([stock_vol_feat, fut_vol_feat, bond_vol_feat], axis=1)  # (T, n_total, 3)

    # 4) 全市场横截面特征 (2维): 跨所有资产的收益率排名和z-score
    all_logret = close_feats[:, :, 0]  # (T, n_total)
    global_cs_feat = compute_global_cross_sectional(all_logret)  # (T, n_total, 2)

    # 5) GARCH(1,1) 条件波动率 (1维): 每个资产独立的波动率估计
    garch_vol = compute_garch_volatility(all_logret)  # (T, n_total)
    garch_feat = garch_vol[..., np.newaxis]  # (T, n_total, 1)

    # 6) 宏观特征 (2维): 利率变化, VIX (尝试加载真实数据, 无则全零)
    macro_feat = compute_macro_features(T, n_total, dates=stock_close.index)  # (T, n_total, 2)

    # 7) 拼接全部特征: 14 + 3 + 1 + 2 + 2 = 22 维
    all_features = np.concatenate([close_feats, vol_feats, garch_feat, global_cs_feat, macro_feat], axis=-1)
    n_feat = all_features.shape[-1]
    print(f"  特征维度: {n_feat} (收益率+波动率+MA+EMA+MACD+RSI+成交量+ATR+GARCH+横截面+宏观)")


    # ---- 数据划分 ----
    n_train = int(T * TRAIN_RATIO)
    n_val = int(T * VAL_RATIO)

    train_feat = all_features[:n_train]
    val_feat = all_features[n_train: n_train + n_val]
    test_feat = all_features[n_train + n_val:]

    # 目标: 预测 N 日累计对数收益率 (信噪比 > 1日收益率)
    # fwd_ret[t] = ln(P[t + N - 1] / P[t - 1])  (N日累计收益, 从 t-1 到 t+N-2)
    # 对齐: 特征窗口 [i, i+seq_len-1] → 预测从 seq_len-1 开始的 N 日收益
    # y[i] = fwd_ret[i + seq_len]
    n_offset = TARGET_HORIZON - 1
    fwd_ret = np.zeros_like(fut_arr)
    for t in range(1, T - n_offset):
        fwd_ret[t] = np.log(fut_arr[t + n_offset] / fut_arr[t - 1])

    # 因 fwd_ret 需要未来 n_offset 天数据, 裁剪末尾避免越界
    train_target = fwd_ret[:n_train - n_offset] if n_train > n_offset else fwd_ret[:n_train]
    val_target = fwd_ret[n_train: n_train + n_val - n_offset]
    test_target = fwd_ret[n_train + n_val: T - n_offset]

    # 同步裁剪特征 (并保留裁剪前的原始对数收益率用于滚动图)
    train_logret = train_feat[:, :, 0].copy()  # 未标准化的对数收益率
    val_logret = val_feat[:, :, 0].copy()
    test_logret = test_feat[:, :, 0].copy()
    train_feat = train_feat[:len(train_target)]
    val_feat = val_feat[:len(val_target)]
    test_feat = test_feat[:len(test_target)]
    train_logret = train_logret[:len(train_target)]
    val_logret = val_logret[:len(val_target)]
    test_logret = test_logret[:len(test_target)]

    # 归一化特征 (MinMax 0-1)
    feat_scaler = MinMaxScaler(feature_range=(0, 1))
    train_feat_flat = train_feat.reshape(-1, n_feat)
    feat_scaler.fit(train_feat_flat)
    train_feat = feat_scaler.transform(train_feat_flat).reshape(train_feat.shape)
    val_feat = feat_scaler.transform(val_feat.reshape(-1, n_feat)).reshape(val_feat.shape)
    test_feat = feat_scaler.transform(test_feat.reshape(-1, n_feat)).reshape(test_feat.shape)

    # 归一化目标 (保存原始值用于基线)
    train_target_raw = train_target.copy()
    val_target_raw = val_target.copy()
    test_target_raw = test_target.copy()
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_target = target_scaler.fit_transform(train_target)
    val_target = target_scaler.transform(val_target)
    test_target = target_scaler.transform(test_target)

    # ---- 构造滑动窗口 ----
    X_train, y_train = create_sequences(train_feat, train_target, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(val_feat, val_target, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_feat, test_target, SEQUENCE_LENGTH)

    print(f"\n序列数据: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    # ---- 预计算滚动图缓存 (每个 batch 一个图, 使用未标准化对数收益率) ----
    # 注意: n_samples 必须与 create_sequences/DaraLoader 的样本数一致,
    #       否则最后 batch 的节点偏移会越界。
    print("\n预计算滚动图缓存 (每个 batch 位置一个时变图)...")
    t_graph = time.time()
    train_graph_cache = build_batch_graph_cache(
        train_logret, n_stock, n_fut, n_bond, SEQUENCE_LENGTH, BATCH_SIZE,
        n_samples=len(X_train))
    val_graph_cache = build_batch_graph_cache(
        val_logret, n_stock, n_fut, n_bond, SEQUENCE_LENGTH, BATCH_SIZE,
        n_samples=len(X_val))
    test_graph_cache = build_batch_graph_cache(
        test_logret, n_stock, n_fut, n_bond, SEQUENCE_LENGTH, BATCH_SIZE,
        n_samples=len(X_test))
    print(f"  训练 {len(train_graph_cache)} batches, "
          f"验证 {len(val_graph_cache)} batches, "
          f"测试 {len(test_graph_cache)} batches | "
          f"耗时 {time.time() - t_graph:.1f}s")

    # ---- DataLoader (时间序列: 使用 shuffle=False 保持滚动图一致性) ----
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # ---- 模型 ----
    model = CMGM(
        n_nodes=n_total,
        in_features=n_feat,
        hidden_dim=HIDDEN_DIM,
        lstm_hidden=LSTM_HIDDEN,
        pred_dim=n_fut,
        gcn_layers=GCN_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=MIN_LR,
    )
    criterion = nn.MSELoss()  # 均方误差

    # ---- 训练 ----
    best_val_loss = float("inf")
    best_epoch = 0
    stop_rounds = 0  # 早停计数器
    t0 = time.time()

    print(f"\n[训练] epochs={EPOCHS} ...")

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, train_graph_cache,
                                 optimizer, criterion)
        val_loss, val_mae, val_mse, val_rmse, val_dir_acc, _, _ = evaluate(
            model, val_loader, val_graph_cache, target_scaler, criterion,
        )

        # 学习率预热: 前 WARMUP_EPOCHS 轮从 0 线性增加到 LEARNING_RATE
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            stop_rounds = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            stop_rounds += 1
            if stop_rounds >= EARLY_STOP_PATIENCE:
                print(f"  >>> 早停: val_loss 连续 {EARLY_STOP_PATIENCE} 轮未下降"
                      f"（当前 {val_loss:.6f} > 最佳 {best_val_loss:.6f}）")
                break

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val MAE: {val_mae:.6f} | Val RMSE: {val_rmse:.6f} | "
                  f"Val Dir Acc: {val_dir_acc:.2%} | "
                  f"LR: {current_lr:.2e}")
        elif (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val Dir Acc: {val_dir_acc:.2%}")

    train_time = time.time() - t0

    # 始终保存最终模型作为兜底
    torch.save(model.state_dict(), f"final_{MODEL_SAVE_PATH}")

    # ---- 测试 ----
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"  加载最佳模型 (epoch {best_epoch})")
    else:
        print(f"  [警告] 未找到最佳模型 '{MODEL_SAVE_PATH}'，使用最终 epoch 模型")

    test_loss, test_mae, test_mse, test_rmse, test_dir_acc, actual_ret, preds_ret = evaluate(
        model, test_loader, test_graph_cache, target_scaler, criterion,
    )

    # ---- 结果输出 (收益率空间) ----
    fut_names = fut_close.columns.tolist()
    print(f"\n{'='*60}")
    print("CMGM 模型测试结果 — 收益率空间 (Return Space)")
    print(f"{'='*60}")

    # Naive 收益率基线: 预测零收益
    naive_mae = np.mean(np.abs(actual_ret))
    naive_mse = np.mean(actual_ret**2)
    naive_rmse = np.sqrt(naive_mse)

    print(f"  {'指标':<20} {'Naive(零收益)':>14} {'CMGM':>14}")
    print(f"  {'-'*48}")
    print(f"  {'MAE':<20} {naive_mae:>14.6f} {test_mae:>14.6f}")
    print(f"  {'MSE':<20} {naive_mse:>14.6f} {test_mse:>14.6f}")
    print(f"  {'RMSE':<20} {naive_rmse:>14.6f} {test_rmse:>14.6f}")
    print(f"  {'-'*48}")
    print(f"  训练时间: {train_time:.1f}s | 最佳 epoch: {best_epoch}/{EPOCHS}")
    print(f"  方向准确率: {test_dir_acc:.4f} (随机: 0.50)")

    ratio = test_rmse / naive_rmse if naive_rmse > 0 else float('inf')
    if ratio < 0.95:
        print(f"  ✓ 模型 RMSE 低于 Naive ({ratio:.2f}x)")
    elif ratio > 1.05:
        print(f"  ⚠️ 模型 RMSE 高于 Naive ({ratio:.1f}x)")
    else:
        print(f"  ~ 模型与 Naive 基线接近 ({ratio:.2f}x)")

    # ---- 分品种详细结果 ----
    print(f"\n分品种预测误差 — 收益率空间:")
    print(f"  {'品种':<12} {'CMGM-MAE':>10} {'CMGM-RMSE':>10} {'Naive-RMSE':>10} {'RMSE比':>8}")
    print(f"  {'-'*52}")
    for i, name in enumerate(fut_names):
        ret_mae_i = mean_absolute_error(actual_ret[:, i], preds_ret[:, i])
        ret_rmse_i = np.sqrt(mean_squared_error(actual_ret[:, i], preds_ret[:, i]))
        naive_rmse_i = np.sqrt(np.mean(actual_ret[:, i]**2))
        r = ret_rmse_i / naive_rmse_i if naive_rmse_i > 0 else float('inf')
        status = "⚠️" if r > 1.5 else "✓" if r < 0.9 else "~"
        print(f"  {name:<12} {ret_mae_i:>10.6f} {ret_rmse_i:>10.6f} {naive_rmse_i:>10.6f} {r:>7.2f}x{status}")

    # ===== 基线模型对比 =====
    print(f"\n{'='*60}")
    print("基线模型对比 (Baselines)")
    print(f"{'='*60}")

    # 1. Historical Mean: 预测训练期均值
    hist_mean = np.mean(train_target_raw, axis=0)
    hist_pred = np.tile(hist_mean, (len(test_target_raw), 1))
    hist_mae = mean_absolute_error(test_target_raw, hist_pred)
    hist_mse = mean_squared_error(test_target_raw, hist_pred)
    hist_rmse = np.sqrt(hist_mse)
    print(f"\n  1. Historical Mean (训练期均值)")
    print(f"     MAE: {hist_mae:.6f} | MSE: {hist_mse:.6f} | RMSE: {hist_rmse:.6f}")

    # 2. LSTM: 仅期货特征, 无 GCN, 无跨市场信息
    print(f"\n  2. LSTM 基线")
    # 提取期货特征 (all_features 中 n_stock:n_stock+n_fut 是期货)
    def _extract_fut(X):
        N, S, _, D = X.shape
        return X[:, :, n_stock:n_stock+n_fut, :].reshape(N, S, n_fut * D)

    X_train_fut = _extract_fut(X_train)
    X_val_fut = _extract_fut(X_val)
    X_test_fut = _extract_fut(X_test)
    print(f"     输入: {n_fut}品种 × {n_feat}特征 = {X_train_fut.shape[2]} 维")

    lstm_model = LSTMBaseline(input_size=n_fut * n_feat, output_size=n_fut).to(DEVICE)
    lstm_opt = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    lstm_crit = nn.HuberLoss(delta=1.0)
    lstm_train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_fut, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True)
    lstm_val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_fut, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False)
    lstm_test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_fut, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False)

    lstm_best_loss = float("inf")
    lstm_best_state = None
    lstm_patience = 15
    lstm_wait = 0
    for ep in range(100):
        lstm_model.train()
        for xb, yb in lstm_train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            lstm_opt.zero_grad()
            pred = lstm_model(xb)
            lstm_crit(pred, yb).backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
            lstm_opt.step()
        lstm_model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in lstm_val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = lstm_model(xb)
                vl += lstm_crit(pred, yb).item()
        vl /= max(len(lstm_val_loader), 1)
        if vl < lstm_best_loss:
            lstm_best_loss = vl
            lstm_best_state = {k: v.cpu().clone() for k, v in lstm_model.state_dict().items()}
            lstm_wait = 0
        else:
            lstm_wait += 1
            if lstm_wait >= lstm_patience:
                break

    if lstm_best_state is not None:
        lstm_model.load_state_dict(lstm_best_state)
    lstm_pred_raw, lstm_actual_raw = [], []
    lstm_model.eval()
    with torch.no_grad():
        for xb, yb in lstm_test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            lstm_pred_raw.append(target_scaler.inverse_transform(
                lstm_model(xb).cpu().numpy()))
            lstm_actual_raw.append(target_scaler.inverse_transform(yb.cpu().numpy()))
    lstm_mae = mean_absolute_error(np.concatenate(lstm_actual_raw),
                                    np.concatenate(lstm_pred_raw))
    lstm_mse = mean_squared_error(np.concatenate(lstm_actual_raw),
                                   np.concatenate(lstm_pred_raw))
    lstm_rmse = np.sqrt(lstm_mse)
    print(f"     MAE: {lstm_mae:.6f} | MSE: {lstm_mse:.6f} | RMSE: {lstm_rmse:.6f}")

    # ---- 3. Ridge Regression (仅最后时间步, L2正则化) ----
    print(f"\n  3. Ridge Regression (仅最后时间步, {n_fut * n_feat}维)")
    X_train_ridge = X_train_fut[:, -1, :]  # (n, n_fut * n_feat)
    X_test_ridge = X_test_fut[:, -1, :]
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_ridge, y_train)
    ridge_pred = target_scaler.inverse_transform(ridge.predict(X_test_ridge))
    ridge_actual = target_scaler.inverse_transform(y_test)
    ridge_mae = mean_absolute_error(ridge_actual, ridge_pred)
    ridge_mse = mean_squared_error(ridge_actual, ridge_pred)
    ridge_rmse = np.sqrt(ridge_mse)
    print(f"     MAE: {ridge_mae:.6f} | MSE: {ridge_mse:.6f} | RMSE: {ridge_rmse:.6f}")

    # ---- 4. SVR (线性核, 仅最后时间步) ----
    print(f"\n  4. SVR (线性核, 仅最后时间步)")
    svr = MultiOutputRegressor(
        SVR(kernel="linear", C=1.0), n_jobs=-1
    )
    svr.fit(X_train_ridge, y_train)
    svr_pred = target_scaler.inverse_transform(svr.predict(X_test_ridge))
    svr_actual = target_scaler.inverse_transform(y_test)
    svr_mae = mean_absolute_error(svr_actual, svr_pred)
    svr_mse = mean_squared_error(svr_actual, svr_pred)
    svr_rmse = np.sqrt(svr_mse)
    print(f"     MAE: {svr_mae:.6f} | MSE: {svr_mse:.6f} | RMSE: {svr_rmse:.6f}")

    # ---- 5. BiLSTM 基线 ----
    print(f"\n  5. BiLSTM 基线")
    bilstm_model = BiLSTM(input_size=n_fut * n_feat, output_size=n_fut).to(DEVICE)
    bilstm_opt = torch.optim.Adam(bilstm_model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    bilstm_crit = nn.HuberLoss(delta=1.0)
    bilstm_train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_fut, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True)
    bilstm_val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_fut, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False)
    bilstm_test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_fut, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False)

    bilstm_best_loss = float("inf")
    bilstm_best_state = None
    bilstm_patience = 15
    bilstm_wait = 0
    for ep in range(100):
        bilstm_model.train()
        for xb, yb in bilstm_train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            bilstm_opt.zero_grad()
            pred = bilstm_model(xb)
            bilstm_crit(pred, yb).backward()
            torch.nn.utils.clip_grad_norm_(bilstm_model.parameters(), 1.0)
            bilstm_opt.step()
        bilstm_model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in bilstm_val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = bilstm_model(xb)
                vl += bilstm_crit(pred, yb).item()
        vl /= max(len(bilstm_val_loader), 1)
        if vl < bilstm_best_loss:
            bilstm_best_loss = vl
            bilstm_best_state = {k: v.cpu().clone() for k, v in bilstm_model.state_dict().items()}
            bilstm_wait = 0
        else:
            bilstm_wait += 1
            if bilstm_wait >= bilstm_patience:
                break

    if bilstm_best_state is not None:
        bilstm_model.load_state_dict(bilstm_best_state)
    bilstm_pred_raw, bilstm_actual_raw = [], []
    bilstm_model.eval()
    with torch.no_grad():
        for xb, yb in bilstm_test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            bilstm_pred_raw.append(target_scaler.inverse_transform(
                bilstm_model(xb).cpu().numpy()))
            bilstm_actual_raw.append(target_scaler.inverse_transform(yb.cpu().numpy()))
    bilstm_mae = mean_absolute_error(np.concatenate(bilstm_actual_raw),
                                     np.concatenate(bilstm_pred_raw))
    bilstm_mse = mean_squared_error(np.concatenate(bilstm_actual_raw),
                                    np.concatenate(bilstm_pred_raw))
    bilstm_rmse = np.sqrt(bilstm_mse)
    print(f"     MAE: {bilstm_mae:.6f} | MSE: {bilstm_mse:.6f} | RMSE: {bilstm_rmse:.6f}")

    # ---- 汇总对比表 ----
    print(f"\n  {'='*52}")
    print(f"  {'模型':<20} {'MAE':>10} {'MSE':>10} {'RMSE':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Naive(零收益)':<20} {naive_mae:>10.6f} {naive_mse:>10.6f} {naive_rmse:>10.6f}")
    print(f"  {'Historical Mean':<20} {hist_mae:>10.6f} {hist_mse:>10.6f} {hist_rmse:>10.6f}")
    print(f"  {'Ridge':<20} {ridge_mae:>10.6f} {ridge_mse:>10.6f} {ridge_rmse:>10.6f}")
    print(f"  {'SVR':<20} {svr_mae:>10.6f} {svr_mse:>10.6f} {svr_rmse:>10.6f}")
    print(f"  {'LSTM':<20} {lstm_mae:>10.6f} {lstm_mse:>10.6f} {lstm_rmse:>10.6f}")
    print(f"  {'BiLSTM':<20} {bilstm_mae:>10.6f} {bilstm_mse:>10.6f} {bilstm_rmse:>10.6f}")
    print(f"  {'CMGM (本文)':<20} {test_mae:>10.6f} {test_mse:>10.6f} {test_rmse:>10.6f}")
    print(f"  {'-'*52}")

if __name__ == "__main__":
    main()
