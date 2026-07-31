"""
每个市场给出整体的描述性统计：股票市场、商品市场、债券市场。
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT = Path("/home/yangxiaotong/projects/myresearch/Commedities/Data")
OUTPUT_DIR = Path("/home/yangxiaotong/projects/myresearch/Commedities/experiments/descriptive_stats")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda x: '%.6f' % x)


def panel_stats(series, name=""):
    """对面板数据（所有观测值合并）计算描述性统计"""
    s = series.dropna()
    stats = {
        "总观测数": len(s),
        "均值": s.mean(),
        "标准差": s.std(),
        "最小值": s.min(),
        "25%分位": s.quantile(0.25),
        "中位数": s.median(),
        "75%分位": s.quantile(0.75),
        "最大值": s.max(),
        "偏度": s.skew(),
        "峰度": s.kurtosis(),
    }
    return stats


# ====================================================================
# 1. 股票市场 — 所有沪深300成分股合并
# ====================================================================
print("=" * 90)
print("  股票市场：沪深300成分股描述性统计")
print("=" * 90)

stock_df = pd.read_csv(DATA_ROOT / "hs300_data" / "hs300_close.csv", index_col=0, parse_dates=True)

# 所有股票全部交易日收盘价合并为一个序列
all_prices = stock_df.values.flatten()
all_prices = all_prices[~np.isnan(all_prices)]

# 收益率：每只股票各自算对数收益率后合并
stock_returns = np.log(stock_df / stock_df.shift(1))
all_returns = stock_returns.values.flatten()
all_returns = all_returns[~np.isnan(all_returns)]

print(f"\n数据范围: {stock_df.index.min().strftime('%Y-%m-%d')} ~ {stock_df.index.max().strftime('%Y-%m-%d')}")
print(f"交易日数: {len(stock_df)}")
print(f"股票数量: {stock_df.shape[1]}")
print(f"总观测值 (股价): {len(all_prices):,}")
print(f"总观测值 (收益率): {len(all_returns):,}")

print("\n--- 收盘价描述统计 ---")
price_stats = panel_stats(pd.Series(all_prices))
for k, v in price_stats.items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n--- 日收益率描述统计 ---")
ret_stats = panel_stats(pd.Series(all_returns))
for k, v in ret_stats.items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")


# ====================================================================
# 2. 商品期货市场 — 所有品种合并
# ====================================================================
print("\n" + "=" * 90)
print("  商品期货市场描述性统计")
print("=" * 90)

futures_df = pd.read_csv(DATA_ROOT / "futures_data" / "全部品种_合并.csv", parse_dates=["date"])
futures_df.sort_values(["品种", "date"], inplace=True)

# 排除股指期货
commodity_df = futures_df[futures_df["类别"] != "股指期货"].copy()

# 收盘价
all_fut_prices = commodity_df["close"].dropna().values

# 收益率：按品种分组计算
commodity_df["return"] = np.nan
for name, group in commodity_df.groupby("品种"):
    idx = group.index
    commodity_df.loc[idx, "return"] = np.log(group["close"] / group["close"].shift(1))
all_fut_returns = commodity_df["return"].dropna().values

# 成交量
all_fut_volume = commodity_df["volume"].dropna().values

print(f"\n品种类别: {commodity_df['类别'].unique()}")
print(f"品种数量: {commodity_df['品种'].nunique()}")
print(f"总记录数: {len(commodity_df):,}")
print(f"数据范围: {commodity_df['date'].min().strftime('%Y-%m-%d')} ~ {commodity_df['date'].max().strftime('%Y-%m-%d')}")

print("\n--- 收盘价描述统计 ---")
for k, v in panel_stats(pd.Series(all_fut_prices)).items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n--- 日收益率描述统计 ---")
for k, v in panel_stats(pd.Series(all_fut_returns)).items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n--- 成交量描述统计 ---")
for k, v in panel_stats(pd.Series(all_fut_volume)).items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")


# ====================================================================
# 3. 债券市场 — 所有债券品种合并
# ====================================================================
print("\n" + "=" * 90)
print("  债券市场描述性统计")
print("=" * 90)

bond_df = pd.read_csv(DATA_ROOT / "bond_data" / "全部债券_合并.csv", parse_dates=["date"])
print(f"\n品种数量: {bond_df['品种'].nunique()}")
print(f"总记录数: {len(bond_df):,}")
print(f"数据范围: {bond_df['date'].min().strftime('%Y-%m-%d')} ~ {bond_df['date'].max().strftime('%Y-%m-%d')}")
print(f"品种: {bond_df['品种'].unique()}")

# 债券数据中有两个价格列：close（期货）和 收盘（指数）
# 统一处理：优先用 close，若全为空则用 收盘
all_bond_prices = []
all_bond_returns = []

for name, group in bond_df.groupby("品种"):
    group = group.sort_values("date")
    close_col = "close" if group["close"].notna().sum() > 0 else "收盘"
    prices = group[close_col].dropna()
    all_bond_prices.extend(prices.values)
    # 收益率
    ret = np.log(prices.astype(float) / prices.astype(float).shift(1))
    all_bond_returns.extend(ret.dropna().values)

all_bond_prices = np.array(all_bond_prices)
all_bond_returns = np.array(all_bond_returns)

print("\n--- 价格/指数描述统计 ---")
for k, v in panel_stats(pd.Series(all_bond_prices)).items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n--- 日收益率描述统计 ---")
for k, v in panel_stats(pd.Series(all_bond_returns)).items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")


# ====================================================================
# 汇总表
# ====================================================================
print("\n" + "=" * 90)
print("  三大市场描述性统计汇总")
print("=" * 90)

summary = pd.DataFrame({
    "股票市场": {k: v for k, v in panel_stats(pd.Series(all_prices)).items()},
    "商品期货": {k: v for k, v in panel_stats(pd.Series(all_fut_prices)).items()},
    "债券市场": {k: v for k, v in panel_stats(pd.Series(all_bond_prices)).items()},
})
summary.index.name = "统计量"
print("\n--- 价格/收盘价 ---")
print(summary.to_string())

summary_ret = pd.DataFrame({
    "股票市场": {k: v for k, v in panel_stats(pd.Series(all_returns)).items()},
    "商品期货": {k: v for k, v in panel_stats(pd.Series(all_fut_returns)).items()},
    "债券市场": {k: v for k, v in panel_stats(pd.Series(all_bond_returns)).items()},
})
summary_ret.index.name = "统计量"
print("\n--- 日收益率 ---")
print(summary_ret.to_string())

# 保存
summary.to_csv(OUTPUT_DIR / "三大市场_价格汇总.csv")
summary_ret.to_csv(OUTPUT_DIR / "三大市场_收益率汇总.csv")
print(f"\n结果已保存至: {OUTPUT_DIR}")
