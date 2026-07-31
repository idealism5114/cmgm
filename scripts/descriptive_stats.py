"""
对所有市场数据进行描述性统计。
计算：样本量、均值、标准差、最小值、25%/50%/75%分位数、最大值、偏度、峰度。
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT = Path("/home/yangxiaotong/projects/myresearch/Commedities/Data")
OUTPUT_DIR = Path("/home/yangxiaotong/projects/myresearch/Commedities/experiments/descriptive_stats")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def descriptive_stats(series):
    """计算一组序列的描述性统计"""
    s = series.dropna()
    if len(s) == 0:
        return {}
    return {
        "样本量": len(s),
        "均值": s.mean(),
        "标准差": s.std(),
        "最小值": s.min(),
        "25%": s.quantile(0.25),
        "50% (中位数)": s.median(),
        "75%": s.quantile(0.75),
        "最大值": s.max(),
        "偏度": s.skew(),
        "峰度": s.kurtosis(),
    }


def print_section(title):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


# ====================================================================
# 1. 股票市场 — 沪深300成分股
# ====================================================================
print_section("1. 股票市场：沪深300成分股收盘价描述统计")

stock_df = pd.read_csv(DATA_ROOT / "hs300_data" / "hs300_close.csv", index_col=0, parse_dates=True)
print(f"数据范围: {stock_df.index.min()} ~ {stock_df.index.max()}")
print(f"交易日数: {len(stock_df)}")
print(f"股票数量: {stock_df.shape[1]}")

# 计算每只股票的收益率 (对数收益率)
stock_returns = np.log(stock_df / stock_df.shift(1))

# 每只股票的统计量
stock_stats = pd.DataFrame({col: descriptive_stats(stock_df[col]) for col in stock_df.columns}).T
stock_stats.index.name = "股票代码"

# 保存完整表格
stock_stats.to_csv(OUTPUT_DIR / "股票_全样本_收盘价统计.csv")
print(f"\n--- 收盘价统计摘要 (646只股票的横截面分布) ---")
summary_fields = ["均值", "标准差", "最小值", "25%", "50% (中位数)", "75%", "最大值", "偏度", "峰度"]
stock_summary = pd.DataFrame({
    field: {
        "横截面均值": stock_stats[field].mean(),
        "横截面标准差": stock_stats[field].std(),
        "横截面最小值": stock_stats[field].min(),
        "横截面中位数": stock_stats[field].median(),
        "横截面最大值": stock_stats[field].max(),
    } for field in summary_fields
}).T
print(stock_summary.to_string())

# 收益率统计
ret_stats = pd.DataFrame({col: descriptive_stats(stock_returns[col]) for col in stock_returns.columns}).T
ret_stats.index.name = "股票代码"
ret_stats.to_csv(OUTPUT_DIR / "股票_收益率统计.csv")

print(f"\n--- 收益率统计摘要 ---")
ret_summary = pd.DataFrame({
    field: {
        "横截面均值": ret_stats[field].mean(),
        "横截面标准差": ret_stats[field].std(),
        "横截面最小值": ret_stats[field].min(),
        "横截面中位数": ret_stats[field].median(),
        "横截面最大值": ret_stats[field].max(),
    } for field in summary_fields
}).T
print(ret_summary.to_string())


# ====================================================================
# 2. 商品期货市场
# ====================================================================
print_section("2. 商品期货市场：描述性统计")

futures_df = pd.read_csv(DATA_ROOT / "futures_data" / "全部品种_合并.csv", parse_dates=["date"])
futures_df.sort_values(["品种", "date"], inplace=True)

# 按品种分组统计
print(f"总记录数: {len(futures_df)}")
print(f"品种数量: {futures_df['品种'].nunique()}")

# 收盘价统计
futures_close_stats = []
for name, group in futures_df.groupby("品种"):
    group = group.sort_values("date")
    stats = descriptive_stats(group["close"])
    stats["品种"] = name
    stats["类别"] = group["类别"].iloc[0]
    stats["合约代码"] = group["合约代码"].iloc[0]
    stats["数据起始"] = group["date"].min()
    stats["数据结束"] = group["date"].max()
    stats["交易日数"] = len(group)
    futures_close_stats.append(stats)

futures_close_df = pd.DataFrame(futures_close_stats)
cols = ["品种", "类别", "合约代码", "数据起始", "数据结束", "交易日数",
        "样本量", "均值", "标准差", "最小值", "25%", "50% (中位数)", "75%", "最大值", "偏度", "峰度"]
futures_close_df = futures_close_df[cols]
futures_close_df = futures_close_df.sort_values(["类别", "品种"])
futures_close_df.to_csv(OUTPUT_DIR / "商品期货_收盘价统计.csv", index=False)
print(futures_close_df.to_string(index=False))

# 收益率统计
print(f"\n--- 商品期货收益率统计 ---")
futures_ret_stats = []
for name, group in futures_df.groupby("品种"):
    group = group.sort_values("date")
    group["return"] = np.log(group["close"] / group["close"].shift(1))
    stats = descriptive_stats(group["return"])
    stats["品种"] = name
    stats["类别"] = group["类别"].iloc[0]
    stats["合约代码"] = group["合约代码"].iloc[0]
    futures_ret_stats.append(stats)

fut_ret_df = pd.DataFrame(futures_ret_stats)
fut_ret_df = fut_ret_df[["品种", "类别", "合约代码", "样本量", "均值", "标准差", "最小值", "25%", "50% (中位数)", "75%", "最大值", "偏度", "峰度"]]
fut_ret_df = fut_ret_df.sort_values(["类别", "品种"])
fut_ret_df.to_csv(OUTPUT_DIR / "商品期货_收益率统计.csv", index=False)
print(fut_ret_df.to_string(index=False))

# 成交量统计（对期货有意义）
print(f"\n--- 商品期货成交量统计 ---")
futures_vol_stats = []
for name, group in futures_df.groupby("品种"):
    group = group.sort_values("date")
    stats = descriptive_stats(group["volume"])
    stats["品种"] = name
    stats["类别"] = group["类别"].iloc[0]
    futures_vol_stats.append(stats)

fut_vol_df = pd.DataFrame(futures_vol_stats)
fut_vol_df = fut_vol_df[["品种", "类别", "样本量", "均值", "标准差", "最小值", "25%", "50% (中位数)", "75%", "最大值", "偏度", "峰度"]]
fut_vol_df = fut_vol_df.sort_values(["类别", "品种"])
fut_vol_df.to_csv(OUTPUT_DIR / "商品期货_成交量统计.csv", index=False)
print(fut_vol_df.to_string(index=False))


# ====================================================================
# 3. 债券市场
# ====================================================================
print_section("3. 债券市场：描述性统计")

bond_df = pd.read_csv(DATA_ROOT / "bond_data" / "全部债券_合并.csv", parse_dates=["date"])
print(f"总记录数: {len(bond_df)}")
print(f"品种数量: {bond_df['品种'].nunique()}")

bond_close_stats = []
for name, group in bond_df.groupby("品种"):
    group = group.sort_values("date")
    # 债券数据中，国债期货有 close 列，指数类在 收盘 列
    close_col = None
    if "close" in group.columns and group["close"].notna().sum() > 0:
        close_col = "close"
    elif "收盘" in group.columns and group["收盘"].notna().sum() > 0:
        close_col = "收盘"

    if close_col:
        stats = descriptive_stats(group[close_col])
    else:
        stats = {"样本量": len(group), "均值": np.nan}

    stats["品种"] = name
    stats["类别"] = group["类别"].iloc[0] if "类别" in group.columns else ""
    stats["数据起始"] = group["date"].min()
    stats["数据结束"] = group["date"].max()
    stats["交易日数"] = len(group)
    bond_close_stats.append(stats)

bond_close_df = pd.DataFrame(bond_close_stats)
bond_cols = ["品种", "类别", "数据起始", "数据结束", "交易日数",
             "样本量", "均值", "标准差", "最小值", "25%", "50% (中位数)", "75%", "最大值", "偏度", "峰度"]
bond_close_df = bond_close_df[[c for c in bond_cols if c in bond_close_df.columns]]
bond_close_df = bond_close_df.sort_values(["类别", "品种"])
bond_close_df.to_csv(OUTPUT_DIR / "债券市场_收盘价统计.csv", index=False)
print(bond_close_df.to_string(index=False))

# 债券收益率
print(f"\n--- 债券市场收益率统计 ---")
bond_ret_stats = []
for name, group in bond_df.groupby("品种"):
    group = group.sort_values("date")
    close_col = "close" if "close" in group.columns and group["close"].notna().sum() > 0 else "收盘"
    if close_col in group.columns:
        group["return"] = np.log(group[close_col].astype(float) / group[close_col].astype(float).shift(1))
        stats = descriptive_stats(group["return"])
        stats["品种"] = name
        stats["类别"] = group["类别"].iloc[0] if "类别" in group.columns else ""
        bond_ret_stats.append(stats)

bond_ret_df = pd.DataFrame(bond_ret_stats)
bond_ret_df = bond_ret_df[["品种", "类别", "样本量", "均值", "标准差", "最小值", "25%", "50% (中位数)", "75%", "最大值", "偏度", "峰度"]]
bond_ret_df = bond_ret_df.sort_values(["类别", "品种"])
bond_ret_df.to_csv(OUTPUT_DIR / "债券市场_收益率统计.csv", index=False)
print(bond_ret_df.to_string(index=False))


# ====================================================================
# 4. 宏观数据
# ====================================================================
print_section("4. 宏观数据：描述性统计")

macro_df = pd.read_csv(DATA_ROOT / "macro_data" / "macro.csv", parse_dates=["date"])
macro_df.set_index("date", inplace=True)

for col in macro_df.columns:
    stats = descriptive_stats(macro_df[col])
    print(f"\n--- {col} ---")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

macro_stats = pd.DataFrame({col: descriptive_stats(macro_df[col]) for col in macro_df.columns}).T
macro_stats.to_csv(OUTPUT_DIR / "宏观数据_统计.csv")
print(f"\n宏观数据已保存")

print(f"\n{'='*100}")
print(f"  所有统计结果已保存至: {OUTPUT_DIR}")
print(f"{'='*100}")
