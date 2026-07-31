"""
宏观数据爬虫: 利率(SHIBOR 1W) + 波动率指数(50ETF QVIX)

数据源 (AKShare):
  - 利率:  rate_interbank → SHIBOR 1周
  - 波动率: index_option_50etf_qvix → 上证50ETF期权隐含波动率 (中国版 VIX)

输出: Data/macro_data/macro.csv
  列: date, interest_rate, vix
  注: 两市场交易日不同, 合并后向前填充缺失值。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "Data" / "macro_data"
OUTPUT = DATA_DIR / "macro.csv"


def fetch_shibor_1w() -> pd.DataFrame:
    """获取 SHIBOR 1 周利率

    Returns:
        DataFrame 列: date (datetime), interest_rate (float, %)
    """
    import akshare as ak

    df = ak.rate_interbank(
        market="上海银行同业拆借市场",
        symbol="Shibor人民币",
        indicator="1周",
    )
    df = df.rename(columns={"报告日": "date", "利率": "interest_rate"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "interest_rate"]]


def fetch_qvix() -> pd.DataFrame:
    """获取上证 50ETF 期权隐含波动率 (QVIX / iVIX)

    QVIX 是基于 50ETF 期权价格反推出的隐含波动率指数,
    与 CBOE VIX 编制原理一致, 可以视为中国版 VIX。

    Returns:
        DataFrame 列: date (datetime), vix (float)
    """
    import akshare as ak

    df = ak.index_option_50etf_qvix()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "close"]].rename(columns={"close": "vix"})


def main():
    print("=" * 55)
    print("  宏观数据爬虫: 利率(SHIBOR 1W) + 波动率(50ETF QVIX)")
    print("=" * 55)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. 利率 ----
    print("\n[1/2] 获取 SHIBOR 1 周利率 ... ", end="", flush=True)
    t0 = time.time()
    try:
        df_rate = fetch_shibor_1w()
        print(f"完成 ({len(df_rate):,} 条, "
              f"{df_rate['date'].min().date()} ~ {df_rate['date'].max().date()}, "
              f"耗时 {time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"\n      [错误] {e}")
        return

    # ---- 2. 波动率 ----
    print("[2/2] 获取 50ETF QVIX 隐含波动率 ... ", end="", flush=True)
    t0 = time.time()
    try:
        df_vix = fetch_qvix()
        print(f"完成 ({len(df_vix):,} 条, "
              f"{df_vix['date'].min().date()} ~ {df_vix['date'].max().date()}, "
              f"耗时 {time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"\n      [错误] {e}")
        return

    # ---- 3. 合并 ----
    print("\n合并数据 ... ", end="", flush=True)
    df = pd.merge(df_rate, df_vix, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)
    # 跨市场对齐: 向前填充缺失值 (SHIBOR 交易日 ≠ 期权交易日)
    df = df.ffill()
    # 丢弃完全没有数据的起始行
    df = df.dropna(subset=["interest_rate", "vix"], how="any").reset_index(drop=True)
    print(f"{len(df):,} 条")

    # ---- 4. 保存 ----
    df.to_csv(OUTPUT, index=False)
    print(f"\n  保存到: {OUTPUT}")
    print(f"  列: {list(df.columns)}")
    print(f"  日期: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  利率: {df['interest_rate'].min():.3f}% ~ {df['interest_rate'].max():.3f}%")
    print(f"  QVIX: {df['vix'].min():.2f} ~ {df['vix'].max():.2f}")
    print(f"\n  前 5 行:")
    print(df.head(5).to_string(index=False))
    print(f"\n  后 5 行:")
    print(df.tail(5).to_string(index=False))
    print("\n完成!")


if __name__ == "__main__":
    main()
