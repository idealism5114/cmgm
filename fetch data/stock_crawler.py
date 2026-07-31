"""
沪深300成分股历史数据下载脚本
数据来源：BaoStock
时间范围：2015-01-01 至 2025-06-30
数据内容：日度收盘价、收益率（前复权）
作者备注：自动处理幸存者偏差，按半年节点获取历史成分股
"""

import baostock as bs
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# ─────────────────────────────────────────────
# 0. 全局配置
# ─────────────────────────────────────────────
START_DATE = "2015-01-01"
END_DATE = "2025-06-30"
ADJUST_FLAG = "2"  # 1=后复权 2=前复权 3=不复权（推荐前复权）
FREQUENCY = "d"  # d=日度  w=周度  m=月度
OUTPUT_DIR = "Data/hs300_data"  # 输出文件夹
SLEEP_BETWEEN = 0.05  # 每只股票请求间隔（秒），避免限流

# 半年节点：覆盖2015-2025，每年1月和7月各取一次成分股
# 沪深300每年1月和7月调整，此处多取几个节点确保覆盖完整
SNAPSHOT_DATES = [
    "2015-01-01", "2015-07-01",
    "2016-01-01", "2016-07-01",
    "2017-01-01", "2017-07-01",
    "2018-01-01", "2018-07-01",
    "2019-01-01", "2019-07-01",
    "2020-01-01", "2020-07-01",
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01", "2023-07-01",
    "2024-01-01", "2024-07-01",
    "2025-01-01",
]

# ─────────────────────────────────────────────
# 1. 初始化输出目录
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[初始化] 输出目录：{os.path.abspath(OUTPUT_DIR)}")


# ─────────────────────────────────────────────
# 2. 登录 BaoStock
# ─────────────────────────────────────────────
def login():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败：{lg.error_msg}")
    print(f"[登录] BaoStock 登录成功")


def logout():
    bs.logout()
    print("[退出] BaoStock 已退出")


# ─────────────────────────────────────────────
# 3. 获取各历史节点的沪深300成分股
#    → 合并去重，得到完整股票池（规避幸存者偏差）
# ─────────────────────────────────────────────
def get_full_stock_pool(snapshot_dates: list) -> pd.DataFrame:
    """
    遍历各快照日期，获取彼时的沪深300成分股列表，
    合并去重后返回完整股票池。

    返回 DataFrame，列：code, code_name, first_appear_date
    """
    all_records = []

    for date in snapshot_dates:
        rs = bs.query_hs300_stocks(date=date)
        if rs.error_code != "0":
            print(f"  [警告] {date} 查询失败：{rs.error_msg}")
            continue

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if rows:
            df_tmp = pd.DataFrame(rows, columns=rs.fields)
            df_tmp["snapshot_date"] = date
            all_records.append(df_tmp)
            print(f"  [成分股] {date}：获取 {len(rows)} 只")
        else:
            print(f"  [警告] {date}：无数据返回")

    if not all_records:
        raise RuntimeError("所有快照日期均未获取到成分股数据，请检查网络或BaoStock服务。")

    # 合并
    combined = pd.concat(all_records, ignore_index=True)

    # 每只股票保留最早出现日期
    pool = (
        combined
        .sort_values("snapshot_date")
        .groupby("code", as_index=False)
        .agg(code_name=("code_name", "first"), first_appear_date=("snapshot_date", "first"))
    )

    print(f"\n[股票池] 合并去重后共 {len(pool)} 只股票（含历史成分，规避幸存者偏差）\n")
    return pool


# ─────────────────────────────────────────────
# 4. 下载单只股票的历史行情
# ─────────────────────────────────────────────
def download_stock(code: str, start: str, end: str,
                   frequency: str, adjust_flag: str) -> pd.DataFrame:
    """
    下载单只股票的日/周/月度行情。
    返回包含 date, close, pct_chg, volume, turn 的 DataFrame。
    出错时返回空 DataFrame。
    """
    fields = "date,code,close,pctChg,volume,turn,tradestatus"
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start,
        end_date=end,
        frequency=frequency,
        adjustflag=adjust_flag,
    )

    if rs.error_code != "0":
        return pd.DataFrame()

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)

    # 类型转换
    df["date"] = pd.to_datetime(df["date"])
    for col in ["close", "pctChg", "volume", "turn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 过滤停牌日（tradestatus=0 表示停牌）
    df = df[df["tradestatus"] == "1"].drop(columns=["tradestatus"])
    df = df.reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# 5. 批量下载全部股票，拼接为宽表
# ─────────────────────────────────────────────
def batch_download(stock_pool: pd.DataFrame,
                   start: str, end: str,
                   frequency: str, adjust_flag: str) -> dict:
    """
    批量下载 stock_pool 中所有股票。
    返回 dict：{"close": wide_df, "returns": wide_df, "log_returns": wide_df}
    宽表格式：行=日期，列=股票代码
    """
    codes = stock_pool["code"].tolist()
    total = len(codes)
    print(f"[下载] 开始批量下载 {total} 只股票，时间范围 {start} ~ {end} ...\n")

    all_close = {}
    all_pctchg = {}
    failed = []

    for i, code in enumerate(codes, 1):
        df = download_stock(code, start, end, frequency, adjust_flag)

        if df.empty:
            print(f"  [{i:>3}/{total}] {code}  ✗ 无数据")
            failed.append(code)
        else:
            all_close[code] = df.set_index("date")["close"]
            all_pctchg[code] = df.set_index("date")["pctChg"]
            print(f"  [{i:>3}/{total}] {code}  ✓ {len(df)} 条")

        time.sleep(SLEEP_BETWEEN)

    print(f"\n[下载完成] 成功 {total - len(failed)} 只，失败 {len(failed)} 只")
    if failed:
        print(f"  失败列表：{failed}")

    # 拼接宽表
    close_wide = pd.DataFrame(all_close).sort_index()
    pctchg_wide = pd.DataFrame(all_pctchg).sort_index()

    # 简单收益率（BaoStock 的 pctChg 已是百分比，除以100转为小数）
    returns_wide = pctchg_wide / 100.0

    # 对数收益率（由收盘价计算）
    log_returns_wide = np.log(close_wide / close_wide.shift(1))

    return {
        "close": close_wide,
        "returns": returns_wide,
        "log_returns": log_returns_wide,
    }


# ─────────────────────────────────────────────
# 6. 数据质量检查
# ─────────────────────────────────────────────
def quality_check(data: dict, stock_pool: pd.DataFrame):
    """输出简要数据质量报告"""
    close = data["close"]
    print("\n" + "=" * 55)
    print("数据质量报告")
    print("=" * 55)
    print(f"时间范围    : {close.index.min().date()} → {close.index.max().date()}")
    print(f"交易日数量  : {len(close)} 天")
    print(f"股票数量    : {close.shape[1]} 只")

    missing_pct = close.isnull().mean() * 100
    print(f"\n缺失率统计（前复权收盘价）：")
    print(f"  平均缺失率  : {missing_pct.mean():.2f}%")
    print(f"  最大缺失率  : {missing_pct.max():.2f}%  ({missing_pct.idxmax()})")
    print(f"  缺失率>20%  : {(missing_pct > 20).sum()} 只")

    # 各年覆盖交易日数
    print(f"\n各年度交易日数：")
    for yr, grp in close.groupby(close.index.year):
        print(f"  {yr} 年：{len(grp)} 个交易日")

    print("=" * 55)


# ─────────────────────────────────────────────
# 7. 保存数据
# ─────────────────────────────────────────────
def save_data(data: dict, stock_pool: pd.DataFrame, output_dir: str):
    """保存四张表到 CSV"""
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    files = {
        f"hs300_close_{ts}.csv": data["close"],
        f"hs300_returns_{ts}.csv": data["returns"],
        f"hs300_log_returns_{ts}.csv": data["log_returns"],
        f"hs300_stock_pool_{ts}.csv": stock_pool,
    }

    for fname, df in files.items():
        path = os.path.join(output_dir, fname)
        df.to_csv(path, encoding="utf-8-sig")
        size_kb = os.path.getsize(path) / 1024
        print(f"  [保存] {fname}  ({size_kb:.0f} KB)")

    print(f"\n[完成] 全部文件已保存至：{os.path.abspath(output_dir)}")


# ─────────────────────────────────────────────
# 8. 主流程
# ─────────────────────────────────────────────
def main():
    start_time = time.time()
    print("=" * 55)
    print("  沪深300成分股历史数据下载脚本")
    print(f"  时间范围：{START_DATE} → {END_DATE}")
    print(f"  频率：{'日度' if FREQUENCY == 'd' else '周度' if FREQUENCY == 'w' else '月度'}")
    print(f"  复权方式：{'前复权' if ADJUST_FLAG == '2' else '后复权' if ADJUST_FLAG == '1' else '不复权'}")
    print("=" * 55 + "\n")

    # Step 1: 登录
    login()

    try:
        # Step 2: 获取完整股票池（含历史成分，规避幸存者偏差）
        print("[Step 1/4] 获取各历史节点沪深300成分股 ...")
        stock_pool = get_full_stock_pool(SNAPSHOT_DATES)

        # Step 3: 批量下载
        print("[Step 2/4] 批量下载行情数据 ...")
        data = batch_download(stock_pool, START_DATE, END_DATE, FREQUENCY, ADJUST_FLAG)

        # Step 4: 数据质量检查
        print("[Step 3/4] 数据质量检查 ...")
        quality_check(data, stock_pool)

        # Step 5: 保存
        print("\n[Step 4/4] 保存数据 ...")
        save_data(data, stock_pool, OUTPUT_DIR)

    finally:
        logout()

    elapsed = time.time() - start_time
    print(f"\n[耗时] 总运行时间：{elapsed / 60:.1f} 分钟")


if __name__ == "__main__":
    main()