#!/usr/bin/env python3
"""
中国大宗商品期货数据爬取工具 (2015-2025)
数据源: 新浪财经 (通过 AKShare)

安装:
    pip install akshare pandas

运行:
    python3 futures_crawler.py
"""

import akshare as ak
import pandas as pd
import time
import logging
from pathlib import Path
from datetime import datetime

# ========== 配置 ==========

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUEST_INTERVAL = 1.0  # 请求间隔(秒)
OUTPUT_DIR = Path("Data/futures_data")
DATA_START = "2015-01-01"
DATA_END = "2025-12-31"

# 品种配置: {类别: {中文名: {code: 新浪代码, exchange: 交易所}}}
PRODUCTS = {
    "能源化工": {
        "原油":           {"code": "SC", "exchange": "INE"},
        "低硫燃料油":     {"code": "LU", "exchange": "INE", "note": "2020年上市"},
        "燃料油":         {"code": "FU", "exchange": "SHFE"},
        "焦煤":           {"code": "JM", "exchange": "DCE"},
        "焦炭":           {"code": "J",  "exchange": "DCE"},
        "液化石油气":     {"code": "PG", "exchange": "DCE", "note": "2020年上市"},
        "甲醇":           {"code": "MA", "exchange": "CZCE"},
    },
    "金属类": {
        "黄金":     {"code": "AU", "exchange": "SHFE"},
        "白银":     {"code": "AG", "exchange": "SHFE"},
        "铜":       {"code": "CU", "exchange": "SHFE"},
        "锌":       {"code": "ZN", "exchange": "SHFE"},
        "铝":       {"code": "AL", "exchange": "SHFE"},
        "镍":       {"code": "NI", "exchange": "SHFE"},
        "锡":       {"code": "SN", "exchange": "SHFE"},
        "螺纹钢":   {"code": "RB", "exchange": "SHFE"},
        "热轧卷板": {"code": "HC", "exchange": "SHFE"},
    },
    "农产品": {
        "玉米":       {"code": "C",  "exchange": "DCE"},
        "黄大豆一号": {"code": "A",  "exchange": "DCE"},
        "豆粕":       {"code": "M",  "exchange": "DCE"},
        "白糖":       {"code": "SR", "exchange": "CZCE"},
        "棉花":       {"code": "CF", "exchange": "CZCE"},
        "菜籽粕":     {"code": "RM", "exchange": "CZCE"},
        "菜籽油":     {"code": "OI", "exchange": "CZCE"},
        "小麦":       {"code": "WH", "exchange": "CZCE", "note": "2023年后数据中断"},
    },
    "股指期货": {
        "沪深300股指期货": {"code": "IF", "exchange": "CFFEX"},
        "上证50股指期货":   {"code": "IH", "exchange": "CFFEX"},
        "中证500股指期货":  {"code": "IC", "exchange": "CFFEX"},
        "中证1000股指期货": {"code": "IM", "exchange": "CFFEX", "note": "2022年上市"},
    },
}


def fetch_futures(symbol: str, max_retries: int = 3) -> pd.DataFrame:
    """获取单个主力连续合约日线数据"""
    for attempt in range(max_retries):
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:
            logger.warning(f"  {symbol} 重试 {attempt+1}/{max_retries}: {e}")
            time.sleep(REQUEST_INTERVAL * (attempt + 1))
    return pd.DataFrame()


def filter_date_range(df: pd.DataFrame) -> pd.DataFrame:
    """截取 2015-2025 时间段"""
    mask = (df["date"] >= DATA_START) & (df["date"] <= DATA_END)
    return df[mask].copy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_category_data = {}

    for category, products in PRODUCTS.items():
        logger.info(f"{'='*50}")
        logger.info(f"类别: {category}")
        logger.info(f"{'='*50}")

        cat_dir = OUTPUT_DIR / category
        cat_dir.mkdir(exist_ok=True)

        cat_dfs = []

        for name, info in products.items():
            symbol = f"{info['code']}0"  # "0" = 主力连续合约
            logger.info(f"  [{symbol}] {name} ...")

            df = fetch_futures(symbol)
            if df.empty:
                logger.warning(f"  ✗ {name} 获取失败，跳过")
                continue

            # 截取时间范围
            df = filter_date_range(df)
            if df.empty:
                logger.warning(f"  ✗ {name} 在 {DATA_START}~{DATA_END} 内无数据")
                continue

            # 补充元信息
            df["品种"] = name
            df["类别"] = category
            df["交易所"] = info["exchange"]
            df["合约代码"] = symbol

            # 按日期排序
            df = df.sort_values("date").reset_index(drop=True)

            # 保存单品种 CSV
            filepath = cat_dir / f"{name}.csv"
            df.to_csv(filepath, encoding="utf-8-sig", index=False)
            logger.info(f"  ✓ {len(df)} 条 ({df['date'].min()} ~ {df['date'].max()}) -> {filepath}")

            cat_dfs.append(df)
            summary_rows.append({
                "类别": category,
                "品种": name,
                "代码": symbol,
                "交易所": info["exchange"],
                "数据量": len(df),
                "开始日期": df["date"].min().strftime("%Y-%m-%d"),
                "结束日期": df["date"].max().strftime("%Y-%m-%d"),
            })

            time.sleep(REQUEST_INTERVAL)

        # 合并该类别的所有品种
        if cat_dfs:
            combined = pd.concat(cat_dfs, ignore_index=True)
            combined = combined.sort_values(["date", "品种"]).reset_index(drop=True)
            cat_combined_path = OUTPUT_DIR / f"{category}_合并.csv"
            combined.to_csv(cat_combined_path, encoding="utf-8-sig", index=False)
            logger.info(f"  [{category}] 合并文件: {cat_combined_path} ({len(combined)} 条)")
            all_category_data[category] = combined

    # ====== 全部品种合并 ======
    if all_category_data:
        all_combined = pd.concat(all_category_data.values(), ignore_index=True)
        all_combined = all_combined.sort_values(["date", "类别", "品种"]).reset_index(drop=True)
        all_path = OUTPUT_DIR / "全部品种_合并.csv"
        all_combined.to_csv(all_path, encoding="utf-8-sig", index=False)
        logger.info(f"\n全部品种合并: {all_path} ({len(all_combined)} 条)")

    # ====== 汇总表 ======
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "数据汇总.csv"
    summary_df.to_csv(summary_path, encoding="utf-8-sig", index=False)

    # ====== 打印汇总 ======
    print(f"\n{'='*70}")
    print(f"爬取完成！数据目录: {OUTPUT_DIR.resolve()}")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))

    total = summary_df["数据量"].sum()
    print(f"\n总计: {len(summary_df)} 个品种, {total} 条日线记录 (2015~2025)")
    print(f"\n输出文件:")
    print(f"  futures_data/")
    for cat in PRODUCTS:
        print(f"    {cat}/")
        for pname in PRODUCTS[cat]:
            print(f"      {pname}.csv")
        print(f"    {cat}_合并.csv")
    print(f"  全部品种_合并.csv")
    print(f"  数据汇总.csv")


if __name__ == "__main__":
    main()
