#!/usr/bin/env python3
"""
中证绿色债券指数爬取工具 (2015-2025)
数据源: 中证指数公司 (通过 AKShare)

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

OUTPUT_DIR = Path("Data/green_bond_data")
DATA_START = "2015-01-01"
DATA_END = "2025-12-31"
REQUEST_INTERVAL = 1.0

# 核心绿色债券指数 (第一档)
GREEN_BOND_INDICES = {
    "931145": {
        "name": "中证绿色债券指数",
        "short_name": "中证绿色债",
        "base_date": "2016-12-30",
        "note": "全市场绿色债券基准指数",
    },
    "930951": {
        "name": "中证交易所绿色债券指数",
        "short_name": "交易所绿色债",
        "base_date": "2016-12-30",
        "note": "交易所上市绿色债券",
    },
    "931147": {
        "name": "中证交易所高等级绿色债券指数",
        "short_name": "交易所高等级绿色债",
        "base_date": "2016-12-30",
        "note": "交易所高等级绿色债",
    },
}


def fetch_green_bond_index(
    code: str, start: str = DATA_START, end: str = DATA_END, max_retries: int = 3
) -> pd.DataFrame:
    """从中证指数官网获取单个绿色债券指数历史行情"""
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_index_hist_csindex(
                symbol=code,
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
            )
            if df is not None and not df.empty:
                # 将日期列排序
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.sort_values("日期").reset_index(drop=True)
                return df
        except Exception as e:
            logger.warning(
                f"  {code} 重试 {attempt+1}/{max_retries}: {e}"
            )
            time.sleep(REQUEST_INTERVAL * (attempt + 1))
    return pd.DataFrame()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    all_dfs = []

    logger.info(f"{'='*50}")
    logger.info("中证绿色债券指数")
    logger.info(f"{'='*50}")

    for code, info in GREEN_BOND_INDICES.items():
        logger.info(f"  [{code}] {info['name']} ...")

        df = fetch_green_bond_index(code)

        if df.empty:
            logger.warning(f"  ✗ {info['name']} 获取失败")
            continue

        # 补充元信息
        df["指数代码"] = code
        df["基日"] = info["base_date"]

        # 保存单文件
        filename = f"{info['short_name']}.csv"
        filepath = OUTPUT_DIR / filename
        df.to_csv(filepath, encoding="utf-8-sig", index=False)

        date_min = df["日期"].min().strftime("%Y-%m-%d")
        date_max = df["日期"].max().strftime("%Y-%m-%d")
        logger.info(f"  ✓ {len(df)} 条 ({date_min} ~ {date_max}) -> {filepath}")

        all_dfs.append(df)
        summary.append({
            "指数代码": code,
            "指数名称": info["name"],
            "简称": info["short_name"],
            "基日": info["base_date"],
            "数据量": len(df),
            "开始日期": date_min,
            "结束日期": date_max,
        })

        time.sleep(REQUEST_INTERVAL)

    # 合并
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(["日期", "指数代码"]).reset_index(drop=True)

        # 只保留关键列，避免合并文件列太多
        key_cols = [c for c in ["日期", "指数代码", "指数中文全称", "收盘", "开盘", "最高", "最低", "涨跌幅", "样本数量", "滚动市盈率"] if c in combined.columns]
        combined_out = combined[key_cols] if len(key_cols) > 1 else combined

        merged_path = OUTPUT_DIR / "绿色债券指数_合并.csv"
        combined_out.to_csv(merged_path, encoding="utf-8-sig", index=False)
        logger.info(f"\n  合并文件: {merged_path} ({len(combined_out)} 条)")

    # 汇总表
    summary_df = pd.DataFrame(summary)
    summary_path = OUTPUT_DIR / "数据汇总.csv"
    summary_df.to_csv(summary_path, encoding="utf-8-sig", index=False)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"绿色债券指数数据已保存到: {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    total = summary_df["数据量"].sum()
    print(f"\n总计: {len(summary_df)} 个指数, {total} 条日线记录")

    print(f"\n文件列表:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        if p.suffix == ".csv":
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
