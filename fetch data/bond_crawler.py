#!/usr/bin/env python3
"""
中国债券市场数据爬取工具 (2015-2025)
数据源: AKShare (新浪财经 / 东方财富)

安装:
    pip install akshare pandas

运行:
    python3 bond_crawler.py
"""

import akshare as ak
import pandas as pd
import time
import logging
from pathlib import Path
from typing import Optional

# ========== 配置 ==========

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUEST_INTERVAL = 1.0
OUTPUT_DIR = Path("Data/bond_data")
DATA_START = "2015-01-01"
DATA_END = "2025-12-31"


# ================================================================
# 第一部分: 国债期货
# ================================================================

TREASURY_FUTURES = {
    "10年期国债期货": {"code": "T",  "exchange": "CFFEX"},
    "5年期国债期货":  {"code": "TF", "exchange": "CFFEX"},
    "2年期国债期货":  {"code": "TS", "exchange": "CFFEX"},
    "30年期国债期货": {"code": "TL", "exchange": "CFFEX"},
}


def fetch_treasury_futures() -> list:
    """爬取国债期货日线数据"""
    logger.info(f"{'='*50}")
    logger.info("国债期货")
    logger.info(f"{'='*50}")

    out_dir = OUTPUT_DIR / "国债期货"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    dfs = []

    for name, info in TREASURY_FUTURES.items():
        symbol = f"{info['code']}0"
        logger.info(f"  [{symbol}] {name} ...")

        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
        except Exception as e:
            logger.warning(f"  ✗ {name} 获取失败: {e}")
            continue

        if df is None or df.empty:
            logger.warning(f"  ✗ {name} 无数据")
            continue

        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= DATA_START) & (df["date"] <= DATA_END)
        df = df[mask].copy()

        if df.empty:
            logger.warning(f"  ✗ {name} 时间范围内无数据")
            continue

        df = df.sort_values("date").reset_index(drop=True)
        df["品种"] = name
        df["交易所"] = info["exchange"]
        df["合约代码"] = symbol

        fp = out_dir / f"{name}.csv"
        df.to_csv(fp, encoding="utf-8-sig", index=False)
        logger.info(f"  ✓ {len(df)} 条 ({df['date'].min()} ~ {df['date'].max()})")

        dfs.append(df)
        rows.append({
            "类别": "国债期货",
            "品种": name,
            "代码": symbol,
            "交易所": info["exchange"],
            "数据量": len(df),
            "开始日期": df["date"].min().strftime("%Y-%m-%d"),
            "结束日期": df["date"].max().strftime("%Y-%m-%d"),
        })
        time.sleep(REQUEST_INTERVAL)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(["date", "品种"]).reset_index(drop=True)
        combined.to_csv(OUTPUT_DIR / "国债期货_合并.csv", encoding="utf-8-sig", index=False)
        logger.info(f"  [国债期货] 合并: {len(combined)} 条")

    return rows


# ================================================================
# 第二部分: 债券 ETF
# ================================================================

BOND_ETFS = {
    "1-3年国债ETF":          {"code": "511020", "exchange": "SH"},
    "7-10年国债ETF":         {"code": "511260", "exchange": "SH"},
    "国开债ETF":             {"code": "511010", "exchange": "SH"},
    "信用债ETF":             {"code": "511360", "exchange": "SH"},
    "高等级同业存单ETF":     {"code": "159649", "exchange": "SZ"},
    "综合债指数ETF":         {"code": "511280", "exchange": "SH"},
    "短融ETF":               {"code": "159976", "exchange": "SZ"},
    "绿色债券ETF(南方)":     {"code": "518890", "exchange": "SH"},
    "绿色电力ETF":           {"code": "159869", "exchange": "SZ"},
}


def fetch_bond_etfs() -> list:
    """爬取债券 ETF 日线数据"""
    logger.info(f"\n{'='*50}")
    logger.info("债券 ETF")
    logger.info(f"{'='*50}")

    out_dir = OUTPUT_DIR / "债券ETF"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    dfs = []

    for name, info in BOND_ETFS.items():
        symbol = info["code"]
        logger.info(f"  [{symbol}] {name} ...")

        # 需加交易所前缀: sh=上海, sz=深圳
        exchange_prefix = info["exchange"].lower()
        try:
            df = ak.fund_etf_hist_sina(symbol=f"{exchange_prefix}{symbol}")
        except Exception as e:
            logger.warning(f"  ✗ {name} 获取失败: {e}")
            continue

        if df is None or df.empty:
            logger.warning(f"  ✗ {name} 无数据")
            continue

        # 统一日期列
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            logger.warning(f"  ✗ {name} 无日期列")
            continue

        mask = (df["date"] >= DATA_START) & (df["date"] <= DATA_END)
        df = df[mask].copy()

        if df.empty:
            logger.warning(f"  ✗ {name} 时间范围内无数据")
            continue

        df = df.sort_values("date").reset_index(drop=True)
        df["品种"] = name
        df["交易所"] = info["exchange"]
        df["基金代码"] = symbol

        fp = out_dir / f"{name}.csv"
        df.to_csv(fp, encoding="utf-8-sig", index=False)
        logger.info(f"  ✓ {len(df)} 条 ({df['date'].min()} ~ {df['date'].max()})")

        dfs.append(df)
        rows.append({
            "类别": "债券ETF",
            "品种": name,
            "代码": symbol,
            "交易所": info["exchange"],
            "数据量": len(df),
            "开始日期": df["date"].min().strftime("%Y-%m-%d"),
            "结束日期": df["date"].max().strftime("%Y-%m-%d"),
        })
        time.sleep(REQUEST_INTERVAL)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(["date", "品种"]).reset_index(drop=True)
        combined.to_csv(OUTPUT_DIR / "债券ETF_合并.csv", encoding="utf-8-sig", index=False)
        logger.info(f"  [债券ETF] 合并: {len(combined)} 条")

    return rows


# ================================================================
# 第三部分: 国债现券
# ================================================================

BOND_SPOTS = {
    "10年期国债现券":  {"code": "110007"},
    "5年期国债现券":   {"code": "110004"},
}


def _fetch_bond_spot(name: str, symbol: str, out_dir: Path) -> Optional[dict]:
    """获取单只债券现券日线数据"""
    logger.info(f"  [{symbol}] {name} ...")

    try:
        df = ak.bond_zh_hs_daily(symbol=f"sh{symbol}")
    except Exception as e:
        logger.warning(f"  ✗ {name} 获取失败: {e}")
        return None

    if df is None or df.empty:
        logger.warning(f"  ✗ {name} 无数据")
        return None

    df["date"] = pd.to_datetime(df["date"])

    mask = (df["date"] >= DATA_START) & (df["date"] <= DATA_END)
    df = df[mask].copy()

    if df.empty:
        logger.warning(f"  ✗ {name} 时间范围内无数据")
        return None

    df = df.sort_values("date").reset_index(drop=True)
    df["品种"] = name
    df["债券代码"] = symbol

    fp = out_dir / f"{name}.csv"
    df.to_csv(fp, encoding="utf-8-sig", index=False)
    logger.info(f"  ✓ {len(df)} 条 ({df['date'].min()} ~ {df['date'].max()})")

    return {
        "类别": "现券",
        "品种": name,
        "代码": symbol,
        "数据量": len(df),
        "开始日期": df["date"].min().strftime("%Y-%m-%d"),
        "结束日期": df["date"].max().strftime("%Y-%m-%d"),
    }


def fetch_bond_spots() -> list:
    """爬取国债现券"""
    logger.info(f"\n{'='*50}")
    logger.info("国债现券")
    logger.info(f"{'='*50}")

    out_dir = OUTPUT_DIR / "现券"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for name, info in BOND_SPOTS.items():
        r = _fetch_bond_spot(name, info["code"], out_dir)
        if r:
            rows.append(r)
        time.sleep(REQUEST_INTERVAL)

    if rows:
        csv_files = sorted(out_dir.glob("*.csv"))
        if len(csv_files) > 1:
            all_df = []
            for f in csv_files:
                d = pd.read_csv(f)
                all_df.append(d)
            combined = pd.concat(all_df, ignore_index=True)
            combined.to_csv(OUTPUT_DIR / "现券_合并.csv", encoding="utf-8-sig", index=False)
            logger.info(f"  [现券] 合并: {len(combined)} 条")

    return rows


# ================================================================
# 第四部分: 绿色债券指数
# ================================================================

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


def fetch_green_bond_indices() -> list:
    """爬取中证绿色债券指数"""
    logger.info(f"\n{'='*50}")
    logger.info("绿色债券指数")
    logger.info(f"{'='*50}")

    out_dir = OUTPUT_DIR / "绿色债券指数"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    dfs = []

    for code, info in GREEN_BOND_INDICES.items():
        name = info["name"]
        logger.info(f"  [{code}] {name} ...")

        try:
            df = ak.stock_zh_index_hist_csindex(
                symbol=code,
                start_date=DATA_START.replace("-", ""),
                end_date=DATA_END.replace("-", ""),
            )
        except Exception as e:
            logger.warning(f"  ✗ {name} 获取失败: {e}")
            continue

        if df is None or df.empty:
            logger.warning(f"  ✗ {name} 无数据")
            continue

        # 统一日期列
        if "日期" in df.columns:
            df["date"] = pd.to_datetime(df["日期"])
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            logger.warning(f"  ✗ {name} 无日期列")
            continue

        df = df.sort_values("date").reset_index(drop=True)
        df["品种"] = name
        df["指数代码"] = code

        fp = out_dir / f"{name}.csv"
        df.to_csv(fp, encoding="utf-8-sig", index=False)
        logger.info(f"  ✓ {len(df)} 条 ({df['date'].min()} ~ {df['date'].max()})")

        dfs.append(df)
        rows.append({
            "类别": "绿色债券指数",
            "品种": name,
            "代码": code,
            "数据量": len(df),
            "开始日期": df["date"].min().strftime("%Y-%m-%d"),
            "结束日期": df["date"].max().strftime("%Y-%m-%d"),
        })
        time.sleep(REQUEST_INTERVAL)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(["date", "品种"]).reset_index(drop=True)
        combined.to_csv(OUTPUT_DIR / "绿色债券指数_合并.csv", encoding="utf-8-sig", index=False)
        logger.info(f"  [绿色债券指数] 合并: {len(combined)} 条")

    return rows


# ================================================================
# 第五部分: 合并所有债券数据
# ================================================================

def merge_all_bond_data():
    """将所有债券数据合并为一个总文件"""
    logger.info(f"\n{'='*50}")
    logger.info("合并所有债券数据")
    logger.info(f"{'='*50}")

    all_csv = sorted(OUTPUT_DIR.rglob("*.csv"))
    # 排除汇总表和总合并表自身
    all_csv = [f for f in all_csv if f.name not in ("数据汇总.csv", "全部债券_合并.csv")]

    if not all_csv:
        logger.warning("  无数据可合并")
        return

    logger.info(f"  发现 {len(all_csv)} 个子文件")

    master_dfs = []
    for f in all_csv:
        try:
            d = pd.read_csv(f)
            rel = f.relative_to(OUTPUT_DIR)
            logger.info(f"    {rel} ({len(d)} 条)")
            master_dfs.append(d)
        except Exception as e:
            logger.warning(f"    跳过 {f.name}: {e}")

    if master_dfs:
        master = pd.concat(master_dfs, ignore_index=True)
        if "date" in master.columns:
            master = master.sort_values(["date", "品种"]).reset_index(drop=True)
        master_path = OUTPUT_DIR / "全部债券_合并.csv"
        master.to_csv(master_path, encoding="utf-8-sig", index=False)
        logger.info(f"  ✓ 总合并: {len(master)} 条 -> {master_path}")


# ================================================================
# 主函数
# ================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # 1. 国债期货
    all_rows.extend(fetch_treasury_futures())

    # 2. 债券 ETF
    all_rows.extend(fetch_bond_etfs())

    # 3. 国债现券
    all_rows.extend(fetch_bond_spots())

    # 4. 绿色债券指数
    all_rows.extend(fetch_green_bond_indices())

    # ====== 汇总表 ======
    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUTPUT_DIR / "数据汇总.csv", encoding="utf-8-sig", index=False)

    # ====== 合并所有数据 ======
    merge_all_bond_data()

    print(f"\n{'='*70}")
    print(f"债券数据爬取完成！目录: {OUTPUT_DIR.resolve()}")
    print(f"{'='*70}")
    print(summary.to_string(index=False))

    total = summary["数据量"].sum() if "数据量" in summary.columns else 0
    print(f"\n总计: {len(all_rows)} 个品种, {total} 条记录")

    print(f"\n输出文件:")
    for p in sorted(OUTPUT_DIR.rglob("*.csv")):
        rel = p.relative_to(OUTPUT_DIR)
        print(f"  bond_data/{rel}")


if __name__ == "__main__":
    main()
