# -*- coding: utf-8 -*-
"""
超跌反弹模型 2026-07-13 回溯测试
对 9 只股票用昨日数据跑超跌反弹模型评分，验证模型能否识别今天大涨的股票。
数据截止 2026-07-13，不使用 2026-07-14 任何数据。

模型函数从 stock_screener.py 导入，脚本无需修改即可复用最新评分/过滤逻辑。
"""

import sys
import os
import traceback
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ─── 路径设置 ────────────────────────────────────
# Project root = parent of scripts/ directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ─── 超跌反弹模型函数直接导入 ─────────────────────
from data.market import read_tdx_day_file
from core.filters import hard_filter_oversold_rebound
from core.scoring.oversold import calculate_oversold_rebound_score
from core.config import TDX_AVAILABLE, TDX_VIPDOC_DIR

# ─── 目标股票 ────────────────────────────────────
STOCKS = {
    "601208": "东材科技",
    "002428": "云南锗业",
    "301200": "大族数控",
    "600010": "包钢股份",
    "300835": "龙磁科技",
    "600259": "中稀有色",
    "002460": "赣锋锂业",
    "300939": "秋田微",
    "002273": "水晶光电",
}

CUTOFF_DATE = "2026-07-13"
OUTPUT_FILE = os.path.join(BASE_DIR, "orb_backtest_result.txt")

print(f"{'='*80}")
print(f"超跌反弹模型回溯测试 — 数据截止: {CUTOFF_DATE}")
print(f"通达信本地数据: {'可用 (' + TDX_VIPDOC_DIR + ')' if TDX_AVAILABLE else '不可用'}")
print(f"{'='*80}\n")


def get_kline_truncated(code, cutoff_date, min_days=120):
    """
    获取股票K线数据并截断到截止日期
    code: 6位股票代码
    cutoff_date: 截止日期字符串 'YYYY-MM-DD'
    min_days: 最少需要的天数
    返回: (DataFrame, error_msg) — error_msg 为 None 表示成功
    """
    df = read_tdx_day_file(code)
    if df is None or len(df) == 0:
        return None, "无本地K线数据"

    # 截断到截止日期
    cutoff = pd.Timestamp(cutoff_date)
    # 保留日期 <= 截止日期的行
    if "日期" in df.columns:
        date_col = "日期"
    elif "date" in df.columns:
        date_col = "date"
    else:
        # 尝试找日期列
        for c in df.columns:
            if "日期" in str(c) or "date" in str(c).lower():
                date_col = c
                break
        else:
            return None, "无法识别日期列"

    df = df[df[date_col] <= cutoff].copy()

    if len(df) < min_days:
        return None, f"K线数据不足{min_days}日（实际{len(df)}日，截止{cutoff_date}）"

    # 取最近 min_days 日，同时保留足够历史用于120日最高点等计算
    # 取全部以让评分函数内部自行截取
    return df.reset_index(drop=True), None


def compute_intermediate_indicators(kline_df):
    """计算关键中间指标"""
    result = {}

    close_col = "收盘" if "收盘" in kline_df.columns else "close"
    high_col = "最高" if "最高" in kline_df.columns else "high"
    low_col = "最低" if "最低" in kline_df.columns else "low"
    volume_col = "成交量" if "成交量" in kline_df.columns else "volume"
    open_col = "开盘" if "开盘" in kline_df.columns else "open"

    closes = kline_df[close_col].values.astype(float)
    highs = kline_df[high_col].values.astype(float)
    lows = kline_df[low_col].values.astype(float)
    volumes = kline_df[volume_col].values.astype(float)
    opens = kline_df[open_col].values.astype(float)
    n = len(closes)

    # 1. 近120日最高点
    n_lookback = min(120, n)
    high_120 = float(np.max(highs[-n_lookback:]))
    result["近120日最高点"] = f"{high_120:.2f}"

    # 2. 当前回调幅度 (从120日最高点)
    if high_120 > 0:
        drawdown = (closes[-1] / high_120 - 1.0) * 100
        result["回调幅度"] = f"{drawdown:.2f}%"
    else:
        result["回调幅度"] = "N/A"

    # 3. 当前收盘价
    result["当前收盘"] = f"{closes[-1]:.2f}"

    # 4. BIAS(MA20) 负乖离
    if n >= 20:
        ma20 = float(np.mean(closes[-20:]))
        if ma20 > 0:
            bias = (closes[-1] / ma20 - 1.0) * 100
            result["BIAS(MA20)"] = f"{bias:.2f}%"
        else:
            result["BIAS(MA20)"] = "N/A"
    else:
        result["BIAS(MA20)"] = "N/A"

    # 5. 量能萎缩比例 (最近5日均量 / 60日最大量)
    if n >= 60:
        vol_ma5 = float(np.mean(volumes[-5:]))
        vol_max_60 = float(np.max(volumes[-60:]))
        if vol_max_60 > 0:
            vol_ratio = vol_ma5 / vol_max_60
            result["量能比(5MA/60max)"] = f"{vol_ratio:.4f} ({vol_ratio*100:.1f}%)"
        else:
            result["量能比(5MA/60max)"] = "N/A"
    else:
        result["量能比(5MA/60max)"] = "N/A"

    # 6. 近3日涨跌幅
    if n >= 4:
        chg_3d = (closes[-1] / closes[-4] - 1.0) * 100
        result["近3日涨跌"] = f"{chg_3d:.2f}%"
    else:
        result["近3日涨跌"] = "N/A"

    # 7. 近5日涨跌
    if n >= 6:
        chg_5d = (closes[-1] / closes[-6] - 1.0) * 100
        result["近5日涨跌"] = f"{chg_5d:.2f}%"
    else:
        result["近5日涨跌"] = "N/A"

    # 8. 近20日涨跌
    if n >= 21:
        chg_20d = (closes[-1] / closes[-21] - 1.0) * 100
        result["近20日涨跌"] = f"{chg_20d:.2f}%"
    else:
        result["近20日涨跌"] = "N/A"

    # 9. 近5日最低点 vs 近20日最低点（判断是否企稳）
    if n >= 5:
        low_5d = float(np.min(lows[-5:]))
        if n >= 20:
            low_20d = float(np.min(lows[-20:]))
            if low_5d > low_20d:
                result["低位企稳"] = "是(5日低>20日低)"
            else:
                result["低位企稳"] = "否(5日低≤20日低)"
        result["近5日最低"] = f"{low_5d:.2f}"

    # 10. MA5 位置
    if n >= 6:
        ma5 = float(np.mean(closes[-5:]))
        ma5_prev = float(np.mean(closes[-6:-1]))
        ma5_status = "站上(MA5向上)" if closes[-1] > ma5 and ma5 >= ma5_prev else (
            "站上(MA5向下)" if closes[-1] > ma5 else "MA5下方")
        result["MA5"] = f"{ma5:.2f} ({ma5_status})"

    # 11. KDJ J值
    if n >= 13:
        try:
            k_vals = [50.0, 50.0]
            d_vals = [50.0, 50.0]
            for idx in range(8, n):
                low_9 = float(np.min(lows[idx - 8:idx + 1]))
                high_9 = float(np.max(highs[idx - 8:idx + 1]))
                rsv = ((closes[idx] - low_9) / (high_9 - low_9) * 100) if high_9 > low_9 else 50.0
                k = 2.0 / 3.0 * k_vals[-1] + 1.0 / 3.0 * rsv
                d = 2.0 / 3.0 * d_vals[-1] + 1.0 / 3.0 * k
                k_vals.append(k)
                d_vals.append(d)
            j_now = 3.0 * k_vals[-1] - 2.0 * d_vals[-1]
            j_prev = 3.0 * k_vals[-2] - 2.0 * d_vals[-2]
            j_trend = "↑" if j_now > j_prev else "↓"
            result["KDJ_J"] = f"{j_now:.2f}({j_trend})"
        except Exception:
            result["KDJ_J"] = "N/A"
    else:
        result["KDJ_J"] = "N/A"

    # 12. MACD 状态
    if n >= 27:
        try:
            ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
            ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
            dif = ema12 - ema26
            dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
            macd_bar = 2.0 * (dif - dea)
            bar_now = macd_bar[-1]
            bar_prev = macd_bar[-2]
            bar_prev2 = macd_bar[-3] if len(macd_bar) >= 3 else bar_prev
            if bar_now < 0 and abs(bar_now) < abs(bar_prev):
                macd_status = "绿柱缩短"
            elif bar_now > bar_prev and bar_prev < 0:
                macd_status = "即将金叉"
            elif bar_now > 0:
                macd_status = "红柱"
            else:
                macd_status = "绿柱放大" if abs(bar_now) > abs(bar_prev) else "绿柱持平"
            result["MACD"] = f"BAR={bar_now:.4f} ({macd_status})"
        except Exception:
            result["MACD"] = "N/A"
    else:
        result["MACD"] = "N/A"

    return result


# ─── 主流程 ──────────────────────────────────────
results = []
all_output_lines = []

for code, name in STOCKS.items():
    print(f"处理 {code} {name}...", end=" ")

    # 获取K线数据（截断到 2026-07-13）
    kline_df, err = get_kline_truncated(code, CUTOFF_DATE, min_days=20)

    if err:
        print(f"✗ {err}")
        all_output_lines.append(f"{code} {name}: ✗ {err}\n")
        continue

    print(f"OK ({len(kline_df)}条K线)", end=" ")

    # 计算中间指标
    indicators = compute_intermediate_indicators(kline_df)

    # 硬过滤
    ok, filter_msg = hard_filter_oversold_rebound(kline_df, None)
    print(f"→ 硬过滤: {'通过' if ok else '淘汰(' + filter_msg + ')'}", end=" ")

    # 评分（无论是否被淘汰都计算，方便分析）
    score = calculate_oversold_rebound_score(kline_df, None)

    total = score.get("综合评分", 0)
    space = score.get("空间维度", 0)
    sentiment = score.get("情绪量能", 0)
    timing = score.get("择时确认", 0)
    sector_bonus = score.get("板块共振", 0)

    print(f"→ 得分: {total} (空间{space} + 情绪{sentiment} + 择时{timing} + 板块{sector_bonus})")

    results.append({
        "代码": code,
        "名称": name,
        "综合评分": total,
        "空间维度": space,
        "情绪量能": sentiment,
        "择时确认": timing,
        "板块共振": sector_bonus,
        "硬过滤": "通过" if ok else f"淘汰",
        "过滤原因": filter_msg if not ok else "",
        "中间指标": indicators,
    })

# ─── 结果排序输出 ────────────────────────────────
results.sort(key=lambda x: x["综合评分"], reverse=True)

# 构建输出文本
out = []
sep = "=" * 90
out.append(sep)
out.append(f"  超跌反弹模型回溯测试报告")
out.append(f"  数据截止: {CUTOFF_DATE} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
out.append(sep)
out.append("")

# ── 总览表格 ──
out.append("【一、评分总览（按综合评分降序）】")
out.append("")
header = f"{'排名':<4} {'代码':<8} {'名称':<10} {'总分':>5} {'空间(40)':>9} {'情绪(30)':>9} {'择时(30)':>9} {'板块(+10)':>9} {'硬过滤':<8} {'过滤原因'}"
out.append(header)
out.append("-" * len(header))

for i, r in enumerate(results, 1):
    filter_status = r["硬过滤"]
    filter_reason = r["过滤原因"]
    line = (
        f"{i:<4} "
        f"{r['代码']:<8} "
        f"{r['名称']:<10} "
        f"{r['综合评分']:>5} "
        f"{r['空间维度']:>9} "
        f"{r['情绪量能']:>9} "
        f"{r['择时确认']:>9} "
        f"{r['板块共振']:>9} "
        f"{filter_status:<8} "
        f"{filter_reason}"
    )
    out.append(line)

out.append("")
out.append("")

# ── 中间指标明细 ──
out.append("【二、关键中间指标明细】")
out.append("")

for i, r in enumerate(results, 1):
    ind = r["中间指标"]
    out.append(f"  {i}. {r['代码']} {r['名称']}  (总分:{r['综合评分']} | 硬过滤:{r['硬过滤']}"
              + (f"({r['过滤原因']})" if r['过滤原因'] else "") + ")")
    out.append(f"     当前收盘: {ind.get('当前收盘','N/A')}  |  近120日最高点: {ind.get('近120日最高点','N/A')}  |  回调幅度: {ind.get('回调幅度','N/A')}")
    out.append(f"     BIAS(MA20): {ind.get('BIAS(MA20)','N/A')}  |  量能比: {ind.get('量能比(5MA/60max)','N/A')}")
    out.append(f"     近3日涨跌: {ind.get('近3日涨跌','N/A')}  |  近5日涨跌: {ind.get('近5日涨跌','N/A')}  |  近20日涨跌: {ind.get('近20日涨跌','N/A')}")
    out.append(f"     MA5: {ind.get('MA5','N/A')}  |  KDJ_J: {ind.get('KDJ_J','N/A')}  |  MACD: {ind.get('MACD','N/A')}")
    out.append(f"     低位企稳: {ind.get('低位企稳','N/A')}  |  近5日最低: {ind.get('近5日最低','N/A')}")
    out.append("")

out.append(sep)
out.append("  报告结束")
out.append(sep)

# ── 写入文件 ──
output_text = "\n".join(out)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"\n结果已写入: {OUTPUT_FILE}")

# ── 也打印到控制台 ──
print("\n" + output_text)
