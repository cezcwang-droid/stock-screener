"""
pytest 共享 fixture 与测试工具。

提供 make_sample_kline() 生成模拟 K 线 DataFrame，
供所有评分模型和回测函数的单元测试使用。
"""

import numpy as np
import pandas as pd
import pytest


def make_sample_kline(
    n=60,
    start_price=10.0,
    trend="up",
    volatility=0.02,
    seed=42,
):
    """生成模拟 K 线 DataFrame，用于测试评分函数。

    Parameters
    ----------
    n : int
        K 线根数（交易日数），默认 60。
    start_price : float
        起始价格，默认 10.0。
    trend : str
        趋势类型: 'up'（上涨）、'down'（下跌）、'v'（先跌后涨）、'high_vol'（高波动）。
    volatility : float
        日波动幅度，默认 0.02（2%）。
    seed : int
        随机种子，保证可复现。

    Returns
    -------
    pd.DataFrame
        包含 '开盘'/'收盘'/'最高'/'最低'/'成交量' 列的 DataFrame，
        索引为日期。
    """
    rng = np.random.default_rng(seed)

    # 生成收盘价序列
    closes = np.zeros(n)
    closes[0] = start_price

    for i in range(1, n):
        ret = rng.normal(0, volatility)

        if trend == "up":
            ret += 0.003  # 微弱正向偏移
        elif trend == "down":
            ret += -0.003
        elif trend == "v":
            # 前半段下跌，后半段上涨
            half = n // 2
            ret += -0.006 if i <= half else 0.008
        elif trend == "high_vol":
            ret = rng.normal(0, volatility * 3)

        # 模拟大幅回撤场景
        if trend == "down" and i > n // 3:
            ret += -0.005  # 加速下跌

        closes[i] = closes[i - 1] * (1 + ret)

    # 确保价格 > 0
    closes = np.maximum(closes, 0.5)

    # 从收盘价反推开盘、最高、最低
    opens = np.zeros(n)
    opens[0] = start_price
    for i in range(1, n):
        half_bar = closes[i] * volatility * 0.3
        opens[i] = closes[i] + rng.uniform(-half_bar, half_bar)

    highs = np.maximum(closes, opens) + abs(rng.normal(0, volatility, n) * closes)
    lows = np.minimum(closes, opens) - abs(rng.normal(0, volatility, n) * closes)
    highs = np.maximum.reduce([highs, closes, opens])
    lows = np.minimum.reduce([lows, closes, opens])

    volumes = np.abs(rng.integers(100000, 10000000, size=n))

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    df = pd.DataFrame(
        {
            # 中文列名（评分模型使用）
            "开盘": opens.round(2),
            "收盘": closes.round(2),
            "最高": highs.round(2),
            "最低": lows.round(2),
            "成交量": volumes,
            # 英文列名（backtest/engine.py 使用）
            "open": opens.round(2),
            "close": closes.round(2),
            "high": highs.round(2),
            "low": lows.round(2),
            "volume": volumes,
        },
        index=dates,
    )
    return df


def make_stock_data(code="000001", name="平安银行", sector="银行"):
    """生成模拟的 stock_data 字典。"""
    return {
        "代码": code,
        "名称": name,
        "板块": sector,
        "3d_gain": 2.5,
        "5d_gain": 4.0,
        "10d_gain": 6.0,
        "6d_gain": 3.0,
        "close": 12.5,
        "ma5": 12.0,
        "ma10": 11.5,
        "ma20": 11.0,
        "ma60": 10.5,
        "ma5_prev": 11.8,
        "ma10_prev": 11.3,
        "ma20_prev": 10.9,
        "ma60_prev": 10.4,
        "vol_today": 5000000,
        "ma5_vol": 4000000,
        "amplitude": 3.5,
        "new_high_2d": False,
        "new_high_today": False,
        "inst_net_sell_2d": False,
        "macd_top_divergence": False,
        "sector_daily_gain": 1.5,
        "sector_limit_up_count_count": 2,
        "sector_net_inflow": 10000000,
        "inst_net_buy_3d": 5000000,
        "north_net_buy": 3000000,
        "pure_hot_money_only": False,
        "pe_hist_percent": 40,
        "pe_raw": 15.0,
        "turnover_rate": 5.0,
        "stock_hot_score": 6.0,
        "sector_fund_outflow": False,
        "float_cap": 1e9,
    }


@pytest.fixture
def sample_kline_up():
    """上涨趋势 K 线（60日）。"""
    return make_sample_kline(n=60, trend="up", seed=42)


@pytest.fixture
def sample_kline_down():
    """下跌趋势 K 线（60日）。"""
    return make_sample_kline(n=60, trend="down", seed=43)


@pytest.fixture
def sample_kline_v():
    """V 型反转 K 线（60日）。"""
    return make_sample_kline(n=60, trend="v", seed=44)


@pytest.fixture
def sample_kline_long():
    """长周期 K 线（250日）。"""
    return make_sample_kline(n=250, trend="v", seed=45)


@pytest.fixture
def sample_kline_short():
    """短周期 K 线（30日）。"""
    return make_sample_kline(n=30, trend="up", seed=46)


@pytest.fixture
def sample_stock_data():
    """标准 stock_data 字典。"""
    return make_stock_data()


@pytest.fixture
def sample_stock_data_golden_cross():
    """为金叉模型准备的 stock_data。"""
    sd = make_stock_data()
    sd["板块"] = "银行"
    return sd
