"""
backtest/engine.run_backtest 自动化验证测试。

策略：patch scan_all_stock_codes + _load_kline_cache，
直接注入构造好的 kline_dict，让回测走 mock 数据，
验证返回的 trades/daily/selection DataFrame 结构正确性。

覆盖范围：
1. 多模型回测（chase_high、buy_low、golden_cross 等）
2. 不同周期的回测区间
3. 空数据/无效区间边界情况
4. 返回数据结构和指标合理性验证
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from tests.conftest import make_sample_kline


# ============================================================
# 辅助函数：构造 mock kline_dict
# ============================================================

def make_mock_kline_dict(
    n_stocks=100,
    n_days=300,
    start_date="2024-06-01",
    codes=None,
    seed=42,
):
    """构造模拟的 kline_dict，用于 run_backtest 测试。

    每只股票独立趋势+随机种子，保证多样性。
    kline_dict 的格式: {code: pd.DataFrame}，
    DataFrame 包含 date/open/high/close/low/volume 列。

    注：n_stocks 必须 >= 100，否则 run_backtest 会因"有效K线太少"返回 None。
    """
    if n_stocks < 100:
        raise ValueError(f"n_stocks={n_stocks} < 100, run_backtest 要求至少 100 只股票")

    base_rng = np.random.default_rng(seed)
    if codes is None:
        codes = [f"{600000 + i:06d}" for i in range(n_stocks)]

    kline_dict = {}

    for idx, code in enumerate(codes):
        trends = ["up", "down", "v", "high_vol"]
        trend = trends[idx % len(trends)]
        stock_seed = int(base_rng.integers(1000, 9999))

        df = make_sample_kline(
            n=n_days,
            start_price=10.0 + (idx % 50) * 0.5,
            trend=trend,
            seed=stock_seed,
        )

        # 转成 backtest/engine.py 需要的格式
        df = df.rename_axis("date").reset_index()
        df = df.drop(columns=["date"])  # 删除 make_sample_kline 生成的日期
        trading_dates = pd.bdate_range(
            start=start_date,
            periods=n_days * 2,
            freq="B",
        )[:n_days]
        df["date"] = trading_dates[: len(df)].values

        df["volume"] = df["volume"].clip(lower=100000)

        kline_dict[code] = df

    return kline_dict


def make_short_kline_dict(n_stocks=5, n_days=30):
    """构造只有少量K线的 kline_dict，用于测试数据不足场景。
    注意：返回结果不符合 run_backtest 的最小要求（<100只），
    专用于测试边界条件。
    """
    base_rng = np.random.default_rng(42)
    codes = [f"{600000 + i:06d}" for i in range(n_stocks)]
    kline_dict = {}
    for idx, code in enumerate(codes):
        trends = ["up", "down", "v", "high_vol"]
        trend = trends[idx % len(trends)]
        stock_seed = int(base_rng.integers(1000, 9999))

        df = make_sample_kline(n=n_days, start_price=10.0, trend=trend, seed=stock_seed)
        df = df.rename_axis("date").reset_index()
        df = df.drop(columns=["date"])
        trading_dates = pd.bdate_range(start="2025-01-01", periods=n_days * 2, freq="B")[:n_days]
        df["date"] = trading_dates[: len(df)].values
        df["volume"] = df["volume"].clip(lower=100000)
        kline_dict[code] = df
    return kline_dict


# ============================================================
# 测试 Mock 基础设施
# ============================================================

class TestMakeMockKlineDict:
    """验证 mock kline_dict 构造器的正确性"""

    def test_basic_structure(self):
        kd = make_mock_kline_dict(n_stocks=100, n_days=100)
        assert len(kd) == 100
        for code, df in kd.items():
            assert "date" in df.columns
            assert "open" in df.columns
            assert "high" in df.columns
            assert "close" in df.columns
            assert "low" in df.columns
            assert "volume" in df.columns
            assert len(df) > 50

    def test_trading_days_only(self):
        """日期列应只包含工作日"""
        kd = make_mock_kline_dict(n_stocks=100, n_days=200)
        df = list(kd.values())[0]
        assert all(df["date"].dt.weekday < 5)

    def test_unique_codes(self):
        kd = make_mock_kline_dict(n_stocks=100)
        assert len(set(kd.keys())) == 100

    def test_short_kline_below_60(self):
        """不足60根K线 — 验证短K线场景（非100只限制）"""
        kd = make_short_kline_dict(n_stocks=3, n_days=30)
        assert all(len(df) < 60 for df in kd.values())
        # 但注意因为 <100 只，run_backtest 会返回 None
        assert len(kd) < 100

    def test_raise_on_less_than_100(self):
        with pytest.raises(ValueError, match="n_stocks=50 < 100"):
            make_mock_kline_dict(n_stocks=50)


# ============================================================
# 共享 fixture：所有回测测试使用同一个 kline_dict
# ============================================================

@pytest.fixture(scope="session")
def _shared_kline_dict():
    """全局共享的唯一 kline_dict，所有回测测试复用。"""
    return make_mock_kline_dict(n_stocks=100, n_days=300, seed=42)


# ============================================================
# run_backtest 端到端测试
# ============================================================

def _run_backtest_patched(
    kline_dict,
    start_date="2025-06-01",
    end_date="2025-07-01",
    model="chase_high",
    **extra_kwargs,
):
    """以 mock TDX 数据运行 run_backtest。

    自动 patch scan_all_stock_codes 和 _load_kline_cache，
    让回测引擎使用注入的 kline_dict 而非真实通达信文件。
    """
    from backtest.engine import run_backtest

    codes = list(kline_dict.keys())

    with patch("backtest.engine.scan_all_stock_codes", return_value=codes):
        with patch("backtest.engine._load_kline_cache", return_value=(kline_dict, True)):
            trades_df, daily_df, selections_df = run_backtest(
                start_date=start_date,
                end_date=end_date,
                model=model,
                **extra_kwargs,
            )

    return trades_df, daily_df, selections_df


class TestRunBacktestChaseHigh:
    """追高模型回测"""

    def test_returns_dataframes(self, _shared_kline_dict):
        """返回三个 DataFrame（trades、daily、selections）"""
        trades_df, daily_df, selections_df = _run_backtest_patched(
            _shared_kline_dict, start_date="2025-05-01", end_date="2025-07-01"
        )
        assert isinstance(trades_df, pd.DataFrame)
        assert isinstance(daily_df, pd.DataFrame)
        assert isinstance(selections_df, pd.DataFrame)

    def test_trades_columns(self, _shared_kline_dict):
        """trades DataFrame 应包含必要列"""
        trades_df, _, _ = _run_backtest_patched(
            _shared_kline_dict, start_date="2025-05-01", end_date="2025-07-01"
        )
        if len(trades_df) > 0:
            expected = {
                "code", "score", "selection_date", "buy_date", "sell_date",
                "buy_price", "sell_price", "shares",
                "gross_return", "net_return", "pnl", "hold_days",
                "resonance", "sell_reason",
            }
            assert expected.issubset(set(trades_df.columns)), \
                f"缺少列: {expected - set(trades_df.columns)}"

    def test_daily_columns(self, _shared_kline_dict):
        """daily DataFrame 应包含必要列"""
        _, daily_df, _ = _run_backtest_patched(
            _shared_kline_dict, start_date="2025-05-01", end_date="2025-07-01"
        )
        if len(daily_df) > 0:
            expected = {
                "date", "portfolio_value", "cash", "position_value",
                "effective_value", "num_positions", "sold_count",
            }
            assert expected.issubset(set(daily_df.columns))

    def test_selections_columns(self, _shared_kline_dict):
        """selections DataFrame 包含选股记录"""
        _, _, selections_df = _run_backtest_patched(
            _shared_kline_dict, start_date="2025-05-01", end_date="2025-07-01"
        )
        if len(selections_df) > 0:
            expected = {
                "selection_date", "rank", "code", "score", "next_buy_date",
            }
            assert expected.issubset(set(selections_df.columns))

    def test_trades_have_reasonable_sell_reasons(self, _shared_kline_dict):
        """sell_reason 应为已知类型之一"""
        trades_df, _, _ = _run_backtest_patched(
            _shared_kline_dict, start_date="2025-05-01", end_date="2025-07-01"
        )
        if len(trades_df) > 0:
            valid_reasons = {
                "stop_loss", "hard_tp", "moving_stop",
                "resonance", "resonance_time", "time_exit",
            }
            actual_reasons = set(trades_df["sell_reason"].unique())
            assert actual_reasons.issubset(valid_reasons), \
                f"未知卖出原因: {actual_reasons - valid_reasons}"

    def test_daily_portfolio_value_positive(self, _shared_kline_dict):
        """daily portfolio_value 应始终大于 0"""
        _, daily_df, _ = _run_backtest_patched(
            _shared_kline_dict, start_date="2025-05-01", end_date="2025-07-01"
        )
        if len(daily_df) > 0:
            assert (daily_df["portfolio_value"] > 0).all()

    def test_num_positions_non_negative(self, _shared_kline_dict):
        """持仓数不应为负"""
        _, daily_df, _ = _run_backtest_patched(
            _shared_kline_dict, start_date="2025-05-01", end_date="2025-07-01"
        )
        if len(daily_df) > 0:
            assert (daily_df["num_positions"] >= 0).all()


class TestRunBacktestMultiModel:
    """多模型回测验证"""

    @pytest.mark.parametrize("model", [
        "chase_high",
        "buy_low",
        "golden_cross",
        "resonance",
    ])
    def test_model_runs_without_error(self, _shared_kline_dict, model):
        """各模型回测不报错"""
        trades_df, daily_df, _ = _run_backtest_patched(
            _shared_kline_dict,
            start_date="2025-04-01",
            end_date="2025-06-01",
            model=model,
        )
        assert isinstance(trades_df, pd.DataFrame)
        assert isinstance(daily_df, pd.DataFrame)

    @pytest.mark.parametrize("model", [
        "chase_high",
        "buy_low",
        "golden_cross",
        "resonance",
    ])
    def test_model_produces_daily_records(self, _shared_kline_dict, model):
        """各模型至少产生 daily 记录"""
        _, daily_df, _ = _run_backtest_patched(
            _shared_kline_dict,
            start_date="2025-04-01",
            end_date="2025-06-01",
            model=model,
        )
        assert len(daily_df) > 0


class TestRunBacktestBoundary:
    """边界条件测试"""

    def test_short_backtest_period(self, _shared_kline_dict):
        """短期回测应返回有效 DataFrame 结构"""
        trades_df, daily_df, _ = _run_backtest_patched(
            _shared_kline_dict,
            start_date="2025-06-01",
            end_date="2025-06-07",
        )
        assert isinstance(daily_df, pd.DataFrame)

    def test_very_short_period_returns_none(self, _shared_kline_dict):
        """极短期（3天以下）交易日太少，返回 None"""
        from backtest.engine import run_backtest
        codes = list(_shared_kline_dict.keys())
        with patch("backtest.engine.scan_all_stock_codes", return_value=codes):
            with patch("backtest.engine._load_kline_cache", return_value=(_shared_kline_dict, True)):
                result = run_backtest(
                    start_date="2025-03-01",
                    end_date="2025-03-02",
                )
        assert result is None

    def test_no_data_returns_none(self):
        """空 kline_dict 应返回 None"""
        from backtest.engine import run_backtest
        with patch("backtest.engine.scan_all_stock_codes", return_value=[]):
            with patch("backtest.engine._load_kline_cache", return_value=({}, True)):
                result = run_backtest(
                    start_date="2025-01-01",
                    end_date="2025-03-01",
                )
        assert result is None

    def test_short_kline_dict_returns_none(self):
        """K线不足 100 只时返回 None（engine.py 第527行检查）"""
        from backtest.engine import run_backtest
        short_kd = make_short_kline_dict(n_stocks=3, n_days=50)
        codes = list(short_kd.keys())
        with patch("backtest.engine.scan_all_stock_codes", return_value=codes):
            with patch("backtest.engine._load_kline_cache", return_value=(short_kd, True)):
                result = run_backtest(
                    start_date="2025-01-01",
                    end_date="2025-03-01",
                )
        assert result is None

    def test_custom_top_n(self, _shared_kline_dict):
        """自定义 top_n 应反映在选股结果的 rank 列"""
        _, _, selections_df_3 = _run_backtest_patched(
            _shared_kline_dict,
            start_date="2025-05-01",
            end_date="2025-06-01",
            top_n=3,
        )
        _, _, selections_df_10 = _run_backtest_patched(
            _shared_kline_dict,
            start_date="2025-05-01",
            end_date="2025-06-01",
            top_n=10,
        )
        if len(selections_df_3) > 0 and len(selections_df_10) > 0:
            assert selections_df_3["rank"].max() <= 3
            assert selections_df_10["rank"].max() <= 10


class TestRunBacktestStats:
    """compute_stats 函数验证"""

    def _make_mock_trades(self):
        return pd.DataFrame({
            "code": ["600001", "600002"],
            "score": [85, 72],
            "buy_date": pd.to_datetime(["2025-06-01", "2025-06-05"]),
            "sell_date": pd.to_datetime(["2025-06-10", "2025-06-12"]),
            "buy_price": [10.0, 15.0],
            "sell_price": [11.0, 14.0],
            "shares": [1000, 600],
            "gross_return": [10.0, -6.67],
            "net_return": [9.7, -6.97],
            "pnl": [970.0, -697.0],
            "hold_days": [9, 7],
            "resonance": [55.0, 30.0],
            "sell_reason": ["hard_tp", "stop_loss"],
        })

    def _make_mock_daily(self):
        dates = pd.bdate_range(start="2025-06-01", periods=20)
        rng = np.random.default_rng(42)
        nav = 1_000_000 * (1 + rng.normal(0.001, 0.015, 20)).cumprod()
        return pd.DataFrame({
            "date": dates,
            "portfolio_value": nav,
            "cash": nav * 0.3,
            "position_value": nav * 0.7,
            "effective_value": nav,
            "total_invested": [800_000] * 20,
        })

    def test_compute_stats_returns_dict(self):
        """compute_stats 返回 dict"""
        from backtest.engine import compute_stats
        stats = compute_stats(self._make_mock_trades(), self._make_mock_daily(),
                              per_stock_amount=10000, top_n=5)
        assert isinstance(stats, dict)

    def test_compute_stats_key_fields(self):
        """stats 包含所有关键指标"""
        from backtest.engine import compute_stats
        stats = compute_stats(self._make_mock_trades(), self._make_mock_daily(),
                              per_stock_amount=10000, top_n=5)
        expected_fields = {
            "total_return", "max_drawdown", "sharpe", "calmar",
            "ann_return", "total_trades", "win_trades", "win_rate",
            "avg_return", "total_pnl", "avg_hold_days",
        }
        assert expected_fields.issubset(set(stats.keys()))

    def test_compute_stats_win_rate_correct(self):
        """胜率计算正确"""
        from backtest.engine import compute_stats
        stats = compute_stats(self._make_mock_trades(), self._make_mock_daily(),
                              per_stock_amount=10000, top_n=5)
        assert stats["total_trades"] == 2
        assert stats["win_trades"] == 1
        assert stats["win_rate"] == 50.0

    def test_compute_stats_none_on_empty(self):
        """空 trades 时返回 None"""
        from backtest.engine import compute_stats
        assert compute_stats(pd.DataFrame(), self._make_mock_daily(),
                             per_stock_amount=10000, top_n=5) is None

    def test_compute_stats_with_real_backtest_output(self, _shared_kline_dict):
        """用真实回测输出调用 compute_stats"""
        from backtest.engine import compute_stats
        trades_df, daily_df, _ = _run_backtest_patched(
            _shared_kline_dict,
            start_date="2025-05-01",
            end_date="2025-07-01",
        )
        if len(trades_df) > 0 and len(daily_df) > 0:
            stats = compute_stats(trades_df, daily_df, per_stock_amount=10000, top_n=5)
            assert isinstance(stats, dict)
            assert stats["total_trades"] == len(trades_df)
            assert 0 <= stats["win_rate"] <= 100


class TestRunBacktestSellReasons:
    """验证回测的卖出原因分布"""

    def test_sell_reasons_distribution(self, _shared_kline_dict):
        """卖出原因应为已知类型"""
        trades_df, _, _ = _run_backtest_patched(
            _shared_kline_dict,
            start_date="2025-04-01",
            end_date="2025-07-01",
            top_n=10,
        )
        if len(trades_df) >= 5:
            known_reasons = {"stop_loss", "hard_tp", "moving_stop",
                           "resonance", "resonance_time", "time_exit"}
            assert set(trades_df["sell_reason"].unique()).issubset(known_reasons)


class TestRunBacktestMultiPeriod:
    """不同时间长度回测"""

    @pytest.mark.parametrize("period", [
        ("2025-03-01", "2025-04-01"),
        ("2025-05-01", "2025-07-01"),
        ("2025-01-01", "2025-06-01"),
    ])
    def test_multi_period_runs(self, _shared_kline_dict, period):
        """不同回测区间都能运行"""
        start_d, end_d = period
        trades_df, daily_df, _ = _run_backtest_patched(
            _shared_kline_dict,
            start_date=start_d,
            end_date=end_d,
        )
        assert isinstance(trades_df, pd.DataFrame)
        assert isinstance(daily_df, pd.DataFrame)
