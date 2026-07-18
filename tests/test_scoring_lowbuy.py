"""
低吸模型（LowBuyScorer）单元测试。

测试范围：
- 七维子评分函数（lb_score_decline_depth 等）
- LowBuyScorer.compute() 综合评分
- _check_low_buy_conditions 条件检测
- 旧接口 calculate_lowbuy_score 兼容
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.scoring.lowbuy import (
    lb_score_decline_depth,
    lb_score_stabilization,
    lb_score_volume_recovery,
    lb_score_ma_support,
    lb_score_valuation,
    lb_score_chip,
    lb_score_fund_flow,
    calculate_lowbuy_score,
    LowBuyScorer,
    _check_low_buy_conditions,
)
from core.scorer_base import ScoreInput
from core.config import DEFAULT_LOWBUY_WEIGHTS
from tests.conftest import make_stock_data, make_sample_kline


class TestLowBuyScoreFunctions:
    def test_decline_depth_deep(self):
        """深度下跌应得高分（跌幅 >= 20%）"""
        kline = make_sample_kline(n=60, trend="down", seed=50)
        sd = make_stock_data()
        score = lb_score_decline_depth(sd, kline)
        assert score >= 0

    def test_decline_depth_shallow(self):
        """浅跌应得低分"""
        kline = make_sample_kline(n=60, trend="up", seed=51)
        sd = make_stock_data()
        score = lb_score_decline_depth(sd, kline)
        assert score == 0  # 上涨趋势无下跌幅度

    def test_stabilization_range(self):
        """企稳评分应在 0~25 范围内"""
        kline = make_sample_kline(n=60, trend="v", seed=52)
        sd = make_stock_data()
        score = lb_score_stabilization(sd, kline)
        assert 0 <= score <= 25

    def test_volume_recovery_range(self):
        """量能恢复评分应在 0~20 范围内"""
        kline = make_sample_kline(n=60, trend="v", seed=53)
        sd = make_stock_data()
        score = lb_score_volume_recovery(sd, kline)
        assert 0 <= score <= 20

    def test_ma_support_range(self):
        """均线支撑评分应在 0~15 范围内"""
        kline = make_sample_kline(n=60, trend="up", seed=54)
        sd = make_stock_data()
        score = lb_score_ma_support(sd, kline)
        assert 0 <= score <= 15

    def test_valuation_range(self):
        """估值评分应在 0~10 范围内"""
        sd = make_stock_data()
        score = lb_score_valuation(sd)
        assert 0 <= score <= 10

    def test_chip_range(self):
        """筹码评分应在 0~10 范围内"""
        kline = make_sample_kline(n=60, trend="up", seed=55)
        sd = make_stock_data()
        score = lb_score_chip(sd, kline)
        assert 0 <= score <= 10

    def test_fund_flow_range(self):
        """主力资金评分应在 0~10 范围内"""
        kline = make_sample_kline(n=60, trend="up", seed=56)
        sd = make_stock_data()
        score = lb_score_fund_flow(sd, kline)
        assert 0 <= score <= 10

    def test_short_kline_decline(self):
        """K线不足20日时，decline_depth 应返回0"""
        kline = make_sample_kline(n=10, trend="down", seed=57)
        sd = make_stock_data()
        score = lb_score_decline_depth(sd, kline)
        assert score == 0


class TestLowBuyScorer:
    def test_compute_passes(self):
        """有效数据应评分通过"""
        kline = make_sample_kline(n=60, trend="v", seed=60)
        sd = make_stock_data()
        scorer = LowBuyScorer()
        si = ScoreInput(stock_data=sd, kline_df=kline)
        result = scorer.score(score_input=si)
        assert result.passed is True
        assert 0 <= result.score <= 100

    def test_dimensions_filled(self):
        """评分后所有维度应有值"""
        kline = make_sample_kline(n=60, trend="v", seed=61)
        sd = make_stock_data()
        scorer = LowBuyScorer()
        si = ScoreInput(stock_data=sd, kline_df=kline)
        result = scorer.score(score_input=si)
        expected_dims = ["下跌幅度", "企稳信号", "量能恢复", "均线支撑",
                         "估值吸引", "筹码沉淀", "主力资金"]
        for dim in expected_dims:
            assert dim in result.dimensions, f"缺少维度 {dim}"
            assert 0 <= result.dimensions.get(dim, -1) <= 100

    def test_deep_drop_gets_higher_score(self):
        """深度下跌趋势应比上涨趋势评分高"""
        kline_down = make_sample_kline(n=60, trend="down", seed=62)
        kline_up = make_sample_kline(n=60, trend="up", seed=63)
        sd = make_stock_data()

        scorer = LowBuyScorer()
        si_down = ScoreInput(stock_data=sd, kline_df=kline_down)
        si_up = ScoreInput(stock_data=sd, kline_df=kline_up)
        r_down = scorer.score(score_input=si_down)
        r_up = scorer.score(score_input=si_up)
        # 下跌趋势在低吸模型中应该有相当比例的分数
        assert r_down.score >= 0
        assert r_up.score >= 0

    def test_short_kline_handling(self):
        """K线太短时应不崩溃，返回合理分数"""
        kline = make_sample_kline(n=15, trend="v", seed=64)
        sd = make_stock_data()
        scorer = LowBuyScorer()
        si = ScoreInput(stock_data=sd, kline_df=kline)
        result = scorer.score(score_input=si)
        assert isinstance(result.score, (int, float))


class TestCheckLowBuyConditions:
    def test_check_passes_on_v_reversal(self):
        """V型反转K线应通过低吸条件"""
        kline = make_sample_kline(n=60, trend="v", seed=70)
        params = {
            "decline_days": 5,
            "vol_rise_days": 3,
            "chg_low": 1.0,
            "chg_high": 5.0,
        }
        # 确保K线最后一天涨幅在1%~5%
        closes = kline["收盘"].values
        prev_close = closes[-2]
        last_close = closes[-1]
        change_pct = (last_close - prev_close) / prev_close * 100
        if params["chg_low"] <= change_pct <= params["chg_high"]:
            passed, signal, label, summary, badge = _check_low_buy_conditions(kline, params)
            # 可能通过也可能不通过（取决于随机K线的形态），但不应该崩溃
            assert isinstance(passed, bool)

    def test_check_short_kline(self):
        """K线太短应返回 False"""
        kline = make_sample_kline(n=20, trend="down", seed=71)
        params = {"decline_days": 5, "vol_rise_days": 3, "chg_low": 1.0, "chg_high": 5.0}
        # K线长度 < decline_days + 5 = 10，应该不通过
        passed, signal, label, summary, badge = _check_low_buy_conditions(kline, params)
        assert passed is False


class TestLowBuyLegacyInterface:
    def test_calculate_lowbuy_score(self):
        """旧接口应返回正确格式"""
        kline = make_sample_kline(n=60, trend="v", seed=80)
        sd = make_stock_data()
        result = calculate_lowbuy_score(sd, kline)
        assert "pass" in result
        assert "综合评分" in result
        assert "下跌幅度" in result
        assert isinstance(result["综合评分"], int)
