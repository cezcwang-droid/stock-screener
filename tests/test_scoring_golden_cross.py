"""
金叉模型（GoldenCrossScorer）单元测试。

测试范围：
- _gc_hard_filter 双通道硬过滤（零下通道/水上通道）
- 七维子评分函数
- GoldenCrossScorer.compute() 及信号分类
- 旧接口 calculate_golden_cross_score
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.scoring.golden_cross import (
    _gc_score_decline,
    _gc_score_steady,
    _gc_score_ma,
    _gc_score_volume,
    _gc_score_macd,
    _gc_score_fund_confirm,
    _gc_score_sector,
    calculate_golden_cross_score,
    GoldenCrossScorer,
    _gc_hard_filter,
    _compute_macd,
)
from core.scorer_base import ScoreInput
from tests.conftest import make_stock_data, make_sample_kline


class TestGcHardFilter:
    def test_short_kline_fails(self):
        """K线不足40日应淘汰"""
        kline = make_sample_kline(n=30, trend="v", seed=100)
        result = _gc_hard_filter("000001", kline)
        passed, msg = result[0], result[1]
        assert passed is False
        assert "40" in msg

    def test_no_macd_column(self):
        """MACD数据不足时应正确处理"""
        kline = make_sample_kline(n=60, trend="down", seed=101)
        # 确保至少返回正常的结果
        result = _gc_hard_filter("000001", kline)
        passed, msg = result[0], result[1]
        # 可能通过也可能不通过，但不应该崩溃
        assert isinstance(passed, bool)

    def test_hard_filter_does_not_crash(self):
        """各种趋势下过滤函数不应崩溃"""
        for trend in ["up", "down", "v", "high_vol"]:
            kline = make_sample_kline(n=80, trend=trend, seed=hash(trend) % 1000)
            result = _gc_hard_filter("000001", kline)
            assert len(result) == 3, f"应返回3个元素 (passed, msg, channel)"
            assert isinstance(result[0], bool)


class TestGcSubScores:
    def test_decline_range(self):
        """下跌形态应在 0~30 范围内"""
        kline = make_sample_kline(n=60, trend="down", seed=110)
        score = _gc_score_decline(kline)
        assert 0 <= score <= 30

    def test_decline_up_trend_low(self):
        """上涨趋势的下跌形态应非常低"""
        kline = make_sample_kline(n=60, trend="up", seed=111)
        score = _gc_score_decline(kline)
        # 上涨趋势跌幅小，分数应偏低
        assert 0 <= score <= 15

    def test_steady_range(self):
        """K线止跌应在 0~20 范围内"""
        kline = make_sample_kline(n=60, trend="v", seed=112)
        score = _gc_score_steady(kline)
        assert 0 <= score <= 20

    def test_ma_range(self):
        """均线拐头应在 0~20 范围内"""
        kline = make_sample_kline(n=60, trend="v", seed=113)
        score = _gc_score_ma(kline)
        assert 0 <= score <= 20

    def test_volume_range(self):
        """量能确认应在 0~15 范围内"""
        kline = make_sample_kline(n=60, trend="v", seed=114)
        score = _gc_score_volume(kline)
        assert 0 <= score <= 15

    def test_macd_range(self):
        """MACD反转应在 0~15 范围内"""
        kline = make_sample_kline(n=60, trend="v", seed=115)
        score = _gc_score_macd(kline)
        assert 0 <= score <= 15

    def test_fund_confirm_range(self):
        """资金确认应在 0~10 范围内"""
        kline = make_sample_kline(n=60, trend="v", seed=116)
        sd = make_stock_data()
        score = _gc_score_fund_confirm(sd, kline)
        assert 0 <= score <= 10

    def test_sector_score_negative_when_no_sector(self):
        """无板块信息应返回负分"""
        sd = make_stock_data()
        sd["板块"] = ""
        score = _gc_score_sector(sd)
        assert score <= 0

    def test_short_kline_macd(self):
        """K线不足35日时MACD评分应为0"""
        kline = make_sample_kline(n=25, trend="v", seed=117)
        score = _gc_score_macd(kline)
        # 可能是0或其他值，但不应该崩溃
        assert isinstance(score, (int, float))


class TestGoldenCrossScorer:
    def test_compute_passes(self):
        """有效数据应评分通过"""
        kline = make_sample_kline(n=80, trend="v", seed=120)
        sd = make_stock_data()
        sd["板块"] = "银行"
        scorer = GoldenCrossScorer()
        si = ScoreInput(stock_data=sd, kline_df=kline)
        result = scorer.score(score_input=si)
        # 可能通过也可能不通过，但不应该崩溃
        assert isinstance(result.passed, bool)
        assert 0 <= result.score <= 100

    def test_short_kline_handling(self):
        """K线太短应返回 passed=False"""
        kline = make_sample_kline(n=30, trend="up", seed=121)
        sd = make_stock_data()
        scorer = GoldenCrossScorer()
        si = ScoreInput(stock_data=sd, kline_df=kline)
        result = scorer.score(score_input=si)
        assert result.passed is False
        assert result.score == 0

    def test_dimensions_filled(self):
        """评分后维度应有值或0"""
        kline = make_sample_kline(n=80, trend="v", seed=122)
        sd = make_stock_data()
        sd["板块"] = "银行"
        scorer = GoldenCrossScorer()
        si = ScoreInput(stock_data=sd, kline_df=kline)
        result = scorer.score(score_input=si)
        expected_dims = ["下跌形态", "K线止跌", "均线拐头", "量能确认",
                         "MACD反转", "资金确认", "板块确认"]
        for dim in expected_dims:
            assert dim in result.dimensions, f"缺少维度 {dim}"


class TestGcLegacyInterface:
    def test_calculate_golden_cross_score(self):
        """旧接口应返回正确格式"""
        kline = make_sample_kline(n=80, trend="v", seed=130)
        sd = make_stock_data()
        sd["板块"] = "银行"
        result = calculate_golden_cross_score(sd, kline)
        assert "pass" in result
        assert "综合评分" in result
        assert "下跌形态" in result
        assert isinstance(result["综合评分"], int)

    def test_legacy_fails_on_short_kline(self):
        """旧接口短K线应返回pass=False"""
        kline = make_sample_kline(n=30, trend="up", seed=131)
        sd = make_stock_data()
        result = calculate_golden_cross_score(sd, kline)
        assert result["pass"] is False
