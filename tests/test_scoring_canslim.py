"""
CAN SLIM 模型（CanslimScorer）单元测试。

测试范围：
- calculate_canslim_score 七因子评分
- CanslimScorer.compute() 
- 边界条件（短K线、无财务数据）
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.scoring.canslim import calculate_canslim_score, CanslimScorer
from core.scorer_base import ScoreInput
from tests.conftest import make_sample_kline


class TestCanslimScorer:
    def test_short_kline_returns_pass_false(self):
        """K线不足60日应返回 passed=False"""
        kline = make_sample_kline(n=40, trend="up", seed=200)
        result = calculate_canslim_score("000001", kline)
        assert result.get("pass") is False

    def test_long_kline_no_context(self):
        """长K线无stock_pool_context时不应崩溃"""
        kline = make_sample_kline(n=250, trend="up", seed=201)
        result = calculate_canslim_score("000001", kline)
        # 无上下文时，C/A 维度为 -1（N/A）
        assert "C_业绩增速" in result
        assert "A_持续增长" in result
        assert result.get("pass") is True

    def test_dimensions_present(self):
        """评分应包含所有七因子维度"""
        kline = make_sample_kline(n=250, trend="up", seed=202)
        result = calculate_canslim_score("000001", kline)
        expected = ["C_业绩增速", "A_持续增长", "N_新催化", "S_中小盘",
                     "L_RPS", "I_流动性", "M_大势"]
        for dim in expected:
            assert dim in result, f"缺少维度 {dim}"

    def test_scorer_short_kline(self):
        """Scorer类短K线处理"""
        kline = make_sample_kline(n=30, trend="up", seed=203)
        scorer = CanslimScorer()
        si = ScoreInput(code="000001", kline_df=kline)
        result = scorer.score(score_input=si)
        assert result.passed is False
        assert result.score == 0

    def test_scorer_long_kline(self):
        """Scorer类长K线评分"""
        kline = make_sample_kline(n=250, trend="up", seed=204)
        scorer = CanslimScorer()
        si = ScoreInput(code="000001", kline_df=kline)
        result = scorer.score(score_input=si)
        assert result.passed is True
        assert 0 <= result.score <= 100

    def test_scorer_with_context(self):
        """Scorer类带stock_pool_context评分"""
        kline = make_sample_kline(n=250, trend="up", seed=205)
        ctx = {
            "fin": {
                "success": True,
                "eps_growth_yoy": 0.30,
                "roe": 0.20,
                "eps_cagr_3y": 0.25,
                "market_cap": 3e10,  # 300亿
                "turnover_rate": 5.0,
            },
            "rps": 85,
            "market_cap": 3e10,
            "turnover_rate": 5.0,
        }
        scorer = CanslimScorer()
        si = ScoreInput(code="000001", kline_df=kline, stock_pool_context=ctx)
        result = scorer.score(score_input=si)
        assert result.passed is True
        # 有财务数据时 C 和 A 应 > 0
        assert result.dimensions.get("C_业绩增速", 0) > 0
        assert result.dimensions.get("A_持续增长", 0) > 0
