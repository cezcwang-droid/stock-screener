"""
超跌反弹模型（OversoldScorer）单元测试。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.scoring.oversold import calculate_oversold_rebound_score, OversoldScorer
from core.scorer_base import ScoreInput
from tests.conftest import make_sample_kline, make_stock_data


class TestOversoldScorer:
    def test_short_kline_returns_nominal(self):
        """K线不足20日应返回但所有维度为0"""
        kline = make_sample_kline(n=10, trend="down", seed=400)
        result = calculate_oversold_rebound_score(kline)
        assert result.get("pass") is True
        assert result["综合评分"] == 0

    def test_long_kline_down_trend(self):
        """下跌趋势K线应给出评分"""
        kline = make_sample_kline(n=120, trend="down", seed=401)
        result = calculate_oversold_rebound_score(kline)
        assert result.get("pass") is True
        assert "空间维度" in result
        assert "情绪量能" in result
        assert "择时确认" in result

    def test_dimensions_present(self):
        """评分应包含所有四维"""
        kline = make_sample_kline(n=120, trend="v", seed=402)
        result = calculate_oversold_rebound_score(kline)
        expected = ["空间维度", "情绪量能", "择时确认", "板块共振"]
        for dim in expected:
            assert dim in result, f"缺少维度 {dim}"

    def test_different_trends_different_scores(self):
        """下跌趋势应有更高的超跌反弹分数"""
        kline_down = make_sample_kline(n=120, trend="down", seed=403)
        kline_up = make_sample_kline(n=120, trend="up", seed=404)
        r_down = calculate_oversold_rebound_score(kline_down)
        r_up = calculate_oversold_rebound_score(kline_up)
        # 下跌趋势的空间维度应更高
        assert r_down["空间维度"] >= r_up["空间维度"]

    def test_scorer_short_kline(self):
        """Scorer类短K线处理"""
        kline = make_sample_kline(n=10, trend="down", seed=405)
        scorer = OversoldScorer()
        si = ScoreInput(kline_df=kline)
        result = scorer.score(score_input=si)
        assert result.passed is False
        assert result.score == 0

    def test_scorer_long_kline(self):
        """Scorer类长K线评分"""
        kline = make_sample_kline(n=120, trend="down", seed=406)
        scorer = OversoldScorer()
        si = ScoreInput(kline_df=kline)
        result = scorer.score(score_input=si)
        assert result.passed is True
        assert 0 <= result.score <= 100

    def test_scorer_with_sector_bonus(self):
        """Scorer类带板块数据评分"""
        kline = make_sample_kline(n=120, trend="down", seed=407)
        sd = make_stock_data(sector="银行")
        scorer = OversoldScorer()
        si = ScoreInput(kline_df=kline, stock_data=sd)
        result = scorer.score(score_input=si)
        assert result.passed is True
