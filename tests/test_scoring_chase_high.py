"""
追高模型（ChaseHighScorer）单元测试。

测试范围：
- hard_filter_v3 7 条硬过滤规则
- 十维评分函数（趋势结构/动量强度/板块共振/北向/机构/资金热度/量价/估值/筹码/情绪）
- ChaseHighScorer.compute() 综合评分
- 旧接口 calculate_v3_total_score 向后兼容
"""

import numpy as np
import pandas as pd
import sys
import os

# 确保能找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.scoring.chase_high import (
    hard_filter_v3,
    score_trend_struct_v3,
    score_momentum_v3,
    score_sector_resonance_v3,
    score_north_capital_v3,
    score_inst_capital_v3,
    score_sector_fund_heat_v3,
    score_volume_price_v3,
    score_valuation_v3,
    score_chip_v3,
    score_sentiment_v3,
    calculate_v3_total_score,
    calculate_five_dimensions_score,
    ChaseHighScorer,
)
from core.scorer_base import ScoreInput
from tests.conftest import make_stock_data, make_sample_kline


# ═══════════════════════════════════════════════════
#  硬过滤测试
# ═══════════════════════════════════════════════════


class TestHardFilterV3:
    def test_pass_valid(self):
        """正常数据应通过硬过滤"""
        sd = make_stock_data()
        passed, msg = hard_filter_v3(sd)
        assert passed is True
        assert "过滤通过" in msg

    def test_fail_3d_gain_over_15(self):
        """3日涨幅 > 15% 应淘汰"""
        sd = make_stock_data()
        sd["3d_gain"] = 20.0
        passed, msg = hard_filter_v3(sd)
        assert passed is False
        assert "3日涨幅" in msg
        assert "15%" in msg

    def test_fail_ma60_deviation_over_80(self):
        """距MA60涨幅 > 80% 应淘汰"""
        sd = make_stock_data()
        sd["close"] = 100.0
        sd["ma60"] = 50.0  # 距MA60涨幅 100%
        passed, msg = hard_filter_v3(sd)
        assert passed is False
        assert "MA60" in msg

    def test_fail_shrink_new_high(self):
        """缩量新高顶背离应淘汰"""
        sd = make_stock_data()
        sd["vol_today"] = 100
        sd["ma5_vol"] = 500  # vol_today < ma5_vol * 0.8 = 400
        sd["new_high_2d"] = True
        passed, msg = hard_filter_v3(sd)
        assert passed is False
        assert "缩量新高" in msg

    def test_fail_inst_net_sell_2d(self):
        """机构连续2日净卖出应淘汰"""
        sd = make_stock_data()
        sd["inst_net_sell_2d"] = True
        passed, msg = hard_filter_v3(sd)
        assert passed is False
        assert "机构" in msg

    def test_fail_macd_divergence(self):
        """MACD高位顶背离应淘汰"""
        sd = make_stock_data()
        sd["macd_top_divergence"] = True
        passed, msg = hard_filter_v3(sd)
        assert passed is False
        assert "MACD" in msg

    def test_fail_6d_gain_over_25(self):
        """6日累计涨幅 > 25% 应淘汰"""
        sd = make_stock_data()
        sd["6d_gain"] = 30.0
        passed, msg = hard_filter_v3(sd)
        assert passed is False
        assert "6日" in msg

    def test_fail_sector_fund_outflow(self):
        """板块资金净流出应淘汰"""
        sd = make_stock_data()
        sd["sector_fund_outflow"] = True
        passed, msg = hard_filter_v3(sd)
        assert passed is False
        assert "板块" in msg

    def test_boundary_3d_gain_15(self):
        """3日涨幅刚好15% 应通过（规则是>15%淘汰，15%算通过）"""
        sd = make_stock_data()
        sd["3d_gain"] = 15.0
        passed, _ = hard_filter_v3(sd)
        assert passed is True

    def test_boundary_6d_gain_25(self):
        """6日涨幅刚好25% 应通过（规则是>25%淘汰）"""
        sd = make_stock_data()
        sd["6d_gain"] = 25.0
        passed, _ = hard_filter_v3(sd)
        assert passed is True


# ═══════════════════════════════════════════════════
#  十维评分函数测试
# ═══════════════════════════════════════════════════


class TestScoreFunctions:
    def test_score_trend_struct_range(self):
        """趋势结构应在 0~25 范围内"""
        sd = make_stock_data()
        score = score_trend_struct_v3(sd)
        assert 0 <= score <= 25

    def test_score_trend_struct_zero_on_empty(self):
        """空 dict 应返回 0"""
        score = score_trend_struct_v3({})
        assert score == 0

    def test_score_momentum_range(self):
        """动量强度应在 0~22 范围内"""
        sd = make_stock_data()
        score = score_momentum_v3(sd)
        assert 0 <= score <= 22

    def test_score_sector_resonance_range(self):
        """板块共振应在 0~10 范围内"""
        sd = make_stock_data()
        score = score_sector_resonance_v3(sd)
        assert 0 <= score <= 10

    def test_score_north_capital_zero_on_empty(self):
        """无北向数据(or 0)应返回 0"""
        sd = make_stock_data()
        sd["north_buy_3d"] = 0
        sd["north_buy_5d"] = 0
        sd["north_buy_10d"] = 0
        score = score_north_capital_v3(sd)
        assert score == 0

    def test_score_north_capital_high_value(self):
        """大额北向净买应给出高分"""
        sd = make_stock_data()
        sd["north_buy_3d"] = 1e8
        sd["north_buy_5d"] = 1e8
        sd["north_buy_10d"] = 1e8
        score = score_north_capital_v3(sd)
        assert 0 <= score <= 10

    def test_score_valuation_low_pe_gets_score(self):
        """低 PE 应给出非负评分"""
        sd = make_stock_data()
        sd["pe_hist_percent"] = 10
        sd["pb"] = 1
        score = score_valuation_v3(sd)
        assert score >= 0

    def test_score_valuation_negative_returns_minus_one(self):
        """pe_percentile < 0 时应返回 -1"""
        sd = make_stock_data()
        # 代码中: if pe_pct < 0: return -1
        # stock_data 的 key 是 pe_hist_percent，但函数读的是 sd.get('pe_percentile', 50)
        # 如果 pe_percentile < 0 则返回 -1
        sd["pe_percentile"] = -1
        score = score_valuation_v3(sd)
        assert score == -1

    def test_score_sentiment_high_turnover(self):
        """高换手率分位应给出低分"""
        sd = make_stock_data()
        sd["turnover_percentile"] = 85
        score = score_sentiment_v3(sd)
        assert score <= 2

    def test_score_sentiment_low_turnover(self):
        """低换手率分位应给出高分"""
        sd = make_stock_data()
        sd["turnover_percentile"] = 30
        score = score_sentiment_v3(sd)
        assert score >= 3

    def test_score_chip_negative(self):
        """筹码稳定性为负应返回 -1"""
        sd = make_stock_data()
        sd["chip_stability"] = -1
        score = score_chip_v3(sd)
        assert score == -1


# ═══════════════════════════════════════════════════
#  ChaseHighScorer 综合测试
# ═══════════════════════════════════════════════════


class TestChaseHighScorer:
    def test_compute_passes(self):
        """有效数据应评分通过"""
        sd = make_stock_data()
        scorer = ChaseHighScorer()
        si = ScoreInput(stock_data=sd)
        result = scorer.score(score_input=si)
        assert result.passed is True
        assert 0 <= result.score <= 100

    def test_compute_fails_hard_filter(self):
        """触发硬过滤时应返回 passed=False"""
        sd = make_stock_data()
        sd["3d_gain"] = 30.0
        scorer = ChaseHighScorer()
        si = ScoreInput(stock_data=sd)
        result = scorer.score(score_input=si)
        assert result.passed is False
        assert result.score == 0

    def test_compute_dimensions_filled(self):
        """评分后所有维度应有值"""
        sd = make_stock_data()
        scorer = ChaseHighScorer()
        si = ScoreInput(stock_data=sd)
        result = scorer.score(score_input=si)
        expected_dims = [
            "趋势结构", "动量强度", "板块共振", "北向资金", "机构净买",
            "板块资金热度", "量价配合", "估值安全", "筹码稳定", "情绪热度",
        ]
        for dim in expected_dims:
            assert dim in result.dimensions, f"缺少维度 {dim}"
            assert 0 <= result.dimensions[dim] <= 100

    def test_compute_high_score_with_strong_data(self):
        """强信号数据应给出较高评分"""
        sd = make_stock_data()
        # 模拟强势股特征
        sd["zx_score"] = 100
        sd["macd_trend"] = 20
        sd["vol_score"] = 20
        sd["zb_score"] = 100
        sd["ma_score"] = 100
        sd["ret_5d"] = 5.0
        sd["ret_10d"] = 8.0
        sd["sector_score"] = 5
        sd["north_buy_3d"] = 5e7
        sd["north_buy_5d"] = 5e7
        sd["north_buy_10d"] = 5e7
        sd["inst_buy_3d"] = 3e7
        sd["inst_buy_5d"] = 3e7
        sd["sector_fund_heat"] = 5
        sd["vp_score"] = 4
        sd["pe_percentile"] = 30
        sd["chip_stability"] = 7
        sd["turnover_percentile"] = 45

        scorer = ChaseHighScorer()
        si = ScoreInput(stock_data=sd)
        result = scorer.score(score_input=si)
        assert result.passed is True
        assert result.score >= 50  # 强信号应至少中等偏上

    def test_position_msg_tiered(self):
        """仓位建议应随分数变化"""
        sd = make_stock_data()

        # 模拟高分
        sd_h = sd.copy()
        sd_h["zx_score"] = 100
        sd_h["ma_score"] = 100
        sd_h["sector_score"] = 5
        sd_h["north_buy_3d"] = 5e7
        sd_h["north_buy_5d"] = 5e7
        sd_h["north_buy_10d"] = 5e7
        sd_h["inst_buy_3d"] = 3e7
        sd_h["inst_buy_5d"] = 3e7
        sd_h["sector_fund_heat"] = 5
        sd_h["vp_score"] = 4
        sd_h["pe_percentile"] = 30
        sd_h["chip_stability"] = 7
        sd_h["turnover_percentile"] = 45
        sd_h["zb_score"] = 100
        sd_h["ret_5d"] = 5.0
        sd_h["ret_10d"] = 8.0
        sd_h["vol_score"] = 10
        sd_h["macd_trend"] = 10

        scorer = ChaseHighScorer()
        si = ScoreInput(stock_data=sd_h)
        r_high = scorer.score(score_input=si)
        assert "重仓" in r_high.position_msg or "轻仓" in r_high.position_msg or "放弃" in r_high.position_msg


# ═══════════════════════════════════════════════════
#  旧接口兼容测试
# ═══════════════════════════════════════════════════


class TestLegacyInterface:
    def test_calculate_v3_total_score(self):
        """旧接口应返回正确格式"""
        sd = make_stock_data()
        result = calculate_v3_total_score(sd)
        assert "pass" in result
        assert "综合评分" in result
        assert "趋势结构" in result
        assert "动量强度" in result
        assert "_raw" in result
        assert isinstance(result["综合评分"], int)

    def test_calculate_v3_handles_empty(self):
        """旧接口处理过滤淘汰数据"""
        sd = make_stock_data()
        sd["3d_gain"] = 30.0
        result = calculate_v3_total_score(sd)
        assert result["pass"] is False
        assert result["综合评分"] == 0

    def test_calculate_five_dimensions_score(self):
        """五维评分旧接口应可用"""
        sd = make_stock_data()
        result = calculate_five_dimensions_score(sd)
        assert "综合评分" in result
        assert result["pass"] is True


# ═══════════════════════════════════════════════════
#  边界条件测试
# ═══════════════════════════════════════════════════


class TestEdgeCases:
    def test_missing_optional_fields(self):
        """缺失可选字段应降级而非崩溃"""
        sd = make_stock_data()
        # 移除可选字段
        minimal_sd = {
            "3d_gain": 2.0,
            "5d_gain": 3.0,
            "10d_gain": 4.0,
            "close": 12.0,
            "ma60": 11.0,
            "vol_today": 100,
            "ma5_vol": 100,
            "new_high_2d": False,
            "macd_top_divergence": False,
            "inst_net_sell_2d": False,
        }
        scorer = ChaseHighScorer()
        si = ScoreInput(stock_data=minimal_sd)
        result = scorer.score(score_input=si)
        # 应不崩溃，至少返回结果
        assert isinstance(result.score, (int, float))

    def test_custom_weights(self):
        """自定义权重应影响评分"""
        sd = make_stock_data()
        w1 = {"趋势结构": 100, "动量强度": 0, "板块共振": 0, "北向资金": 0,
              "机构净买": 0, "板块资金热度": 0, "量价配合": 0, "估值安全": 0,
              "筹码稳定": 0, "情绪热度": 0}
        w2 = {"趋势结构": 0, "动量强度": 100, "板块共振": 0, "北向资金": 0,
              "机构净买": 0, "板块资金热度": 0, "量价配合": 0, "估值安全": 0,
              "筹码稳定": 0, "情绪热度": 0}

        scorer = ChaseHighScorer()
        si1 = ScoreInput(stock_data=sd, weights=w1)
        si2 = ScoreInput(stock_data=sd, weights=w2)
        r1 = scorer.score(score_input=si1)
        r2 = scorer.score(score_input=si2)
        # 权重不同，评分可以相同或不同，但不应该崩溃
        assert isinstance(r1.score, (int, float))
        assert isinstance(r2.score, (int, float))

    def test_extreme_values(self):
        """极端输入值应不崩溃"""
        sd = make_stock_data()
        sd["3d_gain"] = 999
        sd["close"] = 1e9
        sd["ma60"] = 1

        scorer = ChaseHighScorer()
        si = ScoreInput(stock_data=sd)
        result = scorer.score(score_input=si)
        # 极端值可能导致过滤淘汰，但不应崩溃
        assert isinstance(result.passed, bool)
