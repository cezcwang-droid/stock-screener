"""
共振模型（ResonanceScorer）单元测试。

注意：共振模型依赖 DDE 数据（外部 API），
本测试只测试 ResonanceScorer 的非IO部分。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.scoring.resonance import ResonanceScorer, _classify_resonance_style
from core.scorer_base import ScoreInput
from tests.conftest import make_sample_kline


class TestResonanceScorer:
    def test_no_resonance_data_returns_not_passed(self):
        """无共振数据时应返回 passed=False"""
        scorer = ResonanceScorer()
        si = ScoreInput(code="000001")
        result = scorer.score(score_input=si)
        assert result.passed is False
        assert result.score == 0

    def test_with_resonance_data(self):
        """带模拟共振数据时应评分通过"""
        rd = {
            "000001": {
                "money_flow_score": 15.0,
                "dde_proxy_score": 10.0,
                "kline_structure_score": 15.0,
                "sector_heat_score": 12.0,
                "kline_structure_raw": 15.0,
                "sector_heat_raw": 12.0,
            }
        }
        scorer = ResonanceScorer()
        si = ScoreInput(code="000001", resonance_data=rd)
        result = scorer.score(score_input=si)
        assert result.passed is True
        expected_total = 15.0 + 10.0 + 15.0 + 12.0
        assert abs(result.score - expected_total) < 0.1

    def test_dimensions_present(self):
        """评分应包含所有四维"""
        rd = {
            "000001": {
                "money_flow_score": 20.0,
                "dde_proxy_score": 15.0,
                "kline_structure_score": 20.0,
                "sector_heat_score": 15.0,
                "kline_structure_raw": 20.0,
                "sector_heat_raw": 15.0,
            }
        }
        scorer = ResonanceScorer()
        si = ScoreInput(code="000001", resonance_data=rd)
        result = scorer.score(score_input=si)
        assert "资金流向" in result.dimensions
        assert "DDE决策" in result.dimensions
        assert "K线结构" in result.dimensions
        assert "板块热度" in result.dimensions

    def test_high_score_strong_resonance(self):
        """强共振信号应给出对应信号分类"""
        rd = {
            "000001": {
                "money_flow_score": 28.0,
                "dde_proxy_score": 18.0,
                "kline_structure_score": 24.0,
                "sector_heat_score": 22.0,
                "kline_structure_raw": 24.0,
                "sector_heat_raw": 22.0,
            }
        }
        scorer = ResonanceScorer()
        si = ScoreInput(code="000001", resonance_data=rd)
        result = scorer.score(score_input=si)
        assert result.score >= 80
        assert "强烈" in result.signal or "强烈" in result.position_msg

    def test_classify_resonance_style_no_quotes(self):
        """无行情数据时分类应为 '蓄势待发'"""
        style = _classify_resonance_style("000001", {}, None)
        assert style == "蓄势待发"
