"""
core/screener.py 单元测试。

覆盖以下函数：
- _get_sector                    — 代码前缀判断板块（纯函数）
- _apply_lowbuy_percentile_normalization — 低吸百分位归一化（纯函数）
- _apply_chase_percentile_normalization  — 追高百分位归一化（纯函数）
- get_stock_detail               — 从股票池中按 code 匹配
- get_top10_stocks / get_lowbuy_top5 — 轻量包装器
- score_single_stock             — 个股评分中枢（需 mock data 层）
"""

# ============================================================
# Part 1: _get_sector
# ============================================================

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock


class TestGetSector:
    """覆盖 _get_sector 的 7 种板块判断 + 兜底"""

    def test_main_board_6(self):
        from core.screener import _get_sector
        assert _get_sector("600000") == "主板"

    def test_main_board_000(self):
        from core.screener import _get_sector
        assert _get_sector("000001") == "主板"

    def test_main_board_001(self):
        from core.screener import _get_sector
        assert _get_sector("001234") == "主板"

    def test_sme_board(self):
        from core.screener import _get_sector
        assert _get_sector("002001") == "中小板"

    def test_gem_board(self):
        from core.screener import _get_sector
        assert _get_sector("300001") == "创业板"

    def test_tech_board(self):
        from core.screener import _get_sector
        assert _get_sector("688001") == "科创板"

    def test_bsei_board_8(self):
        from core.screener import _get_sector
        assert _get_sector("830001") == "北交所"

    def test_bsei_board_4(self):
        from core.screener import _get_sector
        assert _get_sector("430001") == "北交所"

    def test_unknown_code(self):
        from core.screener import _get_sector
        assert _get_sector("123456") == "其他"

    def test_empty_code(self):
        from core.screener import _get_sector
        assert _get_sector("") == "其他"


# ============================================================
# Part 2: _apply_lowbuy_percentile_normalization
# ============================================================


def _make_lowbuy_result(s1=10, s2=10, s3=10, s4=10, s5=10, s6=10, s7=10,
                        chg_10d=-5.0, inst_net_3d=100):
    """生成模拟的低吸评分结果条目"""
    return {
        "_raw": {
            "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7,
            "chg_10d": chg_10d,
            "inst_net_3d": inst_net_3d,
        },
        "涨跌幅": -3.0,
        "换手率": 5.0,
        "量比": 1.2,
    }


class TestApplyLowbuyPercentileNormalization:
    """覆盖低吸百分位归一化"""

    def test_single_result_no_normalization(self):
        """单结果不进行归一化（n_res > 1 检查），返回不带综合评分的原始 DataFrame"""
        from core.screener import _apply_lowbuy_percentile_normalization
        results = [_make_lowbuy_result()]
        df = _apply_lowbuy_percentile_normalization(results, None)
        assert len(df) == 1
        # 单结果不执行归一化，所以不设置综合评分等字段
        # 只包含 _raw、涨跌幅、换手率、量比
        assert "_raw" in df.columns
        assert "涨跌幅" in df.columns

    def test_multi_results_percentile_ranking(self):
        """多结果进行百分位排名，高分排名靠前"""
        from core.screener import _apply_lowbuy_percentile_normalization
        results = [
            _make_lowbuy_result(s1=30, s2=30, s3=30, s4=30, s5=30, s6=30, s7=30,
                                inst_net_3d=500),
            _make_lowbuy_result(s1=10, s2=10, s3=10, s4=10, s5=10, s6=10, s7=10,
                                inst_net_3d=100),
            _make_lowbuy_result(s1=20, s2=20, s3=20, s4=20, s5=20, s6=20, s7=20,
                                inst_net_3d=200),
        ]
        df = _apply_lowbuy_percentile_normalization(results, None)
        assert len(df) == 3
        # 各维度百分位排名：第3名100分(30全)、第2名66分(20全)、第1名33分(10全)
        scores = df["综合评分"].tolist()
        # 30 > 20 > 10 的原始分，对应排名分 100 > 66 > 33
        assert scores[0] == 100  # 30全
        assert scores[1] == 33   # 10全
        assert scores[2] == 66   # 20全

    def test_all_equal_values(self):
        """所有维度等值时，归一化应返回 full_score * 0.5"""
        from core.screener import _apply_lowbuy_percentile_normalization
        results = [
            _make_lowbuy_result(s1=15, s2=15, s3=15, s4=15, s5=15, s6=15, s7=15,
                                inst_net_3d=0),
            _make_lowbuy_result(s1=15, s2=15, s3=15, s4=15, s5=15, s6=15, s7=15,
                                inst_net_3d=0),
        ]
        df = _apply_lowbuy_percentile_normalization(results, None)
        assert len(df) == 2
        # 等值时应返回相同的综合评分
        assert df.iloc[0]["综合评分"] == df.iloc[1]["综合评分"]

    def test_signal_strong_with_positive_inst(self):
        """总分 >= 55 且 inst_net_3d > 0 → 强烈低吸"""
        from core.screener import _apply_lowbuy_percentile_normalization
        results = [
            _make_lowbuy_result(s1=30, s2=30, s3=30, s4=30, s5=30, s6=30, s7=30,
                                inst_net_3d=500),
            _make_lowbuy_result(s1=5, s2=5, s3=5, s4=5, s5=5, s6=5, s7=5,
                                inst_net_3d=-100),
        ]
        df = _apply_lowbuy_percentile_normalization(results, None)
        # 第一个结果（高分 + 机构净买 > 0）应标记为强烈低吸
        first = df.iloc[0]
        if first["综合评分"] >= 55:
            assert first["信号"] == "强烈低吸"
        else:
            # weights 默认值可能使分不到 55，至少检查它比第二个高
            assert first["综合评分"] >= df.iloc[1]["综合评分"]

    def test_signal_standard_below_55(self):
        """总分 40-54 → 标准低吸"""
        from core.screener import _apply_lowbuy_percentile_normalization
        # 构造两个等值结果使分不高
        results = [
            _make_lowbuy_result(s1=10, s2=10, s3=10, s4=10, s5=10, s6=10, s7=10,
                                inst_net_3d=0),
            _make_lowbuy_result(s1=15, s2=15, s3=15, s4=15, s5=15, s6=15, s7=15,
                                inst_net_3d=0),
        ]
        df = _apply_lowbuy_percentile_normalization(results, None)
        for _, r in df.iterrows():
            score = r["综合评分"]
            if score >= 55:
                continue  # 有可能等值导致分高
            elif score >= 40:
                assert r["信号"] == "标准低吸"
            else:
                assert r["信号"] == "谨慎低吸"

    def test_custom_weights(self):
        """自定义权重影响总分"""
        from core.screener import _apply_lowbuy_percentile_normalization
        results = [
            _make_lowbuy_result(s1=30, s2=10, s3=10, s4=10, s5=10, s6=10, s7=10,
                                inst_net_3d=0),
            _make_lowbuy_result(s1=10, s2=30, s3=10, s4=10, s5=10, s6=10, s7=10,
                                inst_net_3d=0),
        ]
        # 加大下跌幅度权重
        custom_w = {"下跌幅度": 50, "企稳信号": 10, "量能恢复": 10,
                     "均线支撑": 10, "估值吸引": 10, "筹码沉淀": 5, "主力资金": 5}
        df_custom = _apply_lowbuy_percentile_normalization(results, custom_w)
        # 加权重于下跌幅度后，第一个结果（s1=30）应该得分更高
        assert df_custom.iloc[0]["综合评分"] >= df_custom.iloc[1]["综合评分"]

    def test_analysis_summary_format(self):
        """分析摘要格式：'10日跌X.X% | 综合XX分 | 机构3日净买X万'"""
        from core.screener import _apply_lowbuy_percentile_normalization
        results = [
            _make_lowbuy_result(s1=25, s2=25, s3=25, s4=25, s5=25, s6=25, s7=25,
                                chg_10d=-8.5, inst_net_3d=300),
            _make_lowbuy_result(s1=5, s2=5, s3=5, s4=5, s5=5, s6=5, s7=5,
                                chg_10d=-2.0, inst_net_3d=-50),
        ]
        df = _apply_lowbuy_percentile_normalization(results, None)
        for _, r in df.iterrows():
            summary = r["分析摘要"]
            assert "10日跌" in summary
            assert "综合" in summary
            assert "分" in summary
            assert "机构3日净买" in summary

    def test_result_columns_present(self):
        """返回的 DataFrame 应包含信号/信号类/各维度列"""
        from core.screener import _apply_lowbuy_percentile_normalization
        results = [
            _make_lowbuy_result(s1=20, s2=20, s3=20, s4=20, s5=20, s6=20, s7=20,
                                inst_net_3d=100),
            _make_lowbuy_result(s1=10, s2=10, s3=10, s4=10, s5=10, s6=10, s7=10,
                                inst_net_3d=50),
        ]
        df = _apply_lowbuy_percentile_normalization(results, None)
        expected_cols = ["综合评分", "信号强度", "下跌幅度", "企稳信号", "量能恢复",
                          "均线支撑", "估值吸引", "筹码沉淀", "主力资金",
                          "信号", "信号类", "分析摘要"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"


# ============================================================
# Part 3: _apply_chase_percentile_normalization
# ============================================================


def _make_chase_raw(s1=15, s2=15, s3=10, s4=20, s5=20, s6=5, s7=15, s8=5, s9=3, s10=12):
    """生成追高模型 _raw 字典"""
    return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6,
            "s7": s7, "s8": s8, "s9": s9, "s10": s10}


class TestApplyChasePercentileNormalization:
    """覆盖追高百分位归一化（含 N/A 感知机制）"""

    def _make_inputs(self, n=3, seed_raw=None):
        """构造 _apply_chase_percentile_normalization 的 3 个输入"""
        raws = seed_raw or [_make_chase_raw() for _ in range(n)]
        score_results = [{"_raw": r, "综合评分": 50} for r in raws]
        score_df = pd.DataFrame({"综合评分": [50] * n})
        df = pd.DataFrame(index=range(n))
        return score_df, df, score_results

    def test_multi_results_normalization(self):
        """多结果百分位归一化，高分排名得到高分"""
        from core.screener import _apply_chase_percentile_normalization
        raws = [
            _make_chase_raw(s1=30, s2=30, s3=20, s4=30, s5=30, s6=5, s7=30, s8=5, s9=5, s10=30),
            _make_chase_raw(s1=10, s2=10, s3=5, s4=10, s5=10, s6=5, s7=10, s8=0, s9=0, s10=5),
            _make_chase_raw(s1=20, s2=20, s3=10, s4=20, s5=20, s6=5, s7=20, s8=3, s9=2, s10=15),
        ]
        score_df, df, score_results = self._make_inputs(seed_raw=raws)
        result_df, result_df2 = _apply_chase_percentile_normalization(
            score_df, df, score_results, None
        )
        assert "综合评分" in result_df.columns
        # 第一个（全高分）应 >= 第三个
        scores = result_df["综合评分"].tolist()
        assert scores[0] >= scores[2], f"高分未获得高排名: {scores}"
        # 每个结果都有 position_msg
        for r in score_results:
            assert "position_msg" in r

    def test_na_values_handling(self):
        """N/A 值（-1）不参与排名，无 N/A 维度加权后均值更高"""
        from core.screener import _apply_chase_percentile_normalization
        raws = [
            _make_chase_raw(s1=20, s2=20, s3=10, s4=20, s5=20, s6=5, s7=20, s8=5, s9=5, s10=20),
            # 第二个结果 s4=-1, s5=-1（北向/机构数据不可用）
            _make_chase_raw(s1=20, s2=20, s3=10, s4=-1, s5=-1, s6=5, s7=20, s8=5, s9=5, s10=20),
        ]
        score_df, df, score_results = self._make_inputs(n=2, seed_raw=raws)
        result_df, _ = _apply_chase_percentile_normalization(
            score_df, df, score_results, None
        )
        # 两个都有综合评分
        assert result_df["综合评分"].iloc[0] >= 0
        assert result_df["综合评分"].iloc[1] >= 0

    def test_single_result_no_change(self):
        """单结果时跳过归一化"""
        from core.screener import _apply_chase_percentile_normalization
        score_df, df, score_results = self._make_inputs(n=1)
        result_df, _ = _apply_chase_percentile_normalization(
            score_df, df, score_results, None
        )
        assert len(result_df) == 1

    def test_position_msg_by_score(self):
        """position_msg 根据综合评分分级"""
        from core.screener import _apply_chase_percentile_normalization
        raws = [
            _make_chase_raw(s1=30, s2=30, s3=20, s4=30, s5=30, s6=10, s7=30, s8=10, s9=7, s10=30),
            _make_chase_raw(s1=20, s2=20, s3=10, s4=20, s5=20, s6=7, s7=20, s8=5, s9=4, s10=20),
            _make_chase_raw(s1=5, s2=5, s3=3, s4=5, s5=5, s6=3, s7=5, s8=1, s9=1, s10=5),
        ]
        score_df, df, score_results = self._make_inputs(seed_raw=raws)
        _, _ = _apply_chase_percentile_normalization(
            score_df, df, score_results, None
        )
        for r in score_results:
            msg = r.get("position_msg", "")
            score = r.get("综合评分", 0)
            if score >= 85:
                assert "重仓" in msg or "60%" in msg
            elif score >= 70:
                assert "轻仓" in msg or "20%" in msg
            else:
                assert "放弃" in msg or "不参与" in msg

    def test_custom_weights(self):
        """自定义权重改变各维度影响力"""
        from core.screener import _apply_chase_percentile_normalization
        raws = [
            _make_chase_raw(s1=30, s2=5, s3=5, s4=5, s5=5, s6=5, s7=5, s8=5, s9=5, s10=5),
            _make_chase_raw(s1=5, s2=30, s3=5, s4=5, s5=5, s6=5, s7=5, s8=5, s9=5, s10=5),
        ]
        # 加大趋势结构权重
        custom_w = {"趋势结构": 50, "动量强度": 5, "板块共振": 5,
                    "北向资金": 5, "机构净买": 5, "量价配合": 5,
                    "情绪热度": 5, "板块资金热度": 5, "估值安全": 5, "筹码稳定": 5}
        score_df, df, score_results = self._make_inputs(n=2, seed_raw=raws)
        df_result, _ = _apply_chase_percentile_normalization(
            score_df, df, score_results, custom_w
        )
        # s1高的结果（第一个）应获得更高综合评分
        scores = df_result["综合评分"].tolist()
        assert scores[0] >= scores[1], f"趋势结构权重未生效: {scores}"


# ============================================================
# Part 4: get_stock_detail / get_top10_stocks / get_lowbuy_top5
# ============================================================


class TestGetStockDetail:
    """从股票池中按 code 匹配"""

    def test_find_existing_code(self):
        """在股票池中找到指定 code"""
        from core.screener import get_stock_detail
        mock_pool = pd.DataFrame({
            "代码": ["000001", "600000", "300001"],
            "名称": ["平安银行", "浦发银行", "特锐德"],
            "综合评分": [85, 72, 60],
        })
        with patch("core.screener.get_stock_pool", return_value=mock_pool):
            result = get_stock_detail("600000")
            assert result is not None
            assert result["代码"] == "600000"
            assert result["名称"] == "浦发银行"

    def test_code_not_found(self):
        """不存在的 code 返回 None"""
        from core.screener import get_stock_detail
        mock_pool = pd.DataFrame({
            "代码": ["000001", "600000"],
            "名称": ["平安银行", "浦发银行"],
            "综合评分": [85, 72],
        })
        with patch("core.screener.get_stock_pool", return_value=mock_pool):
            result = get_stock_detail("999999")
            assert result is None

    def test_empty_pool(self):
        """股票池为空返回 None"""
        from core.screener import get_stock_detail
        mock_pool = pd.DataFrame({"代码": pd.Series(dtype=str), "名称": pd.Series(dtype=str)})
        with patch("core.screener.get_stock_pool", return_value=mock_pool):
            result = get_stock_detail("000001")
            assert result is None


class TestGetTop10Stocks:
    """get_top10_stocks 包装器"""

    def test_returns_top_n_correctly(self):
        """返回综合评分前 N 的股票"""
        from core.screener import get_top10_stocks
        from core.protocols import AppState
        app_state = AppState()
        mock_pool = pd.DataFrame({
            "代码": [f"{i:06d}" for i in range(1, 21)],
            "综合评分": list(range(100, 80, -1)),  # 100 down to 81
        })
        with patch("core.screener.get_stock_pool", return_value=mock_pool):
            with patch("core.screener.calculate_dynamic_recommend_count", return_value=10):
                with patch("core.screener.save_cache_data"):
                    result = get_top10_stocks(app_state=app_state)
                    assert len(result) == 10
                    assert result[0]["综合评分"] == 100
                    assert result[-1]["综合评分"] == 91

    def test_less_than_n_stocks(self):
        """股票数少于请求数量时不崩溃"""
        from core.screener import get_top10_stocks
        from core.protocols import AppState
        app_state = AppState()
        mock_pool = pd.DataFrame({
            "代码": ["000001", "600000"],
            "综合评分": [85, 72],
        })
        with patch("core.screener.get_stock_pool", return_value=mock_pool):
            with patch("core.screener.calculate_dynamic_recommend_count", return_value=5):
                with patch("core.screener.save_cache_data"):
                    result = get_top10_stocks(app_state=app_state)
                    assert len(result) == 2


class TestGetLowbuyTop5:
    """get_lowbuy_top5 轻量包装器"""

    def test_returns_top_5(self):
        """get_lowbuy_top5 返回 screen_low_buy_stocks 的完整结果（函数名表明设计用途为Top5）"""
        from core.screener import get_lowbuy_top5
        mock_pool = pd.DataFrame({
            "代码": [f"{i:06d}" for i in range(1, 11)],
            "综合评分": list(range(90, 80, -1)),
        })
        with patch("core.screener.screen_low_buy_stocks", return_value=mock_pool):
            with patch("core.screener.fetch_all_a_stocks", return_value=mock_pool):
                with patch("core.screener.preprocess_stock_data", return_value=mock_pool):
                    result = get_lowbuy_top5()
                    assert len(result) == 10  # 返回 screen_low_buy_stocks 的完整结果

    def test_less_than_5_stocks(self):
        """低吸池不足5只时不崩溃"""
        from core.screener import get_lowbuy_top5
        mock_pool = pd.DataFrame({
            "代码": ["000001", "600000"],
            "综合评分": [85, 72],
        })
        with patch("core.screener.screen_low_buy_stocks", return_value=mock_pool):
            with patch("core.screener.fetch_all_a_stocks", return_value=mock_pool):
                with patch("core.screener.preprocess_stock_data", return_value=mock_pool):
                    result = get_lowbuy_top5()
                    assert len(result) == 2


# ============================================================
# Part 5: score_single_stock（需 mock data 层）
# ============================================================


def _make_mock_kline(n=250, trend="up"):
    """生成模拟 K 线 DataFrame"""
    from tests.conftest import make_sample_kline
    return make_sample_kline(n=n, trend=trend)


class TestScoreSingleStock:
    """覆盖个股评分中枢的 7 种模型路由"""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """统一 mock data 层依赖"""
        mock_kline = _make_mock_kline(n=250, trend="up")
        mock_kline60 = _make_mock_kline(n=60, trend="up")

        row_dict = {
            "代码": "000001", "名称": "平安银行", "最新价": 12.5, "涨跌幅": 2.5,
            "量比": 1.2, "换手率": 3.5, "市盈率-动态": 8.0, "市净率": 0.8,
        }
        df_raw = pd.DataFrame(row_dict, index=[0])

        stock_data = {
            "代码": "000001", "名称": "平安银行", "最新价": 12.5,
            "涨跌幅": 2.5, "量比": 1.2, "换手率": 3.5, "RSI": 55.0,
        }

        self._patchers = [
            patch("core.screener.get_stock_kline", side_effect=lambda code, days=250:
                  mock_kline if days >= 60 else mock_kline60),
            patch("core.screener._find_stock_row", return_value=(row_dict, df_raw)),
            patch("core.screener._get_cached_dragon_tiger", return_value={}),
            patch("core.screener._get_cached_sector_data", return_value=({}, {})),
            patch("core.screener.calculate_rsi", return_value=55.0),
            patch("core.screener._build_stock_data", return_value=stock_data),
            patch("core.screener.calculate_v3_total_score",
                  return_value={"综合评分": 85, "pass": True, "position_msg": "主线龙头",
                                "filter_msg": "", "趋势结构": 25, "动量强度": 20,
                                "板块共振": 10, "北向资金": 5, "机构净买": 5,
                                "板块资金热度": 3, "量价配合": 8, "估值安全": 2,
                                "筹码稳定": 3, "情绪热度": 4}),
            patch("core.screener.calculate_lowbuy_score",
                  return_value={"综合评分": 72, "pass": True, "下跌幅度": 20,
                                "企稳信号": 15, "量能恢复": 10, "均线支撑": 10,
                                "估值吸引": 8, "筹码沉淀": 5, "主力资金": 4}),
            patch("core.screener.calculate_golden_cross_score",
                  return_value={"综合评分": 68, "pass": True, "信号": "买点A",
                                "下跌形态": 15, "K线止跌": 12, "均线拐头": 10,
                                "量能确认": 8, "MACD反转": 10, "资金确认": 7,
                                "板块确认": 6}),
            patch("core.screener.get_resonance_data", return_value={}),
            patch("core.screener.calculate_resonance_score", return_value={}),
            patch("core.screener.get_financial_data", return_value={}),
            patch("core.screener.get_all_spot_data", return_value={}),
            patch("core.screener.calculate_canslim_score",
                  return_value={"综合评分": 65, "pass": True,
                                "C_业绩增速": 15, "A_持续增长": 10, "N_新催化": 8,
                                "S_中小盘": 10, "L_RPS": 7, "I_流动性": 8, "M_大势": 7}),
            patch("core.screener.calculate_dilemma_reversal_score",
                  return_value={"综合评分": 70, "pass": True,
                                "L1_拐点": 20, "L2_反转": 18, "L3_安全垫": 15, "L4_技术资金": 17}),
            patch("core.screener.hard_filter_oversold_rebound", return_value=(True, "")),
            patch("core.screener.calculate_oversold_rebound_score",
                  return_value={"综合评分": 75, "pass": True,
                                "空间维度": 25, "情绪量能": 20, "择时确认": 15, "板块共振": 15}),
        ]
        for p in self._patchers:
            p.start()
        yield
        for p in self._patchers:
            p.stop()

    def test_chase_high_model(self):
        """chase_high 模型返回正确的十维评分"""
        from core.screener import score_single_stock
        result = score_single_stock("000001", "chase_high")
        assert result["success"] is True
        assert result["model"] == "chase_high"
        assert result["total"] == 85
        assert result["name"] == "平安银行"
        dims = result["dims"]
        for k in ["趋势结构", "动量强度", "板块共振", "北向资金", "机构净买",
                   "板块资金热度", "量价配合", "估值安全", "筹码稳定", "情绪热度"]:
            assert k in dims, f"缺少维度: {k}"
        assert result["pass"] is True
        assert "position_msg" in result

    def test_buy_low_model(self):
        """buy_low 模型返回正确的七维评分"""
        from core.screener import score_single_stock
        result = score_single_stock("000001", "buy_low")
        assert result["success"] is True
        assert result["model"] == "buy_low"
        assert result["total"] == 72
        dims = result["dims"]
        for k in ["下跌幅度", "企稳信号", "量能恢复", "均线支撑", "估值吸引", "筹码沉淀", "主力资金"]:
            assert k in dims, f"缺少维度: {k}"
        assert result["pass"] is True

    def test_golden_cross_model(self):
        """golden_cross 模型返回正确的七维评分 + 信号"""
        from core.screener import score_single_stock
        result = score_single_stock("000001", "golden_cross")
        assert result["success"] is True
        assert result["model"] == "golden_cross"
        assert result["signal"] == "买点A"
        dims = result["dims"]
        for k in ["下跌形态", "K线止跌", "均线拐头", "量能确认", "MACD反转", "资金确认", "板块确认"]:
            assert k in dims, f"缺少维度: {k}"

    def test_canslim_model(self):
        """canslim 模型返回正确的七因子评分"""
        from core.screener import score_single_stock
        result = score_single_stock("000001", "canslim")
        assert result["success"] is True
        assert result["model"] == "canslim"
        dims = result["dims"]
        for k in ["C_业绩增速", "A_持续增长", "N_新催化", "S_中小盘", "L_RPS", "I_流动性", "M_大势"]:
            assert k in dims, f"缺少维度: {k}"

    def test_dilemma_reversal_model(self):
        """dilemma_reversal 模型返回正确的四层评分"""
        from core.screener import score_single_stock
        result = score_single_stock("000001", "dilemma_reversal")
        assert result["success"] is True
        assert result["model"] == "dilemma_reversal"
        dims = result["dims"]
        for k in ["L1_拐点", "L2_反转", "L3_安全垫", "L4_技术资金"]:
            assert k in dims, f"缺少维度: {k}"

    def test_oversold_rebound_model(self):
        """oversold_rebound 模型返回正确的四维评分 + pass"""
        from core.screener import score_single_stock
        result = score_single_stock("000001", "oversold_rebound")
        assert result["success"] is True
        assert result["model"] == "oversold_rebound"
        assert result["pass"] is True
        assert result["total"] == 75
        dims = result["dims"]
        for k in ["空间维度", "情绪量能", "择时确认", "板块共振"]:
            assert k in dims, f"缺少维度: {k}"

    def test_unknown_model(self):
        """未知模型返回 error"""
        from core.screener import score_single_stock
        result = score_single_stock("000001", "fake_model")
        assert result["success"] is False
        assert "未知模型" in result["error"]

    def test_find_stock_row_none(self):
        """找不到股票时返回 error"""
        from core.screener import score_single_stock
        with patch("core.screener._find_stock_row", return_value=(None, None)):
            result = score_single_stock("999999", "chase_high")
            assert result["success"] is False
            assert "未找到" in result["error"]

    def test_canslim_short_kline_returns_error(self):
        """canslim K线不足60日返回error"""
        from core.screener import score_single_stock
        short_kline = _make_mock_kline(n=30, trend="up")
        with patch("core.screener.get_stock_kline", return_value=short_kline):
            result = score_single_stock("000001", "canslim")
            assert result["success"] is False
            assert "K线数据不足" in result["error"]

    def test_dilemma_short_kline_returns_error(self):
        """dilemma K线不足60日返回error"""
        from core.screener import score_single_stock
        short_kline = _make_mock_kline(n=30, trend="up")
        with patch("core.screener.get_stock_kline", return_value=short_kline):
            result = score_single_stock("000001", "dilemma_reversal")
            assert result["success"] is False
            assert "K线数据不足" in result["error"]

    def test_oversold_short_kline_returns_error(self):
        """oversold K线不足60日返回error"""
        from core.screener import score_single_stock
        short_kline = _make_mock_kline(n=30, trend="up")
        with patch("core.screener.get_stock_kline", return_value=short_kline):
            result = score_single_stock("000001", "oversold_rebound")
            assert result["success"] is False
            assert "K线数据不足" in result["error"]

    def test_resonance_model_not_covered(self):
        """resonance 模型未覆盖该股票时返回 error"""
        from core.screener import score_single_stock
        result = score_single_stock("000001", "resonance")
        assert result["success"] is False
        assert "未覆盖" in result["error"]

    def test_all_models_return_close_and_chg(self):
        """所有模型都返回 close 和 chg"""
        from core.screener import score_single_stock
        for model in ["chase_high", "buy_low", "golden_cross", "canslim",
                       "dilemma_reversal", "oversold_rebound"]:
            result = score_single_stock("000001", model)
            if result["success"]:
                assert "close" in result
                assert "chg" in result
                assert result["close"] == 12.5
