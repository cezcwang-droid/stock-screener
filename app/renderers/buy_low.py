"""
Model renderer: buy_low
Extracted from render_screener tab 2.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, json, time, re, struct, glob, pickle, io, logging, traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import json
from datetime import datetime

from core.config import DEFAULT_GC_PARAMS, DEFAULT_GC_WEIGHTS, DEFAULT_GC_SAMPLE_OPTIONS
from core.config import DEFAULT_LOWBUY_WEIGHTS, WEIGHT_CONFIG, LOWBUY_WEIGHT_CONFIG, CACHE_DATA_JSON, AKSHARE_AVAILABLE
from core.screener import get_stock_pool, get_top10_stocks, get_lowbuy_top5, score_single_stock, _find_stock_row, get_stock_detail, screen_low_buy_stocks, get_stock_pool_chase_high, batch_calculate_rsi, _get_sector
from core.scoring.chase_high import calculate_v3_total_score, calculate_five_dimensions_score, _build_stock_data, _classify_signal
from core.scoring.lowbuy import calculate_lowbuy_score, _check_low_buy_conditions, _get_lowbuy_params
from core.scoring.golden_cross import calculate_golden_cross_score, _run_golden_cross_scan, get_hot_concept_stocks, get_volprice_sectors, _get_gc_params
from core.scoring.canslim import calculate_canslim_score, compute_rps
from core.scoring.dilemma import calculate_dilemma_reversal_score
from core.scoring.oversold import calculate_oversold_rebound_score
from core.filters import hard_filter_oversold_rebound
from core.scoring.resonance import get_dde_confirmation_scores, calculate_resonance_score, _classify_resonance_style, get_resonance_data
from core.filters import hard_filter_v3, _gc_hard_filter
from data.market import fetch_all_a_stocks, preprocess_stock_data, get_stock_kline, get_kline_with_today, get_all_spot_data, _supplement_tencent_quotes, read_tdx_day_file, calculate_dynamic_recommend_count
from data.dde import _load_dde_data, _get_dde_or_fallback, _get_resonance_cross_ref, _get_resonance_cross_ref
from data.sector import _get_cached_dragon_tiger, _get_cached_sector_data, _get_sector_fund_flow, fetch_dragon_tiger_v3, fetch_sector_board_v3, _classify_stock_sector
from data.financial import get_financial_data
from data.tdx_provider import tdx_available, fetch_all_quotes_tdx, fetch_kline_tdx, get_today_quote_single, fetch_sector_data_tdx, resolve_market
from backtest.runner import run_real_backtest_cached, generate_backtest_data
from backtest.metrics import calc_backtest_metrics
from app.charts import create_candlestick_chart, create_radar_chart, create_backtest_equity_chart, create_monthly_heatmap, create_score_bar
from app.components import (
    render_filter_bar, render_top_params_panel, render_stats_chase,
    render_stats_lowbuy, render_stock_buttons, render_table_chase,
    render_table_lowbuy, _render_lowbuy_cards, _render_filter_dashboard,
    _diag_stock, _export_df_to_xlsx,
)
import backtest_engine


def render_tab_buy_low(st, model_tabs, tab_idx=2):
    st.session_state.current_model = 'buy_low'


    # 顶部参数面板

    render_top_params_panel()


    # 样本来源选择

    sc1, sc2 = st.columns([1.8, 1])

    with sc1:

        cur_source = st.session_state.get('lowbuy_sample_source', '全市场A股')

        idx = 0 if cur_source == '全市场A股' else (1 if cur_source == '热门板块' else 2)

        new_source = st.selectbox(

            "样本来源", DEFAULT_GC_SAMPLE_OPTIONS,

            index=idx,

            key="lowbuy_sample_source_select"

        )

        if "全市场" in new_source:

            new_val = "全市场A股"

        elif "量价" in new_source:

            new_val = "量价"

        else:

            new_val = "热门板块"

        if new_val != cur_source:

            st.session_state.lowbuy_cache = None

            st.session_state.lowbuy_sample_source = new_val

            st.session_state.lowbuy_auto_scanned = False

            st.rerun()

        else:

            st.session_state.lowbuy_sample_source = new_val


    _lp = _get_lowbuy_params()

    st.markdown(f"""<div class="header-container" style="border-color:#C8E6C9;background:linear-gradient(135deg, #F0FFF4 0%, #FFFFFF 100%);">

    <div class="main-title" style="color:#2E7D32;">📉 低吸模型 · 抓超跌价值股</div>

    <div class="sub-title">七维评分硬过滤 + K线底部反转信号 · 10日跌{_lp['decline_20d_low']}%~-{-_lp['decline_20d_high']}% · 独立评分 · 精选{_lp['max_results']}只</div>

    </div>""", unsafe_allow_html=True)


    # 自动触发：首次加载时缓存为空则自动扫描

    lb_sample = st.session_state.get('lowbuy_sample_source', '全市场A股')

    lb_cache = st.session_state.get('lowbuy_cache', None)

    if lb_cache is None and not st.session_state.get('lowbuy_auto_scanned', False):

        render_top_low_buy(sample_source=lb_sample)

        st.session_state.lowbuy_auto_scanned = True

        st.rerun()


    render_top_low_buy(sample_source=st.session_state.get('lowbuy_sample_source', '全市场A股'))


    st.markdown("")

    bc1, bc2, bc3 = st.columns([1, 1, 4])

    with bc1:

        lb_source = st.session_state.get('lowbuy_sample_source', '全市场A股')

        if lb_source == "热门板块":

            lb_label = "🔄 加速扫描"

        elif lb_source == "量价":

            lb_label = "🔄 量价扫描"

        else:

            lb_label = "🔄 全市场扫描"

        if st.button(lb_label, width='stretch', key="lowbuy_refresh_data"):

            st.cache_data.clear()

            if os.path.exists(CACHE_DATA_JSON):

                try: os.remove(CACHE_DATA_JSON)

                except: pass

            st.session_state.top10_cache = None

            st.session_state.top10_cache_key = None

            st.session_state.lowbuy_cache = None

            st.session_state.lowbuy_auto_scanned = False

            st.session_state.cache_loaded = False

            st.rerun()


    st.markdown("")

    lb_df = pd.DataFrame(st.session_state.get('lowbuy_cache', []))

    if len(lb_df) > 0:

        render_stats_lowbuy(lb_df)

        st.markdown("")

        st.markdown("---")

        title_col, btn_col = st.columns([5, 1])

        with title_col:

            st.markdown("### 📊 单项评分排行 · 显示前30只")

        with btn_col:

            st.markdown("")

            xlsx_data = _export_df_to_xlsx(lb_df.head(30))

            st.download_button("📥 导出Top30", xlsx_data, f"top30_lowbuy_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width='stretch', key="dl_lb_top30")

        dde_scores = get_dde_confirmation_scores()

        res_scores, res_styles = _get_resonance_cross_ref()

        render_table_lowbuy(lb_df.head(30), dde_scores=dde_scores,

                            resonance_scores=res_scores, resonance_styles=res_styles)

    elif st.session_state.get('lowbuy_cache') is not None:

        st.warning("⚠️ 今日未找到符合低吸条件的股票，请尝试放宽参数后点击「刷新数据」重试。")

        st.stop()

