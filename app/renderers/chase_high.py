"""
Model renderer: chase_high
Extracted from render_screener tab 1.
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


def render_tab_chase_high(st, model_tabs, tab_idx=1):
    st.session_state.current_model = 'chase_high'


    # 顶部参数面板

    render_top_params_panel()


    st.markdown(f"""<div class="header-container"><div class="main-title">🚀 追高模型 · 抓强势龙头股</div>

    <div class="sub-title">八维强势评分 · 全市场量化筛选</div></div>""", unsafe_allow_html=True)


    # 显示缓存状态

    if st.session_state.get('data_status') == 'cached' and st.session_state.get('last_update_time'):

        cache_time = st.session_state.last_update_time

        st.info(f"📌 显示今日 {cache_time.strftime('%H:%M')} 的缓存结果，点击「刷新数据」获取最新")


    # 样本来源选择（紧跟header，与金叉模型一致）

    cr1, cr2 = st.columns([1.8, 1])

    with cr1:

        cur_source = st.session_state.get('chase_sample_source', '全市场A股')

        idx = 0 if cur_source == '全市场A股' else (1 if cur_source == '热门板块' else 2)

        new_source = st.selectbox(

            "样本来源", DEFAULT_GC_SAMPLE_OPTIONS,

            index=idx,

            key="chase_sample_source_select"

        )

        if "全市场" in new_source:

            new_val = "全市场A股"

        elif "量价" in new_source:

            new_val = "量价"

        else:

            new_val = "热门板块"

        if new_val != cur_source:

            st.cache_data.clear()

            st.session_state.top10_cache = None

            st.session_state.top10_cache_key = None

            st.session_state.lowbuy_cache = None

            st.session_state.chase_results = None

            st.session_state.cache_loaded = False

            st.session_state.chase_sample_source = new_val

            st.rerun()

        else:

            st.session_state.chase_sample_source = new_val


    render_top_chase_high()


    st.markdown("")

    chase_sample = st.session_state.get('chase_sample_source', '全市场A股')

    col1, col2 = st.columns([2, 1])

    if chase_sample == "热门板块":

        scan_label = "🔍 加速板块扫描"

    elif chase_sample == "量价":

        scan_label = "🔍 量价扫描"

    else:

        scan_label = "🔍 全市场扫描"

    with col1:

        do_scan = st.button(scan_label, type="primary", width='stretch', key="chase_scan")

    with col2:

        if st.button("🔄 清空缓存", width='stretch', key="chase_clear"):

            st.session_state.chase_results = None

            st.rerun()


    chase_results = st.session_state.get('chase_results', None)


    if do_scan or chase_results is None:

        chase_spin = st.session_state.get('chase_sample_source', '全市场A股')

        if chase_spin == "热门板块":

            spinner_text = "正在扫描资金加速板块，计算追高评分…"

        elif chase_spin == "量价":

            spinner_text = "正在扫描量价反转板块，计算追高评分…"

        else:

            spinner_text = "正在扫描全市场，计算追高评分…"

        with st.spinner(spinner_text):

            st.cache_data.clear()

            if os.path.exists(CACHE_DATA_JSON):

                try: os.remove(CACHE_DATA_JSON)

                except: pass

            st.session_state.top10_cache = None

            st.session_state.top10_cache_key = None

            st.session_state.lowbuy_cache = None

            st.session_state.cache_loaded = False

            df = get_stock_pool()

            st.session_state.chase_results = df

            st.rerun()


    st.markdown("")

    df = chase_results if chase_results is not None else pd.DataFrame()

    if len(df) == 0 or '板块' not in df.columns or '信号' not in df.columns:

        st.warning("⚠️ 数据加载中或接口暂时不可用，请稍后点击扫描按钮重试。")

        st.stop()

    render_stats_chase(df)

    st.markdown("")


    st.markdown("---")

    title_col, btn_col = st.columns([5, 1])

    with title_col:

        st.markdown("### 📊 单项评分排行 · 显示前30只")


    # 三个选择栏排成一行：板块、信号、排序方式

    sectors = ["全部"] + sorted(df["板块"].unique().tolist())

    sr_options = [("全部", None), ("强势买入", ("信号", ["强势买入"])),

        ("逢低吸纳", ("信号", ["逢低吸纳"])), ("观望等待", ("信号", ["观望等待"])),

        ("建议回避", ("信号", ["建议回避"]))]

    fc1, fc2, fc3 = st.columns([1, 1, 1])

    with fc1:

        sec = st.selectbox("板块", sectors, label_visibility="collapsed", key="sf_sec")

    with fc2:

        slabs = [o[0] for o in sr_options]

        sel_idx = slabs.index(st.selectbox("信号筛选", slabs, label_visibility="collapsed", key="sf_sig"))

        sig_filter = sr_options[sel_idx][1]

    with fc3:

        sb = st.selectbox("排序方式", ["综合评分↓", "趋势结构↓", "动量强度↓", "板块共振↓", "北向资金↓", "量价配合↓", "5日涨幅↓"],

            label_visibility="collapsed", key="ch_sort")


    filt = df.copy()

    if sec != "全部": filt = filt[filt["板块"] == sec]

    if sig_filter is not None and sig_filter[0] == "信号": filt = filt[filt["信号"].isin(sig_filter[1])]


    if sb == "5日涨幅↓":

        filt['_chg_n'] = filt["5日涨幅"].str.replace('%','').str.replace('+','').astype(float)

        filt = filt.sort_values('_chg_n', ascending=False)

    else:

        sort_map = {"综合评分↓": ("综合评分", False), "趋势结构↓": ("趋势结构", False),

            "动量强度↓": ("动量强度", False), "板块共振↓": ("板块共振", False),

            "北向资金↓": ("北向资金", False), "量价配合↓": ("量价配合", False)}

        col_name, asc = sort_map.get(sb, ("综合评分", False))

        filt = filt.sort_values(col_name, ascending=asc)


    top30 = filt.head(30).reset_index(drop=True)

    with btn_col:

        st.markdown("")  # align with title

        xlsx_data = _export_df_to_xlsx(top30)

        st.download_button("📥 导出Top30", xlsx_data, f"top30_chase_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width='stretch', key="dl_ch_top30")

    dde_scores = get_dde_confirmation_scores()

    res_scores, res_styles = _get_resonance_cross_ref()

    render_table_chase(top30, dde_scores=dde_scores,

                       resonance_scores=res_scores, resonance_styles=res_styles)

