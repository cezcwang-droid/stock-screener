"""
Model renderer: golden_cross
Extracted from render_screener tab 3.
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


def render_tab_golden_cross(st, model_tabs, tab_idx=3):
    st.session_state.current_model = 'rebound_model'


    render_top_params_panel()


    st.markdown("""<div class="header-container"><div class="main-title">📌 金叉模型 · 抓强势回调股</div>

    <div class="sub-title">强势股回调反弹 — 五维评分（下跌形态·K线止跌·均线拐头·量能确认·MACD反转）</div></div>""", unsafe_allow_html=True)


    # 样本来源 — 独立于参数面板，选择即时生效

    sc1, sc2 = st.columns([1.8, 1])

    with sc1:

        cur_source = st.session_state.gc_params.get("sample_source", "全市场A股")

        idx = 0 if cur_source == "全市场A股" else (1 if cur_source == "热门板块" else 2)

        new_source = st.selectbox(

            "样本来源", DEFAULT_GC_SAMPLE_OPTIONS,

            index=idx,

            key="gc_sample_source_quick"

        )

        if "全市场" in new_source:

            st.session_state.gc_params["sample_source"] = "全市场A股"

        elif "量价" in new_source:

            st.session_state.gc_params["sample_source"] = "量价"

        else:

            st.session_state.gc_params["sample_source"] = "热门板块"


    col1, col2, col3 = st.columns([2, 1, 1])

    gc_sample = st.session_state.gc_params.get("sample_source", "全市场A股")

    if gc_sample == "热门板块":

        scan_label = "🔍 加速板块扫描"

    elif gc_sample == "量价":

        scan_label = "🔍 量价反转扫描"

    else:

        scan_label = "🔍 全市场扫描"

    with col1:

        do_scan = st.button(scan_label, type="primary", width='stretch', key="gc_scan")

    with col2:

        if st.button("🔄 刷新参数", width='stretch', key="gc_refresh_params"):

            st.session_state.gc_params = dict(DEFAULT_GC_PARAMS)

            st.session_state.gc_weights = dict(DEFAULT_GC_WEIGHTS)

            st.rerun()


    gc_cache = st.session_state.get('gc_results', None)


    if do_scan or (gc_cache is None and not st.session_state.get('gc_scanned', False)):

        if gc_sample == "热门板块":

            spinner_text = "正在扫描资金加速板块，筛选金叉标的…"

        elif gc_sample == "量价":

            spinner_text = "正在扫描量价反转板块，筛选金叉标的…"

        else:

            spinner_text = "正在扫描全市场，筛选金叉标的…"

        with st.spinner(spinner_text):

            try:

                quotes_df = fetch_all_a_stocks()

                if quotes_df is None or len(quotes_df) == 0:

                    st.error("获取行情数据失败")

                else:

                    quotes_df = quotes_df[~quotes_df['名称'].str.contains('ST|退市|N|C', na=False)]


                    # 🌟 热门板块过滤

                    if gc_sample == "热门板块":

                        hot_codes = get_hot_concept_stocks(6)

                        if hot_codes:

                            quotes_df['代码'] = quotes_df['代码'].astype(str).str.zfill(6)

                            quotes_df = quotes_df[quotes_df['代码'].isin(hot_codes)]

                            st.info(f"已锁定 {len(quotes_df)} 只热门概念板块成分股")

                        else:

                            st.warning("未能获取热门板块数据，回退为全市场扫描")

                    elif gc_sample == "量价":

                        vp_codes = get_volprice_sectors(6)

                        if vp_codes:

                            quotes_df['代码'] = quotes_df['代码'].astype(str).str.zfill(6)

                            quotes_df = quotes_df[quotes_df['代码'].isin(vp_codes)]

                            st.info(f"已锁定 {len(quotes_df)} 只量价反转板块成分股")

                        else:

                            st.warning("未能获取量价反转板块数据，回退为全市场扫描")


                    # 🔧 填充概念板块（数据源无板块字段，从 pytdx 板块成分股映射获取）

                    _, stock_sector_map = fetch_sector_board_v3()

                    quotes_df['代码'] = quotes_df['代码'].astype(str).str.zfill(6)

                    if stock_sector_map:

                        # 过滤掉非中文板块名（block_fg.dat/block_zs.dat 含内部编码如 W300881W3）

                        import re

                        valid_map = {k: v for k, v in stock_sector_map.items() if re.search(r'[\u4e00-\u9fff]', v)}

                        quotes_df['板块'] = quotes_df['代码'].map(valid_map).fillna('')

                    else:

                        quotes_df['板块'] = ''


                    codes_list = [(str(r['代码']).zfill(6), str(r['名称']), str(r.get('板块', '')))

                                  for _, r in quotes_df.iterrows()]

                    results = _run_golden_cross_scan(tuple(codes_list),

                                                     json.dumps(st.session_state.gc_params),

                                                     json.dumps(st.session_state.gc_weights))

                    st.session_state.gc_results = results

                    st.session_state.gc_scanned = True

                    st.rerun()

            except Exception as e:

                st.error(f"扫描出错: {e}")


    gc_results = st.session_state.get('gc_results', None)


    if gc_results is not None:

        if len(gc_results) == 0:

            st.warning("⚠️ 今日未找到符合条件的金叉标的。")

        else:

            st.markdown(f"### 📊 金叉模型 · Top {min(len(gc_results), 30)}")

            st.caption(f"共筛选出 {len(gc_results)} 只标的")


            # ---- Top N 精选（动态数量）----

            _gc_dyn_n = calculate_dynamic_recommend_count()

            gc_top_n = gc_results[:_gc_dyn_n]

            st.markdown(f"""

            <div class="top10-container">

                <div class="top10-header">

                    <div class="top10-title">📌 金叉精选 Top {_gc_dyn_n} <span class="top10-badge">超跌反弹信号</span></div>

                </div>

            </div>""", unsafe_allow_html=True)

            btn1, btn2, _ = st.columns([1, 1, 4])

            with btn1:

                if st.button("⭐ 一键加入自选", width='stretch', type="primary", key="gc_add_all"):

                    for s in gc_top_n:

                        if s["代码"] not in st.session_state.watchlist:

                            st.session_state.watchlist.append(s["代码"])

                    save_watchlist(st.session_state.watchlist)

                    st.success(f"已将Top {_gc_dyn_n}全部加入自选！"); st.rerun()

            with btn2:

                gc_top_df = pd.DataFrame(gc_top_n)

                gc_top_df['代码'] = gc_top_df['代码'].astype(str).str.zfill(6)

                gc_top_df = gc_top_df.rename(columns={'板块': '概念板块'})

                export_cols = ['代码', '名称', '概念板块',

                               '下跌形态', 'K线止跌', '均线拐头', '量能确认', 'MACD反转', '资金确认',

                               '共振评分', '共振评价', '金叉评分', '信号', '建议']

                gc_top_df = gc_top_df[[c for c in export_cols if c in gc_top_df.columns]]

                xlsx_data = _export_df_to_xlsx(gc_top_df)

                st.download_button(f"📥 导出Top{_gc_dyn_n}", xlsx_data, f"top{_gc_dyn_n}_golden_cross_{datetime.now().strftime('%Y%m%d')}.xlsx",

                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                                   width='stretch', key="dl_gc_t10")


            st.markdown("")

            for i in range(0, len(gc_top_n), 5):

                row_stocks = gc_top_n[i:i+5]

                cols = st.columns(5)

                for j, stock in enumerate(row_stocks):

                    with cols[j]:

                        rank = i + j + 1

                        rank_color = "#FFD700" if rank == 1 else "#C0C0C0" if rank == 2 else "#CD7F32" if rank == 3 else "#CCC"

                        signal = stock.get('信号', '-')

                        st.markdown(f"""

                        <div class="top10-card">

                            <div class="top10-rank" style="color:{rank_color};">{rank}</div>

                            <div style="padding-right:30px;">

                                <div style="font-weight:700;color:#333;font-size:15px;margin-bottom:2px;">{stock['名称']}</div>

                                <div style="font-size:11px;color:#888;margin-bottom:8px;">{str(stock.get('代码','')).zfill(6)} · {stock.get('板块', '其他')[:8]}</div>

                                <div style="display:flex;justify-content:space-between;align-items:center;">

                                    <span style="font-size:20px;font-weight:800;color:#C4842D;">{stock['金叉评分']}</span>

                                </div>

                                <div style="margin-top:6px;"><span class="metric-badge badge-strong">{signal}</span></div>

                            </div>

                        </div>""", unsafe_allow_html=True)

                        code = stock["代码"]; in_wl = code in st.session_state.watchlist

                        if st.button("⭐" if in_wl else "+自选", key=f"gc_t10_{code}",

                            width='stretch', type="primary" if in_wl else "secondary"):

                            if in_wl: st.session_state.watchlist.remove(code)

                            else: st.session_state.watchlist.append(code)

                            save_watchlist(st.session_state.watchlist); st.rerun()


            st.markdown("---")


            # 共振交叉评分

            res_scores, res_styles = _get_resonance_cross_ref()


            fdf = pd.DataFrame(gc_results[:30])

            fdf.index = range(1, len(fdf) + 1)

            fdf['代码'] = fdf['代码'].astype(str).str.zfill(6)

            fdf = fdf.rename(columns={'板块': '概念板块'})


            if res_scores:

                fdf['共振评分'] = fdf['代码'].apply(lambda x: res_scores.get(x, None))

            if res_styles:

                fdf['共振评价'] = fdf['代码'].apply(lambda x: res_styles.get(x, '-'))


            display_cols = ['代码', '名称', '概念板块', '信号', '建议', '金叉评分',

                            '下跌形态', 'K线止跌', '均线拐头', '量能确认', 'MACD反转', '资金确认', '板块确认',

                            '共振评分', '共振评价']

            df_display = fdf[[c for c in display_cols if c in fdf.columns]]


            def color_score(val):

                if isinstance(val, (int, float)):

                    if val >= 80: return 'background-color:#C8E6C9;font-weight:bold'

                    if val >= 70: return 'background-color:#E8F5E9'

                    if val >= 60: return 'background-color:#FFF9C4'

                    return ''

                return ''


            styled = df_display.style.map(color_score, subset=['金叉评分'])

            st.dataframe(styled, width='stretch',

                         column_config={

                             '金叉评分': st.column_config.NumberColumn(format='%.0f'),

                         })


            st.markdown("---")

            st.caption("操作区 — 评分详情 | 加/取消自选")

            gc_for_buttons = []

            for r in gc_results[:30]:

                gc_for_buttons.append({'代码': r.get('代码', ''), '名称': r.get('名称', '')})

            render_stock_buttons(gc_for_buttons, prefix="gc")


            # 导出

            export_df = pd.DataFrame(gc_results)

            export_df['代码'] = export_df['代码'].astype(str).str.zfill(6)

            export_df = export_df.rename(columns={'板块': '概念板块'})

            export_cols = ['代码', '名称', '概念板块', '信号', '建议', '金叉评分',

                           '下跌形态', 'K线止跌', '均线拐头', '量能确认', 'MACD反转', '资金确认', '板块确认',

                           '共振评分', '共振评价']

            export_df = export_df[[c for c in export_cols if c in export_df.columns]]

            csv_data = export_df.to_csv(index=False).encode('utf-8-sig')

            st.download_button("📥 导出 Top30 CSV", csv_data, f"rebound_model_top30_{datetime.now().strftime('%Y%m%d')}.csv",

                               "text/csv", key="dl_gc_top30")

    elif not do_scan:

        st.info("💡 点击「全市场扫描」启动金叉模型选股（基于五维评分 + 硬过滤条件）。")

