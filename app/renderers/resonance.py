"""
Model renderer: resonance
Extracted from render_screener tab 0.
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
from data.dde import _load_dde_data, _get_dde_or_fallback, _get_resonance_cross_ref, get_resonance_cache, save_resonance_cache
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


def render_tab_resonance(st, model_tabs, tab_idx=0):
    st.session_state.current_model = 'resonance'


    # 顶部参数面板

    render_top_params_panel()


    st.markdown(f"""<div class="header-container"><div class="main-title">🎯 共振模型 · 多维共振参考</div>

    <div class="sub-title">资金流向 + DDE决策 + K线结构 + 板块热度 — 四维共振</div></div>""", unsafe_allow_html=True)


    # 刷新 / 应用按钮

    bc1, bc2, bc3 = st.columns([1, 1, 4])

    with bc1:

        refresh_clicked = st.button("🔄 刷新数据", width='stretch', key="resonance_refresh")

    with bc2:

        apply_clicked = st.button("📊 查看结果", width='stretch', key="resonance_apply")


    # 数据状态

    today_str = datetime.now().strftime('%Y%m%d')


    # 自动触发：无缓存或点击刷新时执行扫描

    if 'resonance_auto_scanned' not in st.session_state:

        st.session_state.resonance_auto_scanned = False


    cached_data = get_resonance_cache() if not refresh_clicked else None


    # 判断是否需要扫描（无缓存 或 点击刷新）

    need_scan = refresh_clicked or (cached_data is None and not st.session_state.resonance_auto_scanned)


    if cached_data and not refresh_clicked:

        cache_time = cached_data.get('cache_time', '')

        st.info(f"📌 显示今日 {cache_time} 的缓存结果，点击「刷新数据」获取最新")


    # 加载或刷新数据

    if need_scan:

        with st.spinner("正在计算共振模型（通达信K线资金流向 + DDE代理 + K线结构 + 板块热度）..."):

            # 获取行情数据（走统一缓存兜底链）

            try:

                quotes_df = fetch_all_a_stocks()

            except Exception as e:

                st.error(f"获取行情数据失败: {e}")

                quotes_df = None


            if quotes_df is not None and len(quotes_df) > 0:

                # 过滤 ST/退市

                quotes_df = quotes_df[~quotes_df['名称'].str.contains('ST|退市|N|C', na=False)]


                resonance_data = get_resonance_data(quotes_df)

                if resonance_data:

                    # 补充K线和板块评分

                    scores = calculate_resonance_score(resonance_data, quotes_df)


                    # 按总分排序取 Top30

                    sorted_codes = sorted(scores.keys(), key=lambda x: scores[x]['total'], reverse=True)[:30]


                    # 获取这些股票的名称等信息

                    # 🔧 预取概念板块映射

                    _, res_sec_map = fetch_sector_board_v3()

                    if res_sec_map:

                        import re

                        res_sec_map = {k: v for k, v in res_sec_map.items() if re.search(r'[\u4e00-\u9fff]', v)}


                    result_list = []

                    for code in sorted_codes:

                        sc = scores[code]

                        name = ''

                        sector = ''

                        close = ''

                        row = quotes_df[quotes_df['代码'] == code]

                        if len(row) > 0:

                            name = str(row.iloc[0].get('名称', ''))

                        sector = res_sec_map.get(code, '') if res_sec_map else _get_sector(code)

                        style_tag = _classify_resonance_style(code, resonance_data, quotes_df)

                        result_list.append({

                            '代码': code, '名称': name, '板块': sector,

                            '共振评分': sc['total'],

                            '资金流向': sc['money_flow'],

                            'DDE决策': sc['dde_proxy'],

                            'K线结构': sc['kline_structure'],

                            '板块热度': sc['sector_heat'],

                            '当前走势': style_tag,

                        })


                    cache_data = {

                        'cache_time': datetime.now().strftime('%H:%M'),

                        'results': result_list,

                        'raw_scores': scores,

                    }

                    save_resonance_cache(cache_data)

                    st.session_state['resonance_results'] = result_list

                    st.session_state.resonance_auto_scanned = True

                    st.rerun()

                else:

                    st.warning("⚠️ 共振模型数据源均不可用：DDE Excel 不存在/解析失败，新浪财经资金流向备选链路也未获取到有效数据。可稍后重试或检查网络连接。")

                    st.session_state.resonance_auto_scanned = True

            else:

                st.warning("⚠️ 未获取到行情数据，已跳过共振模型扫描。")

                st.session_state.resonance_auto_scanned = True


    # 显示结果

    resonance_results = st.session_state.get('resonance_results', None)

    if resonance_results is None and cached_data:

        resonance_results = cached_data.get('results', [])

        st.session_state['resonance_results'] = resonance_results


    # 兼容旧缓存：补默认风格标签 / 旧键名迁移

    if resonance_results:

        for s in resonance_results:

            s.setdefault('当前走势', '蓄势待发')

            if '综合评分' in s and '共振评分' not in s:

                s['共振评分'] = s.pop('综合评分')


    if resonance_results and len(resonance_results) > 0:

        _res_dynamic_n = calculate_dynamic_recommend_count()

        top_n = resonance_results[:_res_dynamic_n]


        st.markdown(f"## 🏆 今日精选 Top {_res_dynamic_n}")


        # 导出按钮

        _, btn_export, _ = st.columns([4, 1, 4])

        with btn_export:

            top_n_df = pd.DataFrame(top_n)

            top_n_df['代码'] = top_n_df['代码'].astype(str).str.zfill(6)

            xlsx_data = _export_df_to_xlsx(top_n_df)

            st.download_button(f"📥 导出Top{_res_dynamic_n}", xlsx_data, f"top{_res_dynamic_n}_resonance_{datetime.now().strftime('%Y%m%d')}.xlsx",

                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                               width='stretch', key="dl_resonance_top")


        # Top N 卡片

        cols = st.columns(5)

        rank_colors = {0: '#FFD700', 1: '#C0C0C0', 2: '#CD7F32'}

        style_badge = {'盘中走强': ('#E53935', '#FFEBEE'), '超跌待反转': ('#1E88E5', '#E3F2FD'), '蓄势待发': ('#757575', '#F5F5F5')}

        for i, stock in enumerate(top_n):

            with cols[i % 5]:

                rank_color = rank_colors.get(i, '#888')

                tag = stock.get('当前走势', '蓄势待发')

                tc, bg = style_badge.get(tag, ('#757575', '#F5F5F5'))

                st.markdown(f"""

                <div style="background:white;border-radius:12px;padding:12px;margin:4px;

                border:1px solid #E0E0E0;position:relative;min-height:130px;">

                <div style="position:absolute;top:4px;right:8px;font-size:22px;font-weight:800;color:{rank_color};">#{i+1}</div>

                <div style="font-size:15px;font-weight:700;color:#333;margin-top:4px;">{stock['名称']}

                <span style="display:inline-block;font-size:10px;color:{tc};background:{bg};padding:1px 6px;border-radius:8px;margin-left:4px;vertical-align:middle;">{tag}</span></div>

                <div style="font-size:11px;color:#999;">{stock['代码']}</div>

                <div style="font-size:22px;font-weight:700;color:#C4842D;margin:6px 0;">{stock['共振评分']}<span style="font-size:12px;color:#999;">分</span></div>

                <div style="font-size:10px;color:#666;">

                资金{stock['资金流向']} | DDE{stock['DDE决策']} | K线{stock['K线结构']} | 板块{stock['板块热度']}

                </div>

                <div style="font-size:10px;color:#999;">{stock['板块']}</div>

                </div>

                """, unsafe_allow_html=True)


        # Top 30 详细表格

        st.markdown("---")


        # 当前走势分布摘要

        tags = [s.get('当前走势', '蓄势待发') for s in resonance_results[:30]]

        qs = sum(1 for t in tags if t == '盘中走强')

        dd = sum(1 for t in tags if t == '超跌待反转')

        zx = sum(1 for t in tags if t == '蓄势待发')

        st.caption(f"当前走势分布 — 🔴 盘中走强 {qs} 只 | 🔵 超跌待反转 {dd} 只 | ⚪ 蓄势待发 {zx} 只")


        st.markdown("### 📋 共振模型 Top 30 详细数据")


        df_display = pd.DataFrame(resonance_results[:30])

        df_display = df_display.rename(columns={

            '代码': '代码', '名称': '名称', '板块': '概念板块', '当前走势': '当前走势',

            '共振评分': '共振评分', '资金流向': '资金流向(30)',

            'DDE决策': 'DDE决策(20)', 'K线结构': 'K线结构(25)',

            '板块热度': '板块热度(25)'

        })


        # 带颜色渲染的表格

        def color_score(val):

            if isinstance(val, (int, float)):

                if val >= 80: return 'background-color:#C8E6C9;font-weight:bold'

                if val >= 70: return 'background-color:#E8F5E9'

                if val >= 60: return 'background-color:#FFF9C4'

                return ''

            return ''


        def color_style(val):

            if val == '盘中走强': return 'background-color:#FFEBEE;color:#C62828;font-weight:bold'

            if val == '超跌待反转': return 'background-color:#E3F2FD;color:#1565C0;font-weight:bold'

            return 'background-color:#F5F5F5;color:#757575'


        styled_df = df_display.style.map(color_score, subset=['共振评分']).map(color_style, subset=['当前走势'])

        st.dataframe(styled_df, width='stretch', hide_index=True,

                    column_config={

                        '共振评分': st.column_config.NumberColumn(format='%.1f'),

                        '资金流向(30)': st.column_config.NumberColumn(format='%.1f'),

                        'DDE决策(20)': st.column_config.NumberColumn(format='%.1f'),

                        'K线结构(25)': st.column_config.NumberColumn(format='%.1f'),

                        '板块热度(25)': st.column_config.NumberColumn(format='%.1f'),

                    })


        # 导出

        export_df = pd.DataFrame(resonance_results)

        export_df['代码'] = export_df['代码'].astype(str).str.zfill(6)

        csv_data = export_df.to_csv(index=False).encode('utf-8-sig')

        st.download_button("📥 导出 Top30 CSV", csv_data, f"resonance_top30_{today_str}.csv",

                         "text/csv", key="dl_resonance_top30")


        st.markdown("---")

        st.caption("操作区 — 评分详情 | 加/取消自选")

        render_stock_buttons(resonance_results[:30], prefix="rs")

    elif not need_scan:

        st.info("💡 点击「刷新数据」获取今日共振模型选股结果（基于东方财富DDE数据计算）。")

