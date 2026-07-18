"""
Model renderer: canslim
Extracted from render_screener tab 4.
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


def render_tab_canslim(st, model_tabs, tab_idx=4):
    st.session_state.current_model = 'canslim'


    render_top_params_panel()


    st.markdown("""<div class="header-container"><div class="main-title">📊 CAN SLIM模型（抓主升浪股）</div>

    <div class="sub-title">七因子简化版：C业绩 · A持续增长 · N新催化 · S中小盘 · L_RPS · I流动性 · M大势</div></div>""", unsafe_allow_html=True)


    sc1, sc2 = st.columns([1.8, 1])

    with sc1:

        cur_source = st.session_state.get('canslim_sample_source', '全市场A股')

        idx = 0 if cur_source == '全市场A股' else (1 if cur_source == '热门板块' else 2)

        new_source = st.selectbox(

            "样本来源", DEFAULT_GC_SAMPLE_OPTIONS,

            index=idx,

            key="canslim_sample_source_select"

        )

        if "全市场" in new_source:

            new_val = "全市场A股"

        elif "量价" in new_source:

            new_val = "量价"

        else:

            new_val = "热门板块"

        if new_val != cur_source:

            st.session_state.canslim_results = None

            st.session_state.canslim_sample_source = new_val

            st.rerun()

        else:

            st.session_state.canslim_sample_source = new_val


    col1, col2 = st.columns([2, 1])

    cs_sample = st.session_state.get('canslim_sample_source', '全市场A股')

    if cs_sample == "热门板块":

        scan_label = "🔍 加速板块扫描"

    elif cs_sample == "量价":

        scan_label = "🔍 量价扫描"

    else:

        scan_label = "🔍 全市场扫描"

    with col1:

        do_scan = st.button(scan_label, type="primary", width='stretch', key="cs_scan")

    with col2:

        if st.button("🔄 清空缓存", width='stretch', key="cs_clear"):

            st.session_state.canslim_results = None

            st.rerun()


    cs_results = st.session_state.get('canslim_results', None)


    if do_scan or cs_results is None:

        if cs_sample == "热门板块":

            spinner_text = "正在扫描资金加速板块，计算CAN SLIM评分…"

        elif cs_sample == "量价":

            spinner_text = "正在扫描量价反转板块，计算CAN SLIM评分…"

        else:

            spinner_text = "正在扫描全市场，计算CAN SLIM评分…"

        with st.spinner(spinner_text):

            try:

                quotes_df = fetch_all_a_stocks()

                if quotes_df is None or len(quotes_df) == 0:

                    st.error("获取行情数据失败")

                else:

                    quotes_df = quotes_df[~quotes_df['名称'].str.contains('ST|退市|N|C', na=False)]


                    if cs_sample == "热门板块":

                        hot_codes = get_hot_concept_stocks(6)

                        if hot_codes:

                            quotes_df['代码'] = quotes_df['代码'].astype(str).str.zfill(6)

                            quotes_df = quotes_df[quotes_df['代码'].isin(hot_codes)]

                            st.info(f"已锁定 {len(quotes_df)} 只热门概念板块成分股")

                        else:

                            st.warning("未能获取热门板块数据，回退为全市场扫描")

                    elif cs_sample == "量价":

                        vp_codes = get_volprice_sectors(6)

                        if vp_codes:

                            quotes_df['代码'] = quotes_df['代码'].astype(str).str.zfill(6)

                            quotes_df = quotes_df[quotes_df['代码'].isin(vp_codes)]

                            st.info(f"已锁定 {len(quotes_df)} 只量价反转板块成分股")

                        else:

                            st.warning("未能获取量价反转板块数据，回退为全市场扫描")


                    if '量比' in quotes_df.columns:

                        quotes_df['量比'] = pd.to_numeric(quotes_df['量比'], errors='coerce').fillna(1)

                        quotes_df = quotes_df.sort_values('量比', ascending=False)

                    scan_df = quotes_df.head(300).copy()


                    codes = scan_df['代码'].tolist()

                    names = dict(zip(scan_df['代码'], scan_df['名称']))

                    turnover_map = {}

                    if '换手率' in scan_df.columns:

                        turnover_map = dict(zip(scan_df['代码'],

                            pd.to_numeric(scan_df['换手率'], errors='coerce').fillna(0)))

                    # 同时提取总市值（避免ctx硬编码0导致S维度永远0分）

                    cap_map = {}

                    if '总市值' in scan_df.columns:

                        cap_map = dict(zip(scan_df['代码'],

                            pd.to_numeric(scan_df['总市值'], errors='coerce').fillna(0)))


                    kline_dict = {}

                    status = st.empty()

                    bar = st.progress(0)

                    total = len(codes)

                    with ThreadPoolExecutor(max_workers=10) as ex:

                        futures = {ex.submit(get_stock_kline, c, 250): c for c in codes}

                        done = 0

                        for f in as_completed(futures):

                            done += 1

                            c = futures[f]

                            try:

                                kline = f.result(timeout=15)

                                if kline is not None and len(kline) >= 60:

                                    kline_dict[c] = kline

                            except:

                                pass

                            if done % 20 == 0:

                                bar.progress(done / total)

                                status.text(f"📊 获取K线数据... ({done}/{total})")

                    bar.empty()

                    status.empty()


                    st.text("📊 计算RPS排名...")

                    rps_map = compute_rps(kline_dict, list(kline_dict.keys()))


                    results = []

                    scored_codes = list(kline_dict.keys())

                    # 并发获取财务数据（10线程）

                    fin_data_map = {}

                    fin_bar = st.progress(0)

                    fin_status = st.empty()

                    with ThreadPoolExecutor(max_workers=10) as ex:

                        futures = {ex.submit(get_financial_data, code): code for code in scored_codes}

                        fin_done = 0

                        for f in as_completed(futures):

                            fin_done += 1

                            code = futures[f]

                            try:

                                fin_data_map[code] = f.result(timeout=30)

                            except:

                                fin_data_map[code] = None

                            if fin_done % 30 == 0:

                                fin_bar.progress(fin_done / len(scored_codes))

                                fin_status.text(f"📊 获取财务数据... ({fin_done}/{len(scored_codes)})")

                    fin_bar.empty()

                    fin_status.empty()


                    # 统计财务数据获取情况

                    fin_success_count = sum(1 for v in fin_data_map.values() if v and v.get('success'))

                    if fin_success_count < len(scored_codes) * 0.5:

                        st.warning(f"⚠️ 财务数据获取率仅 {fin_success_count}/{len(scored_codes)}，C/A维度可能显示N/A。请检查网络或稍后重试。")


                    # 串行评分计算

                    bar2 = st.progress(0)

                    for i, code in enumerate(scored_codes):

                        kline_df = kline_dict[code]

                        fin_data = fin_data_map.get(code)

                        ctx = {

                            'rps': rps_map.get(code, 0),

                            'market_cap': cap_map.get(code, 0),

                            'turnover_rate': turnover_map.get(code, 0),

                            'fin': fin_data,

                        }

                        time.sleep(0.01)

                        sr = calculate_canslim_score(code, kline_df, stock_pool_context=ctx)

                        if sr.get('pass') and sr.get('综合评分', 0) > 0:

                            results.append({

                                '代码': code,

                                '名称': names.get(code, ''),

                                '综合评分': sr['综合评分'],

                                'C_业绩增速': sr.get('C_业绩增速', 0),

                                'A_持续增长': sr.get('A_持续增长', 0),

                                'N_新催化': sr.get('N_新催化', 0),

                                'S_中小盘': sr.get('S_中小盘', 0),

                                'L_RPS': sr.get('L_RPS', 0),

                                'I_流动性': sr.get('I_流动性', 0),

                                'M_大势': sr.get('M_大势', 0),

                            })

                        if (i + 1) % 50 == 0:

                            bar2.progress((i + 1) / len(scored_codes))

                    bar2.empty()


                    results.sort(key=lambda x: x['综合评分'], reverse=True)

                    st.session_state.canslim_results = results

                    st.rerun()

            except Exception as e:

                st.error(f"扫描出错: {e}")


    cs_results = st.session_state.get('canslim_results', None)


    if cs_results is not None:

        if len(cs_results) == 0:

            st.warning("⚠️ 今日未找到符合条件的CAN SLIM标的。")

        else:

            top_n = min(len(cs_results), 30)

            st.markdown(f"### 📊 CAN SLIM模型 · Top {top_n}")

            st.caption(f"共筛选出 {len(cs_results)} 只标的")


            _cs_dyn_n = calculate_dynamic_recommend_count()

            cs_top_n = cs_results[:_cs_dyn_n]

            st.markdown(f"""

            <div class="top10-container">

                <div class="top10-header">

                    <div class="top10-title">📊 CAN SLIM 精选 Top {_cs_dyn_n} <span class="top10-badge">主升浪信号</span></div>

                </div>

            </div>""", unsafe_allow_html=True)

            btn1, btn2, _ = st.columns([1, 1, 4])

            with btn1:

                if st.button("⭐ 一键加入自选", width='stretch', type="primary", key="cs_add_all"):

                    for s in cs_top_n:

                        if s["代码"] not in st.session_state.watchlist:

                            st.session_state.watchlist.append(s["代码"])

                    save_watchlist(st.session_state.watchlist)

                    st.success(f"已将Top {_cs_dyn_n}全部加入自选！"); st.rerun()

            with btn2:

                top_n_df = pd.DataFrame(cs_top_n)

                top_n_df['代码'] = top_n_df['代码'].astype(str).str.zfill(6)

                export_cols = ['代码', '名称',

                               '综合评分', 'C_业绩增速', 'A_持续增长', 'N_新催化',

                               'S_中小盘', 'L_RPS', 'I_流动性', 'M_大势']

                top_n_df = top_n_df[[c for c in export_cols if c in top_n_df.columns]]

                xlsx_data = _export_df_to_xlsx(top_n_df)

                st.download_button(f"📥 导出Top{_cs_dyn_n}", xlsx_data,

                    f"top{_cs_dyn_n}_canslim_{datetime.now().strftime('%Y%m%d')}.xlsx",

                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                    width='stretch', key="dl_cs_t10")


            st.markdown("")

            for i in range(0, len(cs_top_n), 5):

                row_stocks = cs_top_n[i:i+5]

                cols = st.columns(5)

                for j, stock in enumerate(row_stocks):

                    with cols[j]:

                        rank = i + j + 1

                        rank_color = "#FFD700" if rank == 1 else "#C0C0C0" if rank == 2 else "#CD7F32" if rank == 3 else "#CCC"

                        st.markdown(f"""

                        <div class="top10-card">

                            <div class="top10-rank" style="color:{rank_color};">{rank}</div>

                            <div style="padding-right:30px;">

                                <div style="font-weight:700;color:#333;font-size:15px;margin-bottom:2px;">{stock['名称']}</div>

                                <div style="font-size:11px;color:#888;margin-bottom:8px;">{str(stock.get('代码','')).zfill(6)}</div>

                                <div style="display:flex;justify-content:space-between;align-items:center;">

                                    <span style="font-size:20px;font-weight:800;color:#C4842D;">{stock['综合评分']}</span>

                                </div>

                                <div style="margin-top:6px;"><span class="metric-badge badge-strong">CAN SLIM</span></div>

                            </div>

                        </div>""", unsafe_allow_html=True)

                        code = stock["代码"]; in_wl = code in st.session_state.watchlist

                        if st.button("⭐" if in_wl else "+自选", key=f"cs_t10_{code}",

                            width='stretch', type="primary" if in_wl else "secondary"):

                            if in_wl: st.session_state.watchlist.remove(code)

                            else: st.session_state.watchlist.append(code)

                            save_watchlist(st.session_state.watchlist); st.rerun()


            st.markdown("---")

            fdf = pd.DataFrame(cs_results[:top_n])

            fdf.index = range(1, len(fdf) + 1)

            fdf['代码'] = fdf['代码'].astype(str).str.zfill(6)

            display_cols = ['代码', '名称', '综合评分',

                            'C_业绩增速', 'A_持续增长', 'N_新催化',

                            'S_中小盘', 'L_RPS', 'I_流动性', 'M_大势']

            df_display = fdf[[c for c in display_cols if c in fdf.columns]].copy()

            # -1 → "N/A" 表示财务数据不可用，与真的0分区分

            for col in ['C_业绩增速', 'A_持续增长', 'S_中小盘', 'I_流动性']:

                if col in df_display.columns:

                    df_display[col] = df_display[col].apply(lambda x: "N/A" if x == -1 else x)

            # Arrow 序列化兼容：含 "N/A" 的 object 列统一转 str

            for c in df_display.columns:

                if df_display[c].dtype == 'object':

                    df_display[c] = df_display[c].astype(str)

            st.dataframe(df_display, width='stretch')


            st.markdown("---")

            st.caption("操作区 — 加/取消自选")

            cs_buttons = [{'代码': r['代码'], '名称': r['名称']} for r in cs_results[:30]]

            render_stock_buttons(cs_buttons, prefix="cs")


            export_df = pd.DataFrame(cs_results)

            export_df['代码'] = export_df['代码'].astype(str).str.zfill(6)

            export_cols = ['代码', '名称', '综合评分',

                           'C_业绩增速', 'A_持续增长', 'N_新催化',

                           'S_中小盘', 'L_RPS', 'I_流动性', 'M_大势']

            export_df = export_df[[c for c in export_cols if c in export_df.columns]]

            csv_data = export_df.to_csv(index=False).encode('utf-8-sig')

            st.download_button("📥 导出 Top30 CSV", csv_data,

                f"canslim_top30_{datetime.now().strftime('%Y%m%d')}.csv",

                "text/csv", key="dl_cs_top30")

    elif not do_scan:

        st.info("💡 点击「全市场扫描」启动CAN SLIM模型选股（基于七因子简化版评分）。")

