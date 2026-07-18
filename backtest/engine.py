# -*- coding: utf-8 -*-
"""
真实回测引擎 v4 - 跌幅超5%止损，共振卖出模式（T日开盘买入）
规则：T-1日收盘选股 -> T日开盘买入 -> 盘中止损或收盘共振卖出
每只股票固定买入1万元，总资金按最终累计投入实际计算
支持追高/低吸/共振/金叉/AI综合评分等多模型

每日执行顺序：
  1. 检查持仓：盘中止损（OHLC模拟）> 共振检查 > 继续持有
  2. 买入昨天选的股（今天开盘价），每只固定1万元
  3. 今天收盘后选股（供明天买入）
"""
import os, sys, struct, time, json, argparse, pickle
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project root = parent of backtest/ directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TDX_VIPDOC_DIR = None
for _p in [r"C:\zd_cjzq\vipdoc", r"C:\new_tdx\vipdoc", r"C:\tdx\vipdoc",
           r"D:\new_tdx\vipdoc", r"D:\tdx\vipdoc"]:
    if os.path.isdir(_p):
        TDX_VIPDOC_DIR = _p
        break

TRADING_COST = 0.00015
PER_STOCK_AMOUNT = 10000
INITIAL_CAPITAL = 10_000_000

# 卖出参数
STOP_LOSS = 0.05
HARD_TAKE_PROFIT = 0.12
HARD_TAKE_PROFIT_CEILING = 0.35
MOVING_STOP_GAIN = 0.15
MOVING_STOP_DRAWDOWN = 0.07
COOLING_PERIOD = 5
RESONANCE_THRESHOLD = 40


def scan_all_stock_codes():
    codes = []
    if not TDX_VIPDOC_DIR:
        return codes
    for market in ['sh', 'sz', 'bj']:
        lday_dir = os.path.join(TDX_VIPDOC_DIR, market, "lday")
        if not os.path.isdir(lday_dir):
            continue
        for f in os.listdir(lday_dir):
            if f.endswith('.day') and len(f) == 12:
                code = f[2:8]
                if code[0] in '012346789':
                    codes.append(code)
    return sorted(set(codes))


def read_tdx_day_file(stock_code):
    if not TDX_VIPDOC_DIR:
        return None
    code = str(stock_code).zfill(6)
    if code.startswith(('6', '9')):
        fp = os.path.join(TDX_VIPDOC_DIR, "sh", "lday", "sh{}.day".format(code))
    elif code.startswith(('0', '3', '2')):
        fp = os.path.join(TDX_VIPDOC_DIR, "sz", "lday", "sz{}.day".format(code))
    elif code.startswith(('4', '8')):
        fp = os.path.join(TDX_VIPDOC_DIR, "bj", "lday", "bj{}.day".format(code))
    else:
        return None
    if not os.path.exists(fp):
        return None
    try:
        data = []
        with open(fp, 'rb') as f:
            while True:
                raw = f.read(32)
                if len(raw) < 32:
                    break
                date_int, open_i, high_i, low_i, close_i, amount, volume, _ = struct.unpack('<IIIIIfII', raw)
                if date_int < 19900101 or date_int > 20991231:
                    continue
                close_p = close_i / 100.0
                if close_p <= 0 or close_p > 100000:
                    continue
                data.append({
                    'date': pd.Timestamp(str(date_int)),
                    'open': open_i / 100.0,
                    'high': high_i / 100.0,
                    'low': low_i / 100.0,
                    'close': close_p,
                    'volume': float(volume),
                    'amount': amount,
                })
        if not data:
            return None
        df = pd.DataFrame(data).sort_values('date').reset_index(drop=True)
        return df
    except Exception:
        return None


def _build_stock_data_from_kline(kline_df):
    """从K线DataFrame构造stock_screener需要的stock_data字典"""
    c = kline_df['close'].astype(float)
    v = kline_df['volume'].astype(float)
    h = kline_df['high'].astype(float)
    l = kline_df['low'].astype(float)
    n = len(c)
    close = c.iloc[-1]
    # 均线
    ma5 = c.rolling(5).mean().iloc[-1] if n >= 5 else close
    ma5_prev = c.rolling(5).mean().iloc[-2] if n >= 6 else ma5
    ma10 = c.rolling(10).mean().iloc[-1] if n >= 10 else close
    ma10_prev = c.rolling(10).mean().iloc[-2] if n >= 11 else ma10
    ma20 = c.rolling(20).mean().iloc[-1] if n >= 20 else close
    ma20_prev = c.rolling(20).mean().iloc[-2] if n >= 21 else ma20
    ma60 = c.rolling(60).mean().iloc[-1] if n >= 60 else close
    ma60_prev = c.rolling(60).mean().iloc[-2] if n >= 61 else ma60
    # 涨跌幅
    g3 = (c.iloc[-1] / c.iloc[-4] - 1) * 100 if n >= 4 else 0
    g5 = (c.iloc[-1] / c.iloc[-6] - 1) * 100 if n >= 6 else 0
    g10 = (c.iloc[-1] / c.iloc[-11] - 1) * 100 if n >= 11 else 0
    # 量价
    vol_today = v.iloc[-1]
    ma5_vol = v.rolling(5).mean().iloc[-1] if n >= 5 else vol_today
    amplitude = (h.iloc[-1] - l.iloc[-1]) / l.iloc[-1] * 100 if l.iloc[-1] > 0 else 0
    # 新高判断
    new_high_today = (close >= h.tail(20).max() * 0.99) if n >= 20 else False
    new_high_2d = (close >= h.tail(2).max() * 0.99) if n >= 2 else False
    return {
        "close": close,
        "ma5": ma5, "ma5_prev": ma5_prev,
        "ma10": ma10, "ma10_prev": ma10_prev,
        "ma20": ma20, "ma20_prev": ma20_prev,
        "ma60": ma60, "ma60_prev": ma60_prev,
        "3d_gain": g3, "5d_gain": g5, "10d_gain": g10,
        "vol_today": vol_today, "ma5_vol": ma5_vol,
        "amplitude": amplitude,
        "new_high_today": new_high_today, "new_high_2d": new_high_2d,
        # 以下字段回测环境下无法获取，设默认值（不影响核心趋势+动量维度）
        "sector": "", "sector_daily_gain": 0, "sector_limit_up_count_count": 0,
        "sector_net_inflow": 0, "inst_net_buy_3d": 0, "north_net_buy": 0,
        "pure_hot_money_only": False, "pe_hist_percent": 50, "turnover_rate": 10,
        "stock_hot_score": 0, "inst_net_sell_2d": False, "macd_top_divergence": False,
    }


def score_stock_from_kline(kline_df, model='chase_high'):
    """调用stock_screener评分函数，返回总分0-100（根据模型区分评分体系）"""
    if kline_df is None or len(kline_df) < 60:
        return 0
    try:
        stock_data = _build_stock_data_from_kline(kline_df)
        import os as _os, warnings as _w, logging as _log
        _w.filterwarnings('ignore')
        _os.environ.setdefault('STREAMLIT_WATCH_MODULES', 'false')
        _os.environ.setdefault('STREAMLIT_SERVER_RUN_ON_SAVE', 'false')
        _log.getLogger('streamlit').setLevel(_log.ERROR)

        if model in ('buy_low',):
            from core.scoring.lowbuy import calculate_lowbuy_score
            from core.scoring.lowbuy import DEFAULT_LOWBUY_WEIGHTS
            # 列名翻译：回测引擎K线用英文列名，低吸评分函数需中文列名
            kline_df_cn = kline_df.rename(columns={
                'close': '收盘', 'volume': '成交量',
                'open': '开盘', 'high': '最高', 'low': '最低',
                'amount': '成交额',
            })
            result = calculate_lowbuy_score(stock_data, kline_df_cn, params=None, weights=DEFAULT_LOWBUY_WEIGHTS)
            return result.get("综合评分", 0) if result.get("pass") else 0
        elif model in ('golden_cross',):
            from core.scoring.golden_cross import calculate_golden_cross_score
            # 列名翻译：回测引擎K线用英文列名，金叉评分函数需中文列名
            kline_df_cn = kline_df.rename(columns={
                'close': '收盘', 'volume': '成交量',
                'open': '开盘', 'high': '最高', 'low': '最低',
                'amount': '成交额',
            })
            result = calculate_golden_cross_score(stock_data, kline_df_cn, weights=None)
            return result.get("综合评分", 0) if result.get("pass") else 0
        elif model in ('canslim',):
            from core.scoring.canslim import calculate_canslim_score
            # 列名翻译：回测引擎K线用英文列名
            kline_df_cn = kline_df.rename(columns={
                'close': '收盘', 'volume': '成交量',
                'open': '开盘', 'high': '最高', 'low': '最低',
                'amount': '成交额',
            })
            # CAN SLIM 需要 market_cap / turnover_rate 上下文，回测环境给合理默认值
            ctx = {
                'rps': 50,
                'market_cap': 200 * 1e8,
                'turnover_rate': 5,
            }
            result = calculate_canslim_score(stock_data.get('code', ''), kline_df_cn, stock_pool_context=ctx)
            return result.get("综合评分", 0) if result.get("pass") else 0
        elif model in ('dilemma_reversal',):
            from core.scoring.dilemma import calculate_dilemma_reversal_score
            # 列名翻译：回测引擎K线用英文列名
            kline_df_cn = kline_df.rename(columns={
                'close': '收盘', 'volume': '成交量',
                'open': '开盘', 'high': '最高', 'low': '最低',
                'amount': '成交额',
            })
            result = calculate_dilemma_reversal_score(stock_data.get('code', ''), kline_df_cn)
            return result.get("综合评分", 0) if result.get("pass") else 0
        else:
            from core.scoring.chase_high import calculate_v3_total_score
            from core.scoring.chase_high import DEFAULT_WEIGHTS
            result = calculate_v3_total_score(stock_data, weights=DEFAULT_WEIGHTS)
            return result.get("综合评分", 0) if result.get("pass") else 0
    except Exception:
        return 0


# ================================================================
#                      股票名称加载 & ST过滤
# ================================================================

_name_map_cache = None

def load_stock_names():
    """加载代码→名称映射（多源合并，含ST标记），进程级缓存"""
    global _name_map_cache
    if _name_map_cache is not None:
        return _name_map_cache
    name_map = {}
    try:
        import json, os
        cache_path = os.path.join(BASE_DIR, 'stock_name_cache.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                name_map = json.load(f)
    except Exception:
        pass
    _name_map_cache = name_map
    return _name_map_cache


def _is_st_stock(code, name_map=None):
    """判断是否为ST/*ST风险警示股"""
    if name_map is None:
        name_map = _name_map_cache or {}
    name = name_map.get(code, '')
    if not name:
        return False
    return 'ST' in str(name).upper()


# ================================================================
#                      K线数据缓存（pickle）
# ================================================================

def _get_cache_path():
    """获取K线缓存文件路径"""
    return os.path.join(BASE_DIR, 'kline_cache.pkl')


def _get_day_files_mtime(vipdoc_dir):
    """获取所有 .day 文件的最大 mtime，用于判断缓存是否过期"""
    max_mtime = 0
    for market in ['sh', 'sz', 'bj']:
        lday_dir = os.path.join(vipdoc_dir, market, "lday")
        if not os.path.isdir(lday_dir):
            continue
        try:
            for f in os.listdir(lday_dir):
                if f.endswith('.day') and len(f) == 12:
                    fp = os.path.join(lday_dir, f)
                    mtime = os.path.getmtime(fp)
                    if mtime > max_mtime:
                        max_mtime = mtime
        except Exception:
            continue
    return max_mtime


def _load_kline_cache(warmup_cutoff, start_date):
    """尝试从 pickle 缓存加载 K 线数据。
    
    返回 (kline_dict, cache_valid)。
    cache_valid=True 表示缓存命中且数据有效。
    """
    cache_path = _get_cache_path()
    if not os.path.exists(cache_path):
        return None, False
    
    try:
        day_mtime = _get_day_files_mtime(TDX_VIPDOC_DIR)
        cache_mtime = os.path.getmtime(cache_path)
        
        if day_mtime > cache_mtime:
            print("  [缓存] .day 文件有更新，缓存失效，重新加载")
            return None, False
        
        print("  [缓存] 命中! 从 {} 加载...".format(os.path.basename(cache_path)))
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 校验缓存结构
        if not isinstance(cache_data, dict) or 'kline_dict' not in cache_data:
            return None, False
        cached_cutoff_str = str(cache_data.get('warmup_cutoff', ''))
        current_cutoff_str = str(warmup_cutoff.date())
        if cached_cutoff_str > current_cutoff_str:
            print("  [缓存] warmup_cutoff 过期 (缓存从 {} 开始，需要从 {} 开始)，重新加载".format(
                cached_cutoff_str, current_cutoff_str))
            return None, False
        
        kline_dict = cache_data['kline_dict']
        # 二次过滤：确保所有数据满足 warmup 和 start_date 条件
        filtered = {}
        for code, df in kline_dict.items():
            df = df[df['date'] >= warmup_cutoff]
            if len(df) < 60:
                continue
            if df['date'].max() < start_date:
                continue
            filtered[code] = df
        return filtered, True
    except Exception as e:
        print("  [缓存] 加载失败: {}，重新加载".format(e))
        return None, False


def _save_kline_cache(kline_dict, warmup_cutoff):
    """将 K 线数据保存到 pickle 缓存"""
    cache_path = _get_cache_path()
    cache_data = {
        'warmup_cutoff': str(warmup_cutoff.date()),
        'kline_dict': kline_dict,
        'saved_at': time.time(),
    }
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("  [缓存] 已保存到 {} ({} 只股票)".format(
            os.path.basename(cache_path), len(kline_dict)))
    except Exception as e:
        print("  [缓存] 保存失败: {}".format(e))


def _read_single_stock(code):
    """读取单只股票的K线数据（供线程池调用）"""
    df = read_tdx_day_file(code)
    if df is None or len(df) < 60:
        return code, None, 'insufficient'
    return code, df, 'ok'


def _load_kline_parallel(all_codes, warmup_cutoff, start_date, max_workers=16):
    """使用线程池并行加载所有股票的K线数据"""
    kline_dict = {}
    loaded = 0
    skipped = 0
    total = len(all_codes)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_read_single_stock, code): code for code in all_codes}
        for i, future in enumerate(as_completed(futures)):
            code, df, status = future.result()
            if (i + 1) % 2000 == 0 or i == total - 1:
                pct = (i + 1) * 100 // total
                print("  加载: {}% ({}/{})".format(pct, i + 1, total))
            
            if status != 'ok':
                skipped += 1
                continue
            
            df = df[df['date'] >= warmup_cutoff]
            if len(df) < 60:
                skipped += 1
                continue
            if df['date'].max() < start_date:
                skipped += 1
                continue
            kline_dict[code] = df
            loaded += 1
    
    return kline_dict, loaded, skipped


def score_all_stocks(kline_dict, up_to_date, top_n=5, skip_st=True, model='chase_high'):
    """在up_to_date收盘后对所有股票打分，返回[(code, score)]。top_n=None 返回全部"""
    name_map = load_stock_names() if skip_st else {}
    scores = []
    for code, df in kline_dict.items():
        if skip_st and _is_st_stock(code, name_map):
            continue
        df_slice = df[df['date'] <= up_to_date]
        if len(df_slice) < 60:
            continue
        c = df_slice['close']
        if len(c) >= 4:
            g3 = (c.iloc[-1] / c.iloc[-4] - 1) * 100
            if g3 > 35 or g3 < -15:
                continue
        if len(c) >= 60:
            ma60v = c.rolling(60).mean().iloc[-1]
            if ma60v > 0 and abs(c.iloc[-1] / ma60v - 1) > 0.8:
                continue
        sc = score_stock_from_kline(df_slice, model=model)
        if sc > 0:
            scores.append((code, sc))
    scores.sort(key=lambda x: -x[1])
    if top_n is not None:
        return scores[:top_n]
    return scores


def get_trading_days(kline_dict, start_date, end_date):
    all_dates = set()
    for df in kline_dict.values():
        dates_in_range = df[(df['date'] >= start_date) & (df['date'] <= end_date)]['date']
        all_dates.update(dates_in_range.tolist())
    return sorted(all_dates)


def _compute_backtest_resonance(code, today, kline_dict):
    """回测版共振评分：仅K线结构（50%）+ 个股动能（50%），真实0-100"""
    df = kline_dict.get(code)
    if df is None:
        return 50
    mask = df['date'] <= today
    if mask.sum() < 60:
        return 50
    d = df[mask].tail(60).copy()
    c = d['close'].astype(float)
    v = d['volume'].astype(float)
    h = d['high'].astype(float)

    # K线结构评分 (0-50)，基础分0
    ma5 = c.rolling(5).mean().iloc[-1]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    close_v = c.iloc[-1]
    k_score = 0
    if close_v > ma5:
        k_score += 10
    if close_v > ma20:
        k_score += 10
    if close_v > ma60:
        k_score += 10
    if ma5 > ma20:
        k_score += 10
    ma5_vol = v.rolling(5).mean().iloc[-1] if len(v) >= 5 else v.iloc[-1]
    if v.iloc[-1] > ma5_vol * 1.2:
        k_score += 10
    k_score = min(k_score, 50)

    # 个股动能评分 (0-50)，替代板块热度，基础分0
    s_score = 0
    if len(c) >= 3:
        d3_gain = (c.iloc[-1] / c.iloc[-4] - 1) * 100
        if d3_gain > 5:
            s_score += 15
        elif d3_gain > 0:
            s_score += 8
    if len(c) >= 10:
        d10_gain = (c.iloc[-1] / c.iloc[-11] - 1) * 100
        if d10_gain > 10:
            s_score += 15
        elif d10_gain > 0:
            s_score += 8
    if len(c) >= 20:
        ma20_trend = c.tail(20).diff().mean()  # 日均涨幅趋势
        if ma20_trend > 0:
            s_score += 10
        if close_v >= h.tail(20).max() * 0.98:
            s_score += 10
    s_score = min(s_score, 50)

    return k_score + s_score


def run_backtest(start_date, end_date, top_n=5, model='chase_high',
                stop_loss=0.05, hard_take_profit=0.12, hard_take_profit_ceiling=0.35,
                moving_stop_gain=0.15, moving_stop_drawdown=0.07, cooling_period=5,
                resonance_threshold=40, min_z_score=0.0):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    print("")
    print("=" * 60)
    model_label = {'chase_high': '追高', 'strong_leaders': '抓强势龙头股', 'ai_scoring': 'AI综合评分', 'buy_low': '低吸', 'golden_cross': '金叉', 'resonance': '共振', 'canslim': 'CAN SLIM模型（抓主升浪股）', 'dilemma_reversal': '困境反转模型（抓周期股）'}.get(model, model)
    print(" 回测引擎 v4 - 跌幅超{:.1f}%止损，共振<{}卖出，移动止盈{:.1f}%→回撤{:.1f}%".format(
        stop_loss * 100, resonance_threshold, moving_stop_gain * 100, moving_stop_drawdown * 100))
    print(" 模型: {}  |  区间: {} ~ {}".format(model_label, start_date, end_date))
    print(" 资金池: {:,.0f}万  |  每只固定投入: {:,.0f}元".format(INITIAL_CAPITAL / 10000, PER_STOCK_AMOUNT))
    print(" Top-N: {}  |  冷却期: {}天  |  硬止盈: {:.1f}%(共振<50) / {:.1f}%(无条件)".format(
        top_n, cooling_period, hard_take_profit * 100, hard_take_profit_ceiling * 100))
    if min_z_score > 0:
        print(" 得分门槛: Z-Score >= {:.1f}".format(min_z_score))
    else:
        print(" 得分门槛: 无")
    print(" 单边成本: {:.3f}%".format(TRADING_COST * 100))
    print("=" * 60)
    print("")

    # [1] 扫描
    print("[1/4] 扫描通达信本地数据...")
    all_codes = scan_all_stock_codes()
    print(" 共发现 {} 只股票".format(len(all_codes)))

    # [2] 加载K线
    print("[2/4] 加载K线数据...")
    warmup_cutoff = start - pd.Timedelta(days=150)
    t0 = time.time()

    # 优先尝试 pickle 缓存（秒级加载）
    kline_dict, cache_hit = _load_kline_cache(warmup_cutoff, start)
    if cache_hit:
        loaded = len(kline_dict)
        skipped = len(all_codes) - loaded
        print(" 已加载 {} 只 (跳过 {} 只), 耗时 {:.1f}s (缓存命中)".format(
            loaded, skipped, time.time() - t0))
    else:
        # 冷缓存：并行读取所有 .day 文件
        kline_dict, loaded, skipped = _load_kline_parallel(
            all_codes, warmup_cutoff, start, max_workers=16)
        elapsed = time.time() - t0
        print(" 已加载 {} 只 (跳过 {} 只), 耗时 {:.1f}s".format(loaded, skipped, elapsed))
        # 保存缓存供下次使用
        if loaded > 100:
            _save_kline_cache(kline_dict, warmup_cutoff)

    if loaded < 100:
        print("错误: 有效K线太少")
        return None

    # [3] 交易日序列
    print("[3/4] 提取交易日序列...")
    trading_days = get_trading_days(kline_dict, start, end)
    print(" 共 {} 个交易日".format(len(trading_days)))
    if len(trading_days) < 3:
        print("错误: 交易日太少")
        return None

    # [4] 回测主循环
    print("[4/4] 开始回测...")
    t0 = time.time()

    trades = []
    daily_records = []
    selection_cache = {}
    # NEW: 记录每日选股结果
    daily_selections = []

    open_positions = {}
    cash = INITIAL_CAPITAL
    total_invested = 0.0
    cumulative_sell_proceeds = 0.0  # 累计卖出回收资金

    for td_idx in range(len(trading_days)):
        today = trading_days[td_idx]

        # 获取今日行情数据（批量）
        today_data = {}
        codes_needed = set(list(open_positions.keys()))
        if td_idx - 1 in selection_cache:
            for c, _ in selection_cache[td_idx - 1]:
                codes_needed.add(c)
        for code in codes_needed:
            df = kline_dict.get(code)
            if df is None:
                continue
            td_rows = df[df['date'] == today]
            if len(td_rows) == 0:
                continue
            today_data[code] = {
                'open': float(td_rows['open'].iloc[0]),
                'high': float(td_rows['high'].iloc[0]),
                'low': float(td_rows['low'].iloc[0]),
                'close': float(td_rows['close'].iloc[0]),
                'volume': float(td_rows['volume'].iloc[0]),
            }

        # === Step 1: 五层卖出机制 ===
        sold_codes = []
        for code, pos in open_positions.items():
            if code not in today_data:
                continue
            today_open = today_data[code]['open']
            today_low = today_data[code]['low']
            today_close = today_data[code]['close']
            hold_days = (today - pos['buy_date']).days

            # 更新最高价（移动止盈用）
            if today_close > pos.get('peak_price', pos['buy_price']):
                pos['peak_price'] = today_close

            # ── 优先级1：盘中止损 ──
            stop_price = pos['buy_price'] * (1 - stop_loss)
            intraday_stop_hit = False
            intraday_sell_price = None
            if today_low <= stop_price:
                intraday_stop_hit = True
                if today_open > stop_price:
                    intraday_sell_price = stop_price
                else:
                    intraday_sell_price = today_open

            should_sell = False
            sell_reason = ""
            sell_price = today_close
            resonance = 0.0
            resonance_computed = False

            if intraday_stop_hit:
                should_sell = True
                sell_reason = "stop_loss"
                sell_price = intraday_sell_price

            # ── 优先级2：硬止盈 ──
            if not should_sell:
                gross_ret = today_close / pos['buy_price'] - 1
                if gross_ret >= hard_take_profit_ceiling:
                    should_sell = True
                    sell_reason = "hard_tp"
                    resonance = _compute_backtest_resonance(code, today, kline_dict)
                    resonance_computed = True
                elif gross_ret >= hard_take_profit:
                    resonance = _compute_backtest_resonance(code, today, kline_dict)
                    resonance_computed = True
                    if resonance < 50:
                        should_sell = True
                        sell_reason = "hard_tp"

            # ── 优先级3：移动止盈 ──
            if not should_sell:
                peak_price = pos.get('peak_price', pos['buy_price'])
                peak_profit = peak_price / pos['buy_price'] - 1
                if peak_profit >= moving_stop_gain:
                    drawdown = (peak_price - today_low) / peak_price
                    if drawdown >= moving_stop_drawdown:
                        should_sell = True
                        sell_reason = "moving_stop"

            # ── 优先级4/5：共振卖出 + 时间止盈 ──
            if not should_sell:
                if not resonance_computed:
                    resonance = _compute_backtest_resonance(code, today, kline_dict)
                if hold_days < cooling_period:
                    pass  # 冷却期：共振不参与
                elif hold_days < 8:
                    if resonance < resonance_threshold:
                        should_sell = True
                        sell_reason = "resonance"
                elif hold_days < 11:
                    if resonance < 50:
                        should_sell = True
                        sell_reason = "resonance_time"
                else:
                    should_sell = True
                    sell_reason = "time_exit"

            if should_sell:
                gross_ret = sell_price / pos['buy_price'] - 1
                net_ret = gross_ret - TRADING_COST * 2
                proceeds = pos['shares'] * sell_price * (1 - TRADING_COST)
                buy_cost = pos['shares'] * pos['buy_price'] * (1 + TRADING_COST)
                pnl = proceeds - buy_cost
                cash += proceeds
                cumulative_sell_proceeds += proceeds
                total_invested -= buy_cost
                sold_codes.append(code)
                trades.append({
                    'code': code, 'score': pos['score'],
                    'selection_date': pos['selection_date'],
                    'buy_date': pos['buy_date'], 'sell_date': today,
                    'buy_price': pos['buy_price'], 'sell_price': sell_price,
                    'shares': pos['shares'],
                    'gross_return': round(gross_ret * 100, 2),
                    'net_return': round(net_ret * 100, 2),
                    'pnl': round(pnl, 2),
                    'hold_days': (today - pos['buy_date']).days,
                    'resonance': round(resonance, 1),
                    'sell_reason': sell_reason,
                })
        for code in sold_codes:
            del open_positions[code]

        # === Step 2: 买入昨天选的股（得分门槛过滤） ===
        prev_idx = td_idx - 1
        if prev_idx in selection_cache:
            candidates = selection_cache[prev_idx]
            filtered_count = len(candidates)

            # 得分门槛过滤：Z-Score 方案，只买入得分超过当日均值 N 个标准差的股票
            if min_z_score > 0:
                all_scores = score_all_stocks(kline_dict, trading_days[prev_idx],
                                              top_n=None, model=model)
                score_values = [s[1] for s in all_scores]
                mean_score = np.mean(score_values)
                std_score = np.std(score_values)
                threshold = mean_score + min_z_score * std_score
                candidates = [(c, s) for c, s in candidates if s >= threshold]
                filtered_count = len(candidates)

            for code, score in candidates:
                if code in open_positions:
                    continue
                if code not in today_data:
                    continue
                buy_price = today_data[code]['open']
                if buy_price <= 0 or buy_price > PER_STOCK_AMOUNT:
                    continue
                shares = int(PER_STOCK_AMOUNT / buy_price / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * buy_price * (1 + TRADING_COST)
                cash -= cost
                total_invested += cost

                df = kline_dict.get(code)
                if df is not None:
                    before_today = df[df['date'] < today]
                    if len(before_today) > 0:
                        prev_close = float(before_today['close'].iloc[-1])
                    else:
                        prev_close = buy_price
                else:
                    prev_close = buy_price

                open_positions[code] = {
                    'code': code, 'score': score,
                    'selection_date': trading_days[prev_idx],
                    'buy_date': today, 'buy_price': buy_price,
                    'shares': shares, 'prev_close': prev_close,
                }

        # === Step 3: 今天收盘后选股（供明天买入） ===
        if td_idx + 1 < len(trading_days):
            top_stocks = score_all_stocks(kline_dict, today, top_n=top_n, model=model)
            selection_cache[td_idx] = top_stocks
            # NEW: 记录每日选股结果
            for rank, (code, score) in enumerate(top_stocks, 1):
                daily_selections.append({
                    'selection_date': today.strftime('%Y-%m-%d'),
                    'rank': rank,
                    'code': code,
                    'score': score,
                    'next_buy_date': trading_days[td_idx + 1].strftime('%Y-%m-%d'),
                })
        else:
            selection_cache[td_idx] = []

        # === 更新持仓的prev_close为今日收盘价 ===
        for code, pos in open_positions.items():
            if code in today_data:
                pos['prev_close'] = today_data[code]['close']

        # === 计算今日总资产（effective_value = 持仓市值 + 累计卖出回收金） ===
        pos_value = 0.0
        for code, pos in open_positions.items():
            if code in today_data:
                pos_value += pos['shares'] * today_data[code]['close']
            else:
                pos_value += pos['shares'] * pos['buy_price']
        portfolio_value = cash + pos_value
        effective_value = pos_value + cumulative_sell_proceeds
        benchmark_return = 0.0  # 回测版无基准

        daily_records.append({
            'date': today,
            'portfolio_value': round(portfolio_value, 2),
            'cash': round(cash, 2),
            'position_value': round(pos_value, 2),
            'effective_value': round(effective_value, 2),
            'benchmark_return': benchmark_return,
            'num_positions': len(open_positions),
            'sold_count': len(sold_codes),
            'total_invested': round(total_invested, 2),
        })

        if (td_idx + 1) % 5 == 0 or td_idx == len(trading_days) - 1:
            filter_info = ""
            if min_z_score > 0 and prev_idx in selection_cache:
                total_picks = len(selection_cache[prev_idx])
                bought_today = len([c for c, _ in candidates if c in open_positions])
                filter_info = " | 买入:{}/{}".format(bought_today, total_picks)
            print("  进度: {}/{} | 损益: {:+,.0f} | 已占用资金: {:,.0f} | 持仓: {}股 | 已卖: {}笔{} | {:.0f}s".format(
                td_idx + 1, len(trading_days), portfolio_value - INITIAL_CAPITAL, total_invested, len(open_positions),
                len(sold_codes), filter_info, time.time() - t0))

    print(" 回测完成! 耗时 {:.1f}s  |  共 {} 笔交易".format(time.time() - t0, len(trades)))
    selections_df = pd.DataFrame(daily_selections) if daily_selections else pd.DataFrame()
    return pd.DataFrame(trades), pd.DataFrame(daily_records), selections_df


def compute_stats(trades_df, daily_df, per_stock_amount, top_n):
    if trades_df is None or len(trades_df) == 0:
        return None
    total_trades = len(trades_df)
    win_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    avg_return = trades_df['net_return'].mean()
    median_return = trades_df['net_return'].median()
    total_pnl = trades_df['pnl'].sum()

    if daily_df is not None and len(daily_df) > 0:
        max_deployed = daily_df['total_invested'].max()
        end_value = daily_df['portfolio_value'].iloc[-1]
        if max_deployed > 0:
            total_return = (end_value / max_deployed - 1) * 100
        else:
            total_return = 0
        dv = daily_df['portfolio_value'].values.astype(float)
        peak = np.maximum.accumulate(np.maximum(dv, 0))
        drawdowns = (peak - dv) / peak * 100
        max_drawdown = np.max(drawdowns[~np.isnan(drawdowns)]) if len(drawdowns) > 0 else 0
        daily_rets_series = pd.Series(dv).pct_change().dropna()
        daily_rets = daily_rets_series[daily_rets_series != np.inf]
        if len(daily_rets) > 1:
            ann_return = (1 + daily_rets.mean()) ** 245 - 1
            ann_vol = daily_rets.std() * np.sqrt(245)
            sharpe = (ann_return - 0.03) / ann_vol if ann_vol > 0 else 0
            calmar = ann_return * 100 / max_drawdown if max_drawdown > 0 else 0
        else:
            ann_return = 0
            sharpe = 0
            calmar = 0
        avg_hold_days = trades_df['hold_days'].mean() if 'hold_days' in trades_df.columns else 0
    else:
        end_value = 0
        total_return = 0
        max_drawdown = 0
        sharpe = 0
        calmar = 0
        ann_return = 0
        max_deployed = 0
        avg_hold_days = 0

    return {
        'per_stock_amount': per_stock_amount,
        'top_n': top_n,
        'total_invested': round(daily_df['total_invested'].iloc[-1], 2) if daily_df is not None and len(daily_df) > 0 else 0,
        'max_deployed': round(max_deployed, 2),
        'end_value': round(end_value, 2),
        'total_return': round(total_return, 2),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe': round(sharpe, 2),
        'calmar': round(calmar, 2),
        'ann_return': round(ann_return * 100, 2),
        'total_trades': total_trades,
        'win_trades': win_trades,
        'win_rate': round(win_rate * 100, 1),
        'avg_return': round(avg_return, 2),
        'median_return': round(median_return, 2),
        'avg_pnl': round(trades_df['pnl'].mean(), 2),
        'total_pnl': round(total_pnl, 2),
        'avg_hold_days': round(avg_hold_days, 1),
    }


def print_report(stats, trades_df, daily_df, start_date, end_date):
    if stats is None:
        print("无回测结果")
        return
    print("")
    print("=" * 60)
    print(" 回测报告 - v4 止损+共振卖出模式")
    print(" 区间: {} ~ {}".format(start_date, end_date))
    print(" 参数: Top{}, 每只{:,.0f}元".format(stats['top_n'], stats['per_stock_amount']))
    print(" 规则: T日开盘买 -> 跌幅超5%止损 / 共振卖出")
    print("=" * 60)
    print(" 累计总投入:  {:>12,.2f}".format(stats['total_invested']))
    print(" 最高部署资金: {:>12,.2f}".format(stats['max_deployed']))
    print(" 期末资产:    {:>12,.2f}".format(stats['end_value']))
    print(" 累计收益率:  {:>+10.2f}%  (相对最高部署资金)".format(stats['total_return']))
    print(" 年化收益:    {:>+10.2f}%".format(stats['ann_return']))
    print(" 最大回撤:    {:>10.2f}%".format(stats['max_drawdown']))
    print(" 夏普比率:    {:>10.2f}".format(stats['sharpe']))
    print(" 卡玛比率:    {:>10.2f}".format(stats['calmar']))
    print("=" * 60)
    print(" 总交易次数: {}".format(stats['total_trades']))
    print(" 胜率:       {:.1f}%".format(stats['win_rate']))
    print(" 平均净收益: {:.2f}%".format(stats['avg_return']))
    print(" 中位净收益: {:.2f}%".format(stats['median_return']))
    print(" 平均盈亏:   {:>+,.2f}".format(stats['avg_pnl']))
    print(" 总盈亏:     {:>+,.2f}".format(stats['total_pnl']))
    print(" 平均持仓:   {:.1f}天".format(stats['avg_hold_days']))

    if trades_df is not None and len(trades_df) > 0:
        print("")
        print("=== 按评分分层统计 ===")
        td = trades_df.copy()
        bins = [0, 40, 50, 60, 70, 80, 100]
        labels = ['0-40', '40-50', '50-60', '60-70', '70-80', '80-100']
        td['score_bin'] = pd.cut(td['score'], bins=bins, labels=labels, right=False)
        layer = td.groupby('score_bin', observed=False).agg(
            count=('net_return', 'count'),
            avg_ret=('net_return', 'mean'),
            win_pct=('pnl', lambda x: (x > 0).mean()),
            median_ret=('net_return', 'median'),
        ).round(4)
        layer.columns = ['count', 'avg_ret', 'win_pct', 'median_ret']
        print(layer.to_string())

        print("")
        print("=== 按持仓天数分布 ===")
        hd_bins = [0, 1, 2, 3, 5, 10, 20, 100]
        hd_labels = ['1天', '2天', '3天', '4-5天', '6-10天', '11-20天', '20天+']
        td['hd_bin'] = pd.cut(td['hold_days'], bins=hd_bins, labels=hd_labels, right=True)
        hd_dist = td.groupby('hd_bin', observed=False).agg(
            count=('net_return', 'count'),
            avg_ret=('net_return', 'mean'),
            win_pct=('pnl', lambda x: (x > 0).mean()),
        ).round(4)
        hd_dist.columns = ['count', 'avg_ret', 'win_pct']
        print(hd_dist.to_string())


def main():
    parser = argparse.ArgumentParser(description='回测引擎 v4 - 跌幅超5%止损+共振卖出')
    parser.add_argument('--start', default='2026-06-01', help='开始日期')
    parser.add_argument('--end', default='2026-07-04', help='结束日期')
    parser.add_argument('--top-n', type=int, default=10, help='每日选股数量')
    parser.add_argument('--stop-loss', type=float, default=0.05, help='止损比例')
    parser.add_argument('--resonance', type=float, default=30, help='共振阈值')
    parser.add_argument('--model', default='chase_high', help='选股模型')
    args = parser.parse_args()

    trades_df, daily_df, selections_df = run_backtest(args.start, args.end, top_n=args.top_n, model=args.model,
                                                        stop_loss=args.stop_loss, resonance_threshold=args.resonance)

    if trades_df is not None:
        stats = compute_stats(trades_df, daily_df, PER_STOCK_AMOUNT, args.top_n)
        print_report(stats, trades_df, daily_df, args.start, args.end)

        trades_path = os.path.join(BASE_DIR, 'backtest_trades.csv')
        daily_path = os.path.join(BASE_DIR, 'backtest_portfolio.csv')
        config_path = os.path.join(BASE_DIR, 'backtest_config.json')
        selections_path = os.path.join(BASE_DIR, 'backtest_selections.csv')

        trades_df.to_csv(trades_path, index=False)
        daily_df.to_csv(daily_path, index=False)
        if not selections_df.empty:
            selections_df.to_csv(selections_path, index=False)
        config = {
            'version': 'v4',
            'model': args.model,
            'start_date': args.start,
            'end_date': args.end,
            'top_n': args.top_n,
            'stop_loss': args.stop_loss,
            'resonance_threshold': args.resonance,
            'per_stock_amount': PER_STOCK_AMOUNT,
            'sell_rule': '跌幅超5%止损 / 共振卖出',
            'trading_cost': TRADING_COST,
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print("")
        print("交易明细: {}".format(trades_path))
        print("净值曲线: {}".format(daily_path))
        print("每日选股: {}".format(selections_path))
        print("配置: {}".format(config_path))
    else:
        print("回测失败")


if __name__ == '__main__':
    main()
