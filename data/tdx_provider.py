
"""
通达信 pytdx 统一数据提供模块
替代 akshare / 东方财富 HTTP 请求，从通达信服务器获取行情数据

数据获取策略：
- 行情快照(全市场): pytdx get_security_quotes 批量获取 OHLCV
- K线数据: 优先 TDX 本地 .day 文件，兜底 pytdx get_security_bars
- 板块数据: pytdx get_and_parse_block_info
- 龙虎榜: 保留 akshare（pytdx 不支持）
- 财务指标(PE/PB等): 从 pytdx quotes 衍生计算 + get_finance_info 补充
"""
import os
import time
import struct
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import requests as _requests_global

import pandas as pd
import numpy as np

# ==================== pytdx 连接管理 ====================

TDX_SERVERS = [
    ('125.39.80.98', 7709),
    ('120.76.152.87', 7709),
    ('119.147.212.81', 7709),
]

# 模块级连接（懒加载，自动重连）
_tdx_api = None
_tdx_connected = False

_MARKET_SZ = 0  # 深圳
_MARKET_SH = 1  # 上海


def _get_api():
    """获取或创建 pytdx 连接（自动重连）"""
    global _tdx_api, _tdx_connected
    if _tdx_connected and _tdx_api is not None:
        return _tdx_api

    from pytdx.hq import TdxHq_API
    _tdx_api = TdxHq_API()
    for ip, port in TDX_SERVERS:
        try:
            if _tdx_api.connect(ip, port):
                _tdx_connected = True
                return _tdx_api
        except Exception:
            continue
    _tdx_connected = False
    return None


def tdx_available():
    """检查 pytdx 是否可用且能连接通达信服务器"""
    try:
        api = _get_api()
        return api is not None
    except Exception:
        return False


# ==================== 股票列表获取 ====================

# 上海A股主板代码范围
_SH_MAIN_RANGES = [
    range(600000, 605000),   # 沪市主板
    range(688000, 689001),   # 科创板
]

# 模块级缓存：上海A股代码列表（通过腾讯API发现）
_sh_codes_cache = None
_sh_name_map_cache = None

def _discover_sh_codes() -> List[str]:
    """通过腾讯微证券API批量查询上海代码范围，返回有效A股代码列表。
    结果在模块级缓存，进程生命周期内只查询一次。
    """
    global _sh_codes_cache, _sh_name_map_cache
    if _sh_codes_cache is not None:
        return _sh_codes_cache

    valid = []
    name_map = {}
    BATCH = 150
    all_ranges = []
    for rng in _SH_MAIN_RANGES:
        all_ranges.extend(list(rng))

    for i in range(0, len(all_ranges), BATCH):
        batch = all_ranges[i:i + BATCH]
        query = ','.join(f'sh{c}' for c in batch)
        try:
            resp = _requests_global.get(f'http://qt.gtimg.cn/q={query}', timeout=15,
                headers={'User-Agent': 'Mozilla/5.0'})
            for line in resp.text.strip().split('\n'):
                if '=\"' in line and '~' in line:
                    parts = line.split('\"')[1].split('~')
                    if len(parts) >= 3 and parts[2]:
                        code = parts[2]
                        name = parts[1] if len(parts) > 1 else code
                        if (code.startswith(('60', '688'))
                                and code not in valid
                                and len(code) == 6):
                            valid.append(code)
                            name_map[code] = name
        except Exception:
            continue

    _sh_codes_cache = valid
    _sh_name_map_cache = name_map
    print(f"[TDX] 通过腾讯API发现 {len(valid)} 只上海A股（含名称映射）")
    return valid


def get_all_stock_codes() -> List[Tuple[int, str]]:
    """获取全市场A股代码列表 [(market, code), ...]
    深圳：通过 pytdx get_security_list 获取（含债券等）
    上海：通过腾讯微证券 API 发现有效代码，再经 pytdx 获取行情
    """
    api = _get_api()
    if not api:
        return []

    all_codes = []

    # 深圳：get_security_list 直接返回 A 股+债券+ETF+国债
    for start in range(0, 50000, 1000):
        try:
            batch = api.get_security_list(_MARKET_SZ, start)
            if not batch:
                break
            for item in batch:
                code = str(item.get('code', ''))
                if len(code) == 6 and code[:3] in ('000', '001', '002', '003', '004', '300', '301'):
                    all_codes.append((_MARKET_SZ, code))
        except Exception:
            break

    # 上海：通过腾讯API发现有效A股代码（pytdx 对无效代码整批返回None）
    sh_codes = _discover_sh_codes()
    for code in sh_codes:
        all_codes.append((_MARKET_SH, code))

    return all_codes


def resolve_market(code: str) -> int:
    """根据代码判断市场（0=深圳, 1=上海）"""
    code = str(code).zfill(6)
    if code.startswith(('6', '9')):
        return _MARKET_SH
    return _MARKET_SZ


# ==================== 行情快照（批量） ====================

def _map_quote_fields(quote: dict, code: str) -> dict:
    """将 pytdx quote 字段映射为与 eastmoney/akshare 兼容的字段名"""
    price = float(quote.get('price', 0) or 0)
    last_close = float(quote.get('last_close', 0) or 0)
    open_p = float(quote.get('open', 0) or 0)
    high = float(quote.get('high', 0) or 0)
    low = float(quote.get('low', 0) or 0)
    vol = float(quote.get('vol', 0) or 0)
    amount = float(quote.get('amount', 0) or 0)

    # 计算衍生字段
    change_pct = round((price - last_close) / last_close * 100, 2) if last_close > 0 else 0
    change_amt = round(price - last_close, 2)
    amplitude = round((high - low) / last_close * 100, 2) if last_close > 0 else 0

    return {
        '代码': code,
        '名称': '',
        '最新价': price,
        '涨跌幅': change_pct,
        '涨跌额': change_amt,
        '今开': open_p,
        '最高': high,
        '最低': low,
        '昨收': last_close,
        '成交量': vol,
        '成交额': amount,
        '振幅': amplitude,
    }


def fetch_all_quotes_tdx() -> Optional[pd.DataFrame]:
    """通过 pytdx 批量获取全市场A股实时行情快照"""
    api = _get_api()
    if not api:
        return None

    codes = get_all_stock_codes()
    if not codes:
        print("[TDX] 获取股票列表失败")
        return None

    # 构建代码->名称映射
    name_map = {}
    for market in (_MARKET_SZ, _MARKET_SH):
        for start in range(0, 50000, 1000):
            try:
                batch = api.get_security_list(market, start)
                if not batch:
                    break
                for item in batch:
                    c = str(item.get('code', ''))
                    if len(c) == 6:
                        if market == _MARKET_SZ and c[:3] in ('000', '001', '002', '003', '004', '300', '301'):
                            name_map[c] = str(item.get('name', c))
                        elif market == _MARKET_SH and c.startswith('6'):
                            name_map[c] = str(item.get('name', c))
            except Exception:
                break

    # 分批获取行情
    BATCH_SIZE = 75
    all_rows = []

    for i in range(0, len(codes), BATCH_SIZE):
        batch = codes[i:i + BATCH_SIZE]
        try:
            quotes = api.get_security_quotes(batch)
            if quotes:
                for q in quotes:
                    if q is None:
                        continue
                    code = str(q.get('code', ''))
                    if not code or len(code) != 6:
                        continue
                    row = _map_quote_fields(q, code)
                    row['名称'] = name_map.get(code, code)
                    all_rows.append(row)
        except Exception as e:
            print(f"[TDX] 行情批次 {i} 失败: {e}")
            continue

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)

    # 补充缺失列
    df['量比'] = np.nan
    df['换手率'] = np.nan
    df['市盈率-动态'] = np.nan
    df['市净率'] = np.nan
    df['总市值'] = np.nan
    df['流通市值'] = np.nan
    df['60日涨跌幅'] = 0.0
    df['年初至今涨跌幅'] = np.nan
    df['主力净流入'] = np.nan
    df['涨跌幅_20d'] = 0.0

    return df


# ==================== 代码→名称映射（全量） ====================

_name_map_cache = None

def get_full_name_map() -> Dict[str, str]:
    """获取全量代码→名称映射（多源合并）。
    深圳：pytdx get_security_list
    上海：腾讯微证券API（_discover_sh_codes 副产物）
    结果在模块级缓存，进程生命周期内只查询一次。
    """
    global _name_map_cache
    if _name_map_cache is not None:
        return _name_map_cache

    name_map = {}
    api = _get_api()

    # 源1: 深圳市场 - pytdx get_security_list
    if api:
        for start in range(0, 50000, 1000):
            try:
                batch = api.get_security_list(_MARKET_SZ, start)
                if not batch:
                    break
                for item in batch:
                    c = str(item.get('code', ''))
                    if len(c) == 6 and not c.startswith('39'):
                        name_map[c] = str(item.get('name', c))
            except Exception:
                break

    # 源1兜底: 深圳市场 - pytdx 连接失败时，扫描本地 .day 文件提取深市代码
    sz_count = sum(1 for c in name_map if c.startswith(('0', '3')))
    if sz_count == 0:
        vipdoc_dir = find_tdx_vipdoc_dir()
        if vipdoc_dir:
            sz_lday = os.path.join(vipdoc_dir, "sz", "lday")
            if os.path.isdir(sz_lday):
                for f in os.listdir(sz_lday):
                    if f.endswith('.day') and len(f) == 12:
                        code = f[2:8]
                        if code[:1] in ('0', '3', '2') and code not in name_map:
                            name_map[code] = code
                sz_fallback = sum(1 for c in name_map if c.startswith(('0', '3')))
                if sz_fallback > 0:
                    print(f"[TDX] pytdx 深圳连接失败，从本地 .day 文件兜底发现 {sz_fallback} 只深市股票（无名称，使用代码占位）")

    # 源2: 上海市场 - 腾讯API（_discover_sh_codes 已缓存名称映射）
    try:
        _discover_sh_codes()
        if _sh_name_map_cache:
            for c, n in _sh_name_map_cache.items():
                if c not in name_map:
                    name_map[c] = n
    except Exception:
        pass

    _name_map_cache = name_map
    print(f"[TDX] 全量名称映射: {len(name_map)} 条 (SZ:{sum(1 for c in name_map if c.startswith('0') or c.startswith('3'))} SH:{sum(1 for c in name_map if c.startswith('6'))})")
    return _name_map_cache


# ==================== K线数据 ====================

def fetch_kline_tdx(code: str, days: int = 100) -> Optional[pd.DataFrame]:
    """通过 pytdx 获取个股日K线数据"""
    api = _get_api()
    if not api:
        return None

    code = str(code).zfill(6)
    market = resolve_market(code)

    try:
        bars = api.get_security_bars(9, market, code, 0, days + 10)
        if not bars:
            return None

        rows = []
        for bar in bars:
            dt_str = bar.get('datetime', '')
            if not dt_str:
                continue
            rows.append({
                '日期': pd.Timestamp(dt_str),
                '开盘': float(bar.get('open', 0) or 0),
                '最高': float(bar.get('high', 0) or 0),
                '最低': float(bar.get('low', 0) or 0),
                '收盘': float(bar.get('close', 0) or 0),
                '成交量': float(bar.get('vol', 0) or 0),
                '成交额': float(bar.get('amount', 0) or 0),
            })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df = df.sort_values('日期').reset_index(drop=True)
        return df.tail(days)
    except Exception as e:
        print(f"[TDX] K线获取失败 {code}: {e}")
        return None


def get_today_quote_single(code: str) -> Optional[dict]:
    """获取单只股票当日实时行情（用于拼接到日K线末尾）"""
    api = _get_api()
    if not api:
        return None

    code = str(code).zfill(6)
    market = resolve_market(code)

    try:
        quotes = api.get_security_quotes([(market, code)])
        if not quotes or quotes[0] is None:
            return None
        return _map_quote_fields(quotes[0], code)
    except Exception as e:
        print(f"[TDX] 单股行情获取失败 {code}: {e}")
        return None


# ==================== 板块数据 ====================

def fetch_sector_data_tdx() -> Tuple[Dict, Dict]:
    """
    通过 pytdx 获取板块/行业数据
    返回: (sector_board: {板块名: {daily_gain, limit_up_count, net_inflow}},
          stock_sector_map: {code: sector_name})
    """
    api = _get_api()
    if not api:
        return {}, {}

    sector_board = {}
    stock_sector_map = {}

    block_files = [
        ('block_gn.dat', '概念'),
        ('block_fg.dat', '风格'),
        ('block_zs.dat', '指数'),
    ]

    for block_file, block_type in block_files:
        try:
            blocks = api.get_and_parse_block_info(block_file)
            if not blocks:
                continue

            # blocks is list of OrderedDict: {blockname, block_type, code_index, code}
            block_groups = defaultdict(list)
            for item in blocks:
                bn = str(item.get('blockname', ''))
                cd = str(item.get('code', ''))
                if bn and cd:
                    block_groups[bn].append(cd)

            for bn, codes in block_groups.items():
                sector_board[bn] = {
                    'daily_gain': 0.0,
                    'limit_up_count': 0,
                    'net_inflow': 0.0,
                }
                for c in codes:
                    c = str(c).zfill(6)
                    if len(c) == 6 and c not in stock_sector_map:
                        stock_sector_map[c] = bn
        except Exception as e:
            print(f"[TDX] 板块数据获取失败 ({block_file}): {e}")

    return sector_board, stock_sector_map


# ==================== TDX 本地 .day 文件 ====================

TDX_COMMON_VIPDOC_PATHS = [
    r"C:\zd_cjzq\vipdoc",
    r"C:\new_tdx\vipdoc",
    r"C:\new_xczq\vipdoc",
    r"C:\tdx\vipdoc",
    r"D:\new_tdx\vipdoc",
    r"D:\tdx\vipdoc",
]


def find_tdx_vipdoc_dir():
    """自动查找通达信 vipdoc 目录"""
    for p in TDX_COMMON_VIPDOC_PATHS:
        if os.path.isdir(p):
            if os.path.isdir(os.path.join(p, "sh", "lday")) or os.path.isdir(os.path.join(p, "sz", "lday")):
                return p
    return None


def read_tdx_day_file(code: str, vipdoc_dir: str = None) -> Optional[pd.DataFrame]:
    """读取通达信本地日线 .day 文件"""
    if vipdoc_dir is None:
        vipdoc_dir = find_tdx_vipdoc_dir()
    if not vipdoc_dir:
        return None

    code = str(code).zfill(6)
    if code.startswith(('6', '9')):
        filepath = os.path.join(vipdoc_dir, "sh", "lday", f"sh{code}.day")
    elif code.startswith(('0', '3', '2')):
        filepath = os.path.join(vipdoc_dir, "sz", "lday", f"sz{code}.day")
    elif code.startswith(('4', '8')):
        filepath = os.path.join(vipdoc_dir, "bj", "lday", f"bj{code}.day")
    else:
        return None

    if not os.path.exists(filepath):
        return None

    record_size = 32
    data = []
    try:
        with open(filepath, 'rb') as f:
            while True:
                raw = f.read(record_size)
                if len(raw) < record_size:
                    break
                date_int, open_i, high_i, low_i, close_i, amount, volume, _ = struct.unpack(
                    '<IIIIIfII', raw[:32])
                open_p = open_i / 100.0
                high_p = high_i / 100.0
                low_p = low_i / 100.0
                close_p = close_i / 100.0
                if date_int < 19900101 or date_int > 20991231:
                    continue
                if close_p <= 0 or close_p > 100000:
                    continue
                data.append({
                    '日期': pd.Timestamp(str(date_int)),
                    '开盘': open_p,
                    '最高': high_p,
                    '最低': low_p,
                    '收盘': close_p,
                    '成交量': volume,
                    '成交额': amount,
                })
    except Exception:
        return None

    if not data:
        return None
    df = pd.DataFrame(data)
    df = df.sort_values('日期').reset_index(drop=True)
    return df


# ==================== 组合接口 ====================

def get_kline(code: str, days: int = 60, vipdoc_dir: str = None) -> Optional[pd.DataFrame]:
    """获取个股K线：优先 TDX 本地 .day，兜底 pytdx get_security_bars"""
    code = str(code).zfill(6)

    df = read_tdx_day_file(code, vipdoc_dir)
    if df is not None and len(df) > 0:
        return df.tail(days)

    return fetch_kline_tdx(code, days)


def get_kline_with_today(code: str, days: int = 60, vipdoc_dir: str = None):
    """获取K线并在交易时段附加当日实时K线"""
    df = get_kline(code, days, vipdoc_dir)
    if df is None or len(df) == 0:
        return df, False

    now = datetime.now()
    if now.weekday() >= 5:
        return df, False
    hour_min = now.hour * 100 + now.minute
    if hour_min < 915 or hour_min > 1505:
        return df, False

    last_date_str = str(df.iloc[-1].get('日期', ''))
    today_str = now.strftime('%Y-%m-%d')
    today_str2 = now.strftime('%Y%m%d')
    if today_str in last_date_str or today_str2 in last_date_str:
        return df, False

    try:
        today_quote = get_today_quote_single(code)
        if today_quote is None:
            return df, False
        today_close = today_quote.get('最新价', 0)
        today_open = today_quote.get('今开', 0)
        if today_close <= 0:
            return df, False
        today_row = {
            '日期': today_str,
            '开盘': today_open,
            '最高': today_quote.get('最高', today_close),
            '最低': today_quote.get('最低', today_close),
            '收盘': today_close,
            '成交量': today_quote.get('成交量', 0),
            '成交额': today_quote.get('成交额', 0),
        }
        new_row = pd.DataFrame([today_row])
        df = pd.concat([df, new_row], ignore_index=True)
        return df, True
    except Exception as e:
        print(f"[TDX] 附加当日K线失败 {code}: {e}")
        return df, False
