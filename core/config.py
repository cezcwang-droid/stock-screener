"""Global configuration: paths, weight configs, constants, caches."""
import os, json, time

# ════════════════════════════════════════════════════════════════
#  Verbosity
# ════════════════════════════════════════════════════════════════
_VERBOSE = False

# ════════════════════════════════════════════════════════════════
#  Path configuration
# ════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCHLIST_JSON = os.path.join(BASE_DIR, "watchlist.json")
CACHE_DATA_JSON = os.path.join(BASE_DIR, "cache_data.json")
CACHE_DIR = os.path.join(BASE_DIR, "resonance_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
#  Optional dependencies
# ════════════════════════════════════════════════════════════════
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    import pytdx
    TDX_PYTORCH_AVAILABLE = True
except ImportError:
    TDX_PYTORCH_AVAILABLE = False

# ════════════════════════════════════════════════════════════════
#  TDX path configuration
# ════════════════════════════════════════════════════════════════
TDX_COMMON_PATHS = [
    os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Roaming", "TDX", "vipdoc"),
    "C:\\(new_tdx)\\vipdoc",
    "D:\\new_tdx\\vipdoc",
    "D:\\通达信\\vipdoc",
    "C:\\通达信\\vipdoc",
    "D:\\new_zd\\vipdoc",
]

def find_tdx_vipdoc():
    """Find TDX vipdoc directory."""
    import struct
    for p in TDX_COMMON_PATHS:
        if os.path.isdir(p):
            return p
    # Search common drives
    for drive in ["C", "D", "E", "F"]:
        for name in ["new_tdx", "通达信", "new_zd", "zd_zszq", "tdx"]:
            p = os.path.join(drive + ":", name, "vipdoc")
            if os.path.isdir(p):
                return p
    return None

TDX_VIPDOC_DIR = find_tdx_vipdoc()
TDX_AVAILABLE = TDX_VIPDOC_DIR is not None

# ════════════════════════════════════════════════════════════════
#  DDE data paths
# ════════════════════════════════════════════════════════════════
DDE_DATA_FILE = os.path.join(BASE_DIR, "DDE_data", "Table.xlsx")
_SINA_MONEYFLOW_URL = (
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
    "?_s_a_a_a=MoneyFlow&page=1&num=3000&sort=netamount&asc=0&bankuai=hs_a"
)
_SINA_PREFIX_MAP = {"sh": "", "sz": "", "bj": ""}

# ════════════════════════════════════════════════════════════════
#  Weight configurations — 追高模型 (v3八维)
# ════════════════════════════════════════════════════════════════
WEIGHT_CONFIG = {
    "趋势结构": {"desc": "MA5/10/20/60多头排列", "default": 15, "max": 35, "icon": "📐", "color": "#1A73E8", "full_score": 25},
    "动量强度": {"desc": "5日+10日涨幅",        "default": 18, "max": 35, "icon": "🚀", "color": "#E65100", "full_score": 22},
    "板块共振": {"desc": "板块涨幅/涨停/资金流",  "default": 8, "max": 25, "icon": "🌐", "color": "#00897B", "full_score": 10},
    "北向资金": {"desc": "北向资金近3日净买",     "default": 15, "max": 25, "icon": "👆", "color": "#1565C0", "full_score": 10},
    "机构净买": {"desc": "机构3日净买额",         "default": 10, "max": 25, "icon": "🏚", "color": "#C62828", "full_score": 10},
    "板块资金热度": {"desc": "板块资金排名映射",   "default": 5, "max": 15, "icon": "🔥", "color": "#E65100", "full_score": 10},
    "量价配合": {"desc": "量比/振幅/缩量新高",    "default": 14, "max": 30, "icon": "📈", "color": "#F57C00", "full_score": 12},
    "估值安全": {"desc": "PE历史分位(赛道差异化)","default": 3,  "max": 25, "icon": "🛡", "color": "#7B1FA2", "full_score": 10},
    "筹码稳定": {"desc": "换手率接力",            "default": 6,  "max": 20, "icon": "🔒", "color": "#2E7D32", "full_score": 7},
    "情绪热度": {"desc": "热度分(主线/冷门差异化)","default": 6,  "max": 15, "icon": "🎯", "color": "#AD1457", "full_score": 4},
}
DEFAULT_WEIGHTS = {k: v["default"] for k, v in WEIGHT_CONFIG.items()}

# 赛道分类
GROWTH_SECTOR = {"半导体", "AI算力", "光模块", "EDA", "存储芯片", "云计算",
                 "半导体/芯片", "AI/人工智能", "软件/云计算", "芯片", "算力", "数据"}
CYCLE_SECTOR = {"锂电设备", "物流", "影视传媒", "电力设备", "CXO医药",
                "新能源/锂电", "化工/材料", "机械设备"}

# 仓位分层
SCORE_TIERS = [
    (85, 0.7, "💰 核心龙头·重仓追高", "主线龙头，重仓60%~80%"),
    (70, 0.3, "📈 支线趋势·轻仓试错", "支线趋势，轻仓20%~40%"),
    (0,  0.0, "⚠️ 不建议参与",       "分数不足70，放弃不参与"),
]

# ════════════════════════════════════════════════════════════════
#  低吸模型参数
# ════════════════════════════════════════════════════════════════
DEFAULT_LOWBUY_PARAMS = {
    "_params_version": 5,
    "max_results": 20,
    "pre_filter_decline": 5,
    "decline_20d_low": -40,
    "decline_20d_high": 0,
    "max_vol_ratio": 2.5,
    "no_new_low_days": 2,
    "reversal_bottom_pct": 0.2,
    "reversal_require_uptrend": False,
    "min_decline_depth": 5,
    "min_stabilization": 8,
    "min_volume_recovery": 5,
    "min_ma_support": 4,
    "min_valuation_attr": 3,
    "min_chip_settle": 3,
    "min_fund_flow": 2,
    "min_total_score": 35,
    "fund_weight": 0.08,
}

LOWBUY_WEIGHT_CONFIG = {
    "下跌幅度": {"default": 23, "max": 40, "color": "#E74C3C"},
    "企稳信号": {"default": 18, "max": 35, "color": "#F39C12"},
    "量能恢复": {"default": 14, "max": 30, "color": "#3498DB"},
    "均线支撑": {"default": 14, "max": 30, "color": "#2ECC71"},
    "估值吸引": {"default": 14, "max": 30, "color": "#9B59B6"},
    "筹码沉淀": {"default":  9, "max": 25, "color": "#1ABC9C"},
    "主力资金": {"default":  8, "max": 20, "color": "#E67E22"},
}
DEFAULT_LOWBUY_WEIGHTS = {k: v["default"] for k, v in LOWBUY_WEIGHT_CONFIG.items()}

# ════════════════════════════════════════════════════════════════
#  超跌反弹模型
# ════════════════════════════════════════════════════════════════
ORB_WEIGHT_CONFIG = {
    "空间维度": {"default": 40, "max": 60, "color": "#E74C3C"},
    "情绪量能": {"default": 30, "max": 50, "color": "#3498DB"},
    "择时确认": {"default": 30, "max": 50, "color": "#F39C12"},
    "板块加成": {"default": 10, "max": 20, "color": "#27AE60"},
}
DEFAULT_ORB_WEIGHTS = {k: v["default"] for k, v in ORB_WEIGHT_CONFIG.items()}
DEFAULT_ORB_PARAMS = {
    "min_drawdown_pct": 30,
    "min_decline_20d": 15,
    "min_price": 3.0,
    "vol_shrink_threshold": 0.5,
}

# ════════════════════════════════════════════════════════════════
#  金叉模型
# ════════════════════════════════════════════════════════════════
DEFAULT_GC_PARAMS = {
    "main_rise_pct": 20,
    "decline_pct": 15,
    "volume_ratio": 1.2,
    "steady_days": 2,
    "sample_source": "全市场A股",
    "fund_weight": 0.10,
}
DEFAULT_GC_SAMPLE_OPTIONS = ["全市场A股", "热门板块（资金加速Top6）", "板块反转（量价+资金）"]
DEFAULT_GC_WEIGHTS = {
    "下跌形态": 28,
    "K线止跌": 18,
    "均线拐头": 18,
    "量能确认": 13,
    "MACD反转": 13,
    "资金确认": 10,
}

# ════════════════════════════════════════════════════════════════
#  Module-level caches
# ════════════════════════════════════════════════════════════════
_app_cache = {
    "raw_market_data": None,
    "raw_market_time": None,
    "dragon_tiger": None,
    "dragon_tiger_time": None,
    "sector_data": None,
    "sector_map": None,
    "sector_time": None,
    "sector_fund_flow": None,
    "sector_fund_flow_time": None,
    "stock_pool": None,
    "stock_pool_time": None,
    "stock_pool_key": None,
}

_industry_pe_cache = {"stats": None, "timestamp": 0}

# Export ALL names (including underscore-prefixed) for `from core.config import *`
__all__ = [
    # Verbosity
    "_VERBOSE",
    # Paths
    "BASE_DIR", "WATCHLIST_JSON", "CACHE_DATA_JSON", "CACHE_DIR",
    # Optional deps
    "AKSHARE_AVAILABLE", "TDX_PYTORCH_AVAILABLE",
    # TDX
    "TDX_COMMON_PATHS", "TDX_VIPDOC_DIR", "TDX_AVAILABLE", "find_tdx_vipdoc",
    # DDE
    "DDE_DATA_FILE", "_SINA_MONEYFLOW_URL", "_SINA_PREFIX_MAP",
    # Chase high weights
    "WEIGHT_CONFIG", "DEFAULT_WEIGHTS", "GROWTH_SECTOR", "CYCLE_SECTOR", "SCORE_TIERS",
    # Lowbuy
    "DEFAULT_LOWBUY_PARAMS", "LOWBUY_WEIGHT_CONFIG", "DEFAULT_LOWBUY_WEIGHTS",
    # Oversold rebound
    "ORB_WEIGHT_CONFIG", "DEFAULT_ORB_WEIGHTS", "DEFAULT_ORB_PARAMS",
    # Golden cross
    "DEFAULT_GC_PARAMS", "DEFAULT_GC_SAMPLE_OPTIONS", "DEFAULT_GC_WEIGHTS",
    # Caches
    "_app_cache", "_industry_pe_cache",
]
