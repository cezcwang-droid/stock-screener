"""Backward-compatible shim — step 1: proxy env cleanup + re-export external deps only.
External callers still do:
    from stock_screener import TDX_AVAILABLE, read_tdx_day_file, ...

New code should import directly from the target module:
    from core.scoring.chase_high import calculate_v3_total_score
    from data.market import read_tdx_day_file
"""
# Proxy env cleanup (must run before any requests)
import os
for _k in list(os.environ.keys()):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from utils.network import _patch_requests_no_proxy, _patch_push2_http_fallback
_patch_requests_no_proxy()
_patch_push2_http_fallback()

# ── Re-export only the names that external files actually use ──
from data.market import read_tdx_day_file
from core.filters import hard_filter_oversold_rebound
from core.scoring.oversold import calculate_oversold_rebound_score
from core.config import TDX_AVAILABLE, TDX_VIPDOC_DIR, BASE_DIR

# backtest/engine.py lazy-imports these internally
from core.scoring.lowbuy import calculate_lowbuy_score, DEFAULT_LOWBUY_WEIGHTS
from core.scoring.golden_cross import calculate_golden_cross_score
from core.scoring.canslim import calculate_canslim_score
from core.scoring.dilemma import calculate_dilemma_reversal_score
from core.scoring.chase_high import calculate_v3_total_score
from core.config import DEFAULT_WEIGHTS

# backtest_engine.py re-exports everything from backtest/engine (kept as a shim)
import backtest_engine

if __name__ == "__main__":
    from app.main import main
    main()
