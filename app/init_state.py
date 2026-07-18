"""Streamlit session state 初始化 —— 在 app 层运行。
替代原来 data/cache.py 的 init_session_state()，剥离 UI 依赖。
"""
import streamlit as st
from datetime import datetime

from core.config import (
    DEFAULT_WEIGHTS, DEFAULT_LOWBUY_PARAMS, DEFAULT_LOWBUY_WEIGHTS,
    DEFAULT_ORB_PARAMS, DEFAULT_ORB_WEIGHTS,
)
from data.cache import load_watchlist, load_cache_data, is_cache_today


def init_session_state():
    """初始化 Streamlit session_state —— 在 app 启动时调用一次。"""
    defaults = dict(DEFAULT_WEIGHTS)
    for key in ['weights', 'watchlist', 'selected_stock', 'current_page',
                'backtest_result', 'bt_params', 'top10_cache', 'last_update_time',
                'data_status', 'raw_stock_data', 'current_model', 'lowbuy_params',
                'lowbuy_cache', 'lowbuy_auto_scanned', 'top10_cache_key']:
        if key not in st.session_state:
            if key == 'watchlist':
                st.session_state[key] = load_watchlist()
            elif key == 'weights':
                st.session_state[key] = defaults
            elif key == 'current_page':
                st.session_state[key] = 'screener'
            elif key == 'current_model':
                st.session_state[key] = 'chase_high'
            elif key == 'lowbuy_params':
                st.session_state[key] = dict(DEFAULT_LOWBUY_PARAMS)
            elif key == 'lowbuy_weights':
                st.session_state[key] = dict(DEFAULT_LOWBUY_WEIGHTS)
            elif key in ('top10_cache', 'lowbuy_cache', 'last_update_time'):
                st.session_state[key] = None
            elif key == 'data_status':
                st.session_state[key] = 'normal'
            elif key == 'lowbuy_auto_scanned':
                st.session_state[key] = False
            else:
                st.session_state[key] = None

    # 超跌反弹模型参数初始化
    if 'orb_params' not in st.session_state:
        st.session_state.orb_params = dict(DEFAULT_ORB_PARAMS)
    if 'orb_weights' not in st.session_state:
        st.session_state.orb_weights = dict(DEFAULT_ORB_WEIGHTS)
    if 'orb_results' not in st.session_state:
        st.session_state.orb_results = None
    if 'oversold_rebound_auto_scanned' not in st.session_state:
        st.session_state.oversold_rebound_auto_scanned = False

    # 从文件加载缓存（启动时恢复上次选股结果）
    if 'cache_loaded' not in st.session_state:
        cache_data = load_cache_data()
        if cache_data and is_cache_today(cache_data):
            if cache_data.get('chase_high_top10'):
                st.session_state.top10_cache = cache_data['chase_high_top10']
                st.session_state.top10_cache_key = f"top10_{datetime.now().strftime('%Y%m%d')}"
            if cache_data.get('lowbuy_top5'):
                st.session_state.lowbuy_cache = cache_data['lowbuy_top5']
            st.session_state._lb_dbg = cache_data.get('lowbuy_dbg') or {}
            st.session_state.last_update_time = datetime.fromisoformat(cache_data['timestamp'])
            st.session_state.data_status = 'cached'
            cached_lb_ver = cache_data.get('lowbuy_params_version', 0)
            current_lb_ver = DEFAULT_LOWBUY_PARAMS.get('_params_version', 0)
            if cached_lb_ver < current_lb_ver:
                st.session_state.lowbuy_cache = None
                st.session_state.last_update_time = None
                st.session_state._lb_dbg = {}
                st.session_state.data_status = 'normal'
        st.session_state.cache_loaded = True
        if '_lb_dbg' not in st.session_state:
            st.session_state._lb_dbg = {}


class StreamlitProgressSink:
    """ProgressSink 的 Streamlit 实现 —— 将回调映射到 Streamlit 组件。"""

    def __init__(self):
        self._progress_bar = None
        self._status_text = None

    def on_progress(self, current: int, total: int, msg: str = ""):
        import streamlit as st
        if self._progress_bar is None:
            self._progress_bar = st.progress(0)
            self._status_text = st.empty()
        self._progress_bar.progress(min(current / total, 1.0) if total > 0 else 0)
        if msg:
            self._status_text.text(msg)
        if current >= total:
            self._progress_bar.empty()
            self._status_text.empty()
            self._progress_bar = None
            self._status_text = None

    def on_warning(self, msg: str):
        import streamlit as st
        st.warning(msg)

    def on_error(self, msg: str):
        import streamlit as st
        st.error(msg)

    def on_info(self, msg: str):
        import streamlit as st
        st.info(msg)

    def on_status(self, msg: str):
        import streamlit as st
        if msg:
            st.markdown(msg)
