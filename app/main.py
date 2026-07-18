""""""
import os, json, time, re
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
from app.init_state import init_session_state, StreamlitProgressSink
from utils.network import _patch_requests_no_proxy, _patch_push2_http_fallback
from app.pages import render_screener, render_detail, render_watchlist, render_backtest
from app.components import render_sidebar

# ============ Proxy disable (must run before any requests) ============
for _k in list(os.environ.keys()):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)

# Apply network patches
_patch_requests_no_proxy()
_patch_push2_http_fallback()

# Initialize session state
init_session_state()


def main():
    render_sidebar()
    if st.session_state.selected_stock:
        render_detail(st.session_state.selected_stock)
    elif st.session_state.current_page == "watchlist":
        render_watchlist()
    elif st.session_state.current_page == "backtest":
        render_backtest()
    else:
        render_screener()

if __name__ == "__main__":
    main()
