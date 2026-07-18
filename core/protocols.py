"""
UI 交互协议层 —— core/data 层通过此协议与 UI 框架解耦。

core/data 层只调用 ProgressSink 的回调方法，不直接引用任何 UI 框架（streamlit 等）。
UI 层（app/）实现 StreamlitProgressSink，将回调映射到 streamlit 组件。
"""
from typing import Protocol, Optional, Any


class ProgressSink(Protocol):
    """core/data 层通知 UI 进度的唯一通道。

    core/data 层在函数签名中使用 `sink: Optional[ProgressSink] = None`，
    默认使用 NullSink()，使 core 层可以脱离 UI 独立运行。
    """

    def on_progress(self, current: int, total: int, msg: str = "") -> None:
        """更新进度。current=total 表示完成。"""

    def on_warning(self, msg: str) -> None:
        """显示警告消息。"""

    def on_error(self, msg: str) -> None:
        """显示错误消息。"""

    def on_info(self, msg: str) -> None:
        """显示信息消息。"""

    def on_status(self, msg: str) -> None:
        """设置状态文本。"""


class NullSink:
    """默认空实现 —— core 层独立运行时使用，所有方法不执行任何操作。"""

    def on_progress(self, current: int, total: int, msg: str = "") -> None:
        pass

    def on_warning(self, msg: str) -> None:
        pass

    def on_error(self, msg: str) -> None:
        pass

    def on_info(self, msg: str) -> None:
        pass

    def on_status(self, msg: str) -> None:
        pass


# ── 简单应用状态传递（替代 st.session_state） ──

class AppState:
    """纯数据容器，用于在 core/data 层之间传递状态。

    UI 层（app/）负责将 streamlit session_state 映射到此对象，
    并回写变化。
    """

    def __init__(self):
        self.data_status: str = "normal"
        self.last_update_time: Any = None
        self.raw_stock_data: Any = None
        self.current_model: str = "chase_high"
        self.chase_sample_source: str = "全市场A股"
        self.weights: dict = {}
        self.lowbuy_weights: dict = {}
        self.lowbuy_params: dict = {}
        self.top10_cache: Any = None
        self.top10_cache_key: str = ""
        self._chase_dynamic_n: int = 0
        self.lowbuy_cache: Any = None
        self._lb_dbg: dict = {}
        self.selected_stock: Any = None
        self.current_page: str = "screener"
        self.gc_params: dict = {}
        self.gc_weights: dict = {}
        self.resonance_results: list = []
        self.orb_params: dict = {}
        self.orb_weights: dict = {}
        self.orb_results: Any = None
        self.oversold_rebound_auto_scanned: bool = False
        self.cache_loaded: bool = False
        self.raw_market_data: Any = None
        self.raw_market_time: float = 0
