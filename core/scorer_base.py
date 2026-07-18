"""
评分模型抽象基类 —— 统一 7 个评分模型的接口规范。

使用方式:
    class MyScorer(BaseScorer):
        model_name = "my_model"
        def compute(self, score_input: ScoreInput) -> dict:
            ...
    
    scorer = get_scorer("chase_high")
    result = scorer.score(code, kline_df, ...)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import pandas as pd
import numpy as np


@dataclass
class ScoreInput:
    """评分模型的统一输入数据容器。

    各模型从该 dataclass 中按需取字段。
    """

    # ── 基础标识 ──
    code: str = ""
    name: str = ""
    sector: str = ""

    # ── K 线数据（所有模型都需要） ──
    kline_df: Optional[pd.DataFrame] = None

    # ── 股票数据字典（由 _build_stock_data 生成） ──
    stock_data: dict = field(default_factory=dict)

    # ── 额外上下文 ──
    stock_pool_context: dict = field(default_factory=dict)

    # ── 共振模型专用 ──
    resonance_data: dict = field(default_factory=dict)
    quotes_df: Optional[pd.DataFrame] = None

    # ── 配置参数 ──
    params: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)


@dataclass
class ScoreResult:
    """评分模型的统一输出数据容器。"""

    score: float = 0.0           # 综合评分（归一化 0~100）
    passed: bool = True           # 是否通过硬过滤
    filter_msg: str = ""          # 未通过时的原因
    position_msg: str = ""        # 仓位建议
    signal: str = ""              # 信号分类
    signal_class: str = ""        # 信号 CSS class
    advice: str = ""              # 操作建议
    dimensions: dict = field(default_factory=dict)   # 各维度评分 {名称: 分数}
    raw: dict = field(default_factory=dict)          # 原始维度分（归一化前）


class BaseScorer(ABC):
    """评分模型抽象基类。

    子类需定义：
        model_name: str      — 模型唯一标识符
        compute(input)       — 核心评分逻辑，返回 ScoreResult

    可选改写：
        prepare(input)       — 评分前的数据预处理
        finalize(input, result) — 评分后的后处理/归一化
    """

    model_name: str = "base"

    @abstractmethod
    def compute(self, score_input: ScoreInput) -> ScoreResult:
        """核心评分逻辑。"""
        ...

    def prepare(self, score_input: ScoreInput) -> ScoreInput:
        """评分前的数据预处理（可选改写）。"""
        return score_input

    def finalize(self, score_input: ScoreInput, result: ScoreResult) -> ScoreResult:
        """评分后的后处理（可选改写）。"""
        return result

    def score(self, **kwargs) -> ScoreResult:
        """对外统一评分入口。

        可接受关键字参数构造 ScoreInput，或直接传入 ScoreInput 实例。
        """
        if "score_input" in kwargs:
            si = kwargs["score_input"]
        else:
            si = ScoreInput(**{k: v for k, v in kwargs.items()
                               if k in ScoreInput.__dataclass_fields__})
        si = self.prepare(si)
        result = self.compute(si)
        result = self.finalize(si, result)
        return result


# ── 工厂 ──

_SCORER_REGISTRY: dict[str, type[BaseScorer]] = {}


def register_scorer(scorer_cls: type[BaseScorer]):
    """注册 Scorer 到全局工厂。"""
    name = scorer_cls.model_name
    _SCORER_REGISTRY[name] = scorer_cls
    return scorer_cls


def get_scorer(model_name: str) -> BaseScorer:
    """根据模型名称获取 Scorer 实例。"""
    if model_name not in _SCORER_REGISTRY:
        raise KeyError(f"Unknown scorer model: {model_name}. "
                       f"Available: {list(_SCORER_REGISTRY.keys())}")
    return _SCORER_REGISTRY[model_name]()


def list_scorers() -> list[str]:
    """列出所有已注册的评分模型名称。"""
    return list(_SCORER_REGISTRY.keys())
