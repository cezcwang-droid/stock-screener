---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 80bc8ebecb165bb4bb346dbe5308e6ae_a224420f754b11f1aabe5254007bceed
    ReservedCode1: 0bJUvWExWVjwH5RO8TtNPamH90nsNuH5qnCp9ZyZphJwRBqr0Hadz0RybtWeVu9vGDnGoJri+cINw1VuMEwGYO0U8DMkn/0PTZrJPcDwEEZsQP9fAYTXAOZ7S/wiGe1mA0KuBbQpoceAqw1YdhQutY4T4svajFBlR9h+3iZFbXLp2saR/0pq1T/eC9o=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 80bc8ebecb165bb4bb346dbe5308e6ae_a224420f754b11f1aabe5254007bceed
    ReservedCode2: 0bJUvWExWVjwH5RO8TtNPamH90nsNuH5qnCp9ZyZphJwRBqr0Hadz0RybtWeVu9vGDnGoJri+cINw1VuMEwGYO0U8DMkn/0PTZrJPcDwEEZsQP9fAYTXAOZ7S/wiGe1mA0KuBbQpoceAqw1YdhQutY4T4svajFBlR9h+3iZFbXLp2saR/0pq1T/eC9o=
---

# 变更日志

## 2026-07-01

### 策略回测：新增低吸模型支持

**需求**：原策略回测仅支持追高模型，需要让用户可在回测页面选择低吸模型进行历史回测。

**涉及文件**：`stock_screener.py`、`backtest_engine.py`

#### stock_screener.py

1. `calculate_lowbuy_score` — 签名增加 `weights=None` 参数，使回测引擎可显式传入权重，避免依赖 `st.session_state`
2. `run_real_backtest_cached` ×3 副本 — 统一增加 `model='chase_high'` 参数，透传给 `backtest_engine.run_backtest`
3. `render_backtest` — 新增追高/低吸模型切换 radio 组件；权重展示区根据模型动态切换 `WEIGHT_CONFIG` / `LOWBUY_WEIGHT_CONFIG`；`run_real_backtest_cached` 调用传入 `model=model_key`

#### backtest_engine.py

1. 新增 `score_stock_from_kline_lowbuy` — 调用 `calculate_lowbuy_score` 的低吸六维评分函数
2. 新增 `score_all_stocks_lowbuy` — 低吸模型的全股票打分函数
3. `run_backtest` — 签名增加 `model='chase_high'`；新增 `_score_func` 分发逻辑；打印信息动态显示模型名称；3 处 `score_all_stocks` 调用统一改为 `_score_func`

**回退兼容**：`model` 默认值为 `'chase_high'`，未传入时完全走原有追高路径，不影响现有功能。
*（内容由AI生成，仅供参考）*
