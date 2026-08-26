# Benchmarks / 记忆系统评估

本目录是 Grace 记忆系统的第三方基准评估（2026-08-26，全程沙盒隔离运行，未触碰正式记忆库）。

## LongMemEval（腾讯 AI Lab / UCLA，ICLR 2025）

[LongMemEval](https://github.com/xiaowu0162/LongMemEval)（arXiv:2410.10813）是长对话记忆基准，
500 题、6 类能力（单会话-用户 / 多会话推理 / 时间推理 / 知识更新 / 单会话-偏好 / 单会话-助手）。
数据经 `HF_ENDPOINT=https://hf-mirror.com` 下载 `xiaowu0162/longmemeval-cleaned`。

### 评估阶梯（同一 500 题，四种设定）

| 设定 | 检索 R@5 | 说明 |
|---|---|---|
| 全量混合库（500 题共享一个 L2 库） | 0.25 | 地狱难度：干扰放大 ~100 倍，非官方口径 |
| 纯余弦 per-question（组件上限） | 0.93 | bge-m3 嵌入 + 裸余弦，无检索器 |
| **系统级 per-question（完整 search）** | **0.96** | 每题独立 L2 库 + ANN/FTS/RRF/时间加权 |
| 端到端（检索 + 27B reader） | **Accuracy 0.29** | 官方 Table3 Session top-5 对照：GPT-4o 0.67 / L3.1-70B 0.59 |

### 关键结论

1. **系统检索器 = 组件上限，甚至微超**（0.96 > 纯余弦 0.93）——FTS 补充补齐向量遗漏
2. **L3 boost 只在"用户自己的记忆库"上生效**——评估无关数据源时必须禁用（官方设定无 L3 概念）
3. 端到端差距在 reader 模型（本地 27B vs GPT-4o 级），检索层已超官方记忆系统

### 过程中的正式系统改进（评估反哺）

- FTS 英文停用词过滤（短英文问题 OR 匹配噪声）
- FTS 腿权重 ×0.5（向量腿为主，FTS 补充）
- 候选池 pool 可配置（默认 max(k*3,60)）

## 自设评估题（每日事件流）

`评估题-每日事件流-2026-08-26.md`：构造"某天会发生的事"（临时重要/临时不重要/易被覆盖但必须提醒），
验证时间感知优先级（24h 加权 / L3 时间排序 / L3→L2 降级）的端到端行为。

## 脚本

`scripts/` 下为评估脚手架（需适配你的仓库结构 + llama-cpp venv）：
- `longmemeval_eval.py` — 抽样摄入 + 检索 Recall
- `longmemeval_reading.py` — Reading（检索 → LLM 作答 → 双口径评判）
- `longmemeval_perq.py` — 纯余弦 per-question 检索
- `longmemeval_perq_pipeline.py` — 系统级 per-question（独立 L2 库 + 完整 search）
