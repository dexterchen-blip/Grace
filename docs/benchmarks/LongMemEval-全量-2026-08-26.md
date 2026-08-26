# LongMemEval 全量评估（500 题）— 2026-08-26

## Retrieval（Recall@k，pool=200）

| 类别 | N | R@1 | R@3 | R@5 | 未命中 |
|---|---|---|---|---|---|
| knowledge-update | 78 | 0.19 | 0.28 | 0.28 | 56 |
| multi-session | 133 | 0.12 | 0.19 | 0.19 | 108 |
| single-session-assistant | 56 | 0.50 | 0.73 | 0.73 | 15 |
| single-session-preference | 30 | 0.00 | 0.00 | 0.00 | 30 |
| single-session-user | 70 | 0.01 | 0.01 | 0.01 | 69 |
| temporal-reasoning | 133 | 0.16 | 0.27 | 0.27 | 97 |
| **总体** | 500 | 0.16 | 0.25 | 0.25 | 375 |

## Reading（检索 top5 会话 → 27B 作答，严格+核心词部分）

| 类别 | N | 检索命中 | 严格 | 严格+部分 |
|---|---|---|---|---|
| knowledge-update | 78 | 22 | 10 | 12 |
| multi-session | 133 | 25 | 4 | 5 |
| single-session-assistant | 56 | 41 | 23 | 35 |
| single-session-preference | 30 | 0 | 0 | 3 |
| single-session-user | 70 | 1 | 4 | 6 |
| temporal-reasoning | 133 | 36 | 2 | 4 |
| **总体** | 500 | 125 | 43 | 65 |

**端到端：Retrieval R@5 = 0.25，Reading Accuracy(含部分) = 0.13，检索命中→答对转化率 = 0.52（命中 125 题）**