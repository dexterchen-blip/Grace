# LongMemEval per-question × L2 管线版（系统级官方分）— 2026-08-26

| 类别 | N | search R@5 | 严格 | 严格+部分 |
|---|---|---|---|---|
| knowledge-update | 78 | 0.97 | 26 | 36 |
| multi-session | 133 | 0.98 | 23 | 27 |
| single-session-assistant | 56 | 1.00 | 27 | 41 |
| single-session-preference | 30 | 0.87 | 0 | 2 |
| single-session-user | 70 | 0.97 | 17 | 19 |
| temporal-reasoning | 133 | 0.94 | 10 | 19 |
| **总体** | 500 | 0.96 | 103 | 144 |

**端到端 Accuracy(含部分) = 0.29，严格 = 0.21，search 命中→答对转化率 = 0.30（命中 481 题）**

**对照：纯余弦 per-question 检索 R@5=0.93；官方 Table3 Session top-5：GPT-4o 0.67 / L3.1-70B 0.59 / L3.1-8B 0.52**