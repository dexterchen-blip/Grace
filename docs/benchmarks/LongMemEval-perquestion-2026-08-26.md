# LongMemEval per-question 隔离评估（模拟官方 Session 粒度）— 2026-08-26

| 类别 | N | R@1 | R@3 | R@5 | 未命中 |
|---|---|---|---|---|---|
| knowledge-update | 78 | 0.83 | 0.94 | 0.95 | 4 |
| multi-session | 133 | 0.88 | 0.97 | 0.98 | 3 |
| single-session-assistant | 56 | 1.00 | 1.00 | 1.00 | 0 |
| single-session-preference | 30 | 0.53 | 0.77 | 0.83 | 5 |
| single-session-user | 70 | 0.46 | 0.74 | 0.80 | 14 |
| temporal-reasoning | 133 | 0.77 | 0.90 | 0.95 | 7 |
| **总体** | 500 | 0.78 | 0.91 | 0.93 | 33 |

**耗时 1778s；对照混合库 R@1=0.16/R@5=0.25**