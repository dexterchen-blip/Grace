# 双模型分层唤醒架构(2026-08-30)

> 资源最优:5B 哨兵常驻(轻,只读)感知,完整系统(27B)按需唤醒。
> 设计重点:①她如何开启对话 ②她如何判断合适时机 ③突发情况要不要提醒。

## 架构

```
5B 哨兵(Llama-3.2-3B 4bit,~2GB,常驻,launchd 每 5 分钟)
  巡检 .daytime(L1 瞬时层,水位只扫新增——无完整注意力,只看更新)
  → 规则分级(urgent/important/routine,★关键词由完整系统决定,哨兵只读)
  → urgent 命中 → 5B 兜底确认(考试/截止/缴费=需要)
  → 写唤醒信号 sentinel-signal.json
        ↓ 唤醒
wake_handler(唤醒入口): urgent 且 5b≠不需要 → 通知主人 + 若 27B 未跑则启动
timing_decision(时机决策): 夜班后三因子 → proactive-plan.json
        ↓ 合适时间
完整系统主动开启对话(主动消息=事件+她的读心)
```

## 时机三因子(她如何判断"合适的时间")

| 因子 | 问什么 |
|---|---|
| ① 事件重要性 | 这件事值得打扰吗?(urgent > important > routine) |
| ② 主人可打扰度 | 深夜(23-8)不打扰 / 晚上(19-22)最佳 / 白天可 |
| ③ 对话时机 | 距上次主动 > 6h / 当日主动 ≤ 2 次 |

## 突发分级(5B 判断要不要提醒)

- 🔴 紧急(考试/截止/缴费/异常)→ 立即唤醒+提醒
- 🟡 重要(缴费/报名/签证)→ 记录,等合适时机
- 🟢 日常(广告/newsletter)→ 不打扰,进夜班汇总

## 关键文件

- v2/engine/sentinel.py(哨兵)+ sentinel_keywords.json(★Grace 决定,哨兵只读)
- v2/engine/wake_handler.py(唤醒入口,防重复)
- v2/engine/timing_decision.py(时机决策)
- 模型: models/sentinel-3b(Llama-3.2-3B 4bit)
