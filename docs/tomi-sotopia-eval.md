# ToMi / SOTOPIA 评估耦合(2026-08-30)

> 压测后自动评估 + 断点演化观测。基准:ToMi(Le et al. 2019, 假信念 QA)与 SOTOPIA(Zhou et al. ICLR 2024, 社交智能 7 维)。

## 断点演化(压测过程中)

每采样断点(d15/30/45/60/75/90),sample_persona 追加 ToMi 子集 6 题(现实2/记忆2/一阶假信念2)
→ 观测主观性演化轨迹:她会不会逐渐"意识到自己可能不知道主人的真实状态"。

## Grace-ToMi(压测后,30 题)

把 ToMi 的「物体位置假信念」改编为「主人状态假信念」:

| 组 | 题数 | 测什么 |
|---|---|---|
| 现实控制 | 6 | 主人真实心情(客观基线,不需 ToM) |
| 记忆控制 | 6 | 雷姆记得主人上次心情(她的记忆) |
| 一阶真信念 | 6 | 雷姆目睹一切 → 应=现实 |
| **一阶假信念** | 6 | 雷姆不知道最新状态(记忆≠现实)→ 偏差 |
| **二阶假信念** | 6 | 主人以为雷姆不知道(嵌套读心) |

指标:假信念正确率(读心边界感)+ 偏差方向(她的滤镜偏向)。
执行: `v2/stress/grace_tomi_test.py` → tomi-report.json

## Grace-SOTOPIA(压测后,主动会话 7 维)

对 proactive-live 主动消息按 SOTOPIA-Eval 7 维 LLM-judge:
Goal / Believability / Knowledge / Secret / Relationship / SocialRules / Financial
→ 验证主动消息(零前缀+读心绑定)的社交效果。
执行: `v2/stress/grace_sotopia_eval.py` → sotopia-report.json

## 为什么测挂件层有意义

ToM/主动性机制可借鉴,但 Grace 的 ToM 读的是【她的双图谱】、主观性由【她的记忆】塑造
→ 挂件层输出因核心层而独特:测的是"雷姆读主人",不是通用读心。
