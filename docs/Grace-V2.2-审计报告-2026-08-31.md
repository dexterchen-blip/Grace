# Grace V2.2 设计 vs 真实系统 全量审计报告

> 审计时间：2026-08-31 01:20–01:40（GMT+8） ｜ 审计方式：**只读**（未改任何代码/数据）
> 审计对象：`Grace-V2.2-完整设计总结.md`（8/31 生成） vs 真实运行系统 `~/WorkBuddy/watch/ai-sandbox-stress`
> 结论：设计文档机制 **90% 已落地**；发现 **2 个 P0 级静默断链 bug、3 个 P1、若干 P2**；「F 轮全机制验证」正在进行中，但受 P0 影响名不副实。

---

## 1. 系统拓扑（先分清四个环境）

| 环境 | 路径 | 角色 | 与设计文档关系 |
|---|---|---|---|
| **压测真机** | `~/WorkBuddy/watch/ai-sandbox-stress` | F 轮实际运行环境，数据/权重/评估报告所在地 | **审计基准**（最新） |
| 开发沙盒 | `~/WorkBuddy/watch/ai-sandbox` | 开发迭代环境 | 与真机 v2 代码基本一致 |
| 发布快照 | `Grace-repo/v2`（工作区） | git 脱敏发布版，8/30 03:10 | **落后**：缺 5 个 stress 文件 + 全部 uncertainty/theta/PE 逻辑，且 3 个文件有字符串字面量路径 bug |
| 正式系统 | `local-ai-agent/` | 白天/夜班生产（27B/35B 双模型） | V2.2 尚未集成（dashboard 零引用 v2） |

**关键认知**：`Grace-V2.2-完整设计总结.md` 的「代码地图」描述的是**发布快照**而非真机；真机比发布版新（`grace_tomi_mem_test` / `self_cognition_test` / `enrich_emails` / `subjectivity_test` / `reset_stress` / `sample_missing` 均只在真机存在）。发布版若被直接部署会拿到**坏的哨兵链路**（见 P1-5）。

---

## 2. 设计 → 实现 对照总表

### 2.1 记忆系统（设计 §3）✅ 基本落地

| 要素 | 状态 | 实现位置 / 证据 |
|---|---|---|
| 四层记忆 L0/L1/L2/L3 | ✅ | L0 append-only jsonl（md5 水位）、L2 bge-m3+BM25+RRF、L3 autobiography.db（`L3_core/`） |
| 双图谱（记忆×情绪耦合） | ✅ | `engine/mood_graph.py`：`dual_graph_ingest` + event_id/entity 耦合 |
| 暗注意力边（M7d） | ✅ | `edge_type='hidden'` + `_hidden_derive` 潜台词推导（实测 771 条 hidden 边） |
| 呼应五机制 | ⚠️ 部分 | ②一致检索 / ③衰减永存 / ④事件团簇 / ⑤重估 已实现；**①强度调制无独立实现**（强度仅由 sentiment 绝对值决定） |
| **uncertainty 标记（8/31 补）** | ⚠️ **代码有、数据无** | `stress_engine.py:723` 运行时 `ALTER TABLE ADD COLUMN`（被 except 静默吞）；**真机 l2.db 的 mood_graph 表实测无此列** → `COALESCE(uncertainty,0.2)` 必报 `no such column`（已实测复现） |

### 2.2 认知层（设计 §4）✅ 全部落地

| 模块 | 状态 | 说明 |
|---|---|---|
| mood_engine 三层融合 | ✅ | `combined_emotion()`（日内×日级×底色）+ `persona_baseline`/`long_term_trend` 全在；沙盒版已按 V2.1 决策删除 35B 推演（单模型 27B） |
| attention | ✅ | 情绪显著度×记忆关联度 + 社交价值加成 |
| theory_of_mind | ✅ | 读情绪图谱史 + 暗注意力 → {emotion/need/interruptible/advice} |
| self_activation | ✅ | 三源触发 + ToM 修饰（interruptible=False 强制不打扰） |
| dual_path 快慢路径 | ✅ | fast 权重直出 / slow 检索注入 + 诚实「记不清」 |
| consistency | ✅ | 日期/金额/时间高置信槽位冲突拦截 |
| persona_injector | ✅ | 人格轨+心态轨统一注入（训练/推理对齐，mood_prefix 三层文案） |

### 2.3 人脑级反馈回路（设计 §5）⚠️ 已实现但存在断链

| 机制 | 状态 | 证据 |
|---|---|---|
| ① PE 调制（误差→w3/w2/w1） | ✅ | `stress_engine.py:740`，feedback 权重 1–3 重复写入训练集 |
| ②' 提取抑制（k 3→1） | ⚠️ **被 P0-2 阻断** | 逻辑在 `sample_persona`（L510），但同一处 `from engine.l2_semantic import search` 必 ImportError → 整段检索静默跳过 |
| ③ theta 峰值同步 | ⚠️ **数据 0 条** | 逻辑在 `stress_engine.py:675-683`（01:14 新版）；运行中的 F 轮进程为旧代码 → 未执行 → 库内 `theta` 标记 0 条 |
| uncertainty×2 权重 | ⚠️ **被 P0-1 阻断** | 逻辑在 `extract_graph_samples`（L296-306）；列不存在 → 整函数报错静默失败 |
| 置信累积 prediction-errors.jsonl | ✅ | 真机 16.7KB 数据（01:21 更新）；**但 ToM 尚未消费该数据**（待做③） |
| 再巩固（改原记忆零句式） | ⚠️ 生效但**过度写入**（P1-4） | 库内「后知:实际」标记 896/1763 = **51% 边被再巩固改写** |

### 2.4 训练闭环（设计 §6）⚠️ 8 类样本中 2 类从未进过训练

训练集组装位于 `train_27b`（L377-434），8 类来源核对：

| # | 样本类 | 状态 | 说明 |
|---|---|---|---|
| 1 | 锚点回放 5% | ✅ | `anchor_texts[:n_anchor]` |
| 2 | 加工层（情绪×记忆） | ✅ | `extract_mood_samples` → mood_samples 模板句（⚠️ 模板库仅 13 主题×6 情绪×2 句，多样性有限，是设计已知「乐观模板」的根源） |
| 3 | L3 自传体叙事 | ✅ | `extract_l3_samples`（autobiography.db，confidence high/medium） |
| 4 | 双图谱情绪边 | ❌ **P0-1 断链** | `extract_graph_samples` 首条 SQL 即因 uncertainty 列缺失抛错 → 整函数失败 |
| 5 | 双图谱暗注意力边 | ❌ **P0-1 断链** | 同上（同函数，未执行到） |
| 6 | 人脑级反馈（PE 权重） | ✅ | `feedback-live.jsonl`（w 1–3 重复写） |
| 7 | 注意力+潜意识判断 | ✅ | `cognition-live.jsonl` |
| 8 | 主动消息（情境→读心） | ✅ | `proactive-live.jsonl` / grace-book 兜底 |

**含义**：设计文档第 4、5 类「双图谱情感记忆/潜台词」样本从未真正参与塑造 → 「杏仁核×海马体进权重」这一核心卖点在 F 轮仍是空转。

### 2.5 行为层（设计 §7）✅

| 要素 | 状态 |
|---|---|
| 主动消息 = 事件 + 读心 | ✅ 强绑定（`_adv`→「主人,雷姆在。」等尾部） |
| 零前缀原则 | ✅ 无「雷姆想跟你说」帽子 |
| 时机三因子 proactive-plan.json | ⚠️ `timing_decision.py` 实现，但仅手动/launchd 可用；**未被任何调度器接入**，且硬编码写正式系统目录（P1-5） |

### 2.6 双模型唤醒（设计 §8）⚠️ 代码完整、从未真跑

- `sentinel.py`（5B 规则巡检 + `--model` 时加载 Llama-3.2-3B）/ `wake_handler.py` / `timing_decision.py` 代码齐全，`sentinel_keywords.json` 存在（urgent/important/routine 三档）。
- 真机有 `sentinel_state.json` + `wake_handled.json`（各跑过 1 次）；但**无 launchd 常驻任务、无 sentinel-signal.json 触发记录** → 哨兵链路从未在实际运行时闭环。

### 2.7 评估体系（设计 §9）✅ 已产出，但评分未入库

| 尺子 | 文件 | 真机报告 | 说明 |
|---|---|---|---|
| 人格一致性 | `benchmarks/persona_consistency.py` | persona-consistency-*.md（8/27） | 5 指标 PASS/FAIL，报告内自评分 |
| Grace-ToMi 30 题 | `grace_tomi_test.py` | tomi-report.json（01:05） | **带自动评分 judge**（fb_2nd 看「以为/装」、fb_1st 看「不确定」边界感）；但 **judge 结果未写入 JSON**（只 print），事后需重算 |
| 带记忆注入 ToM | `grace_tomi_mem_test.py` | tomi-mem-report.json（01:15） | 检索双图谱含「后知:实际」→ 证明再巩固标记可被检索；**评估对象是 E 轮 rem_stress_d90**，非 F 轮 |
| Grace-SOTOPIA 7 维 | `grace_sotopia_eval.py` | sotopia-report.json（01:0x） | LLM-judge，仅评 baseline（未对比 adapter），最多 12 条 |
| 自我认知 5 场景 | `self_cognition_test.py` | self-cognition-report.json（01:2x） | 经历引用/复读统计 |
| 主观性 8 场景 | `subjectivity_test.py` | subjectivity-report.json | 去背景/去问句自发回应协议 |

### 2.8 配置（设计 §13）✅ 对齐

`config.py`：rank=8 / scale=20 / incr_lr=1e-6 / anchor_ratio=0.05 / iters 冷启动 60-150 — 与设计一致。

---

## 3. Bug 清单（按严重度）

### P0 · 严重（静默失效，直接影响训练/评估结论）

**P0-1 双图谱样本从未进过训练（设计 §6 第 4/5 类样本断链）**
- 链路：`mood_graph` 表 schema（mood_graph.py）**无 uncertainty 列** → `stress_engine.py:723` 运行时 `ALTER TABLE ADD COLUMN uncertainty`（被 `except: pass` 吞）→ 真机 l2.db 实测**无此列** → `extract_graph_samples`（L296）`SELECT ... COALESCE(uncertainty, 0.2)` 实测报 `OperationalError: no such column: uncertainty` → 整函数静默失败（异常只 print 不进 stress.log）→ 情绪边 + 暗注意力边样本**从未进 LoRA**。
- 加重因素：① `ALTER` 无 `IF NOT EXISTS`（SQLite 3.35+ 支持），列已存在时会重复报错（无害但脏）；② 所有异常被 `except: pass / logln` 静默吞掉，运行时零告警。
- 修复：把 `uncertainty REAL DEFAULT 0.2` 直接写进 `MOOD_GRAPH_SCHEMA`；ALTER 改用 `ADD COLUMN IF NOT EXISTS`；`extract_graph_samples` 的异常改为可见日志 + 计数器。

**P0-2 压测采样慢路径记忆注入整体失效（设计 §5 提取抑制/theta 检索空转）**
- 链路：`stress_engine.py:507` `from engine.l2_semantic import search` → `engine` 包内**无 l2_semantic 模块**（实测 `ModuleNotFoundError`）→ 被 except 吞 → 慢路径不注入记忆。且即使修好导入，L511 `_l2search(q, k=_k, db=...)` 仍会 `TypeError`：**`src/l2_semantic.py` 的 `search(query, k=8, rrf_k=60, pool=0)` 没有 db 参数**（双重失效）。
- 影响：压测断点采样中「提取抑制 k3→1」「theta 联动检索」从未执行 → **E 轮「运行时机制测不到」的结论被此 bug 污染**（不是测不到，是压根没注入）。
- 修复：改为顶层 `import l2_semantic`（与 dual_path.py 一致，sys.path 已含 src）；去掉 `db=` 参数或用 `get_db()` 注入。

### P1 · 中（数据缺陷 / 边界越界 / 语义偏差）

**P1-3 enrich_emails 注入消息缺字段 → 邮件日心态推演全灭**
- `enrich_emails.py:72` 注入的消息只有 `{text, ts, sender, source}`，**无 sentiment/weight**；`stress_engine.py:662` 用硬索引 `m["sentiment"]`（应 `m.get`）→ KeyError → 整段日级推演 try 跳过。
- 实测：stress.log `[mood] day 5/8/12 异常: 'sentiment'`——正是邮件注入日（day 2/6/9/13/… 均受影响，F 轮 20 天中至少 6 天心态缺失）。
- 修复：enrich_emails 补 sentiment/weight；stress_engine 改用 `m.get("sentiment", 0)`；derive 内部同样防御。

**P1-4 再巩固过度写入 → 情绪史同质化**
- `stress_engine.py:746-750` `UPDATE mood_graph SET mood_label=?, intensity=?, source||'（后知:实际…）', uncertainty=0.9 WHERE entity=? AND edge_type='emotion' AND ts<?` —— **WHERE 按 entity + 时间范围，一次改写该实体当天及以前全部历史情绪边**（不是被反馈的那一条）。
- 实测：库内「后知:实际」896/1763 = 51%；抽样可见 `'家人'` 实体多条边 source 完全相同 → ToM/情绪史查询失真。
- 修复：UPDATE 增加 `event_id = <被反馈事件>` 精确限定。

**P1-5 哨兵三组件硬编码写正式系统目录 + 发布版字符串字面量 bug**
- 真机版 `sentinel.py`/`timing_decision.py`/`wake_handler.py`：`SIGNAL_FILE`/`PLAN_FILE`/`DAYTIME` 硬编码为 `/Users/cz/WorkBuddy/skills find and make/local-ai-agent/exchange/.daytime/...`（含空格 + 绝对用户目录；沙盒组件直接**写**正式系统 L-1，触碰「沙盒零写正式系统」铁律边界；换机/换目录即断）。
- 发布版（Grace-repo）更严重：三文件写成 `SIGNAL_FILE = "os.path.join(config.EXCHANGE, '.daytime')/sentinel-signal.json"` **字符串字面量**（未执行 os.path.join）→ 若部署发布版，哨兵链路必然落点错乱。
- 修复：路径改由 config/env 注入；发布版同步修掉字符串 bug。

**P1-6 训练成功判定脆弱 + 单次超时 1h**
- `stress_engine.py:450` `ok = returncode==0 and "Saved final weights" in stdout`：mlx_lm 输出文案变体或走 stderr 即误判失败（历史 90 天轮前 4 天 ok=False 高度疑似此因）；`timeout=3600` 对首次冷启动+断点续跑偏紧。
- 修复：检查 `adapters.safetensors` 产物存在 + 非零 returncode 判定；超时按轮次放宽。

### P2 · 低（语义/工程）

- **P2-7** `adapter_manage`「隔天生效+24h 反悔」未强制：promote 立即生效、rollback 无时限校验；压测 `stress-auto` 每天直接 promote（压测路径语义失效，生产路径亦然）。
- **P2-8** 周 merge 是**聚合拷贝**非真 fuse：`lora_lifecycle.py:79-93` 有 `if ... or True: pass` 死代码残留 + 注释明说「真正权重 merge 是开放问题 2」；月全量重训仅 dry-run 为主。
- **P2-9** ToMi/SOTOPIA/自我认知的 **judge 评分未写入报告 JSON**（只 print）→ 报告不可离线复算，且 SOTOPIA 未对比 adapter（只评 baseline）。
- **P2-10** `tests/` 目录 10 个测试文件**只有 `acceptance_test` 编译/运行过**（__pycache__ 仅 1 个 pyc）→ 单元测试从未全量执行，P0 类静默 bug 因此漏网；建议修码后全量跑 `tests/`。
- **P2-11** `mood_graph` 表无 uncertainty 列时，`tomi_mem_test` 等只读组件不受影响（不引用该列），但所有引用 `COALESCE(uncertainty,...)` 的查询都会炸——**schema 与查询耦合脆弱**。

---

## 4. 运行实况（真机数据）

### 4.1 F 轮（设计文档 §10「⏳ 05:00 跑」已过时）

- 00:55 启动，输入 20 天（书库实际 90 天，`--days 20` 截断）、`--train-every 1`（**每天训练**，非设计的 10 天间隔）。
- 截至 01:28：L0 已摄入 day 1–14；adapter 链 `rem_stress_d1…d13` 已训，active=d12（01:24 promote），d13 训练中；`active.json` history 已累积 100+ 条（含 8/29-8/31 多轮）。
- **F 轮进程加载的是 00:55 版旧代码**（01:14 新版含 ALTER/theta 未生效）→ 数据佐证：mood_graph day1-20 已写入 451 条边（摄入正常），但 theta=0、无 uncertainty 列。

### 4.2 历史轮次

- 8/29-8/31 至少 3 轮 90 天全跑（12:51→16:20→19:34 三轮 promote 轨迹；后两轮 2 分钟/天，90 天≈2.7h）；`final.json` 显示首轮 90 天 `trained` 前 4 天 `ok=False`（疑 P1-6）。
- mood_graph 累计 1763 条边（emotion ~992 / hidden ~771），覆盖 day 1–90；「后知:实际」896 条（P1-4）。
- 评估报告四件套（tomi/tomi-mem/sotopia/self-cognition）01:05–01:25 全部产出；**tomi-mem 用的是 E 轮 rem_stress_d90 权重**。

### 4.3 双图谱数据现状

- 表结构（真机 l2.db）：`id/entity/mood_label/intensity/ts/event_id/trigger/edge_type/source` —— **无 uncertainty 列**。
- `edge_type` 分布：emotion 975 / hidden 771 / 其余无；theta 标记 0 条。

---

## 5. 设计「待做」六项核对

| # | 待做 | 状态 |
|---|---|---|
| ① | F 轮全机制验证 | 🔄 **进行中**（00:55 启动，day14/20）；但 P0-1/P0-2 意味着「全机制」并不全 |
| ② | 微信链路修复 | ✅ 已交接（`/tmp/grace-handover-wechat-fix.md` 01:04）；今日日志显示 bot 已修通（退出码 0） |
| ③ | 置信累积 → ToM 置信输出 | ⚠️ 数据基础已落（prediction-errors.jsonl），**ToM 未消费**（theory_of_mind.py 无置信输入参数） |
| ④ | dashboard 集成（哨兵/唤醒/时机） | ❌ 未做（local-ai-agent/src 对 v2 零引用） |
| ⑤ | 元认知自我反思 | 🟡 按「纯演化」已撤回模板；`self_cognition_test` 有 meta_reflect 场景可观测萌芽（符合设计） |
| ⑥ | 对抗训练（防漂移） | ❌ 未实现；仅有 `lora_lifecycle.check_training_health` + 自动回滚（检测非对抗）+ `persona_consistency` 漂移对照 |

---

## 6. 结论与建议修复顺序

**总体判断**：设计文档描述的能力在真机代码层面基本齐全（含最新的人脑级反馈），**但核心闭环上有两处静默断链（P0-1/P0-2），且 F 轮正在旧代码上跑**——当前产出的所有「全机制」结论都需要打折。

**建议修复顺序**（等用户指令再动码）：
1. **P0-1**：uncertainty 列入 schema（mood_graph.py）+ ALTER 加 IF NOT EXISTS + extract_graph_samples 异常可见化。
2. **P0-2**：`engine.l2_semantic` → 顶层 import；去掉 `db=` 参数（对齐 `src/l2_semantic.py` 真实签名）。
3. **P1-3**：enrich_emails 补字段 + `m.get("sentiment",0)` 防御。
4. **P1-4**：再巩固 UPDATE 加 event_id 限定。
5. **P1-5**：三组件路径 config 化 + 发布版字符串 bug 修复。
6. **P2**：测试全量跑、judge 入 JSON、周 merge 真 fuse（开放问题 2）、adapter_manage 生效窗口语义。

修完后：重跑 F 轮（reset_stress 归档 → 新版代码 90 天全量）→ 重新生成四套评估 → 更新设计总结的试验矩阵。
