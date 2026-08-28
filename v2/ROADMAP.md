# Grace V2 开发路线图（ROADMAP）

> 状态：2026-08-27 用户拍板版本。三阶段递进：权重轨闭环 → 心智×主动 → 加固×证明。
> 铁律：一切开发/实验只在 `ai-sandbox` 内；正式系统只读；任何时刻单模型驻留（48GB）。

## 总原则（排序依据）

1. **依赖驱动**：心态轨/自发依赖权重轨跑通；一致性加固依赖前两者在线。
2. **风险递减**：先复用已验证管线（惠惠），再谈自动化；先规则引擎，再上 35B。
3. **资源约束**：48GB 单驻留铁律。**27B 训练必须与白天 day-model 错峰**（停 8100 → 训练 → 恢复），惠惠 v6 已验证此流程。

---

## M0 · 框架（✅ 已完成 2026-08-27）

- `v2/` 三轨骨架：config / persona(rem) / engine / api(:18300) / skill `grace-v2-train`
- 隔离复检 6/6 全绿

---

## M1 · 权重轨最小闭环 —— 参考惠惠（Megumin）已训模型

**参考资产（正式系统只读）**：
- 训练配方：`models/megumin-lora/adapters/megumin_v6/adapter_config.json`
  `rank=8 / scale=20.0 / lr=3e-5 / iters=1000 / num_layers=16 / max_seq=2048 / batch=1 / save_every=100 / seed=42`
- 数据格式：ChatML `{"messages":[system人设, user前一句台词, assistant角色台词]}`
  （v6-beta 方案：对话块级提取 + 叙述主语归属，纯正则零 LLM）
- 数据管线：`konosuba-dialogue-extractor` skill（正则提取「」引号台词 → 长度 5-150 → 排叙述/复杂句 → 去重）
- 成品参考：`megumin_v6/` 每 100 步 checkpoint + 最终 `adapters.safetensors`
- 质检门：20 题盲测（身份/知识/口癖/复读），教训 = 数据增强重复样本 + 身份错乱 → ChatML 每条带 system 修复

**M1 动作**：
1. 数据源：**Re:Zero 轻小说文本（待用户确认提供）**；无文本前先用原创锚点样本扩充 30~50 条跑通流程
2. 提取器改造：复制惠惠提取逻辑 → 雷姆（说话人「雷姆/蕾姆」，口癖/自称规则见 `v2/persona/rem.md`）
3. 构建 ChatML 数据集（system = rem.md 人设 + user = 前一句 + assistant = 雷姆台词）→ `experiments/lora/datasets/rem/`
4. 训练：`mlx_lm.lora`（抄惠惠配方）→ `experiments/lora/adapters/rem_v1/`；**与 day-model 错峰**（停 8100 → 训 → 恢复）
5. 训前/训后语气对比 + 20 题盲测门 + 训练报告（每次一个）
6. 可选：先 qwen2.5-1.5b 走通全流程再上 27B（小模型不占白天时段）

**验收**：第一个真实雷姆适配器 + 报告 + 盲测通过（身份不跑偏、口癖稳定）。

---

## M2 · 夜班引擎真实化 —— 训练候选直接吃已有文件与记忆（不等夜班）

**修正**：候选源不依赖夜班流水线产出，直接读取正式系统**已有**文件与记忆（只读引用，写仍为零）：
- `local-ai-agent/memory/L0_raw/`：`chat.jsonl` / `exchange:inbox.jsonl` / `wechat.jsonl` / `school.jsonl` 等
- 注入方式：`AIAGENT_PROD_L0=<正式L0路径>` 环境变量（沙盒内只读打开），`candidate_extract.py` 支持该源
- 隔离不变：读正式系统 OK（红线=只读），沙盒内所有训练产物仍只落沙盒

**M2 动作**：
1. ✅ **源信息按天排序（2026-08-27 已实现）**：`candidate_extract.py` 按本地日期分组（epoch/ts → Asia/Shanghai），`--days` 看按天统计、`--date YYYY-MM-DD` 只提炼那一天、默认提炼最新一天；候选 id 带日期 `lora-YYYYMMDD-<seq>`；多源支持 `--l0 <path>` / `AIAGENT_PROD_L0`（正式 L0 只读引用）；文本提取按 source 适配（exchange→payload.text / wechat、chat→payload.messages[].text）
2. ✅ **风格/事实分类器升级**：词表扩充（金额/学期节点 + 雷姆口癖词：巴鲁斯/昴君/鬼族/呜呣…），冒烟通过（"学费 5000 美金"→fact，"巴鲁斯,蕾姆…"→style）
3. **35B 夜班错峰集成**（代码骨架已就绪：night_engine_v2 --real + 前置 8100 检查；35B 真跑待夜班）
4. ✅ **隔天生效 + 24h 反悔窗口**：`v2/engine/adapter_manage.py`（promote/rollback/list，active.json 记录生效版本）；训练产物改 `rem_v1_YYYYMMDD` 命名
5. ✅ **7 天滑动窗口 + 周 merge 脚本**：`v2/engine/lora_lifecycle.py`（prune/status/weekly_merge；真权重 merge 为开放问题 2）
4. 隔天生效 + 24h 反悔窗口（适配器轮换：昨天 adapter 保留可回退）
5. 7 天滑动窗口 + 周 merge 脚本（LoRA 累积管理）

**验收**：白天可随时用已有记忆生成训练候选；夜班自动「提炼→人审→训练→快照→报告」连续 7 天不翻车。

---

## M3 · 心态轨真实化（B1，与 M4 并行）

1. ✅ `mood_engine` 规则版 + **35B 推演骨架**（`derive_with_35b`，回退规则引擎；35B 真跑待夜班）
2. ✅ **心态注入器** `persona_injector.py`：mood_states → 显式文案前缀 + 雷姆人设段 → build_v2_system；铁律=只调语气强度不动价值观
2b. ✅ **日内变化机制**（M3 增强）：mood_intraday 表 + `apply_intraday_event`（事件即时拨动，impact 0.4/大事 0.6）+ `current_intraday`（指数衰减 exp(-Δt/τ)，τ=2h 回归当日 base）+ `intraday_timeline`；注入器优先读日内（「现在你心情很好（刚才：…）」），跨天自动回退日级
2c. ✅ **三层情绪融合引擎**（LoRA 人格底色 × 长期趋势 × 短期日内）：`combined_emotion` = 慢变量 anchor(0.6×7天趋势均值 + 0.4×人格底色 mood_baseline) + 快变量（有日内事件 combined=0.3×anchor+0.7×日内；无日内=0.5×anchor+0.5×日级）；`long_term_trend` 跨窗口对比（当前7天 vs 上一7天 → 回升/下行/平稳）；`derive`/`long_term_trend` 支持 ts/now 注入（时间线正确性）。性能实测 **0.19 ms/次**（SQLite 本地，×1000 基准）
3. ✅ **心态一致性尺子 v0** `benchmarks/mood_consistency.py`：4/4（标签合法性/平滑性/方向性）

**验收**：对话能感知「今天心情」，只调语气强度、不动价值观 ✅（注入文案已可生成）

## M4 · 自发分级（B2，与 M3 并行）

1. ✅ **三级分级** `initiative.py`：L1 播报自动 / L2 限时窗口 / L3 必须人审
2. ✅ L2 限时窗口（30 分钟无异议自动放行）+ 心态只调 L1/L2 频率（bias，不动风险等级）
3. ✅ **L3 → proposal_queue 人审**（type=dispatch，实测 prop-xxx 落闸门）
4. ✅ **自发恰当性尺子 v0** `benchmarks/initiative_appropriateness.py`：10/10

**验收**：主动消息 + dashboard 审批流 ✅（分级/放行/L3 提案全通；自发触发源与 dashboard 推送待接）

---

## M5 · 一致性加固（C）

1. ✅ **事实校验拦截器** `engine/consistency.py`：回答中日期/时间/金额槽位 vs 外挂事实 → 冲突拦截（外挂优先）；测试 6/6（正确放行/错误日期/金额/时间拦截/无事实不误杀）
2. ✅ **训练健康检查 + 自动回滚** `lora_lifecycle.check_training_health`（val 发散→bad / train<0.15∧val>1.0→overfit / 否则 ok）+ `auto_rollback_if_needed`；测试 3/3（rem_v1 过拟合日志正确判 overfit）
3. ✅ **月全量重训流程** `lora_lifecycle.monthly_full_retrain`（整合当月全部数据 → 全量重训建议）

## §15 快慢双路径（2026-08-27 用户洞察：类人双系统记忆）

> 潜意识+人格 = 快速回答（权重轨 LoRA）；记忆 = 要想一下（外挂轨检索）；每日微调 = 记忆塑造人格。
> 已写入 Grace_v2_融合设计.md §15。

1. ✅ `engine/dual_path.py`：三级路由（强事实词→slow / 人格词→fast / 弱时间词→slow 兜底）
2. ✅ 慢路径：检索（想一下）→ 注入参考记忆 → 生成 → 一致性校验；**回忆失败 → 诚实"记不清"（防幻觉第一道闸）**
3. ✅ 快路径：人格秒答（不检索）；测试 13/13（路由 6 + 慢路径 3 + 回忆失败 1 + 冲突联动 1 + 快路径 1 + 注入 1）

## M6 · 评测收尾（C）

1. ✅ **三把尺子正式版**：人格一致性 **5/5**（rem_v2 24 生成：自称 1.00/称呼 0.62/身份 0.78/零复读/口癖 0.21）+ 自发恰当性 **10/10** + 心态一致性 **4/4**
2. ✅ **LongMemEval 回归**：检索核心 md5 与正式系统逐字节一致（零改动→无退化）；引用基准 R@5=0.93-0.96；完整回归列夜间任务
3. ✅ **48GB 复盘**：训练峰值 39.1GB + 服务 15.2GB = 54.3GB 超载 → 单模型铁律实证；v2/README 全面更新
- **全量测试 76 项断言全绿**（M6-评测收尾报告.md）

---

## 开放问题（动手前需拍板，Grace_v2 设计 §14）

1. **雷姆数据源**：Re:Zero 轻小说文本是否有？（无 → M1 先用原创样本 demo）
2. 锚点数据长期维护（谁负责标注/扩充）
3. 周 merge 失真阈值（什么指标触发回滚）
4. 风格/事实分类器精度目标（规则 vs 模型辅助）
5. 心态标签体系（已暂定固定词表 6 个）
6. 自发 L2 限时窗口时长
