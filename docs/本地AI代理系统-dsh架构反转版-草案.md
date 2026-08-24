# 本地 AI 代理系统 · dsh 架构反转版设计文档（草案 v2）

> **草案状态**：2026-08-15 基于七轮 grill（Q1–Q53）固化。本文件为**反转版草案**，待用户确认后替换原《本地AI代理系统设计文档.md》。
> **原则**：先别急改项目资料——本草案仅作确认用，原文档未被修改。

> **⚠️ 2026-08-17 实测修正（取代 08-15 模型选型）**：夜班模型定稿为 **Qwen3.5-35B-A3B Q6_K（llama.cpp / GGUF，~29GB）**，非原 §3 的 Qwen3.6-40B；白天 **Qwen3.8-27B（MLX，GDN hybrid，峰值 15.5GB 实测 17.3 tok/s）**。**dsh 实测为 Node/TypeScript（Cordis 框架），非 Python**；本地模型经 **OpenAI 兼容 HTTP（mlx_lm server / llama-server）注册为 ctx.llm provider**（M3 接缝已证通）。本草案 §2/§3/§4/§7/§14/§17/§18 的「40B」字样均已过时，由主对话于 2026-08-17 深夜修订。见 `local-ai-agent/审计-任务对照-2026-08-17.md`。

---

## §0 架构反转总纲（本次最核心变更）

原假设：手写编排器，dsh 是可选组件。
**反转后站位**：

- **本地 AI 系统 ＝ dsh 架构的基础与主体（主权系统）**。保留独立完整性、可自我运行。
- **dsh ＝ 被本地 AI 使用的本地编排引擎**（Cordis 插件树 + `ctx.*` 服务总线）。dsh 是本地软件、非云，无损主权。
- **线上 agent（DeepSeek / Kimi-K3 等云端模型）及其驱动的爬虫 ＝ 可插拔外设插件**。

一句话：**本地自主运行是系统的主要目的**；云端是按需调用的外设，不是核心。

---

## §1 目标与定位

- 一套运行于未来 M5 Pro（48GB）Mac 上的**本地主权 AI 代理系统**。
- 能力：编辑文件、本地知识库、本地记忆库、网页搜索、与 WorkBuddy（主 agent / 洛琪希）直接文件夹互传、独立常驻桌面信息窗口。
- 核心哲学：**本地优先、离线可活、云端仅作外设**。

---

## §2 目标机器事实（M5 Pro "阉割版"）

- M5 Pro，48GB 统一内存，1TB SSD，约 14.5GB/s SSD 读写，307GB/s 内存带宽。
- 内存预算现实：macOS 10–12GB + 浏览器/IDE 8–10GB → 白天约 26–30GB 给模型。70B 4-bit ≈ 38GB（仅权重）**白天放不下，且含 KV 后 45–52GB 连夜班 36GB 也装不下**；04:00 清桌后约 36GB 可用 → **夜班实际选型为 Qwen3.5-35B-A3B Q6_K（~29GB，含 KV，llama.cpp/GGUF），70B 留作 64GB+ DLC**（2026-08-17 定稿已将 40B 方案替换为 35B-A3B MoE）（详见 `70B部署可行性研究-kv缓存.md`）。
- 当前 M2 8GB、磁盘 91% 满、近退役；目标机 2026-03 后发布。Phase 0 先在 M2 验脚手架，Phase 1 迁 M5。

---

## §3 模型分层（白天 / 夜班）

- **白天**：20GB 级模型经 Ollama（OpenAI 兼容 API），`ctx.llm` adapter 即插即用。
- **夜班 04:00**：**Qwen3.5-35B-A3B Q6_K（~29GB）经 llama.cpp** 作"夜班分析师"，白天绝不加载。保守备选：Qwen3.8-27B（MLX 原生，15.5GB 峰值）。
  - **⚠️ 上线前核查（2026-08-17 实测更新）**：① **确认取干净 base**（私人记忆系统建议用未改写 base，避免 persona 污染事实记忆）；② **夜班框架锁定 llama.cpp（GGUF）**——35B-A3B 的 GDN+MoE 层在 llama.cpp 量化成熟、Metal 稳定；**MLX 已实测支持 GDN（白天 Qwen3.8-27B 即 GDN hybrid，15.5GB/17.3 tok/s），故「MLX 不支持 GDN」的旧结论已证伪**。选型依据见 `模型选型总结-2026-08-16.md`。
- 模型无关：`ctx.llm` adapter seam，本地 Ollama / MLX 皆可。
- **本地自主运行是主要目的**：整套核心（夜班/记忆/爬虫）靠本地模型 + launchd 自跑，永不依赖互联网。

---

## §4 四层记忆（系统灵魂，dsh 不给，必须自建）

- **L0 原始层**：只追加不删（append-only），微信/邮件/学校/云端投放等所有摄入落这里。对应 dsh 的 append-only `SessionEvent` 不变量（"Model-visible means logged"）。
- **L1 工作层**：30 天滚动工作记忆。
- **L2 语义层**：sqlite-vec + bge-m3 + BM25 混合检索。
- **L3 核心层**：常驻约 2000 token 上下文，git 版本化快照（可回滚）。
- **睡眠期巩固**：夜班 **（Qwen3.5-35B-A3B Q6_K，llama.cpp）** 将 L0/L1 巩固进 L2/L3；遗忘＝从索引降级，绝不删 L0。

---

## §5 Megumin persona + 隔离（Q34）

- persona 模式对话打 `mode` 标签（normal / persona）。
- 巩固阶段**只把 `mode=normal` 喂进事实记忆**；`persona` 对话隔离到 `persona/` 子树，永不进 L0–L3 语义索引，不参与"想起某事"检索。
- 实现：dsh agent preset + `isolate` realm + `deployment:persona`，热插拔 LoRA（`--adapter-path`，**绝不 fuse**）。

---

## §6 隐私红线（Q7 / Q47）

- `sensitive:true` 数据只落 `shared/`，**永不出境**。
- **云端插件（DeepSeek / Kimi-K3）永不碰 L0–L3**：调用时只给具体任务片段 + 最小 briefing；产出回流本地。
- 例外口子（开工前定）：仅当用户显式许可，才给云端只读脱敏记忆视图（RAG over L2）。
- 实现：dsh `ctx.sandbox`（macOS Seatbelt）+ approval policy + `fs/*``tools/*` 事件做 policy listener。

---

## §7 夜间流水线（Q51：全 `ctx.jobs`，非自由 agent）

- 03:00 摄入（微信解密 + 邮件 + ucsb + 本地爬虫产出 + 扫描 cloud-drop）→ 03:30 增量嵌入 → 04:00 **巩固（Qwen3.5-35B-A3B Q6_K）** → 最晚 08:00 看门狗交付 changelog。
- **实现**：摄入/嵌入是确定性 `ctx.jobs`（无模型推理）；04:00 巩固是 `ctx.jobs` 调 `ctx.llm`（35B-A3B 模型调用，非 agent）。整体由 **launchd 在 04:00 触发** dsh 跑这串 job。
- 看门狗：每阶段 heartbeat + 退出码；08:00 无 changelog → 侧边栏红点告警；记忆核心每晚 git 快照回滚；每阶段 `--verify` 自检。
- **门控（2026-08-18 用户定）**：03:00 摄入段全部完成后才允许进 04:00 巩固——**资料未抓取完毕则不巩固**。
- **交付物规格（2026-08-18 用户定）**：巩固完成后产出「夜班报告」= changelog + **本地 AI 自主判断的「今日应在意的学校 / 微信信息」优先级清单**，供 dashboard 首屏展示（§16）。

---

## §8 邮件深度抓取（已落地，Q9/Q35 更新）

- `gmail-cookie-reader`：Gmail OAuth + 复用本地 Edge 会话 + cookie 注入（2026-08-14 验证可读全文）+ `gmail_llm.py` 本地桥。
- `outlook-cdp-reader`（2026-08-15 验证）。
- 每日邮件摘要自动化 `automation-1786267403854`（UCSB+Gmail+Outlook，DAILY）已上线。
- 默认深度抓取；仅当用户点名某封才需额外深抓（已基本无缺口）。

---

## §9 提案队列 + 看门狗（非阻塞）

- 睡眠 agent **永不询问**，只写提案到队列；看门狗 08:00 硬截止、单任务 10 分钟超时；连续两晚无 changelog 告警。
- 被否决提案**永不再提**。

---

## §10 写回安全边界（Q32）

- 只有"用户显式动作"才写回：(a) 侧边栏状态按钮点击；(b) 聊天命中白名单动词（"交了/done/完成" + 具体作业名）。
- AI 推断绝不写回，一律落提案队列。
- 每次写回追加 `audit.log`（只追加不删，单记录单文件），git 一键回滚。

---

## §11 学校文件夹 + 云端投放文件夹（Q26 / Q47 / Q53）

- **学校文件夹**：独立子目录存日程表/作业表等，本地 AI 可读写（经写回边界）。
- **云端投放文件夹 `exchange/cloud-drop/`**（Q47 新增）：
  - 线上 AI 抓取到的信息落此（如 `cloud-drop/kimi-scrape/`、`cloud-drop/deepseek-fetch/`）。
  - **云端＝生产者，本地＝主权消费者**：数据先隔离，夜班 ingest 同批扫描，经提案队列门控后才巩固进 L0–L3。
  - **hot 子文件夹**（Q53 新增）：`cloud-drop/urgent/` 存少部分**时间敏感性文件**，本地 AI **高密度扫描**（高频/事件驱动，非仅夜班），便于及时告警（如注册截止、成绩发布）。

---

## §12 exchange/ 总线（Q13）

- inbox / outbox / shared / archive 目录；`fs/*` 事件驱动；`sensitive:true` 只落 shared。
- 与 dsh `ctx.fs` provider seam 天然同构。

---

## §13 爬虫适配（Q39B / Q50；**2026-08-18 用户定夺降级**）

- **用户定夺（2026-08-18）：爬虫不强制全部重写为 `ctx.tools`**。要求降为三条：① dsh 可调用（经 `ctx.jobs` / bash 包装现有脚本即可，M4 的 `m4_ingest.py` 管线保留）；② 本地 AI 可经 dsh 下达指令触发抓取；③ **每晚记忆巩固前资料必须抓取完毕**（§7 门控）。
- 原设计（存档）：全部爬虫重写为 dsh `ctx.tools` 上的 tool（`wechat-decrypt` / `ucsb-fetch` / `firecrawl` tool 等），由 dsh 统一管理生命周期、超时、重试、审批（复用 `tools/pre-execute`→guard→`tools/execute`→`tools/post-execute`）。后续若需统一治理再演进回此路线。
- **归属切分**：
  - 本地自主爬虫（微信/ucsb/邮件/本地 Firecrawl）＝ **本地 AI 主体的核心 tool**（"主体器官"，非插件），仍保持外部进程经 launchd/Docker 触发、写 `exchange/`、dsh 经 `ctx.fs` 消费。
  - 线上 agent 驱动的爬虫 ＝ **可选插件**（云端模型临时决定抓取时挂载）。

---

## §14 skill / 自动化资产盘点（摄入层大半已建）

| 子功能 | 已有资产 | 状态 |
|---|---|---|
| 微信解密+摘要 | macos-wechat-db-decrypt + wechat-summary-bot + wechat-summary-report | ✅ |
| 邮件 Gmail 全文 | gmail-cookie-reader（OAuth+注入+本地桥） | ✅ |
| 邮件 Outlook | outlook-cdp-reader | ✅ |
| 邮件每日摘要 | 自动化 1786267403854（DAILY） | ✅ |
| 本地模型 | local-ai（Ollama API） | ✅ |
| UCSB 课表 | ucsb-data-bot + 自动化 1786173428851 | ✅ |
| 定时调度 | macos-launchd-scheduled-task（微信已修 04:00） | ✅ |
| agent 交接 | handoff | ✅ |
| DeepSeek Harness | github.com/deepseek-ai/deepseek-harness（**Node/TS，Cordis 框架**，MIT, dev preview） | ✅ M3 接缝已证通（OpenAI 兼容 HTTP 注册 ctx.llm） |

→ 绿地（主体自建）：四层记忆 + **夜间巩固（Qwen3.5-35B-A3B Q6_K / llama.cpp）**、夜班 job 链、UI（社区 dsh UI 插件）、Megumin LoRA、dispatch、`exchange/` 总线、cloud-drop、persona 隔离。

---

## §15 多 agent 分发 ＝ 开工时的构建分工（Q16 / Q52）

- **不是运行时能力，是 Phase 1 实现时的构建方式**。由 hy3（主 agent / 洛琪希）当主调度，把活分给不同 AI。
- 分工：
  - **hy3（我本人 / 洛琪希）**＝ 主对话监督 + 任务分配 + 收口剩余开放工作。
  - **DeepSeek**＝ 后台（backend）代码完整实现。
  - **Kimi-K3**＝ 少部分高难/救火；**死贵，平时省着用，高难与出 bug 不吝啬**。
- **dsh 映射（Q52 A）**：DeepSeek / Kimi-K3 各注册为 `ctx.llm` 端点 + 薄任务包装；hy3 用 `agent.inject()` / 分发派活。它们是"被调用的模型"，非自由 agent。
- **触发（Q49 A）**：手动路由；**引信＝"多次出错、代码已成一坨屎山"时点 Kimi-K3 救火**。
- **开发平台（Q52）**：本方案在**当前平台（WorkBuddy）完成开发**，无需担心。

---

## §16 UI：基于社区 dsh UI 插件（Q48）

- dsh 本身是插件且有社区 UI 插件；界面**参考/复用线上已有开源 dsh UI 项目**构建，不纯自研 Tauri。
- 本地 AI dashboard（**首屏＝夜班报告**：「今日应在意的学校/微信信息」优先级清单 + changelog，2026-08-18 用户定；另含今日关注 + 睡眠态 + 写回按钮）**内容主权自有**，渲染底座借社区。
- dsh 跑 headless 作编排后端，喂事件/状态给壳。

---

## §17 搭建顺序

- **Phase 0（现在 M2，仅修 bug）**：✅ 修微信 launchd（已完成 04:00）。其余（记忆骨架 / 夜间编排器 / UI 原型 / 模型部署）**全部推迟到 M5 到货后**，遵循「除 bug 外一切正式开发等 M5」铁律（2026-08-15 用户定夺）。
- **Phase 1（M5 到货）**：部署 **Qwen3.5-35B-A3B Q6_K 夜班（llama.cpp）**、接 Megumin LoRA、记忆规模化巩固、学校/cloud-drop 写回闭环、云端插件接入（DeepSeek/Kimi-K3 端点）。**⚠️ 开工前核查（2026-08-17 实测更新）**：① 确认取**干净 base**（私人记忆系统建议用未改写 base，避免 persona 污染事实记忆）；② 夜班框架锁定 **llama.cpp（GGUF）**——35B-A3B 的 GDN+MoE 层在 llama.cpp 量化成熟、Metal 稳定；**MLX 已实测支持 GDN（白天 Qwen3.8-27B GDN hybrid 跑通），「MLX 不支持 GDN」旧结论已证伪**，MLX 可作白天主力。
- 设计可移植：M2 验证脚手架原样搬 M5，只换模型与机器。

---

## §18 待钉死 / 开工前清单

> **🔴 夜班模型两项硬核查（开工前必须先过，详见 `70B部署可行性研究-kv缓存.md`）**：
> 1. **确认取干净 base**：私人记忆系统建议用未改写 base，避免 persona 污染事实记忆。
> 2. **夜班框架锁定 llama.cpp（GGUF）**：35B-A3B 的 GDN+MoE 层在 llama.cpp 量化成熟、Metal 稳定；**MLX 已实测支持 GDN（白天 Qwen3.8-27B GDN hybrid 跑通），「MLX 不支持 GDN」旧结论已证伪**，MLX 可作白天主力。

- ~~DeepSeek / Kimi-K3 / hy3 实际 API 端点~~（2026-08-18 用户定夺：**不管了**，开工时再说）。
- ~~Kimi-K3 成本护栏具体数值~~（2026-08-18 用户定夺：**不管了**）。
- ~~`cloud-drop/urgent` 高频扫描的具体周期~~（2026-08-18 已定：**15 分钟 StartInterval 轮询**，`com.local-ai-agent.urgent-watch` 已装机）。
- dsh 稳定性：developer preview 破坏性变更，M5 时预计已稳，需再核验版本（与上面硬核查②共同核验 llama.cpp 主线版本是否已达 PR #22673）。
- 新机解密密钥重捕保险（Q36，迁移后一次性处理）。
- Megumin LoRA 需基于**干净 base** 重训（非社区微调版），开工前定训练语料与基座。
