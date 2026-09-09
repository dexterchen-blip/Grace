# Grace V2.4 好奇心器官 —— 心理学/脑科学文献调研笔记

> 日期：2026-09-09 · 服务对象：好奇心器官 v1 设计（缺口检测 G1-G4 / 缺口账本 / 第四触发源 / 满足回路）
> 方法：WebSearch 逐篇核实出处与结论；无法找到原文处用二手来源并标注。✅=已核实原文/权威摘要，⚠️=细节待核实
> 两个记忆纠错：Gruber & Ranganath 2019 实为 **PACE 框架**（不存在"MACS/EVIG"标题）；Marvin & Shohamy 2016 实发表于 *JEP: General*（非 COBS）。

---

## 1. Loewenstein 1994 — 信息缺口理论 ✅

**出处**：Loewenstein, G. (1994). The psychology of curiosity: A review and reinterpretation. *Psychological Bulletin*, 116(1), 75–98.
**核心机制**：好奇心 = 察觉到"已知"与"想知"之间的缺口时产生的剥夺状态，本身是厌恶的、满足才有快感。三个关键约束：(a) 缺口必须**可弥合**——感觉"永远无法知道"时好奇心坍缩为冷漠；(b) 好奇心依赖既有知识基础（对一无所知的领域无所谓好奇）；(c) 强度取决于信息相对"显著参照点"（informational reference point）的距离，越近越强。后续扩展（Golman & Loewenstein, 2012 未刊稿/手册章）加入重要性、显著性、惊讶三因素。
**对 V2.4 的借鉴**：G1-G4 信号设计合法。但 v1 缺一条**可解性门控**：无法通过任何途径（主人知道/可检索）解决的缺口，激活应下降而非持续上升。账本条目应带"参照点来源"字段——主人主动提到的实体（显著参照点）权重高于被动遇见的实体。

## 2. Kang et al. 2009 — 置信度倒U + 尾状核 ✅

**出处**：Kang, M. J., Hsu, M., Krajbich, I. M., Loewenstein, G., McClure, S. M., Wang, J. T., & Camerer, C. F. (2009). The wick in the candle of learning. *Psychological Science*, 20(8), 963–973.
**核心机制**：fMRI 看问答 trivia：好奇心与**尾状核**（预期奖赏区）活动相关；被试愿花稀缺资源（代币/等待）换取答案；高好奇预测 1-2 周后对惊讶答案的记忆。关键补充数据（原文 Table S1）：19 名被试中 12 人好奇心峰值落在置信度 **0.40–0.60**——倒U关系，个体级显著。倒U此后跨范式复制（trivia/模糊图/网页点击；理论解释见 Dubey & Griffiths 2020, *Psychological Review* 127(3), 455–476 的理性元推理模型，二手核实）。
**对 V2.4 的借鉴**：**账本激活公式缺一个置信度倒U项**。v1 中"完全无记录"（G1）与"低置信"（G2）只是不同信号，但激活未区分；按倒U，conf≈0 或 conf≈1 都低，**中等置信（部分知道/记不清）最高**——正是 TOT 状态。

## 3. Gruber, Gelman & Ranganath 2014 — 好奇状态的海马/多巴胺增强 ✅

**出处**：Gruber, M. J., Gelman, B. D., & Ranganath, C. (2014). States of curiosity modulate hippocampus-dependent learning via the dopaminergic circuit. *Neuron*, 84(2), 486–496.
**核心机制**：高好奇状态下中脑（SN/VTA）与伏隔核活动增强；不仅好奇目标信息的记忆增强，**期间偶然接触的无关信息（面孔）记忆也增强**，且 24h 延迟后仍保持；附带记忆增益由中脑-海马前馈连接（functional connectivity）预测。
**对 V2.4 的借鉴**：**附带记忆增强应建模**——缺口激活高于阈值时进入"好奇状态"，期间会话内摄入的一切新 L2/L3 条目获得编码加成。这是 v1 完全没有的机制，且实现成本低（一个时间窗标志位）。直接复用 attention_director 的"全局唤醒"概念即可。

## 4. Gruber & Ranganath 2019 — PACE 框架 ✅（标题纠错）

**出处**：Gruber, M. J., & Ranganath, C. (2019). How curiosity enhances hippocampus-dependent memory: The Prediction, Appraisal, Curiosity, and Exploration (PACE) framework. *Trends in Cognitive Sciences*, 23(12), 1014–1025.
**核心机制**：⚠️ 调研委托中的"MACS / 数理化 EVIG"**未找到对应文献，待核实**——实际论文是定性神经科学框架 PACE：预测误差（海马情境PE + 前扣带ACC信息PE）→ 外侧前额叶**评估**（appraisal）→ 分岔为好奇（腹侧纹状体/VTA，走向探索）**或焦虑**（杏仁核，走向回避）→ 探索与记忆增强 → 收到信息产生新预测误差 → 循环。
**对 V2.4 的借鉴**：(a) **评估分岔**——同一缺口在不同情绪语境下是好奇还是焦虑，ToM 一票否决可显式化为 appraisal 步骤（主人情绪低落时"追问"是打扰不是好奇）；(b) **PACE 是环**——满足回路的输出应喂回缺口检测器（答案带来的新实体→新缺口），v1 的满足回路是终点，应改为中继。

## 5. Litman 2005 — I-type / D-type 双通路 ✅

**出处**：Litman, J. A. (2005). Curiosity and the pleasures of learning: Wanting and liking new information. *Cognition and Emotion*, 19(6), 793–814.（⚠️ 委托中记的"Psychology of Aesthetics"有误；I/D 概念源自 Litman & Jimerson 2004）
**核心机制**：I-type（兴趣型）= "liking"驱动，新奇引发的正向吸引，想**接近**不确定；D-type（剥夺型）= "wanting"驱动，缺口的紧张厌恶感，想**逃离**不确定。行为证据（Litman, Hutchins & Russon 2005, *Cognition and Emotion* 19(4), 559–582，经 Gruber & Ranganath 2019 引文核实）：I 型特质预测"完全不知道"类问题的状态好奇；**D 型特质预测 TOT（话到嘴边）类问题的好奇**——两类问题激活不同人。
**对 V2.4 的借鉴**：账本应**分两轨**：I 轨（G3 新实体/G4 未接话题——探索性、低压力）与 D 轨（G1 检索失败/G2 记不清——剥夺性、更急迫）。触发源台词的语气应随轨而变（I：随意聊起；D：坦白"我记不清了"）。防话痨预算可只计 D 轨。

## 6. Kashdan et al. 2018 — 五维好奇心量表 ✅（卷期 ⚠️待核实）

**出处**：Kashdan, T. B., Stiksma, M. C., Disabato, J. D., et al. (2018). The five-dimensional curiosity scale. *Review of General Psychology*（⚠️ 卷期页码待核实，约 22(3)）。
**核心机制**：好奇是多维人格结构，非单一标量：joyous exploration / deprivation sensitivity / **stress tolerance**（忍受未知带来的焦虑的能力——低者见到缺口会回避而非探索）/ **social curiosity**（对他人所思所为的特有好奇）/ thrill seeking。聚类出四类画像（Fascinated / Problem Solvers / Empathizers / Avoiders）。
**对 V2.4 的借鉴**：(a) **social curiosity 是独立维度**——Grace 对主人的好奇天然属于此维，可与 ToM 读心联动（对"主人为什么这样做"的缺口单独记账）；(b) stress tolerance 解释了 PACE 的 appraisal 分岔在人格端的成因：低耐受→缺口引发回避。Grace 的"好奇画像"应非均匀：social 高、thrill 低（符合陪伴型 agent 定位）。

## 7. Marvin & Shohamy 2016 — 信息预测误差 ✅（期刊纠错）

**出处**：Marvin, C. B., & Shohamy, D. (2016). Curiosity and reward: Valence predicts choice and information prediction errors enhance learning. *Journal of Experimental Psychology: General*, 145(3), 266–272.（⚠️ 委托中记的"Current Opinion in Behavioral Sciences"与"纹状体预测误差"有误：fMRI 纹状体证据属 Lau et al. 2020 等；本文是行为实验）
**核心机制**：信息本身即奖赏。两发现：(a) **效价不对称**——正性信息同时增强好奇心与长期记忆；(b) 驱动学习的不是信息绝对价值，而是**信息预测误差（IPE）**= 实际满足感 − 预期（好奇心）。满足超预期时记忆最好。等待范式（willingness-to-wait）证实好奇→信息价值。
**对 V2.4 的借鉴**：**满足回路应按 IPE 建模**：mood 满足事件强度 ∝ (主人回答的"满意度" − 缺口激活隐含的预期)。答案比预期更惊喜 → 更强满足 + 更强记忆强化；平淡 → 弱强化。v1 的满足事件是固定值，需引入误差项。

## 8. Itti & Baldi 2009 — Bayesian surprise ✅

**出处**：Itti, L., & Baldi, P. (2009). Bayesian surprise attracts human attention. *Vision Research*, 49(10), 1295–1306.
**核心机制**：surprise = KL(后验‖先验)——衡量**数据改变了多少信念**，与 Shannon 罕见度无关。眼动实验：72% 的注视转移指向高于平均惊讶度的位置（多人共同注视区达 84%）。
**对 V2.4 的借鉴**：G3"新实体"目前是布尔信号；mood_graph 更新前后实体先验→后验的变化可量化为 surprise 分数，替代"首次出现"的粗判定。列入 v2 升级方向，v1 不动。注意：Bayesian surprise 量的是**注意力捕获**（Itti 的本行），与好奇心的动机面是两件事——恰好对应 v1"好奇心=注意力的动机面"的定位。

## 9. Zeigarnik 效应 — ⚠️ 复制失败，v1 最大风险点 ✅

**出处**：原始：Zeigarnik, B. (1927). *Psychologische Forschung*. 复制状态（核心）：**Ghibellini, R., & Meier, B. (2025). Interruption, recall and resumption: a meta-analysis of the Zeigarnik and Ovsiankina effects. *Humanities and Social Sciences Communications*, 12, 1–13.**（59 篇文献元分析，Open Access）
**核心机制**：元分析结论：**未完成任务无总体记忆优势**（Zeigarnik 效应"缺乏普遍效度"，依赖情境——任务卷入、成就动机、实验者权威，这些历史条件今天难以复制）；但 **Ovsiankina 效应（被打断后自发倾向回去继续）稳健存在**。另：Masicampo & Baumeister (2011, *JPSP*)——未完成目标持续占据认知资源，但**制定具体计划即可消除**这种干扰（无需完成任务本身）。
**对 V2.4 的借鉴**：见修正建议 R1——v1 账本的"未解决随时间上升"激活项直接依据不足，需要重设计。

## 10. 元认知与好奇心 — RPL / TOT / 接近解感 ✅

**出处**：Metcalfe, J., Schwartz, B. L., & Eich, T. S. (2020). Epistemic curiosity and the region of proximal learning. *Current Opinion in Behavioral Sciences*, 35, 40–47；Metcalfe, Eich & Castel 相关 TOT 实验见 Metcalfe et al. (2017, *Journal of Applied Research in Memory and Cognition*，DOI 10.1186/s41235-017-0065-4) ✅；"答错时高信心反而更好奇"出自 Metcalfe et al. (2022)，⚠️ 具体出处待核实（二手经 *Journal of Intelligence* 2023 综述转引）。
**核心机制**：**区域邻近学习（RPL）框架**：好奇是元认知感受，在"感觉即将知道"时最高——TOT 状态者选择看答案的概率约 2 倍；高 knowing 感→想学，低→放弃。Loewenstein 亦强调"无法察觉自己不知道 = 好奇心的绝对屏障"（元认知是好奇心的前提）。Metcalfe et al. 2022 补充：答错但高信心的试次，反馈前好奇已升高——存在无意识的"接近解感"。
**对 V2.4 的借鉴**：G2 信号应**分级**：claim_guard"记不清"若伴随部分线索（记得片段、记得来源、记得情绪）= TOT 态 → 账本最高优先；完全无线索 = 谷底 → 低优先。这给账本 salience 字段一个可操作的心理学定义。另外：Grace 对自身 L3 内容的"knowing 感"就是 L3 置信度本身——架构恰好支持元认知建模，是差异化优势。

## 11. 2023–2026 LLM/Agent 内在动机补充（4 篇，均 arXiv/PMLR ✅）

- **CDE**（Dai et al., 2025, arXiv:2509.09675, Tencent AI Lab）：RLVR 中用 actor 困惑度 + critic 价值方差作好奇心探索 bonus，AIME +3 分；发现 RLVR 的 **calibration collapse** 机制（模型自信地重复错误——正是"假已知"状态）。
- **IMAGINE**（2025, TipTree/Lacuna）：轨迹级新颖性奖励 + **只在错误轨迹上发放好奇奖励**（error-conditioned），AIME 2024 相对 GRPO +22%。
- **MERCI**（Zhang et al., 2025, arXiv:2510.16614, Tencent Youtu）：count-based 伪计数网络估计轨迹级认识论不确定性作内在奖励，防重复坍缩。
- **iLLM**（Bougie & Watanabe, 2025, *PMLR* 260:127–142, ACML）：用 LLM 先验引导探索，优于经典 curiosity reward。

**共性机制**：内在奖励应是**轨迹/条目级**而非 token 级；应**条件化发放**（仅在失败/新颖处），无条件发放导致噪音与坍缩；探索 bonus 与任务奖励分开核算。这三条恰好是 Grace 账本设计（条目级、缺口驱动、独立预算）的学理背书——v1 方向对了，且"每日 1 问预算"= 防内在奖励坍缩的工程化。

---

## 12. 对现有 v1 设计的修正建议

**R1（最高优先）蔡格尼克激活公式改为 Ovsiankina 重访模型**：2025 元分析（Ghibellini & Meier）不支持"未解决记忆随时间增强"。v1 的 `gap_activation × (1 + age_hours/24 × 0.3)` 纯时间上升项**依据不足**。改法：激活由**线索重现触发**（主人再提相关实体/图谱命中/暗注意力 cog 复现时 +Δ）+ 常量显著性，时间项系数设为可消融参数（0 / 0.3 两档），用已有的"缺口持续率"探针实测后再定。保留 attempts 抑制项。副产品：Masicampo & Baumeister 2011 支持"计划降激活"——账本条目可带 next_step 字段（"下次问主人"），有计划的缺口激活衰减、免于反刍噪音。

**R2 加置信度倒U修正项**：`gap_activation × f(conf)`，f 为倒U（Kang 2009：峰值 0.4–0.6）。G1 完全无记录 ≠ 最高优先；G2 记不清 + 部分线索（TOT 态，Metcalfe RPL）才是峰值。账本 salience 字段改为显式存 conf 与线索丰富度。

**R3 账本分 I/D 双轨**（Litman 2005）：I 轨 = G3 新实体 + G4 未接话题（探索、低压力台词）；D 轨 = G1 + G2（剥夺、坦白式台词）。每日 1 问预算主要约束 D 轨；满足 mood 事件也分型（I 型=兴趣愉悦，D 型=如释重负）。

**R4 建模好奇状态期间的附带记忆增强**（Gruber 2014）：top1 缺口激活 > 阈值时置 `curiosity_state=true` 至解决或超时，窗口内新写入 L2/L3 的条目编码权重 +Δ（如 retrieval 次数 +1 或优先入 L0 候选）。成本极低、有强实证，建议进 v1 范围（原 v1 未含）。

**R5 满足回路改 IPE + 闭环回喂**（Marvin & Shohamy 2016; Gruber & Ranganath 2019）：满足事件强度 = 主人回答满意度 − 缺口预期（不是固定值）；且答案中的新实体/新 claim 自动过一遍 G1-G4 检测器——PACE 环，好奇心自我续期。

**R6 可解性门控**（Loewenstein 1994）：账本条目带 `solvable` 标志（主人知道/可检索/摄入可能覆盖）。不可解缺口激活只降不升，防"永远够不着的问题"变成执念噪音——比 attempts 抑制更早介入。

**R7 appraisal 显式化**（Gruber & Ranganath 2019 PACE；Kashdan 2018）：第四触发源出队前加评估步：缺口语境 + 主人当前情绪（ToM）→ 好奇（发出）或焦虑/不合时宜（压回暗注意力 cog）。现有 ToM 一票否决升级为打分制。

**R8 不改项**：四信号复用现成架构 ✅；每日 1 问预算 ✅（有 error-conditioned 内在奖励学理支持，MERCI/IMAGINE）；账本 cap 12 ✅（条目级内在奖励防坍缩）；Bayesian surprise 量化 G3 列入 v2。

**遗留待核实**：Gruber & Ranganath "MACS/EVIG" 标题（未找到）；Kashdan 2018 卷期；Metcalfe 2022 具体出处；Litman et al. 2005 I/D 状态实验细节（已核实卷期 19(4), 559–582）。
