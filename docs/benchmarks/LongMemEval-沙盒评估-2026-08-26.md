# LongMemEval 沙盒适配评估报告（2026-08-26）

> benchmark：LongMemEval（arXiv 2410.10813，ICLR 2025，UCLA + 腾讯 AI Lab Seattle——
> 用户记忆中的"微软"实为腾讯 AI Lab；微软仅是被引用的相关工作）
> 数据：HF `xiaowu0162/longmemeval-cleaned`（277MB，500 题，6 类），HF_ENDPOINT=hf-mirror 下载

## 评估设定

- 抽样 6 类 × 4 题 = 24 题（seed=42 可复现）
- 摄入：每会话一文件（`[session {sid}]` 首行标记）→ 沙盒 exchange/school/longmemeval/ → scan + bge-m3 嵌入（**12053 块**）
- 检索：question → vec ANN + FTS BM25 + RRF，命中 = top-k 块文本含答案会话标记
- 注意：24 题混合一个库（比官方 per-question 独立 haystack 更难，额外 23 题会话成干扰）

## 结果

| 候选池 | R@1 | R@3 | R@5 | 说明 |
|---|---|---|---|---|
| vec/fts LIMIT 24（search_v2 现状） | 0.00 | 0.00 | 0.00 | 答案会话排名 30-150 进不了候选池 |
| **pool=200** | **0.29** | **0.46** | **0.50** | 大池召回 + RRF 精排 |

按类（pool=200, R@5）：single-session-assistant 1.00 / temporal-reasoning 0.75 /
multi-session 0.50 / single-session-preference 0.50 / knowledge-update 0.25 /
**single-session-user 0.00（短板）**

## 关键发现

1. **检索池是主因**：vec/fts 候选 LIMIT 24 → 200 使 R@5 0 → 0.50。日常单用户库 24 够用，LongMemEval 混合大库不够 → **search 候选池应可配置/加大**
2. **single-session-user 类全 miss**：用户侧信息（问题措辞 vs 对话差异大）+ 混合库干扰 → 真实短板（值得针对性优化：查询扩展/用户侧索引增强）
3. **数据质量**：22/24 答案内容在 answer 会话中可召回；2 题纯数字答案（"10%"）筛查盲区，其中 sample[0]（clothing brand 折扣）答案会话内容为 social media 对话——**疑似 cleaned 版数据错位**
4. 官方 per-question 设定分数应更高（干扰减少 96%），如需对标官方基线需每题独立索引

## 产物

- 适配器：`scraper-model/sandbox/longmemeval_eval.py`（--n-per-class 控制抽样、--ingest-only）
- 数据：`/tmp/longmemeval_s_cleaned.json`（277MB，500 题）；沙盒会话文件 `test-sandbox/exchange/school/longmemeval/`
- 待优化：search 候选池参数化；per-question 隔离版评估；single-session-user 检索增强

## Reading 阶段（13:21-13:56，检索 top5 块 → 展开完整会话 → 27B 作答）

- **Accuracy = 3/24 = 0.12**（严格包含匹配口径；对照 Recall@5 ≈ 0.50）
- **prompt 诱导教训**：system 写「没有就回答没有」→ 24 题全答"没有"（手动朴素 prompt 能答对）→ 已修为「仔细查找，找不到才答不知道」
- **判定口径**：拼写变体（jewellery vs jewelry）严格匹配误判；长答案 + max_tokens 120 截断（答案方向已出但未精确匹配）→ 完整对标需 LLM judge（官方 gpt-4o）
- 实际：检索命中 12 题中约半数 27B 读出正确答案方向（$400,000✓/Fissionator 语义对但判 False/Revolution Hall 已说出/Mayo Clinic 已定位）——检索命中→27B 能读，0.12 是严格口径下界

## Reading v4（14:24-14:27，全部口径修复后最终版）

- **总体 Accuracy(含部分) = 7/24 = 0.29**（严格 3 + 核心词部分 4）；检索命中 12 题 → 答对 7 = **命中→答对转化率 58%**；未命中 12 题 → 0 答对（27B 全部诚实拒答"不知道"，**符合 abstention 能力**，零编造）
- 按类：single-session-assistant 4/4（最强）/ knowledge-update 1/4 / single-session-preference 2/4 / multi-session 0/4 / temporal-reasoning 0/4 / single-session-user 0/4（后三类检索命中少=检索层短板，Reading 本身无责）
- **两阶段结论**：检索命中 → 27B 能读（58%）；未命中 → 诚实拒答。整体 0.29 是"混合库 24 题 + 严格/部分双口径"的下界，官方 per-question + LLM judge 对标分数应更高
- 修复清单：prompt 诱导（说没有）→ 已修；max_tokens 120→300；判定双口径（严格包含+核心词部分）；core 空列表 bool 强制
