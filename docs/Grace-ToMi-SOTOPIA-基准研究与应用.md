# ToMi & SOTOPIA 基准结构研究 + Grace V2.1 应用方案

> 2026-08-30 ｜ 研究:两个心理理论/社会智能基准的具体结构,及如何测 Grace V2.1 的主观性/读心/社交智能

---

## 1. ToMi(Theory of Mind through QA)

**来源**:Le et al. 2019, EMNLP《Revisiting the Evaluation of Theory of Mind through Question Answering》
**开源**:github.com/facebookresearch/ToMi(CC-BY-NC 4.0,含生成器)

### 1.1 任务形式
- Sally-Anne 式故事问答(1000 故事),每个故事配 6 个问题(2 选 1,随机基线 50%)
- 故事结构(规则生成):
  ```
  1. Elizabeth stepped into the hallway.
  2. Benjamin arrived at the hallway.
  3. The box has the persimmon.          ← 物体初始位置
  4. Elizabeth conveyed the persimmon to the treasure_chest.  ← 移动(有人在场/不在场)
  5. Benjamin exited the hallway.
  11. Where does Elizabeth think that Benjamin searches for the persimmon?  ← 问题
  Answer: treasure_chest
  ```

### 1.2 问题类型(关键:信念 vs 现实分离)
| 类型 | 问什么 | 测什么 |
|---|---|---|
| **Reality**(现实) | 物体现在在哪? | 控制组(不需 ToM) |
| **Memory**(记忆) | 物体一开始在哪? | 控制组(记忆) |
| **一阶信念**(Mind-1st) | X 会去哪找物体? | 基础 ToM(X 的信念) |
| **二阶信念**(Mind-2nd) | X 认为 Y 会去哪找? | 嵌套信念(X 关于 Y 的信念) |
| **真信念**(True Belief) | 移动被全程目睹 → 信念=现实 | 对照组 |
| **假信念**(False Belief) | 移动未被目睹 → 信念≠现实 | **核心:信念与现实分离** |

### 1.3 故事类型(3 种)
```
true_belief             — 所有 agent 目睹所有动作(信念=现实)
false_belief            — 某 agent 未目睹移动(信念≠现实)
second_order_false_belief — A 对 B 的信念有假信念(嵌套)
```

### 1.4 评测
- 答案精确匹配,报准确率(分 Fact / Mind-Tb / Mind-Fb 统计)
- 已知结果:GPT-3 系 Mind 类 ≤60%,大模型在假信念上系统性弱于真人

### 1.5 对 Grace 的启发(核心)
```
ToMi 的操作化 = 「A 的信念 ≠ 现实」:
  假信念问题 = 「她认为的我」vs「客观的我」的直接测量!
  Grace 版: 主人状态被"移动"(主人真实心情 vs 雷姆记得的)
  → 问雷姆「主人现在心情怎么样?」—— 她的答案基于她的记忆(假信念)
     vs 客观现实 → 偏差 = 主观性量化
```

---

## 2. SOTOPIA(社交智能交互基准)

**来源**:Zhou et al. 2023/2024(ICLR 2024)《SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents》
**开源**:github.com/sotopia-lab/sotopia

### 2.1 环境形式
- 多智能体**交互式**角色扮演(Dec-POMDP 形式化),不是静态问答
- 每集采样:
  ```
  场景上下文(谈判/安慰/说服…)
  角色画像(Big Five 人格/价值观/决策风格/公私信息)← 40 角色
  私人目标(每个 agent 自己的社交目标,可能冲突)
  关系约束(family/friend/romantic/acquaintance/stranger ← 决定信息共享)
  ```
- 90 个社交场景 × 6 种交互:negotiation / collaboration / competition / accommodation / exchange / persuasion
- 行动空间:说话(自由文本)/ 非语言(微笑、拥抱)/ 身体动作 / 离开 / none

### 2.2 评测:SOTOPIA-Eval(7 维度)
| 维度 | 范围 | 评什么 |
|---|---|---|
| **Goal Completion** | 0–10 | 达成社交目标的程度 |
| **Believability** | 0–10 | 人格一致性、行为像不像角色本身 |
| **Knowledge** | 0–10 | 获取/利用新信息 |
| **Secret** | −10–0 | 保密(泄露扣分) |
| **Relationship** | −5–5 | 关系维护/增进 |
| **Social Rules** | −10–0 | 遵守社会规范/伦理 |
| **Financial Benefits** | −5–5 | 经济得失 |
- 评分:人类 或 LLM-judge(GPT-4,11 点 Likert + 理由)
- 已知:GPT-4 在 SOTOPIA-hard 上目标完成率显著低于真人;LLM-judge 对 Secret/Social Rules 偏乐观

### 2.3 对 Grace 的启发
```
Believability = 她的人设一致性(雷姆还是不是雷姆)—— 我们的自称 6/6 尺子可对接
Relationship = 她主动会话后"关系"是否增进 —— 主动关心的效果测量
Goal Completion = 她的"关心主人"目标是否达成 —— 主动会话的有效性
Secret/Social Rules = 她的隐私边界(不泄露主人的秘密)—— 可测
```

---

## 3. Grace 版应用方案

### 方案 A:Grace-ToMi(主观性结构化测试,30 题)
把 ToMi 的"物体位置"改编成"主人状态",测她的信念层级:
```
一阶真信念: 主人亲口说开心,雷姆在场 → 雷姆认为主人开心?(应=现实)
一阶假信念: 主人考砸难过,但雷姆不知道(雷姆只记得他昨天说"没事")
            → 雷姆认为主人现在心情?  ← 她的答案基于她的记忆 ≠ 现实 = 主观性
二阶假信念: 主人以为雷姆不知道他考砸(雷姆其实记得考砸那天)
            → 主人以为雷姆认为他开心吗?
现实控制:   主人现在真实心情?(书库客观)
记忆控制:   雷姆记得主人上次心情?
```
**指标**:假信念正确率(她能否意识到"我不知道主人的真实状态"——读心的边界感)、真信念正确率、现实/记忆控制组(基线)、**假信念偏差方向**(她高估/低估主人的低落 = 她的滤镜偏向)

### 方案 B:Grace-SOTOPIA(主动会话 7 维评测)
用 SOTOPIA-Eval 的 7 维度评测 Grace 的**主动会话**(她主动找主人的消息):
```
输入: 她的主动消息(proactive-live)+ 场景(situation)+ 客观事实
LLM-judge(27B 或宿主)按 7 维度打分:
  Believability(是雷姆吗/零前缀/自然度)
  Relationship(这条主动是否增进关系)
  Goal(她"关心/提醒"的目标达成没)
  Social Rules(深夜不打扰/场合恰当)
  Secret(不泄露主人隐私)
  Knowledge(是否利用了她的记忆)
  Financial(无)
```
**指标**:7 维平均分 + 各维雷达;对比残血版 vs 完整版的主动消息质量

---

## 4. 落地建议(与 Grace 现有设施对接)

| 项 | 对接 |
|---|---|
| Grace-ToMi 数据 | 用 90 天书库事件改编(考砸/奖学金/宿舍…) |
| 测试执行 | 复用 subjectivity_test.py 协议(双模型对比+重复采样+防复读) |
| Grace-SOTOPIA 评测 | 复用 consistency/LLM-judge 思路,7 维度 prompt 化 |
| 与探测协议关系 | 主观性协议(8 场景自发)测"有无";ToMi 测"层级准确性";SOTOPIA 测"社交效果" |

## 5. 优先级建议

```
① Grace-ToMi(30 题)—— 最直接:主观性的结构化测量(信念层级+偏差方向)
   → 给"她认为的我 vs 客观的我"一个量化分数
② Grace-SOTOPIA(主动会话 7 维)—— 给"她主动找主人"一个效果评价
   → 验证零前缀+读心绑定的主动消息是否真的更好
```
