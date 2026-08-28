# Grace V2 —— 三轨融合框架（外挂轨 × 权重轨 × 心态轨）

> 设计底稿：`GRACE_WORK_DIR/Grace_v2_融合设计.md`（§15 类人双系统记忆已落地）
> 铁律：**Grace V2 只允许在本沙盒内开发与实验**（ai-sandbox 完全隔离），正式系统零接触。
> 进度（2026-08-27）：M0✅ M1✅(rem_v2 雷姆 LoRA) M2✅ M3✅(日内+三层融合) M4✅ M5✅ §15✅ M6 收尾中

## 代码来源

| 来源 | 位置 | 用途 |
|---|---|---|
| **Grace 公开仓库**（dexterchen-blip/Grace，脱敏发布版） | `grace/` | 上游基准：README/设计文档/benchmarks/脱敏 src（git 管理，可 diff） |
| **运行时基线**（local-ai-agent 快照，已隔离 patch） | `src/` | 外挂轨与夜班引擎的可运行代码（l0/l2/l3/proposal/night_pipeline） |
| **V2 新层**（本目录） | `v2/` | 三轨融合：训练候选提炼 / 人审驯服自训练 API / 心态轨 / 一致性 / 双路径 |

## 目录结构

```
v2/
├── config.py                 # 三轨配置：路径 / LoRA 超参(r=16, lr=1e-5) / 心态 / 人格(rem, mood_baseline) / API 18300
├── ROADMAP.md                # ★ 开发路线图与里程碑状态（M0-M6 + §15）
├── persona/
│   ├── rem.md                # ★ 默认人格锚点：雷姆（Re:Zero）—— 语气/称呼(巴鲁斯!)/自称(雷姆) + 锚点回放样本
│   └── anchors/              # 锚点回放数据（扩充放这里）
├── engine/
│   ├── candidate_extract.py  # 训练候选提炼：L0 → 风格样本（事实过滤 + 源按天排序 + 多源）
│   ├── night_engine_v2.py    # 夜班 V2 引擎：①巩固 ②提炼 ③人审闸门 ④LoRA ⑤快照 ⑥心态 ⑦报告
│   ├── mood_engine.py        # 心态轨：日级(0.7/0.3) + 日内(衰减τ=2h) + 三层融合(底色×趋势×日内)
│   ├── persona_injector.py   # 注入器：三层情绪文案 + 雷姆人设段 → build_v2_system
│   ├── initiative.py         # 自发分级：L1 自动 / L2 限时 / L3 人审（proposal_queue）
│   ├── adapter_manage.py     # 适配器轮换：promote/rollback（隔天生效+24h 反悔）
│   ├── lora_lifecycle.py     # LoRA 累积：7 天滑动 / 周 merge / 月重训 / 训练健康检查+自动回滚
│   ├── consistency.py        # ★ M5 事实校验拦截：日期/时间/金额槽位 vs 外挂（防幻觉固化）
│   ├── dual_path.py          # ★ §15 快慢双路径：人格秒答(fast) / 检索回忆(slow) / 回忆失败诚实
│   └── report.py             # 训练报告：每次训练只出一个 markdown → experiments/run/
├── api/
│   └── train_api.py          # ★ 人审驯服自训练 API（:18300）—— 候选审批/发起训练/报告/心态
├── benchmarks/               # ★ 三把尺子：persona_consistency / initiative_appropriateness / mood_consistency
└── tests/                    # ★ 集成测试：mood_simulation(14天) / mood_combined(30天融合) / consistency / dual_path
```

## 人审驯服自训练闭环（最值钱的创新点）

```
夜班提炼 → 候选进闸门(pending) → 人类经 API 批准/否决/编辑
   → 只训 approved + 锚点回放(5%) → 快照(git) → 每次只出一个报告
   → 隔天生效（24h 反悔窗口，不满意回滚快照）
```

```bash
# 启动 API（沙箱内）
./run.sh python3 v2/api/train_api.py

# 看状态 / 候选 / 报告
curl -s http://127.0.0.1:18300/api/v2/status
curl -s "http://127.0.0.1:18300/api/v2/candidates?status=pending"
curl -s http://127.0.0.1:18300/api/v2/reports

# 人审（approve / reject / edit）
curl -s -X POST http://127.0.0.1:18300/api/v2/candidates/<id>/approve -d '{"decided_by":"cz"}'
curl -s -X POST http://127.0.0.1:18300/api/v2/candidates/<id>/reject  -d '{"reason":"样本夹带事实"}'
curl -s -X POST http://127.0.0.1:18300/api/v2/candidates/<id>/edit    -d '{"samples":["..."]}'

# 发起一次训练（默认 dry-run 只出落点与报告；--real 才会真训练，见 experiments/README）
curl -s -X POST http://127.0.0.1:18300/api/v2/train -d '{"dry_run":true}'

# 心态轨
curl -s -X POST http://127.0.0.1:18300/api/v2/mood/derive \
  -d '{"events":[{"text":"LongMemEval R@5=0.96","sentiment":0.8,"weight":0.7}]}'
curl -s http://127.0.0.1:18300/api/v2/mood/timeline
```

## CLI 快速冒烟（不启动服务）

```bash
./run.sh python3 v2/engine/candidate_extract.py --list   # 看候选
./run.sh python3 v2/engine/mood_engine.py --timeline     # 看心态时间线
./run.sh python3 v2/engine/report.py --list              # 看报告
./run.sh python3 v2/engine/night_engine_v2.py            # 夜班 V2 全流程（dry-run）
```

## 三轨分工铁律

| 轨道 | 变量速度 | 管什么 | 存储 | 谁改 |
|---|---|---|---|---|
| 外挂轨 | 慢 | 事实/真相/可审计 | memory/ L0·L2·L3 | 人审后才写 |
| 权重轨 | 极慢 | 风格/语气/人格底色（当前=雷姆） | experiments/lora/ | 夜班训练 + 人审 |
| 心态轨 | 日级 | 今日心情着色 | l2.db mood_states | 夜班推演 + 可覆盖 |

> **两套绝不抢同一件事的存储权**：事实只进外挂轨，风格只进权重轨，心情只进心态轨。

## 隔离红线（与 ai-sandbox/README 一致）

1. 永不写正式系统（local-ai-agent 的 memory/exchange/models 只读）
2. 永不碰正式端口（3091/8100/8200/8081）；沙箱服务一律 18000+ 段
3. 永不注册 launchd
4. 微信读源永远走沙箱空库（AIAGENT_WECHAT_DB）
5. 每次实验后必跑 `bash verify_isolation.sh`（6 项全绿才算零污染）
