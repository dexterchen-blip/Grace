#!/usr/bin/env python3
"""训练报告生成 —— 每次训练只出一个报告（用户定下的产物规格）。

报告落点：experiments/run/YYYYMMDD-HHMM-lora-report.md
内容：数据集摘要 / 超参 / 训练结果 / 快照 / 验证建议。纯本地 markdown，供 dashboard 与人类阅读。
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime

import config  # noqa: E402


def _fmt_list(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items)


def build_report(*, candidate_ids: list[str], dataset_stats: dict,
                 lora_cfg: dict, adapter_path: str, snapshot_hash: str,
                 loss_points: list[dict] | None = None,
                 notes: str = "") -> str:
    """生成一次训练的报告（markdown）。"""
    now = datetime.now()
    loss_txt = "（训练未执行 / 无 loss 数据）"
    if loss_points:
        loss_txt = "```\n" + "\n".join(
            f"iter={p.get('iter')} loss={p.get('loss')}" for p in loss_points) + "\n```"

    md = f"""# LoRA 训练报告（Grace V2 权重轨）

> 生成时间：{now.strftime('%Y-%m-%d %H:%M:%S')}
> 人格锚点：{config.PERSONA['display']}（`{config.PERSONA['name']}`）

## 1. 本次训练数据集

- 来源候选：{_fmt_list(candidate_ids) if candidate_ids else '（无候选，纯锚点回放）'}
- 样本统计：{json.dumps(dataset_stats, ensure_ascii=False)}
- 双轨过滤：事实类样本已在提炼阶段剔除（见 candidate_extract），本报告不再含事实。

## 2. 超参（低秩 · 微量）

```json
{json.dumps(lora_cfg, ensure_ascii=False, indent=2)}
```

## 3. 训练结果

- 适配器产物：`{adapter_path}`（若未训练则为 `-`）
- 权重快照：`{snapshot_hash or '-'}`（git 可回滚点）

{loss_txt}

## 4. 验证建议（人格一致性尺子）

1. 同情境多次生成：语气/自称/短句稳定，无占位符泄漏。
2. 事实冲突检测：LoRA 输出与 L2 检索冲突时，外挂轨优先（系统层拦截）。
3. 本适配器**隔天生效**（24h 反悔窗口；不满意可回滚快照 {snapshot_hash or '-'}）。

## 5. 备注

{notes or '（无）'}
"""
    os.makedirs(config.REPORTS, exist_ok=True)
    fn = now.strftime("%Y%m%d-%H%M-lora-report.md")
    path = os.path.join(config.REPORTS, fn)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[v2-report] 报告已生成：{path}")
    return path


def list_reports() -> list[dict]:
    out = []
    if os.path.isdir(config.REPORTS):
        for fn in sorted(os.listdir(config.REPORTS), reverse=True):
            p = os.path.join(config.REPORTS, fn)
            if fn.endswith(".md"):
                out.append({"name": fn, "path": p,
                            "size": os.path.getsize(p),
                            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))})
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        for r in list_reports():
            print(f"{r['mtime']}  {r['name']}  {r['size']}B")
        sys.exit(0)
    print(build_report(candidate_ids=[], dataset_stats={"style": 0},
                       lora_cfg=config.LORA, adapter_path="-", snapshot_hash="-",
                       notes="CLI 冒烟：手动生成示例报告"))
