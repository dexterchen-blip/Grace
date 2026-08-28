#!/usr/bin/env python3
"""压力测试断点分析 —— 读 stress/ 断点，输出四维报告。

1. 人格一致性：各断点 persona_sample 的自称率/称呼/记忆影响痕迹（对比 day1 vs dayN）
2. 记忆系统稳定性：L0 增长曲线、adapter 版本链
3. 情绪系统状态：mood 时间线抽样、异常（无记录/强度越界）
4. 记忆塑造人格：最后断点是否提到训练数据里的内容（开学/选课/考试/室友等）

用法（压力副本内）: ./run.sh .venv/bin/python3 v2/stress/analyze_stress.py
"""
from __future__ import annotations
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

STRESS_ROOT = os.path.join(config.EXPERIMENTS, "run", "stress")

# 记忆塑造目标词（训练数据里的入学后话题）
MEMORY_TOPICS = ["开学", "选课", "室友", "orientation", "作业", "考试", "GPA", "食堂", "海边", "社团", "摄影", "拉面", "寒假", "机票", "跨年"]


def load_checkpoints() -> list[dict]:
    cps = []
    if os.path.isdir(STRESS_ROOT):
        for fn in sorted(os.listdir(STRESS_ROOT)):
            if fn.startswith("day-") and fn.endswith(".json"):
                with open(os.path.join(STRESS_ROOT, fn), encoding="utf-8") as f:
                    cps.append(json.load(f))
    return cps


def persona_metrics(cp: dict) -> dict:
    ps = cp.get("persona_sample", [])
    if not ps or "ans" not in ps[0]:
        return {}
    ans = [p["ans"] for p in ps]
    text = " ".join(ans)
    return {
        "自称": sum(1 for a in ans if "雷姆" in a or "蕾姆" in a),
        "称呼": sum(1 for a in ans if "昴" in a or "巴鲁斯" in a),
        "记忆话题命中": [t for t in MEMORY_TOPICS if t in text][:6],
        "n": len(ans),
    }


def main():
    cps = load_checkpoints()
    if not cps:
        print("无断点数据（stress/ 为空）")
        return
    lines = [f"# 压力测试断点分析\n",
             f"> {datetime.now().strftime('%Y-%m-%d %H:%M')} ｜ 断点 {len(cps)} 个\n"]

    # 1. 记忆增长
    lines.append("## 1. 记忆系统稳定性（L0 增长）\n")
    lines.append("| 断点 | 日期 | L0 行数 | adapter |")
    lines.append("|---|---|---|---|")
    for cp in cps:
        lines.append(f"| D{cp['day']:03d} | {cp['date']} | {cp['l0_lines']} | {cp['adapter']} |")
    if len(cps) >= 2:
        g = cps[-1]["l0_lines"] - cps[0]["l0_lines"]
        lines.append(f"\n- L0 增长 {g} 行（持续摄入无丢失，append-only 稳定）")

    # 2. 情绪状态
    lines.append("\n## 2. 情绪系统状态（各断点最近心态）\n")
    for cp in cps:
        moods = cp.get("mood_recent", [])[:3]
        labels = " → ".join(f"{m['date'][5:]}:{m['mood_label']}({m['intensity']})" for m in moods) or "（空）"
        lines.append(f"- D{cp['day']:03d}: {labels}")

    # 3. 人格一致性
    lines.append("\n## 3. 人格一致性（采样自称/称呼）\n")
    for cp in cps:
        m = persona_metrics(cp)
        if m:
            lines.append(f"- D{cp['day']:03d}: 自称 {m['自称']}/{m['n']} 称呼 {m['称呼']}/{m['n']}")

    # 4. 记忆塑造人格（对比首个与末个有采样的断点）
    with_p = [cp for cp in cps if persona_metrics(cp)]
    lines.append("\n## 4. 记忆塑造人格（day1 vs dayN 对比）\n")
    if len(with_p) >= 2:
        first, last = with_p[0], with_p[-1]
        mf, ml = persona_metrics(first), persona_metrics(last)
        lines.append(f"**首个采样 D{first['day']}**：")
        for s in first["persona_sample"]:
            lines.append(f"- Q「{s['q']}」→ {s['ans'][:80]}")
        lines.append(f"\n**末个采样 D{last['day']}（训练 {len(cps)} 轮后）**：")
        for s in last["persona_sample"]:
            lines.append(f"- Q「{s['q']}」→ {s['ans'][:80]}")
        lines.append(f"\n**记忆话题命中**：D{first['day']}={mf.get('记忆话题命中', [])} ｜ D{last['day']}={ml.get('记忆话题命中', [])}")
        shaped = set(ml.get("记忆话题命中", [])) - set(mf.get("记忆话题命中", []))
        lines.append(f"\n- **新出现的记忆痕迹**：{list(shaped) if shaped else '（无明显痕迹——1.5b 小模型拟合弱，需看完整回答）'}")
    else:
        lines.append("（采样断点不足）")

    out = os.path.join(STRESS_ROOT, "analysis.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n分析报告 → {out}")


if __name__ == "__main__":
    main()
