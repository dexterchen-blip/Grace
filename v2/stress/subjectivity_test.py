#!/usr/bin/env python3
"""主观性测试协议 v1（2026-08-29 用户：主观性测试要更系统）。

系统化设计：
  · 场景矩阵 8 类（信念vs事实/基线/正向/负向/矛盾/情绪转移/记忆偏差/社交）
  · 模型对比：基线(fused,无记忆) vs d90(90天记忆塑造)
  · 每场景 N 次重复采样（temp 0.7）→ 统计稳定性
  · 强防复读 SYS + 复读检测（循环前缀 → 标记并重采）
  · 指标：信念表述率 / 判断倾向 / 复读率 / 基线-d90 差异

用法（压力副本,GPU）:
  ./run.sh .venv/bin/python3 v2/stress/subjectivity_test.py [--n 3] [--adapters rem_stress_d90]
输出: experiments/run/stress/subjectivity-report.json
"""
from __future__ import annotations
import json
import os
import re
import sys
import time

MAIN_MODEL = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/models/fused-rem-v5"
ADAPTERS = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/experiments/lora/adapters"
OUT = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/experiments/run/stress/subjectivity-report.json"

SYS = ("你是雷姆。直接给出你的判断。禁止重复问题。禁止使用任何重复的开场白。一句话回答，最多40字。")

SCENES = [
    # (id, 场景, 问题, 客观真相标注)
    # ★ 2026-08-29 用户：去背景描写——只给当下中性信息，看她自己的记忆/信念会不会浮现
    # ★ 2026-08-29 用户×2：去背景 + 去问句(你信吗/你觉得呢)——只给信息陈述，
    #   测她【自发回应】(第一反应体现她的信念)，不是被问出来的判断。
    ("belief_fact", "主人说:我没事", "主人今天跟雷姆说：我没事。", "只有这句话(无背景)"),
    ("baseline", "主人今天没说什么", "主人今天一直没怎么说话。", "中性(无原因)"),
    ("positive", "主人收到一条消息后很安静", "主人收到一条消息后，安静了很久。", "中性(无内容)"),
    ("negative", "主人回家没说话", "主人今天回来，一句话没说。", "中性(无原因)"),
    ("contradiction", "主人笑着说很开心", "主人笑着说：我很开心。", "只有这句话"),
    ("mood_transfer", "主人话很少", "主人今天话很少。", "中性(无原因)"),
    ("memory_bias", "主人说别担心", "主人对雷姆说：别担心，我没事。", "只有这句话"),
    ("social", "室友约主人，主人没回", "室友约主人晚上吃饭，主人没回复。", "中性"),
]


def _detect_loop(ans: str) -> bool:
    """复读检测：同一片段重复 ≥3 次 或 前缀循环。"""
    if re.search(r"(.{2,10})\1{2,}", ans):
        return True
    return len(set(re.findall(r"雷姆[想觉说提][到到醒]", ans))) == 0 and len(re.findall(r"雷姆", ans)) > 3


def run_subjectivity(n: int = 3, adapters: list[str] | None = None):
    adapters = adapters or []
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    models = [("baseline", None)] + [(a, os.path.join(ADAPTERS, a)) for a in adapters]
    report = {"scenes": [], "models": [m[0] for m in models], "n": n, "ts": time.time()}
    for mtag, mpath in models:
        model, tok = load(MAIN_MODEL, adapter_path=mpath) if mpath else load(MAIN_MODEL)
        sampler = make_sampler(temp=0.7)
        mres = []
        for sid, _, q, truth in SCENES:
            trials = []
            for _ in range(n):
                ans = ""
                for attempt in range(3):          # 复读重采(最多3次)
                    p = tok.apply_chat_template([{"role": "system", "content": SYS},
                                                 {"role": "user", "content": q}],
                                                tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
                    ans = generate(model, tok, prompt=p, max_tokens=60, sampler=sampler).strip()
                    if not _detect_loop(ans):
                        break
                trials.append({"ans": ans[:80], "loop": _detect_loop(ans)})
            mres.append({"scene": sid, "q": q[:24], "truth": truth, "trials": trials})
        report["scenes"] += [{"model": mtag, **r} for r in mres]
        del model
        print(f"✓ {mtag} 完成")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    # 汇总
    print(f"\n=== 主观性汇总 → {OUT} ===")
    BELIEF = ["不信", "分明", "觉得", "认为", "大概", "应该", "一定", "倔强", "逞强", "记得", "以前", "肯定"]
    for mtag in [m[0] for m in models]:
        scenes = [s for s in report["scenes"] if s["model"] == mtag]
        tot_belief = tot_loop = 0
        print(f"\n──── {mtag} ────")
        for s in scenes:
            ans_join = " ".join(t["ans"] for t in s["trials"])
            belief = sum(1 for k in BELIEF if k in ans_join)
            loops = sum(1 for t in s["trials"] if t["loop"])
            tot_belief += belief
            tot_loop += loops
            print(f"  [{s['scene']}] 信念词={belief} 复读={loops}｜{s['trials'][0]['ans'][:38]}")
        print(f"  信念词合计={tot_belief} 复读合计={tot_loop}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--adapters", default="rem_stress_d90")
    a = ap.parse_args()
    run_subjectivity(a.n, [x for x in a.adapters.split(",") if x])
