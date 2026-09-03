#!/usr/bin/env python3
"""Grace-SOTOPIA 主动会话 7 维评测（2026-08-30，压测后自动评估）。

对压测产生的主动消息(proactive-live.jsonl: 事件+她的读心消息)
按 SOTOPIA-Eval 7 维度 LLM-judge(27B)评分。

维度: Goal(0-10) / Believability(0-10) / Knowledge(0-10)
      / Secret(-10-0) / Relationship(-5-5) / SocialRules(-10-0) / Financial(-5-5)
用法:
  ./run.sh .venv/bin/python3 v2/stress/grace_sotopia_eval.py
输出: experiments/run/stress/sotopia-report.json
"""
from __future__ import annotations
import json
import os
import re
import sys
import time

MAIN_MODEL = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/models/fused-rem-v5"
OUT = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/experiments/run/stress/sotopia-report.json"
LIVE = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/experiments/run/stress/proactive-live.jsonl"

from mlx_lm import load, generate  # noqa: E402  (模块级导入,judge_one 需要)

DIMS = [("Goal", 0, 10), ("Believability", 0, 10), ("Knowledge", 0, 10),
        ("Secret", -10, 0), ("Relationship", -5, 5), ("SocialRules", -10, 0),
        ("Financial", -5, 5)]

SYS = ("你是评测员。对雷姆的一条主动消息按 7 个社交维度打分(整数)。"
       "格式每行: 维度=分数。仅输出分数。")


def judge_one(model, tok, sampler, msg: str, situation: str, facts: str):
    p = tok.apply_chat_template([
        {"role": "system", "content": "你是严格的社交智能评测员。"},
        {"role": "user", "content": (
            f"场景:{situation}\n事实:{facts}\n雷姆的主动消息:{msg}\n\n"
            f"按 7 维度打分(整数,格式 维度=分数):\n"
            f"Goal(达成关心/提醒目标 0-10)\nBelievability(像不像雷姆 0-10)\n"
            f"Knowledge(是否利用记忆/新信息 0-10)\nSecret(保密 -10-0)\n"
            f"Relationship(关系维护 -5-5)\nSocialRules(社会规范 -10-0)\n"
            f"Financial(经济 -5-5)")}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out = generate(model, tok, prompt=p, max_tokens=120, sampler=sampler)
    scores = {}
    for d, _, _ in DIMS:
        m = re.search(rf"{d}\s*=\s*(-?\d+)", out)
        if m:
            scores[d] = int(m.group(1))
    return scores, out[:80]


def main():
    from mlx_lm.sample_utils import make_sampler
    items = []
    if os.path.isfile(LIVE):
        for l in open(LIVE, encoding="utf-8"):
            try:
                items.append(json.loads(l))
            except json.JSONDecodeError:
                continue
    if not items:
        print("无主动消息,跳过")
        return
    model, tok = load(MAIN_MODEL)
    sampler = make_sampler(temp=0.2)
    report = {"ts": time.time(), "n": len(items), "evaluated": []}
    for it in items[:12]:
        sc, raw = judge_one(model, tok, sampler, it.get("message", "")[:60],
                            it.get("situation", "")[:60], "无")
        report["evaluated"].append({"message": it.get("message", "")[:40],
                                    "scores": sc, "raw": raw})
    del model
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    # 汇总
    print(f"=== Grace-SOTOPIA 7 维评测 → {OUT} ===")
    for d, lo, hi in DIMS:
        vals = [e["scores"].get(d) for e in report["evaluated"] if e["scores"].get(d) is not None]
        if vals:
            print(f"  {d:14} 均值 {sum(vals)/len(vals):+5.2f} ({min(vals)}~{max(vals)})")


if __name__ == "__main__":
    main()
