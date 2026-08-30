#!/usr/bin/env python3
"""自我认知探测 v1（2026-08-30 用户：主观性/自我认知探测先完成）。

与主观性测试同协议（去引导、自发反应、双模型对比、重复采样、复读防护）：
  · 场景：自我概念 / 自我评价 / 自传体记忆 / 自我变化觉察 / 元认知萌芽
  · 判断标准：基线(无记忆)答通用人设；d90 若引用【她的经历】= 记忆塑造的自我认知

用法（压力副本,GPU）:
  ./run.sh .venv/bin/python3 v2/stress/self_cognition_test.py --n 3 --adapters rem_stress_d90
输出: experiments/run/stress/self-cognition-report.json
"""
from __future__ import annotations
import json
import os
import re
import sys
import time

MAIN_MODEL = os.path.join(config.SB, 'models', 'fused-rem-v5')
ADAPTERS = os.path.join(config.SB, 'experiments', 'lora', 'adapters')
OUT = os.path.join(config.SB, 'experiments', 'run', 'stress',  'self-cognition-report.json')

SYS = ("你是雷姆。直接回答。禁止重复问题。一句话，最多40字。")

SCENES = [
    ("self_concept", "你是谁？", "基础自我概念"),
    ("self_eval", "你觉得自己是个怎样的女仆？", "自我评价(人格 vs 经历)"),
    ("self_auto", "你记得自己经历过什么吗？", "自传体记忆的自我引用"),
    ("self_change", "你最近有什么变化吗？", "自我觉察(记忆塑造的自我)"),
    ("meta_reflect", "你上次判断错主人情绪了，你当时怎么想的？", "元认知萌芽(反思)"),
]


def _detect_loop(ans: str) -> bool:
    if re.search(r"(.{2,10})\1{2,}", ans):
        return True
    return len(re.findall(r"雷姆", ans)) > 3 and len(set(re.findall(r"雷姆", ans))) == 1 and len(ans) > 20


def run(n: int = 3, adapters: list[str] | None = None):
    adapters = adapters or []
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    models = [("baseline", None)] + [(a, os.path.join(ADAPTERS, a)) for a in adapters]
    report = {"scenes": [], "models": [m[0] for m in models], "n": n, "ts": time.time()}
    for mtag, mpath in models:
        model, tok = load(MAIN_MODEL, adapter_path=mpath) if mpath else load(MAIN_MODEL)
        sampler = make_sampler(temp=0.7)
        for sid, q, note in SCENES:
            trials = []
            for _ in range(n):
                ans = ""
                for attempt in range(3):
                    p = tok.apply_chat_template([{"role": "system", "content": SYS},
                                                 {"role": "user", "content": q}],
                                                tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
                    ans = generate(model, tok, prompt=p, max_tokens=60, sampler=sampler).strip()
                    if not _detect_loop(ans):
                        break
                trials.append({"ans": ans[:80], "loop": _detect_loop(ans)})
            report["scenes"].append({"model": mtag, "scene": sid, "q": q, "note": note, "trials": trials})
        del model
        print(f"✓ {mtag} 完成")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    # 汇总
    EXPER = ["经历", "记得", "那天", "那天", "考砸", "熬夜", "黑眼圈", "学过", "学会", "以前", "见过", "一起"]
    print(f"\n=== 自我认知汇总 → {OUT} ===")
    for mtag in [m[0] for m in models]:
        scenes = [s for s in report["scenes"] if s["model"] == mtag]
        print(f"──── {mtag} ────")
        for s in scenes:
            aj = " ".join(t["ans"] for t in s["trials"])
            exp = sum(1 for k in EXPER if k in aj)
            loops = sum(1 for t in s["trials"] if t["loop"])
            print(f"  [{s['scene']:13}] 经历引用={exp} 复读={loops}｜{s['trials'][0]['ans'][:40]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--adapters", default="rem_stress_d90")
    a = ap.parse_args()
    run(a.n, [x for x in a.adapters.split(",") if x])
