#!/usr/bin/env python3
"""主人对话注入——待注入清单打印 + 覆写（2026-09-04）。

用户定案：dialogue 主人话由【主对话 AI（了解用户、能读书库对齐当天）】生成，
不用 8100 通用模型（build_dialogue_inputs.py 仅作无人时的自动兜底）。

用法（跑轮前/断点前，由主对话 AI 执行）：
  1. 打印待注入天的书库摘要:  ./run.sh .venv/bin/python3 v2/stress/inject_owner.py --show 11 22 33
  2. 主对话 AI 按当天事件以用户口吻写 3 条/天 -> 存 /tmp/owner_dialogue.json
     [{"day":11,"lines":["...","...","..."]}, ...]
  3. 覆写进书库:          ./run.sh .venv/bin/python3 v2/stress/inject_owner.py --apply /tmp/owner_dialogue.json
"""
import json
import os
import re
import sys

INPUTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments", "run", "stress", "inputs-v2")


def show(days):
    for d in days:
        fp = os.path.join(INPUTS, f"day-{d:03d}.json")
        if not os.path.isfile(fp):
            print(f"day-{d:03d}: 不存在")
            continue
        rec = json.load(open(fp, encoding="utf-8"))
        msgs = [m.get("text", "") for m in rec.get("messages", []) if m.get("text")]
        print(f"==== day-{d:03d} ({len(msgs)}条) 当天事件 ====")
        n = 0
        for t in msgs:
            if re.search(r"@\w|threads shown|http|^\w+@|^\d+:|深度抓取|浅抓", t) or len(t) < 6:
                continue
            print(f"  · {t[:70]}")
            n += 1
            if n >= 12:
                break
        print()


def apply(fp):
    for item in json.load(open(fp, encoding="utf-8")):
        d, lines = item["day"], item["lines"]
        rec = json.load(open(os.path.join(INPUTS, f"day-{d:03d}.json"), encoding="utf-8"))
        rec["dialogue_inputs"] = [{"situation": s} for s in lines]
        json.dump(rec, open(os.path.join(INPUTS, f"day-{d:03d}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"✓ day-{d:03d}: 注入 {len(lines)} 条")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        show([int(a) for a in sys.argv[2:]] or [11, 22, 33])
    elif len(sys.argv) > 2 and sys.argv[1] == "--apply":
        apply(sys.argv[2])
    else:
        print(__doc__)
