#!/usr/bin/env python3
"""书库 sentiment 重标 —— Phase 0 情绪标注修复（2026-09-03）。

遍历 inputs-v2/day-*.json，用统一 engine/sentiment.assess() 重算每条消息的 sentiment
（分层效价词表），保留 gist/cog/events 等全部其他字段。幂等：重复跑结果一致。
用法: ./run.sh .venv/bin/python3 v2/stress/relabel_sentiment.py [--dir experiments/run/stress/inputs-v2]
"""
import json
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # v2/..
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
from engine.sentiment import assess  # noqa: E402

IN_DIR = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else \
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
                 "experiments", "run", "stress", "inputs-v2")


def main() -> int:
    files = sorted(glob.glob(os.path.join(IN_DIR, "day-*.json")))
    total = changed = 0
    dist = {}
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        dirty = False
        for m in d.get("messages", []):
            if not m.get("text"):
                continue
            total += 1
            old = m.get("sentiment", 0)
            a = assess(m["text"])
            new = a["valence"]
            bucket = "neu" if new == 0 else ("pos" if new > 0 else "neg")
            dist[bucket] = dist.get(bucket, 0) + 1
            if old != new:
                m["sentiment"] = new
                dirty = True
                changed += 1
        if dirty:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"重标完成: {total} 条消息, {changed} 条变更")
    print(f"分布: {dist} (非零 {100 - dist.get('neu', 0) / max(total, 1) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
