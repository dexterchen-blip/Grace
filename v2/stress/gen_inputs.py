#!/usr/bin/env python3
"""阶段一：生成 3 个月（90 天）完整输入数据（可查看/审计）。

输出：experiments/run/stress/inputs/day-NNN.json
  每条: {"day", "date", "messages": [{"text", "sentiment", "weight"}], "events": [...]}
模拟用户（UCSB 新生）入学后与雷姆的沟通，覆盖三个学期阶段。
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402
import scenarios  # noqa: E402

START_DAY = datetime(2026, 8, 28)   # 入学后（虚拟日历）
OUT = os.path.join(config.EXPERIMENTS, "run", "stress", "inputs")


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    os.makedirs(OUT, exist_ok=True)
    total_msgs = 0
    manifest = {"days": days, "start": START_DAY.strftime("%Y-%m-%d"), "files": []}
    for day in range(1, days + 1):
        msgs = scenarios.gen_day(day)
        rec = {
            "day": day,
            "date": (START_DAY + timedelta(days=day - 1)).strftime("%Y-%m-%d"),
            "messages": msgs,
            "events": [{"text": m["text"], "sentiment": m["sentiment"], "weight": m["weight"]} for m in msgs],
        }
        with open(os.path.join(OUT, f"day-{day:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        total_msgs += len(msgs)
        manifest["files"].append(f"day-{day:03d}.json")
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"✅ 已生成 {days} 天输入 → {OUT}")
    print(f"   总消息 {total_msgs} 条（日均 {total_msgs/days:.1f}）")
    print(f"   阶段分布：入学初期(1-30) / 学期中(31-60) / 期末(61-90)")


if __name__ == "__main__":
    main()
