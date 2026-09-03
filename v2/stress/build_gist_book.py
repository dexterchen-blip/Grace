#!/usr/bin/env python3
"""build_gist_book.py — V2 书库 gist 离线预生成(2026-09-02).

对 inputs-v2/ 每一天的真实经历调用 27B 提 gist(生成效应), 写回 day-NNN.json 的 gist 字段。
27B 离线(8100 停)时跳过(打印提示)——由 V2 轮启动前手动跑(需 8100 在线)。

用法: ./run.sh .venv/bin/python3 v2/stress/build_gist_book.py [--book inputs-v2]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

from gist_extractor import extract_day_gist, extract_day_cognition  # noqa: E402


def main() -> None:
    book = sys.argv[sys.argv.index("--book") + 1] if "--book" in sys.argv else "inputs-v2"
    in_dir = os.path.join(config.EXPERIMENTS, "run", "stress", book)
    if not os.path.isdir(in_dir):
        print(f"❌ 未找到 {in_dir}")
        return
    from gist_extractor import _server_alive
    if not _server_alive():
        print("⚠️  8100 27B 离线 —— 先恢复 day-model 再跑(launchctl bootstrap gui/501 ...day-model.plist)")
        return

    files = sorted(f for f in os.listdir(in_dir) if f.startswith("day-") and f.endswith(".json"))
    done = skipped = 0
    for fn in files:
        fp = os.path.join(in_dir, fn)
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # ★2026-09-02 耦合: 认知重构器与 gist 同一次离线批量。已含 gist+cog 的天跳过。
        if d.get("gist") and d.get("cog"):
            skipped += 1
            continue
        g = extract_day_gist(d.get("messages", []))
        cog = extract_day_cognition(d.get("messages", []))
        if g or cog:
            if g:
                d["gist"] = g
            if cog:
                d["cog"] = cog
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
            done += 1
            print(f"  ✓ {fn}: gist {len(g)} 条 + cog {len(cog)} 条(认知重构)", flush=True)
        else:
            print(f"  - {fn}: 无产出(27B 空返回/当天无消息)", flush=True)
    print(f"=== gist+cog 预生成完成: {done} 天生成, {skipped} 天已有 ===")


if __name__ == "__main__":
    main()
