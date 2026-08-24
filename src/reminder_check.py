#!/usr/bin/env python3
"""定时提醒检查（2026-08-23 用户方案：疫苗 9/6 后提醒等 remind_after 待办到期自动提醒）。

扫描 L3 core.md 中带 `remind_after=YYYY-MM-DD` 的待办行：
  - 今天 >= remind_after 且尚未提醒（水位）→ 写 .daytime/reminder-<today>.md + macOS 通知
  - 未来日期 → 不触发（提前 = bug）
每天 08:00 由 launchd（com.local-ai-agent.reminder）触发；沙盒跟随 AIAGENT_SANDBOX。

用法：
  python3 reminder_check.py        # 检查并提醒（幂等：每天每条只提醒一次）
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L3_PATH = os.path.join(REPO, "memory", "L3_core", "core.md")
DAYTIME = os.path.join(REPO, "exchange", ".daytime")
STATE_PATH = os.path.join(REPO, "memory", "L1_working", "reminder_state.json")

SANDBOX = os.environ.get("AIAGENT_SANDBOX", "")
if SANDBOX:
    L3_PATH = os.path.join(SANDBOX, "memory", "L3_core", "core.md")
    DAYTIME = os.path.join(SANDBOX, "exchange", ".daytime")
    STATE_PATH = os.path.join(SANDBOX, "memory", "L1_working", "reminder_state.json")

REMINDER_RE = re.compile(r"remind_after=(\d{4}-\d{2}-\d{2})")


def run() -> int:
    if not os.path.exists(L3_PATH):
        print("[reminder] L3 不存在，跳过")
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    state = {"reminded": {}}
    if os.path.exists(STATE_PATH):
        try:
            state = json.load(open(STATE_PATH, encoding="utf-8"))
        except Exception:
            pass
    due = []
    for line in open(L3_PATH, encoding="utf-8"):
        m = REMINDER_RE.search(line)
        if not m:
            continue
        ra = m.group(1)
        if ra > today:
            print(f"[reminder] {ra} 未到（今天 {today}），不触发")
            continue
        item = line.strip()[:80]
        key = f"{ra}|{item}"
        if state["reminded"].get(key):
            continue
        due.append((ra, item))
        state["reminded"][key] = today
    if due:
        os.makedirs(DAYTIME, exist_ok=True)
        path = os.path.join(DAYTIME, f"reminder-{today}.md")
        with open(path, "a", encoding="utf-8") as f:
            for ra, item in due:
                f.write(f"- ⏰ [{ra} 到期] {item}\n")
        # macOS 系统通知（非沙盒时；沙盒跳过避免打扰）
        if not SANDBOX:
            for ra, item in due[:3]:
                subprocess.run(
                    ["osascript", "-e",
                     f'display notification "{item[:60]}" with title "本地 AI 提醒 ⏰"'],
                    capture_output=True, timeout=10)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
        print(f"[reminder] 触发 {len(due)} 条 → {path}")
    else:
        print(f"[reminder] 今天 {today} 无到期提醒")
    return len(due)


if __name__ == "__main__":
    run()
