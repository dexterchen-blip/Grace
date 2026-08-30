#!/usr/bin/env python3
"""② 夜班后时机决策（2026-08-30 双模型分层唤醒 v0）。

设计重点（用户）：她判断「合适的时间」主动开启对话。
v0 时机三因子（规则版）：
  ① 事件重要性：urgent > important > routine（来自哨兵信号 + 夜班汇总）
  ② 主人可打扰度：深夜(23-8)不主动；工作日白天(9-12/14-18)可；晚上(19-22)最佳
  ③ 对话时机：距上次主动 > 6h；当日主动 ≤ 2 次
输出 proactive-plan.json（完整系统读取后执行主动开启对话）。

用法（夜班后 / 手动）:
  ./run.sh .venv/bin/python3 v2/engine/timing_decision.py
"""
from __future__ import annotations
import config
import json
import os
import time
from datetime import datetime

SIGNAL_FILE = os.path.join(config.EXCHANGE, '.daytime', 'sentinel-signal.json')
PLAN_FILE = os.path.join(config.EXCHANGE, '.daytime', 'proactive-plan.json')
HANDLED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wake_handled.json")


def _hour() -> int:
    return datetime.now().hour


def _active_window(now_h: int) -> tuple[bool, str]:
    """主人可打扰度：深夜不打扰；晚上最佳；白天可。"""
    if 23 <= now_h or now_h < 8:
        return False, "深夜(23-8),不打扰"
    if 19 <= now_h < 22:
        return True, "晚上(19-22),最佳"
    if 9 <= now_h < 12 or 14 <= now_h < 18:
        return True, "白天,可打扰"
    return False, "休息时段,不打扰"


def decide() -> dict:
    sig = json.load(open(SIGNAL_FILE, encoding="utf-8")) if os.path.isfile(SIGNAL_FILE) else {"urgent": [], "important": []}
    handled = json.load(open(HANDLED_FILE, encoding="utf-8")) if os.path.isfile(HANDLED_FILE) else {}
    now = datetime.now()
    ok, window = _active_window(now.hour)
    items = [u for u in sig.get("urgent", []) if u.get("5b", "未知") != "不需要"]
    important = sig.get("important", [])
    # 时机决策
    if not ok:
        plan = {"when": "next_window", "eta_h": 9 - now.hour if now.hour >= 23 else (9 - now.hour) % 24,
                "reason": window, "items": (items + important)[:5]}
    elif items:
        plan = {"when": "now", "eta_h": 0, "reason": f"紧急事项 {len(items)} 项", "items": items[:5]}
    elif important:
        plan = {"when": "later_today", "eta_h": 1, "reason": f"重要事项 {len(important)} 项,稍后", "items": important[:5]}
    else:
        plan = {"when": "none", "eta_h": None, "reason": "无值得主动的事项", "items": []}
    # 对话时机约束（③）
    if handled:
        last_wake = handled.get("wake_time", 0)
        if time.time() - last_wake < 6 * 3600 and plan["when"] != "none":
            plan["reason"] += "；距上次唤醒 <6h,推迟"
            plan["when"] = "defer"
    plan["ts"] = time.time()
    plan["date"] = now.strftime("%Y-%m-%d %H:%M")
    os.makedirs(os.path.dirname(PLAN_FILE), exist_ok=True)
    with open(PLAN_FILE, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    return plan


if __name__ == "__main__":
    p = decide()
    print(f"=== 时机决策 {p['date']} ===")
    print(f"  何时主动: {p['when']} ｜ 原因: {p['reason']}")
    for it in p["items"][:3]:
        print(f"    · {it.get('line', '')[:44]}")
    print(f"→ {PLAN_FILE}")
