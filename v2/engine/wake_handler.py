#!/usr/bin/env python3
"""① 完整系统唤醒入口（2026-08-30 双模型分层唤醒 v0）。

读 5B 哨兵信号 sentinel-signal.json：
  · urgent 非空 → 通知主人（macOS 通知）+ 若 8100 未跑则启动 27B
  · 处理后标记已处理（防重复唤醒）
用法（launchd / 手动）:
  ./run.sh .venv/bin/python3 v2/engine/wake_handler.py
"""
from __future__ import annotations
import config
import json
import os
import subprocess
import time
from datetime import datetime

SIGNAL_FILE = os.path.join(os.environ.get('AIAGENT_EXCHANGE_DAYTIME', os.path.join(config.EXCHANGE, '.daytime')), 'sentinel-signal.json')
HANDLED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wake_handled.json")
DAY_PLIST = os.path.expanduser('~/Library/LaunchAgents/com.local-ai-agent.day-model.plist')


def _port_open(port: int = 8100) -> bool:
    try:
        import socket
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _notify(title: str, body: str):
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{body}" with title "{title}"'],
                       capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001
        pass


def _start_27b():
    """8100 未跑时启动 day-model（临时唤醒完整系统）。"""
    if _port_open():
        return "8100 已在跑"
    try:
        subprocess.run(["launchctl", "bootstrap", "gui/501", DAY_PLIST], capture_output=True, timeout=15)
        time.sleep(6)
        return "8100 已启动" if _port_open() else "8100 启动中(可能失败)"
    except Exception as e:  # noqa: BLE001
        return f"启动失败: {e}"


def main():
    if not os.path.isfile(SIGNAL_FILE):
        print("无信号文件,无事可做")
        return
    sig = json.load(open(SIGNAL_FILE, encoding="utf-8"))
    handled = json.load(open(HANDLED_FILE, encoding="utf-8")) if os.path.isfile(HANDLED_FILE) else {}
    last = handled.get("last_ts", 0)
    if sig["ts"] <= last:
        print("信号已处理,跳过")
        return
    urgent = sig.get("urgent", [])
    if not urgent:
        print("无紧急项,不唤醒")
        return
    # 5B 确认: 只对 5b=="需要" 或未知 的紧急项唤醒(5b=="不需要" 降级)
    wake_items = [u for u in urgent if u.get("5b", "未知") != "不需要"]
    if not wake_items:
        print("5B 判定均为'不需要',不唤醒")
        return
    body = "；".join(u["line"][:40] for u in wake_items[:3])
    _notify("🔴 Grace 哨兵: 有紧急事项", body[:80])
    status = _start_27b()
    # 标记已处理
    json.dump({"last_ts": sig["ts"], "wake_time": time.time(),
               "items": [u["line"][:60] for u in wake_items[:5]],
               "27b": status}, open(HANDLED_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"🔴 已唤醒: {len(wake_items)} 项 | {status} | 通知已发")
    print(f"  {body[:80]}")


if __name__ == "__main__":
    main()
