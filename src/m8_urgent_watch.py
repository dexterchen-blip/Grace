#!/usr/bin/env python3
"""M8 urgent 高频扫描器（设计 §11，grill Q53）。

cloud-drop/urgent/ 与 school/urgent/ 存**时间敏感文件**（注册截止、成绩发布、
orientation 预约、BARC 账单截止等）。本扫描器以高频（launchd 每 15 分钟）运行，
不等夜班——新文件立即：
  1. 入 L0（source=urgent:*，原始留存，append-only）
  2. 写告警到 exchange/shared/alerts/alert-*.json（dashboard 红点数据源）

注意：urgent 的「立即」只到 L0 + 告警；巩固进 L2/L3 仍走夜班提案门控（§11 主权最后拍板）。

幂等：memory/L1_working/urgent_watch_state.json（relpath → [mtime_ns, size]）。

用法：
  python3 m8_urgent_watch.py scan     # 扫一遍（launchd 入口）
  python3 m8_urgent_watch.py alerts   # 列出未读告警
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l0_ingest import L0Writer  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCHANGE = os.path.join(REPO, "exchange")
L0_ROOT = os.path.join(REPO, "memory", "L0_raw")
STATE_PATH = os.path.join(REPO, "memory", "L1_working", "urgent_watch_state.json")
ALERTS_DIR = os.path.join(EXCHANGE, "shared", "alerts")

TZ_CN = timezone(timedelta(hours=8))

# urgent 监视点：子目录 → L0 source 名
WATCH_DIRS = {
    os.path.join("cloud-drop", "urgent"): "urgent:cloud-drop",
    os.path.join("school", "urgent"): "urgent:school",
}


def now_cn() -> str:
    return datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S")


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}}


def save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


def scan() -> int:
    """扫所有 urgent 监视点。返回新文件数。"""
    st = load_state()
    w = L0Writer(L0_ROOT)
    new_count = 0

    for sub, source in WATCH_DIRS.items():
        base = os.path.join(EXCHANGE, sub)
        if not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base)):
            if fn.startswith(".") or fn.endswith((".tmp", ".part")):
                continue
            p = os.path.join(base, fn)
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, REPO)
            try:
                stat = os.stat(p)
            except OSError:
                continue
            sig = [stat.st_mtime_ns, stat.st_size]
            if st["files"].get(rel) == sig:
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                text = "(非文本文件)"
            # 1. 入 L0（原始留存）
            w.append(source, {"path": rel, "text": text},
                     sensitive=False,
                     meta={"ingest": "urgent_watch", "urgent": True})
            # 2. 写告警
            os.makedirs(ALERTS_DIR, exist_ok=True)
            alert_id = f"alert-{datetime.now(TZ_CN).strftime('%Y%m%d%H%M%S')}-{fn[:20].replace(' ', '_')}"
            alert = {
                "id": alert_id,
                "source": source,
                "file": rel,
                "snippet": text[:200],
                "detected_at": now_cn(),
                "status": "new",
            }
            with open(os.path.join(ALERTS_DIR, alert_id + ".json"), "w", encoding="utf-8") as f:
                json.dump(alert, f, ensure_ascii=False, indent=2)
            st["files"][rel] = sig
            new_count += 1
            print(f"[urgent] 新文件 {rel} → L0({source}) + 告警 {alert_id}")

    save_state(st)
    if new_count == 0:
        print(f"[urgent] {now_cn()} 无新文件")
    return new_count


def list_alerts() -> None:
    """列出未读告警（status=new）。"""
    if not os.path.isdir(ALERTS_DIR):
        print("无告警")
        return
    alerts = []
    for fn in sorted(os.listdir(ALERTS_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(ALERTS_DIR, fn), encoding="utf-8") as f:
            a = json.load(f)
        if a.get("status") == "new":
            alerts.append(a)
    if not alerts:
        print("无未读告警")
        return
    print(f"=== {len(alerts)} 条未读 urgent 告警 ===")
    for a in alerts:
        print(f"  [{a['detected_at']}] {a['source']}: {a['file']}")
        print(f"    {a['snippet'][:100]}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "scan":
        scan()
    elif sys.argv[1] == "alerts":
        list_alerts()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
