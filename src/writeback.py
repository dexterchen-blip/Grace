#!/usr/bin/env python3
"""写回安全边界（设计 §10/§3，2026-08-19 补齐 #11 缺的一半）。

职责：
  1. audit.log 基建：每次「写回」动作落 `exchange/audit/`（单条单文件，只追加，
     配合 memory git 一键回滚；sensitive 永不出 exchange）。
  2. L3 modify 提案执行器：消费 `exchange/proposals/approved/` 里 type=modify 且
     target.path 指向 memory/L3_core/core.md 的提案 —— 用户点「批准」按钮 = §10 的
     显式写回授权，批准后立即把 old→new 应用进 core.md，记 audit，标 executed。

边界（严格按设计）：
  - 只有用户显式动作（dashboard 按钮 / CLI 手动）才会 approve → 才会触发 apply。
  - AI 推断绝不写回：任何代码路径都不得自行调用 apply_approved 之外的写。
  - 找不到 old 文本的提案：记 audit fail，提案留在 approved/ 待人工处理，绝不猜测替换。

用法：
  python3 writeback.py apply              # 执行所有已批准且可应用的 L3 modify 提案
  python3 writeback.py log "action" "..." # 追加一条审计记录（内部用）
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCHANGE = os.path.join(REPO, "exchange")
AUDIT_DIR = os.path.join(EXCHANGE, "audit")
PROPOSALS_DIR = os.path.join(EXCHANGE, "proposals")
L3_PATH = os.path.join(REPO, "memory", "L3_core", "core.md")
TZ_CN = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(TZ_CN).strftime("%Y-%m-%dT%H:%M:%S%z")


def log_audit(entry: dict) -> str | None:
    """追加一条审计记录：单条单文件 exchange/audit/<ts>-<seq>.json + index.log 行。
    返回落盘文件名，失败返回 None。"""
    try:
        os.makedirs(AUDIT_DIR, exist_ok=True)
        seq = len([f for f in os.listdir(AUDIT_DIR) if f.endswith(".json")]) + 1
        name = f"{datetime.now(TZ_CN).strftime('%Y%m%d-%H%M%S')}-{seq:03d}.json"
        entry.setdefault("ts", _now_iso())
        path = os.path.join(AUDIT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        with open(os.path.join(AUDIT_DIR, "index.log"), "a", encoding="utf-8") as f:
            f.write(f"{entry['ts']} {entry.get('action', '?')} {entry.get('target', '')} {name}\n")
        return name
    except Exception as e:
        print(f"[writeback] audit 写入失败: {e}")
        return None


def _list_approved() -> list[dict]:
    """approved/ 下所有提案（按文件名排序，保持 FIFO）。"""
    out = []
    d = os.path.join(PROPOSALS_DIR, "approved")
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception:
                continue
    return out


def _apply_l3_modify(prop: dict, path: str) -> tuple[bool, str]:
    """把一条 modify 提案应用进 core.md。返回 (成功?, 说明)。绝不猜测替换。"""
    target = prop.get("target", {})
    details = target.get("details", {}) if isinstance(target, dict) else {}
    old_txt = details.get("old", "")
    new_txt = details.get("new", "")
    if not old_txt:
        return False, "提案缺少 old 文本，跳过（人工处理）"
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return False, f"读 core.md 失败: {e}"
    if old_txt not in content:
        return False, "core.md 中找不到 old 原文，跳过（人工处理）"
    content = content.replace(old_txt, new_txt, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return True, "已应用 old→new"


def apply_approved() -> list[dict]:
    """执行所有已批准且可应用的 L3 modify 提案。返回每条的执行结果。"""
    sys.path.insert(0, os.path.join(REPO, "src"))
    from proposal_queue import ProposalQueue
    pq = ProposalQueue(PROPOSALS_DIR)
    results = []
    for prop in _list_approved():
        pid = prop.get("id", "?")
        target = prop.get("target", {}) if isinstance(prop.get("target"), dict) else {}
        tpath = str(target.get("path", ""))
        if not tpath.startswith("memory/L3_core/"):
            results.append({"pid": pid, "ok": False, "reason": "非 L3 modify 提案，跳过"})
            continue
        ok, reason = _apply_l3_modify(prop, L3_PATH)
        if ok:
            pq.mark_executed(pid)
            log_audit({
                "action": "writeback.apply_l3",
                "actor": "user-approved-button",
                "target": tpath,
                "detail": {"proposal": pid, "old": target.get("details", {}).get("old", "")[:100],
                           "new": target.get("details", {}).get("new", "")[:100]},
                "outcome": "applied",
            })
        else:
            # 应用失败：提案留在 approved/ 待人工处理，只记审计不破坏状态
            log_audit({
                "action": "writeback.apply_l3",
                "actor": "user-approved-button",
                "target": tpath,
                "detail": {"proposal": pid, "reason": reason},
                "outcome": "failed",
            })
        results.append({"pid": pid, "ok": ok, "reason": reason})
        print(f"[writeback] {pid}: {'✅' if ok else '⚠️'} {reason}")
    return results


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "apply":
        res = apply_approved()
        print(f"[writeback] 共处理 {len(res)} 条，成功 {sum(1 for r in res if r['ok'])} 条")
        sys.exit(0 if all(r["ok"] for r in res) else 0)  # 失败不 exit 非零，避免阻断流水线
    if cmd == "log" and len(sys.argv) >= 4:
        name = log_audit({"action": sys.argv[2], "actor": "cli", "target": sys.argv[3]})
        print(f"[writeback] audit → {name}")
        sys.exit(0 if name else 1)
    print(f"未知命令: {cmd}")
    sys.exit(2)


if __name__ == "__main__":
    main()
