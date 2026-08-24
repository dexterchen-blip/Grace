#!/usr/bin/env python3
"""提案队列 + 门控机制（设计 §9/§10，grill Q22/Q27）。

核心规则：
  1. 睡眠 agent 永不等待人类输入——只写提案到 pending/，继续干活。
  2. 被否决提案永不再提——content_hash 进 blacklist.jsonl，create 时自动挡回。
  3. 超期提案批量打包升级——08:00 看门狗把过期 pending 移到 expired/，
     多条打包成一个 escalation batch 写 escalated/，dashboard 红点告警。
  4. 所有「要删/要改/要派」操作经 gate_check() 门控——只有 approved 才放行。

存储：
  exchange/proposals/
    pending/     — 待决策（每条一个 JSON 文件）
    approved/    — 已批准，待执行
    rejected/    — 已否决（hash 已入 blacklist）
    expired/     — 超期未决
    escalated/   — 批量打包的升级包
    executed/    — 已执行归档
    blacklist.jsonl — 永久否决名单（只追加）

用法：
  from proposal_queue import ProposalQueue
  pq = ProposalQueue("/abs/path/to/exchange/proposals")
  pq.create(type="modify", title="归档 8 月旧会话", target={"path": "...", "action": "archive"})
  pq.approve("prop-20260818-001")
  pq.reject("prop-20260818-002", reason="不需要")
  pq.expire_and_escalate()  # 08:00 看门狗调
  pq.gate_check("consolidate", target_path="cloud-drop/kimi-scrape/")
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROPOSALS_DIR = os.path.join(BASE_DIR, "exchange", "proposals")

# 测试沙盒（2026-08-22）：提案目录跟随 AIAGENT_SANDBOX。
_SANDBOX = os.environ.get("AIAGENT_SANDBOX", "")
if _SANDBOX:
    PROPOSALS_DIR = os.path.join(_SANDBOX, "exchange", "proposals")

STATUSES = ("pending", "approved", "rejected", "expired", "escalated", "executed")
DEFAULT_EXPIRY_HOURS = 28  # 04:00 写 → 次日 08:00 过期 = 28h；但看门狗 08:00 主动扫


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _now_ts() -> float:
    return time.time()


def _content_hash(prop: dict) -> str:
    """对 type + target 做哈希，同一操作被否决后永不再提。"""
    key = json.dumps({
        "type": prop["type"],
        "target": prop.get("target", {}),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode()).hexdigest()


def _prop_id() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = os.urandom(3).hex()
    return f"prop-{ts}-{suffix}"


class ProposalQueue:
    """提案队列：文件系统存储，一个提案一个 JSON 文件。"""

    def __init__(self, root: str = PROPOSALS_DIR):
        self.root = root
        self.blacklist_path = os.path.join(root, "blacklist.jsonl")
        for s in STATUSES:
            os.makedirs(os.path.join(root, s), exist_ok=True)
        if not os.path.exists(self.blacklist_path):
            # 初始化空 blacklist
            with open(self.blacklist_path, "w", encoding="utf-8") as f:
                f.write("")

    # ---- 内部 ----

    def _path(self, status: str, pid: str) -> str:
        return os.path.join(self.root, status, f"{pid}.json")

    def _find(self, pid: str) -> str | None:
        """在所有状态目录里找提案文件。"""
        for s in STATUSES:
            p = self._path(s, pid)
            if os.path.exists(p):
                return p
        return None

    def _load(self, pid: str) -> dict | None:
        p = self._find(pid)
        if not p:
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def _move(self, pid: str, from_status: str, to_status: str) -> str:
        src = self._path(from_status, pid)
        dst = self._path(to_status, pid)
        shutil.move(src, dst)
        return dst

    def _is_blacklisted(self, content_hash: str) -> bool:
        if not os.path.exists(self.blacklist_path):
            return False
        with open(self.blacklist_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("content_hash") == content_hash:
                        return True
                except json.JSONDecodeError:
                    continue
        return False

    def _append_blacklist(self, content_hash: str, pid: str, reason: str) -> None:
        rec = {
            "content_hash": content_hash,
            "rejected_proposal_id": pid,
            "reason": reason,
            "blacklisted_at": _now_iso(),
        }
        with open(self.blacklist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _list_dir(self, status: str) -> list[dict]:
        """列出某状态目录的提案（escalated 目录含 batch 文件，自动跳过）。"""
        d = os.path.join(self.root, status)
        props = []
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    rec = json.load(f)
                if "status" not in rec:
                    # batch 文件（escalated/ 里的 esc-batch-*.json）→ 跳过
                    continue
                props.append(rec)
            except (json.JSONDecodeError, KeyError):
                continue
        return props

    # ---- 公开 API ----

    def create(
        self,
        *,
        type: str,
        title: str,
        description: str = "",
        target: dict | None = None,
        source: str = "night-consolidation",
        priority: str = "medium",
        expires_at: str | None = None,
        extra: dict | None = None,
    ) -> str:
        """创建提案。若 content_hash 在黑名单则拒绝（返回空串 + 打印警告）。

        type: consolidate | delete | modify | dispatch | writeback
        target: {"path": "...", "action": "...", "details": {...}}
        """
        assert type in ("consolidate", "delete", "modify", "dispatch", "writeback"), \
            f"bad type: {type}"
        assert priority in ("low", "medium", "high", "urgent"), \
            f"bad priority: {priority}"

        prop = {
            "id": _prop_id(),
            "type": type,
            "title": title,
            "description": description,
            "target": target or {},
            "source": source,
            "priority": priority,
            "status": "pending",
            "created_at": _now_iso(),
            "expires_at": expires_at or _default_expiry(),
            "decided_at": None,
            "decided_by": None,
            "reject_reason": None,
            "executed_at": None,
        }
        if extra:
            prop.update(extra)

        chash = _content_hash(prop)
        prop["content_hash"] = chash

        if self._is_blacklisted(chash):
            print(f"[proposal] ⚠ 被否决提案永不再提：{title} (hash={chash[:12]})")
            return ""

        path = self._path("pending", prop["id"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(prop, f, ensure_ascii=False, indent=2)
        print(f"[proposal] 创建 {prop['id']} → pending/ ({title})")
        return prop["id"]

    def approve(self, pid: str, decided_by: str = "user") -> bool:
        prop = self._load(pid)
        if not prop:
            print(f"[proposal] 未找到 {pid}")
            return False
        if prop["status"] != "pending":
            print(f"[proposal] {pid} 状态={prop['status']}，无法批准")
            return False
        prop["status"] = "approved"
        prop["decided_at"] = _now_iso()
        prop["decided_by"] = decided_by
        old_path = self._find(pid)
        new_path = self._path("approved", pid)
        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(prop, f, ensure_ascii=False, indent=2)
        os.remove(old_path)
        print(f"[proposal] ✅ 批准 {pid}")
        return True

    def reject(self, pid: str, reason: str = "", decided_by: str = "user") -> bool:
        prop = self._load(pid)
        if not prop:
            print(f"[proposal] 未找到 {pid}")
            return False
        if prop["status"] != "pending":
            print(f"[proposal] {pid} 状态={prop['status']}，无法否决")
            return False
        prop["status"] = "rejected"
        prop["decided_at"] = _now_iso()
        prop["decided_by"] = decided_by
        prop["reject_reason"] = reason
        chash = prop.get("content_hash", _content_hash(prop))
        # 入黑名单
        self._append_blacklist(chash, pid, reason)
        old_path = self._find(pid)
        new_path = self._path("rejected", pid)
        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(prop, f, ensure_ascii=False, indent=2)
        os.remove(old_path)
        print(f"[proposal] ❌ 否决 {pid}（已入黑名单，永不再提）")
        return True

    def mark_executed(self, pid: str) -> bool:
        prop = self._load(pid)
        if not prop:
            return False
        if prop["status"] != "approved":
            print(f"[proposal] {pid} 状态={prop['status']}，无法标记已执行")
            return False
        prop["status"] = "executed"
        prop["executed_at"] = _now_iso()
        old_path = self._find(pid)
        new_path = self._path("executed", pid)
        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(prop, f, ensure_ascii=False, indent=2)
        os.remove(old_path)
        print(f"[proposal] 🔧 已执行 {pid}")
        return True

    def expire_and_escalate(self) -> int:
        """08:00 看门狗调用：把过期 pending 移到 expired，多条打包升级到 escalated。

        返回升级的提案数。
        """
        now = _now_ts()
        pending = self._list_dir("pending")
        expired_list = []
        for prop in pending:
            exp = prop.get("expires_at", "")
            if not exp:
                continue
            try:
                exp_ts = datetime.fromisoformat(exp.replace("Z", "+00:00")).timestamp()
            except (ValueError, AttributeError):
                continue
            if exp_ts < now:
                # 移到 expired
                old_path = self._path("pending", prop["id"])
                prop["status"] = "expired"
                prop["expired_at"] = _now_iso()
                new_path = self._path("expired", prop["id"])
                with open(new_path, "w", encoding="utf-8") as f:
                    json.dump(prop, f, ensure_ascii=False, indent=2)
                os.remove(old_path)
                expired_list.append(prop)

        if not expired_list:
            print("[proposal] 无过期提案")
            return 0

        # 打包升级
        batch_id = f"esc-batch-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        batch = {
            "batch_id": batch_id,
            "created_at": _now_iso(),
            "count": len(expired_list),
            "proposals": [{"id": p["id"], "title": p["title"], "type": p["type"],
                           "target": p.get("target", {}), "priority": p["priority"]}
                          for p in expired_list],
        }
        batch_path = os.path.join(self.root, "escalated", f"{batch_id}.json")
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        print(f"[proposal] ⏰ {len(expired_list)} 条过期提案已打包升级 → {batch_id}")
        return len(expired_list)

    def gate_check(self, gate_type: str, target_path: str = "") -> list[dict]:
        """门控：返回所有 approved 提案中匹配 gate_type + target_path 的。

        M5 巩固前调 gate_check("consolidate", target_path) 检查是否有批准。
        M8 cloud-drop 巩固前调 gate_check("consolidate", "cloud-drop/...")。
        M11 写回前调 gate_check("writeback", target_path)。
        无批准 → 返回空列表 → 调用方应跳过该操作。
        """
        approved = self._list_dir("approved")
        matched = []
        for prop in approved:
            if prop["type"] != gate_type:
                continue
            tpath = prop.get("target", {}).get("path", "")
            if target_path and tpath and target_path not in tpath and tpath not in target_path:
                continue
            matched.append(prop)
        return matched

    def list(self, status: str = "pending") -> list[dict]:
        """列出指定状态的提案。"""
        if status == "all":
            result = []
            for s in STATUSES:
                result.extend(self._list_dir(s))
            return result
        return self._list_dir(status)

    def stats(self) -> dict:
        """队列统计。escalated 显示 batch 数（batch 内含过期提案摘要）。"""
        counts = {}
        for s in STATUSES:
            if s == "escalated":
                # escalated/ 存的是 batch 文件，不是单条提案
                d = os.path.join(self.root, s)
                counts[s] = len([f for f in os.listdir(d)
                                 if f.endswith(".json")]) if os.path.isdir(d) else 0
            else:
                counts[s] = len(self._list_dir(s))
        bl_count = 0
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, encoding="utf-8") as f:
                bl_count = sum(1 for line in f if line.strip())
        counts["blacklisted"] = bl_count
        return counts

    def is_blacklisted_content(self, type: str, target: dict) -> bool:
        """外部预检：某个操作是否已被永久否决。"""
        dummy = {"type": type, "target": target}
        return self._is_blacklisted(_content_hash(dummy))


def _default_expiry() -> str:
    """默认过期时间：下一个本地（UTC+8）08:00，转 UTC ISO。

    2026-08-20 修复：旧实现在 UTC 时间上直接 replace(hour=0)——本地凌晨
    （00:00-08:00）创建的提案会被算成「本地昨天 08:00」，创建即过期，
    08:00 看门狗立即误移 expired/（夜班 03:00 巩固产生的提案全部中招）。
    现在先在本地时区算「下一个 08:00」，再转 UTC。"""
    import datetime as dt
    tz8 = dt.timezone(dt.timedelta(hours=8))
    now_local = dt.datetime.now(tz8)
    target = now_local.replace(hour=8, minute=0, second=0, microsecond=0)
    if now_local.hour >= 8:
        target += dt.timedelta(days=1)
    return target.astimezone(dt.timezone.utc).isoformat(timespec="seconds")
    return target.isoformat(timespec="seconds")


# ---- CLI ----

def main():
    ap = argparse.ArgumentParser(description="提案队列管理")
    sub = ap.add_subparsers(dest="cmd")

    sp_create = sub.add_parser("create", help="创建提案")
    sp_create.add_argument("--type", required=True,
                           choices=["consolidate", "delete", "modify", "dispatch", "writeback"])
    sp_create.add_argument("--title", required=True)
    sp_create.add_argument("--desc", default="")
    sp_create.add_argument("--target-path", default="")
    sp_create.add_argument("--target-action", default="")
    sp_create.add_argument("--source", default="night-consolidation")
    sp_create.add_argument("--priority", default="medium",
                           choices=["low", "medium", "high", "urgent"])

    sp_approve = sub.add_parser("approve", help="批准提案")
    sp_approve.add_argument("pid")

    sp_reject = sub.add_parser("reject", help="否决提案（永不再提）")
    sp_reject.add_argument("pid")
    sp_reject.add_argument("--reason", default="")

    sp_list = sub.add_parser("list", help="列出提案")
    sp_list.add_argument("--status", default="pending",
                         choices=list(STATUSES) + ["all"])

    sp_expire = sub.add_parser("expire", help="08:00 看门狗：过期 + 批量升级")

    sp_gate = sub.add_parser("gate", help="门控检查")
    sp_gate.add_argument("--type", required=True,
                         choices=["consolidate", "delete", "modify", "dispatch", "writeback"])
    sp_gate.add_argument("--target-path", default="")

    sp_stats = sub.add_parser("stats", help="队列统计")

    sp_exec = sub.add_parser("executed", help="标记已执行")
    sp_exec.add_argument("pid")

    sp_check_bl = sub.add_parser("check-blacklist", help="检查某操作是否已被永久否决")
    sp_check_bl.add_argument("--type", required=True)
    sp_check_bl.add_argument("--target-path", default="")
    sp_check_bl.add_argument("--target-action", default="")

    args = ap.parse_args()
    pq = ProposalQueue()

    if args.cmd == "create":
        pid = pq.create(
            type=args.type, title=args.title, description=args.desc,
            target={"path": args.target_path, "action": args.target_action},
            source=args.source, priority=args.priority,
        )
        if pid:
            print(f"  id={pid}")
    elif args.cmd == "approve":
        sys.exit(0 if pq.approve(args.pid) else 1)
    elif args.cmd == "reject":
        sys.exit(0 if pq.reject(args.pid, args.reason) else 1)
    elif args.cmd == "list":
        props = pq.list(args.status)
        if not props:
            print(f"(无 {args.status} 提案)")
        for p in props:
            pri = p.get("priority", "?")
            print(f"  [{p['status']}] {p['id']} pri={pri} | {p['title']}")
            if p.get("target", {}).get("path"):
                print(f"         target: {p['target']['path']} ({p['target'].get('action', '')})")
    elif args.cmd == "expire":
        n = pq.expire_and_escalate()
        print(f"升级 {n} 条")
    elif args.cmd == "gate":
        matched = pq.gate_check(args.type, args.target_path)
        if matched:
            print(f"✅ 门控通过：{len(matched)} 条 approved 提案匹配")
            for m in matched:
                print(f"  {m['id']}: {m['title']}")
        else:
            print(f"⛔ 门控未通过：无 approved 的 {args.type} 提案"
                  f"{f' (target={args.target_path})' if args.target_path else ''}")
    elif args.cmd == "stats":
        s = pq.stats()
        for k, v in s.items():
            print(f"  {k}: {v}")
    elif args.cmd == "executed":
        sys.exit(0 if pq.mark_executed(args.pid) else 1)
    elif args.cmd == "check-blacklist":
        bl = pq.is_blacklisted_content(
            args.type,
            {"path": args.target_path, "action": args.target_action},
        )
        print("⛔ 已被永久否决" if bl else "✅ 不在黑名单")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
