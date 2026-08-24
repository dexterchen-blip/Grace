#!/usr/bin/env python3
"""L0 原始层摄入原语（系统灵魂底座，框架无关）。

设计对应：
  §4 四层记忆 —— L0 只追加不删（append-only），对应 dsh 的 append-only SessionEvent 不变量。
  §5 Megumin persona —— mode=persona 的对话隔离到 persona/ 子树，永不进语义索引。
  §11 来源/sensitive —— 每条记录带 source 与 sensitive 标签；sensitive 仅落本地。

核心不变量：L0 永不删除、永不覆盖。遗忘 = 从上层索引降级，绝不触碰 L0（见 §4）。

用法：
  from l0_ingest import L0Writer
  w = L0Writer("/abs/path/to/memory/L0_raw")
  w.append(source="wechat", mode="normal", payload={"text": "..."})
  w.append(source="persona_megumin", mode="persona", payload={"text": "..."})  # 隔离子树
"""
from __future__ import annotations
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


class L0Writer:
    def __init__(self, l0_root: str):
        self.l0_root = l0_root
        self.persona_dir = os.path.join(l0_root, "persona")
        os.makedirs(self.l0_root, exist_ok=True)
        os.makedirs(self.persona_dir, exist_ok=True)

    def _record(self, source: str, mode: str, payload: Any,
                sensitive: bool, meta: Optional[dict]) -> dict:
        rec = {
            "id": uuid.uuid4().hex,
            "ts": datetime.now(timezone.utc).isoformat(),
            "epoch": time.time(),
            "source": source,
            "mode": mode,            # normal | persona
            "sensitive": bool(sensitive),
            "payload": payload,
        }
        if meta:
            rec["meta"] = meta
        return rec

    def append(self, source: str, payload: Any, *,
               mode: str = "normal", sensitive: bool = False,
               meta: Optional[dict] = None) -> str:
        """追加一条原始记录，返回记录 id。永远追加，永不删改。"""
        if mode not in ("normal", "persona"):
            raise ValueError(f"mode must be normal|persona, got {mode!r}")
        rec = self._record(source, mode, payload, sensitive, meta)
        # persona 隔离到子树，永不进事实记忆的语义索引
        target_dir = self.persona_dir if mode == "persona" else self.l0_root
        # 按来源分子文件，便于夜班 ingest 扫描（§11 同构于 exchange）
        fname = f"{source}.jsonl"
        fpath = os.path.join(target_dir, fname)
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec["id"]

    def iter_source(self, source: str, *, include_persona: bool = False):
        """遍历某来源的全部原始记录（只追加，故按行读即可）。"""
        fpath = os.path.join(self.l0_root, f"{source}.jsonl")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        if include_persona:
            pf = os.path.join(self.persona_dir, f"{source}.jsonl")
            if os.path.exists(pf):
                with open(pf, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)


# ----------------------------------------------------------------------------
# 摄入粒度策略（设计 §3 / 审计 A5）：在写入 L0 前执行，L0 仍只追加不删。
#   · 按会话切片：相邻消息间隔 <30min 归同一段（SESSION_GAP_SECONDS）
#   · 丢公众号：sender/id 以 "gh_" 开头（微信公众号）不进 L0
#   · 丢纯表情/图片：剥离占位符与 emoji 后无实质文本，不进 L0
#   · 保留 msg_id 范围：每段记录 min/max msg_id，便于回捞 L0 原文
# ----------------------------------------------------------------------------
SESSION_GAP_SECONDS = 1800  # 30 分钟

_EMOJI_RE = re.compile(
    "[" 
    "\U0001F000-\U0001FAFF"  # 象形/符号/扑克牌等
    "\U00002600-\U000027BF"  # 杂项符号
    "\U0001F1E6-\U0001F1FF"  # 旗帜
    "\U00002190-\U000021FF"  # 箭头
    "\U00002B00-\U00002BFF"  # 符号箭头
    "\U0000FE00-\U0000FE0F"  # 变体选择符
    "\U0000200D"             # ZWJ
    "]"
)


def _strip_symbols(text: str) -> str:
    """去 [图片]/[表情] 占位符、emoji、空白，返回实质文本。"""
    t = re.sub(r"\[[^\]]*\]", "", text or "")
    t = _EMOJI_RE.sub("", t)
    t = re.sub(r"\s+", "", t)
    return t


def is_public_account(sender: str) -> bool:
    """微信公众号：wxid 以 gh_ 开头，不进事实记忆。"""
    return bool(sender) and sender.startswith("gh_")


def is_emoji_or_media_only(text: str) -> bool:
    """纯表情 / 纯图片占位 / 空 —— 无实质文本，丢弃。"""
    return _strip_symbols(text) == ""


def slice_sessions(messages: list[dict], gap: int = SESSION_GAP_SECONDS) -> list[list[dict]]:
    """按时间相邻(间隔<gap)切成会话段。messages 需含 'ts'(epoch float) 且已排序。"""
    sessions: list[list[dict]] = []
    cur: list[dict] = []
    prev_ts = None
    for m in messages:
        ts = m.get("ts")
        if prev_ts is not None and ts is not None and (ts - prev_ts) > gap:
            if cur:
                sessions.append(cur)
            cur = []
        cur.append(m)
        prev_ts = ts
    if cur:
        sessions.append(cur)
    return sessions


def ingest_wechat(writer: "L0Writer", messages: list[dict], *,
                  source: str = "wechat", sensitive: bool = False) -> list[str]:
    """摄入微信消息列表：过滤公众号/纯表情图片 → 会话切片 → 写 L0。

    每段记录含 session_id、msg_id 范围、消息数、参与者，便于夜班巩固回捞。
    sensitive=True 时整批标记敏感（私人聊天按 §6/§11 红线应传 True）。
    返回写入的 L0 记录 id 列表。
    """
    kept_ids: list[str] = []
    filtered = [
        m for m in messages
        if not is_public_account(str(m.get("sender", "")))
        and not is_emoji_or_media_only(str(m.get("text", "")))
    ]
    for sess in slice_sessions(filtered):
        msg_ids = [m.get("msg_id") for m in sess if m.get("msg_id") is not None]
        rec_payload = {
            "session_id": uuid.uuid4().hex,
            "msg_id_min": min(msg_ids) if msg_ids else None,
            "msg_id_max": max(msg_ids) if msg_ids else None,
            "msg_count": len(sess),
            "participants": sorted({str(m.get("sender", "")) for m in sess}),
            "messages": sess,
        }
        rid = writer.append(
            source, rec_payload, sensitive=sensitive,
            meta={"ingest": "wechat_session", "granularity": "30min"},
        )
        kept_ids.append(rid)
    return kept_ids


if __name__ == "__main__":
    import tempfile
    d = tempfile.mkdtemp()
    w = L0Writer(d)
    # 基础原语自测（用独立 source，避免与下方摄入粒度自测混源）
    rid = w.append("base_test", {"text": "明天交物理作业"}, sensitive=True)
    w.append("persona_megumin", {"text": " Explosion! 本小姐才不记得呢"}, mode="persona")
    print("base append wrote:", rid)
    # 摄入粒度自测
    now = time.time()
    msgs = [
        {"msg_id": 1, "sender": "friend_a", "text": "在吗", "ts": now},
        {"msg_id": 2, "sender": "friend_a", "text": "晚上一起写代码", "ts": now + 60},
        {"msg_id": 3, "sender": "gh_news", "text": "今日头条推送", "ts": now + 120},          # 公众号 → 丢
        {"msg_id": 4, "sender": "friend_b", "text": "[图片]", "ts": now + 130},               # 纯图片 → 丢
        {"msg_id": 5, "sender": "friend_b", "text": "🔥🔥", "ts": now + 140},                 # 纯表情 → 丢
        {"msg_id": 6, "sender": "friend_a", "text": "对了作业pdf发你", "ts": now + 2000},     # >30min → 新段
    ]
    ids = ingest_wechat(w, msgs)
    recs = list(w.iter_source("wechat"))
    print("ingest wrote sessions:", len(ids))
    print("kept sessions:", len(recs), "(应为 2：段1=friend_a 两条, 段2=friend_a 一条)")
    for r in recs:
        print("  session", r["payload"]["session_id"][:8],
              "msg_id", r["payload"]["msg_id_min"], "-", r["payload"]["msg_id_max"],
              "count", r["payload"]["msg_count"],
              "participants", r["payload"]["participants"])
    assert len(recs) == 2, "粒度过滤异常"
    print("L0 granularity OK ✅")
