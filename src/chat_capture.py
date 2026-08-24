#!/usr/bin/env python3
"""#16 对话捕获：dsh session log → L0 记忆桥（设计预设：旧文档 §6 mode 标签 /
草案 §5 persona 隔离 / dsh 调研「Model-visible means logged」）。

dsh web/headless 的对话只落在 ~/.dsh/sessions/<cwd-slug>/session-<uuid>/session.jsonl.zstd，
本桥把它们增量摄入 L0（source=chat），让夜班巩固看得到「用户和本地模型聊了什么」。

提取规则（session.jsonl 条目类型）：
  user/message     data.content[].type==text，且 source.kind==user；
                   跳过运行时注入（"Current runtime context" 开头等非真人内容）
  assistant/message data.message.content[] 只取 type==text 部分（reasoning/tool 不要）
  tool/*, step/*, turn/* 等结构性条目不摄入（巩固只关心对话内容）

mode 标签：
  默认 normal；session 标题/cwd 命中 persona 关键词（megumin/惠惠/persona）→ persona。
  persona 不进 L2 语义索引（l2_semantic 摄入侧按 mode 过滤，见 §5）。

幂等水位：memory/L1_working/chat_capture_state.json，per-session 记录已摄入的最大 seq。
解压：session.jsonl.zstd 走 zstd CLI（anaconda3 自带，~/anaconda3/bin/zstd）。

用法：
  python3 chat_capture.py run      # 增量摄入（m4_ingest chat 步调它）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l0_ingest import L0Writer  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSH_SESSIONS = os.path.expanduser("~/.dsh/sessions")
STATE_PATH = os.path.join(REPO, "memory", "L1_working", "chat_capture_state.json")
L0_ROOT = os.path.join(REPO, "memory", "L0_raw")
ZSTD = "~/anaconda3/bin/zstd"

PERSONA_KEYWORDS = ("megumin", "惠惠", "persona")
# 运行时注入/系统条目特征：非真人消息，跳过
INJECT_MARKERS = ("Current runtime context", "This snapshot supersedes", "<system-reminder")


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"sessions": {}}


def save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


def _decompress(path: str) -> str:
    r = subprocess.run([ZSTD, "-d", "-c", path], capture_output=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"zstd 解压失败: {path}")
    return r.stdout.decode("utf-8", errors="replace")


def _text_parts(content: list) -> str:
    return "\n".join(c.get("text", "") for c in content if c.get("type") == "text").strip()


def extract_messages(raw: str, after_seq: int) -> tuple[list[dict], int, str]:
    """从解压后的 jsonl 提取对话对。返回 (消息列表, 最大 seq, session 标题)。"""
    msgs = []
    max_seq = after_seq
    title = ""
    pending_user: dict | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        seq = int(d.get("seq", 0) or 0)
        ts = int(d.get("time", 0) or 0) / 1000.0

        if t == "session/title":
            title = str(d.get("data", {}).get("title", "") or "")
            continue

        if t == "user/message" and seq > after_seq:
            data = d.get("data", {})
            if data.get("source", {}).get("kind") != "user":
                continue
            text = _text_parts(data.get("content", []))
            if not text or any(m in text[:200] for m in INJECT_MARKERS):
                continue
            pending_user = {"role": "user", "text": text, "ts": ts}
            max_seq = max(max_seq, seq)

        elif t == "assistant/message" and seq > after_seq and pending_user is not None:
            content = d.get("data", {}).get("message", {}).get("content", [])
            text = _text_parts(content)
            if text:
                msgs.append(pending_user)
                msgs.append({"role": "assistant", "text": text, "ts": ts})
            pending_user = None
            max_seq = max(max_seq, seq)

    return msgs, max_seq, title


def run() -> None:
    if not os.path.isdir(DSH_SESSIONS):
        print(f"[chat] dsh sessions 目录不存在: {DSH_SESSIONS}")
        return
    st = load_state()
    w = L0Writer(L0_ROOT)
    total = 0

    for proj in os.listdir(DSH_SESSIONS):
        proj_dir = os.path.join(DSH_SESSIONS, proj)
        if not os.path.isdir(proj_dir):
            continue
        for sess in os.listdir(proj_dir):
            fp = os.path.join(proj_dir, sess, "session.jsonl.zstd")
            if not os.path.isfile(fp):
                continue
            key = f"{proj}/{sess}"
            after = int(st["sessions"].get(key, {}).get("seq", 0))
            try:
                raw = _decompress(fp)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"[chat] 跳过 {key}: {e}")
                continue
            msgs, max_seq, title = extract_messages(raw, after)
            if not msgs:
                if max_seq > after:
                    st["sessions"][key] = {"seq": max_seq}
                continue
            # mode 标签：标题或会话路径命中 persona 关键词 → persona（§5 隔离）
            haystack = (title + " " + key).lower()
            mode = "persona" if any(k in haystack for k in PERSONA_KEYWORDS) else "normal"
            w.append("chat", {
                "session": sess,
                "title": title,
                "messages": msgs,
                "turns": len(msgs) // 2,
            }, mode=mode, sensitive=True, meta={"ingest": "chat_capture"})
            st["sessions"][key] = {"seq": max_seq}
            total += len(msgs)
            print(f"[chat] {sess[:24]}… +{len(msgs)} 条消息（mode={mode}）入 L0")

    save_state(st)
    if total == 0:
        print("[chat] 无新对话")
    else:
        print(f"[chat] 共摄入 {total} 条消息")


if __name__ == "__main__":
    run()
