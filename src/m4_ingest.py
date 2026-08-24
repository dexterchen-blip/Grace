#!/usr/bin/env python3
"""M4 摄入管线：外部数据源 → exchange 总线 → L0 原始层（框架无关，标准库 only）。

数据源接线（设计 §11/§13，审计 B2/B4）：
  wechat  微信 summary.db（wechat-summary-bot 每日产出）→ 会话切片 → L0（sensitive）
  school  学校初始语料（工作区根 ucsb_*.md 等，B2 点名）→ exchange/school/initial-corpus/ → L0
  email   邮箱摘要 md（自动化产出，B4 落点=exchange/inbox/email/）→ L0（sensitive）
  scan    exchange 热文件夹（inbox/cloud-drop/school）新文件 → L0（增量，通用收口）
  chat    dsh 会话日志（~/.dsh/sessions/*.zstd）→ L0（#16 对话捕获桥，chat_capture.py）
  all     以上全部（夜班/手动一键跑）

幂等：memory/L1_working/ingest_state.json 记水位（wechat=max ts；文件=relpath+内容 md5，2026-08-20 起）。
微信全量历史（decrypted/ 137 Msg_* 表，13.2 万条）回填是独立数据作业，不在本管线内。

用法：
  python3 m4_ingest.py all        # 全量接线跑一遍
  python3 m4_ingest.py wechat     # 只跑微信增量
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l0_ingest import L0Writer, ingest_wechat  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # local-ai-agent/
WS = os.path.dirname(REPO)                                            # 工作区根
WECHAT_DB = os.path.join(WS, "wechat-summary-bot", "summary.db")
L0_ROOT = os.path.join(REPO, "memory", "L0_raw")
STATE_PATH = os.path.join(REPO, "memory", "L1_working", "ingest_state.json")
EXCHANGE = os.path.join(REPO, "exchange")

# 测试沙盒（2026-08-22）：AIAGENT_SANDBOX=<dir> 时全部写路径重定向到沙盒，
# 与正式记忆/交换系统完全隔离（沙盒目录须含 memory/ + exchange/ 结构）。
SANDBOX = os.environ.get("AIAGENT_SANDBOX", "")
if SANDBOX:
    L0_ROOT = os.path.join(SANDBOX, "memory", "L0_raw")
    STATE_PATH = os.path.join(SANDBOX, "memory", "L1_working", "ingest_state.json")
    EXCHANGE = os.path.join(SANDBOX, "exchange")

# B2 点名的学校初始语料（工作区根既有文件）
SCHOOL_CORPUS = [
    "ucsb_barc.md", "ucsb_gold.md", "ucsb_orientation.md",
    "ucsb_orientation_session.md", "选课建议-2026-Fall.md",
    "分组名册-按intern.md", "orientation-zoom-简报.md",
]
# B4 自动化产出物落点：邮箱摘要 → exchange/inbox/email/
EMAIL_CORPUS = ["邮箱摘要-2026-08-09.md"]


# ---------------------------------------------------------------- 状态（幂等水位）
def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"wechat_max_ts": 0, "files": {}}


def save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


def _content_sig(text: str) -> str:
    """文件水位签名 = 内容 md5（2026-08-20 起替代 [mtime, size]）。
    修复：夜班每晚重写摘要导致 mtime 变 → 内容没变也整文件重摄入（重复进 L0）。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _migrate_sig(st: dict) -> int:
    """旧水位 [mtime,size] → 内容 md5（一次性迁移，不重摄任何内容）。
    迁移时读文件算当前内容 hash 写回账本；文件已不存在 → 删条目。"""
    files = st.setdefault("files", {})
    n = 0
    for rel, sig in list(files.items()):
        if isinstance(sig, list):
            for root in (REPO, WS):  # 账本 key 两种根：REPO 相对（落点副本）/ WS 相对（源）
                p = os.path.join(root, rel)
                if os.path.isfile(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            files[rel] = _content_sig(f.read())
                        n += 1
                    except (UnicodeDecodeError, OSError):
                        files.pop(rel, None)
                    break
            else:
                files.pop(rel, None)  # 文件不存在，条目作废
    return n


# ---------------------------------------------------------------- wechat
def run_wechat(w: L0Writer, st: dict) -> None:
    if not os.path.exists(WECHAT_DB):
        print(f"[wechat] 库不存在，跳过: {WECHAT_DB}")
        return
    since = float(st.get("wechat_max_ts", 0))
    con = sqlite3.connect(f"file:{WECHAT_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT conversation, msg_id, display_name, is_send, ts, content"
        " FROM messages WHERE ts > ? ORDER BY conversation, ts, msg_id",
        (since,),
    ).fetchall()
    con.close()
    if not rows:
        print("[wechat] 无新消息")
        return
    by_conv: dict[str, list[dict]] = {}
    for conv, msg_id, name, is_send, ts, content in rows:
        by_conv.setdefault(conv, []).append({
            "msg_id": msg_id,
            "sender": conv,              # wxid；gh_ 公众号在此被粒度规则过滤
            "display_name": name,
            "is_send": bool(is_send),
            "text": content or "",
            "ts": float(ts),
        })
    total_sessions = 0
    for conv, msgs in by_conv.items():
        ids = ingest_wechat(w, msgs, source="wechat", sensitive=True)
        total_sessions += len(ids)
    st["wechat_max_ts"] = max(float(r[4]) for r in rows)
    print(f"[wechat] 新消息 {len(rows)} 条 / {len(by_conv)} 会话 → {total_sessions} 个会话段入 L0（sensitive）")


# ---------------------------------------------------------------- 文件类语料
def _ingest_file(w: L0Writer, st: dict, src_path: str, dst_dir: str,
                 source: str, sensitive: bool) -> bool:
    """单个 md 文件：拷入落点目录（若来源在外部）+ 写一条 L0。返回是否摄入。
    水位 = 内容 md5：内容一字未变则跳过（重写/改 mtime 不重摄）。"""
    rel = os.path.relpath(src_path, WS)
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()
    sig = _content_sig(text)
    if st["files"].get(rel) == sig:
        return False
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if os.path.abspath(src_path) != os.path.abspath(dst_path):
        shutil.copy2(src_path, dst_path)
    # 落点副本也登记水位（内容相同 → hash 一致）——否则 scan 步骤会把
    # 落点副本当新文件再摄入一遍（首跑已踩：8 个语料双重入库）。
    st["files"][os.path.relpath(dst_path, REPO)] = sig
    w.append(source, {"filename": os.path.basename(src_path), "text": text},
             sensitive=sensitive,
             meta={"ingest": "corpus_file", "landing": os.path.relpath(dst_path, REPO)})
    st["files"][rel] = sig
    return True


def run_school(w: L0Writer, st: dict) -> None:
    dst = os.path.join(EXCHANGE, "school", "initial-corpus")
    n = 0
    for name in SCHOOL_CORPUS:
        p = os.path.join(WS, name)
        if os.path.exists(p):
            n += _ingest_file(w, st, p, dst, "school", sensitive=False)
        else:
            print(f"[school] 缺文件: {name}")
    print(f"[school] 初始语料新摄入 {n} 个 → exchange/school/initial-corpus/ + L0")


def run_email(w: L0Writer, st: dict) -> None:
    dst = os.path.join(EXCHANGE, "inbox", "email")
    n = 0
    for name in EMAIL_CORPUS:
        p = os.path.join(WS, name)
        if os.path.exists(p):
            n += _ingest_file(w, st, p, dst, "email", sensitive=True)
    print(f"[email] 摘要新摄入 {n} 个 → exchange/inbox/email/ + L0（sensitive）")


# ---------------------------------------------------------------- exchange 通用扫描
SCAN_DIRS = ["inbox", "cloud-drop", "school"]


def run_scan(w: L0Writer, st: dict) -> None:
    n = 0
    for top in SCAN_DIRS:
        base = os.path.join(EXCHANGE, top)
        for dirpath, _dirs, files in os.walk(base):
            for fn in sorted(files):
                if fn.startswith(".") or fn.endswith((".tmp", ".part")):
                    continue
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, REPO)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        text = f.read()
                except (UnicodeDecodeError, OSError):
                    continue            # 非文本，留给专门管线
                sig = _content_sig(text)  # 内容 md5：重写但内容不变 → 跳过（不重摄）
                if st["files"].get(rel) == sig:
                    continue
                w.append(f"exchange:{top}", {"path": rel, "text": text},
                         sensitive=(top == "inbox" or "ucsb-scrape" in rel),
                         meta={"ingest": "exchange_scan"})
                st["files"][rel] = sig
                n += 1
    print(f"[scan] exchange 热文件夹新文件 {n} 个入 L0")


# ---------------------------------------------------------------- main
def _run_chat(_w: L0Writer, _st: dict) -> None:
    """#16 对话捕获桥（chat_capture.py 自管水位，不经 ingest_state）。"""
    import chat_capture
    chat_capture.run()


def main(argv: list[str]) -> None:
    steps = {
        "wechat": run_wechat,
        "school": run_school,
        "email": run_email,
        "scan": run_scan,
        "chat": _run_chat,
    }
    which = argv[1:] or ["all"]
    if which == ["all"]:
        which = list(steps)
    w = L0Writer(L0_ROOT)
    st = load_state()
    mig = _migrate_sig(st)
    if mig:
        print(f"[state] 水位签名迁移 {mig} 条 → 内容 md5（旧 [mtime,size] 作废）")
    for name in which:
        if name not in steps:
            print(f"未知步骤: {name}（可选 {list(steps)} / all）")
            sys.exit(2)
        steps[name](w, st)
    save_state(st)
    print("state →", os.path.relpath(STATE_PATH, REPO))


if __name__ == "__main__":
    main(sys.argv)
