# -*- coding: utf-8 -*-
"""LongMemEval per-question × 沙盒 L2 管线版（系统级官方分）。

每题：独立临时 l2.db（SCHEMA + vec0 + fts5，该题 ~50 会话）→ 完整 search()（ANN+BM25+RRF+
时间加权）→ 27B 作答 → 双口径。真正走"系统检索器"，非纯余弦。

用法：python longmemeval_perq_pipeline.py > /tmp/longmemeval-perq-pipeline.log 2>&1
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "local-ai-agent", "src"))

DATA = "/tmp/longmemeval_s_cleaned.json"
LLM = "http://127.0.0.1:8100/v1/chat/completions"
VCACHE = "/tmp/longmemeval-perq-vecs"
TMPDB = "/tmp/longmemeval-perq-l2"
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "research", "LongMemEval-perquestion-pipeline-2026-08-26.md")


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s%.$€£]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def ask27b(context: str, question: str) -> str:
    body = json.dumps({
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "messages": [
            {"role": "system", "content": "你是记忆检索助手。根据给定的对话记忆仔细回答问题。"
             "记忆里可能包含多个无关对话，请从中找出与问题相关的信息并作答。"
             "回答要简短（一句话）。确实找不到相关信息时，直接回答「不知道」。"},
            {"role": "user", "content": f"对话记忆内容（可能含无关对话）：\n{context}\n\n"
             f"问题：{question}\n\n回答："}],
        "max_tokens": 300, "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(LLM, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def build_perq_db(q, emb, DIM=1024) -> tuple[sqlite3.Connection, list[str], list[str]]:
    """建该题独立临时 L2 库；返回 (conn, texts, sids)。"""
    import sqlite_vec
    import l2_semantic as l2
    os.makedirs(TMPDB, exist_ok=True)
    dbp = os.path.join(TMPDB, f"{q['question_id']}.db")
    if os.path.exists(dbp):
        os.remove(dbp)
    conn = sqlite3.connect(dbp)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(l2.SCHEMA)
    conn.execute(f"CREATE VIRTUAL TABLE vec_docs USING vec0(doc_id TEXT PRIMARY KEY, embedding float[{DIM}])")
    conn.execute("CREATE VIRTUAL TABLE fts_docs USING fts5(doc_id UNINDEXED, text)")
    conn.execute("PRAGMA journal_mode=MEMORY")

    sids = q["haystack_session_ids"]
    sessions = q["haystack_sessions"]
    texts = ["\n".join(m.get("content", "") for m in msgs) for msgs in sessions]
    cache_p = os.path.join(VCACHE, f"{q['question_id']}.npy")
    if os.path.exists(cache_p):
        import numpy as np
        vecs = np.load(cache_p)
    else:
        import numpy as np
        vecs = np.array(emb.embed(texts))
        np.save(cache_p, vecs)

    now = time.time()
    seen = set()
    for sid, txt, v in zip(sids, texts, vecs):
        if sid in seen:              # 数据含重复 sid（13/500 题）→ 只插一次（vec0 不支持 OR IGNORE）
            continue
        seen.add(sid)
        did = hashlib.md5(f"{sid}|{txt}".encode()).hexdigest()
        conn.execute("INSERT INTO docs(doc_id, source, ref, text, ts, stale) VALUES(?,?,?,?,?,0)",
                     (did, "longmemeval", sid, txt[:400000], now))
        conn.execute("INSERT INTO vec_docs(doc_id, embedding) VALUES(?,?)",
                     (did, json.dumps(v.tolist())))
        conn.execute("INSERT INTO fts_docs(doc_id, text) VALUES(?,?)", (did, txt[:400000]))
    conn.commit()
    return conn, texts, sids


def main():
    import l2_semantic as l2
    # 官方 LongMemEval 设定无「用户 L3 事实」概念：禁用 boost，否则 L3 查询在无关
    # 临时库上加权无关会话、挤掉答案（2026-08-26 诊断：R@5 0.27→0.9+ 的关键差异）
    l2._load_important_facts = lambda *a, **k: []
    emb = l2.Embedder()
    data = json.load(open(DATA, encoding="utf-8"))
    print(f"per-question × L2 管线：{len(data)} 题（每题独立库 + 完整 search，禁 boost）", flush=True)

    stop = set("the a an of to in on for with and or is are was be i you your my me "
               "it this that what how can do did not would".split())
    orig_get_db = l2.get_db
    t0 = time.time()
    stats = {}
    for i, q in enumerate(data, 1):
        conn, texts, sids = build_perq_db(q, emb)
        l2.get_db = lambda c=conn: c        # 完整 search() 用该题临时库
        ans = set(q["answer_session_ids"])
        try:
            res = l2.search(q["question"], k=5)
        except Exception as e:
            print(f"  search 失败 {q['question_id']}: {e}", flush=True)
            res = []
        conn.close()
        l2.get_db = orig_get_db

        hit = any(r.get("ref") in ans for r in res)
        # context = top5 会话全文（按 search 返回 ref 取原始文本）
        picked = []
        for r in res[:5]:
            sid = r.get("ref")
            if sid and sid in sids:
                picked.append(texts[sids.index(sid)])
        context = "\n---\n".join(s[:3000] for s in picked)[:9000]

        gen = ask27b(context, q["question"])
        n_ans, n_gen = norm(str(q["answer"])), norm(gen)
        strict = bool(n_ans) and n_ans in n_gen
        core = [t for t in re.findall(r"[a-z']{4,}", n_ans) if t not in stop]
        if not core:
            core = [d.strip(".,") for d in re.findall(r"\d[\d.,$%]*", n_ans)]
        partial = bool(not strict and core and any(c in n_gen for c in core)
                       and "不知道" not in gen and "没有" not in gen[:6])
        stats.setdefault(q["question_type"], []).append(
            {"ok": bool(strict or partial), "strict": strict, "hit": hit})
        if i % 50 == 0:
            print(f"  {i}/{len(data)}（{time.time()-t0:.0f}s）", flush=True)

    lines = ["# LongMemEval per-question × L2 管线版（系统级官方分）— 2026-08-26", "",
             "| 类别 | N | search R@5 | 严格 | 严格+部分 |", "|---|---|---|---|---|"]
    n = ok = st = hh = 0
    for cat, rs in sorted(stats.items()):
        n += len(rs); ok += sum(r["ok"] for r in rs); st += sum(r["strict"] for r in rs); hh += sum(r["hit"] for r in rs)
        lines.append(f"| {cat} | {len(rs)} | {sum(r['hit'] for r in rs)/len(rs):.2f} | "
                     f"{sum(r['strict'] for r in rs)} | {sum(r['ok'] for r in rs)} |")
        print(f"  [{cat}] search R@5={sum(r['hit'] for r in rs)/len(rs):.2f} "
              f"严格={sum(r['strict'] for r in rs)} 严格+部分={sum(r['ok'] for r in rs)} (N={len(rs)})", flush=True)
    lines.append(f"| **总体** | {n} | {hh/n:.2f} | {st} | {ok} |")
    lines.append(f"\n**端到端 Accuracy(含部分) = {ok/n:.2f}，严格 = {st/n:.2f}，"
                 f"search 命中→答对转化率 = {ok/hh:.2f}（命中 {hh} 题）**")
    lines.append(f"\n**对照：纯余弦 per-question 检索 R@5=0.93；官方 Table3 Session top-5：GPT-4o 0.67 / L3.1-70B 0.59 / L3.1-8B 0.52**")
    print(f"\n=== 系统级端到端：Accuracy={ok/n:.2f}（严格 {st/n:.2f}），search R@5={hh/n:.2f} ===", flush=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
