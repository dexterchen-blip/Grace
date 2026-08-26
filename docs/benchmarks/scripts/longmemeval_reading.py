# -*- coding: utf-8 -*-
"""LongMemEval Reading 阶段评估（沙盒）：检索 → 27B 生成回答 → 与 ground truth 比对。

链路：24 题 sample（seed 42）→ search_v2(池200) top5 块作 context → :8100 27B 作答 →
归一化包含匹配评判。对照 Recall@5 看 Retrieval→Reading 衰减。

用法：AIAGENT_SANDBOX=<test-sandbox> python longmemeval_reading.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "local-ai-agent", "src"))

DATA = "/tmp/longmemeval_s_cleaned.json"
LLM = "http://127.0.0.1:8100/v1/chat/completions"


def norm(s: str) -> str:
    """归一化：小写、去标点（保留 % 数字）、压空格。"""
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
    req = urllib.request.Request(LLM, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"].strip()


def load_session_text(sid: str) -> str | None:
    """从沙盒 exchange 读完整会话文本（块索引定位 → 完整记忆读取）。"""
    import glob
    base = os.path.join(os.environ.get("AIAGENT_SANDBOX", ""), "exchange", "school", "longmemeval")
    for p in glob.glob(f"{base}/Q*/*.txt"):
        if p.split("/")[-1].replace(".txt", "") == sid:
            with open(p, encoding="utf-8") as f:
                return f.read()
    return None


def search_pool(q, k=5, pool=200):
    import l2_semantic as l2
    emb = l2.Embedder(); db = l2.get_db()
    qv = json.dumps(emb.embed([q])[0])
    vec = db.execute("SELECT doc_id FROM vec_docs WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                     (qv, pool)).fetchall()
    fq = " OR ".join(f'"{t}"' for t in re.findall(r"[A-Za-z0-9\-]{2,}", q)) or f'"{q}"'
    fts = db.execute("SELECT doc_id FROM fts_docs WHERE fts_docs MATCH ? ORDER BY bm25(fts_docs) LIMIT ?",
                     (fq, pool)).fetchall()
    scores = {}
    for rank, (d,) in enumerate(vec): scores[d] = scores.get(d, 0) + 1.0 / (60 + rank + 1)
    for rank, (d,) in enumerate(fts): scores[d] = scores.get(d, 0) + 1.0 / (60 + rank + 1)
    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    # 命中块 → 展开为完整会话文本（去重，最多 3 个会话）
    seen, sessions = set(), []
    for d, _ in top:
        r = db.execute("SELECT text FROM docs WHERE doc_id=?", (d,)).fetchone()
        if not r: continue
        m = re.search(r"\[session ([^\]]+)\]", r[0])
        if not m or m.group(1) in seen: continue
        seen.add(m.group(1))
        full = load_session_text(m.group(1))
        if full:
            sessions.append(full)
        if len(sessions) >= 3:
            break
    return sessions


def main():
    data = json.load(open(DATA, encoding="utf-8"))
    random.seed(42)
    by_cat = {}
    for q in data: by_cat.setdefault(q["question_type"], []).append(q)
    sample = []
    for cat, qs in by_cat.items(): sample.extend(random.sample(qs, 4))

    print(f"Reading 评估：{len(sample)} 题（检索 top5 块 → 展开完整会话 → 27B 作答）\n")
    stats = {}
    for i, q in enumerate(sample, 1):
        sessions = search_pool(q["question"], k=5)
        context = "\n---\n".join(s[:3000] for s in sessions)[:9000]
        ans = ask27b(context, q["question"])
        # 命中答案会话？
        hit = any(f"[session {s}]" in ctx for s in set(q["answer_session_ids"])
                  for ctx in sessions)
        # 评判双口径：严格包含 + 核心词部分命中（处理拼写变体/长答案）
        n_ans, n_gen = norm(str(q["answer"])), norm(ans)
        strict = bool(n_ans) and n_ans in n_gen
        # 核心词：answer 中 ≥4 字母的实词（排除停用词/数字单独处理）
        stop = set("the a an of to in on for with and or is are was be i you your my me "
                   "it this that what how can do did did not would".split())
        core = [t for t in re.findall(r"[a-z']{4,}", n_ans) if t not in stop]
        if not core:
            digits = re.findall(r"\d[\d.,$%]*", n_ans)
            core = [d.strip(".,") for d in digits] if digits else []
        partial = bool(not strict and core and any(c in n_gen for c in core)
                       and "不知道" not in ans and "没有" not in ans[:6])
        correct = bool(strict or partial)
        stats.setdefault(q["question_type"], []).append({
            "correct": correct, "strict": strict, "partial": partial, "hit": hit,
            "q": q["question"][:50], "ans": str(q["answer"])[:30],
            "gen": ans[:60]})
        flag = "✓" if strict else ("~" if partial else "✗")
        print(f"[{i:>2}] {flag} {q['question_type'][:14]:<14} 严格={strict} 部分={partial} "
              f"检索命中={hit} 期望={str(q['answer'])[:20]!r} 生成={ans[:40]!r}")

    print(f"\n{'类别':<24}{'N':>3}{'检索命中':>8}{'答对(严格+部分)':>14}")
    n = ok = st = 0
    for cat, rs in sorted(stats.items()):
        n += len(rs)
        k = sum(r["correct"] for r in rs)
        s = sum(r["strict"] for r in rs)
        h = sum(r["hit"] for r in rs)
        ok += k; st += s
        print(f"{cat:<24}{len(rs):>3}{h:>8}{k:>6}（严格 {s}）")
    print(f"{'总体':<24}{n:>3}{sum(r['hit'] for rs in stats.values() for r in rs):>8}{ok:>6} "
          f"（严格 {st}，Accuracy(含部分)={ok/n:.2f}）")


if __name__ == "__main__":
    main()
