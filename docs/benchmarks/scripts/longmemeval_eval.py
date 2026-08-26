# -*- coding: utf-8 -*-
"""LongMemEval 检索级评估（沙盒）：微软系先进记忆 benchmark 适配。

方法：抽样 6 类 × N 题 → 会话转文本入沙盒 L2（sid 文件名）→ 每题 question 检索 top-k
→ Recall@k（答案会话的块是否在 top-k，按 docs.ref 判定）→ 按类统计。

用法：AIAGENT_SANDBOX=<test-sandbox> python longmemeval_eval.py [--n-per-class 4] [--ingest-only]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

DATA = "/tmp/longmemeval_s_cleaned.json"
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "local-ai-agent")
OUT_EX = os.environ.get("AIAGENT_SANDBOX", "")  # 沙盒根
if not OUT_EX:
    sys.exit("需 AIAGENT_SANDBOX 环境变量（沙盒隔离）")
EXCHANGE = os.path.join(OUT_EX, "exchange", "school", "longmemeval")


def sessions_to_text(sessions: list, session_ids: list) -> list[tuple[str, str]]:
    """[(sid, 文本)]；对话转可读文本。"""
    out = []
    for sid, msgs in zip(session_ids, sessions):
        lines = [f"[session {sid}]"]
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        out.append((sid, "\n".join(lines)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=4)
    ap.add_argument("--ingest-only", action="store_true")
    args = ap.parse_args()

    data = json.load(open(DATA, encoding="utf-8"))
    by_cat: dict[str, list] = {}
    for q in data:
        by_cat.setdefault(q["question_type"], []).append(q)
    # 抽样（固定种子可复现）
    import random
    random.seed(42)
    sample: list = []
    for cat, qs in by_cat.items():
        sample.extend(random.sample(qs, min(args.n_per_class, len(qs))))
    print(f"抽样 {len(sample)} 题（{args.n_per_class}×{len(by_cat)} 类）")

    # ① 生成会话文件
    os.makedirs(EXCHANGE, exist_ok=True)
    n_files = 0
    for q in sample:
        qdir = os.path.join(EXCHANGE, f"Q{q['question_id']}")
        os.makedirs(qdir, exist_ok=True)
        for sid, text in sessions_to_text(q["haystack_sessions"],
                                          q["haystack_session_ids"]):
            with open(os.path.join(qdir, f"{sid}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            n_files += 1
    print(f"已生成 {n_files} 个会话文件 → {EXCHANGE}")
    if args.ingest_only:
        return

    # ② 摄入（scan → L0/L2）
    print("\n② 摄入沙盒 L0/L2 ...")
    env = {**os.environ, "AIAGENT_SANDBOX": OUT_EX}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    r = subprocess.run([sys.executable, "src/m4_ingest.py", "scan"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    print("  scan:", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
    r2 = subprocess.run([os.environ.get("LLAMA_PY", "python3"),  # 适配你的 llama-cpp venv python
                         "src/l2_semantic.py", "build"],
                        cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    print("  build:", r2.stdout.strip().splitlines()[-1] if r2.stdout else r2.stderr[-200:])

    # ③ 检索评估
    print("\n③ 检索评估（Recall@k）...")
    sys.path.insert(0, os.path.join(REPO, "src"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import l2_semantic as l2
    from l3_time_sort_experiment import parse_date  # noqa: F401

    def search_v2(q, k=8, rw=1.0, bw=0.8, iw=1.5):
        emb = l2.Embedder(); db = l2.get_db(); now = time.time()
        qv = json.dumps(emb.embed([q])[0])
        vec = db.execute("SELECT doc_id FROM vec_docs WHERE embedding MATCH ? ORDER BY distance LIMIT 24", (qv,)).fetchall()
        fq = " OR ".join(f'"{t}"' for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9\-]{2,}", q)) or f'"{q}"'
        fts = db.execute("SELECT doc_id FROM fts_docs WHERE fts_docs MATCH ? ORDER BY bm25(fts_docs) LIMIT 24", (fq,)).fetchall()
        scores = {}
        for rank, (d,) in enumerate(vec): scores[d] = scores.get(d, 0) + 1.0 / (60 + rank + 1)
        for rank, (d,) in enumerate(fts): scores[d] = scores.get(d, 0) + 1.0 / (60 + rank + 1)
        for fact in (l2._load_important_facts() or []):
            bv = json.dumps(emb.embed([fact])[0])
            for rank, (d,) in enumerate(db.execute("SELECT doc_id FROM vec_docs WHERE embedding MATCH ? ORDER BY distance LIMIT 8", (bv,)).fetchall()):
                scores[d] = scores.get(d, 0) + bw * 1.5 / (60 + rank + 1)
        out = []
        for d, sc in scores.items():
            r = db.execute("SELECT ref, text FROM docs WHERE doc_id=?", (d,)).fetchone()
            if not r: continue
            out.append((d, sc, r[0] or "", r[1] or ""))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    stats: dict[str, list] = {}
    for q in sample:
        hits = search_v2(q["question"], k=5)
        ans_sids = set(q["answer_session_ids"])
        # 命中 = top-k 块文本含答案会话标记 [session <sid>]（docs.ref 是哈希，不可用）
        hit_at = None
        for i, (_, _, _, txt) in enumerate(hits):
            if any(f"[session {s}]" in (txt or "") for s in ans_sids):
                hit_at = i + 1
                break
        stats.setdefault(q["question_type"], []).append(hit_at)

    print(f"\n{'类别':<24}{'N':>3} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'未命中':>5}")
    all_hits = []
    for cat, hits in sorted(stats.items()):
        n = len(hits)
        r1 = sum(1 for h in hits if h and h <= 1) / n
        r3 = sum(1 for h in hits if h and h <= 3) / n
        r5 = sum(1 for h in hits if h and h <= 5) / n
        miss = sum(1 for h in hits if not h)
        all_hits += [h for h in hits if h]
        print(f"{cat:<24}{n:>3} {r1:>6.2f} {r3:>6.2f} {r5:>6.2f} {miss:>5}")
    n = len(all_hits) + sum(1 for c, hs in stats.items() for h in hs if not h)
    print(f"{'总体':<24}{n:>3} "
          f"{sum(1 for h in all_hits if h<=1)/n:>6.2f} "
          f"{sum(1 for h in all_hits if h<=3)/n:>6.2f} "
          f"{sum(1 for h in all_hits if h<=5)/n:>6.2f} "
          f"{n-len(all_hits):>5}")


if __name__ == "__main__":
    main()
