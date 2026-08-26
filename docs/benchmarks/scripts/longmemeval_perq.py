# -*- coding: utf-8 -*-
"""LongMemEval per-question 隔离评估（模拟官方设定，Session 粒度）。

每题独立：haystack 会话 → bge-m3 嵌入（会话级向量）→ question 余弦检索 top-k →
Recall@1/3/5（answer 会话是否在 top-k）。500 题全量。

用法：python longmemeval_perq.py > /tmp/longmemeval-perq.log 2>&1
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "local-ai-agent", "src"))

DATA = "/tmp/longmemeval_s_cleaned.json"
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "research", "LongMemEval-perquestion-2026-08-26.md")


def main():
    import numpy as np
    import l2_semantic as l2
    emb = l2.Embedder()
    data = json.load(open(DATA, encoding="utf-8"))
    print(f"per-question 评估：{len(data)} 题（每题独立嵌入 haystack 会话）", flush=True)

    t0 = time.time()
    stats = {}
    for i, q in enumerate(data, 1):
        # 每题：会话文本列表 + 答案会话 id 集合
        sids = q["haystack_session_ids"]
        sessions = q["haystack_sessions"]
        texts = []
        for msgs in sessions:
            texts.append("\n".join(m.get("content", "") for m in msgs))
        ans = set(q["answer_session_ids"])

        # 会话嵌入（批量）
        vecs = np.array(emb.embed(texts))          # (N, 1024)
        qvec = np.array(emb.embed([q["question"]])[0])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        sims = (vecs @ qvec) / (norms[:, 0] * (np.linalg.norm(qvec) + 1e-9))
        order = np.argsort(-sims)                   # 相似度降序

        hit_at = None
        for rank in range(min(5, len(order))):
            if sids[order[rank]] in ans:
                hit_at = rank + 1
                break
        stats.setdefault(q["question_type"], []).append(hit_at)
        if i % 100 == 0:
            print(f"  {i}/{len(data)}（{time.time()-t0:.0f}s）", flush=True)

    lines = ["# LongMemEval per-question 隔离评估（模拟官方 Session 粒度）— 2026-08-26", "",
             "| 类别 | N | R@1 | R@3 | R@5 | 未命中 |", "|---|---|---|---|---|---|"]
    n = 0; allh = []
    for cat, hs in sorted(stats.items()):
        n += len(hs); allh += [h for h in hs if h]
        r1 = sum(1 for h in hs if h and h <= 1) / len(hs)
        r3 = sum(1 for h in hs if h and h <= 3) / len(hs)
        r5 = sum(1 for h in hs if h and h <= 5) / len(hs)
        lines.append(f"| {cat} | {len(hs)} | {r1:.2f} | {r3:.2f} | {r5:.2f} | "
                     f"{sum(1 for h in hs if not h)} |")
        print(f"  [{cat}] R@1={r1:.2f} R@3={r3:.2f} R@5={r5:.2f} (N={len(hs)})", flush=True)
    r1 = sum(1 for h in allh if h <= 1) / n
    r3 = sum(1 for h in allh if h <= 3) / n
    r5 = sum(1 for h in allh if h <= 5) / n
    lines.append(f"| **总体** | {n} | {r1:.2f} | {r3:.2f} | {r5:.2f} | {n-len(allh)} |")
    lines.append(f"\n**耗时 {time.time()-t0:.0f}s；对照混合库 R@1=0.16/R@5=0.25**")
    print(f"\n=== 总体 R@1={r1:.2f} R@3={r3:.2f} R@5={r5:.2f}（耗时 {time.time()-t0:.0f}s）===", flush=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
