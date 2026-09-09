#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""seed_l2.py — V2.4 书库 L2 种子（2026-09-09）。

背景：reset_stress 每轮归档 L0/索引 → 语义检索无料可查（静默降级一周未觉）。
本脚本在每轮 reset 后、引擎启动前运行：把 inputs-v3 真实书库日转成 L0 格式
(book.jsonl)，随后由 auto_round 调 l2_semantic.py build 重建索引。
幂等：每次全量重写 book.jsonl（内容只依赖 inputs-v3）。

用法（由 auto_round.sh 调用，勿手动）：
    HF_HUB_OFFLINE=1 <llama-cpp-venv>/bin/python3 v2/stress/seed_l2.py
"""
from __future__ import annotations
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))            # ai-sandbox-stress/
SRC_DAYS = os.path.join(ROOT, "experiments", "run", "stress", "inputs-v3")
OUT = os.path.join(ROOT, "memory", "L0_raw", "book.jsonl")


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for fp in sorted(glob.glob(os.path.join(SRC_DAYS, "day-*.json"))):
            try:
                d = json.load(open(fp, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            day = d.get("day") or os.path.basename(fp)[4:7]
            msgs = d.get("messages") or []
            if not msgs:
                continue
            rec = {"id": f"book-v3-d{day}", "epoch": None, "source": "book",
                   "payload": {"messages": [{"sender": f"书库d{day}", "text": m.get("text", "")}
                                            for m in msgs if m.get("text")],
                               "title": f"书库 day {day}"},
                   "meta": {"ingest": "book-l2-seed"}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[seed-l2] book.jsonl 重写完成: {n} 源天")
    # ★2026-09-09 清残留: reset 只清 docs 清不了 vec0 虚拟表（普通连接无 vec 模块）——
    #   残留 vec 行会让 build 撞 UNIQUE。本进程（llama-cpp venv）能加载 sqlite_vec，负责全清。
    import sqlite3
    try:
        import sqlite_vec
        db = sqlite3.connect(os.path.join(ROOT, "memory", "L2_semantic", "l2.db"))
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        total = 0
        for t in ("docs", "doc_entities", "relations", "fts_docs",
                  "vec_docs", "vec_docs_info", "vec_docs_chunks",
                  "vec_docs_rowids", "vec_docs_vector_chunks00"):
            try:
                cur = db.execute("DELETE FROM " + t)
                total += cur.rowcount
            except Exception:
                pass  # 表不存在——跳过
        db.commit(); db.close()
        print("[seed-l2] 语义索引残留清除: %d 行（vec+docs 全清，重建从零）" % total)
    except Exception as e:
        print("[seed-l2] 残留清除失败（build 可能撞 UNIQUE）: %s" % e)


if __name__ == "__main__":
    main()
