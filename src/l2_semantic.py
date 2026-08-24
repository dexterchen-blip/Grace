#!/usr/bin/env python3
"""L2 语义层：sqlite-vec + bge-m3 + FTS5(BM25) 混合检索（设计 §3/§4，任务 #9）。

- 消费 L0（只读，绝不写 L0）；persona/ 子树永不进索引（§5 隔离红线）。
- 一 L0 记录 = 一文档；超长语料按 ~1200 字切块。
- 检索 = 向量 ANN + BM25，RRF 融合（k=60）。
- 遗忘 = 从本索引删除/降级，L0 原文永在。

用法（llama-cpp venv 运行）：
  python3 l2_semantic.py build           # 全量/增量建索引
  python3 l2_semantic.py search "orientation 截止时间" -k 5   # 混合检索
  python3 l2_semantic.py stats           # 索引规模

环境：llama-cpp venv（llama_cpp + sqlite_vec）。嵌入模型 = 真·bge-m3（ggml-org Q8_0 GGUF，
Metal 加速，本地 models/embed/）。HF 直连/代理均被墙，模型经 hf-mirror 下载。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time

import sqlite_vec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L0_ROOT = os.path.join(REPO, "memory", "L0_raw")
DB_PATH = os.path.join(REPO, "memory", "L2_semantic", "l2.db")
EMBED_PATH = os.path.join(REPO, "models", "embed", "bge-m3-q8_0.gguf")

# 测试沙盒（2026-08-22）：AIAGENT_SANDBOX=<dir> 时 L0 读取源与 l2.db 全进沙盒。
SANDBOX = os.environ.get("AIAGENT_SANDBOX", "")
if SANDBOX:
    L0_ROOT = os.path.join(SANDBOX, "memory", "L0_raw")
    DB_PATH = os.path.join(SANDBOX, "memory", "L2_semantic", "l2.db")

DIM = 1024
CHUNK = 1200          # 语料切块阈值（字）
CHUNK_TRIGGER = 1500  # 超过才切


class Embedder:
    """bge-m3 GGUF 经 llama.cpp（Metal）。输出 L2 归一化向量（余弦≈单调于 L2 距离）。"""

    def __init__(self, path: str = EMBED_PATH):
        from llama_cpp import Llama
        self.llm = Llama(model_path=path, embedding=True,
                         n_gpu_layers=-1, n_ctx=8192, verbose=False)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = self.llm.embed(t[:6000])          # bge-m3 上限 8192 token，留余量
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(
  doc_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  ref    TEXT NOT NULL,          -- L0 记录 id 或文件路径（回捞用）
  text   TEXT NOT NULL,
  ts     REAL,
  meta   TEXT,
  hits   INTEGER DEFAULT 0,      -- 命中次数（强化）
  last_hit REAL,                 -- 最后命中时间（复活/遗忘判据）
  stale  INTEGER DEFAULT 0       -- 陈旧标记（L2 遗忘降权）
);
CREATE TABLE IF NOT EXISTS meta_kv(k TEXT PRIMARY KEY, v TEXT);
"""


def _ensure_columns(db: sqlite3.Connection) -> None:
    """旧库补列（hits/last_hit/stale，2026-08-21 分层遗忘 + 强化）。"""
    cols = {r[1] for r in db.execute("PRAGMA table_info(docs)")}
    for name, ddl in [("hits", "INTEGER DEFAULT 0"),
                      ("last_hit", "REAL"),
                      ("stale", "INTEGER DEFAULT 0")]:
        if name not in cols:
            db.execute(f"ALTER TABLE docs ADD COLUMN {name} {ddl}")
    db.commit()


def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.executescript(SCHEMA)
    _ensure_columns(db)
    # vec0 虚拟表（幂等创建）
    exists = db.execute(
        "SELECT name FROM sqlite_master WHERE name='vec_docs'").fetchone()
    if not exists:
        db.execute(f"CREATE VIRTUAL TABLE vec_docs USING vec0(doc_id TEXT PRIMARY KEY, embedding float[{DIM}])")
    exists = db.execute(
        "SELECT name FROM sqlite_master WHERE name='fts_docs'").fetchone()
    if not exists:
        db.execute("CREATE VIRTUAL TABLE fts_docs USING fts5(doc_id UNINDEXED, text)")
    return db


def doc_id(source: str, ref: str, idx: int) -> str:
    return hashlib.sha1(f"{source}|{ref}|{idx}".encode()).hexdigest()


def chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_TRIGGER:
        return [text]
    return [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]


def l0_record_to_text(rec: dict) -> str:
    """L0 记录 → 可嵌入文本。会话段拼接消息；语料文件取正文。"""
    p = rec.get("payload", {})
    if "messages" in p:                       # wechat 会话段
        lines = []
        for m in p["messages"]:
            who = m.get("display_name") or m.get("sender", "?")
            lines.append(f"{who}: {m.get('text', '')}")
        return "\n".join(lines)
    return str(p.get("text", ""))


def iter_l0_docs(only_source: str | None = None):
    """遍历 L0（跳过 persona 子树），yield (source, ref, text, ts, meta)。"""
    for fn in sorted(os.listdir(L0_ROOT)):
        if not fn.endswith(".jsonl"):
            continue
        source = fn[:-6]
        if only_source and source != only_source:
            continue
        with open(os.path.join(L0_ROOT, fn), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = l0_record_to_text(rec)
                if not text.strip():
                    continue
                yield source, rec["id"], text, rec.get("epoch"), rec.get("meta")


def build(only_source: str | None = None, batch: int = 16) -> None:
    emb = Embedder()
    db = get_db()
    existing = {r[0] for r in db.execute("SELECT doc_id FROM docs")}
    todo = []
    for source, ref, text, ts, meta in iter_l0_docs(only_source):
        for i, chunk in enumerate(chunk_text(text)):
            did = doc_id(source, ref, i)
            if did not in existing:
                todo.append((did, source, ref, chunk, ts, json.dumps(meta or {}, ensure_ascii=False)))
    if not todo:
        print("[build] 无新文档")
        return
    print(f"[build] 新文档 {len(todo)} 块，嵌入中（bge-m3 Q8_0/llama.cpp）…")
    done = 0
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        vecs = emb.embed([t[3] for t in part])
        with db:
            for (did, source, ref, text, ts, meta), vec in zip(part, vecs):
                db.execute("INSERT OR REPLACE INTO docs(doc_id,source,ref,text,ts,meta) VALUES(?,?,?,?,?,?)",
                           (did, source, ref, text, ts, meta))
                db.execute("INSERT OR REPLACE INTO vec_docs(doc_id, embedding) VALUES(?,?)",
                           (did, json.dumps(vec)))
                db.execute("INSERT OR REPLACE INTO fts_docs(doc_id, text) VALUES(?,?)",
                           (did, text))
        done += len(part)
        print(f"  {done}/{len(todo)}")
    print(f"[build] 完成，索引总量 {db.execute('SELECT COUNT(*) FROM docs').fetchone()[0]} 块")


def _load_important_facts(limit: int = 5) -> list[str]:
    """L3 core.md 中 [x/high]/[x/medium] 事实（夜班 35B 判定重要）→ 常驻加权查询。
    实现「AI 重要的事自动加权」：这些事实在每次检索时作为额外查询，命中文档权重 ×1.5。"""
    l3 = os.path.join(REPO, "memory", "L3_core", "core.md")
    if not os.path.exists(l3):
        return []
    facts = []
    for line in open(l3, encoding="utf-8"):
        line = line.strip()
        if line.startswith("- [") and ("/high]" in line or "/medium]" in line):
            text = line.split("] ", 1)[-1].strip()
            if len(text) >= 8:
                facts.append(text[:200])
        if len(facts) >= limit:
            break
    return facts


def decay(stale_days: int = 90, revive_days: int = 30) -> None:
    """分层遗忘（2026-08-21）：L2 语义层超 stale_days 未命中 → 标陈旧（检索 ×0.3 降权）；
    近 revive_days 有命中 → 复活。L0 永存不删（append-only 档案），L3 走人审提案治理。"""
    db = get_db()
    now = time.time()
    with db:
        db.execute(
            "UPDATE docs SET stale=1 WHERE stale=0 AND"
            " ((last_hit IS NULL AND ts < ?) OR (last_hit IS NOT NULL AND last_hit < ?))",
            (now - stale_days * 86400, now - stale_days * 86400))
        db.execute("UPDATE docs SET stale=0 WHERE stale=1 AND last_hit > ?",
                   (now - revive_days * 86400,))
    n_stale = db.execute("SELECT COUNT(*) FROM docs WHERE stale=1").fetchone()[0]
    print(f"[decay] L2 遗忘扫描完成：陈旧 {n_stale} 条（{stale_days} 天未命中降权，{revive_days} 天内命中复活）")


def search(query: str, k: int = 8, rrf_k: int = 60) -> list[dict]:
    """混合检索：向量 ANN + BM25 + RRF 融合。
    2026-08-21 增强：① 时间权重（1/(1+age/30)，新记忆占优）② L3 high/medium 事实常驻
    加权查询（AI 判定重要 ×1.5）③ 陈旧降权（stale ×0.3）④ 命中即强化（hits/last_hit）。"""
    emb = Embedder()
    db = get_db()
    now = time.time()
    qv = json.dumps(emb.embed([query])[0])
    vec_rows = db.execute(
        "SELECT doc_id, distance FROM vec_docs WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (qv, k * 3)).fetchall()
    # 主查询 FTS5 转义（2026-08-21 修）：查询含 -/数字/空格等特殊字符会被 FTS5 当语法
    # （如「DS-160」→ column 160）→ 整个检索异常。拆词 + 双引号，保证不炸。
    fts_query = " OR ".join(f'"{t}"' for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9\-]{2,}", query)) or f'"{query}"'
    fts_rows = db.execute(
        "SELECT doc_id, bm25(fts_docs) AS score FROM fts_docs WHERE fts_docs MATCH ? ORDER BY score LIMIT ?",
        (fts_query, k * 3)).fetchall()
    scores: dict[str, float] = {}
    for rank, (did, _d) in enumerate(vec_rows):
        scores[did] = scores.get(did, 0) + 1.0 / (rrf_k + rank + 1)
    for rank, (did, _s) in enumerate(fts_rows):
        scores[did] = scores.get(did, 0) + 1.0 / (rrf_k + rank + 1)
    # 重要加权：L3 high/medium 事实作为常驻查询（AI 重要的事自动加权）
    boost = _load_important_facts()
    if boost:
        for fact in boost:
            bv = json.dumps(emb.embed([fact])[0])
            bvec = db.execute(
                "SELECT doc_id FROM vec_docs WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (bv, k)).fetchall()
            # FTS5 查询转义：事实拆词 + 双引号（避免 DS-160 之类被当列名）
            tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9\-]{2,}", fact)[:8]
            bfts = db.execute(
                "SELECT doc_id FROM fts_docs WHERE fts_docs MATCH ? ORDER BY bm25(fts_docs) LIMIT ?",
                (" OR ".join(f'"{t}"' for t in tokens) if tokens else fact, k)).fetchall()
            for rank, (did,) in enumerate(bvec):
                scores[did] = scores.get(did, 0) + 1.5 / (rrf_k + rank + 1)
            for rank, (did,) in enumerate(bfts):
                scores[did] = scores.get(did, 0) + 1.5 / (rrf_k + rank + 1)
    # 计算最终分：RRF × 时间权重 × 陈旧降权，排序取 top-k
    ranked = []
    for did, sc in scores.items():
        r = db.execute("SELECT source, ref, text, ts, stale FROM docs WHERE doc_id=?", (did,)).fetchone()
        if not r:
            continue
        age_days = (now - (r[3] or now)) / 86400.0
        time_w = 1.0 / (1.0 + age_days / 30.0)     # 时间权重：30 天半衰（新记忆占优）
        stale_w = 0.3 if r[4] else 1.0              # 陈旧降权（L2 遗忘）
        ranked.append((did, sc * time_w * stale_w, r))
    ranked.sort(key=lambda x: -x[1])
    out = []
    for did, final, r in ranked[:k]:
        db.execute("UPDATE docs SET hits=hits+1, last_hit=? WHERE doc_id=?", (now, did))  # 强化
        out.append({"rrf": round(final, 4), "source": r[0], "ref": r[1],
                    "text": r[2][:300], "ts": r[3]})
    db.commit()
    return out


def stats() -> None:
    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    by_src = db.execute("SELECT source, COUNT(*) FROM docs GROUP BY source ORDER BY 2 DESC").fetchall()
    print(f"[stats] 共 {total} 块")
    for s, c in by_src:
        print(f"  {s}: {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--source", default=None)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=8)
    d = sub.add_parser("decay"); d.add_argument("--stale-days", type=int, default=90)
    sub.add_parser("stats")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.source)
    elif args.cmd == "search":
        for hit in search(args.query, args.k):
            print(f"[{hit['rrf']}] {hit['source']} | {hit['text'][:150].replace(chr(10), ' / ')}")
            print("---")
    elif args.cmd == "decay":
        decay(args.stale_days)
    else:
        stats()
