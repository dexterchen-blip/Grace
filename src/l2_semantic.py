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

# 2026-09-04 boost 相关性门槛：cos(fact, query) < BOOST_COS_MIN → 不 boost；
# ≥ MIN 后线性升至 (MIN+RANGE) 满权重。标定：相关 0.61-0.73，不相关 0.27-0.44。
BOOST_COS_MIN = 0.45
BOOST_COS_RANGE = 0.25


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


_URGENT = re.compile(r"immediately|urgent|asap|right away|尽快|紧急", re.IGNORECASE)
_ISO_DATE = re.compile(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}")


def _parse_deadline(text: str) -> float | None:
    """提取文本中第一个未来日期的时间戳（ISO 格式），无则 None。"""
    for m in _ISO_DATE.finditer(text):
        try:
            t = time.mktime(time.strptime(m.group(0).replace("/", "-"), "%Y-%m-%d"))
            return t
        except ValueError:
            continue
    return None


def _time_sort_facts(facts: list[str]) -> list[str]:
    """时间排序（2026-08-26 沙盒验证）：紧急 > 未来日期(近→远) > 无日期 > 已过期。
    不删除事实，只调 boost 优先级（即将发生/紧急的优先进加权查询）。"""
    now = time.time()
    ranked = []
    for f in facts:
        dl = _parse_deadline(f)
        if _URGENT.search(f):
            rank, key = 0, 0.0
        elif dl and dl >= now:
            rank, key = 1, dl - now
        elif dl:
            rank, key = 3, now - dl
        else:
            rank, key = 2, 0.0
        ranked.append((rank, key, f))
    return [f for _, _, f in sorted(ranked, key=lambda x: (x[0], x[1]))]


def _load_important_facts(limit: int = 5) -> list[str]:
    """L3 core.md 中 [x/high]/[x/medium] 事实（夜班 35B 判定重要）→ 常驻加权查询。
    2026-08-26：加时间排序（紧急/即将发生优先，已过期垫底）。"""
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
    return _time_sort_facts(facts)[:limit]


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


def search(query: str, k: int = 8, rrf_k: int = 60, pool: int = 0) -> list[dict]:
    """混合检索：向量 ANN + BM25 + RRF 融合。
    2026-08-21 增强：① 时间权重（1/(1+age/30)，新记忆占优）② L3 high/medium 事实常驻
    加权查询（AI 判定重要 ×0.8）③ 陈旧降权（stale ×0.3）④ 命中即强化（hits/last_hit）。
    2026-08-26 增强（沙盒验证）：⑤ 候选池 pool（默认 max(k*3,60)，混合大库需放大）
    ⑥ recency：24h 内新内容 ×2.0 ⑦ imminence：文本含 24h 内未来日期 ×1.5。"""
    emb = Embedder()
    db = get_db()
    now = time.time()
    if not pool:
        pool = max(k * 3, 60)
    q_vec = emb.embed([query])[0]                     # 2026-09-04: 留原始向量算 boost 相关度
    qv = json.dumps(q_vec)
    vec_rows = db.execute(
        "SELECT doc_id, distance FROM vec_docs WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (qv, pool)).fetchall()
    # 主查询 FTS5 转义（2026-08-21 修）：查询含 -/数字/空格等特殊字符会被 FTS5 当语法
    # （如「DS-160」→ column 160）→ 整个检索异常。拆词 + 双引号，保证不炸。
    # 2026-08-26 修：英文停用词（how/is/my/to 等高频词）不进 OR，否则 FTS 命中大量
    # 无关会话、RRF 被 FTS 主导、向量腿被挤掉（LongMemEval 全英文暴露）。
    _EN_STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "with", "and", "or",
                "is", "are", "was", "were", "be", "been", "i", "you", "your", "my",
                "me", "it", "this", "that", "what", "how", "can", "do", "did", "not",
                "would", "should", "could", "have", "has", "had", "from", "at", "by",
                "about", "up", "out", "into", "over", "after"}
    fts_tokens = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9\-]{2,}", query)
                  if t.lower() not in _EN_STOP]
    fts_query = " OR ".join(f'"{t}"' for t in fts_tokens) or f'"{query}"'
    fts_rows = db.execute(
        "SELECT doc_id, bm25(fts_docs) AS score FROM fts_docs WHERE fts_docs MATCH ? ORDER BY score LIMIT ?",
        (fts_query, pool)).fetchall()
    scores: dict[str, float] = {}
    for rank, (did, _d) in enumerate(vec_rows):
        scores[did] = scores.get(did, 0) + 1.0 / (rrf_k + rank + 1)
    # FTS 腿权重 ×0.5（2026-08-26 修）：短英文问题 OR 匹配的 BM25 噪声会主导 RRF、
    # 挤掉向量腿正确命中（LongMemEval 全英文暴露）。向量腿为主、FTS 为补充。
    for rank, (did, _s) in enumerate(fts_rows):
        scores[did] = scores.get(did, 0) + 0.5 / (rrf_k + rank + 1)
    # 重要加权：L3 high/medium 事实作为常驻查询（时间排序后，权重 ×0.8 防霸榜）。
    # 2026-09-04 修（护照/签证主题查询 top-k 被挤光的根因）：boost 原来无条件给
    # 每个 fact 的近邻文档加票、与当前查询无关 → 5 个 fact 的票(~0.8·1.5/61≈0.02/篇)
    # 把纯查询命中（vec rank1≈1/61，还要乘 time_w≈0.6）整体挤出 top-k。现在 boost
    # 只对与查询语义相关（cos≥0.45）的 fact 启用，幅度按 (cos-0.45)/0.25 线性升到满权重：
    #   - 问护照/签证（与全部 fact cos≤0.44）→ boost 全关，主题文档正常回来
    #   - 问 ELPE/疫苗（cos 0.61-0.73）→ 保留同义改写召回（LongMemEval 场景不回退）
    boost = _load_important_facts()
    if boost:
        for fact in boost:
            bv_list = emb.embed([fact])[0]
            # bge-m3 向量已 L2 归一，点积即余弦
            cos_qf = sum(a * b for a, b in zip(q_vec, bv_list))
            factor = min(1.0, max(0.0, (cos_qf - BOOST_COS_MIN) / BOOST_COS_RANGE))
            if factor <= 0:
                continue
            bv = json.dumps(bv_list)
            bvec = db.execute(
                "SELECT doc_id FROM vec_docs WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (bv, k)).fetchall()
            # FTS5 查询转义：事实拆词 + 双引号（避免 DS-160 之类被当列名）
            tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9\-]{2,}", fact)[:8]
            bfts = db.execute(
                "SELECT doc_id FROM fts_docs WHERE fts_docs MATCH ? ORDER BY bm25(fts_docs) LIMIT ?",
                (" OR ".join(f'"{t}"' for t in tokens) if tokens else fact, k)).fetchall()
            for rank, (did,) in enumerate(bvec):
                scores[did] = scores.get(did, 0) + 0.8 * 1.5 * factor / (rrf_k + rank + 1)
            for rank, (did,) in enumerate(bfts):
                scores[did] = scores.get(did, 0) + 0.8 * 1.5 * factor / (rrf_k + rank + 1)
    # 计算最终分：RRF × 时间权重 × 陈旧降权 × recency(24h) × imminence(24h 内到期)，排序取 top-k
    ranked = []
    for did, sc in scores.items():
        r = db.execute("SELECT source, ref, text, ts, stale FROM docs WHERE doc_id=?", (did,)).fetchone()
        if not r:
            continue
        age_days = (now - (r[3] or now)) / 86400.0
        time_w = 1.0 / (1.0 + age_days / 30.0)     # 时间权重：30 天半衰（新记忆占优）
        stale_w = 0.3 if r[4] else 1.0              # 陈旧降权（L2 遗忘）
        recency_w = 2.0 if (r[3] and (now - r[3]) < 86400) else 1.0   # 24h 新内容
        dl = _parse_deadline(r[2] or "")
        imminence_w = 1.5 if (dl and 0 <= dl - now <= 86400) else 1.0  # 24h 内到期
        ranked.append((did, sc * time_w * stale_w * recency_w * imminence_w, r))
    ranked.sort(key=lambda x: -x[1])
    out = []
    for did, final, r in ranked[:k]:
        db.execute("UPDATE docs SET hits=hits+1, last_hit=? WHERE doc_id=?", (now, did))  # 强化
        out.append({"doc_id": did, "rrf": round(final, 4), "source": r[0], "ref": r[1],
                    "text": r[2][:300], "ts": r[3]})  # doc_id: 2026-09-04 图谱跳转用
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
    s.add_argument("--json", action="store_true",
                   help="输出 JSON {hits,related}（hits 含 doc_id，供 dashboard 图谱跳转）")
    s.add_argument("--graph", type=int, default=0,
                   help=">0 时沿知识图谱跳 N 跳召回关联文档（lazy import knowledge_graph，失败只丢 related）")
    d = sub.add_parser("decay"); d.add_argument("--stale-days", type=int, default=90)
    sub.add_parser("stats")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.source)
    elif args.cmd == "search":
        hits = search(args.query, args.k)
        # 图谱跳转（2026-09-04 第二步）：命中 doc → graph_hops → 关联文档文本。
        # best-effort：任何异常只丢 related，检索本身照常；知识图谱只读 l2.db。
        related: list[dict] = []
        if args.graph > 0 and hits:
            try:
                if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from knowledge_graph import graph_hops  # 懒加载：默认路径零依赖
                rel_ids = graph_hops([h["doc_id"] for h in hits], depth=args.graph, limit=8)
                if rel_ids:
                    db = get_db()
                    for did in rel_ids:
                        r = db.execute(
                            "SELECT source, ref, text FROM docs WHERE doc_id=?", (did,)).fetchone()
                        if r:
                            related.append({"doc_id": did, "source": r[0], "ref": r[1],
                                            "text": (r[2] or "")[:200]})
            except Exception:
                related = []
        if args.json:
            print(json.dumps({"hits": hits, "related": related}, ensure_ascii=False))
        else:
            for hit in hits:
                print(f"[{hit['rrf']}] {hit['source']} | {hit['text'][:150].replace(chr(10), ' / ')}")
                print("---")
            for d in related:
                print(f"[图谱关联] {d['source']} | {d['text'][:150].replace(chr(10), ' / ')}")
    elif args.cmd == "decay":
        decay(args.stale_days)
    else:
        stats()
