#!/usr/bin/env python3
"""L2 知识图谱层（2026-08-21 第一步：实体-关系三元组抽取 + 落库，检索跳转第二步再做）。

设计（用户 2026-08-21 拍板「按设计来」，分两步）：
  第一步（本模块）：夜班 35B 巩固后顺带把增量 L2 文档抽成三元组
    (subject, relation, object)，存 l2.db 图谱三表；图谱随每晚夜班自动积累。
  第二步（后续开发）：检索跳转 —— 查询命中 doc 后沿图谱 1-2 跳召回关联记忆，
    接进 _l2_search_context（本模块预留 graph_hops() 接口，图谱积累后再启用）。

存储（并入 l2.db，与向量/BM25 同库）：
  entities(name TEXT PRIMARY KEY)                  -- 实体（35B 归一化到规范名）
  relations(id, s, r, o, ts, doc_id)               -- 三元组 + 来源 + 时间
  doc_entities(doc_id, entity)                     -- 文档 ↔ 实体 连接（跳转用）

幂等：memory/L1_working/kg_state.json 记已抽取的 doc_id；L2 新块增量处理。

用法：
  python3 knowledge_graph.py extract --port 8200 --model <id>   # 夜班 seg3 调
  python3 knowledge_graph.py stats                               # 图谱规模
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l2_semantic import get_db  # noqa: E402  (图谱并入 l2.db)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_STATE = os.path.join(REPO, "memory", "L1_working", "kg_state.json")

# 测试沙盒（2026-08-22）：水位与 l2.db（经 l2_semantic）跟随 AIAGENT_SANDBOX。
SANDBOX = os.environ.get("AIAGENT_SANDBOX", "")
if SANDBOX:
    KG_STATE = os.path.join(SANDBOX, "memory", "L1_working", "kg_state.json")

TZ_CN = timezone(timedelta(hours=8))

GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities(
  name TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS relations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  s TEXT NOT NULL, r TEXT NOT NULL, o TEXT NOT NULL,
  ts REAL, doc_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_rel_s ON relations(s);
CREATE INDEX IF NOT EXISTS idx_rel_o ON relations(o);
CREATE TABLE IF NOT EXISTS doc_entities(
  doc_id TEXT NOT NULL, entity TEXT NOT NULL,
  PRIMARY KEY(doc_id, entity)
);
"""

EXTRACT_SYSTEM = (
    "你是记忆图谱抽取器。从给定文本中抽取明确的事实关系三元组 (subject, relation, object)。\n"
    "规则：1) 抽取文本中明确出现的事实关系，包括汇总/摘要中的明确信息"
    "（谁、何时、何事、截止、归属、地址等——如\"6/9 数学分级\"\"课表由学院安排\"\"房间分配为 1335c\""
    "都算明确事实），但仍是文本中出现的，禁止编造/猜测；2) 同一实体的不同叫法归一到规范名"
    "（昵称/简称→全名，如英文名→中文名）；3) 关系用简洁中文动词或名词；"
    "4) 人名、学校/机构名、事件名、待办事项都可以是实体；5) 只输出 JSON 数组。\n"
    "JSON 数组的每个元素是 {\"s\": 主语实体名, \"r\": 关系, \"o\": 宾语实体名}，"
    "s/r/o 必须是文本中出现的**真实实体名**，禁止使用占位符、示例名或自造词。"
    "不要输出任何其他文字。"
)


def now_iso() -> str:
    return datetime.now(TZ_CN).isoformat(timespec="seconds")


def ensure_graph_schema() -> None:
    db = get_db()
    db.executescript(GRAPH_SCHEMA)
    db.commit()


def load_kg_state() -> dict:
    if os.path.exists(KG_STATE):
        try:
            return json.load(open(KG_STATE, encoding="utf-8"))
        except Exception:
            pass
    return {"extracted_doc_ids": []}


def save_kg_state(st: dict) -> None:
    os.makedirs(os.path.dirname(KG_STATE), exist_ok=True)
    tmp = KG_STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, KG_STATE)


def _pending_docs(db: sqlite3.Connection, done: set[str], batch_chars: int = 3000) -> list[tuple[str, str]]:
    """L2 中未抽取的文档（增量），按【源优先级】排序（2026-08-23 修：原 ts 顺序导致
    每晚先抽到微信闲聊块 → 实体迟迟不出现；改为 school/email/inbox/doc 高价值源优先，
    wechat/chat 最后——图谱在事实数据上才有产出）。合并到约 batch_chars 一组。"""
    rows = db.execute("""
        SELECT doc_id, text FROM docs
        ORDER BY CASE source
          WHEN 'email' THEN 0 WHEN 'doc:file' THEN 1 WHEN 'chat' THEN 2
          WHEN 'exchange:inbox' THEN 3 WHEN 'school' THEN 4 WHEN 'wechat' THEN 5
          ELSE 6 END, ts ASC
    """).fetchall()
    pending = [(did, text) for did, text in rows if did not in done and text.strip()]
    batches: list[tuple[str, str]] = []
    cur_docs: list[str] = []
    cur_chars = 0
    for did, text in pending:
        cur_docs.append(did)
        cur_chars += len(text)
        if cur_chars >= batch_chars or len(cur_docs) >= 8:
            batches.append((" | ".join(cur_docs), "\n---\n".join(
                db.execute("SELECT text FROM docs WHERE doc_id=?", (d,)).fetchone()[0][:600]
                for d in cur_docs)))
            cur_docs, cur_chars = [], 0
    if cur_docs:
        batches.append((" | ".join(cur_docs), "\n---\n".join(
            db.execute("SELECT text FROM docs WHERE doc_id=?", (d,)).fetchone()[0][:600]
            for d in cur_docs)))
    return batches


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _extract_triples(port: int, model_id: str, text: str, max_tokens: int = 3000) -> list[dict]:
    """调本地模型抽三元组（复用巩固的 :8200 35B / :8100 27B）。返回解析后的三元组列表。"""
    payload = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"待分析文本：\n{text[:4000]}"},
        ],
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    content = _strip_think(resp["choices"][0]["message"].get("content") or "")
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for t in arr:
        if isinstance(t, dict) and t.get("s") and t.get("r") and t.get("o"):
            s = str(t["s"]).strip()[:80]
            r = str(t["r"]).strip()[:40]
            o = str(t["o"]).strip()[:80]
            if s and r and o:
                out.append({"s": s, "r": r, "o": o})
    return out


def _store_triples(db: sqlite3.Connection, doc_ids: list[str], triples: list[dict]) -> int:
    """三元组落库（entities/relations/doc_entities）。按 (s,r,o) 去重。返回新增关系数。"""
    n = 0
    ts = time.time()
    with db:
        for t in triples:
            s, r, o = t["s"], t["r"], t["o"]
            db.execute("INSERT OR IGNORE INTO entities(name) VALUES(?)", (s,))
            db.execute("INSERT OR IGNORE INTO entities(name) VALUES(?)", (o,))
            exists = db.execute(
                "SELECT 1 FROM relations WHERE s=? AND r=? AND o=? LIMIT 1",
                (s, r, o)).fetchone()
            if not exists:
                db.execute("INSERT INTO relations(s,r,o,ts,doc_id) VALUES(?,?,?,?,?)",
                           (s, r, o, ts, ",".join(doc_ids)[:500]))
                n += 1
            for e in (s, o):
                for did in doc_ids:
                    db.execute("INSERT OR IGNORE INTO doc_entities(doc_id, entity) VALUES(?,?)",
                               (did, e))
    return n


def extract_pending(port: int, model_id: str, max_tokens: int = 3000) -> int:
    """扫 L2 新文档 → 分批调模型抽三元组 → 落库。返回新增关系数（幂等，失败跳过）。"""
    ensure_graph_schema()
    db = get_db()
    st = load_kg_state()
    done = set(st.get("extracted_doc_ids", []))
    batches = _pending_docs(db, done)
    if not batches:
        return 0
    total = 0
    for key, text in batches:
        doc_ids = [d for d in key.split(" | ") if d]
        try:
            triples = _extract_triples(port, model_id, text, max_tokens)
        except Exception as e:
            print(f"[kg] 抽取失败（跳过本批）: {e}")
            continue
        if triples:
            total += _store_triples(db, doc_ids, triples)
        done.update(doc_ids)
        save_kg_state(st | {"extracted_doc_ids": sorted(done)})
    print(f"[kg] 本次新增关系 {total} 条，已处理文档 {len(done)}")
    return total


def graph_hops(doc_ids: list[str], depth: int = 1, limit: int = 10) -> list[str]:
    """【第二步：检索跳转预留接口】给命中文档，沿图谱跳 depth 跳，返回关联 doc_id。
    图谱积累（第一步跑一段时间）后接进 _l2_search_context 启用。"""
    if depth < 1 or not doc_ids:
        return []
    db = get_db()
    cur = set(doc_ids)
    seen = set(doc_ids)
    for _ in range(depth):
        ents = db.execute(
            "SELECT DISTINCT entity FROM doc_entities WHERE doc_id IN (%s)"
            % ",".join("?" * len(cur)), tuple(cur)).fetchall()
        if not ents:
            break
        enames = [e[0] for e in ents]
        rel_rows = db.execute(
            "SELECT s, o FROM relations WHERE s IN (%s) OR o IN (%s)"
            % (",".join("?" * len(enames)), ",".join("?" * len(enames))),
            tuple(enames) + tuple(enames)).fetchall()
        hops: set[str] = set()
        for s, o in rel_rows:
            for e in enames:
                if s == e:
                    hops.add(o)
                if o == e:
                    hops.add(s)
        if not hops:
            break
        next_docs = db.execute(
            "SELECT DISTINCT doc_id FROM doc_entities WHERE entity IN (%s) AND doc_id NOT IN (%s)"
            % (",".join("?" * len(hops)), ",".join("?" * len(seen))),
            tuple(hops) + tuple(seen)).fetchall()
        new_docs = [d[0] for d in next_docs][:limit]
        if not new_docs:
            break
        cur = set(new_docs)
        seen.update(new_docs)
    return [d for d in seen if d not in doc_ids]


def stats() -> None:
    ensure_graph_schema()
    db = get_db()
    e = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    r = db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    d = db.execute("SELECT COUNT(*) FROM doc_entities").fetchone()[0]
    top = db.execute(
        "SELECT entity, COUNT(*) c FROM doc_entities GROUP BY entity ORDER BY c DESC LIMIT 8").fetchall()
    print(f"[kg] 实体 {e} / 关系 {r} / 文档连接 {d}")
    print("Top 实体:", ", ".join(f"{x[0]}({x[1]})" for x in top))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "extract":
        port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8200
        model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else ""
        if not model:
            print("用法: knowledge_graph.py extract --port <port> --model <model_id>")
            sys.exit(2)
        n = extract_pending(port, model)
        print(f"[kg] 完成，新增 {n} 条")
    elif cmd == "stats":
        stats()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
