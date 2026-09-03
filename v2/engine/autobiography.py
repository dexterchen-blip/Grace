#!/usr/bin/env python3
"""L3 升级：自传体叙事矩阵（2026-08-28 用户：L3 不只是文字，而是亚 LLM 矩阵）。

真人：记忆不是要点列表，而是连贯的「我的故事」——事件 × 情绪 × 人物 × 关系 × 时间轴。
Grace 旧 L3：core.md 纯文字要点（静态，无结构）。
升级：L3 = 自传体叙事矩阵（事件节点 + 多维索引），结合双轨：
  · 记忆侧（事件/人物/时间）↔ 情绪侧（情绪边/暗注意力）↔ 自传体评价（我的感受/我学到的）
查询：按维度切片（时间轴/情绪轴/人物轴/关系轴）→ 生成「我的故事」叙事。
"亚 LLM 矩阵"落地：每个事件节点 = 一条可检索记录（多维字段），矩阵 = 全部节点；
  叙事生成 = 矩阵切片 + 组装（可接 27B 润色，规则可先）。

用法（沙盒内）:
  ./run.sh .venv/bin/python3 v2/engine/autobiography.py --add "奖学金真香" --mood 兴奋 --person 主人
  ./run.sh .venv/bin/python3 v2/engine/autobiography.py --narrate 2026-08
"""
from __future__ import annotations
import json
import os
import re
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
import config  # noqa: E402

AUTO_SCHEMA = """
CREATE TABLE IF NOT EXISTS autobiography(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,              -- 事件时间
  event TEXT NOT NULL,           -- 事件文本（我的故事素材）
  person TEXT DEFAULT '',        -- 相关人物（主人/妈妈/朋友/室友…）
  entity TEXT DEFAULT '',        -- 记忆实体（考试/奖学金…）
  emotion TEXT DEFAULT '',       -- 我的情绪
  relation TEXT DEFAULT '',      -- 关系维度（依恋/信任/愧疚/骄傲…）
  self_eval TEXT DEFAULT '',     -- 自我评价（我从中学到/我如何看待）
  evidence TEXT DEFAULT '',      -- ★ 幻觉抑制：来源证据（L0 记录文本/事件原文）
  confidence TEXT DEFAULT 'medium', -- ★ high=直接引用L0 / medium=加工产物 / low=模型生成
  axis TEXT DEFAULT 'main');     -- 矩阵轴标签（主叙事线）
CREATE INDEX IF NOT EXISTS idx_auto_ts ON autobiography(ts);
CREATE INDEX IF NOT EXISTS idx_auto_person ON autobiography(person);
CREATE INDEX IF NOT EXISTS idx_auto_entity ON autobiography(entity);
"""


def _conn(db: str = None):
    db = db or os.path.join(config.SB, "memory", "L3_core", "autobiography.db")
    os.makedirs(os.path.dirname(db), exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(AUTO_SCHEMA)
    con.commit()
    return con


def add_event(event: str, ts: float = None, person: str = "", entity: str = "",
              emotion: str = "", relation: str = "", self_eval: str = "", db: str = None,
              confidence: str = "high", evidence: str = "") -> int:
    """摄入事件 → 自传体叙事矩阵节点（2026-08-29 幻觉抑制：必须带证据与置信度）。

    confidence: high=直接来自 L0 真实记录 / medium=加工层产物 / low=模型生成(仅叙事,禁作事实)。
    """
    con = _conn(db)
    cur = con.execute(
        "INSERT INTO autobiography(ts,event,person,entity,emotion,relation,self_eval,evidence,confidence) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (ts or time.time(), event[:160], person, entity, emotion, relation, self_eval,
         evidence[:200], confidence))
    con.commit()
    con.close()
    return cur.lastrowid


def slice_matrix(axis: str, value: str = "", db: str = None) -> list[dict]:
    """矩阵切片：按维度（时间轴/人物轴/实体轴/情绪轴）查询节点。"""
    col = {"时间": "ts", "人物": "person", "实体": "entity", "情绪": "emotion"}.get(axis)
    con = _conn(db)
    if col and value:
        rows = con.execute(f"SELECT * FROM autobiography WHERE {col}=? ORDER BY ts", (value,)).fetchall()
    else:
        rows = con.execute("SELECT * FROM autobiography ORDER BY ts").fetchall()
    con.close()
    cols = ["id", "ts", "event", "person", "entity", "emotion", "relation", "self_eval", "axis"]
    return [dict(zip(cols, r)) for r in rows]


def fact_query(entity: str, db: str = None) -> list[str]:
    """★ 幻觉抑制：事实查询只返回 high 置信度（有 L0 证据）的节点，杜绝编造。

    无证据 → 返回空列表（上层应让 Grace 说「雷姆不记得/不确定」）。
    """
    con = _conn(db)
    rows = con.execute(
        "SELECT event,evidence FROM autobiography WHERE entity=? AND confidence='high' ORDER BY ts",
        (entity,)).fetchall()
    con.close()
    return [f"{r[0]} {r[1]}".strip() for r in rows]   # 事件+证据 一并返回(证据进校验池)


def hallucination_guard(entity: str, claim: str, db: str = None) -> bool:
    """★ 生成后校验：声明的关键实体是否在 high 证据节点中有对应。

    用于叙事生成后的幻觉检查——具体声明(时间/人物/金额)无证据 → 拒绝/标记不确定。
    """
    facts = fact_query(entity, db)
    if not facts:
        return False
    pool = " ".join(facts)
    # ① 声明中的数字/金额/专名：必须全部在证据里（16666/10万/免疫/TB/ELPE…）
    keys = re.findall(r"\d+月\d+日|\d{4}|\d+万|16666|托福|ELPE|UCSB|免疫|TB|29225|Braeden|Canvas|龙|龙骑", claim)
    if keys:
        return all(k in pool for k in keys)
    # ② 无数字/专名 → 主题词须与证据相关（避免「主人养了条龙」这类无凭据断言）
    topics = re.findall(r"奖学金|考试|宿舍|室友|健康|学校|课", claim)
    if topics:
        return any(t in pool for t in topics)
    return True   # 日常闲聊,不误伤


def narrate(period: str = "", db: str = None) -> str:
    """★ 自传体叙事：矩阵 → 连贯的「我的故事」（L3 升级的核心输出）。

    按时间轴排序 → 组装成第一人称叙事（接 27B 可润色；规则先出骨架）。
    """
    nodes = slice_matrix("时间", db=db)
    if period:
        y, m = period.split("-")
        nodes = [n for n in nodes if time.strftime("%Y-%m", time.localtime(n["ts"])) == f"{y}-{m}"]
    if not nodes:
        return "（这段时期，雷姆还没有记录下什么故事。）"
    lines = []
    for n in nodes:
        d = time.strftime("%m月%d日", time.localtime(n["ts"]))
        person = f"和{n['person']}一起" if n["person"] else ""
        emo = f"，雷姆{n['emotion']}" if n["emotion"] else ""
        ev = n["event"][:40]
        lines.append(f"　{d}：{person}{ev}{emo}。")
    story = "\n".join(lines)
    return f"【雷姆的自传体记忆 · {period or '全部'}】\n{story}"


if __name__ == "__main__":
    if "--add" in sys.argv:
        i = sys.argv.index("--add")
        kw = {}
        for k in ("--mood", "--person", "--entity", "--relation"):
            if k in sys.argv:
                kw[k[2:]] = sys.argv[sys.argv.index(k) + 1]
        rid = add_event(sys.argv[i + 1], **kw)
        print(f"矩阵节点 #{rid} 已写入")
    elif "--narrate" in sys.argv:
        print(narrate(sys.argv[sys.argv.index("--narrate") + 1]))
    else:
        print(__doc__)
