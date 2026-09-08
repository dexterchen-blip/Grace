#!/usr/bin/env python3
"""情绪图谱 + 双图谱耦合摄入 —— M7 增强（2026-08-27 用户洞察）。

背景（用户诊断）：
  正式系统 L2 图谱是「事实侧」（实体/关系），情绪在提炼时被丢掉 → wechat 摘要去情绪化。
  Grace V2 情绪系统直接建在记忆系统上，却没有自己的图谱结构（只有 mood_states 时序表）。

设计：**双图谱耦合**
  摄入事件时并行生成两张图谱，共享 event_id/实体：
    ① 记忆图谱（事实侧）：事件 → 实体/关系（规则主题抽取；V2.1 单模型 27B，不预留 35B 边缘抽取）
    ② 情绪图谱（感受侧）：实体 → 情绪边（什么 → 什么感受，强度 + 时间）
  耦合：同一事件的两个侧面通过 event_id + entity 连接；查询时"实体"同时返回
        记忆事实 + 情绪历史（考试 → 事实 + [焦虑0.7@8-27, 专注0.6@8-20]）。

用法（沙盒内）:
  ./run.sh .venv/bin/python3 v2/engine/mood_graph.py --add "考砸了" --mood 低落 --int 0.7 --entity 考试
  ./run.sh .venv/bin/python3 v2/engine/mood_graph.py --query 考试
"""
from __future__ import annotations
import json
import os
import re
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                        # v2/engine/（同目录模块）
import config  # noqa: E402
from attention import _sentiment_of  # noqa: E402（复用事件情绪初判）

# 情绪图谱表（独立于 mood_states 时序；图谱式：实体 ↔ 情绪边）
MOOD_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS mood_graph(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity TEXT NOT NULL,          -- 事件/对象实体（考试/奖学金/妈妈）
  mood_label TEXT NOT NULL,      -- 情绪标签（兴奋/低落/焦虑/专注/平静）
  intensity REAL NOT NULL,       -- 情绪强度 0-1
  ts REAL NOT NULL,              -- 事件时间
  event_id TEXT DEFAULT '',      -- 耦合：与记忆图谱共享的事件 id
  trigger TEXT DEFAULT '',       -- 触发文本摘要
  edge_type TEXT DEFAULT 'emotion',  -- ★ 2026-08-28：emotion(半暗,影响语气) / hidden(全暗,暗注意力)
  uncertainty REAL DEFAULT 0.2,    -- ★ 2026-08-31：再巩固标记(反馈 0.9/普通 0.2)
  source TEXT DEFAULT 'dual_graph');
CREATE INDEX IF NOT EXISTS idx_mg_entity ON mood_graph(entity);
CREATE INDEX IF NOT EXISTS idx_mg_event ON mood_graph(event_id);
CREATE INDEX IF NOT EXISTS idx_mg_type ON mood_graph(edge_type);
"""

# ★ 暗注意力 = 真实思考（2026-09-03 架构统一，用户：思考过程本身就是暗注意力）
# 废除规则模板推导（_HIDDEN_RULES 4 句「雷姆担心但不会说」= 假潜台词 + V2 轮模板污染源 +
# 分布倒置源头：hidden 727 条兴奋 562/低落 165、0% 平静 vs emotion 98.3% 平静）。
# 真实暗注意力唯一来源 = cog 认知重构（27B 自由内心独白：联想/读心/潜台词/顾虑）。
# 写边 = 把 cog 句写入 hidden 边（全暗：只注入决策上下文,绝不直接输出）


def add_hidden_text(entity: str, hidden_text: str, ts: float = None,
                    trigger: str = "", db: str = None) -> int:
    """写真实暗注意力边（全暗）。hidden_text = cog 重构的真实思考句（非规则模板）。"""
    from engine.sentiment import assess as _assess
    _a = _assess(hidden_text)
    mood_label = mood_label_of(_a["valence"], _a["arousal"] + 0.3)
    con = _conn(db)
    cur = con.execute(
        "INSERT INTO mood_graph(entity,mood_label,intensity,ts,event_id,trigger,edge_type,source) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (entity, mood_label, round(max(0.1, _a["arousal"]), 3), ts or time.time(), "",
         (trigger or hidden_text)[:120], "hidden", f"hidden:{hidden_text[:140]}"))
    con.commit()
    con.close()
    return cur.lastrowid


def query_hidden(entity: str = "", event_id: str = "", k: int = 3, db: str = None) -> list[str]:
    """★ 暗注意力查询：该实体/事件的潜台词（全暗,注入生成上下文但不输出）。

    与情绪边同一套机制：按时间衰减 + 强度排序。
    """
    import math
    con = _conn(db)
    if entity:
        rows = con.execute(
            "SELECT source,intensity,ts FROM mood_graph WHERE edge_type='hidden' "
            "AND entity=? ORDER BY ts DESC LIMIT 20", (entity,)).fetchall()
    else:
        rows = con.execute(
            "SELECT source,intensity,ts FROM mood_graph WHERE edge_type='hidden' "
            "AND event_id=? ORDER BY ts DESC LIMIT 20", (event_id,)).fetchall()
    con.close()
    now = time.time()
    out = []
    for src, intensity, ts in rows:
        age = max(0.0, (now - ts) / 86400.0) if ts else 0.0
        w = intensity * math.exp(-age / 60.0)   # 暗注意力衰减慢(τ=60天,底色持久)
        out.append((src.replace("hidden:", ""), w))
    out.sort(key=lambda x: -x[1])
    return [t for t, _ in out[:k]]

# 实体抽取（记忆侧主题 → 图谱实体；可与 attention.topic_of 对齐）
_ENTITY_MAP = [
    ("考试", r"考|期中|期末|测验|成绩|GPA|分数"),
    ("作业", r"作业|截止|ddl|due|project"),
    ("选课", r"选课|课程"),
    ("宿舍", r"宿舍|室友"),
    ("家人", r"想家|爸妈|妈妈|父亲|回家"),
    ("朋友", r"朋友|同学"),
    ("游玩", r"海边|旅行|社团|摄影|拉面|跨年|假期"),
    ("健康", r"累|失眠|生病|药|感冒"),
    ("奖学金", r"奖学金|补助|助学金"),
    ("邮件", r"邮件|邮箱|inbox"),
]
_MOOD_LABELS = ("兴奋", "轻微兴奋", "平静", "专注", "低落", "焦虑")


def entity_of(text: str) -> str:
    """记忆侧实体抽取（事件主题 → 实体名）。"""
    for ent, pat in _ENTITY_MAP:
        if re.search(pat, text):
            return ent
    return "日常"


def mood_label_of(sentiment: float, intensity: float) -> str:
    """情绪侧标签（sentiment 方向 + 强度 → 词表）。"""
    if sentiment > 0.3:
        return "兴奋" if intensity > 0.7 else "轻微兴奋"
    if sentiment < -0.3:
        return "低落" if intensity > 0.5 else "焦虑"
    return "平静"


def _conn(db: str = None):
    db = db or config.L2_DB
    os.makedirs(os.path.dirname(db), exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(MOOD_GRAPH_SCHEMA)
    con.commit()
    return con


def add_mood_edge(entity: str, mood_label: str, intensity: float, ts: float = None,
                  event_id: str = "", trigger: str = "", db: str = None) -> int:
    """写情绪图谱边（实体 ↔ 情绪，带时间与事件耦合 id）。"""
    ts = ts or time.time()
    con = _conn(db)
    cur = con.execute(
        "INSERT INTO mood_graph(entity,mood_label,intensity,ts,event_id,trigger,source) "
        "VALUES(?,?,?,?,?,?,?)",
        (entity, mood_label, round(intensity, 3), ts, event_id, trigger[:120], "dual_graph"))
    con.commit()
    con.close()
    return cur.lastrowid


def query_mood_history(entity: str, k: int = 5, db: str = None) -> list[dict]:
    """查实体的情绪历史（考试 → [低落0.7@8-27, 专注0.6@8-20]）。"""
    con = _conn(db)
    rows = con.execute(
        "SELECT entity,mood_label,intensity,ts,event_id,trigger FROM mood_graph "
        "WHERE entity=? ORDER BY ts DESC LIMIT ?", (entity, k)).fetchall()
    con.close()
    return [{"entity": r[0], "mood_label": r[1], "intensity": r[2], "ts": r[3],
             "event_id": r[4], "trigger": r[5]} for r in rows]


def dual_graph_ingest(event_text: str, event_id: str = "", db: str = None,
                      sentiment: float | None = None, ts: float = None) -> dict:
    """★ 双图谱耦合摄入：同一事件并行生成 记忆图谱 + 情绪图谱。

    ① 记忆侧：实体抽取（entity_of）→ 写记忆图谱边（本实现落 mood_graph 的实体边，
       规则主题抽取；V2.1 不预留 35B（2026-08-29））
    ② 情绪侧：sentiment → 情绪标签 → 写情绪图谱边（add_mood_edge）
    ★2026-09-03 架构统一（用户：思考=暗注意力）：暗注意力不再在此由情绪规则推导
       （废 _HIDDEN_RULES 4 模板：假潜台词/分布倒置源头）。真实暗注意力 = cog 认知重构
       产物，由 add_hidden_text 单独写入（stress_engine 当日 cog 归档段）。
    """
    ts = ts or time.time()
    entity = entity_of(event_text)
    # ★2026-09-03 Phase 0: intensity 由 arousal 驱动（Russell 双轴）——原 abs(sentiment) 把
    #   高唤醒中性事件(面签/截止/登录提醒)压成 0.1「平静」→ 图谱情绪边扁平。
    #   valence 决定标签方向, arousal 决定边强度; 外部传 sentiment 时仍以它为准(书库重标值)。
    from engine.sentiment import assess as _assess
    _a = _assess(event_text)
    sentiment = sentiment if sentiment is not None else _a["valence"]
    intensity = max(abs(sentiment), _a["arousal"])
    mood_label = mood_label_of(sentiment, intensity + 0.3)
    eid = add_mood_edge(entity, mood_label, max(0.1, intensity), ts, event_id, event_text, db)
    return {
        "event_id": event_id, "entity": entity,
        "memory_side": {"entity": entity, "fact": event_text[:80]},
        "mood_side": {"entity": entity, "mood_label": mood_label,
                      "intensity": round(max(0.1, intensity), 2)},
        "hidden_side": {"derived": False, "edge_id": 0,
                        "note": "真实暗注意力走 cog→add_hidden_text 通道"},
        "coupled": {"edge_id": eid, "shared": ["event_id", "entity"]},
    }


def dual_query(entity: str, db: str = None, mood: dict | None = None) -> dict:
    """耦合查询：实体 → 记忆事实 + 情绪历史（两图谱联动返回）。

    ★ 呼应机制（2026-08-27 用户：钥匙独立但需呼应）：
      ② 情绪一致检索线索（mood-congruent）——注入当前心态后，情绪相近的边权重加成重排
      ③ 情绪衰减内容永存（fading affect bias）——负面情绪边衰减更快，事实本身不丢（L0 永存）
      返回的每条边带 weight（= 调制后的呼应强度）。
    """
    import math
    mood_label = (mood or {}).get("label") or (mood or {}).get("mood_label")
    con = _conn(db)
    rows = con.execute(
        "SELECT mood_label,intensity,ts,trigger FROM mood_graph WHERE entity=? ORDER BY ts DESC LIMIT 20",
        (entity,)).fetchall()
    con.close()
    now = time.time()
    mood_hist = []
    for r in rows:
        label, intensity, ts, trigger = r
        age_days = max(0.0, (now - ts) / 86400.0) if ts else 0.0
        # ③ 情绪衰减：负面(低落/焦虑)τ=15 衰减快，正面/平静 τ=60 衰减慢（fading affect bias）
        tau = 15.0 if label in ("低落", "焦虑") else 60.0
        decay = math.exp(-age_days / tau)
        # ② mood-congruent：当前心态与历史情绪相近 → 加成（开心时易想起开心事）
        congruent = 1.0
        if mood_label:
            congruent = 0.4 + 0.6 * _mood_sim(mood_label, label)
        weight = round(intensity * decay * congruent, 3)
        mood_hist.append({"mood_label": label, "intensity": intensity,
                          "ts": time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "",
                          "trigger": trigger[:40], "decay": round(decay, 2),
                          "congruent": round(congruent, 2), "weight": weight})
    # 呼应强度排序（衰减×心态相近）
    mood_hist.sort(key=lambda m: -m["weight"])
    return {"entity": entity, "mood_history": mood_hist,
            "coupled": bool(mood_hist), "mood_congruent": mood_label}


def query_event(event_id: str, db: str = None) -> dict:
    """④ 事件团簇：同 event_id 的全部实体 + 情绪（"何时何地"时间轴上的事件节点）。"""
    con = _conn(db)
    rows = con.execute(
        "SELECT entity,mood_label,intensity,ts,trigger FROM mood_graph WHERE event_id=? ORDER BY ts",
        (event_id,)).fetchall()
    con.close()
    return {"event_id": event_id,
            "clusters": [{"entity": r[0], "mood_label": r[1], "intensity": r[2],
                          "ts": time.strftime("%Y-%m-%d", time.localtime(r[3])) if r[3] else "",
                          "trigger": r[4][:40]} for r in rows]}


def reevaluate(entity: str, event_text: str, sentiment: float = None,
               db: str = None, ts: float = None) -> dict:
    """⑤ 回固化情绪重估：事件重访/回忆时，以最新心境重新评估情绪边（情绪可被重新评估，内容永存）。"""
    from engine.sentiment import assess as _assess
    _a = _assess(event_text)
    sentiment = sentiment if sentiment is not None else _a["valence"]
    intensity = max(0.1, abs(sentiment), _a["arousal"])
    label = mood_label_of(sentiment, intensity + 0.3)
    eid = add_mood_edge(entity, label, intensity, ts or time.time(),
                        f"re-{int(time.time())}", f"(重估) {event_text[:80]}", db)
    return {"entity": entity, "new_mood": label, "intensity": round(intensity, 2),
            "note": "内容(记忆图谱)永存，情绪(情绪图谱)以新心境重新评估", "edge_id": eid}


# 情绪相近度（呼应调制核心）：0 无关 .. 1 相同
_MOOD_SIM = {
    "兴奋": {"兴奋": 1.0, "轻微兴奋": 0.85, "平静": 0.4, "专注": 0.3, "低落": 0.1, "焦虑": 0.1},
    "轻微兴奋": {"兴奋": 0.85, "轻微兴奋": 1.0, "平静": 0.6, "专注": 0.4, "低落": 0.15, "焦虑": 0.15},
    "平静": {"兴奋": 0.4, "轻微兴奋": 0.6, "平静": 1.0, "专注": 0.5, "低落": 0.3, "焦虑": 0.3},
    "专注": {"兴奋": 0.3, "轻微兴奋": 0.4, "平静": 0.5, "专注": 1.0, "低落": 0.25, "焦虑": 0.3},
    "低落": {"兴奋": 0.1, "轻微兴奋": 0.15, "平静": 0.3, "专注": 0.25, "低落": 1.0, "焦虑": 0.6},
    "焦虑": {"兴奋": 0.1, "轻微兴奋": 0.15, "平静": 0.3, "专注": 0.3, "低落": 0.6, "焦虑": 1.0},
}


def _mood_sim(m1: str, m2: str) -> float:
    return _MOOD_SIM.get(m1, {}).get(m2, 0.3)


if __name__ == "__main__":
    if "--add" in sys.argv:
        i = sys.argv.index("--add")
        t = sys.argv[i + 1]
        m = sys.argv[sys.argv.index("--mood") + 1] if "--mood" in sys.argv else None
        ent = sys.argv[sys.argv.index("--entity") + 1] if "--entity" in sys.argv else entity_of(t)
        eid = sys.argv[sys.argv.index("--event") + 1] if "--event" in sys.argv else f"ev-{int(time.time())}"
        r = dual_graph_ingest(t, event_id=eid)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif "--query" in sys.argv:
        ent = sys.argv[sys.argv.index("--query") + 1]
        print(json.dumps(dual_query(ent), ensure_ascii=False, indent=2))
    else:
        print(__doc__)
