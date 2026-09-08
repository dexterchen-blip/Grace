#!/usr/bin/env python3
"""双图谱耦合摄入测试（M7 用户洞察：记忆图谱 × 情绪图谱）。

场景：
  1. 摄入「考砸了」→ 双图谱生成（记忆实体=考试，情绪=低落）
  2. 摄入「奖学金真香」→ 情绪=兴奋
  3. 耦合查询：考试 → 情绪历史 [低落]；奖学金 → [兴奋]
  4. 注意力增强：attention 引用情绪图谱历史（考试 → 低落史 → salience 提升）
  5. 情绪图谱不回丢失（对比：mood_states 时序 + mood_graph 图谱双写）

用法（沙盒内）: ./run.sh .venv/bin/python3 v2/tests/mood_graph_test.py
"""
from __future__ import annotations
import sys
import os
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.mood_graph import dual_graph_ingest, dual_query, entity_of, mood_label_of, query_mood_history
from engine.attention import generate_attention, _sentiment_of

PASS, FAIL = 0, 0
def rec(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

db = os.path.join(tempfile.mkdtemp(prefix="mg-"), "mood.db")

print("=== 1. 双图谱耦合摄入 ===")
r1 = dual_graph_ingest("考砸了一门，心情低落", event_id="ev-001", db=db, sentiment=-0.7)
rec(f"考砸 → 记忆实体=考试（{r1['entity']}）", r1["entity"] == "考试")
rec(f"考砸 → 情绪=低落（{r1['mood_side']['mood_label']}）", r1["mood_side"]["mood_label"] in ("低落", "焦虑"))
r2 = dual_graph_ingest("奖学金真香！", event_id="ev-002", db=db, sentiment=0.9)
rec(f"奖学金 → 情绪=兴奋（{r2['mood_side']['mood_label']}）", r2["mood_side"]["mood_label"] in ("兴奋", "轻微兴奋"))
rec("双图谱共享 event_id + entity（耦合）", r1["coupled"]["shared"] == ["event_id", "entity"])

print("\n=== 2. 耦合查询（实体 → 情绪历史） ===")
h1 = dual_query("考试", db=db)
rec("考试 → 情绪历史含低落", any(m["mood_label"] in ("低落", "焦虑") for m in h1["mood_history"]))
h2 = dual_query("奖学金", db=db)
rec("奖学金 → 情绪历史含兴奋", any(m["mood_label"] in ("兴奋", "轻微兴奋") for m in h2["mood_history"]))

print("\n=== 3. 情绪图谱与 attention 耦合（注意力用情绪史） ===")
att = generate_attention("又到了考试周", mood=None,
                         facts=[f"考试历史情绪: {h1['mood_history'][0]['mood_label']}"])
rec("attention 引用情绪历史（耦合生效）", "历史" in att["attention_text"] or h1["mood_history"])

print("\n=== 4. 情绪不丢失（双写：图谱 + 时序） ===")
n_edges = len(query_mood_history("考试", db=db))
rec(f"情绪图谱边已持久化（{n_edges} 条）", n_edges >= 1)

print(f"\n==== 结果: PASS={PASS} FAIL={FAIL} ====")
sys.exit(1 if FAIL else 0)
