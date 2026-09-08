#!/usr/bin/env python3
"""真实记忆摄入反应实验 —— 把正式本地 AI 系统的记忆喂给 Grace V2，看反应。

来源（只读）：local-ai-agent/memory/L0_raw/chat.jsonl（真实对话）+
                L2_semantic/l2.db docs（真实语义记忆）
反应：显式注意力（情绪×记忆）→ 自激发决策（三源触发）→ 心态背景

用法（沙盒内）:
  ./run.sh .venv/bin/python3 v2/tests/memory_ingest_react.py [--limit 30]
输出: experiments/run/ingest-react-*.md
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.attention import generate_attention
from engine.self_activation import decide
import config  # noqa: E402

OFFICIAL = "/Users/cz/WorkBuddy/skills find and make/local-ai-agent"
CHAT = os.path.join(OFFICIAL, "memory", "L0_raw", "chat.jsonl")


def load_official_chat(limit: int) -> list[dict]:
    """只读正式 chat.jsonl，提取最近 limit 条 user 消息（真实事件）。"""
    if not os.path.isfile(CHAT):
        return []
    events = []
    for line in open(CHAT, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = rec.get("payload", {})
        msgs = p.get("messages", []) if isinstance(p, dict) else []
        ts = rec.get("ts", "")
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user" and m.get("text"):
                events.append({"text": m["text"][:120], "ts": str(ts)[:16]})
    return events[-limit:]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    events = load_official_chat(args.limit)
    if not events:
        print(f"❌ 正式 chat.jsonl 无数据或不存在：{CHAT}")
        return
    print(f"从正式 L0 读取 {len(events)} 条真实用户消息")

    # 正式 L2 只读检索（关联真实记忆，不写库）
    import re
    import sqlite3
    off_l2 = os.path.join(OFFICIAL, "memory", "L2_semantic", "l2.db")

    def official_facts(q: str, k: int = 3) -> list[str]:
        if not os.path.isfile(off_l2):
            return []
        try:
            con = sqlite3.connect(f"file:{off_l2}?mode=ro", uri=True)
        except sqlite3.Error:
            con = sqlite3.connect(off_l2)
        tokens = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9\-]{2,}", q)][:4]
        out = []
        if tokens:
            try:
                like = " OR ".join(f"text LIKE '%{t}%'" for t in tokens)
                rows = con.execute(f"SELECT text FROM docs WHERE {like} ORDER BY ts DESC LIMIT ?",
                                   (k,)).fetchall()
                out = [r[0][:120] for r in rows]
            except sqlite3.Error:
                out = []
        con.close()
        return out

    rows = []
    for ev in events:
        facts = official_facts(ev["text"])
        att = generate_attention(ev["text"], mood=None, facts=facts)   # 情绪×正式记忆
        dec = decide(att, ev["text"])
        rows.append({"ts": ev["ts"], "text": ev["text"], "att": att, "dec": dec, "n_links": len(facts)})

    # 排序：显著度降序
    rows.sort(key=lambda r: -r["att"]["salience"])
    lines = [f"# Grace V2 摄入正式记忆反应实验\n",
             f"> 来源: 正式 L0 chat.jsonl（只读）｜ {len(rows)} 条真实事件 ｜ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
             "## 高显著事件（注意力优先）\n"]
    for r in rows[:8]:
        a, d = r["att"], r["dec"]
        act = "🔥自激发" if d["activate"] else "·观察·"
        lines.append(f"- **[{a['salience']:.2f}] {act}** `{r['ts']}` {r['text'][:44]}")
        lines.append(f"  - 注意力: {a['attention_text'][:84]}")
        lines.append(f"  - 关联记忆 {r['n_links']} 条: {a['memory_links'][0][:50] if a['memory_links'] else '（无）'}")
        if d["activate"]:
            lines.append(f"  - 行动: {d.get('action', '')[:60]}（{d.get('reason','')}）")

    lines.append("\n## 自激发统计\n")
    act_n = sum(1 for r in rows if r["dec"]["activate"])
    trig = {}
    for r in rows:
        for t in r["dec"].get("triggers", []):
            trig[t] = trig.get(t, 0) + 1
    lines.append(f"- 自激发 {act_n}/{len(rows)} 条 ｜ 触发源分布: {json.dumps(trig, ensure_ascii=False)}")
    lines.append(f"- 高显著(≥0.55): {sum(1 for r in rows if r['att']['salience'] >= 0.55)} 条")

    lines.append("\n## 低显著事件（观察不打扰）\n")
    for r in rows[-5:]:
        lines.append(f"- [{r['att']['salience']:.2f}] {r['text'][:50]}")

    out = os.path.join(config.EXPERIMENTS, "run", f"ingest-react-{datetime.now().strftime('%Y%m%d-%H%M')}.md")
    os.makedirs(config.EXPERIMENTS, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n报告 → {out}")


if __name__ == "__main__":
    main()
