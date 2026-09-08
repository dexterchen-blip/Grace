#!/usr/bin/env python3
"""夜班摄入反应实验 —— 用夜班方式摄入正式系统的多源记忆，看 Grace V2 反应。

背景（用户）：正式系统绝大多数记忆走夜班摄入（wechat/邮件/学校信息），不是聊天框。
本实验：把正式 L0 的夜班源（wechat 570 / email / school / exchange:inbox）当作
        "夜班输入"，喂给 Grace V2：显式注意力（情绪×正式 L2）→ 自激发 → 心态。

注意：正式源只读；wechat sensitive=true 文本截断显示，不泄露完整内容。

用法（沙盒内）:
  ./run.sh .venv/bin/python3 v2/tests/night_ingest_react.py [--limit 60]
输出: experiments/run/night-ingest-react-*.md
"""
from __future__ import annotations
import json
import os
import re
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.attention import generate_attention
from engine.self_activation import decide
from engine.mood_engine import derive
import config  # noqa: E402

OFFICIAL = "/Users/cz/WorkBuddy/skills find and make/local-ai-agent"
OFF_L0 = os.path.join(OFFICIAL, "memory", "L0_raw")
OFF_L2 = os.path.join(OFFICIAL, "memory", "L2_semantic", "l2.db")

SOURCES = ["wechat.jsonl", "email.jsonl", "school.jsonl", "exchange:inbox.jsonl", "doc:file.jsonl"]


def extract_events(source: str, limit: int) -> list[dict]:
    """从正式 L0 单源提取消息文本（夜班摄入的原始内容）。"""
    path = os.path.join(OFF_L0, source)
    if not os.path.isfile(path):
        return []
    events = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = rec.get("payload", {})
        if not isinstance(p, dict):
            continue
        # 通用：payload.text / messages[].text / title+body（邮件）
        texts = []
        if p.get("text"):
            texts.append(p["text"])
        for m in p.get("messages", []):
            if isinstance(m, dict) and m.get("text"):
                texts.append(m["text"])
            elif isinstance(m, dict) and m.get("content"):
                texts.append(m["content"])
        if p.get("subject"):
            texts.insert(0, p["subject"])
        for t in texts:
            t = str(t)[:200]
            if len(t) >= 6 and t not in [e["text"] for e in events]:
                events.append({"text": t, "src": source.split(":")[0]})
        if len(events) >= limit:
            break
    return events[:limit]


def official_facts(q: str, k: int = 3) -> list[str]:
    """正式 L2 只读检索（关键词 LIKE 匹配）。"""
    if not os.path.isfile(OFF_L2):
        return []
    tokens = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9\-]{2,}", q)][:4]
    if not tokens:
        return []
    try:
        con = sqlite3.connect(f"file:{OFF_L2}?mode=ro", uri=True)
        like = " OR ".join(f"text LIKE '%{t}%'" for t in tokens)
        rows = con.execute(f"SELECT text FROM docs WHERE {like} ORDER BY ts DESC LIMIT ?", (k,)).fetchall()
        con.close()
        return [r[0][:120] for r in rows]
    except sqlite3.Error:
        return []


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    all_events = []
    for src in SOURCES:
        all_events += extract_events(src, args.limit)
    print(f"夜班源事件: {len(all_events)} 条（wechat/email/school/inbox/doc）")

    results = []
    for ev in all_events:
        facts = official_facts(ev["text"])
        att = generate_attention(ev["text"], mood=None, facts=facts)
        dec = decide(att, ev["text"])
        results.append({"src": ev["src"], "text": ev["text"], "att": att, "dec": dec})

    # 心态：全部事件累进 derive（夜班心态推演）
    evs = [{"text": r["text"], "sentiment": r["att"]["sentiment"], "weight": 0.5} for r in results]
    mood = None
    try:
        mood = derive(evs, db=None) if False else None
    except Exception:  # noqa: BLE001
        pass

    lines = [f"# Grace V2 夜班摄入反应实验\n",
             f"> 来源: 正式 L0 夜班源（只读）｜ {len(results)} 条 ｜ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
             "## 分源反应统计\n",
             "| 源 | 事件数 | 自激发 | attention_driven | mood_driven | memory_driven |",
             "|---|---|---|---|---|---|"]
    for src in sorted({r["src"] for r in results}):
        sub = [r for r in results if r["src"] == src]
        act = sum(1 for r in sub if r["dec"]["activate"])
        trig = {}
        for r in sub:
            for t in r["dec"].get("triggers", []):
                trig[t] = trig.get(t, 0) + 1
        lines.append(f"| {src} | {len(sub)} | {act} | {trig.get('attention_driven',0)} | {trig.get('mood_driven',0)} | {trig.get('memory_driven',0)} |")

    lines.append("\n## 高显著反应（top 8）\n")
    results.sort(key=lambda r: -r["att"]["salience"])
    for r in results[:8]:
        a, d = r["att"], r["dec"]
        act = "🔥自激发" if d["activate"] else "·观察·"
        t = r["text"] if r["src"] != "wechat" else r["text"][:40] + "…"   # 微信截断
        lines.append(f"- **[{a['salience']:.2f}] {act}** `{r['src']}` {t[:42]}")
        lines.append(f"  - 注意力: {a['attention_text'][:80]}")
        if d["activate"]:
            lines.append(f"  - 行动: {d.get('action','')[:56]}（{d.get('reason','')}）")

    lines.append("\n## mood_driven 反应（情绪驱动，wechat 生活内容）\n")
    mood_hits = [r for r in results if "mood_driven" in r["dec"].get("triggers", [])]
    for r in mood_hits[:5]:
        a = r["att"]
        lines.append(f"- [{a['salience']:.2f}] {r['text'][:46]}")
        lines.append(f"  - 注意力: {a['attention_text'][:76]}")

    out = os.path.join(config.EXPERIMENTS, "run", f"night-ingest-react-{datetime.now().strftime('%Y%m%d-%H%M')}.md")
    os.makedirs(config.EXPERIMENTS, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n报告 → {out}")


if __name__ == "__main__":
    main()
