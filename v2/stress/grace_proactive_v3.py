#!/usr/bin/env python3
"""Grace 主动会话 v3 —— ToM 实时化（2026-08-28 用户：主人状态实时变化 → 主动时间/决定实时变）。

核心：主人情绪不是静态传入，而是**随当天书库事件实时推演**（真实感）：
  每天书库流入 → 事件 sentiment 加权 → 主人当天情绪(实时)
  → 注意力(事件) + ToM(主人实时状态) → 自激发 decide(tom=) → 主动消息
  → 主动消息进训练样本 + L3 自传体矩阵摄入（配合压力测试）

验证：① 她主动找的时间/事件决定随主人状态实时变化 ② L3 矩阵 90 天积累。

用法: ./run.sh .venv/bin/python3 grace-book/grace_proactive_v3.py [--days 90]
输出: grace-book/run/proactive-v3-*.md + proactive-train-v3.jsonl
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2", "engine"))
from engine.attention import generate_attention, _sentiment_of  # noqa: E402
from engine.self_activation import decide  # noqa: E402
from engine.theory_of_mind import infer_owner_state  # noqa: E402
from engine.autobiography import add_event  # noqa: E402
from engine.mood_graph import entity_of  # noqa: E402

GRACE = os.path.dirname(os.path.abspath(__file__))
INCOMING = os.path.join(GRACE, "incoming")
RUN = os.path.join(GRACE, "run")
AUTO_DB = os.path.join(GRACE, "memory", "L3_auto", "autobiography.db")


def owner_mood_of(items: list[str]) -> tuple[str, float]:
    """★ 主人当天情绪实时推演：取当天**情绪峰值**（最强烈经历决定一天心情）。

    加权平均会被中性内容稀释 → 全"平静"；真人一天的情绪由最强事件主导。
    """
    best = None
    for t in items:
        s = _sentiment_of(t)
        if best is None or abs(s) > abs(best[1]):
            best = (t, s)
    s = best[1] if best else 0.0
    if s > 0.3:
        return ("兴奋" if s > 0.55 else "轻微兴奋"), s
    if s < -0.3:
        return ("低落" if s < -0.5 else "焦虑"), s
    return "平静", s


def run(days: int | None = None, max_cand: int = 4):
    os.makedirs(RUN, exist_ok=True)
    os.makedirs(os.path.dirname(AUTO_DB), exist_ok=True)
    files = sorted(f for f in os.listdir(INCOMING) if f.startswith("day-"))
    if days:
        files = files[:days]
    log = [f"# Grace 主动会话 v3（ToM 实时化 + L3 矩阵）\n",
           f"> {time.strftime('%Y-%m-%d %H:%M')} ｜ 书库 {len(files)} 天渐进 ｜ 主人情绪实时推演\n"]
    train = []
    stats = {"proactive": 0, "mood_map": {}}
    reminded = set()
    for fn in files:
        day = fn.replace("day-", "").replace(".jsonl", "")
        items = [json.loads(l)["text"] for l in open(os.path.join(INCOMING, fn), encoding="utf-8")]
        owner_mood, owner_s = owner_mood_of(items)          # ★ 主人当天情绪(实时)
        stats["mood_map"][owner_mood] = stats["mood_map"].get(owner_mood, 0) + 1
        cands = []
        for t in items:
            att = generate_attention(t, mood=None, facts=[])
            cands.append((t, att))
        cands.sort(key=lambda x: -x[1]["salience"])
        day_msgs = []
        for t, att in cands[:max_cand]:
            tom = infer_owner_state(t, owner_mood)          # ★ ToM：主人实时状态
            r = decide(att, t, tom=tom)
            if not r["activate"]:
                continue
            key = t.strip()[:50]
            if key in reminded:
                continue
            reminded.add(key)
            msg = (f"昴君，雷姆注意到：{t[:30]}……" if tom["event_type"] == "社交" and owner_mood in ("兴奋", "轻微兴奋")
                   else f"昴君，雷姆想跟你说——{t[:30]}。")
            day_msgs.append({"text": t[:56], "owner_mood": owner_mood, "tom": tom["advice"][:22],
                             "msg": msg})
            train.append({"day": day, "owner_mood": owner_mood, "situation": t[:110], "message": msg})
            # ★ L3 自传体矩阵摄入（事件 → 矩阵节点，与双轨配合）
            #   2026-08-29 审计修复：entity 抽取 / confidence=medium(模拟书库,非直接L0) /
            #   evidence=事件原文(溯源) / relation+self_eval 按情绪推导
            _rel = {"低落": "思念与守护", "焦虑": "担忧与陪伴", "兴奋": "分享与骄傲",
                    "轻微兴奋": "温暖", "平静": "日常守护"}.get(owner_mood, "陪伴")
            _eval = {"低落": "雷姆想把这一天也记住,因为主人的感受对雷姆很重要",
                     "焦虑": "雷姆想为主人分担一点不安",
                     "兴奋": "主人开心,雷姆也开心",
                     "轻微兴奋": "这样的日子,雷姆很喜欢",
                     "平静": "这一天的事,雷姆记住了"}.get(owner_mood, "雷姆记住了")
            add_event(t, person="", entity=entity_of(t), emotion=owner_mood,
                      relation=_rel, self_eval=_eval, db=AUTO_DB,
                      confidence="medium", evidence=t[:120],
                      ts=datetime.fromisoformat(day).timestamp())
        stats["proactive"] += len(day_msgs)
        if day_msgs:
            log.append(f"## {day}（主人情绪:{owner_mood}｜她主动 {len(day_msgs)} 次）")
            for m in day_msgs:
                log.append(f"- 〔ToM:{m['tom']}〕 {m['msg'][:52]}")
    with open(os.path.join(RUN, "proactive-train-v3.jsonl"), "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.append(f"\n## 总计：她主动 {stats['proactive']} 次 ｜ 主人情绪分布 {json.dumps(stats['mood_map'], ensure_ascii=False)}")
    log.append(f"训练样本 {len(train)} 条 → proactive-train-v3.jsonl ｜ L3 矩阵节点 {len(train)} 个 → L3_auto/")
    with open(os.path.join(RUN, f"proactive-v3-{time.strftime('%H%M')}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"\n→ {RUN}/proactive-v3-*.md")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--max-cand", type=int, default=4)
    args = ap.parse_args()
    run(args.days, args.max_cand)
