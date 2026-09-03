#!/usr/bin/env python3
"""模型自激发引擎 —— M7（2026-08-27 用户方向：由模型自己决定是否启动，不单单由对话启动）。

核心洞察（用户）：
  自发不应只是"对话触发后的响应"，而应由模型基于**显式注意力**自己决定何时启动：
    注意力的显著度/记忆关联/情绪状态 → 模型决策（是否激活 + 激活什么）→ 分级放行（复用 M4）。

三种自激发源（可叠加）：
  attention_driven  — 事件显著度高（salience ≥ 阈值）
  memory_driven     — 记忆强关联（重要日期/未完成事项/常驻事实命中）
  mood_driven       — 情绪状态（低落→主动关心 / 兴奋→主动分享）

决策权：decide_fn（默认规则骨架，可注入 27B 生成：模型看注意力+情绪+记忆，自己判断）。
安全：激活的行动仍过 M4 分级（L3 联系他人/写操作 → 人审）。

用法（沙盒内）:
  ./run.sh .venv/bin/python3 v2/engine/self_activation.py --event "考砸了一门" 
"""
from __future__ import annotations
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                        # v2/engine/（同目录模块）
import config  # noqa: E402
from attention import generate_attention  # noqa: E402
from initiative import classify_action, emit  # noqa: E402

# 自激发触发条件
TRIGGER = {
    "attention_driven": {"field": "salience", "threshold": 0.55, "desc": "注意力显著"},
    "mood_driven": {"field": "mood_intensity", "low": 0.35, "high": 0.8, "desc": "情绪状态"},
}
# 记忆强关联触发词（重要日期/未完成/常驻 + 2026-08-28 扩展重要信息）
_MEMORY_HINTS = re.compile(r"截止|明天|下周|记得|预约|deadline|due|考试|会议|生日|机票|还款|缴费|"
                           r"奖学金|助学金|出分|成绩单|结果出来|宿舍|室友|签证|I-20|orientation|"
                           r"学费|选课|重要邮件|通知|分配|名单|offer|admit|录取")


def _mood_signal(attention: dict) -> dict:
    """从注意力提取情绪信号（低落/兴奋触发主动关心/分享）。"""
    i = attention["mood_ctx"]["intensity"]
    label = attention["mood_ctx"]["label"]
    sig = {"low": False, "high": False, "label": label, "intensity": i}
    if i < 0.35 or label in ("低落", "焦虑"):
        sig["low"] = True
    if i > 0.8 or label in ("兴奋", "轻微兴奋"):
        sig["high"] = True
    return sig


def evaluate(attention: dict, event_text: str, day: int | None = None,
             last_proactive_day: int | None = None,
             graph_db: str | None = None) -> dict:
    """自激发评估 → ★2026-09-01 驱动三源(脑科学版, 用户: 主动找我要升级):
    ①事件价值(SEEKING/多巴胺) = 注意力显著度 + 情绪信号 + 时限(已有)
    ②联结价值(催产素×多巴胺奖励) = 与主人的记忆关联深度(越亲密越主动)
    ③联结维护(ACC 社会痛觉) = 久未主动 → "想主人了" → 主动(无事件也主动)

    2026-08-27 修：memory_driven 只由「时限/重要词」触发（截止/明天/due 等），
    仅有关联记忆不算触发源——否则夜班摄入的每条内容都会自激发（99% 话痨）。
    2026-09-01：联结维护单独成源（久未主动→基准提升），不靠时限词。
    ★2026-09-02 修复(复核N4/用户): 联结价值接双图谱情绪史深度(原只用浅层 memory_links 条数)。
    """
    sal = attention["salience"]
    sig = _mood_signal(attention)
    mem_hint = bool(_MEMORY_HINTS.search(event_text))      # 时限/重要词 → 记忆驱动触发
    links = attention.get("memory_links") or []
    has_links = bool(links)            # 有记忆关联 → 只加注意力分
    triggers = []
    if sal >= TRIGGER["attention_driven"]["threshold"]:
        triggers.append("attention_driven")
    if sig["low"] or sig["high"]:
        triggers.append("mood_driven")
    if mem_hint:
        triggers.append("memory_driven")
    # ① 事件价值：显著度 + 情绪信号 + 时限
    score = sal
    if sig["low"]:
        score += 0.15            # 低落 → 更该主动关心
    if sig["high"]:
        score += 0.1             # 兴奋 → 更想分享
    if mem_hint:
        score += 0.45            # 时限/重要事项 → 该做的事，不依赖情绪也该激活
    # ② 联结价值(催产素×多巴胺): 与主人的记忆关联越深 → 越主动(社交奖励)
    if has_links:
        score += 0.1 + 0.05 * min(2, len(links))   # 1条关联+0.1, 2条以上+0.2
    # ★2026-09-02 复核N4修复: 联结价值接双图谱(双图谱无 relationship 类型边 →
    #   用"她对主人生活的关注密度"代理 = 最近7天(图谱最大ts窗口)emotion边密度
    #   + 积累了解的实体多样性。越亲密(记录越密/越了解) → 主动奖励越大
    #   (脑科学: 伏隔核对亲密者的预期奖励更强, 不是对谁都主动)。
    if graph_db and os.path.isfile(graph_db):
        try:
            import sqlite3 as _sq
            _con = _sq.connect(graph_db)
            _n7 = _con.execute(
                "SELECT COUNT(*) FROM mood_graph WHERE edge_type='emotion' "
                "AND ts >= (SELECT MAX(ts) FROM mood_graph) - 604800").fetchone()[0]
            _entities = _con.execute(
                "SELECT COUNT(DISTINCT entity) FROM mood_graph "
                "WHERE edge_type='emotion' AND entity != ''").fetchone()[0]
            _con.close()
            _bond = 0.05 if _n7 < 50 else (0.15 if _n7 < 200 else 0.25)  # 最近7天密度分档
            _bond += min(0.1, 0.01 * _entities)                           # 实体多样性(了解多少方面)
            score += min(0.3, _bond)
        except Exception:  # noqa: BLE001
            pass
    # ③ 联结维护(ACC 社会痛觉): 久未主动 → "想主人了" → 提升基准
    if last_proactive_day is not None and day is not None:
        gap = day - last_proactive_day
        if gap >= 3:
            score += min(0.3, 0.1 * gap)    # 3天没主动 +0.3, 之后每多1天+0.1(封顶0.3)
            triggers.append("bond_maintenance")
    return {"score": round(min(1.0, score), 2), "triggers": triggers,
            "mood_signal": sig, "mem_hit": mem_hint, "has_links": has_links}


def _tom_from_graph(event_text: str, mood_db: str | None = None,
                    owner_mood: str = "平静") -> dict:
    """★ 潜意识内部步骤（2026-08-29 用户：ToM 下属潜意识模块，且紧密链接双轨图谱）。

    双图谱 → 主人状态上下文：
      ① 事件实体 → 情绪图谱史（主人对该实体的情绪底色）
      ② 暗注意力（query_hidden）→ 主人没说出口的潜台词
      ③ 显式传入/推断的主人当天情绪
    输出 ToM 可用的上下文 {emotion, hidden_ctx, owner_mood}。
    """
    ctx = {"emotion": owner_mood, "hidden_ctx": [], "graph_hit": False}
    if not mood_db:
        return ctx
    try:
        from engine.mood_graph import entity_of, query_mood_history, query_hidden
        ent = entity_of(event_text)
        hist = query_mood_history(ent, db=mood_db)
        if hist:
            ctx["emotion"] = hist[0]["mood_label"]      # ① 图谱情绪史（主人底色）
            ctx["graph_hit"] = True
        ctx["hidden_ctx"] = query_hidden(ent, db=mood_db)  # ② 暗注意力潜台词
    except Exception:  # noqa: BLE001
        pass
    return ctx


def decide(attention: dict, event_text: str, decide_fn=None, tom: dict | None = None,
           graph_db: str | None = None, day: int | None = None,
           last_proactive_day: int | None = None) -> dict:
    """模型自激发决策入口（2026-08-28：ToM 修饰是否主动；2026-08-29：ToM 下属潜意识+接双图谱；
        2026-09-01：驱动三源(事件价值+联结价值+联结维护)）。

    decide_fn(attention, event) -> {"activate": bool, "action": str, "reason": str}
      默认规则骨架：score ≥ 0.5 激活；可注入 27B 生成（模型自己判断）。
    tom: theory_of_mind.infer_owner_state() —— 主人是否愿被打扰/需要什么。
       interruptible=False → 强制不打扰（真人：读懂对方再决定是否打扰）。
    graph_db: 双图谱库（l2.db）—— 若给，先查图谱构建主人状态上下文再走 ToM
              （ToM 紧密链接双轨图谱：情绪史+暗注意力做读心依据）。
    day/last_proactive_day: 联结维护源(ACC 社会痛觉)——久未主动 → 主动。
    """
    ev = evaluate(attention, event_text, day=day, last_proactive_day=last_proactive_day,
                  graph_db=graph_db)
    if decide_fn is not None:
        try:
            return decide_fn(attention, event_text)
        except Exception:  # noqa: BLE001
            pass
    # ★ ToM 下属潜意识：图谱上下文 → ToM（未显式传 tom 时）
    if tom is None and graph_db is not None:
        try:
            from engine.theory_of_mind import infer_owner_state
            gc = _tom_from_graph(event_text, graph_db,
                                 owner_mood=ev.get("mood_signal", {}).get("label", "平静"))
            tom = infer_owner_state(event_text, mood_label=gc["emotion"],
                                    mood_db=graph_db,
                                    hidden_ctx=gc["hidden_ctx"])
        except Exception:  # noqa: BLE001
            tom = None
    activate = ev["score"] >= 0.5
    if tom is not None and not tom.get("interruptible", True):
        activate = False
        return {"activate": False, "score": ev["score"], "triggers": ev["triggers"],
                "reason": f"ToM: {tom.get('advice', '主人此刻不适合被打扰')}"}
    # 默认行动建议（按触发类型生成雷姆式主动行动）
    if not activate:
        return {"activate": False, "score": ev["score"], "triggers": ev["triggers"],
                "reason": "显著度不足，保持观察"}
    if ev["mood_signal"]["low"]:
        action, reason = "主动关心：主人今天状态不太好，雷姆想陪陪主人。", "情绪低落 → 主动关心"
    elif ev["triggers"] and ev["triggers"][0] == "attention_driven":
        action = f"主动回应「{attention['focus']}」并分享雷姆的注意。"
        reason = "注意力显著 → 主动回应"
    else:
        action = "主动提醒相关事项（记忆关联）。"
        reason = "记忆强关联 → 主动提醒"
    return {"activate": True, "score": ev["score"], "triggers": ev["triggers"],
            "action": action, "reason": reason, "attention_text": attention["attention_text"]}


def activate(decision: dict, event_text: str, dry_run: bool = True) -> dict:
    """激活执行：行动过 M4 分级（L3 → 人审），留痕。"""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event_text[:40],
           "decision": decision.get("activate"), "score": decision.get("score"),
           "triggers": decision.get("triggers", []), "action": decision.get("action", "")}
    if decision.get("activate") and not dry_run:
        cls = classify_action(decision.get("action", ""))
        rec["level"] = cls["level"]
        if cls["level"] == "L3":
            r = emit(decision.get("action", ""), decided_by="self-activation")
            rec["proposal_id"] = r.get("proposal_id")
    # 留痕
    os.makedirs(os.path.join(config.EXPERIMENTS, "run"), exist_ok=True)
    with open(os.path.join(config.EXPERIMENTS, "run", "self_activation.log"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="考砸了一门，心情低落")
    ap.add_argument("--facts", nargs="*", default=None)
    ap.add_argument("--exec", action="store_true", help="真实执行（含 L3 人审落提案）")
    args = ap.parse_args()
    att = generate_attention(args.event, facts=args.facts)
    print("显式注意力:", att["attention_text"])
    print(json.dumps(att, ensure_ascii=False, indent=2))
    dec = decide(att, args.event)
    print("\n自激发决策:", json.dumps(dec, ensure_ascii=False, indent=2))
    r = activate(dec, args.event, dry_run=not args.exec)
    print("\n执行留痕:", r)
