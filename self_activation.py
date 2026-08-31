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


def evaluate(attention: dict, event_text: str) -> dict:
    """自激发评估：注意力显著度 + 情绪信号 + 记忆关联 → 是否值得激活。

    2026-08-27 修：memory_driven 只由「时限/重要词」触发（截止/明天/due 等），
    仅有关联记忆不算触发源——否则夜班摄入的每条内容都会自激发（99% 话痨）。
    """
    sal = attention["salience"]
    sig = _mood_signal(attention)
    mem_hint = bool(_MEMORY_HINTS.search(event_text))      # 时限/重要词 → 记忆驱动触发
    has_links = bool(attention["memory_links"])            # 有记忆关联 → 只加注意力分
    triggers = []
    if sal >= TRIGGER["attention_driven"]["threshold"]:
        triggers.append("attention_driven")
    if sig["low"] or sig["high"]:
        triggers.append("mood_driven")
    if mem_hint:
        triggers.append("memory_driven")
    # 激活分：显著度 + 情绪信号 + 记忆关联
    score = sal
    if sig["low"]:
        score += 0.15            # 低落 → 更该主动关心
    if sig["high"]:
        score += 0.1             # 兴奋 → 更想分享
    if mem_hint:
        score += 0.45            # 时限/重要事项 → 该做的事，不依赖情绪也该激活
    elif has_links:
        score += 0.1             # 普通记忆关联 → 轻加分（有相关记忆，但非紧急）
    return {"score": round(min(1.0, score), 2), "triggers": triggers,
            "mood_signal": sig, "mem_hit": mem_hint, "has_links": has_links}


def decide(attention: dict, event_text: str, decide_fn=None, tom: dict | None = None) -> dict:
    """模型自激发决策入口（2026-08-28：ToM 修饰是否主动）。

    decide_fn(attention, event) -> {"activate": bool, "action": str, "reason": str}
      默认规则骨架：score ≥ 0.5 激活；可注入 27B 生成（模型自己判断）。
    tom: theory_of_mind.infer_owner_state() —— 主人是否愿被打扰/需要什么。
       interruptible=False → 强制不打扰（真人：读懂对方再决定是否打扰）。
    """
    ev = evaluate(attention, event_text)
    if decide_fn is not None:
        try:
            return decide_fn(attention, event_text)
        except Exception:  # noqa: BLE001
            pass
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
