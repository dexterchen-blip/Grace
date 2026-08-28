#!/usr/bin/env python3
"""显式注意力生成器 —— M7（2026-08-27 用户方向：情绪系统 × 记忆系统共同生成注意力）。

核心洞察（用户）：
  注意力不是凭空产生的，是「这件事让我什么感受（情绪）× 这关联到我记得的什么（记忆）」的共同产物。
  显式注意力 = 把加工层的隐式过程显式化，模型"看见自己注意到了什么"。

输出（每次事件/对话）：
  {
    "focus": 注意焦点（事件要点 + 关联记忆主题）,
    "salience": 显著度 0-1（= 0.5×情绪显著度 + 0.5×记忆关联度）,
    "mood_ctx": 情绪背景（三层融合标签 + 强度）,
    "memory_links": 关联记忆（L2 检索 top-k）,
    "attention_text": 显式注意力文案（可注入 system）
  }

用法（沙盒内）:
  ./run.sh .venv/bin/python3 v2/engine/attention.py --event "考砸了一门" --facts "他期中考试压力很大" 
"""
from __future__ import annotations
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import config  # noqa: E402


def _sentiment_of(text: str) -> float:
    """事件情绪初判（-1..1）：情感词匹配 + 社交价值（2026-08-28）。

    社交邀约（一起/等我/约/聚餐/找我）→ 微正：真人会把「朋友约我」当重要社交信号。
    """
    neg = re.compile(r"考砸|难过|焦虑|压力|累|失眠|想家|失败|担心|紧张|低落|难受|烦")
    pos = re.compile(r"开心|高兴|成功|通过|顺利|棒|喜欢|不错|好消息|期待|兴奋|好棒|通过")
    social = re.compile(r"一起|等我|找我|约|聚餐|吃饭|跑步|接龙|社团|外拍|来我家|操场|集合")
    s = 0.0
    if pos.search(text):
        s += 0.6
    if neg.search(text):
        s -= 0.7
    if social.search(text) and "广告" not in text and "优惠" not in text:
        s += 0.35               # 社交价值：邀约/陪伴信号（真人会注意）
    return max(-1.0, min(1.0, s))


def generate_attention(event_text: str, mood: dict | None = None,
                       facts: list[str] | None = None, k: int = 3) -> dict:
    """显式注意力 = 情绪显著度 × 记忆关联度。

    mood: combined_emotion 输出（三层融合）或 None（自动取当前心态）。
    facts: 注入的关联记忆；None 时用 L2 检索（生产）。
    """
    # 情绪侧
    sentiment = _sentiment_of(event_text)
    if mood is None:
        try:
            from engine.mood_engine import combined_emotion
            mood = combined_emotion()
        except Exception:  # noqa: BLE001
            mood = {"label": "平静", "intensity": 0.5}
    mood_label = mood.get("label", mood.get("mood_label", "平静"))
    mood_intensity = float(mood.get("combined", mood.get("intensity", 0.5)))
    # 情绪显著度：事件强度 × 当前情绪敏感度（情绪波动大时更敏感）
    mood_sens = 0.5 + abs(mood_intensity - 0.5)          # 情绪偏离中性越远越敏感
    mood_sal = abs(sentiment) * mood_sens

    # 记忆侧
    if facts is None:
        try:
            import l2_semantic
            facts = [h["text"] for h in l2_semantic.search(event_text, k=k)]
        except Exception:  # noqa: BLE001
            facts = []
    mem_sal = min(1.0, len(facts) / k * 0.9 + 0.05) if facts else 0.1   # 有强关联记忆 → 高
    mem_sal = round(mem_sal, 2)

    salience = round(0.5 * min(1.0, mood_sal) + 0.5 * mem_sal, 2)

    # 注意焦点：事件要点 + 关联记忆主题
    focus = event_text[:30]
    links = facts[:k]
    attention_text = _attention_text(focus, salience, mood_label, links, sentiment)
    return {
        "focus": focus, "salience": salience,
        "mood_ctx": {"label": mood_label, "intensity": round(mood_intensity, 2)},
        "memory_links": links,
        "sentiment": round(sentiment, 2),
        "attention_text": attention_text,
    }


def _attention_text(focus: str, salience: float, mood_label: str,
                    links: list[str], sentiment: float) -> str:
    """显式注意力文案（雷姆视角，可注入 system/自激发决策）。"""
    mood_ph = {
        "兴奋": "此刻雷姆心里是雀跃的", "轻微兴奋": "雷姆心里是暖的",
        "平静": "雷姆心情平静", "专注": "雷姆很专注",
        "低落": "雷姆的心情跟着沉了一下", "焦虑": "雷姆有些不安",
    }.get(mood_label, "雷姆心情平静")
    link_ph = f"，这让雷姆想起：{links[0][:40]}…" if links else ""
    if salience >= 0.7:
        head = "【高显著】雷姆注意到了"
    elif salience >= 0.4:
        head = "【中显著】雷姆注意到"
    else:
        head = "【低显著】雷姆瞥见"
    return f"{head}「{focus}」——{mood_ph}{link_ph}。"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="考砸了一门，心情低落")
    ap.add_argument("--facts", nargs="*", default=None)
    args = ap.parse_args()
    r = generate_attention(args.event, facts=args.facts)
    print(json.dumps(r, ensure_ascii=False, indent=2))
