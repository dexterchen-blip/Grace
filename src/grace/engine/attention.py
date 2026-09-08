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
    """事件情绪初判（-1..1）：统一效价评估（engine/sentiment.py, 2026-09-03 双轴化）。

    ★2026-09-03 Phase 0: 原词典仅 14+14 词 + 社交值 → 88.8% sentiment=0 → ToM 现实标签
    84% 假"平静"。改走统一 sentiment.assess() 分层效价词表（含否定削弱）。
    """
    from engine.sentiment import valence_of
    return valence_of(text)


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
