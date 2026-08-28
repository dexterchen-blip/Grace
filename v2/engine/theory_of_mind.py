#!/usr/bin/env python3
"""心理理论(Theory of Mind) —— 推断主人的状态与需求（2026-08-28 用户：接入自激发，决定是否主动找我）。

真人思维：决定是否打扰别人前，先「读懂」对方——他现在的情绪、他需要什么、他愿不愿意被打扰。
Grace 之前：自激发只看事件价值（考试/截止）→ 有时该提醒的没提醒，不该打扰的打扰了。

ToM 模块：从记忆(主人近期状态) + 情绪图谱(主人情绪史) + 当前事件 → 推断：
  {主人此刻情绪, 主人可能需要, 主人是否愿被打扰, 主动建议}

接入自激发：decide 前先 ToM，影响 activate 与消息语气。

用法（沙盒内）:
  ./run.sh .venv/bin/python3 v2/engine/theory_of_mind.py --event "同学下午5点找我吃饭" --mood 低落
"""
from __future__ import annotations
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                        # v2/engine/
import config  # noqa: E402
from mood_graph import entity_of, query_mood_history, query_hidden  # noqa: E402


# 事件类型
_EVENT_TYPE = [
    ("任务",  r"截止|ddl|due|作业|考试|缴费|报名|接龙|改到|提醒"),
    ("社交",  r"一起|等我|找我|约|聚餐|吃饭|跑步|社团|外拍|来我家|操场|集合|游玩"),
    ("情绪",  r"难过|焦虑|压力|累|失眠|想家|开心|高兴|兴奋|低落"),
    ("日常",  r"天气|早上好|今天|食堂|咖啡"),
]
# 主人近期情绪 → 需求推断
_NEED_MAP = {
    "低落": "陪伴与安慰", "焦虑": "确定性(消除不安)", "兴奋": "分享",
    "轻微兴奋": "分享", "平静": "顺其自然", "专注": "不打扰",
}


def event_type_of(text: str) -> str:
    for name, pat in _EVENT_TYPE:
        if re.search(pat, text):
            return name
    return "日常"


def infer_owner_state(event_text: str, mood_label: str = "平静",
                      mood_db: str = None, memories: list[str] | None = None) -> dict:
    """★ ToM：推断主人当前状态（情绪 / 需求 / 打扰意愿 / 主动建议）。

    输入：当前事件 + 主人近期情绪(mood_label 或情绪图谱历史) + 记忆。
    输出：{emotion, need, interruptible, advice} —— 接入自激发决策。
    """
    # 主人近期情绪：显式传入 or 情绪图谱查询（哪个实体近期的情绪史）
    emotion = mood_label
    if mood_db and emotion == "平静":
        # 从事件实体查情绪史（考试→低落史 说明主人最近压力大）
        ent = entity_of(event_text)
        hist = query_mood_history(ent, db=mood_db)
        if hist:
            emotion = hist[0]["mood_label"]
    et = event_type_of(event_text)
    need = _NEED_MAP.get(emotion, "顺其自然")
    # 打扰意愿：深夜/专注 → 不打扰；低落+社交 → 鼓励；任务+焦虑 → 提醒
    late = bool(re.search(r"深夜|凌晨|晚上1[0-2]|23点|睡不着|半夜", event_text))
    interruptible = True
    advice = "顺其自然"
    if et == "社交":
        if emotion in ("低落", "焦虑"):
            advice = "鼓励主人去（社交是放松机会），雷姆支持而不是打扰"
        else:
            advice = "主人的事，雷姆不打扰，让他去放松"
            interruptible = False
    elif et == "任务":
        if emotion in ("焦虑", "专注"):
            advice = "提醒主人（他需要确定性），但用简短的确认语气"
        else:
            advice = "温和提醒一句即可"
    elif et == "情绪":
        if emotion in ("低落", "焦虑"):
            advice = "主动关心（他需要陪伴）"
        else:
            advice = "陪主人开心"
    if late:
        interruptible = False
        advice += "（深夜了，不打扰）"
    return {"emotion": emotion, "need": need, "event_type": et,
            "interruptible": interruptible, "advice": advice}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="同学下午5点找我吃饭")
    ap.add_argument("--mood", default="平静")
    args = ap.parse_args()
    print(json.dumps(infer_owner_state(args.event, args.mood), ensure_ascii=False, indent=2))
