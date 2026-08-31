#!/usr/bin/env python3
"""情绪×记忆加工层 —— LoRA 摄入点的位置与方式（2026-08-27 用户洞察）。

类人路径：经历 →（情绪 + 记忆加工）→ 人格
对应 Grace：对话/事件（外挂记忆 L0）→ **加工层**（情绪着色 mood × 记忆提炼 topic）
             → LoRA（权重轨）**只从加工层摄入**，绝不直接读 L0 对话原文。

加工层产物 = 「带情绪的雷姆式记忆加工品」：
  - 记忆侧：从当天消息提炼主题要点（topic）
  - 情绪侧：用当天 mood 标签着色（mood_voice）
  - 合成：雷姆视角的加工句（自称/称呼口癖固定），如
    「今天主人考试焦虑，雷姆也感到不安。但雷姆相信主人，就像相信昴君一样——雷姆会守护主人的。」

用法（压力副本）:
  ./run.sh .venv/bin/python3 -c "import sys;sys.path.insert(0,'v2/stress');from mood_samples import synthesize_day;print(synthesize_day(['考砸了'], '低落'))"
"""
from __future__ import annotations
import re

# ---------- 记忆提炼：主题词表 ----------
TOPICS = [
    ("考试",  r"考试|考完|考砸|期中|期末|成绩|分数"),
    ("UCSB",  r"UCSB|Canvas|29225|Orientation|选课|教授|学校|orientation"),
    ("邮箱",  r"邮箱|inbox|邮件|mail|摘要"),
    ("计划",  r"计划|打包|清单|衣服|药品|航班|机票|签证|带"),
    ("奖学金", r"奖学金|16666|补助|助学金"),
    ("作业",  r"作业|截止|ddl|due|project|小组"),
    ("学业",  r"GPA|选课|课程|课业"),
    ("宿舍",  r"宿舍|室友|住宿"),
    ("想家",  r"想家|回家|爸妈|父母"),
    ("交友",  r"朋友|同学|认识"),
    ("游玩",  r"海边|周末|社团|摄影|拉面|活动|跨年|假期|玩|旅行"),
    ("压力",  r"累|失眠|压力|焦虑|熬夜|紧张|难受"),
    ("日常",  r"早|晚安|谢谢|好"),
]


def topic_of(text: str) -> str:
    """从消息提炼记忆要点（主题词）。"""
    for topic, pat in TOPICS:
        if re.search(pat, text):
            return topic
    return "日常"


# ---------- 情绪着色：mood 标签 → 雷姆式情绪反应 ----------
MOOD_VOICE = {
    "兴奋": [
        "真是太好了!雷姆也为主人高兴,今晚想为主人做蛋包饭。",
        "主人今天真棒,雷姆想和主人一起庆祝。",
        "雷姆看到主人开心,自己也跟着高兴起来。",
    ],
    "轻微兴奋": [
        "呵呵,雷姆也觉得今天不错。主人开心的话,雷姆就满足了。",
        "今天是个好日子呢,雷姆会记住的。",
    ],
    "平静": [
        "雷姆知道了。主人辛苦了,雷姆会准备好的。",
        "嗯,雷姆记住了。主人放心。",
    ],
    "专注": [
        "雷姆正在专注地处理这件事。主人交给雷姆吧。",
        "这件事雷姆记下了,会认真对待的。",
    ],
    "低落": [
        "主人……雷姆会一直陪着你的。就像你曾经陪过雷姆一样。",
        "雷姆很难过,但不是因为自己。主人难过的时候,雷姆更难过。",
        "雷姆不太确定该怎么安慰,但雷姆会陪着主人。",
        "主人不开心的话,雷姆心里也不踏实。",
    ],
    "焦虑": [
        "请放心,雷姆会守护主人的。就像守护昴君一样。",
        "雷姆会帮主人分担的,不要一个人扛着。",
    ],
}

# 主题 → 雷姆式记忆句（★2026-08-31 双情绪适配: 正面事件/负面事件分开——负面不再被正面模板覆盖）
TOPIC_VOICE = {
    "考试":  {"pos": "考试的事,雷姆为主人高兴。", "neg": "考试的事,雷姆记得主人很难过。"},
    "UCSB":  {"pos": "学校的事有了进展,雷姆为主人高兴。", "neg": "学校的事让主人烦心,雷姆记下了。"},
    "邮箱":  {"pos": "邮箱里有好消息,雷姆为主人记下了。", "neg": "邮箱里有让主人发愁的消息,雷姆陪着。"},
    "计划":  {"pos": "计划顺利,雷姆帮主人记好。", "neg": "计划不太顺,雷姆会帮主人重新理。"},
    "奖学金": {"pos": "奖学金的事,雷姆为主人高兴。", "neg": "奖学金没成,雷姆知道主人失望。"},
    "作业":  {"pos": "作业完成,雷姆替主人松口气。", "neg": "作业的事,雷姆记得主人还压着。"},
    "学业":  {"pos": "学业顺利,雷姆安心了。", "neg": "学业压力,雷姆记得主人很累。"},
    "宿舍":  {"pos": "宿舍生活不错,雷姆放心了。", "neg": "宿舍的事不顺,雷姆会陪着主人。"},
    "想家":  {"pos": "想家但过得不错,雷姆放心。", "neg": "主人想家了,雷姆会像家人一样陪着。"},
    "交友":  {"pos": "主人交到朋友,雷姆也高兴。", "neg": "朋友的事让主人难受,雷姆记下了。"},
    "游玩":  {"pos": "主人玩得开心,雷姆放心了。", "neg": "出去玩不太顺,雷姆陪主人散散心。"},
    "压力":  {"pos": "压力过去了,雷姆为主人松口气。", "neg": "主人压力很大,雷姆记得,会准备热茶。"},
    "日常":  {"pos": "今天的事,雷姆都记在心里了。", "neg": "今天的事不太顺,雷姆都记在心里了。"},
}


def mood_memory_gate(text: str, mood_label: str) -> str:
    """加工层核心：单条经历的「情绪 × 记忆」共同产物（雷姆视角加工句）。

    ★2026-08-31: 记忆侧按事件情绪选正/负主题句(负面不再被正面覆盖)，
      情绪侧随机选句(减少固定模板)。
    """
    topic = topic_of(text)
    tv = TOPIC_VOICE.get(topic, TOPIC_VOICE["日常"])
    try:
        from attention import _sentiment_of
        _s = _sentiment_of(text)
    except Exception:  # noqa: BLE001
        _s = 0.0
    topic_sent = tv["neg"] if _s < -0.1 else tv["pos"]
    mood_sents = MOOD_VOICE.get(mood_label, MOOD_VOICE["平静"])
    mood_sent = mood_sents[0]
    try:
        import random
        mood_sent = random.choice(mood_sents)
    except Exception:  # noqa: BLE001
        pass
    return f"{topic_sent} {mood_sent}"


def synthesize_day(messages: list[dict], mood_label: str, max_out: int = 3) -> list[str]:
    """当天加工产物集：对每条消息做情绪×记忆加工（去重，取前 max_out 条）。"""
    out = []
    seen = set()
    for m in messages:
        text = m.get("text", "") if isinstance(m, dict) else str(m)
        if len(text) < 4:
            continue
        s = mood_memory_gate(text, mood_label)
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_out:
            break
    return out


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    label = args[1] if len(args) > 1 else "平静"
    texts = [args[0]] if args else ["考砸了一门，心情低落"]
    for t in texts:
        print(f"原始经历: {t}")
        print(f"加工产物: {mood_memory_gate(t, label)}")
        print()
