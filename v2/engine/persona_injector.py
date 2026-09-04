#!/usr/bin/env python3
"""心态轨 + 人格轨注入器 —— M3 推理时给 27B 加 system 前缀（Grace_v2 设计 §8.2）。

两路消费：
  ① 心态注入：从 l2.db mood_states 读当日心态 → 显式文案前缀（"今早你轻微兴奋，因为…"）
  ② 人格注入：从 persona/<name>.md 提炼人设段（雷姆：自称/称呼/语气）
  ③ build_v2_system：把人格段 + 心态段拼到基础 system 后

铁律：
  - 心态**只调语气强度，不改价值观立场**（难过时话少，但不能说反话）
  - 心态只读 approved 状态（人审闸门放行过的）
  - L3 高风险自发永不受心态影响（本模块不管自发，只管语气）

用法（沙盒内）:
  ./run.sh python3 v2/engine/persona_injector.py            # 打印当前注入段
  ./run.sh python3 v2/engine/persona_injector.py --mood 兴奋 0.8 "跑分通过"   # 手动造心态并打印
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
from mood_engine import latest as mood_latest  # noqa: E402（同目录模块）

# 心态 → 语气强度调节（只调频率/长度，不调立场）
_MOOD_TONE = {
    "平静":     {"prefix": "今天你的心情平静如常。", "energy": 0.5, "proactive_bias": 0.0},
    "轻微兴奋": {"prefix": "今早你有一点轻微兴奋。", "energy": 0.6, "proactive_bias": 0.2},
    "兴奋":     {"prefix": "你今天心情很好，有些兴奋。", "energy": 0.8, "proactive_bias": 0.4},
    "低落":     {"prefix": "今天你情绪有些低落。", "energy": 0.3, "proactive_bias": -0.3},
    "焦虑":     {"prefix": "你今天有些焦虑。", "energy": 0.4, "proactive_bias": -0.2},
    "专注":     {"prefix": "今天你非常专注。", "energy": 0.7, "proactive_bias": 0.1},
}


def mood_prefix(mood: dict | None = None, db: str | None = None, now: float | None = None,
                combined: bool = True) -> str:
    """心态注入文案：三层情绪融合（人格底色×长期趋势×短期日内）优先；无日内事件回退日级。

    combined=True（默认）时输出完整三层文案：
      "现在你有些低落（刚才：考砸了…）。这周整体情绪平稳，雷姆的人格底色是外冷内热——"
      "注意：这只影响你的语气和主动性，绝不改变你的价值观与立场。"
    db/now 可指定心态库与时刻（测试用临时库/模拟时间）。
    """
    if combined:
        try:
            from mood_engine import combined_emotion
            ce = combined_emotion(db=db, now=now)
            if ce.get("intraday") is not None and ce.get("intraday_trigger"):
                label = ce["label"]
                cause = ce["intraday_trigger"][:40]
                line = f"现在你{_present_phrase(label)}。"
                if cause:
                    line += f"（刚才：{cause}…）"
                if ce["trend_direction"] != "平稳" or abs(ce["combined"] - ce["anchor"]) > 0.05:
                    line += f"这一周整体情绪{ce['trend_direction']}，长期底色{ce['baseline']:.2f}。"
                line += "注意：这只影响你的语气和主动性，绝不改变你的价值观与立场。"
                return line
            # 无日内事件 → 日级 + 长期
            if ce["trend_direction"] != "平稳":
                return (f"今天你{_present_phrase(ce['label'])}。"
                        f"这一周整体情绪{ce['trend_direction']}（长期锚点{ce['anchor']:.2f}）。"
                        f"注意：这只影响你的语气和主动性，绝不改变你的价值观与立场。")
        except Exception:  # noqa: BLE001
            pass
    # 兜底：日级
    intra = None
    try:
        from mood_engine import current_intraday
        intra = current_intraday(db=db, now=now)
    except Exception:  # noqa: BLE001
        intra = None
    if intra and intra.get("event_text"):
        label = intra.get("mood_label", "平静")
        cause = intra.get("event_text", "")[:40]
        line = f"现在你{_present_phrase(label)}。"
        if cause:
            line += f"（刚才：{cause}…）"
        line += "注意：这只影响你的语气和主动性，绝不改变你的价值观与立场。"
        return line
    mood = mood or mood_latest()
    if not mood:
        return ""
    label = mood.get("mood_label", "平静")
    tone = _MOOD_TONE.get(label, _MOOD_TONE["平静"])
    triggers = mood.get("triggers", "")
    try:
        trig = json.loads(triggers) if triggers else []
    except json.JSONDecodeError:
        trig = []
    cause = trig[0][:40] if trig else ""
    line = f"{tone['prefix']}"
    if cause:
        line += f"（触发：{cause}…）"
    line += "注意：这只影响你的语气和主动性，绝不改变你的价值观与立场。"
    return line


def _present_phrase(label: str) -> str:
    """日内文案的当下措辞。"""
    return {
        "平静": "心情平静",
        "轻微兴奋": "有点兴奋",
        "兴奋": "心情很好",
        "低落": "有些低落",
        "焦虑": "有点焦虑",
        "专注": "很专注",
    }.get(label, "心情" + label)


def persona_prefix(persona: str | None = None) -> str:
    """从 persona/<name>.md 提炼人设段（一行）。默认雷姆。"""
    persona = persona or config.PERSONA["name"]
    p = os.path.join(config.PERSONA["anchor_file"]) if persona == config.PERSONA["name"] else None
    if not p or not os.path.isfile(p):
        return ""
    # 从 rem.md 提炼：自称/称呼/语气一句话
    # ★2026-09-04 裸 prompt 对照实验瘦身(实验证据: AB persona 版自称漂移"妹妹我"=prompt 里
    #   "拉姆的妹妹"触发; OOC"不会英语"=异世界女仆设定被模型强化; 裸版自称稳且 UCSB 知识正确):
    #   ①删"鬼族,拉姆的妹妹"(自称漂移源——模型把"妹妹"演绎成自称"妹妹我")
    #   ②删称呼细节(巴鲁斯/昴君/姐姐大人)——对话场景少用, 反而诱导错位称呼
    #   ③加"能协助主人处理现实事务(英文邮件/学校系统等)"——明示能力范围, 防"不会英语"式 OOC
    #   ④保留: 自称「雷姆」(锚) + 说话形态(短句/口语/当面说话不念旁白/黑色幽默)——这是 persona
    #      版唯一显著优于裸版处(裸版满屏小说旁白体, persona 规范压住)。语气风格最终应由权重学。
    return ("你是雷姆。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
            "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。"
            "带黑色幽默与毒舌吐槽的一面。她能协助主人处理现实事务（英文邮件、学校系统等）。"
            "事实问题请参考检索到的记忆回答。")


def build_v2_system(base_system: str, persona: str | None = None) -> str:
    """把人格段 + 心态段拼到基础 system 后（M3 注入点）。"""
    parts = [base_system]
    pp = persona_prefix(persona)
    if pp:
        parts.append(pp)
    mp = mood_prefix()
    if mp:
        parts.append(f"【今日心态】{mp}")
    return "\n\n".join(p for p in parts if p)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--mood":
        # 手动造心态测试注入文案
        label, intensity = sys.argv[2], float(sys.argv[3])
        cause = sys.argv[4] if len(sys.argv) > 4 else ""
        fake = {"mood_label": label, "intensity": intensity, "triggers": json.dumps([cause], ensure_ascii=False)}
        print("心态注入:", mood_prefix(fake))
        print("人设段 :", persona_prefix())
    else:
        print("=== 当前心态注入段 ===")
        print(mood_prefix() or "（mood_states 无心态记录）")
        print()
        print("=== 人设段 ===")
        print(persona_prefix())
        print()
        print("=== 完整 v2 system 追加（示例基础 system）===")
        print(build_v2_system("你是本地 AI 助手，负责处理用户的日程、邮件与记忆。")[:600])
