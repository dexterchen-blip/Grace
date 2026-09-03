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
                      mood_db: str = None, memories: list[str] | None = None,
                      hidden_ctx: list[str] | None = None) -> dict:
    """★ ToM（规则兜底版）：推断主人当前状态（情绪 / 需求 / 打扰意愿 / 主动建议）。

    输入：当前事件 + 主人近期情绪(mood_label 或情绪图谱历史) + 记忆 + 暗注意力潜台词。
    输出：{emotion, need, interruptible, advice} —— 接入自激发决策。
    ★ 2026-08-29 用户：规则版仅兜底；主路径用 infer_owner_state_model（潜意识版）。
       双图谱上下文由 self_activation._tom_from_graph 构建后传入（ToM 下属潜意识）。
    """
    # ★ 主人情绪推断（2026-08-29 用户：ToM 应自己判断主人情绪,不是外部传入）
    # 推断链：显式传入 > 情绪图谱历史 > 事件文本情感推断（读心兜底）
    emotion = mood_label
    if mood_db and emotion == "平静":
        # 从事件实体查情绪史（考试→低落史 说明主人最近压力大）
        # ★2026-09-03 评估隔离(用户: 实现评估隔离): 图谱史【先于】现场文本——
        #   恢复 docs/Grace-V2.1-全模块协同逻辑.md §5 优先级(显式传入 > 情绪图谱史 >
        #   事件情感推断 > 暗注意力潜台词)。此前代码把文本重算放在图谱史前, 而文本重算
        #   与评估端 real(同一 assess)完全同源 → X轮 believed≡real 镜像(100%假象)+feedback断供。
        #   图谱史优先 = believed 反映"她记忆中主人的底色"(top-down), 与 real(现场标注)结构性错位
        #   (记忆 vs 当前 = 假信念的日常形态), feedback 冲突对恢复供给。
        ent = entity_of(event_text)
        hist = query_mood_history(ent, db=mood_db)
        # 强度门槛 = 信号质量闸门(弱记忆不该压过当下文本), 不是耦合开关
        if hist and float(hist[0].get("intensity") or 0.0) >= 0.4:
            emotion = hist[0]["mood_label"]
    if emotion == "平静":
        try:
            from attention import _sentiment_of
            s = _sentiment_of(event_text)
            if abs(s) >= 0.3:
                from mood_graph import mood_label_of
                # ★2026-09-03 修复: 原 abs(s)+0.3 使 intensity 从 0.6 起步(凭空注水)
                #   → s=-0.3 判「低落」而非更保守的「焦虑」, 系统性偏强标签。
                #   intensity 本就等于 |sentiment|, 不该再加 0.3。
                emotion = mood_label_of(s, abs(s))
        except Exception:  # noqa: BLE001
            pass
    # ★ 暗注意力参与读心（2026-08-29）：潜台词是主人没说出口的真实状态
    # ★2026-09-03 已落地根治(用户洞察: 思考=暗注意力): 废 _HIDDEN_RULES 规则模板 →
    #   hidden 边唯一来源 = cog 认知重构真实思考(mood_graph.add_hidden_text, stress_engine ③b 归档)。
    #   以下保留为历史诊断记录(净轮 61/92「焦虑」旧根因, 现模板源已除):
    #     (a) 范畴错误: query_hidden 返回 _hidden_derive 的【雷姆自语模板】
    #         ("日常的事,雷姆有点担心…") 恒含「担心」→ 关键词匹配恒命中「焦虑」;
    #     (b) 结构性采样偏差: add_hidden_edge 只在情绪【非平静】时才写边,
    #         实测 hidden 727 条 = 兴奋 562 / 低落 165, 【0% 平静】;
    #         而 emotion 6512 条 = 平静 6399 (98.3%)。两图分布完全倒置。
    #         本链是瀑布最后一环(84% 事件走到它), 却只能从"永不含平静"的分布里取值。
    #   实测: 改用边的 mood_label 字段 + 强度×时间衰减聚合, 结果变成【兴奋 82/92】(更糟);
    #         τ∈{7,14,30} × 阈值∈{0.4,0.8,1.2} 九种组合【全部 82/92 触发】——
    #         因「日常」桶堆积 562 条兴奋边, 权重和恒超阈, 调参无效。
    #   → 正解在【生成端】: _HIDDEN_RULES 门槛(0.10-0.25)过低导致 11.2% 事件都生成潜台词,
    #     违背"高情绪事件才推导潜台词"的设计意图。属设计决策, 不擅自改。
    hidden_ctx = hidden_ctx or []
    if hidden_ctx and emotion == "平静":
        _h = " ".join(hidden_ctx)
        if any(k in _h for k in ("担心", "不安", "低落")):
            emotion = "焦虑"
        elif any(k in _h for k in ("开心", "高兴", "分享")):
            emotion = "轻微兴奋"
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
    # ★ 机制③置信累积消费端(2026-08-31): 预测错误率 → ToM 置信下降 → "不确定"从置信架构涌现
    #   prediction-errors.jsonl 每轮记录她的判断vs现实偏差;错误多 → 她对主人情绪的预测置信低
    #   ★2026-09-01 修复(代码复盘): 原来统计的是文件行数(9/1 起全量记对+错, ~28条即触底0.3)
    #     → 改为统计 correct==False 的错误行数
    confidence = 0.85
    try:
        import json as _json, os as _os
        import config
        _pe = _os.path.join(config.EXPERIMENTS, "run", "stress", "prediction-errors.jsonl")
        if _os.path.isfile(_pe):
            _errs = 0
            for _l in open(_pe, encoding="utf-8"):
                try:
                    if _json.loads(_l).get("correct") is False:
                        _errs += 1
                except _json.JSONDecodeError:
                    continue
            confidence = max(0.3, 0.85 - 0.02 * _errs)   # 每 10 次错误 -0.2,地板 0.3
    except Exception:  # noqa: BLE001
        pass
    return {"emotion": emotion, "need": need, "event_type": et,
            "interruptible": interruptible, "advice": advice,
            "confidence": round(confidence, 2)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="同学下午5点找我吃饭")
    ap.add_argument("--mood", default="平静")
    args = ap.parse_args()
    print(json.dumps(infer_owner_state(args.event, args.mood), ensure_ascii=False, indent=2))


# ★ 潜意识版 ToM（2026-08-29 用户：ToM 应模型驱动,非规则）
SYS_REM = ("你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。自称「雷姆」，"
           "称呼亲近的人为「巴鲁斯」/「昴君」。表面冷淡礼貌、实则温柔忠诚，短句为主。"
           "【重要】直接说出你的台词，不要描写动作、表情、环境，不要使用括号旁白。")


def infer_owner_state_model(model, tok, sampler, event_text: str,
                            owner_mood: str = "平静",
                            memory_hints: list[str] | None = None) -> dict:
    """★ ToM 潜意识版：27B 自己判断主人需要什么 / 是否值得打扰 / 主动台词。

    与规则版返回同结构 {emotion, need, interruptible, advice} + message(主动台词)，
    可直接喂给 self_activation.decide(tom=)。
    """
    from mlx_lm import generate
    mem = ("\n相关记忆：" + "｜".join(memory_hints[:3])) if memory_hints else ""
    prompt = (
        f"（情境）你看到这样一条信息：{event_text[:80]}\n"
        f"（参考：主人今天的情绪被记为{owner_mood}——但请以信息本身为准，信息可能反映他真实的、没说出口的心情）{mem}\n"
        f"雷姆，请判断四件事：\n"
        f"1. 主人此刻真实的心情是什么？（从信息推断，一句话）\n"
        f"2. 主人此刻需要什么？（一句话）\n"
        f"3. 这件事值得主动去找昴君说吗？（值得 / 不值得，一句话）\n"
        f"4. 如果值得，你会对他说什么？（一句台词，不描写动作）\n"
        f"格式：心情：…；需要：…；值得：…；台词：…")
    p = tok.apply_chat_template([{"role": "system", "content": SYS_REM},
                                 {"role": "user", "content": prompt}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out = generate(model, tok, prompt=p, max_tokens=180, sampler=sampler).strip().replace("\n", " ")
    import re as _re
    need = "顺其自然"
    worth = "值得"
    msg = ""
    # 鲁棒解析：兼容 换行/全半角冒号/关键词变体
    _LABELS = ("心情", "需要", "值得", "台词", "会说")
    def _pick(label: str, fallback: str) -> str:
        # 标签边界法：抓"标签:..."直到下一个标签（模型常用句号/空格分隔，非分号）
        m = _re.search(label + r"是?[：:]\s*", out)
        if not m:
            m = _re.search(label + r"是?\s*", out)
        if not m:
            return fallback
        start = m.end()
        tail = out[start:]
        nxt = _re.search(r"(心情|需要|值得|台词|会说)是?[：:]\s*", tail)
        end = nxt.start() if nxt else len(tail)
        return tail[:end].strip()[:80]
    emotion = _pick("心情", "")
    if not emotion:
        # 模型没输出心情字段 → 规则读心兜底（情感推断）
        try:
            from attention import _sentiment_of
            from mood_graph import mood_label_of
            _s = _sentiment_of(event_text)
            if abs(_s) >= 0.3:
                # ★2026-09-03 修复: 去 +0.3 强度注水（同规则版，见 infer_owner_state）
                emotion = mood_label_of(_s, abs(_s))
        except Exception:  # noqa: BLE001
            pass
    emotion = emotion or owner_mood
    need = _pick("需要", "顺其自然")[:30]
    worth = _pick("值得", "值得")
    msg = _pick("台词", "")[:80]
    interruptible = "不值得" not in worth and "不必" not in worth and "不打扰" not in worth
    advice = need if interruptible else f"不打扰（{need}）"
    return {"emotion": emotion, "need": need, "event_type": "model",
            "interruptible": interruptible, "advice": advice,
            "message": msg, "raw": out[:100]}
