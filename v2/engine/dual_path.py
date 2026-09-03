#!/usr/bin/env python3
"""快慢双路径路由器 —— §15 类人双系统记忆（2026-08-27 用户洞察工程化）。

系统1（快）＝权重轨 LoRA：寒暄/主观/人格问题 → 秒答（脱口而出，潜意识）。
系统2（慢）＝外挂轨检索：事实/日期/日程问题 → "想一下"：L2 检索 → 注入参考事实 → 生成 → 一致性校验。
回忆失败（检索为空）→ 诚实回答"记不清"，绝不幻觉编造（防幻觉第一道闸）。

用法（沙盒内）:
  ./run.sh python3 v2/engine/dual_path.py --route "学费什么时候截止"     # slow
  ./run.sh python3 v2/engine/dual_path.py --route "你是谁呀"            # fast
  ./run.sh python3 v2/engine/dual_path.py --demo                        # 模拟问答演示
"""
from __future__ import annotations
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import config  # noqa: E402

# 快路径信号（人格/寒暄/主观/当下感受）
_FAST = re.compile(
    r"(?:你是谁|你叫什么|你的名字|你喜欢|你讨厌|爱(?:你|我)|想你|"
    r"心情|感觉|累不累|晚安|早安|早上好|你好|谢谢|对不起|"
    r"夸|聪明|可爱|漂亮|巴鲁斯|昴君|姐姐大人|女仆|鬼族|魔法)"
)
# 慢路径强信号（事实/检索词，不会出现在人格问题里）
_SLOW_STRONG = re.compile(
    r"(?:什么时候|几点|几号|多少钱|多少|截止|deadline|due|学费|GPA|成绩|邮件|日程|会议|预约|"
    r"航班|车次|日期|时间|金额|"
    r"记得吗|还记得|发生过|当时|地址|电话|密码|账号|签证|I-20|作业|课程|考试|"
    r"哪(?:天|里)|为什么)"
)
# 慢路径弱信号（中性词：时间词/问法，默认 slow 兜底）
_SLOW_WEAK = re.compile(r"(?:是谁|怎么样|怎么办|哪里|今天|明天|昨天|下周|上周)")


def classify_question(q: str) -> dict:
    """路由：fast（人格秒答/潜意识）| slow（回忆检索）。三级判定。"""
    if _SLOW_STRONG.search(q):
        return {"path": "slow", "reason": "强事实词，走回忆路径（想一下）"}
    if _FAST.search(q):
        return {"path": "fast", "reason": "人格/寒暄/主观，走潜意识秒答路径"}
    if _SLOW_WEAK.search(q):
        return {"path": "slow", "reason": "弱事实词，走回忆路径"}
    return {"path": "slow", "reason": "默认走慢路径（宁可检索，不冒险编造）"}


def answer_slow(query: str, search_fn=None, generate_fn=None, verify_fn=None,
                k: int = 5, persona_ctx: str = "") -> dict:
    """慢路径：检索（想一下）→ 注入事实 → 生成 → 一致性校验。

    search_fn(query) -> list[str]（检索到的记忆文本）；None 时用 l2_semantic。
    generate_fn(prompt) -> str；None 时返回"生成钩子未注入"。
    回忆失败（检索空）→ 诚实回答（不幻觉）。
    """
    if search_fn is None:
        try:
            import l2_semantic
            def _s(q):
                return [h["text"] for h in l2_semantic.search(q, k=k)]
            search_fn = _s
        except Exception:  # noqa: BLE001
            search_fn = lambda q: []   # noqa: E731

    facts = search_fn(query)
    if not facts:
        # 回忆失败 → 诚实，不编造
        return {"path": "slow", "retrieved": 0, "recall": "fail",
                "answer": "（回忆失败·诚实）这个……雷姆记不太清了。可以告诉雷姆更多细节吗？",
                "conflicts": []}
    prompt = f"{persona_ctx or '你是雷姆。'}\n\n参考记忆（请以这些为准）：\n" + "\n".join(f"- {f[:200]}" for f in facts) + \
             f"\n\n问题：{query}\n请结合参考记忆回答："
    if generate_fn is None:
        return {"path": "slow", "retrieved": len(facts), "recall": "ok",
                "prompt": prompt, "answer": None, "conflicts": [],
                "note": "generate_fn 未注入（集成时传真实生成器）"}
    answer = generate_fn(prompt)
    conflicts = []
    if verify_fn:
        vr = verify_fn(answer, query, facts=facts)
        conflicts = vr.get("conflicts", [])
        if conflicts:
            answer += "（⚠ 与记忆冲突，以上为准：外挂轨优先）"
    return {"path": "slow", "retrieved": len(facts), "recall": "ok",
            "answer": answer, "conflicts": conflicts, "facts": facts[:3]}


def answer_fast(query: str, generate_fn=None, persona_ctx: str = "") -> dict:
    """快路径：人格秒答（不检索，潜意识脱口而出）。"""
    prompt = f"{persona_ctx or '你是雷姆。'}\n\n（直接以你的性格回答，无需检索记忆）\n{query}"
    if generate_fn is None:
        return {"path": "fast", "answer": None, "prompt": prompt,
                "note": "generate_fn 未注入（集成时传真实生成器）"}
    return {"path": "fast", "answer": generate_fn(prompt)}


def route(query: str, search_fn=None, generate_fn=None, verify_fn=None,
          persona_ctx: str = "") -> dict:
    """入口：路由 + 执行。集成点：dashboard 对话 / 自发引擎。

    ★2026-09-01 修复(代码复盘): 原实现把 cls["reason"](路由原因文本,如"人格/寒暄/主观,走潜意识秒答路径")
      当 persona_ctx 塞进生成 prompt → 路由原因污染生成上下文。改为透传外部 persona_ctx;
      reason 只记录在 r["route"] 供观测。
    """
    cls = classify_question(query)
    if cls["path"] == "fast":
        r = answer_fast(query, generate_fn, persona_ctx=persona_ctx)
    else:
        r = answer_slow(query, search_fn, generate_fn, verify_fn, persona_ctx=persona_ctx)
    r["route"] = cls
    return r


if __name__ == "__main__":
    if "--route" in sys.argv:
        i = sys.argv.index("--route")
        q = sys.argv[i + 1]
        cls = classify_question(q)
        print(f"问题: {q}\n路由: {cls['path']} ｜ {cls['reason']}")
    elif "--demo" in sys.argv:
        for q in ["你是谁呀？", "学费什么时候截止？", "昨天发生了什么？", "你喜欢昴吗？"]:
            cls = classify_question(q)
            print(f"  「{q}」 → {cls['path']}（{cls['reason']}）")
        print()
        print("回忆失败演示（检索为空 → 诚实回答）：")
        r = answer_slow("学费什么时候截止", search_fn=lambda q: [])
        print(f"  {r['answer']}")
    else:
        print(__doc__)
