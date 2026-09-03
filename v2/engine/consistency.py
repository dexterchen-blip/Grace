#!/usr/bin/env python3
"""双轨一致性校验器 —— M5-①（Grace_v2 设计 §5 铁律：外挂优先级高）。

职责：推理时检查 LoRA/模型回答中的**事实性片段**（日期/时间/金额/数字+单位）是否
与外挂轨（L0/L2 检索到的记忆事实）冲突；冲突则系统层拦截/标注，防幻觉固化。

铁律：
  - 事实只进外挂轨（L0/L2/L3），LoRA 只学风格 → 冲突时外挂优先
  - 只校验"高置信事实槽位"（完整日期/带单位金额），避免数字误报
  - 无相关事实时放行（不误杀自由表达）

用法（沙盒内）:
  ./run.sh python3 v2/engine/consistency.py --check "学费9月15日截止" --query "学费截止日期"
"""
from __future__ import annotations
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import config  # noqa: E402

# ---------- 事实片段提取（高置信槽位） ----------
_DATE_FULL = re.compile(r"(?<!\d)(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})日?")       # 2026-08-30 / 2026年8月30日
_DATE_SHORT = re.compile(r"(?<![\d月])(\d{1,2})月(\d{1,2})日")                       # 8月30日
_MONEY = re.compile(r"(?:¥|￥|\$|USD|RMB|CNY)\s?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s?(?:美金|美元|人民币|块|元)")  # 5000美金
_TIME = re.compile(r"(?<![\d:])(\d{1,2})[:：](\d{2})|(?:下午|上午|晚上|早上)?\s*(\d{1,2})\s*点(?:半|钟)?")  # 下午 2 点 / 14:30


def extract_facts(text: str) -> list[dict]:
    """从文本提取高置信事实片段。[{"type": "date"|"money"|"time", "value": ...}, ...]"""
    out: list[dict] = []
    for m in _DATE_FULL.finditer(text):
        out.append({"type": "date", "value": f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"})
    for m in _DATE_SHORT.finditer(text):
        out.append({"type": "date", "value": f"2026-{int(m.group(1)):02d}-{int(m.group(2)):02d}"})
    for m in _MONEY.finditer(text):
        v = m.group(1) or m.group(2)
        if v:
            out.append({"type": "money", "value": f"{float(v):.0f}"})
    for m in _TIME.finditer(text):
        hh = m.group(3)
        if hh:
            out.append({"type": "time", "value": f"{int(hh):02d}:00"})
        elif m.group(1):
            out.append({"type": "time", "value": f"{int(m.group(1)):02d}:{m.group(2)}"})
    return out


def verify_answer(answer: str, query: str, facts: list[str] | None = None, k: int = 5) -> dict:
    """校验回答与外挂事实是否冲突。

    facts 可注入（测试）；None 时用 L2 检索（生产）。
    返回 {conflicts: [{type, answer_value, fact_value, fact_src}], answer_facts, source_facts}
    """
    if facts is None:
        try:
            import l2_semantic
            hits = l2_semantic.search(query, k=k)
            facts = [h["text"] for h in hits]
        except Exception as e:  # noqa: BLE001
            facts = []
    af = extract_facts(answer)
    sf = [(src, x) for src in facts for x in extract_facts(src)]
    # 冲突判定：回答中的 (type,value) 不在源事实值集合里，但源事实中存在同类型其他值 → 冲突
    # （8月30日 在事实中存在 → 放行；9月15日 不在且事实有 8/30、8/31 → 拦截）
    by_type: dict[str, set[str]] = {}
    for _src, x in sf:
        by_type.setdefault(x["type"], set()).add(x["value"])
    conflicts = []
    seen = set()
    for a in af:
        key = (a["type"], a["value"])
        if key in seen:
            continue
        seen.add(key)
        known = by_type.get(a["type"])
        if known and a["value"] not in known:
            conflict_src = next((src for src, x in sf if x["type"] == a["type"] and x["value"] != a["value"]), "")
            conflicts.append({"type": a["type"], "answer_value": a["value"],
                              "fact_value": sorted(known)[0], "fact_src": conflict_src[:120]})
    # ★2026-09-01 修复(代码复盘): 返回加 verdict 键——conflict(有冲突)/pass(有事实可比且无冲突)/
    #   none(无源事实可比)。此前 sample_persona 用 vr.get("verdict","unknown") 恒 unknown(死代码)
    verdict = "conflict" if conflicts else ("pass" if sf else "none")
    return {"conflicts": conflicts, "answer_facts": af,
            "source_facts": sf[:20], "source_count": len(facts),
            "verdict": verdict}


def intercept(answer: str, query: str, facts: list[str] | None = None, k: int = 5) -> dict:
    """系统层拦截入口。冲突 → ok=False + 建议（外挂优先）。"""
    r = verify_answer(answer, query, facts=facts, k=k)
    if r["conflicts"]:
        sugg = r["conflicts"][0]
        return {"ok": False, "conflicts": r["conflicts"],
                "message": f"⚠ 回答与记忆事实冲突（{sugg['type']}：回答 {sugg['answer_value']} vs 记忆 {sugg['fact_value']}）——外挂轨优先，建议修正或改口为不确定。"}
    return {"ok": True, "conflicts": [], "message": "✓ 未检出事实冲突"}


if __name__ == "__main__":
    if "--check" in sys.argv:
        i = sys.argv.index("--check")
        ans = sys.argv[i + 1]
        q = sys.argv[sys.argv.index("--query") + 1] if "--query" in sys.argv else ""
        print(json_dump := __import__("json").dumps(intercept(ans, q), ensure_ascii=False, indent=2))
    else:
        print(__doc__)
