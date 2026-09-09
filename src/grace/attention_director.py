#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""attention_director.py — 注意力导向器（2026-09-09，用户洞察："几层记忆系统应用得
好，可以形成非常明确的注意力导向"）。

理论锚: ACT-R —— 注意力与记忆是同一个系统。记忆条目的激活值即注意力：
    激活 = 新近衰减(位置) × 显著度(写入时情绪) × 焦点关联(与当前消息重叠)
      → 本实现取 0.4×新近 + 0.3×显著 + 0.3×关联 的线性近似。

职责: 在 prompt 组装前，把候选上下文（工作记忆条目等）按激活值导向为
【焦点区】【背景区】【标注】三段 —— 图形/背景分明，治注意力稀释
（"？"回声/上下文劫持/身份冲突概率性，全部源于等权竞争）。

纯 stdlib。formal(src/grace/) 与沙盒(v2/engine/) 同源双份，改动需双写。
"""
from __future__ import annotations
import re

_STRIP = re.compile(r"[\s，。,.:：;；!！?？@~～]+")
_TAG = re.compile(r"^\[(主人说|旁听)\]")
_CONFLICT = re.compile(r"陈泽|蕾姆|拉姆|你叫什么|你是谁\b")


def _norm(t: str) -> str:
    return _STRIP.sub("", t or "")


def _overlap(a: str, b: str) -> float:
    """CJK bigram Jaccard 重叠（无依赖的相关度近似）。"""
    na, nb = _norm(a), _norm(b)
    if len(na) < 2 or len(nb) < 2:
        return 0.0
    ga = {na[i:i + 2] for i in range(len(na) - 1)}
    gb = {nb[i:i + 2] for i in range(len(nb) - 1)}
    return len(ga & gb) / len(ga | gb)


def direct(user_text: str, wm_entries: list, focus_hint: str = "") -> dict:
    """导向主函数。

    wm_entries: 工作记忆条目列表，元素为 str（可带 [主人说]/[旁听] 前缀标签）
                或 (text, salience) 元组。
    focus_hint: 可选的焦点提示（如上一轮话题），与 user_text 一起参与关联计算。

    返回 {"focus": [...], "background": [...], "annotations": [...]}，
    focus/background 元素为 {"text","act","rel","tag"}。
    """
    _ut = _norm(user_text)
    scored = []
    n = max(1, len(wm_entries))
    for i, e in enumerate(wm_entries):
        nov = None
        if isinstance(e, dict):                    # ★V2.4 神经记忆: 条目带学出来的 nov
            text, nov, sal = e.get("t", ""), e.get("nov"), None
            tag0 = e.get("tag", "")
        else:
            text, sal = (e if isinstance(e, tuple) else (e, None))
            tag0 = ""
        m = _TAG.match(text or "")
        tag = m.group(1) if m else tag0
        t = _TAG.sub("", text or "").strip()
        if not t:
            continue
        rel = _overlap(t, user_text) if _ut else 0.0
        if focus_hint:
            rel = max(rel, _overlap(t, focus_hint))
        recency = (i + 1) / n                      # 列表位置≈时间顺序，越靠后越新
        if nov is not None:
            sal_term = 0.2 + 0.8 * nov             # 学出来的新奇度替换手设显著度
        else:
            sal_term = 0.35 if sal is None else abs(sal) + 0.2
        act = round(0.4 * recency + 0.3 * sal_term + 0.3 * rel, 3)
        scored.append({"text": t, "act": act, "rel": rel, "tag": tag, "nov": nov})
    scored.sort(key=lambda x: -x["act"])
    # 焦点区: 与当前消息相关（rel≥0.25）或主人亲口说的，取 top3；旁听不进焦点
    focus = [s for s in scored if (s["rel"] >= 0.25 or s["tag"] == "主人说")][:3]
    used = {id(s) for s in focus}
    background = [s for s in scored if id(s) not in used][:8]
    annotations = []
    if _CONFLICT.search(user_text or ""):
        annotations.append("（注意力提示：这句话涉及称呼/名字——你是雷姆，主人的名字是陈泽，"
                           "名字不是你的；以这个事实为准，自然回应）")
    return {"focus": focus, "background": background, "annotations": annotations}


def render_blocks(d: dict, focus_title: str = "你此刻在注意",
                  bg_title: str = "背景·记得但不必展开") -> str:
    """导向结果 → 可注入 prompt 的文本块（空条目自动省略）。"""
    parts = []
    if d["focus"]:
        parts.append(f"【{focus_title}】\n" + "\n".join(f"· {s['text']}" for s in d["focus"]))
    if d["background"]:
        parts.append(f"【{bg_title}】\n" + "\n".join(f"· {s['text']}" for s in d["background"]))
    return "\n".join(parts)
