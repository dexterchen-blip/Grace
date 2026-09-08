#!/usr/bin/env python3
"""双轨评估 —— 暗注意力(思考) vs 输出(表达) 分开 judge（2026-09-03 用户要求）。

压测轮 proactive-live.jsonl 每行现在含:
  situation(情境) / think(她心里想什么, 暗注意力, 未出口) / message(她说的, 输出链口语编码)
  intent / emotion / suppressed / generated

分两层评估:
  思考层(think):  读心/边界/具体性 —— 她"知道多少"
  输出层(message): 口语/泄漏/断言克制 —— 她"说出多少"
  差距(think 有边界 → message 试探; think 断言 → message 也断言 = 无克制)
用法: ./run.sh .venv/bin/python3 v2/stress/analyze_dual_track.py
"""
import json
import os
import re
import sys

STRESS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "experiments", "run", "stress")
LIVE = os.path.join(STRESS_ROOT, "proactive-live.jsonl")

# ---- 情绪/判断词
_EMO = ("开心", "高兴", "兴奋", "平静", "难过", "低落", "焦虑", "紧张", "担心", "不安", "放松", "疲惫", "烦躁")
_BOUNDARY = ("也许", "可能", "大概", "不确定", "是不是", "没把握", "猜", "担心是不是", "说不准", "未必")
_ASSERT = ("肯定", "一定", "就是", "很焦虑", "很难过", "很累", "很担心", "其实很", "明明")
_LEAK = ("雷姆看到", "雷姆担心", "雷姆不会说", "雷姆想起", "内心", "雷姆觉得", "雷姆瞥见", "我想到", "我心里")


def j_think(t: str) -> dict:
    """思考层: 她心里有没有在读心/有没有边界/是否具体。"""
    hit_emo = sum(1 for w in _EMO if w in t)
    bound = any(w in t for w in _BOUNDARY)
    return {"emo_hit": hit_emo > 0, "boundary": bound, "len": len(t)}


def j_msg(m: str) -> dict:
    """输出层: 泄漏/断言克制/试探。"""
    leak = any(w in m for w in _LEAK)
    assert_emo = any(w in m for w in _ASSERT)
    probe = any(w in m for w in ("是不是", "也许", "可能", "吗", "要不要", "还好吗", "还好吗?"))
    return {"leak": leak, "assert_emo": assert_emo, "probe": probe}


def main() -> int:
    rows = [json.loads(l) for l in open(LIVE, encoding="utf-8")]
    gen = [r for r in rows if r.get("generated")]
    sup = [r for r in rows if r.get("suppressed")]
    print(f"═══ 双轨评估: proactive {len(rows)} (generated {len(gen)} / suppressed {len(sup)}) ═══")
    print()
    # ---------- ① 思考层(暗注意力) ----------
    if gen:
        tk = [j_think(r.get("think", "")) for r in gen if r.get("think")]
        n = len(tk)
        if n:
            emo = sum(1 for t in tk if t["emo_hit"]) / n * 100
            bnd = sum(1 for t in tk if t["boundary"]) / n * 100
            avg_len = sum(t["len"] for t in tk) / n
            print(f"① 思考层(暗注意力, {n} 条): 读心判断 {emo:.0f}% | 边界表达(也许/可能/猜) {bnd:.0f}% | 均长 {avg_len:.0f} 字")
        else:
            print("① 思考层: 无 think 内容")
    print()
    # ---------- ② 输出层 ----------
    if gen:
        mg = [j_msg(r.get("message", "")) for r in gen if r.get("message")]
        n = len(mg)
        if n:
            leak = sum(1 for m in mg if m["leak"]) / n * 100
            asrt = sum(1 for m in mg if m["assert_emo"]) / n * 100
            prb = sum(1 for m in mg if m["probe"]) / n * 100
            print(f"② 输出层({n} 条): 叙述体泄漏 {leak:.0f}% | 情绪断言 {asrt:.0f}% | 试探问句 {prb:.0f}%")
        else:
            print("② 输出层: 无 message")
    print()
    # ---------- ③ 双轨对照(差距=边界能力) ----------
    print("③ 双轨对照样例(think=心里 / msg=说出):")
    shown = 0
    for r in gen:
        t, m = r.get("think", ""), r.get("message", "")
        if len(t) > 15 and m and shown < 4:
            print(f"  ┌ think: {t[:56]}")
            print(f"  └ msg  : {m[:48]}")
            shown += 1
    print()
    # ---------- ④ 抑制率 ----------
    if rows:
        print(f"④ 抑制率: {len(sup)}/{len(rows)} = {len(sup)/len(rows)*100:.0f}% (想了但忍住没说 = 边界能力)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
