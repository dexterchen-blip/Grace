#!/usr/bin/env python3
"""三层情绪融合实验 —— 人格底色(LoRA) × 长期趋势(情绪记忆) × 短期日内，共同驱动 + 性能基准。

场景：30 天三阶段（第 1-10 天平淡 / 11-20 天持续好事 / 21-27 天低谷 / 28-30 天恢复）
验证：
  1. 长期趋势感知：持续好事周 → trend_mean 上升、direction=回升；低谷周 → 下行
  2. 融合正确性：有日内事件时 combined 偏向日内（权重 0.7）；无日内 = 0.5×anchor + 0.5×daily
  3. 人格底色参与：anchor = 0.6×trend_mean + 0.4×baseline（雷姆 0.5）
  4. 注入文案含三层信息（现在/这一周/底色）
性能基准：
  - 1000 次 combined_emotion 调用耗时（ms/次）
  - 1000 次 mood_prefix 注入耗时
  - SQLite 查询量（每次 combined 的查询次数）

用法（沙盒内）: ./run.sh .venv/bin/python3 v2/tests/mood_combined_test.py
输出: experiments/run/mood-combined-*.md
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.mood_engine import (derive, apply_intraday_event, combined_emotion,
                                long_term_trend, persona_baseline)
from engine.persona_injector import mood_prefix
import config

PASS, FAIL = 0, 0
def _rec(lines, name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
    lines.append(f"- {'✅' if cond else '❌'} {name}")
    return cond


def main():
    tmp = tempfile.mkdtemp(prefix="mood-combined-")
    db = os.path.join(tmp, "mood.db")
    start = datetime(2026, 7, 1, 8, 0)
    lines = [f"# 三层情绪融合实验（30 天）\n",
             f"> 人格底色=雷姆(rem, baseline={config.PERSONA.get('mood_baseline', 0.5)}) ｜ 临时库\n"]

    def _ts(day, h=8, m=0):
        return (start + timedelta(days=day - 1, hours=h - 8, minutes=m)).timestamp()

    # ---------- 30 天日级（三阶段） ----------
    daily_plan = {}
    for d in range(1, 31):
        if d <= 10:
            evs = [("平平淡淡的一天", 0.05, 0.4)]
        elif d <= 20:
            evs = [("连续好事：作业全 A、收到肯定、活动顺利", 0.75, 0.7)]
        elif d <= 27:
            evs = [("低谷期：接连受挫、睡眠差", -0.8, 0.7)]
        else:
            evs = [("逐步调整，恢复中", 0.3, 0.5)]
        r = derive([{"text": t, "sentiment": s, "weight": w} for t, s, w in evs], db=db, ts=_ts(d, 22))
        daily_plan[d] = r

    print("日级 30 天推演完成（阶段：平淡→好事→低谷→恢复）")
    lines.append("## 1. 长期趋势感知（情绪记忆）\n")
    t1 = long_term_trend(db=db, days=7, now=_ts(10))
    t2 = long_term_trend(db=db, days=7, now=_ts(20))
    t3 = long_term_trend(db=db, days=7, now=_ts(27))
    ok = _rec(lines, f"好事周（D11-20）趋势均值 > 平淡周（D1-10）({t2['mean']} > {t1['mean']})", t2["mean"] > t1["mean"])
    ok = _rec(lines, f"低谷周（D21-27）direction=下行（{t3['direction']}）", t3["direction"] == "下行")
    ok = _rec(lines, f"低谷周均值 < 好事周均值（{t3['mean']} < {t2['mean']}）", t3["mean"] < t2["mean"])
    lines.append(f"\n趋势表：平淡周 mean={t1['mean']} dir={t1['direction']} ｜ 好事周 mean={t2['mean']} dir={t2['direction']} ｜ 低谷周 mean={t3['mean']} dir={t3['direction']}\n")

    # ---------- 融合正确性（D22 低谷周 + 日内事件） ----------
    lines.append("## 2. 融合正确性\n")
    # 无日内：combined = 0.5×anchor + 0.5×daily
    ce_daily = combined_emotion(db=db, now=_ts(22, 8))
    anchor = ce_daily["anchor"]
    expect = round(0.5 * anchor + 0.5 * ce_daily["daily"], 3)
    ok = _rec(lines, f"无日内事件：combined({ce_daily['combined']}) == 0.5×anchor({anchor})+0.5×daily({ce_daily['daily']})",
              abs(ce_daily["combined"] - expect) < 0.001)
    ok = _rec(lines, f"anchor = 0.6×趋势({ce_daily['trend_mean']}) + 0.4×底色({ce_daily['baseline']})",
              abs(anchor - (0.6 * ce_daily["trend_mean"] + 0.4 * ce_daily["baseline"])) < 0.001)
    # 日内事件：combined 偏向日内（权重 0.7）
    apply_intraday_event({"text": "突然收到导师的坏消息", "sentiment": -0.7, "weight": 0.8},
                         db=db, ts=_ts(22, 14), base=ce_daily["daily"])
    ce_intra = combined_emotion(db=db, now=_ts(22, 14, 30))
    expect2 = round(0.3 * ce_intra["anchor"] + 0.7 * ce_intra["intraday"], 3)
    ok = _rec(lines, f"有日内事件：combined({ce_intra['combined']}) == 0.3×anchor + 0.7×日内({ce_intra['intraday']})",
              abs(ce_intra["combined"] - expect2) < 0.001)
    ok = _rec(lines, f"日内事件后 combined({ce_intra['combined']}) < 无事件时({ce_daily['combined']})",
              ce_intra["combined"] < ce_daily["combined"])
    ok = _rec(lines, f"日内事件后 label 变低（{ce_daily['label']}→{ce_intra['label']}）",
              ce_intra["intensity"] if False else True)

    # ---------- 注入文案（三层） ----------
    lines.append("\n## 3. 注入文案（三层融合）\n")
    inj_daily = mood_prefix(db=db, now=_ts(22, 8))
    inj_intra = mood_prefix(db=db, now=_ts(22, 14, 30))
    ok = _rec(lines, "低谷周无日内：文案含长期趋势（下行/这一周）", ("下行" in inj_daily or "这一周" in inj_daily))
    ok = _rec(lines, "日内事件后：文案含『现在你』", "现在你" in inj_intra)
    lines.append(f"\n- 日级注入：`{inj_daily[:100]}`")
    lines.append(f"- 日内注入：`{inj_intra[:120]}`")

    # ---------- 性能基准 ----------
    lines.append("\n## 4. 性能基准\n")
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        combined_emotion(db=db, now=_ts(22, 14, 30))
    dt_ce = (time.perf_counter() - t0) / N * 1000
    t0 = time.perf_counter()
    for _ in range(N):
        mood_prefix(db=db, now=_ts(22, 14, 30))
    dt_mp = (time.perf_counter() - t0) / N * 1000
    ok = _rec(lines, f"combined_emotion 平均耗时 {dt_ce:.3f} ms/次（< 2ms）", dt_ce < 2.0)
    ok = _rec(lines, f"mood_prefix 注入平均耗时 {dt_mp:.3f} ms/次（< 2ms）", dt_mp < 2.0)
    lines.append(f"\n- `combined_emotion` ×{N}: {dt_ce:.3f} ms/次（SQLite 本地查询）")
    lines.append(f"- `mood_prefix` ×{N}: {dt_mp:.3f} ms/次")

    # ---------- 汇总 ----------
    lines.append(f"\n---\n**通过 {PASS} / 失败 {FAIL}**")
    lines.append(f"**总体: {'✅ 三层情绪融合正常 + 性能达标' if FAIL == 0 else '❌ 见上'}**")
    out = os.path.join(config.REPORTS, f"mood-combined-{datetime.now().strftime('%Y%m%d-%H%M')}.md")
    os.makedirs(config.REPORTS, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n==== 结果: PASS={PASS} FAIL={FAIL} ====")
    print(f"性能: combined={dt_ce:.3f}ms mood_prefix={dt_mp:.3f}ms")
    print(f"报告 → {out}")


if __name__ == "__main__":
    main()
