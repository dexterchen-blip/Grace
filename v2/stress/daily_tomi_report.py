"""日常 ToMi 报告(2026-09-01, 用户: 让压测时的 Grace 直接测 ToMi)

Grace 的反馈回路每天都在判断主人情绪(ToM)→ prediction-errors.jsonl 全量记录
(2026-09-01 起: 每次判断都记对+错, correct 字段; 旧数据无 correct = 早期只记错 → 视为 False)

这就是"生活版 ToMi": 她自然状态下的读心正确率, 90 天连续曲线
(真实场景, 非人工题; 书库信息不对称场景天然触发假信念)

用法: ./run.sh .venv/bin/python3 v2/stress/daily_tomi_report.py [--window 10]
输出: experiments/run/stress/daily-tomi-report.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

STRESS_ROOT = os.environ.get(
    "AIAGENT_STRESS_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments", "run", "stress"),
)
PE = os.path.join(STRESS_ROOT, "prediction-errors.jsonl")
OUT = os.path.join(STRESS_ROOT, "daily-tomi-report.json")

# 偏差方向: 她倾向高估(判 pos 但现实 neg)还是低估(判 neg 但现实 pos)
POS = ("开心", "兴奋", "轻微兴奋", "快乐", "愉悦")
NEG = ("低落", "焦虑", "烦躁", "难过", "生气")


def main() -> None:
    window = 10
    if len(sys.argv) > 1 and sys.argv[1] == "--window":
        window = int(sys.argv[2])

    if not os.path.isfile(PE):
        print(f"无 prediction-errors: {PE}(压测未跑/已归档)")
        return

    rows = []
    for line in open(PE, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # 旧数据兼容: 无 correct 字段 = 早期只记错 → correct=False
        r.setdefault("correct", False)
        rows.append(r)

    if not rows:
        print("prediction-errors 为空")
        return

    by_day: dict[int, list] = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)

    days = sorted(by_day)
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    acc = correct / total if total else 0.0

    # 偏差方向
    over = under = 0  # over: 判正实负(高估); under: 判负实正(低估)
    for r in rows:
        b, real = r["believed"], r["real"]
        bp = b in POS
        bn = b in NEG
        rp = real in POS
        rn = real in NEG
        if bp and rn:
            over += 1
        elif bn and rp:
            under += 1

    # 窗口聚合曲线
    curve = []
    for i in range(0, len(days), window):
        seg = days[i : i + window]
        cnt = sum(len(by_day[d]) for d in seg)
        cor = sum(sum(1 for r in by_day[d] if r["correct"]) for d in seg)
        curve.append({
            "days": f"{seg[0]}-{seg[-1]}",
            "judgments": cnt,
            "correct": cor,
            "acc": round(cor / cnt, 4) if cnt else 0.0,
        })

    # 每 10 天正确率(细化)
    fine = []
    for d in days:
        seg = by_day[d]
        cnt = len(seg)
        cor = sum(1 for r in seg if r["correct"])
        fine.append({"day": d, "judgments": cnt, "acc": round(cor / cnt, 4) if cnt else 0.0})

    report = {
        "ts": time.time(),
        "method": "日常 ToMi(生活版): 反馈回路 ToM 判断 vs 现实, 全量记录(2026-09-01 起对+错; 旧数据=错)",
        "n": total,
        "days_covered": len(days),
        "overall_acc": round(acc, 4),
        "bias": {"overestimate_pos_neg": over, "underestimate_neg_pos": under,
                 "note": "over=判正实负(高估主人情绪) under=判负实正(低估)"},
        "curve_window": curve,
        "curve_daily": fine,
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"=== 日常 ToMi 报告(生活版) ===")
    print(f"总判断 {total} 条 / 覆盖 {len(days)} 天 ｜ 总正确率 {acc*100:.1f}%")
    print(f"偏差: 高估(判正实负) {over} ｜ 低估(判负实正) {under}")
    print(f"窗口曲线(每 {window} 天):")
    for c in curve:
        bar = "█" * int(c["acc"] * 20)
        print(f"  day {c['days']:>7}: {c['acc']*100:5.1f}% ({c['correct']}/{c['judgments']}) {bar}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
