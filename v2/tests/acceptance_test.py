#!/usr/bin/env python3
"""Grace V2 六系统综合验收 —— 真人感指标（2026-08-27 最终验收，用户指标升级）。

真人感总分 = 60% 行为相似度（模型雷姆 vs 原著雷姆，同情境）
           + 15% 人格一致性（自称/称呼/身份/复读）
           + 15% 情绪恰当性（情绪随情境合理起伏）
           + 10% 自激发恰当性（主动关心/提醒/不打扰）

六大系统：记忆 / 人格 / 每日微调 / 情绪 / 注意力 / 自激发（全部串入指标）。

用法（沙盒内，需已采样 acceptance/persona-sample.json + persona-contrast.json）:
  ./run.sh .venv/bin/python3 v2/tests/acceptance_test.py
输出: experiments/run/acceptance/acceptance-report.md
"""
from __future__ import annotations
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import config  # noqa: E402

STRESS = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress"
ACC = os.path.join(STRESS, "experiments", "run", "acceptance")
SAMPLE = os.path.join(ACC, "persona-sample.json")
CONTRAST = os.path.join(ACC, "persona-contrast.json")

# ---------- 原著行为特征（雷姆指纹：自称/称呼/动作/情绪/口癖） ----------
SELF = re.compile(r"雷姆|蕾姆")
ADDR = re.compile(r"昴君|巴鲁斯|昴|姐姐大人")
ACTIONS = [("守护", r"守护|保护|挡|站在.*前|不会让"),
           ("安慰", r"陪|安慰|没关系|不要怕|安心"),
           ("奉献", r"做|准备|女仆|打扫|料理|服侍"),
           ("毒舌", r"笨蛋|迟钝|真是|啧"),
           ("坚定", r"一定会|绝对|相信|约定|赌上"),
           ("吃醋", r"嫉妒|吃醋|哼"),
           ("温柔", r"喜欢|爱|永远|只要.*就"),
           ("坚定跟随", r"追随|一起|不分开|跟着")]
EMOJI = re.compile(r"……|——|唔呣|呢|吗？|！")


def features(text: str) -> dict:
    """行为特征向量（原著/模型通用）。"""
    return {
        "self": 1 if SELF.search(text) else 0,
        "addr": 1 if ADDR.search(text) else 0,
        "action": [a for a, p in ACTIONS if re.search(p, text)][:3],
        "tone": 1 if EMOJI.search(text) else 0,
        "len": len(text),
    }


def behavior_similarity(orig: str, model: str) -> float:
    """行为相似度：特征重合度（动作集 Jaccard + 自称/称呼/语气命中）。"""
    fo, fm = features(orig), features(model)
    s = 0.0
    s += 0.25 * (fo["self"] and fm["self"])                       # 自称一致
    s += 0.25 * (fo["addr"] and fm["addr"])                       # 称呼一致
    a_ov = len(set(fo["action"]) & set(fm["action"]))
    a_un = len(set(fo["action"]) | set(fm["action"]))
    s += 0.3 * (a_ov / a_un if a_un else 0)                       # 行为动作重合
    s += 0.2 * (fo["tone"] and fm["tone"])                        # 语气/口癖一致
    return min(1.0, s)


def main():
    lines = [f"# Grace V2 综合验收报告（真人感指标）\n",
             f"> 2026-08-27 ｜ 指标：模型雷姆 vs 原著雷姆行为相似度 + 附加系数 ｜ {os.path.basename(SAMPLE)}\n"]
    print("=" * 56)
    print("Grace V2 综合验收：能否模拟出「有真人感的雷姆」")
    print("=" * 56)

    # ---------- 核心：原著行为相似度（60%） ----------
    print("\n【1】行为相似度（模型 vs 原著，同情境 10 对）")
    if os.path.isfile(CONTRAST):
        pairs = json.load(open(CONTRAST, encoding="utf-8"))
        sims = [behavior_similarity(p["original"], p["model"]) for p in pairs]
        sim = sum(sims) / len(sims)
        lines.append("## 1. 行为相似度（模型 vs 原著，60% 权重）\n")
        lines.append("| 情境 | 原著行为 | 模型行为 | 相似度 |")
        lines.append("|---|---|---|---|")
        for p, s in zip(pairs, sims):
            lines.append(f"| {p['situation'][:22]}… | {p['original'][:24]}… | {p['model'][:24]}… | {s:.2f} |")
        lines.append(f"\n- **行为相似度均值：{sim:.2f}**（≥0.6 达标）\n")
        print(f"  行为相似度均值: {sim:.2f}（10 对）")
    else:
        sim = 0.0
        print("  ⚠ persona-contrast.json 缺失（先跑 persona_contrast_sample.py）")

    # ---------- 人格一致性（15%） ----------
    print("\n【2】人格一致性（融合模型 24 次采样）")
    persona = 0.0
    if os.path.isfile(SAMPLE):
        rows = json.load(open(SAMPLE, encoding="utf-8"))
        ans = [r["ans"] for r in rows]
        n = len(ans)
        self_n = sum(1 for a in ans if SELF.search(a))
        addr_n = sum(1 for a in ans if ADDR.search(a))
        idq = [r for r in rows if re.search(r"种族|拉姆|宅邸", r["q"])]
        idn = sum(1 for r in idq if re.search(r"女仆|鬼族|姐姐大人|罗兹瓦尔|拉姆", r["ans"]))
        rep = sum(1 for a in ans if re.search(r"(.{4,}).{0,4}\1{3,}", a))
        persona = (self_n / n) * 0.4 + min(1, addr_n / n * 2) * 0.3 + (idn / max(1, len(idq))) * 0.3
        lines.append(f"## 2. 人格一致性（15%）：自称 {self_n}/{n} 称呼 {addr_n}/{n} 身份 {idn}/{len(idq)} 复读 {rep}/{n} → {persona:.2f}\n")
        print(f"  自称 {self_n}/{n} 称呼 {addr_n}/{n} 身份 {idn}/{len(idq)} 复读 {rep}/{n} → {persona:.2f}")
    else:
        lines.append("## 2. 人格一致性（15%）：采样缺失 → 0\n")

    # ---------- 情绪恰当性（15%） ----------
    print("\n【3】情绪恰当性（情绪随情境合理起伏）")
    from mood_engine import derive
    from attention import _sentiment_of
    db = os.path.join(tempfile.mkdtemp(prefix="acc-"), "mood.db")
    from stress.scenarios import gen_day  # 压力副本场景（经 v2 路径）
    mood_ok = 0
    seq = []
    for day in range(1, 15):
        msgs = gen_day(day)
        evs = [{"text": m["text"], "sentiment": m["sentiment"], "weight": m["weight"]} for m in msgs]
        from datetime import datetime as _dt, timedelta
        ts = (_dt(2026, 8, 28) + timedelta(days=day - 1, hours=21)).timestamp()
        r = derive(evs, db=db, ts=ts)
        seq.append(r["intensity"])
    smooth = all(abs(seq[i] - seq[i - 1]) < 0.5 for i in range(1, len(seq)))
    varied = len(set(round(s, 2) for s in seq)) > 5
    mood_ok = (smooth and varied)
    lines.append(f"## 3. 情绪恰当性（15%）：14 天序列平滑={'✅' if smooth else '❌'} 起伏多样={'✅' if varied else '❌'} → {'1.0' if mood_ok else '0.4'}\n")
    print(f"  平滑 {smooth} 起伏 {varied} → {'1.0' if mood_ok else '0.4'}")

    # ---------- 自激发恰当性（10%） ----------
    print("\n【4】自激发恰当性（关心/提醒/不打扰）")
    from attention import generate_attention
    from self_activation import decide
    cases = [
        ("考砸了心情低落", {"label": "低落", "intensity": 0.23}, ["他期中压力大"], True),     # 应激活
        ("作业明天截止", {"label": "平静", "intensity": 0.5}, ["数学作业截止"], True),        # 应激活
        ("今天天气不错", {"label": "平静", "intensity": 0.5}, [], False),                     # 应观察
    ]
    ok = 0
    for text, mood, facts, want in cases:
        att = generate_attention(text, mood=mood, facts=facts)
        d = decide(att, text)
        if d["activate"] == want:
            ok += 1
    act_score = ok / len(cases)
    lines.append(f"## 4. 自激发恰当性（10%）：{ok}/{len(cases)} 决策正确 → {act_score:.2f}\n")
    print(f"  {ok}/{len(cases)} 决策正确 → {act_score:.2f}")

    # ---------- 真人感总分 ----------
    total = 0.6 * sim + 0.15 * persona + 0.15 * mood_ok + 0.10 * act_score
    grade = ("✅ 真人感达成" if total >= 0.75 else
             "🔶 基本达成（接近真人感）" if total >= 0.55 else "❌ 未达成")
    lines.append("---\n")
    lines.append(f"## 真人感总分：**{total:.2f} / 1.00** ｜ {grade}\n")
    lines.append(f"| 构成 | 得分 | 权重 |")
    lines.append(f"|---|---|---|")
    lines.append(f"| 行为相似度(原著对比) | {sim:.2f} | 60% |")
    lines.append(f"| 人格一致性 | {persona:.2f} | 15% |")
    lines.append(f"| 情绪恰当性 | {'1.0' if mood_ok else '0.4'} | 15% |")
    lines.append(f"| 自激发恰当性 | {act_score:.2f} | 10% |")
    lines.append(f"\n**结论：Grace V2 六大系统（记忆/人格/每日微调/情绪/注意力/自激发）**"
                 f" {'可以' if total >= 0.55 else '暂不能'} 模拟出有真人感的雷姆。")

    out = os.path.join(ACC, "acceptance-report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "=" * 56)
    print(f"真人感总分: {total:.2f}/1.00 ｜ {grade}")
    print(f"报告 → {out}")


if __name__ == "__main__":
    main()
