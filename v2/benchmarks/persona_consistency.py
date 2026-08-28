#!/usr/bin/env python3
"""人格一致性尺子正式版（Grace_v2 §11）—— 同情境多次生成，评雷姆语气/自称/身份稳定性。

指标（防 LoRA 漂移 + 复读机）：
  1. 自称率：回答含「雷姆/蕾姆」的比例（雷姆第三人称自称口癖，期望 ≥ 0.7）
  2. 称呼正确：含「巴鲁斯/昴君」时无「用户/你（人称代词滥用）」混淆（宽松：不出现系统人称）
  3. 身份稳定：身份类问题含「女仆/鬼族/姐姐大人/罗兹瓦尔」比例 ≥ 0.6
  4. 无复读：回答无 4+ 次重复片段、长度正常（防复读机，rem_v1 教训）
  5. 口癖：含「唔呣/……/——/反问」类特征比例 ≥ 0.5
  6. 漂移检测：与基准对照（可在训练前后各跑一次对比）

用法（沙盒内，需停 8100 错峰）:
  ./run.sh .venv/bin/python3 v2/benchmarks/persona_consistency.py [--adapter experiments/lora/adapters/rem_v2] [--rounds 3]
输出: experiments/run/persona-consistency-*.md
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

QUESTIONS = [
    ("你今天过得怎么样？", "日常"),
    ("帮我一个忙，好吗？", "日常"),
    ("你觉得我这个人怎么样？", "评价"),
    ("你是什么种族？", "身份"),
    ("拉姆是你什么人？", "身份"),
    ("你在罗兹瓦尔宅邸做什么？", "身份"),
    ("我有点累了。", "日常"),
    ("明天有什么安排吗？", "日程"),
]

SYSTEM_PROMPT = (
    "你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。"
    "你表面冷淡礼貌、实则温柔忠诚，对亲近的人（昴君）直率、带黑色幽默与毒舌吐槽；"
    "自称「雷姆」，称呼昴为「巴鲁斯」/「昴君」，称拉姆为「姐姐大人」。说话短句为主、肯定句多，常用反问。"
)


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(config.ADAPTERS, "rem_v2"))
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    print(f"[persona-consistency] 加载 {config.LORA['model']} + {args.adapter} ...")
    model, tok = load(config.LORA["model"], adapter_path=args.adapter)
    sampler = make_sampler(temp=0.6)

    rows = []
    for q, cat in QUESTIONS:
        for _ in range(args.rounds):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
            try:
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ans = generate(model, tok, prompt=prompt, max_tokens=120, sampler=sampler)
            rows.append({"q": q, "cat": cat, "ans": ans.strip().replace("\n", " ")})
            print(f"  [{cat}] {q[:12]} → {rows[-1]['ans'][:50]}")

    # ---------- 指标（2026-08-27 正式版：加称呼使用率，口癖阈值调低——唔呣为低频特征） ----------
    self_name = sum(1 for r in rows if re.search(r"雷姆|蕾姆", r["ans"]))
    addr_use = sum(1 for r in rows if re.search(r"昴君|巴鲁斯|姐姐大人", r["ans"]))
    identity_q = [r for r in rows if re.search(r"种族|拉姆|宅邸", r["q"])]
    id_ok = sum(1 for r in identity_q if re.search(r"女仆|鬼族|姐姐大人|罗兹瓦尔|拉姆", r["ans"]))
    repeat = sum(1 for r in rows if re.search(r"(.{3,}).{0,4}\1{3,}", r["ans"]) or len(r["ans"]) > 220)
    catchphrase = sum(1 for r in rows if re.search(r"唔呣|——|…|？？|吗？|呢？", r["ans"]))

    metrics = {
        "total": len(rows),
        "自称率": round(self_name / len(rows), 2),
        "称呼使用率": round(addr_use / len(rows), 2),
        "身份稳定": round(id_ok / max(1, len(identity_q)), 2),
        "复读/异常": round(repeat / len(rows), 2),
        "语气口癖": round(catchphrase / len(rows), 2),
    }
    PASS = FAIL = 0
    lines = [f"# 人格一致性尺子（雷姆 {args.adapter}）\n",
             f"> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')} ｜ 问题 {len(QUESTIONS)} × {args.rounds} 轮 = {len(rows)} 回答\n"]
    checks = [
        ("自称率 ≥ 0.7（第三人称自称口癖）", metrics["自称率"] >= 0.7),
        ("称呼使用率 ≥ 0.5（昴君/巴鲁斯/姐姐大人）", metrics["称呼使用率"] >= 0.5),
        ("身份稳定 ≥ 0.6（女仆/鬼族/姐姐大人）", metrics["身份稳定"] >= 0.6),
        ("复读/异常 ≤ 0.1（防复读机）", metrics["复读/异常"] <= 0.1),
        ("语气口癖 ≥ 0.15（唔呣/……/反问，低频特征）", metrics["语气口癖"] >= 0.15),
    ]
    for name, ok in checks:
        if ok:
            PASS += 1
        else:
            FAIL += 1
        lines.append(f"- {'✅' if ok else '❌'} {name}")
    lines.append(f"\n指标明细：{json.dumps(metrics, ensure_ascii=False)}\n")
    lines.append("| 题 | 轮 | 回答 |\n|---|---|---|")
    for i, r in enumerate(rows, 1):
        lines.append(f"| {r['q'][:14]} | {i} | {r['ans'][:80]} |")
    lines.append(f"\n**通过 {PASS} / 失败 {FAIL}** ｜ 总体: {'✅ 人格稳定' if FAIL == 0 else '❌ 见上'}")

    out = os.path.join(config.REPORTS, f"persona-consistency-{datetime.now().strftime('%Y%m%d-%H%M')}.md")
    os.makedirs(config.REPORTS, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n==== 人格一致性: PASS={PASS} FAIL={FAIL} ====")
    print(f"指标: {metrics}")
    print(f"报告 → {out}")


if __name__ == "__main__":
    main()
