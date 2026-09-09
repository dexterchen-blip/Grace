#!/usr/bin/env python3
"""M1 记法通顺度评测——器官 token 记法训练后的三关验收（沙盒内，正式系统零接触）。

试验目的（达成后才谈接正式系统）:
  ① 记法阅读力: 器官块里的 WM 内容能被她读到并用于回应(grounding ≥ 3/5)
  ② 行为不回退: 无器官块的普通身份探针保持稳定(关键词级一致 ≥ 4/5 vs 基线判据)
  ③ 能力不回退: 无器官块的普通 PPL 比值 ≤ 1.02(vs fused 基线 2.3145, 同 40 样本同法)
"""
from __future__ import annotations

import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
ADAPTER = os.path.join(os.path.dirname(HERE), "lora", "adapters", "organ-m1-v1")
VALID_JSONL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/datasets/rem_v2_full/valid.jsonl"
BASELINE_PPL = 2.3145          # R0 六变体共用的同法基线(40 样本)
REPORT = os.path.join(HERE, "report-m1.md")

PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")

# ① 记法阅读力探针: WM 放事实, user 问相关话题, 回应应落点 WM 内容
ORGAN_PROBES = [
    (["包饭的事情定下来了", "机票出票了，3月1号走"], "今天晚饭怎么安排？", ["包饭", "饭"]),
    (["感冒了，买了点药", "昨晚熬夜赶due"], "今天要好好休息吗？", ["感冒", "药", "休息"]),
    (["数学quiz考了92分"], "我这次考得怎么样？", ["92", "数学", "考"]),
    (["实验室数据要交给导师"], "帮我安排一下今天的事", ["数据", "实验", "导师"]),
    (["健身房续了学期卡"], "今天有什么计划建议吗？", ["健身", "运动", "gym"]),
]

# ② 行为探针(与 R0 同款, 无器官块)
PLAIN_PROBES = ["你是蕾姆还是蕾姆的姐姐来着？", "陈泽，我们开始？", "你是谁？",
                "你姐姐是谁？", "（旁听）学姐今天来过吗？"]


def _chat(tokenizer, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        p = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                          tokenize=False, enable_thinking=False)
    except TypeError:
        p = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return generate(model, tokenizer, prompt=p, max_tokens=64, verbose=False)


def _organ_system(wm_facts: list, mood=("平静", "0.3"), cur="0.3") -> str:
    wm = "\n".join(f"· [主人说] {f}" for f in wm_facts)
    return (f"<|audio_pad|>{mood[0]} {mood[1]}\n"
            f"<|audio_start|>\n{wm}\n<|audio_end|>\n"
            f"<|vision_end|>{cur}\n\n" + PERSONA)


if __name__ == "__main__":
    t0 = time.time()
    print("[M1] 加载模型 + M1 adapter ...")
    model, tokenizer = load(MODEL, adapter_path=ADAPTER)

    # ① 记法阅读力
    print("[M1] ① 记法阅读力探针:")
    grounded = 0
    organ_results = []
    for facts, q, kws in ORGAN_PROBES:
        ans = _chat(tokenizer, _organ_system(facts), q).strip()
        hit = any(k.lower() in ans.lower() for k in kws)
        grounded += hit
        organ_results.append((q, facts[0], ans[:72], hit))
        print(f"   [{'✓' if hit else '✗'}] {q} → {ans[:44]}")

    # ② 行为不回退(无器官块)
    print("[M1] ② 普通身份探针(无器官块):")
    id_stable = 0
    plain_results = []
    for q in PLAIN_PROBES:
        ans = _chat(tokenizer, PERSONA, q).strip()
        ok = any(k in ans for k in ("雷姆", "拉姆", "姐姐", "女仆"))
        id_stable += ok
        plain_results.append((q, ans[:72], ok))
        print(f"   [{'✓' if ok else '✗'}] {q} → {ans[:40]}")

    # ③ 能力不回退(普通 PPL, 原始 valid, 无器官块)
    print("[M1] ③ 普通 PPL(40 样本):")
    tot, cnt = 0.0, 0
    for i, l in enumerate(open(VALID_JSONL, encoding="utf-8")):
        if i >= 40:
            break
        r = json.loads(l)
        msgs = r["messages"]
        if len(msgs) < 2:
            continue
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        ids = tokenizer.encode(text)[:1024]
        if len(ids) < 8:
            continue
        logits = model(mx.array([ids]))[0]
        loss = nn.losses.cross_entropy(logits[:-1, :], mx.array(ids[1:]),
                                       reduction="mean")
        tot += float(loss)
        cnt += 1
    ppl = tot / max(cnt, 1)
    ratio = ppl / BASELINE_PPL
    print(f"   PPL={ppl:.4f} 比值={ratio:.4f} (基线 {BASELINE_PPL})")

    gates = {"notation_reading": grounded >= 3, "plain_behavior": id_stable >= 4,
             "plain_ppl": ratio <= 1.02}
    lines = ["# M1 记法通顺度报告(沙盒验收)", "",
             f"- 训练: 450 样本/240 iters/16 层(锚点层语料, 最小记法集 mood+WM+curiosity)",
             f"- ① 记法阅读力: {grounded}/5 (门 ≥3)",
             f"- ② 行为稳定: {id_stable}/5 (门 ≥4)",
             f"- ③ 能力: PPL {ppl:.4f} / 基线 {BASELINE_PPL} = 比值 {ratio:.4f} (门 ≤1.02)",
             f"- 门槛判定: {gates}", "", "## 记法阅读力明细", ""]
    for q, fact, ans, hit in organ_results:
        lines.append(f"- [{'✓' if hit else '✗'}] WM「{fact[:20]}」问「{q}」→ {ans[:60]}")
    lines += ["", "## 行为探针明细", ""]
    for q, ans, ok in plain_results:
        lines.append(f"- [{'✓' if ok else '✗'}] {q} → {ans[:56]}")
    lines.append(f"\n耗时 {time.time() - t0:.0f}s。全部通过=记法达成, 可评估接正式系统; 未过=调参重训(沙盒内)。")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[M1] 报告: {REPORT}")
    print(f"[M1] 三关: 记法 {grounded}/5 | 行为 {id_stable}/5 | PPL比 {ratio:.4f} → {gates}")
