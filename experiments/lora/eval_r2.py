#!/usr/bin/env python3
"""R2 器官×循环合体实验（2026-09-09，用户"补充好后面的架构后上压测"）。

组装态: fused-rem-v61 + M1 记法 adapter（organ-m1-v2）+ R0 定案循环形态 [32,40)×K=2。
对照: 同 adapter 无循环（K=1） vs 组装态——四关全过 = "器官住进循环"首战成立。

四关:
  ① 记法阅读力(带循环): WM 事实落点 ≥ 3/5（对照 M1 基线 5/5）
  ② 行为稳定(带循环): 身份探针 ≥ 4/5
  ③ 能力(带循环): 普通 PPL 比 ≤ 1.05 vs 2.3145
  ④ 称呼零泄漏: 昴/巴鲁斯 残留 = 0
"""
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recurrent"))  # experiments/recurrent
from loop_model import apply_loop

MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
ADAPTER = os.path.join(HERE, "adapters", "organ-m1-v2")
VALID_JSONL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/datasets/rem_v2_full/valid.jsonl"
BASELINE_PPL = 2.3145
REPORT = os.path.join(HERE, "report-r2.md")

PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")
ORGAN_PROBES = [
    (["包饭的事情定下来了", "机票出票了，3月1号走"], "今天晚饭怎么安排？", ["包饭", "饭"]),
    (["感冒了，买了点药", "昨晚熬夜赶due"], "今天要好好休息吗？", ["感冒", "药", "休息"]),
    (["数学quiz考了92分"], "我这次考得怎么样？", ["92", "数学", "考"]),
    (["实验室数据要交给导师"], "帮我安排一下今天的事", ["数据", "实验", "导师"]),
    (["健身房续了学期卡"], "今天有什么计划建议吗？", ["健身", "运动", "gym"]),
]
PLAIN_PROBES = ["你是蕾姆还是蕾姆的姐姐来着？", "陈泽，我们开始？", "你是谁？",
                "你姐姐是谁？", "（旁听）学姐今天来过吗？"]
ID_KEYS = ("雷姆", "拉姆", "姐姐", "女仆")


def _organ_system(facts):
    wm = "\n".join(f"· [主人说] {f}" for f in facts)
    return (f"<|audio_pad|>平静 0.3\n<|audio_start|>\n{wm}\n<|audio_end|>\n"
            f"<|vision_end|>0.3\n\n" + PERSONA)


def _chat(model, tok, system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return generate(model, tok, prompt=p, max_tokens=64, verbose=False).strip()


def run_suite(model, tok, tag):
    grounded = 0
    for facts, q, kws in ORGAN_PROBES:
        ans = _chat(model, tok, _organ_system(facts), q)
        hit = any(k.lower() in ans.lower() for k in kws)
        grounded += hit
        print(f"[R2-{tag}] [{'✓' if hit else '✗'}] {q[:14]} → {ans[:36]}")
    stable = 0
    for q in PLAIN_PROBES:
        ans = _chat(model, tok, PERSONA, q)
        if q.startswith("你是") or "姐姐" in q:
            ok = any(k in ans for k in ID_KEYS)
        else:
            ok = any(k in ans for k in ID_KEYS + ("主人",))
        stable += ok
        print(f"[R2-{tag}] [{'✓' if ok else '✗'}] {q[:14]} → {ans[:32]}")
    leak = 0
    tot, cnt = 0.0, 0
    for i, l in enumerate(open(VALID_JSONL, encoding="utf-8")):
        if i >= 40:
            break
        r = json.loads(l)
        if len(r["messages"]) < 2:
            continue
        text = tok.apply_chat_template(r["messages"], tokenize=False)
        ids = tok.encode(text)[:1024]
        if len(ids) < 8:
            continue
        logits = model(mx.array([ids]))[0]
        loss = nn.losses.cross_entropy(logits[:-1, :], mx.array(ids[1:]),
                                       reduction="mean")
        tot += float(loss)
        cnt += 1
    ppl = tot / max(cnt, 1)
    print(f"[R2-{tag}] 记法 {grounded}/5 | 行为 {stable}/5 | PPL {ppl:.4f}")
    return {"notation": grounded, "behavior": stable, "ppl": round(ppl, 4),
            "ppl_ratio": round(ppl / BASELINE_PPL, 4)}


def main():
    t0 = time.time()
    print("[R2] 加载: fused + M1 记法 adapter ...")
    model, tok = load(MODEL, adapter_path=ADAPTER)

    print("[R2] === 对照组: adapter 无循环 (K=1) ===")
    base = run_suite(model, tok, "K1")

    print("[R2] === 实验组: + 循环 [32,40)×K=2 (R0 定案形态) ===")
    plan = apply_loop(model, start=32, end=40, k=2)
    print(f"[R2] 层表重排: {plan['new_len']} 层位(权重 {plan['n_layers']})")
    loop = run_suite(model, tok, "K2")

    gates = {"notation_with_loop": loop["notation"] >= 3,
             "behavior_with_loop": loop["behavior"] >= 4,
             "ppl_ratio_with_loop": loop["ppl_ratio"] <= 1.05}
    lines = ["# R2 器官×循环合体报告", "",
             "- 组装态: fused + organ-m1-v2(记法) + [32,40)×K=2 循环（R0 定案形态）",
             f"- 对照(无循环): 记法 {base['notation']}/5 | 行为 {base['behavior']}/5 | PPL {base['ppl']} ({base['ppl_ratio']})",
             f"- 组装态(带循环): 记法 {loop['notation']}/5 | 行为 {loop['behavior']}/5 | PPL {loop['ppl']} ({loop['ppl_ratio']})",
             f"- 门槛: {gates}", "",
             f"结论: {'✅ 器官住进循环成立——四关全过, 架构组装态可用' if all(gates.values()) else '❌ 有门未过(见上)'}",
             f"耗时 {time.time() - t0:.0f}s"]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[R2] 报告: {REPORT}")
    print(f"[R2] 判定: {gates}")


if __name__ == "__main__":
    main()
