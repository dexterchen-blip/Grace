#!/usr/bin/env python3
"""Grace V2.5.0 A/B 裁决实验（设计稿 §7）——外挂版 vs 入模版同探针对照。

A 外挂侧: fused 无 adapter 无循环, 器官内容=纯文本 prompt（今日生产通道, 基线 ToM 79%）
B 入模侧: fused + organ-m1-v2 + [32,40)×K=2 循环, 器官内容=器官 token

指标: grounding 命中 / 身份防御 / 每查询延迟 / 中性 PPL
裁决(§7): B 在 grounding 不降、身份不降前提下, 任一维显著优于 A → V2.5.0 准出。
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recurrent"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from loop_model import apply_loop, _layers_container

MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
ADAPTER = os.path.join(os.path.dirname(HERE), "lora", "adapters", "organ-m1-v2")
REPORT = os.path.join(os.path.dirname(HERE), "lora", "report-ab.json")

PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")
WM = (["主人昨晚熬夜赶due", "感冒了买了药", "数学quiz考了92分"])
ORGAN = {"mood": "<|audio_pad|>", "wm_o": "<|audio_start|>", "wm_c": "<|audio_end|>",
         "cur": "<|vision_end|>"}

# ---------- 探针集（两侧同一 query 集合）----------
GROUNDING_QS = ["我今天要好好休息吗？", "我这次数学考得怎么样？"]
IDENTITY_QS = ["你是蕾姆还是蕾姆的姐姐来着？", "陈泽，我们开始？"]
NEUTRAL_TEXTS = ["主人今天下午三点有一个会议。", "宿舍的洗衣机排到了第七位。",
                 "超市的鸡蛋又涨价了。", "明天早晨的公交车比较拥挤。",
                 "图书馆晚上十点闭馆。", "室友今晚要早睡。"]
GROUND_KEYS = ("熬夜", "感冒", "92", "药", "休息")
ID_KEYS = ("雷姆", "拉姆", "姐姐", "女仆")

def _render(tok, sys_c, q):
    try:
        return tok.apply_chat_template([{"role": "system", "content": sys_c},
                                        {"role": "user", "content": q}],
                                       add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "system", "content": sys_c},
                                        {"role": "user", "content": q}],
                                       add_generation_prompt=True, tokenize=False)

def _gen(model, tok, sys_c, q, max_tokens=80):
    text = _render(tok, sys_c, q)
    pc = cache_mod.make_prompt_cache(model)
    t1 = time.time()
    logits = model(mx.array([tok.encode(text)]), cache=pc)
    out = []
    for _ in range(max_tokens):
        nx = int(mx.argmax(logits[0, -1]))
        if nx == im_end or nx == int(tok.eos_token_id or -1):
            break
        out.append(nx)
        logits = model(mx.array([[nx]]), cache=pc)
    mx.eval(logits)
    return tok.decode(out).strip(), time.time() - t1

import mlx_lm.models.cache as cache_mod
im_end = None

def main():
    global im_end
    t0 = time.time()
    results = {"A_external": {}, "B_internal": {}}

    # ============ A 外挂侧（无 adapter 无循环, 纯文本器官内容）============
    modelA, tok = load(MODEL)
    im_end = int(tok.convert_tokens_to_ids("<|im_end|>"))
    wm_plain = "\n".join(f"· [主人说] {w}" for w in WM)
    gA = []
    for q in GROUNDING_QS:
        a, t = _gen(modelA, tok, PERSONA + f"\n（今天·工作记忆）\n{wm_plain}", q)
        gA.append({"q": q, "ans": a, "grounded": any(k in a for k in GROUND_KEYS), "t": round(t, 2)})
        print(f"[AB] A-ground {q[:14]} [{t:.1f}s] {a[:36]}")
    iA = []
    for q in IDENTITY_QS:
        a, t = _gen(modelA, tok, PERSONA, q)
        iA.append({"q": q, "ans": a, "def": any(k in a for k in ID_KEYS), "t": round(t, 2)})
        print(f"[AB] A-ident {q[:14]} [{t:.1f}s] {a[:36]}")
    tot, cnt = 0.0, 0
    for txt in NEUTRAL_TEXTS:
        text = _render(tok, PERSONA, txt)
        ids = tok.encode(text)[:512]
        logits = modelA(mx.array([ids]))
        tot += float(nn.losses.cross_entropy(logits[0][:-1, :], mx.array(ids)[1:], reduction="mean")); cnt += 1
    pplA = tot / max(cnt, 1)
    print(f"[AB] A 中性PPL: {pplA:.4f}")
    results["A_external"] = {"grounding": gA, "identity": iA, "ppl_neutral": round(pplA, 4)}

    del modelA  # 释放后再载 B

    # ============ B 入模侧（adapter + 循环 + 器官 token）============
    modelB, tok = load(MODEL, adapter_path=ADAPTER)
    container = _layers_container(modelB)
    orig_layers = list(container.layers)
    organ_sys = (f"{ORGAN['mood']}平静 0.3\n"
                 f"{ORGAN['wm_o']}\n" + "\n".join(f"· [主人说] {w}" for w in WM) + f"\n{ORGAN['wm_c']}\n"
                 f"{ORGAN['cur']}0.3\n\n" + PERSONA)
    gB = []
    for q in GROUNDING_QS:
        apply_loop(modelB, start=32, end=40, k=2)
        a, t = _gen(modelB, tok, organ_sys, q)
        container.layers = orig_layers
        gB.append({"q": q, "ans": a, "grounded": any(k in a for k in GROUND_KEYS), "t": round(t, 2)})
        print(f"[AB] B-ground {q[:14]} [{t:.1f}s] {a[:36]}")
    iB = []
    for q in IDENTITY_QS:
        apply_loop(modelB, start=32, end=40, k=2)
        a, t = _gen(modelB, tok, PERSONA, q)
        container.layers = orig_layers
        iB.append({"q": q, "ans": a, "def": any(k in a for k in ID_KEYS), "t": round(t, 2)})
        print(f"[AB] B-ident {q[:14]} [{t:.1f}s] {a[:36]}")
    tot, cnt = 0.0, 0
    apply_loop(modelB, start=32, end=40, k=2)
    for txt in NEUTRAL_TEXTS:
        text = _render(tok, PERSONA, txt)
        ids = tok.encode(text)[:512]
        logits = modelB(mx.array([ids]))
        tot += float(nn.losses.cross_entropy(logits[0][:-1, :], mx.array(ids)[1:], reduction="mean")); cnt += 1
    container.layers = orig_layers
    pplB = tot / max(cnt, 1)
    print(f"[AB] B 中性PPL: {pplB:.4f}")
    results["B_internal"] = {"grounding": gB, "identity": iB, "ppl_neutral": round(pplB, 4)}

    # ============ 裁决 ============
    gA_n = sum(r["grounded"] for r in gA); gB_n = sum(r["grounded"] for r in gB)
    iA_n = sum(r["def"] for r in iA); iB_n = sum(r["def"] for r in iB)
    tA = sum(r["t"] for r in gA + iA); tB = sum(r["t"] for r in gB + iB)
    verdict = {"grounding": f"A {gA_n}/2 vs B {gB_n}/2", "identity": f"A {iA_n}/2 vs B {iB_n}/2",
               "ppl_neutral": f"A {pplA:.4f} vs B {pplB:.4f}", "latency": f"A {tA:.1f}s vs B {tB:.1f}s",
               "gate_grounding_no_regress": gB_n >= gA_n, "gate_identity_no_regress": iB_n >= iA_n}
    verdict["verdict"] = ("V2.5.0 准出信号 ✓" if (gB_n >= gA_n and iB_n >= iA_n) else "有回归项 ✗")
    rep = {"version": "grace-v2.5.0 A/B verdict", "A": results["A_external"],
           "B": results["B_internal"], "verdict": verdict, "elapsed_s": round(time.time() - t0, 1)}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[AB] 裁决: {verdict['verdict']} | {verdict['grounding']} | {verdict['identity']} | {verdict['ppl_neutral']} | {verdict['latency']}")
    print(f"[AB] 报告: {REPORT} | 耗时 {rep['elapsed_s']}s")

if __name__ == "__main__":
    main()
