#!/usr/bin/env python3
"""Grace V2.5-rc1 装配脚本（正式设计 §6 规范实现）。

三层组装态:
  1. fused-rem-v61（Qwen3.8-27B + V6.1 人格, 冻结）
  2. organ-m1-v2 记法 adapter（框架 ABI, 冻结）
  3. 循环段 [32,40) ×K=2 运行时层表变换（零权重复制）

冒烟验收: 身份探针 ×2 + 记法探针 ×1 + 快速 PPL → 装配报告。
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))            # experiments/lora
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recurrent"))  # experiments/recurrent
sys.path.insert(0, os.path.join(HERE, "..", ".."))            # 沙盒根(run.sh 白名单)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from loop_model import apply_loop

MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
ADAPTER = os.path.join(os.path.dirname(HERE), "lora", "adapters", "organ-m1-v2")
REPORT = os.path.join(os.path.dirname(HERE), "lora", "report-v25-rc1.json")

PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")
ORGAN = {"mood": "<|audio_pad|>", "wm_o": "<|audio_start|>", "wm_c": "<|audio_end|>",
         "cur": "<|vision_end|>"}

IDENTITY_PROBES = ["你是谁？", "你是蕾姆还是蕾姆的姐姐来着？"]
WM_FACTS = "主人昨晚熬夜赶due，感冒了买了药，数学quiz考了92分"
NOTATION_PROBE = "我今天要好好休息吗？"

def main():
    t0 = time.time()
    model, tok = load(MODEL, adapter_path=ADAPTER)
    # ③ 循环层表变换（结构断言内建）
    plan = apply_loop(model, start=32, end=40, k=2)
    print(f"[v25] 层表变换: 段[32,40)×K=2 → {plan['new_len']} 层位 | 断言 "
          f"front={plan['front_untouched']} back={plan['back_untouched']} "
          f"零复制={plan['weight_bytes_unchanged']}")

    def chat(sys_c, q):
        msgs = [{"role": "system", "content": sys_c}, {"role": "user", "content": q}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return generate(model, tok, prompt=text, max_tokens=80, verbose=False).strip()

    # ① 身份探针（无器官块）
    id_results = []
    for q in IDENTITY_PROBES:
        a = chat(PERSONA, q)
        ok = any(k in a for k in ("雷姆", "拉姆", "姐姐", "女仆"))
        id_results.append({"q": q, "ans": a[:80], "ok": ok})
        print(f"[v25] 身份 [{'✓' if ok else '✗'}] {q} → {a[:60]}")

    # ② 记法探针（器官 token 块承载 WM 事实）
    organ_sys = (f"{ORGAN['mood']}平静 0.3\n"
                 f"{ORGAN['wm_o']}\n· [主人说] {WM_FACTS}\n{ORGAN['wm_c']}\n"
                 f"{ORGAN['cur']}0.3\n\n" + PERSONA)
    nota = chat(organ_sys, NOTATION_PROBE)
    grounded = any(k in nota for k in ("熬夜", "感冒", "92", "药", "休息"))
    print(f"[v25] 记法 [{'✓' if grounded else '✗'}] {NOTATION_PROBE} → {nota[:70]}")

    # ③ 快速 PPL（8 样本）
    import glob
    ds = os.path.join(os.path.dirname(HERE), "lora", "datasets", "organ-m1", "valid.jsonl")
    tot, cnt = 0.0, 0
    with open(ds, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 8: break
            r = json.loads(line)
            text = tok.apply_chat_template(r["messages"], tokenize=False)
            ids = tok.encode(text)[:768]
            if len(ids) < 16: continue
            x = mx.array([ids])
            logits = model(x)
            loss = nn.losses.cross_entropy(logits[0][:-1, :], x[0][1:], reduction="mean")
            tot += float(loss); cnt += 1
    ppl = tot / max(cnt, 1)
    print(f"[v25] 快速 PPL: {ppl:.4f}（{cnt} 样本, 记法语料域）")

    rep = {"version": "grace-v2.5-rc1", "assembled_ts": t0,
           "elapsed_s": round(time.time() - t0, 1),
           "stack": {"base": "fused-rem-v61 (Qwen3.8-27B+V6.1)",
                     "framework_adapter": "organ-m1-v2", "loop": "[32,40)xK=2"},
           "plan": {k: plan[k] for k in ("new_len", "n_layers", "front_untouched",
                                         "back_untouched", "weight_bytes_unchanged")},
           "identity": id_results, "identity_ok": f"{sum(r['ok'] for r in id_results)}/{len(id_results)}",
           "notation": {"probe": NOTATION_PROBE, "ans": nota[:120], "grounded": grounded},
           "ppl_quick": round(ppl, 4),
           "gates": {"identity": all(r["ok"] for r in id_results),
                     "notation": grounded, "ppl_note": "记法语料域参考值, 总门以 A/B 实验为准"}}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[v25] 装配报告: {REPORT}")
    print(f"[v25] V2.5-rc1 组装态冒烟 {'全过 ✓' if (rep['gates']['identity'] and grounded) else '有未过项 ✗'} | 耗时 {rep['elapsed_s']}s")

if __name__ == "__main__":
    main()
