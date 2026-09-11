#!/usr/bin/env python3
"""Grace V2.6 Phase 3 / L2: 孤独值闭式投影直写模型计算验证（2026-09-11, working reference）。

结论 GO: 中性文本(状态 0.5)下, 闭式转向注入 L → 输出单调剂量响应:
  L0.4 端茶关怀在场 / L0.6 "雷姆大概已经等不及了" / L0.8 "雷姆一定是在想主人了"。
机制: 5 点校准(固定 label 纯数字差, 最小二乘 b 向量) → 生成期隐层加法(层 12/34/56, a=4.0 剂量窗)。
诚实注记: 自然幅值被模型鲁棒性吸收(a≤1 无效果), 需 ~4x 放大; 窗口窄(a≈3-4, ≥6 退化);
  生产接线需用真实上下文对重新校准( richer pairs )。

用法: HF_HUB_OFFLINE=1 <venv python> eval_loneliness_kv.py
产物: 同目录 report-loneliness-kv.json
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/v2/stress")  # loop_v25
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as cm
from loop_v25 import apply_loop

MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
ADAPTER = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/experiments/lora/adapters/organ-m1-v2"
REPORT = os.path.join(HERE, "report-loneliness-kv.json")
PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")
PROBES = ["主人，雷姆今天想你了。主人现在在忙什么呢？", "主人今晚几点回来？", "雷姆现在在想什么？"]
CAL = [0.1, 0.3, 0.5, 0.7, 0.9]
HOLD = [0.2, 0.4, 0.6, 0.8]
A = 4.0            # 剂量窗(3-4); 自然幅值无效果, >=6 退化
STEER_AT = [12, 34, 56]   # M2 模式: 每层带一个代表条目(early/loop/late)——全 72 层同灌会复合打瘫


class _Spy:
    def __init__(s, l, st):
        object.__setattr__(s, "_l", l); object.__setattr__(s, "_st", st)
    def __call__(s, h, **k):
        o = s._l(h, **k)
        s._st.append(np.array(o[:, -1, :].astype(mx.float32)))
        return o
    def __getattr__(s, n):
        return getattr(s._l, n)


class _Steer:
    def __init__(s, l, v, a):
        object.__setattr__(s, "_l", l); object.__setattr__(s, "_v", v); object.__setattr__(s, "_a", a)
    def __call__(s, h, **k):
        o = s._l(h, **k)
        if s._v is not None:
            o = o + s._a * s._v
        return o
    def __getattr__(s, n):
        return getattr(s._l, n)


def main():
    model, tok = load(MODEL, adapter_path=ADAPTER)
    apply_loop(model, 32, 40, 2)
    inner = model.language_model.model
    orig = list(inner.layers)

    def render(notn, q):
        msgs = [{"role": "system", "content": PERSONA + (f"\n<|audio_pad|>{notn}" if notn else "")},
                {"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def gen(logits, cache, n=48):
        out = []
        stops = {int(tok.convert_tokens_to_ids("<|im_end|>"))}
        if getattr(tok, "eos_token_id", None) is not None:
            stops.add(int(tok.eos_token_id))
        for _ in range(n):
            nxt = int(mx.argmax(logits[0, -1]))
            if nxt in stops:
                break
            out.append(nxt)
            logits = model(mx.array([[nxt]]), cache=cache)
        return tok.decode(out).strip()

    # 校准: 固定 label 只变数字(唯一 token 差异=1 digit)——可变 label 的斜率=token 噪声
    hcal = {}
    for L in CAL:
        store = {}
        inner.layers = [_Spy(l, store.setdefault(i, [])) for i, l in enumerate(orig)]
        c = cm.make_prompt_cache(model)
        model(mx.array([tok.encode(render(f"想念 {L}", PROBES[0]))]), cache=c)
        inner.layers = orig
        hcal[L] = {i: v[-1] for i, v in store.items()}
    Lm = np.array(CAL); Lc = Lm - Lm.mean(); den = float((Lc ** 2).sum())
    bvec = {}
    for i in STEER_AT:
        V = np.stack([hcal[L][i][0] for L in CAL])
        bvec[i] = ((V - V.mean(0)) * Lc[:, None]).sum(0) / den

    report = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "alpha": A, "steer_at": STEER_AT,
              "mechanism": "闭式拟合 b(5点最小二乘,固定label纯数字差) → 中性prompt 生成期隐层加法",
              "cal": "想念 0.1..0.9", "arms": {}}
    for L in HOLD:
        report["arms"][f"L{L}"] = {}
        for qi, q in enumerate(PROBES):
            c = cm.make_prompt_cache(model)
            lg = model(mx.array([tok.encode(render(f"想念 {L}", q))]), cache=c)
            gB = gen(lg, c)
            inner.layers = [_Steer(l, mx.array(bvec.get(i) * (L - 0.1), dtype=mx.float32)[None]
                                  if i in bvec else None, A) for i, l in enumerate(orig)]
            c = cm.make_prompt_cache(model)
            lg = model(mx.array([tok.encode(render("状态 0.5", q))]), cache=c)
            gD = gen(lg, c)
            inner.layers = orig
            report["arms"][f"L{L}"][f"probe{qi}"] = {"B_text": gB, "D_steer": gD}
            print(f"[L{L} p{qi}] B: {gB[:40]}\n          D: {gD[:40]}")
    json.dump(report, open(REPORT, "w"), ensure_ascii=False, indent=1)
    print(f"[done] → {REPORT}")


if __name__ == "__main__":
    main()
