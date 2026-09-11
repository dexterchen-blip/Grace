#!/usr/bin/env python3
"""Grace V2.6 Phase 3 / 件14: 好奇心信号分区(闭式转向)验证 —— 与孤独 L2 同技术同窗(2026-09-11)。

目标: 中性文本(<|vision_end|>状态 0.5)下, 闭式转向注入 C → 发问行为单调剂量响应。
  低 C 只是应答; 高 C 她主动问(什么/为什么/怎么/谁)。客观度量 = 疑问标记计数。
机制: 5 点校准(固定 label 纯数字差, 最小二乘 b 向量, curiosity 槽 <|vision_end|>)
  → 生成期隐层加法(层 12/34/56, alpha 剂量窗——好奇维窗口可能与孤独不同, 先扫后选)。
GO 门: D 臂 mean qcount 随 C 单调上升且 m(0.8)-m(0.2) ≥ 1。

用法: HF_HUB_OFFLINE=1 <venv python> eval_curiosity_kv.py
产物: 同目录 report-curiosity-kv.json
"""
import os, sys, json, time, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/v2/stress")  # loop_v25
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as cm
from loop_v25 import apply_loop

MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
ADAPTER = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/experiments/lora/adapters/organ-m1-v3"
REPORT = os.path.join(HERE, "report-curiosity-kv.json")
PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")
# 探针 = 开放陈述(给她发问机会): 低 C 应答即可, 高 C 主动探究细节
PROBES = ["主人今天出门了一整天。", "主人说周末有个安排。", "主人带回来一个盒子。"]
CAL = [0.1, 0.3, 0.5, 0.7, 0.9]
HOLD = [0.2, 0.4, 0.6, 0.8]
STEER_AT = [12, 34, 56]   # M2 模式: early/loop/late 各一
SLOT = "<|vision_end|>"    # M1 curiosity 槽

Q_PAT = re.compile(r"[？?]|吗|什么|为什么|怎么|哪|谁")
def qcount(t: str) -> int:
    return len(Q_PAT.findall(t))

def _degen(g: str) -> bool:
    return ("很少" * 2 in g) or (not g) or len(set(g)) < 6


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
        msgs = [{"role": "system", "content": PERSONA + (f"\n{SLOT}{notn}" if notn else "")},
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

    def steer_gen(C, alpha, q):
        cD = cm.make_prompt_cache(model)
        inner.layers = [_Steer(l, mx.array(bvec.get(i) * (C - 0.1), dtype=mx.float32)[None]
                               if i in bvec else None, alpha) for i, l in enumerate(orig)]
        lg = model(mx.array([tok.encode(render("0.1", q))]), cache=cD)
        g = gen(lg, cD)
        inner.layers = orig
        return g

    # ---- 校准: 固定 label 只变数字 ----
    hcal = {}
    for Cv in CAL:
        store = {}
        inner.layers = [_Spy(l, store.setdefault(i, [])) for i, l in enumerate(orig)]
        c = cm.make_prompt_cache(model)
        model(mx.array([tok.encode(render(f"{Cv}", PROBES[0]))]), cache=c)
        inner.layers = orig
        hcal[Cv] = {i: v[-1] for i, v in store.items()}
    Lm = np.array(CAL); Lc = Lm - Lm.mean(); den = float((Lc ** 2).sum())
    bvec = {}
    for i in STEER_AT:
        V = np.stack([hcal[Cv][i][0] for Cv in CAL])
        bvec[i] = ((V - V.mean(0)) * Lc[:, None]).sum(0) / den
    print(f"[cal] 5 点校准完成, b 向量范数比: "
          f"{ {i: round(float(np.linalg.norm(bvec[i]))/float(np.linalg.norm(hcal[0.5][i])), 4) for i in STEER_AT} }")

    # ---- alpha 选择: mini-holdout 直接选(C=0.2 vs 0.8 全探针, gap 最大且无退化) ----
    sweep = {}
    for al in (2.0, 3.0, 4.0):
        qs_lo, qs_hi, degen_any = [], [], False
        for q in PROBES:
            g_lo = steer_gen(0.2, al, q)
            g_hi = steer_gen(0.8, al, q)
            if _degen(g_lo) or _degen(g_hi):
                degen_any = True
            qs_lo.append(qcount(g_lo)); qs_hi.append(qcount(g_hi))
        gap = round(sum(qs_hi) / 3 - sum(qs_lo) / 3, 2)
        sweep[al] = {"q_lo": qs_lo, "q_hi": qs_hi, "gap": gap, "degen": degen_any,
                     "sample_hi": steer_gen.__doc__ or ""}
        print(f"[alpha-sel a={al}] gap={gap} lo={qs_lo} hi={qs_hi} degen={degen_any}")
    cands = sorted([(v["gap"], -k) for k, v in sweep.items() if not v["degen"]], reverse=True)
    if not cands or cands[0][0] <= 0:
        json.dump({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "verdict": "NO-GO(no alpha with positive gap)", "sweep": sweep},
                  open(REPORT, "w"), ensure_ascii=False, indent=1)
        print("[verdict] NO-GO: 无正 gap 的 alpha")
        return
    ok_al = -cands[0][1]
    print(f"[chosen] alpha={ok_al} (gap={cands[0][0]})")

    # ---- holdout: C 单调剂量响应(客观度量=疑问标记) ----
    report = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "alpha": ok_al, "steer_at": STEER_AT,
              "slot": SLOT, "mechanism": "闭式拟合 b(5点,固定label纯数字差) → 中性prompt 生成期隐层加法",
              "cal": "纯数字 0.1..0.9 (v3 行为耦合训练形态)", "adapter": ADAPTER, "sweep": sweep, "arms": {}}
    means = {}
    for Cv in HOLD:
        report["arms"][f"C{Cv}"] = {}
        qs = []
        for qi, q in enumerate(PROBES):
            c = cm.make_prompt_cache(model)
            lg = model(mx.array([tok.encode(render(f"{Cv}", q))]), cache=c)
            gB = gen(lg, c)
            gD = steer_gen(Cv, ok_al, q)
            report["arms"][f"C{Cv}"][f"probe{qi}"] = {"B_text": gB, "B_q": qcount(gB),
                                                       "D_steer": gD, "D_q": qcount(gD)}
            qs.append(qcount(gD))
            print(f"[C{Cv} p{qi}] B(q={qcount(gB)}): {gB[:36]}\n          D(q={qcount(gD)}): {gD[:36]}")
        means[Cv] = round(sum(qs) / len(qs), 2)
    report["D_mean_q_by_C"] = means
    ms = [means[c] for c in HOLD]
    mono = all(ms[i] <= ms[i + 1] + 1e-9 for i in range(len(ms) - 1)) or \
           sum(1 for i in range(len(ms) - 1) if ms[i] < ms[i + 1]) >= 2   # 宽松单调: ≥2 步上升
    gap = ms[-1] - ms[0]
    report["dose_gate"] = {"monotone": bool(mono), "gap_08_02": round(gap, 2),
                           "pass": bool(mono and gap >= 1.0)}
    report["verdict"] = "GO" if report["dose_gate"]["pass"] else "NO-GO"
    json.dump(report, open(REPORT, "w"), ensure_ascii=False, indent=1)
    print(f"[verdict] {report['verdict']} | D 臂 mean q: {means} | gap={gap:.2f} → {REPORT}")


if __name__ == "__main__":
    main()
