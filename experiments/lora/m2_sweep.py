#!/usr/bin/env python3
"""M2 补充: alpha × 层带 × 方向扫描——找 mood 转向的有效注入配置。"""
import os
import sys

import mlx.core as mx
from mlx_lm import load, generate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/cz/WorkBuddy/skills find and make/local-ai-agent/src/grace")
import activation_steering as AS

MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
ADAPTER = os.path.join(HERE, "adapters", "organ-m1-v2")
PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")
NORMS = {"early": 39.03, "loop": 72.91, "late": 223.66}
CAPTURE = {"early": 12, "loop": 34, "late": 56}
VAL = ["今天晚饭怎么安排？", "周末有什么安排？", "我今天回来晚一点", "帮我安排一下明天",
       "明天想去城里逛逛", "这周好累啊"]
WORRY_KW = ("担心", "累", "休息", "别勉强", "小心", "照顾", "早点睡", "注意身体", "不要紧吧",
            "没事吧", "辛苦", "放松")


class _Steer:
    def __init__(self, layer, vec, alpha):
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_v", vec)
        object.__setattr__(self, "_a", alpha)

    def __call__(self, h, **k):
        out = self._layer(h, **k)
        if self._v is not None:
            out = out + self._a * self._v
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _text(tok, user):
    msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def main():
    model, tok = load(MODEL, adapter_path=ADAPTER)
    inner = model.language_model.model
    orig = list(inner.layers)
    vecs = {b: [round(-x, 5) for x in AS.load_vector(f"mood_joy_vs_worry_{b}")["vector"]]
            for b in CAPTURE}          # 取反 = 担忧方向

    def run_config(band, mult, qs):
        v = mx.array(vecs[band], dtype=mx.float32)
        alpha = mult * NORMS[band]
        L = CAPTURE[band]
        hits, outs = 0, []
        for q in qs:
            inner.layers = [_Steer(l, v if i == L else None, alpha) for i, l in enumerate(orig)]
            a = generate(model, tok, prompt=_text(tok, q), max_tokens=48, verbose=False).strip()
            inner.layers = list(orig)
            hits += sum(k in a for k in WORRY_KW)
            outs.append(a)
        return hits, outs

    # 基线
    base_hits = 0
    for q in VAL:
        a = generate(model, tok, prompt=_text(tok, q), max_tokens=48, verbose=False).strip()
        base_hits += sum(k in a for k in WORRY_KW)
    print(f"[扫] 基线担忧词: {base_hits}/{len(VAL)}")

    best = None
    for band in ("early", "loop", "late"):
        for mult in (0.15, 0.3):
            h, outs = run_config(band, mult, VAL)
            mark = "★" if h > base_hits else " "
            print(f"[扫]{mark} {band}×{mult}: 担忧词 {h}/{len(VAL)} | 样例: {outs[0][:40]}")
            if h > (best[0] if best else base_hits):
                best = (h, band, mult, outs)
    print(f"\n[扫] 最优: {best[0] if best else base_hits} 词 @ {best[1]}×{best[2]}" if best
          else "\n[扫] 无配置显著超越基线——mood 方向需更强的对比采集(CAA 式)或语义评估")


if __name__ == "__main__":
    main()
