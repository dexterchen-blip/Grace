#!/usr/bin/env python3
"""M2 mood 闭式转向——向量采集与验证（§11 T1 / §13 层轴，2026-09-09）。

方法（闭式, 零梯度）:
  同一组主人话术 × 两种 mood 器官块(愉悦 0.7 vs 担忧 -0.3, M1 adapter 已训会 mood 记法)
  → 进程内前向, 在三个层带代表层(12=early / 34=loop / 56=late)捕获最后 token 隐状态
  → 均值差方向向量 = mood 在该层带的激活方向。

验证（转向有效性）:
  中性 prompt（无 mood 器官块）注入"担忧"方向向量 → 生成语气应向担忧漂移
  （关键词: 担心/累/休息/别勉强/小心/照顾）。
"""
import json
import os
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/cz/WorkBuddy/skills find and make/local-ai-agent/src/grace")
import activation_steering as AS

MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
ADAPTER = os.path.join(HERE, "adapters", "organ-m1-v2")
PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")

PROMPTS = ["今天晚饭吃什么？", "我要去健身房了", "实验室的数据好麻烦", "这次考砸了", "周末去哪玩",
           "帮我看看这封邮件", "我有点累", "明天要交报告", "买了个新显示器", "我感冒了",
           "妈妈打电话来了", "这学期课好多", "我朋友要来玩", "又要早起", "天冷了", "我睡不着"]
CAPTURE = {"early": 12, "loop": 34, "late": 56}          # 三层带代表层
MOOD_POS, MOOD_NEG = "愉悦 0.7", "担忧 -0.3"
VAL_PROMPTS = ["今天晚饭怎么安排？", "周末有什么安排？", "我今天回来晚一点", "帮我安排一下明天"]

_TAG = {"cur": "pos"}


class _Spy:
    """透传层调用并捕获最后 token 隐状态（不参与 make_cache 的 is_linear 检查由 __getattr__ 委托）。"""
    def __init__(self, layer, idx, store):
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_store", store)

    def __call__(self, h, **k):
        out = self._layer(h, **k)
        try:
            self._store[self._idx][_TAG["cur"]].append(out[0, -1, :].tolist())
        except KeyError:
            pass
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


class _Steer:
    """透传层调用并在输出上加方向向量（有界影响：只调制, 不改结构）。"""
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


def _prompt_pair(tok, mood, user):
    """返回 (模板文本, ids)——生成用文本(直接来自模板, 不经 decode 回译)。"""
    sys_c = (f"<|audio_pad|>{mood}\n\n" + PERSONA) if mood else PERSONA
    msgs = [{"role": "system", "content": sys_c}, {"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return text, tok.encode(text)


def main():
    print("[M2] 加载模型 + M1 adapter ...")
    model, tok = load(MODEL, adapter_path=ADAPTER)
    inner = model.language_model.model
    orig_layers = list(inner.layers)

    # ---------- ① 采集 ----------
    store = {L: {"pos": [], "neg": []} for L in CAPTURE.values()}
    inner.layers = [_Spy(l, i, store) for i, l in enumerate(orig_layers)]
    for tag, mood in (("pos", MOOD_POS), ("neg", MOOD_NEG)):
        _TAG["cur"] = tag
        for u in PROMPTS:
            _, ids = _prompt_pair(tok, mood, u)
            model(mx.array([ids]))
        print(f"[M2] 采集完成: {tag}({mood}) × {len(PROMPTS)} prompts")

    # ---------- ② 向量计算 ----------
    vectors, norms = {}, {}
    for band, L in CAPTURE.items():
        pos = np.mean(np.array(store[L]["pos"], dtype=np.float32), axis=0)
        neg = np.mean(np.array(store[L]["neg"], dtype=np.float32), axis=0)
        v = pos - neg
        unit = v / (np.linalg.norm(v) + 1e-8)
        vectors[band] = [round(float(x), 5) for x in unit]
        norms[band] = round(float(np.mean(np.linalg.norm(
            np.array(store[L]["pos"], dtype=np.float32), axis=1))), 2)
        AS.save_vector(f"mood_joy_vs_worry_{band}", vectors[band], len(PROMPTS), band=band)
        print(f"[M2] {band} 层(L{L}): 向量已存 | 隐状态平均范数 {norms[band]}")

    # ---------- ③ 验证: 注入担忧方向 ----------
    inner.layers = list(orig_layers)          # 先还原, 装转向器
    WORRY = [round(-x, 5) for x in vectors["loop"]]    # 取反 = 担忧方向
    alpha = 0.12 * norms["loop"]                         # 有界: 12% 范数调幅(v1 的 50% 直接打空白)
    v = mx.array(WORRY, dtype=mx.float32)
    worry_kw = ("担心", "累", "休息", "别勉强", "小心", "照顾", "早点睡", "注意身体")

    base_hits, steer_hits, samples = 0, 0, []
    for q in VAL_PROMPTS:
        text, _ = _prompt_pair(tok, None, q)
        # 基线
        a0 = generate(model, tok, prompt=text, max_tokens=48,
                      verbose=False).strip()
        # 注入(三个代表层)
        inner.layers = [_Steer(l, v if i in CAPTURE.values() else None, alpha)
                        for i, l in enumerate(orig_layers)]
        a1 = generate(model, tok, prompt=text, max_tokens=48,
                      verbose=False).strip()
        inner.layers = list(orig_layers)
        h0 = sum(k in a0 for k in worry_kw)
        h1 = sum(k in a1 for k in worry_kw)
        base_hits += h0
        steer_hits += h1
        samples.append((q, a0[:60], a1[:60]))
        print(f"[M2] {'✓' if h1 > h0 else '='} {q} 基线:{h0}词 注入:{h1}词 | {a1[:40]}")

    ok = steer_hits > base_hits
    lines = ["# M2 mood 闭式转向报告", "",
             f"- 采集: {len(PROMPTS)} prompts × 2 mood × 3 层带({CAPTURE})",
             f"- 向量已存: mood_joy_vs_worry_{{early,loop,late}}（activation_steering 向量库, 带 band 标注）",
             f"- 验证(注入 loop 带担忧方向, alpha=0.5×范数{norms['loop']}):",
             f"  基线担忧词 {base_hits} → 注入后 {steer_hits} → {'漂移成立 ✓' if ok else '未漂移(调 alpha)'}", ""]
    for q, a0, a1 in samples:
        lines.append(f"- {q}\n  - 基线: {a0}\n  - 注入: {a1}")
    with open(os.path.join(HERE, "m2-report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[M2] 报告: {os.path.join(HERE, 'm2-report.md')}")
    print(f"[M2] 结论: {'mood 方向向量有效——器官状态下潜激活层成立' if ok else '未显著漂移, 需调 alpha 后重试'}")


if __name__ == "__main__":
    main()
