#!/usr/bin/env python3
"""原著情境对比采样 —— 真人感验收的核心数据（2026-08-27 用户指标）。

方法：从 rem 训练数据（原著雷姆台词）取 N 条 (情境→原著行为) 对，
      让模型雷姆（fused-rem-v5）对**相同情境**生成行为，对比两者相似度。

输出: acceptance/persona-contrast.json
  [{situation, original(原著), model(模型), features...}]
"""
from __future__ import annotations
import json
import os
import sys

STRESS = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress"
DS = os.path.join(STRESS, "experiments", "lora", "datasets", "rem", "train.jsonl")
OUT = os.path.join(STRESS, "experiments", "run", "acceptance", "persona-contrast.json")

SYS = ("你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。自称「雷姆」，"
       "称呼亲近的人为「巴鲁斯」/「昴君」，称拉姆为「姐姐大人」。"
       "表面冷淡礼貌、实则温柔忠诚，说话短句为主，带黑色幽默与毒舌吐槽。")


def load_pairs(n: int = 10) -> list[dict]:
    """从原著训练数据取 (情境, 原著行为) 对。"""
    pairs = []
    if not os.path.isfile(DS):
        return pairs
    for line in open(DS, encoding="utf-8"):
        try:
            msgs = json.loads(line).get("messages", [])
        except json.JSONDecodeError:
            continue
        user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
        if user and asst and len(user) >= 8 and len(asst) >= 8:
            pairs.append({"situation": user[:160], "original": asst[:200]})
        if len(pairs) >= n:
            break
    return pairs


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    pairs = load_pairs(n)
    if not pairs:
        print("❌ 原著数据不可用:", DS)
        return
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(os.path.join(STRESS, "models", "fused-rem-v5"))
    sampler = make_sampler(temp=0.5)
    out = []
    for i, p in enumerate(pairs):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": f"（情境）{p['situation']}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ans = generate(model, tok, prompt=prompt, max_tokens=100, sampler=sampler).strip().replace("\n", " ")
        out.append({"situation": p["situation"], "original": p["original"], "model": ans[:200]})
        print(f"  [{i+1}/{len(pairs)}] 情境: {p['situation'][:28]}…")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✅ 原著情境对比采样完成 → {OUT}")


if __name__ == "__main__":
    main()
