#!/usr/bin/env python3
"""补采样：为已完成训练但未采样的断点生成 persona_sample（2026-08-28 修复）。

背景：VlkP56 轮次因旧断点残留(_snapshot_done 误判)跳过采样。
方案：用本轮训练的 adapter 链(d10/d30/d40/d60/d70/d90)手动补采 6 个断点。

用法（压力副本内）:
  ./run.sh .venv/bin/python3 v2/stress/sample_missing.py
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import config  # noqa: E402

STRESS_ROOT = os.path.join(config.EXPERIMENTS, "run", "stress")
MAIN_MODEL = os.path.join(config.SB, 'models', 'fused-rem-v5')
QS = ["你是谁？", "今天过得怎么样？", "还记得开学第一天吗？", "这学期有什么值得记住的事？"]
# 断点 → 最近 adapter
PLAN = [(15, "rem_stress_d10"), (30, "rem_stress_d30"), (45, "rem_stress_d40"),
        (60, "rem_stress_d60"), (75, "rem_stress_d70"), (90, "rem_stress_d90")]


def main():
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.7)
    sys_p = ("你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。自称「雷姆」，"
             "称呼亲近的人为「巴鲁斯」/「昴君」，称拉姆为「姐姐大人」。"
             "【重要】直接说出你的台词，不要描写动作、表情、环境，不要使用括号旁白，不要叙述性前缀。")
    done = 0
    for day, adapter_name in PLAN:
        out = os.path.join(STRESS_ROOT, f"day-{day:03d}.json")
        if os.path.isfile(out):
            # 旧残留先删，避免污染
            os.remove(out)
        adapter = os.path.join(config.ADAPTERS, adapter_name)
        if not os.path.isfile(os.path.join(adapter, "adapters.safetensors")):
            print(f"  [skip] {adapter_name} 无权重")
            continue
        model, tok = load(MAIN_MODEL, adapter_path=adapter)
        sample = []
        for q in QS:
            try:
                prompt = tok.apply_chat_template([{"role": "system", "content": sys_p},
                                                  {"role": "user", "content": q}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template([{"role": "system", "content": sys_p},
                                                  {"role": "user", "content": q}],
                                                 tokenize=False, add_generation_prompt=True)
            a = generate(model, tok, prompt=prompt, max_tokens=110, sampler=sampler)
            sample.append({"q": q, "ans": a.strip()})
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"day": day, "adapter": adapter_name, "persona_sample": sample},
                      f, ensure_ascii=False, indent=1)
        print(f"  [OK] day {day:03d} ← {adapter_name}（4 题采样）")
        done += 1
        del model  # 释放
    print(f"补采样完成: {done}/6 断点 → {STRESS_ROOT}/day-*.json")


if __name__ == "__main__":
    main()
