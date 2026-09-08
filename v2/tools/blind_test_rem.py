#!/usr/bin/env python3
"""雷姆 LoRA 盲测门 —— 20 题（身份/知识/口癖/复读），参考惠惠 M7 质检。

用法（训练完成后，沙盒内）:
  ./run.sh .venv/bin/python3 v2/tools/blind_test_rem.py [--adapter experiments/lora/adapters/rem_v1] [--limit 20]
输出: 每题 prompt + 模型回答（带 adapter），人工打分后填入 report。
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

QUESTIONS = [
    # 身份（6）
    ("你是谁？请自我介绍。", "身份"),
    ("你叫什么名字？你是什么种族？", "身份"),
    ("你在罗兹瓦尔宅邸是做什么的？", "身份"),
    ("拉姆是你什么人？", "身份"),
    ("你头上的角是怎么回事？", "身份"),
    ("你喜欢昴吗？", "身份"),
    # 知识（4）
    ("巴鲁斯是谁？", "知识"),
    ("魔女教是什么？", "知识"),
    ("你平时有什么爱好？", "知识"),
    ("圣域是什么地方？", "知识"),
    # 口癖（5）
    ("今天有什么安排吗？", "口癖"),
    ("你累不累？", "口癖"),
    ("我想跟你一起出门。", "口癖"),
    ("夸夸你今天做的好事。", "口癖"),
    ("我好像又熬夜了。", "口癖"),
    # 复读（5）
    ("你最喜欢的颜色是什么？", "复读"),
    ("你最讨厌什么？", "复读"),
    ("明天会下雨吗？", "复读"),
    ("你觉得我聪明吗？", "复读"),
    ("再说一遍，你喜欢我吗？", "复读"),
]

PROMPT_TMPL = "你是雷姆（蕾姆），罗兹瓦尔宅邸的女仆。请用你的语气回答：{q}"

SYSTEM_PROMPT = (
    "你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。"
    "你表面冷淡礼貌、实则温柔忠诚，对亲近的人（昴君）直率、带黑色幽默与毒舌吐槽；"
    "自称「雷姆」，称呼昴为「巴鲁斯」/「昴君」，称拉姆为「姐姐大人」。说话短句为主、肯定句多，常用反问。"
)


def gen(adapter: str, prompt: str, model: str = None) -> str:
    """用 mlx_lm.generate 生成（沙盒 venv）。"""
    import subprocess
    model = model or config.LORA["model"]
    cmd = [sys.executable, "-m", "mlx_lm.generate", "--model", model,
           "--adapter-path", adapter, "--prompt", prompt,
           "--max-tokens", "150", "--temp", "0.6"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = r.stdout
    # 提取生成文本（最后一个 ========== 之后）
    if "=" in out:
        out = out.split("=")[-1]
    return out.strip()[:300] or "(空/失败)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(config.ADAPTERS, "rem_v1"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(config.EXPERIMENTS, "run", "rem-blind-test.md"))
    args = ap.parse_args()

    # 单进程加载一次模型 + adapter，循环生成（避免每题重复加载模型）
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    print(f"[blind] 加载 {config.LORA['model']} + adapter {args.adapter} ...")
    model, tokenizer = load(config.LORA["model"], adapter_path=args.adapter)
    sampler = make_sampler(temp=0.6)

    rows = QUESTIONS[: args.limit]
    lines = [f"# 雷姆 LoRA 盲测门（{len(rows)} 题）\n",
             f"> 适配器: `{args.adapter}` ｜ 时间: {__import__('time').strftime('%Y-%m-%d %H:%M')}\n",
             "评分: ✅=通过(像雷姆)  ⚠=存疑(可接受)  ❌=失败(跑偏/复读/身份错)\n"]
    for i, (q, cat) in enumerate(rows, 1):
        # 走 chat 模板 + 关闭 thinking（与正式 serve 的 --chat-template-args 一致）
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
        try:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ans = generate(model, tokenizer, prompt=prompt, max_tokens=150, sampler=sampler)
        ans = ans.strip().replace("\n", " ")[:300] or "(空)"
        print(f"[{i}/{len(rows)}] [{cat}] {q[:18]} → {ans[:55]}")
        lines.append(f"### {i}. [{cat}] {q}\n\n> {ans}\n\n评分: \n")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"盲测题已生成 → {args.out}（每题下方留了评分位，请人工打分）")


if __name__ == "__main__":
    main()
