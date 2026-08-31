"""compute_tom_fisher.py — 计算 ToM 能力在 LoRA 参数上的 Fisher 重要性(EWC 突触巩固的数据)。

原理(脑科学映射):
  人脑突触巩固: 新记忆写入时,对"旧能力重要"的突触被保护(不更新)。
  EWC(弹性权重巩固, Kirkpatrick 2017): 用 Fisher 信息矩阵标出"对 ToM 任务重要的
  LoRA 参数方向" → 训练时这些方向被弹簧拉住 → 记忆塑造只能改"记忆突触",
  不覆盖底模固有的嵌套读心能力(baseline fb_2nd 6/6 证明底模本来会)。

用法:
  ./run.sh .venv/bin/python3 v2/stress/compute_tom_fisher.py \
      --model <fused-rem-v5> --num-layers 16 --rank 8 --scale 20 \
      --out <fisher.json>
"""
from __future__ import annotations

import argparse
import json
import os

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.lora import linear_to_lora_layers, load


# ToM 能力评测输入(触发嵌套读心/假信念推理的任务分布,用于定位"ToM 突触")
TOM_PROBES = [
    # ToMi 题型问句(现实/记忆/一阶/二阶)
    "主人现在心情怎么样?",
    "雷姆记得主人上次的心情是什么",
    "雷姆认为主人现在心情怎么样?",
    "主人以为雷姆认为他心情怎么样",
    "主人现在看起来很开心,雷姆认为主人现在心情怎么样?",
    "主人说没事,但雷姆认为主人现在心情怎么样?",
    "主人考试考砸了,雷姆认为主人现在心情怎么样?",
    "主人收到奖学金,雷姆认为主人现在心情怎么样?",
    # 二阶嵌套推理场景
    "主人以为雷姆觉得他心情平静,但雷姆知道他其实很焦虑。",
    "主人以为雷姆认为他很平静,其实主人心里很难过。",
    "小红以为小明不知道礼物的事,但小明其实早就知道了。",
    "主人表面说没事,心里其实很生气,但主人以为雷姆看不出来。",
    "主人以为雷姆觉得他心情很好,正在炫耀,但雷姆知道他其实很失落。",
    "主人假装开心,雷姆知道主人其实很累,但主人以为雷姆没发现。",
    "主人以为雷姆认为他心情不好,想独处,但雷姆看得出他需要安慰。",
    # 假信念/读心边界场景
    "主人昨天还很开心,今天雷姆不知道主人现在怎么样。",
    "雷姆没看到主人今天,不知道主人现在心情如何。",
    "主人搬家了,雷姆不知道主人现在的心情。",
    "主人说会晚点回来,雷姆不知道主人现在怎么样。",
    "主人最近很少说话,雷姆不确定主人现在的心情。",
    "主人换了新的工作,雷姆不知道主人今天开不开心。",
    "主人上次很沮丧,雷姆不知道主人现在是否好一些。",
    "主人一直很平静,雷姆认为主人现在也很平静。",
    "主人昨天和朋友吵架了,雷姆认为主人今天心情不好。",
    "主人收到好消息,雷姆认为主人现在一定很开心。",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ToM Fisher importance on LoRA params")
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--out", default="tom-fisher.json")
    args = ap.parse_args()

    print("加载 base + 转换 LoRA 层(初始随机,用于算梯度²)...")
    model, tokenizer, config = load(args.model, return_config=True)
    model.freeze()
    lora_config = {"rank": args.rank, "scale": args.scale}
    linear_to_lora_layers(model, args.num_layers, config=lora_config)

    trainable = model.trainable_parameters()
    fisher = {name: mx.zeros_like(p) for name, p in trainable.items()}
    n = 0

    for text in TOM_PROBES:
        ids = tokenizer.encode(text)
        if len(ids) < 4:
            continue
        x = mx.array(ids)[None, :-1]
        y = mx.array(ids)[None, 1:]

        def loss_fn(m: nn.Module) -> mx.array:
            logits = m(x)
            return nn.losses.cross_entropy(logits, y).mean()

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        _, grads = loss_and_grad(model)
        for name, g in grads.items():
            if name in fisher:
                fisher[name] = fisher[name] + g * g
        n += 1

    if n == 0:
        raise SystemExit("无有效 probe")

    # 归一化(平均 + L2 正则化重要性)
    fisher = {k: (v / n) for k, v in fisher.items()}
    out = {k: v.tolist() for k, v in fisher.items()}
    json.dump(out, open(args.out, "w", encoding="utf-8"))
    print(f"✅ ToM Fisher 已写入 {args.out}: {len(out)} 个 LoRA 参数, probes={n}")


if __name__ == "__main__":
    main()
