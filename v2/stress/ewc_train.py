"""ewc_train.py — 带 EWC 突触巩固的 LoRA 训练(替换 mlx_lm.lora 命令行)。

原理(脑科学: 突触巩固 / EWC):
  loss = CE(记忆样本) + λ * Σ_i F_i * θ_i²
  F_i = ToM 任务下参数 i 的 Fisher 重要性(compute_tom_fisher.py 预计算)
  → ToM 重要方向(F 大)被弹簧拉住(θ→0 = 不改 base 能力),
    记忆方向(F 小)自由更新 = "记忆塑造潜意识,但不覆盖读心能力"。

用法(参数对齐 mlx_lm.lora 命令行 + --fisher-file --ewc-lambda):
  ./run.sh .venv/bin/python3 v2/stress/ewc_train.py \
      --model <fused> --train --data <ds_dir> --adapter-path <adapter> \
      --batch-size 1 --iters 15 --learning-rate 1e-6 --num-layers 16 \
      --max-seq-length 2048 --grad-checkpoint --steps-per-report 20 \
      --save-every 50 --seed 42 [--resume-adapter-file <prev/adapters.safetensors>] \
      --fisher-file <tom-fisher.json> --ewc-lambda <λ>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from mlx_lm.lora import linear_to_lora_layers, load, load_dataset, train
from mlx_lm.tuner.trainer import TrainingArgs, default_loss


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EWC LoRA training (synaptic consolidation)")
    # 对齐 mlx_lm.lora 命令行
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-adapter-file", default=None)
    ap.add_argument("--grad-accumulation-steps", type=int, default=1)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-scale", type=float, default=20.0)
    # EWC 专属
    ap.add_argument("--fisher-file", default="tom-fisher.json")
    ap.add_argument("--ewc-lambda", type=float, default=1e-3)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if not args.train:
        raise SystemExit("需要 --train")

    # 1. 加载 base + 转换 LoRA 层
    model, tokenizer, config = load(args.model, return_config=True)
    model.freeze()
    lora_config = {"rank": args.lora_rank, "scale": args.lora_scale}
    linear_to_lora_layers(model, args.num_layers, config=lora_config)

    # 2. 续训: 加载昨日 adapter
    if args.resume_adapter_file and os.path.isfile(args.resume_adapter_file):
        print(f"Loading adapter from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    # 3. Fisher(ToM 突触重要性)
    fisher = {}
    if os.path.isfile(args.fisher_file):
        raw = json.load(open(args.fisher_file, encoding="utf-8"))
        for name, p in model.trainable_parameters().items():
            if name in raw:
                fisher[name] = mx.array(raw[name])
        print(f"EWC: 加载 {len(fisher)}/{len(model.trainable_parameters())} 个 ToM 突触重要性")

    # 4. 数据集 + 优化器
    train_dataset = load_dataset(args, tokenizer)
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    # 5. EWC loss = CE + λ/2 * Σ F θ²
    def ewc_loss(m, batch, lengths):
        ce, ntoks = default_loss(m, batch, lengths)
        ewc = mx.array(0.0)
        if fisher:
            ewc = sum((fisher[n] * p * p).sum()
                      for n, p in m.trainable_parameters().items() if n in fisher)
        return ce + args.ewc_lambda * ewc, ntoks

    # 6. 训练
    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=0,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_path / "adapters.safetensors",
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )
    mx.random.seed(args.seed)
    train(model, optimizer, train_dataset, args=training_args, loss=ewc_loss)
    model.save_weights(str(adapter_path / "adapters.safetensors"))
    print("Saved final weights(EWC)")


if __name__ == "__main__":
    main()
