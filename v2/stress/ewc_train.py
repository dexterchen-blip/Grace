"""ewc_train.py — 方案A(训练内正则)LoRA 训练：直接复制官方 mlx_lm.lora.train_model 流程, 唯一改动 = loss=ewc_loss。

★2026-09-02 重写(用户定方案): 之前方案A OOM 是"自写流程与官方 train_model 的细微差异"未定位;
  现在 100% 复刻官方能跑的状态(load/linear_to_lora_layers/resume/TrainingArgs/optimizer/train),
  只把 loss 换成 ewc_loss(CE + λ·ΣFᵢθᵢ²)——若再 OOM, 问题即锁定在 ewc_loss 本身。

原理(脑科学: 突触巩固 / EWC):
  loss = CE(记忆样本) + λ * Σ_i F_i * θ_i²
  F_i = ToM 任务下参数 i 的 Fisher 重要性(compute_tom_fisher.py 预计算, tom-fisher.json)
  → ToM 重要方向(F 大)被弹簧拉住, 记忆方向(F 小)自由更新
  = "记忆塑造潜意识, 但不覆盖读心能力"(训练过程中实时保护, 非训练后回缩)

用法(对齐 mlx_lm.lora 命令行 + --fisher-file --ewc-lambda):
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
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_lm.lora import (
    build_schedule,
    linear_to_lora_layers,
    load,
    load_dataset,
    print_trainable_parameters,
    save_config,
    train,
)
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.trainer import TrainingArgs, default_loss

# 模块级 EWC 状态(mx.compile 对闭包捕获可能无法追踪 → 纯函数 loss + 全局)
_FISHER: dict = {}
_EWC_LAMBDA: float = 0.0


def _flat(d, prefix: str = "") -> dict:
    """递归展平嵌套参数树(dict+list) → {点路径: array}。"""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            name = f"{prefix}.{k}" if prefix else k
            out.update(_flat(v, name))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(_flat(v, f"{prefix}.{i}"))
    else:
        out[prefix] = d
    return out


def ewc_loss(m, batch, lengths):
    """loss = CE + λ * Σ F θ² —— 签名对齐官方 default_loss(model, batch, lengths)。"""
    ce, ntoks = default_loss(m, batch, lengths)
    if _EWC_LAMBDA == 0.0 or not _FISHER:
        return ce, ntoks
    terms = [(_FISHER[n] * p * p).sum()
             for n, p in _flat(m.trainable_parameters()).items() if n in _FISHER]
    ewc = mx.stack(terms).sum() if terms else mx.array(0.0)
    return ce + _EWC_LAMBDA * ewc, ntoks


def train_model_ewc(
    args,
    model: nn.Module,
    train_set,
    valid_set,
):
    """★2026-09-02 方案A: 直接复制官方 mlx_lm.lora.train_model(lora.py:216)流程,
    唯一改动 = train(..., loss=ewc_loss)。最小差异 → 若还 OOM 即 ewc_loss 本身问题。
    """
    mx.random.seed(args.seed)
    model.freeze()
    if args.num_layers > len(model.layers):
        raise ValueError(
            f"Requested to train {args.num_layers} layers "
            f"but the model only has {len(model.layers)} layers."
        )

    # 与官方一致: lora 转换(默认 fine_tune_type="lora", use_dora=False)
    linear_to_lora_layers(model, args.num_layers, args.lora_parameters)

    # ★ Fisher(ToM 突触重要性)→ 全局(ewc_loss 模块级函数使用)。必须在转换后、训练前。
    global _FISHER, _EWC_LAMBDA
    _FISHER = {}
    if getattr(args, "fisher_file", None) and os.path.isfile(args.fisher_file):
        raw = json.load(open(args.fisher_file, encoding="utf-8"))
        tp = _flat(model.trainable_parameters())
        for name in tp:
            if name in raw:
                _FISHER[name] = mx.array(raw[name])
        _EWC_LAMBDA = args.ewc_lambda
        print(f"EWC-A: 加载 {len(_FISHER)}/{len(tp)} 个 ToM 突触重要性")
    else:
        print("EWC-A: 无 fisher-file, 退化为纯 CE 训练")

    # Resume from weights if provided (与官方一致)
    if args.resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )

    lr = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
    optimizer = optim.Adam(learning_rate=lr)

    # ★ 唯一改动: loss=ewc_loss(官方默认 default_loss)
    train(
        model=model,
        args=training_args,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set) if len(valid_set) > 0 else None,
        loss=ewc_loss,
    )
    # ★2026-09-02 修复(自动化小测抓到): model.save_weights 会把**整个模型(15GB bf16)**覆写到
    #   adapters.safetensors(应为 ~58MB float32 LoRA) → SHY 巩固读 bf16 崩 TypeError + resume 加载 15GB 风险。
    #   官方 mlx_lm.lora 无此行——train() 内部按 save_every 周期保存(248 keys float32)即正确。已删除。


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EWC-A LoRA training (synaptic consolidation, in-training)")
    # 对齐 mlx_lm.lora 命令行(官方 CONFIG_DEFAULTS 字段)
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
    ap.add_argument("--val-batches", type=int, default=0)
    ap.add_argument("--lr-schedule", default=None)
    ap.add_argument("--test", action="store_true", default=False)  # mlx load_dataset 检查该字段
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-scale", type=float, default=20.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    # EWC 专属
    ap.add_argument("--fisher-file", default="tom-fisher.json")
    ap.add_argument("--ewc-lambda", type=float, default=1e-3)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if not args.train:
        raise SystemExit("需要 --train")
    # 构造官方 lora_parameters(对齐 CONFIG_DEFAULTS)
    args.lora_parameters = {"rank": args.lora_rank, "dropout": args.lora_dropout, "scale": args.lora_scale}

    # 1. 官方 load(与 run() 一致: 2 元组)
    print("Loading pretrained model")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})

    # 2. 数据集(官方 load_dataset)
    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    # 3. 训练(Fisher 在 train_model_ewc 内转换后加载)
    train_model_ewc(args, model, train_set, valid_set)
    print("EWC-A 训练完成")


if __name__ == "__main__":
    main()
