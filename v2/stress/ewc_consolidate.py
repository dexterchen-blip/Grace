"""ewc_consolidate.py — 训练后「突触回缩」(方案B,替代训练时 EWC)。

原理(脑科学: 突触巩固 / 睡眠期整合):
  记忆训练(mlx 官方管线,零改动)写入了所有 LoRA 突触(包括 ToM 重要方向)。
  训练后做一步巩固: 对高 Fisher(ToM 重要)的参数按比例收缩,低 Fisher(记忆)几乎不动
    θ_i ← θ_i × (1 − α·F_i/maxF)
  等效于 EWC 的梯度收缩,但:
    · 不改 mlx 一行代码 → 无 OOM/框架摩擦
    · 训练 = 白天记忆编码,回缩 = 夜间突触巩固(更贴人脑)

用法:
  ./run.sh .venv/bin/python3 v2/stress/ewc_consolidate.py \
      --adapter <adapter_path/adapters.safetensors> \
      --fisher <tom-fisher.json> [--alpha 0.7]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-train synaptic consolidation (EWC-B)")
    ap.add_argument("--adapter", required=True, help="adapter safetensors 路径")
    ap.add_argument("--fisher", default="tom-fisher.json", help="ToM Fisher 重要性 json")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="收缩强度(高 Fisher 收缩比例,0-1;默认 0.7)")
    ap.add_argument("--delta", type=float, default=0.0,
                    help="★SHY 睡眠全局下调率(突触稳态假说 Tononi&Cirelli): 弱突触按 δ 衰减,强突触豁免;0=关闭")
    args = ap.parse_args()

    if not os.path.isfile(args.adapter) or not os.path.isfile(args.fisher):
        print(f"[consolidate] 缺文件: adapter={os.path.isfile(args.adapter)} fisher={os.path.isfile(args.fisher)}")
        return

    # 1. 读 adapter 权重
    with safe_open(args.adapter, framework="np") as f:
        weights = {k: f.get_tensor(k) for k in f.keys()}

    # 2. 读 Fisher(嵌套 list → np 数组)
    fisher_raw = json.load(open(args.fisher, encoding="utf-8"))
    fisher = {k: np.asarray(v, dtype=np.float32) for k, v in fisher_raw.items()}

    # 3. 归一化 + 回缩: θ ← θ·(1 − α·F/maxF)
    n_scaled = 0
    for k, w in weights.items():
        if k not in fisher:
            continue
        F = fisher[k]
        if F.shape != w.shape:
            print(f"[consolidate] ⚠ shape 不匹配 {k}: F={F.shape} w={w.shape},跳过")
            continue
        fmax = float(F.max()) if F.size else 0.0
        # ★SHY 突触稳态下调(2026-09-01, 治 30 天寿命): 全局按比例衰减,但强突触豁免
        #   scale_sleep = 1 - δ×(1 - |θ|/θ_max) —— 弱参数衰减 δ,强参数(≈θ_max)豁免
        #   净效应: 污染弱信号每天被洗掉,核心人格/记忆保留,adapter 不饱和
        if args.delta > 0 and w.size:
            wmax = float(np.abs(w).max())
            if wmax > 0:
                scale_sleep = 1.0 - args.delta * (1.0 - np.abs(w) / wmax)
                w = w * scale_sleep
        # EWC: ToM 方向拉回(护能力,防覆盖)
        if fmax > 0:
            scale_ewc = 1.0 - args.alpha * (F / fmax)
            w = w * scale_ewc
            n_scaled += 1
        weights[k] = w

    # 4. 写回
    save_file(weights, args.adapter)
    _sleep = f"+SHY 全局下调 δ={args.delta}" if args.delta > 0 else ""
    print(f"[consolidate] ✅ 突触巩固完成: {n_scaled}/{len(weights)} 参数(α={args.alpha}{_sleep})")


if __name__ == "__main__":
    main()
