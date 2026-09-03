"""ewc_consolidate.py — 训练后「突触稳态巩固」(2026-09-02 A 重构: EWC+SHY 合一).

原理(脑科学: BCM 突触缩放 / 突触稳态——强活跃不缩, 弱不活跃缩):
  记忆训练(mlx 官方管线,零改动)写入了所有 LoRA 突触(包括 ToM 重要方向)。
  训练后一步巩固 = 单一"突触稳态函数":
    θ ← θ · clip(1 − α_f·(1−F̂) − α_s·exp(−|θ|/τ), 0.5, 1.0)
    F̂ = F/maxF ∈[0,1] (Fisher 归一化重要性 = 该参数对 ToM 的活跃度代理)
    · 高 F(F̂→1):  第一项→0 → 不动 = 保护 ToM 能力(原 EWC 角色, 防覆盖)
    · 小 |θ|:      exp(−|θ|/τ)→1 → 第二项→α_s → 剪 = 洗噪声(原 SHY 角色)
    · 连续函数替代硬分层(noise/keep floor 绝对阈值废弃) —— 同一稳态法则的两面
    τ = 权重 |θ| 分布特征尺度(自适应自动估, 零旋钮)

  等效于: 白天记忆编码(LoRA), 夜间突触稳态(一个函数护能力+洗噪) —— 更贴人脑
  旧版 EWC 回缩对高 F 参数也收缩(F-norm 几乎不变=保护空转的根因);
  新版高 F 参数 F̂→1 时 g→1 真不动 → 保护真正生效。

用法(CLI 兼容):
  ./run.sh .venv/bin/python3 v2/stress/ewc_consolidate.py \
      --adapter <adapter_path/adapters.safetensors> \
      --fisher <tom-fisher.json> [--alpha 0.7] [--delta 0.03]
  --alpha = α_f(Fisher 保护强度, 护 ToM; 0=关闭)   --delta = α_s(稳态剪枝强度, 洗噪; 0=关闭)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# 突触稳态函数下限(防过度收缩; 0.5 = 单夜最多缩一半)
_CLIP_LO = 0.5


def _tau_from(weights: dict) -> float:
    """τ 自适应: 全局 |θ| 分布的 68 分位(≈1σ 尺度)——噪声 |θ|≪τ 被剪, 记忆 |θ|≳τ 豁免。
    LoRA 权重尺度随训练漂移, 绝对阈值(旧 noise/keep floor)会失效 → 分布自适应。"""
    all_a = np.concatenate([np.abs(w).ravel() for w in weights.values()]) if weights else np.array([1e-4])
    return float(np.percentile(all_a, 68)) if all_a.size else 1e-4


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-train synaptic homeostasis (EWC+SHY unified)")
    ap.add_argument("--adapter", required=True, help="adapter safetensors 路径")
    ap.add_argument("--fisher", default="tom-fisher.json", help="ToM Fisher 重要性 json")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="α_f: Fisher 保护强度(高 F 参数不动, 护 ToM 防覆盖; 0=关闭). 默认 0.7")
    ap.add_argument("--delta", type=float, default=0.03,
                    help="α_s: 稳态剪枝强度(小 |θ| 噪声按 exp(−|θ|/τ) 衰减; 0=关闭). 默认 0.03")
    args = ap.parse_args()

    if not os.path.isfile(args.adapter) or not os.path.isfile(args.fisher):
        print(f"[consolidate] 缺文件: adapter={os.path.isfile(args.adapter)} fisher={os.path.isfile(args.fisher)}")
        return

    # 1. 读 adapter 权重(★2026-09-01: 记录原 dtype——numpy 会把 fp16/fp32 提升 float64, 保存前还原)
    with safe_open(args.adapter, framework="np") as f:
        weights = {k: f.get_tensor(k) for k in f.keys()}
        dtypes = {k: f.get_tensor(k).dtype for k in f.keys()}

    # 2. 读 Fisher(嵌套 list → np 数组)
    fisher_raw = json.load(open(args.fisher, encoding="utf-8"))
    fisher = {k: np.asarray(v, dtype=np.float32) for k, v in fisher_raw.items()}

    # 3. τ 自适应估计
    tau = _tau_from(weights)
    _before = float(np.sqrt(sum(float((v.astype(np.float32) ** 2).sum()) for v in weights.values())))

    # 4. ★2026-09-02 A 重构: 突触稳态——分工正确版(数学验证修正):
    #    consolidate 只做两件事, 不缩记忆:
    #    ① 保护: 高 F(ToM 重要, F̂>保阈) 完全不碰(g=1)——防覆盖(EWC 本质, 不是"缩低F")
    #    ② 剪噪: 低 F 且 |θ|≪τ 的噪声参数轻剪(α_s·exp(−|θ|/τ))——洗噪(SHY)
    #    低 F 大 |θ|(刚写入的记忆): g≈1 不动——记忆塑造不被 consolidate 抵消(旧版把低 F 缩 70% = 杀记忆)
    #    g = 1 − α_s·(1−F̂)·exp(−|θ|/τ)  (α_s=0 或全高F 时无操作)
    tau = _tau_from(weights)
    _before = float(np.sqrt(sum(float((v.astype(np.float32) ** 2).sum()) for v in weights.values())))
    n_f_protect = n_s_prune = 0
    for k, w in weights.items():
        if args.delta <= 0:
            break
        if k not in fisher:
            continue
        F = fisher[k]
        if F.shape != w.shape:
            print(f"[consolidate] ⚠ shape 不匹配 {k}: F={F.shape} w={w.shape},跳过")
            continue
        fmax = float(F.max()) if F.size else 0.0
        fhat = (F / fmax) if fmax > 0 else np.zeros_like(F, dtype=np.float32)
        # 保护门控: alpha>0 时高 F 完全不剪; alpha=0(SHY-only) 时全部参数按 |θ| 剪
        gate = (1.0 - fhat) if args.alpha > 0 else np.ones_like(fhat, dtype=np.float32)
        prune = args.delta * np.exp(-np.abs(w) / max(tau, 1e-9))
        g = 1.0 - gate * prune
        w = w * np.clip(g, _CLIP_LO, 1.0)
        weights[k] = w
        if args.alpha > 0:
            n_f_protect += 1
        n_s_prune += 1

    # 5. 写回(dtype 还原防 float64)
    for k in weights:
        if weights[k].dtype != dtypes[k]:
            weights[k] = weights[k].astype(dtypes[k])
    # ★2026-09-02 审计修复: EWC 默认开 → 每日原地覆写 adapter, 覆写前留 .pre 备份
    #   (防 consolidate 异常污染 LoRA 链; .pre 不参与加载, 仅人工回滚用)
    try:
        import shutil as _sh
        _sh.copy2(args.adapter, args.adapter + ".pre")
    except Exception as _bke:  # noqa: BLE001
        print(f"[consolidate] ⚠ 备份失败(继续不阻断): {_bke}")
    save_file(weights, args.adapter)
    _after = float(np.sqrt(sum(float((v.astype(np.float32) ** 2).sum()) for v in weights.values())))
    print(f"[consolidate] ✅ 突触稳态: {n_f_protect} 参护F + {n_s_prune} 参剪噪"
          f"(α_f={args.alpha} α_s={args.delta} τ={tau:.2e}) F-norm {_before:.4e}→{_after:.4e}")


if __name__ == "__main__":
    main()
