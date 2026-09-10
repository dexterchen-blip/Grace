#!/usr/bin/env python3
"""Grace V2.5 LoRA 巩固层——每日 adapter 堆叠 → 周期性合并（设计稿 §8/§14）。

脑科学同构: 睡眠巩固——白天(每日 LoRA)的学习夜里合并成"永久的"皮层记忆。
三模式:
  linear  任务算术: Δ = Σ w_i·Δ_i（保守默认）
  ties    修剪-符号共识-合并: 每任务保留 top-k 幅度, 逐位符号选举, 只平均同号者（抗干扰）
  dare    随机丢弃 p 并按 1/(1-p) 重缩放后平均（DARE 论文配方）

安全: 基座永不动(V6.1 权重不参与)——合并发生在 adapter 参数空间(ΔW = B·A·scale),
输出是"巩固层 adapter"（与每日轨同构, 可被后续每日轨叠加, 也可整体拔除回滚）。
SVD 秩保持: Δ 重分解为 A'/B' 时保留输入秩与(min(合并秩, r_max))的较小者。
"""
import os, sys, json, argparse, glob
import numpy as np

def _load_adapter_state(adapter_dir):
    """读 adapter 目录的 safetensors → dict（支持 mlx 与 torch 两种保存格式）。"""
    try:
        from safetensors.numpy import load_file
        st = {}
        for fp in sorted(glob.glob(os.path.join(adapter_dir, "*.safetensors"))):
            st.update(load_file(fp))
        return st
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"读取 {adapter_dir} 失败: {e}")

def _deltas(state):
    """从 adapter state 提取 (layer_key → (A, B, scale))。mlx_lm 命名: *.lora_A / *.lora_B / *.scale。"""
    out = {}
    scales = {k.rsplit(".", 1)[0]: float(np.asarray(v).reshape(-1)[0])
              for k, v in state.items() if k.endswith(".scale")}
    for k in state:
        if not k.endswith(".lora_A"):
            continue
        base = k[:-len(".lora_A")]
        bk = base + ".lora_B"
        if bk not in state:
            continue
        A = np.asarray(state[k], dtype=np.float64)
        B = np.asarray(state[bk], dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        out[base] = (A, B, scales.get(base, 1.0))
    return out

def _repack(delta, rank, scale):
    """Δ(out,in) → SVD 低秩重分解 → (A'(rank,in), B'(out,rank))。"""
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    r = min(rank, len(S))
    sq = np.sqrt(S[:r])
    A = (Vt[:r] * sq[:, None])
    B = (U[:, :r] * sq[None, :])
    return A, B, scale

def merge(adapters, weights=None, mode="linear", rank=8, drop_p=0.5, seed=42):
    """合并多个 adapter → 巩固层 state dict。adapters: [(dir, weight)]。"""
    if not adapters:
        raise ValueError("无 adapter 输入")
    w = weights or [1.0 / len(adapters)] * len(adapters)
    states = [(d, _load_adapter_state(d)) for d, _ in adapters]
    pairs = [_deltas(st) for _, st in states]
    keys = set()
    for p in pairs:
        keys |= set(p.keys())
    rng = np.random.default_rng(seed)
    merged, ranks = {}, {}
    for key in sorted(keys):
        ds, ws = [], []
        for pi, (dpath, wgt) in enumerate(adapters):
            if key not in pairs[pi]:
                continue
            A, B, scale = pairs[pi][key]
            ds.append((B @ A) * scale)
            ws.append(wgt)
        if not ds:
            continue
        D = np.stack(ds)                                   # (n, out, in)
        if mode == "linear":
            delta = sum(wi * Di for wi, Di in zip(ws, D))
        elif mode == "ties":
            k = max(1, int(D[0].size * (1 - drop_p)))
            flat = D.reshape(len(D), -1)
            mag = np.abs(flat)
            thr = np.sort(mag, axis=1)[:, -k]              # 每任务 top-k 保留
            trimmed = np.where(mag >= thr[:, None], flat, 0.0)
            sign = np.sign(trimmed.sum(axis=0))            # 逐位符号选举
            agree = (np.sign(trimmed) == sign[None, :]) & (trimmed != 0)
            merged_flat = (trimmed * agree).sum(axis=0) / np.maximum(agree.sum(axis=0), 1)
            delta = merged_flat.reshape(D.shape[1:])
        elif mode == "dare":
            mask = (rng.random(D.shape) > drop_p).astype(D.dtype) / (1 - drop_p)
            delta = (D * mask * np.asarray(ws)[:, None, None]).sum(axis=0)
        else:
            raise ValueError(mode)
        # 秩策略: linear=输入秩之和(任务算术保真, 两个rank-8之和是rank-16, 截8丢信息——
        #   实测重构误差0.38的教训), 上限64; ties/dare 靠稀疏化抗干扰, 保持目标秩。
        if mode == "linear":
            r_eff = min(sum(p_[key][0].shape[0] for p_ in pairs if key in p_), 64)
        else:
            r_eff = rank
        A, B, scale = _repack(delta, max(r_eff, 1), 1.0)
        merged[key + ".lora_A"] = A.astype(np.float32)
        merged[key + ".lora_B"] = B.astype(np.float32)
        merged[key + ".scale"] = np.array([scale], dtype=np.float32)   # safetensors 需数组非标量
        ranks[key] = int(A.shape[0])
    # 非 LoRA 键(scales 等)从首个 adapter 透传
    for k, v in states[0][1].items():
        if not (k.endswith(".lora_A") or k.endswith(".lora_B")):
            merged.setdefault(k, v)
    return merged, ranks

def consolidate(adapter_dirs, out_dir, weights=None, mode="linear", rank=8, drop_p=0.5):
    os.makedirs(out_dir, exist_ok=True)
    adapters = [(d, w) for d, w in zip(adapter_dirs, weights or [None] * len(adapter_dirs))]
    adapters = [(d, w if w is not None else 1.0 / len(adapter_dirs)) for d, w in adapters]
    merged, ranks = merge(adapters, weights=[w for _, w in adapters], mode=mode,
                          rank=rank, drop_p=drop_p)
    from safetensors.numpy import save_file
    save_file({k: np.asarray(v) for k, v in merged.items()},
              os.path.join(out_dir, "adapters.safetensors"))
    cfg = os.path.join(adapter_dirs[0], "adapter_config.json")
    if os.path.isfile(cfg):
        json.dump(json.load(open(cfg)), open(os.path.join(out_dir, "adapter_config.json"), "w"),
                  ensure_ascii=False, indent=1)
    meta = {"mode": mode, "sources": adapter_dirs, "weights": [w for _, w in adapters],
            "rank": rank, "drop_p": drop_p, "layers": len(ranks), "ts": time.time()}
    json.dump(meta, open(os.path.join(out_dir, "consolidation-meta.json"), "w"),
              ensure_ascii=False, indent=1)
    return meta

import time

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="linear", choices=["linear", "ties", "dare"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--drop-p", type=float, default=0.5)
    a = ap.parse_args()
    meta = consolidate(a.adapters, a.out, a.weights, a.mode, a.rank, a.drop_p)
    print(f"[consolidate] {a.mode}: {len(a.adapters)} adapter → {a.out} "
          f"({meta['layers']} 层位合并)")
