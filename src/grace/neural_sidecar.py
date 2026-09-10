#!/usr/bin/env python3
"""Grace V2.5 N3: delta-rule 联想记忆 sidecar（设计稿 §14.1, Titans 台阶A 重试）。

与 v1 失败版(在线自编码器)的本质区别: 联想记忆损失——W 从 key 直接重构 value,
更新是闭式 delta rule(在线一步, 无迭代优化), 惊奇门控写入(Titans 惊奇度=预测误差)。

动力学:
  recall(k)     = W @ k
  surprise(k,v) = ||W@k − v|| / (||v||+ε)            ← 惊奇度(信息预测误差)
  update:       W ← W + η_eff · (v − W@k) ⊗ k / (||k||²+ε)    ← delta rule
  η_eff         = η · (surprise 压缩进 [η_lo, η_hi])   ← 惊奇门控: 意外才强写
  decay:        W ← (1−λ)·W (周期性)                  ← 自适应遗忘(过时信息淡出)

与器官咬合: 惊奇门控信号与好奇心账本同源(IPE/缺口)——两器官共享"什么是意外"。
持久化: npz(W 矩阵)。维度: bge-m3 d=1024 → W 4MB(线性映射, 蚊子腿)。
"""
import os
import numpy as np

class DeltaSidecar:
    def __init__(self, d: int = 1024, eta: float = 0.5, lam: float = 1e-4,
                 eta_lo: float = 0.05, eta_hi: float = 1.0, path: str | None = None):
        self.d, self.eta, self.lam = d, eta, lam
        self.eta_lo, self.eta_hi = eta_lo, eta_hi
        self.W = np.zeros((d, d), dtype=np.float32)
        self.n_updates = 0
        if path and os.path.isfile(path):
            z = np.load(path)
            self.W = z["W"]; self.n_updates = int(z["n"])

    # ---------- 核心动力学 ----------
    def recall(self, k: np.ndarray) -> np.ndarray:
        return self.W @ k

    def surprise(self, k: np.ndarray, v: np.ndarray) -> float:
        err = self.recall(k) - v
        return float(np.linalg.norm(err) / (np.linalg.norm(v) + 1e-8))

    def update(self, k: np.ndarray, v: np.ndarray, surprise: float | None = None) -> float:
        """写入一条 (k→v)。返回该条的惊奇度(供账本/日志消费)。"""
        s = self.surprise(k, v) if surprise is None else surprise
        # 惊奇门控: 平淡(已会)→弱写, 意外(新奇)→强写。s∈[0,∞] → η_eff∈[η_lo, η_hi]
        gate = self.eta_lo + (self.eta_hi - self.eta_lo) * (s / (s + 1.0))
        resid = v - self.W @ k
        kk = k / (np.dot(k, k) + 1e-8)
        self.W += (self.eta * gate * np.outer(resid, kk)).astype(np.float32)
        self.n_updates += 1
        return s

    def decay(self, steps: int = 1):
        """自适应遗忘: 未被重访的记忆随时间淡出(weight decay 式)。"""
        for _ in range(steps):
            self.W *= (1.0 - self.lam)

    # ---------- 持久化 ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, W=self.W, n=np.int64(self.n_updates))

    # ---------- 指标 ----------
    def fidelity(self, keys: list, vals: list) -> float:
        """已存键值对的重构保真度(1−相对误差)。"""
        if not keys:
            return 1.0
        errs, refs = 0.0, 0.0
        for k, v in zip(keys, vals):
            errs += np.linalg.norm(self.recall(k) - v)
            refs += np.linalg.norm(v)
        return 1.0 - errs / (refs + 1e-8)

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.expanduser(
        "~/WorkBuddy/watch/ai-sandbox-stress/experiments/lora/sidecar.npz"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        # ★动力学自测(与 v1 失败教训对照): delta rule η<1 单步不收敛——重复接触才收敛
        #   = 习惯化。空记忆首条惊奇=1(全意外), 同条反复写惊奇递减, 多 epoch 收敛。
        rng = np.random.default_rng(0)
        m = DeltaSidecar(d=256)
        K = [rng.normal(size=256) for _ in range(64)]
        V = [rng.normal(size=256) for _ in range(64)]
        curve = [m.update(K[0], V[0]) for _ in range(6)]
        print("习惯化曲线(同条×6 惊奇):", [f"{x:.2f}" for x in curve])
        for ep in range(8):
            for k, v in zip(K, V):
                m.update(k, v)
            if ep in (0, 3, 7):
                print(f"  epoch {ep+1}: 保真度 {m.fidelity(K, V):.3f}")
        knew, vnew = rng.normal(size=256), rng.normal(size=256)
        print(f"熟键惊奇: {m.surprise(K[0], V[0]):.3f} | 陌生键惊奇: {m.surprise(knew, vnew):.3f}")
        m.decay(steps=3000)
        print(f"衰减后保真度: {m.fidelity(K, V):.3f}")
        m.save(a.path)
        m2 = DeltaSidecar(d=256, path=a.path)
        print(f"持久化往返: {np.allclose(m.W, m2.W)} | 总更新 {m2.n_updates}")
