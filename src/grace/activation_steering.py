#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""activation_steering.py — S1.5 激活转向骨架（Grace V2.4→V2.5，2026-09-09）。

目标：器官输出（mood/好奇心/身份）从 prompt 层下潜到激活层——
  从真实运行采集对比对 → 计算方向向量 → 推理时注入隐状态。

数据需求：~32 对比样本/向量（远小于训练）。
前提（诚实标注）：隐状态提取需要**进程内生成**（夜班窗 in-process 路径）——
  mlx_lm server 模式不暴露激活。本骨架提供采集/计算/注入的纯函数核心，
  进程内接线留待夜班生成器改造（v1.1）。

向量库设计：
  steering-vectors.jsonl: {"name": "curious"/"anxious"/..., "vector": [...],
                           "pairs": N, "born_ts", "source": "formal-runs"}
"""
from __future__ import annotations
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEC_PATH = os.path.join(REPO, "exchange", "grace", "steering-vectors.jsonl")


def collect_pairs(journal_path: str, state_key: str, lo: float, hi: float) -> list:
    """从轨迹日志采集对比对：(高状态情境隐状态, 低状态情境隐状态)。

    state_key: "curiosity"(缺口激活) / "mood_valence" / "novelty" 等。
    前提：journal 条目需带隐状态（进程内生成时记录 hs）——骨架期返回空并说明。
    """
    pairs = []
    if not os.path.isfile(journal_path):
        return pairs
    hi_pool, lo_pool = [], []
    for l in open(journal_path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        v = (r.get("hidden_state") or {}).get(state_key)
        if v is None:
            continue
        (hi_pool if v >= hi else lo_pool if v <= lo else []).append(v)
    # 配对：随机高-低组合（骨架期 hi/lo 池为空 → 返回空，待进程内采集上线）
    import random
    random.seed(42)
    n = min(len(hi_pool), len(lo_pool))
    return list(zip(hi_pool[:n], lo_pool[:n]))


def steering_vector(pos: list, neg: list) -> list:
    """方向向量 = mean(pos) − mean(neg)，L2 归一。pos/neg: 隐状态向量列表。"""
    import numpy as np
    if not pos or not neg:
        return []
    p = np.mean(np.array(pos, dtype=np.float32), axis=0)
    n = np.mean(np.array(neg, dtype=np.float32), axis=0)
    v = p - n
    norm = float(np.linalg.norm(v)) + 1e-8
    return [round(float(x) / norm, 6) for x in v]


def apply_steering(hidden_states, vector, alpha: float = 1.0):
    """推理时注入：hidden += alpha × vector（逐 token 广播）。

    进程内用法（夜班生成器接线点）：
        hs = model(inputs)            # mlx 调用前 hook 每层残差流
        hs = apply_steering(hs, vec["vector"], alpha=0.8)
    """
    import numpy as np
    v = np.array(vector, dtype=np.float32)
    return [np.asarray(h, dtype=np.float32) + alpha * v for h in hidden_states]


def save_vector(name: str, vector: list, pairs: int, vec_path: str = VEC_PATH) -> None:
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)
    rows = []
    if os.path.isfile(vec_path):
        rows = [json.loads(l) for l in open(vec_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("name") != name]
    rows.append({"name": name, "vector": vector, "pairs": pairs,
                 "born_ts": __import__("time").time()})
    with open(vec_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_vector(name: str, vec_path: str = VEC_PATH) -> dict | None:
    if not os.path.isfile(vec_path):
        return None
    for l in open(vec_path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("name") == name:
            return r
    return None
