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

# ★§13 KV 双轴·层轴（2026-09-09 用户："给不同的模型层加不同的 KV"）：
#   器官状态按层位注入——慢变量（persona/mood 底色）进早层、快变量（wm/gap）进
#   中段循环层、表达整形（语气/称呼/ToM 微调）进后段。与脑的时间尺度分层同构。
#   层位以 Qwen3.8-27B（64 层，架构族 qwen3_5）为基准；R 线循环段=[32,36)。
LAYER_BANDS = {"early": (1, 24), "loop": (32, 36), "late": (48, 64)}


def band_layers(band: str, n_layers: int = 64) -> list:
    """层带 → 层索引列表。band ∈ {early, loop, late}；越界自动夹取。"""
    if band not in LAYER_BANDS:
        raise ValueError(f"未知层带: {band}（可选 {list(LAYER_BANDS)}）")
    lo, hi = LAYER_BANDS[band]
    return [i for i in range(max(0, lo), min(hi, n_layers))]


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


def apply_steering(hidden_states, vector, alpha: float = 1.0, layers: list | None = None):
    """推理时注入：hidden += alpha × vector（逐 token 广播）。

    layers: ★§13 层位选择。None=全部层；传层索引列表（如 band_layers("early")）
    则只注入指定层——器官更新频率与层位绑定的实现点。

    进程内用法（夜班生成器接线点）：
        hs = model(inputs)            # mlx 调用前 hook 每层残差流
        hs = apply_steering(hs, vec["vector"], alpha=0.8, layers=vec.get("layers"))
    """
    import numpy as np
    v = np.array(vector, dtype=np.float32)
    if layers is None:
        return [np.asarray(h, dtype=np.float32) + alpha * v for h in hidden_states]
    out = []
    for i, h in enumerate(hidden_states):
        h = np.asarray(h, dtype=np.float32)
        if i in layers:
            h = h + alpha * v
        out.append(h)
    return out


def save_vector(name: str, vector: list, pairs: int, vec_path: str = VEC_PATH,
                band: str | None = None) -> None:
    """★band: 层带名（early/loop/late，§13 层轴）——向量按层位存储。"""
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)
    rows = []
    if os.path.isfile(vec_path):
        rows = [json.loads(l) for l in open(vec_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("name") != name]
    rows.append({"name": name, "vector": vector, "pairs": pairs,
                 "band": band, "layers": band_layers(band) if band else None,
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
