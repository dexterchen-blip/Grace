#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""willingness.py — 训练意愿器官 v1.0 骨架（Grace V2.6，2026-09-11）。

设计依据：《Grace-V2.5-自涌现迭代设计-2026-09-09.md》P3（训练意愿）
    "LoRA 训练意愿: 她自己判断'值得记住'（新奇积分+情绪峰+缺口解决量的她自己的阈值）
     → 训练请求 → 夜班硬件窗执行（内容照过铁律滤）"
    + 争点收敛记录（2026-09-10）：外部通量乘子已被【孤独泵】取代（用户方案优于乘子）；
    R5 约束：自服务解决打折、仅外部填充全额入账。

与本体系其他件的关系：
    - 体量触发器（volume_trigger.sh, 阈值 300 唯一文本）= 【量的门】——未筛够多少才开检查点；
    - 本模块 = 【质的信号】——"这批经历值不值得进权重"（她判断的原则①载体）。
      L1 外挂版：阈值/权重是 T0 参数（我们设）；
      矩阵化版：换成她的体感（"她感觉到该学了"）——衔接点已定案。
    - 孤独机制不乘进本公式（泵在恢复环自己转，两个账本分开）。

纯 stdlib。环境门控：GRACE_WILLINGNESS=1。
【接线位（惰性，未接）】检查点判官完成 → 本模块对 pass 批算意愿 → 过阈才入训练队列
    （现状：触发器按体量直接训练——质与量合流是接线时的升级点）。
"""
from __future__ import annotations
import json
import os
import time

# ---------------- T0 配置（判定不可演化：权重与阈值永不进 direction.md） ----------------
WIL = {
    "w_novelty": 0.35,       # 新奇积分权重
    "w_mood_peak": 0.25,     # 情绪峰权重
    "w_gap_resolved": 0.25,  # 缺口解决量权重
    "w_cog_density": 0.15,   # cog 密度权重（§2.5 第四信号：暗注意力想的越多=经历越稠）
    "threshold": 0.60,       # 训练意愿过阈线（"值得记住"门）
    "self_resolution_discount": 0.4,   # R5：她自服务解决的缺口按 40% 计（仅外部填充全额）
    "queue_cap": 200,
}


def _now() -> float:
    return time.time()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ---------------- 四信号 → 意愿分 ----------------
def compute(novelty_integral: float, mood_peak: float,
            gap_resolved_external: int, gap_resolved_self: int = 0,
            cog_density: float = 0.0) -> dict:
    """四信号合成训练意愿（§2.5 原文四项：新奇积分/情绪峰/缺口解决数/cog 密度）。

    novelty_integral: 当期新奇积分（0-1 归一——接线层从 neural_memory/好奇心账本取）
    mood_peak:        当期情绪峰强度（0-1——mood_engine intraday 峰值幅度）
    gap_resolved_external: 外部填充解决的缺口数（主人回答/新真实 L0）
    gap_resolved_self:     她自服务解决的缺口数（R5 打折）
    cog_density:      cog 暗流密度（0-1 归一——当日 hidden 念头量/思考长度，接线层取）
    """
    g_res = (gap_resolved_external
             + WIL["self_resolution_discount"] * gap_resolved_self)
    gap_n = _clamp01(g_res / 5.0)                     # 5 个当量缺口 ≈ 满格
    n = _clamp01(novelty_integral)
    m = _clamp01(mood_peak)
    c = _clamp01(cog_density)
    w = (WIL["w_novelty"] * n
         + WIL["w_mood_peak"] * m
         + WIL["w_gap_resolved"] * gap_n
         + WIL["w_cog_density"] * c)
    return {
        "willingness": round(w, 3),
        "should_train": w >= WIL["threshold"],
        "factors": {"novelty": round(n, 3), "mood_peak": round(m, 3),
                    "gap_resolved": round(gap_n, 3), "cog_density": round(c, 3),
                    "g_ext": gap_resolved_external, "g_self": gap_resolved_self},
        "ts": _now(),
    }


# ---------------- 训练请求队列（夜班/检查点消费） ----------------
def queue_request(root: str, factors: dict, reason: str) -> dict:
    """意愿过阈 → 训练请求入队（夜班硬件窗执行，物理窗不受意愿影响）。"""
    q_path = os.path.join(root, "training-queue.jsonl")
    os.makedirs(root, exist_ok=True)
    rows = []
    if os.path.isfile(q_path):
        rows = [json.loads(l) for l in open(q_path, encoding="utf-8") if l.strip()]
    rows = rows[-WIL["queue_cap"]:]
    row = {"ts": _now(), "reason": reason[:120], "factors": factors,
           "status": "pending"}
    rows.append(row)
    with open(q_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"queued": True, "position": len(rows),
            "pending": sum(1 for r in rows if r["status"] == "pending")}


def pending(root: str) -> list[dict]:
    q_path = os.path.join(root, "training-queue.jsonl")
    if not os.path.isfile(q_path):
        return []
    return [json.loads(l) for l in open(q_path, encoding="utf-8")
            if l.strip() and json.loads(l).get("status") == "pending"]
