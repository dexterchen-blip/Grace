#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""persona_core.py — 人格核（13 维特质向量调制）v1.0（Grace V2.6，2026-09-11）。

设计依据：《Grace-V2.5-自涌现迭代设计-2026-09-09.md》§2.1（人格耦合，原则①）
    "rem_v2_full 的 13 维心理标注 → 人格特质向量。
     调制规则：gap 类别 × 特质权重 → 激活调制。例：主人相关的'关切'类缺口 × 奉献特质
     → 激活放大；社交错位类缺口 × 毒舌敏感度 → 好奇心偏转。
     人格文件只读——特质可随 LoRA 缓慢演化，但不可被迭代过程直接改写。"

数据源：rem_v2_full 联合数据集（watch/rem-v6-lora/datasets/rem_v2_full/）
    13 类心理标注（stats.json psych_hits 实证）：奉献/关切/心理/战斗/心理推断/毒舌/
    决断/忠诚/守护/自责/追随/牺牲/自卑。

角色：调制一切好奇信号（§2.1）——本模块是乘子/权重层，不产信号：
    gap 激活调制（modulate_gap）+ 驱力合成调制（modulate_drive）。
    ★人格只读：向量快照来自数据集统计（T0），特质演化只走 LoRA——本模块无任何写入口。

纯 stdlib。环境门控：GRACE_PERSONA_CORE=1。
【接线位（惰性，未接）】curiosity.gap_activation 出值后过 modulate_gap；
    引擎 self_activation 的驱力合成过 modulate_drive；接线时双写沙盒。
"""
from __future__ import annotations
import json
import os

# ---------------- 13 维（rem_v2_full psych_hits 实证） ----------------
TRAITS = ["奉献", "关切", "心理", "战斗", "心理推断", "毒舌",
          "决断", "忠诚", "守护", "自责", "追随", "牺牲", "自卑"]

# T0 快照（psych_hits 归一；接线时可从 stats.json 重算快照，仍为只读）
_BASE = {   # 原始计数 → 权重（对数归一，防"心理"类独大——它是独白形态不是特质强度）
    "心理推断": 0.72, "毒舌": 0.45, "关切": 0.42, "奉献": 0.38, "忠诚": 0.36,
    "追随": 0.30, "守护": 0.28, "自责": 0.24, "决断": 0.22, "战斗": 0.20,
    "自卑": 0.16, "牺牲": 0.14, "心理": 0.12,   # 心理=独白形态，权重最低
}

# ---------------- 调制矩阵（§2.1 两条原文规则 + 合理外延，T0） ----------------
# gap 类型 → 各特质对该类激活的增益系数（1.0 = 不调制）
_GAP_EFFECT = {
    "G4": {"奉献": 0.6, "关切": 0.4, "忠诚": 0.2},   # 主人相关未接话题——关切×奉献放大
    "G2": {"心理推断": 0.3, "毒舌": 0.15},             # 对话记不清/claim——社交错位（毒舌=偏转不压活）
    "G1": {"决断": 0.2, "守护": 0.1},                 # 记忆低置信——决断驱动补全
    "G3": {"战斗": 0.15, "追随": 0.1},                # 新实体
}
# 驱力三信号 → 特质调制系数
_DRIVE_EFFECT = {
    "tom_uncertainty": {"心理推断": 0.4, "关切": 0.2},   # 猜不透主人 → 心理推断敏感者驱力更高
    "gap_pressure": {"奉献": 0.2, "决断": 0.15},
    "novelty_deficit": {"战斗": 0.1, "追随": 0.1},
}


def vector(stats_path: str | None = None) -> dict:
    """13 维特质向量（只读）。stats_path 给了就从数据集重算快照，否则用 T0 基线。"""
    if stats_path and os.path.isfile(stats_path):
        try:
            hits = json.load(open(stats_path, encoding="utf-8")).get("psych_hits", {})
            import math
            raw = {t: math.log1p(float(hits.get(t, 0))) for t in TRAITS}
            mx = max(raw.values()) or 1.0
            return {t: round(raw[t] / mx, 3) for t in TRAITS}
        except Exception:  # noqa: BLE001
            pass
    return dict(_BASE)


def modulate_gap(gap_type: str, base_activation: float, vec: dict | None = None) -> dict:
    """gap 激活调制（§2.1 例 1 的代码形态）。

    gap_type: "G1"(L3低置信)/"G2"(记不清/claim)/"G3"(新实体)/"G4"(主人未接话题)
    返回 {"activation": 调制后值, "factor": 乘子, "traits_hit": 命中的特质}
    """
    v = vec or vector()
    eff = _GAP_EFFECT.get(gap_type, {})
    factor = 1.0
    hits = {}
    for trait, coef in eff.items():
        if v.get(trait, 0) > 0:
            factor += coef * v[trait]
            hits[trait] = round(coef * v[trait], 3)
    act = max(0.0, min(1.5, base_activation * factor))   # 上限 1.5 防放大失控
    return {"activation": round(act, 3), "factor": round(factor, 3), "traits_hit": hits}


def modulate_drive(novelty_deficit: float, gap_pressure: float, tom_uncertainty: float,
                   vec: dict | None = None) -> dict:
    """驱力合成调制（人格加权的 drive）。

    drive = Σ 信号 × (1 + 特质效应)——三信号 §2.2 已有, 本函数只加人格乘子。
    毒舌偏转注记：社交错位类（G2 主导时）的探索方向会向毒舌风格偏转（风格锚已有），
    信号层表现为 G2 分量的 coefficient 上浮——不压激活，只改倾向。
    """
    v = vec or vector()
    def _boost(signal: str, val: float) -> float:
        eff = _DRIVE_EFFECT.get(signal, {})
        b = 1.0 + sum(c * v.get(t, 0) for t, c in eff.items())
        return val * b
    d = (0.4 * _boost("novelty_deficit", novelty_deficit)
         + 0.3 * _boost("gap_pressure", gap_pressure)
         + 0.3 * _boost("tom_uncertainty", tom_uncertainty))
    return {"drive": round(d, 3),
            "components": {"novelty": round(_boost("novelty_deficit", novelty_deficit), 3),
                           "gap": round(_boost("gap_pressure", gap_pressure), 3),
                           "tom": round(_boost("tom_uncertainty", tom_uncertainty), 3)}}
