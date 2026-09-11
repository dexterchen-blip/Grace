#!/usr/bin/env python3
"""V2.5 循环段（沙盒引擎版）——R0 定案形态 [32,40)×K=2 的层表重排。

移植自 Grace-repo/experiments/recurrent/loop_model.py 的核心三件
(plan_loop / _layers_container / apply_loop, R0/R2/rc2/A-B 全链实测零失败),
供 stress_engine GRACE_V25=1 时对进程内受测模型挂循环段。
机制: 纯内存对象引用重排——零权重复制 / 零梯度路径 / 重新 load 即完全还原。
"""
from __future__ import annotations

PERIOD = 4  # Qwen3.5 混合注意力周期: 3 linear + 1 full


def plan_loop(n_layers: int, start=None, end=None, k: int = 2,
              align_period: bool = True) -> dict:
    if start is None:
        mid = n_layers // 2
        start = (mid // PERIOD) * PERIOD if align_period else mid - 3
    if end is None:
        end = start + PERIOD if align_period else mid + 3
    if not (0 < start < end <= n_layers):
        raise ValueError(f"段选择非法: start={start}, end={end}, n={n_layers}")
    if k < 2:
        raise ValueError("循环至少 K=2")
    order = list(range(start)) + list(range(start, end)) * k + list(range(end, n_layers))
    return {"n_layers": n_layers, "start": start, "end": end, "k": k,
            "seg_len": end - start, "order": order, "new_len": len(order),
            "front_untouched": list(range(start)) == order[:start],
            "weight_bytes_unchanged": True}


def _layers_container(model):
    for path in ("language_model.model", "model", ""):
        obj = model
        try:
            for attr in [a for a in path.split(".") if a]:
                obj = getattr(obj, attr)
            if hasattr(obj, "layers"):
                return obj
        except AttributeError:
            continue
    raise AttributeError("找不到 layers 容器（架构路径变化需更新 _layers_container）")


def apply_loop(model, start=32, end=40, k: int = 2) -> dict:
    """对已加载模型执行层表重排（原地内存操作）。返回排布报告。
    结构浓度断言内建: 层对象零复制(引用同一组对象)。"""
    inner = _layers_container(model)
    layers = list(inner.layers)
    plan = plan_loop(len(layers), start, end, k)
    new_layers = [layers[i] for i in plan["order"]]
    inner.layers = new_layers
    cfg = getattr(inner, "config", None) or getattr(model, "config", None)
    for f in ("num_hidden_layers", "n_layers", "num_layers"):
        if cfg is not None and hasattr(cfg, f):
            try:
                setattr(cfg, f, plan["new_len"])
            except Exception:  # noqa: BLE001
                pass
    objs = {id(x) for x in new_layers}
    assert len(objs) == len(layers), "结构浓度断言失败: 层对象必须零复制"
    plan = dict(plan)
    plan["struct_ok"] = True
    return plan
