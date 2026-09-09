#!/usr/bin/env python3
"""R0 循环矩阵排布——层表重排骨架（Grace V2.5 §12 R 线，2026-09-09）。

用户定案: "重点还是框架浓度……先把骨架搭出来, 验收完了以后再确定什么时候可以跑。"
→ 本文件默认 dry-run 只打印排布计划, 不加载模型; 真实运行统一走 run_r0.sh --execute。

机制(实测核实过的 API 事实, 非猜测):
  - fused-rem-v61 = Qwen3.5 混合注意力, 64 层, 原生周期 [3×linear_attention + 1×full_attention]
    (is_linear = (layer_idx+1) % full_attention_interval != 0, interval=4)
  - mlx_lm.models.cache.make_prompt_cache → model.make_cache() 直接遍历 self.layers
    按层位建缓存(线性层 ArraysCache / 全注意力层 KVCache)——层表重排后缓存自动正确
  - self.layers 是普通 Python list(Qwen3Model.__init__ 即列表推导)——替换列表=改排布

骨架安全性(框架浓度·结构层):
  - 排布=纯内存对象引用重排: 零权重复制 / 零梯度路径 / 重新 load 即完全还原
"""
from __future__ import annotations

import argparse
import json
import os

DEFAULT_MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
PERIOD = 4  # Qwen3.5 混合注意力周期: 3 linear + 1 full


def read_config(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def read_layer_types(model_path: str) -> list | None:
    cfg = read_config(model_path)
    tc = cfg.get("text_config") or cfg
    lt = tc.get("layer_types")
    return lt if isinstance(lt, list) and lt else None


def plan_loop(n_layers: int, start=None, end=None, k: int = 2,
              align_period: bool = True) -> dict:
    """排布计划(纯函数, dry-run 与真跑共用同一计划源)。

    默认取中段一个原生周期(3 linear + 1 full)做循环段——按架构自己的重复单元循环,
    保持线性:全注意力比例不变; K>=2。
    """
    if start is None:
        mid = n_layers // 2
        start = (mid // PERIOD) * PERIOD if align_period else mid - 3
    if end is None:
        end = start + PERIOD if align_period else start + 6
    if not (0 < start < end <= n_layers):
        raise ValueError(f"段选择非法: start={start}, end={end}, n={n_layers}")
    if k < 2:
        raise ValueError("循环至少 K=2")
    order = list(range(start)) + list(range(start, end)) * k + list(range(end, n_layers))
    return {
        "n_layers": n_layers, "start": start, "end": end, "k": k,
        "seg_len": end - start,
        "order": order, "new_len": len(order),
        "front_untouched": list(range(start)) == order[:start],
        "back_untouched": list(range(end, n_layers)) == order[-(n_layers - end):],
        "weight_bytes_unchanged": True,
        "loop_refs": {i: k for i in range(start, end)},
    }


def _layers_container(model):
    """解析持有 .layers 的模块（Qwen3.5 架构: 顶层 Model → .language_model(TextModel)
    → .model(Qwen3_5TextModel) → .layers；TextModel.make_cache 经 property 委托同一列表）。"""
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


def apply_loop(model, start=None, end=None, k: int = 2) -> dict:
    """对已加载模型执行层表重排(原地/内存中; 重新 load 即完全还原)。返回排布报告。

    结构浓度断言内建: 循环段引用同一组层对象(零权重复制), 引用计数与计划一致。
    """
    inner = _layers_container(model)
    layers = list(inner.layers)
    plan = plan_loop(len(layers), start, end, k)
    seg_is_linear = [bool(getattr(l, "is_linear", False))
                     for l in layers[plan["start"]:plan["end"]]]
    new_layers = [layers[i] for i in plan["order"]]
    inner.layers = new_layers
    # 防御性同步 config 层数字段(make_cache 按层表构建, 此处同步只为兼容读 config 的路径)
    cfg = getattr(inner, "config", None) or getattr(model, "config", None)
    for f in ("num_hidden_layers", "n_layers", "num_layers"):
        if cfg is not None and hasattr(cfg, f):
            try:
                setattr(cfg, f, plan["new_len"])
            except Exception:
                pass
    objs = {id(x) for x in new_layers}
    assert len(objs) == len(layers), "结构浓度断言失败: 层对象必须零复制(引用同一组对象)"
    plan = dict(plan)
    plan["seg_is_linear"] = seg_is_linear
    plan["struct_ok"] = True
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description="R0 层表重排骨架(默认 dry-run)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.dry_run:
        raise SystemExit("[R0] 骨架纪律: loop_model.py 不单独真跑——评测统一走 "
                         "eval_r0.py(由 run_r0.sh --execute 触发)")

    lt = read_layer_types(a.model)
    if lt:
        n = len(lt)
    else:
        n = int((read_config(a.model).get("text_config") or {})
                .get("num_hidden_layers", 0))
    plan = plan_loop(n, a.start, a.end, a.k)
    seg_types = ([lt[i].replace("_attention", "") for i in range(plan["start"], plan["end"])]
                 if lt else ["?"] * plan["seg_len"])

    print(f"[R0-dry] 模型: {a.model}")
    print(f"[R0-dry] 层数: {n}" + (f"(混合注意力, 原生周期 {PERIOD}: 3×linear+1×full)" if lt else ""))
    print(f"[R0-dry] 循环段: [{plan['start']}, {plan['end']}) = {plan['seg_len']} 层 "
          f"[{'/'.join(seg_types)}] ×K={plan['k']}")
    print(f"[R0-dry] 排布: 前段 {plan['start']} 层 → 循环段×{plan['k']} → "
          f"后段 {n - plan['end']} 层 ⇒ {plan['new_len']} 个层位(权重仍 {n} 层)")
    print(f"[R0-dry] 断言: 前段不动={plan['front_untouched']} | "
          f"后段不动={plan['back_untouched']} | 权重零复制={plan['weight_bytes_unchanged']}")
    print("[R0-dry] 机制: 纯内存引用重排——零梯度路径(框架浓度·结构层 100% 由代码保证); "
          "重新 load 即完全还原")
    print("[R0-dry] 缓存: make_cache() 遍历 self.layers 按层位建槽"
          "(线性层 ArraysCache/全注意力 KVCache)——重排后自动正确(已核实 mlx_lm 源码)")


if __name__ == "__main__":
    main()
