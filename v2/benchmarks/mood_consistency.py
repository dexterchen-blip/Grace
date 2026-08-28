#!/usr/bin/env python3
"""心态一致性尺子 v0（Grace_v2 §11）—— 事件→心态映射是否合理、是否平滑。

检查项：
  1. 标签合法性：mood_label ∈ 词表
  2. 平滑性：单件事件导致强度跳变不超过 0.4（防"精神分裂"）
  3. 方向合理性：正面事件 → 强度不降反升
用法（沙盒内）: ./run.sh python3 v2/benchmarks/mood_consistency.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.mood_engine import derive, latest
import config

PASS = 0
FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {name}")
    else:
        FAIL += 1; print(f"  [FAIL] {name}")

print("=== 心态一致性尺子 v0 ===")
# 1. 标签合法性
ev = [{"text": "跑分通过，很开心", "sentiment": 0.9, "weight": 0.8}]
r = derive(ev)
check(f"标签在词表内 ({r['mood_label']})", r["mood_label"] in config.MOOD["labels"])
check(f"强度 0-1 ({r['intensity']})", 0 <= r["intensity"] <= 1)
# 2. 平滑性：连续两次推演（第二次事件中性）强度跳变应 < 0.4
prev_i = r["intensity"]
r2 = derive([{"text": "普通的一天", "sentiment": 0.1, "weight": 0.2}])
check(f"平滑：强度跳变 {abs(r2['intensity']-prev_i):.2f} < 0.4",
      abs(r2["intensity"] - prev_i) < 0.4)
# 3. 方向合理性：负面事件后强度 ≤ 正面后强度
r_neg = derive([{"text": "出了点问题", "sentiment": -0.8, "weight": 0.8}])
check(f"方向：负面事件强度({r_neg['intensity']}) ≤ 正面({r['intensity']})",
      r_neg["intensity"] <= r["intensity"] + 1e-9)
print(f"\n结果: PASS={PASS} FAIL={FAIL}")
sys.exit(1 if FAIL else 0)
