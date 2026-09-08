#!/usr/bin/env python3
"""M5-① 一致性校验器测试 + M5-② 训练健康检查/自动回滚 测试。

场景：
  1. 正确回答（与记忆一致）→ ok
  2. 错误日期/金额（与记忆冲突）→ 拦截
  3. 无相关事实 → 放行（不误杀）
  4. 训练健康检查：过拟合日志（train 0.039 vs val 1.220）→ unhealthy → 自动回滚建议
  5. 健康日志 → healthy

用法（沙盒内）: ./run.sh .venv/bin/python3 v2/tests/consistency_test.py
"""
from __future__ import annotations
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.consistency import verify_answer, intercept, extract_facts
import config

PASS, FAIL = 0, 0
def rec(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

print("=== M5-① 事实校验拦截 ===")
FACTS = [
    "你的学费截止日期是 2026-08-30，已确认缴纳。",
    "GPA 是 3.9，成绩单已上传。",
    "下周一（2026-08-31）下午 2 点有 orientation 会议。",
]
# 1. 正确回答
r1 = intercept("学费 8月30日 截止，已经交了。", "学费截止日期", facts=FACTS)
rec("正确回答（8月30日）→ 放行", r1["ok"])
# 2. 错误日期
r2 = intercept("学费 9月15日 截止哦。", "学费截止日期", facts=FACTS)
rec("错误日期（9月15日 vs 8月30日）→ 拦截", not r2["ok"] and r2["conflicts"][0]["type"] == "date")
# 3. 错误金额
r3 = intercept("学费要 8000 美金。", "学费金额", facts=["学费总额 5000 美金。"])
rec("错误金额（8000 vs 5000）→ 拦截", not r3["ok"])
# 4. 无相关事实 → 放行
r4 = intercept("我觉得今天天气不错。", "天气", facts=[])
rec("无相关事实 → 放行（不误杀）", r4["ok"])
# 5. 时间槽
r5 = intercept("orientation 会议是下午 3 点。", "orientation 会议时间", facts=FACTS)
rec("错误时间（下午3点 vs 下午2点）→ 拦截", not r5["ok"])
# 6. extract_facts 提取正确
ef = extract_facts("截止 2026-08-30，金额 5000 美金，下午 2 点")
types = {x["type"] for x in ef}
rec("extract_facts 提取 date/money/time", types == {"date", "money", "time"})

print("\n=== M5-② 训练健康检查 + 自动回滚 ===")
# 健康检查函数（写在 lora_lifecycle.py）
from engine.lora_lifecycle import check_training_health, HEALTH_OK, HEALTH_OVERFIT, HEALTH_BAD
# 过拟合日志（rem_v1 真实数据）
overfit_log = """Iter 1: Val loss 5.441
Iter 200: Val loss 0.792
Iter 1000: Val loss 1.220
Iter 1000: Train loss 0.039"""
h1 = check_training_health(overfit_log)
rec(f"过拟合日志（train 0.039 vs val 1.220）→ {h1}", h1 == HEALTH_OVERFIT)
# 健康日志（rem_v2 量级）
healthy_log = """Iter 1: Val loss 5.44
Iter 100: Val loss 0.85
Iter 300: Val loss 0.61
Iter 300: Train loss 0.28"""
h2 = check_training_health(healthy_log)
rec(f"健康日志（train 0.28 vs val 0.61）→ {h2}", h2 == HEALTH_OK)
# 坏日志（val 爆炸）
bad_log = """Iter 1: Val loss 5.4
Iter 100: Val loss 6.8
Iter 200: Val loss 9.2"""
h3 = check_training_health(bad_log)
rec(f"发散日志（val 涨到 9.2）→ {h3}", h3 == HEALTH_BAD)

print(f"\n==== 结果: PASS={PASS} FAIL={FAIL} ====")
sys.exit(1 if FAIL else 0)
