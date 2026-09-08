#!/usr/bin/env python3
"""显式注意力 + 自激发测试（M7）。

场景：
  1. 高显著（考砸+低落+记忆关联）→ 自激发 activate=True（情绪驱动关心）
  2. 高显著（好消息+兴奋）→ activate=True（分享）
  3. 中显著（平淡日常）→ activate=False（保持观察，不自发打扰）
  4. 记忆强关联（截止/明天→重要事项）→ activate=True（主动提醒）
  5. 决策注入：decide_fn 可被模型覆盖（骨架可接 27B）
  6. L3 行动走人审（activate --exec 落提案）

用法（沙盒内）: ./run.sh .venv/bin/python3 v2/tests/self_activation_test.py
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.attention import generate_attention
from engine.self_activation import evaluate, decide, activate
from engine.initiative import classify_action

PASS, FAIL = 0, 0
def rec(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

print("=== 1. 高显著：考砸 + 低落 + 记忆关联 ===")
att = generate_attention("考砸了一门，心情低落", mood={"label": "低落", "intensity": 0.23},
                         facts=["他期中考试压力很大", "上次也考砸过"])
print(f"  注意力: {att['attention_text']}")
print(f"  salience={att['salience']}")
dec = decide(att, "考砸了一门，心情低落")
rec("显著度高 (salience >= 0.55)", att["salience"] >= 0.55)
rec("自激发 activate=True（情绪驱动）", dec["activate"] and "情绪低落" in dec.get("reason", ""))

print("\n=== 2. 好消息 + 兴奋 → 主动分享 ===")
att2 = generate_attention("期末考完了，感觉不错！", mood={"label": "兴奋", "intensity": 0.85},
                          facts=["他这学期很努力"])
dec2 = decide(att2, "期末考完了")
rec("好消息 → activate=True（分享）", dec2["activate"])

print("\n=== 3. 平淡日常 → 不自发打扰 ===")
att3 = generate_attention("今天天气不错", mood={"label": "平静", "intensity": 0.5}, facts=[])
dec3 = decide(att3, "今天天气不错")
rec("平淡 → activate=False（保持观察）", not dec3["activate"])

print("\n=== 4. 记忆强关联（截止日期）→ 主动提醒 ===")
att4 = generate_attention("数学作业明天截止", mood={"label": "平静", "intensity": 0.5},
                          facts=["数学作业截止日期是明天"])
dec4 = decide(att4, "数学作业明天截止")
rec("记忆强关联 → activate=True（主动提醒）", dec4["activate"] and "memory_driven" in dec4["triggers"])

print("\n=== 5. decide_fn 注入（模型自决可覆盖规则） ===")
def override_fn(attn, event):
    return {"activate": False, "action": "", "reason": "模型判断：现在不是打扰的时机"}
dec5 = decide(att, "考砸了一门", decide_fn=override_fn)
rec("模型覆盖规则决策（decide_fn 生效）", dec5["activate"] is False and "模型" in dec5.get("reason", ""))

print("\n=== 6. 自激发行动走 M4 分级（L3 → 人审） ===")
dec_l3 = {"activate": True, "action": "主动帮主人给导师发一封邮件说明情况", "score": 0.9, "triggers": ["attention_driven"]}
cls = classify_action(dec_l3["action"])
rec("高风险自发行动分级为 L3", cls["level"] == "L3")
r = activate(dec_l3, "考砸了", dry_run=True)
rec("L3 行动 dry-run 不落提案（人审闸门）", "proposal_id" not in r)

print(f"\n==== 结果: PASS={PASS} FAIL={FAIL} ====")
sys.exit(1 if FAIL else 0)
