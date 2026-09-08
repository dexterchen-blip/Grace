#!/usr/bin/env python3
"""快慢双路径测试（§15 类人双系统记忆）。

场景：
  1. 路由分类：人格问题→fast；事实问题→slow；含事实词优先 slow
  2. 慢路径：检索到事实 → 注入 → 生成钩子收到含记忆的 prompt
  3. 回忆失败：检索空 → 诚实回答（不幻觉）
  4. 冲突拦截联动：生成回答与检索事实冲突 → 标注外挂优先
  5. 快路径：生成钩子收到"无需检索"的 prompt

用法（沙盒内）: ./run.sh .venv/bin/python3 v2/tests/dual_path_test.py
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.dual_path import classify_question, answer_slow, answer_fast, route
from engine.consistency import intercept

PASS, FAIL = 0, 0
def rec(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

print("=== 1. 路由分类 ===")
rec("「你是谁呀」→ fast", classify_question("你是谁呀？")["path"] == "fast")
rec("「你喜欢昴吗」→ fast", classify_question("你喜欢昴吗？")["path"] == "fast")
rec("「学费什么时候截止」→ slow", classify_question("学费什么时候截止？")["path"] == "slow")
rec("「昨天发生了什么」→ slow", classify_question("昨天发生了什么？")["path"] == "slow")
rec("「今天心情怎么样」→ fast（主观）", classify_question("今天心情怎么样？")["path"] == "fast")
rec("「GPA 是多少」→ slow", classify_question("我的 GPA 是多少？")["path"] == "slow")

print("\n=== 2. 慢路径（检索→注入→生成→校验） ===")
FACTS = ["学费截止日期是 2026-08-30，已确认缴纳。"]
captured = {}
def gen(prompt):
    captured["prompt"] = prompt
    return "学费 8月30日 截止，已经交了。"
r = answer_slow("学费什么时候截止", search_fn=lambda q: FACTS, generate_fn=gen,
                verify_fn=lambda a, q, facts: intercept(a, q, facts=facts))
rec("慢路径检索到 1 条事实", r["recall"] == "ok" and r["retrieved"] == 1)
rec("生成 prompt 注入了参考记忆", "参考记忆" in captured.get("prompt", ""))
rec("回答正确且无冲突", r["conflicts"] == [])

print("\n=== 3. 回忆失败（检索空 → 诚实） ===")
r2 = answer_slow("学费什么时候截止", search_fn=lambda q: [])
rec("检索空 → recall=fail", r2["recall"] == "fail")
rec("诚实回答（不幻觉）", "记不太清" in r2["answer"])

print("\n=== 4. 冲突拦截联动 ===")
def gen_wrong(prompt):
    return "学费 9月15日 截止哦。"
r3 = answer_slow("学费什么时候截止", search_fn=lambda q: FACTS, generate_fn=gen_wrong,
                 verify_fn=lambda a, q, facts: intercept(a, q, facts=facts))
rec("回答与记忆冲突 → 标注外挂优先", len(r3["conflicts"]) > 0 and "外挂轨优先" in r3["answer"])

print("\n=== 5. 快路径（不检索秒答） ===")
r4 = answer_fast("你是谁呀", generate_fn=lambda p: "雷姆，罗兹瓦尔宅邸的女仆。")
rec("快路径直接人格回答", r4["answer"] == "雷姆，罗兹瓦尔宅邸的女仆。")

print(f"\n==== 结果: PASS={PASS} FAIL={FAIL} ====")
sys.exit(1 if FAIL else 0)
