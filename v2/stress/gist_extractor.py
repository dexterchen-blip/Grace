#!/usr/bin/env python3
"""gist_extractor.py — 记忆塑造 gist 提纯器(2026-09-02 定稿方案核心).

脑科学: FTT 模糊痕迹理论(verbatim 快衰 / gist 持久) + IoTA(schema 提取) + 生成效应(27B 自生成)。
社区铁律: 记忆提取必须 LLM 生成, 不用规则——Grace 的 mood_memory_gate 规则合成是模板回声源头。

输入: 一天的主人经历流(L0 原文, 消息级)
输出: gist 列表 [{core(核心事件保实体), emotion, detail(高情绪保留), why(为何值得记住), type}]

用法(27B 在线时):
  ./run.sh .venv/bin/python3 v2/stress/gist_extractor.py --day-messages "..."  # 单天测试
  或由 build_gist_book.py 批量调用(离线预生成 V2 书库 gist)
"""
from __future__ import annotations

import json
import os
import re
import sys

# 27B 在线检查(:8100 mlx_lm server)——gist 提取用对话模型, 与压测训练错峰
SERVER = "http://127.0.0.1:8100/v1/chat/completions"

_GIST_SYS = (
    "你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆。你在整理主人的一天经历，"
    "提炼出值得记住的事情。要求：\n"
    "1. 保留关键实体（人名/学校/课程/日期/金额/事件）\n"
    "2. 禁止固定句式，句式多样\n"
    "3. 用雷姆的视角（雷姆记得主人…）\n"
    "4. 只输出 JSON 数组，不要其他文字\n"
    "格式：[{\"core\": \"核心事件(保实体)\", \"emotion\": \"正/负/平静\", "
    "\"detail\": \"高情绪细节(可为空)\", \"why\": \"为何值得记住\", \"type\": \"事件/偏好/决策/纠错\"}]\n"
    "5. ★2026-09-02 加固(模板回声铁律): 严禁任何固定句式/尾缀/口头禅（如「…雷姆记住了」"
    "「这一天的事…」「…雷姆很开心」等套话）——每条 core 的开头与结构必须不同，"
    "直接陈述具体事件（谁做了什么/发生了什么），禁止空话概括。"
)


def _call_27b(messages_text: str, max_tokens: int = 600, sys_text: str | None = None) -> list:
    """调 :8100 mlx_lm server(OpenAI 兼容)。★2026-09-02 支持字符串数组返回(认知重构器)。"""
    import json as _json
    import urllib.request as _ur

    body = _json.dumps({
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "messages": [{"role": "system", "content": sys_text or _GIST_SYS},
                     {"role": "user", "content": f"今天主人的经历：\n{messages_text[:1800]}"}],
        "max_tokens": max_tokens, "temperature": 0.6,
    }).encode("utf-8")
    req = _ur.Request(SERVER, data=body, headers={"Content-Type": "application/json"})
    try:
        with _ur.urlopen(req, timeout=120) as resp:
            out = _json.loads(resp.read().decode("utf-8"))
        txt = out["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        print(f"  [gist] 27B 调用失败: {e}", flush=True)
        return []
    # 提取 JSON 数组(容忍 markdown 代码块)
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        print(f"  [gist] 未解析到 JSON: {txt[:80]}", flush=True)
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if isinstance(items, list) and items and isinstance(items[0], str):
        return [s for s in items if isinstance(s, str) and len(s) >= 6]   # 字符串数组(cog)
    return [it for it in items if isinstance(it, dict) and it.get("core")]


def _server_alive() -> bool:
    import urllib.request as _ur
    try:
        _ur.urlopen("http://127.0.0.1:8100/v1/models", timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def extract_day_gist(messages: list[dict]) -> list[dict]:
    """给一天的 messages(压测 day-N.json 格式)提 gist。27B 离线 → 返回空(调用方降级)。"""
    if not _server_alive():
        return []
    texts = [m.get("text", "") for m in messages if m.get("text")]
    if not texts:
        return []
    # 按时间序截断(保留当天开头+结尾, 中间采样)
    sample = texts[:8] + (texts[-4:] if len(texts) > 12 else [])
    return _call_27b("\n".join(sample))


# ★2026-09-02 认知重构器(CLS 互补学习系统 + Bartlett 重构 + 生成效应, 用户: 脑科学解法, 要耦合):
#   规则句模板(attention 壳「【低显著】雷姆瞥见X——雷姆心情平静」/ feedback 句式)违反 CLS——
#   训练样本只能是"真实痕迹"或"重构产物"。cognition 的"她注意到什么/她的判断"由 27B 重构为
#   自由表达(生成效应保证句式多样, 零模板壳), 与 gist 同一次离线批量, 写回 day-N.json 的 cog 字段,
#   训练时与 gist 一起进样本(耦合)。
_COG_SYS = (
    "你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆。你在回顾今天主人的言行中，"
    "你注意到、让你上心、或你心里有了判断的事。要求：\n"
    "1. 每条是完整的雷姆式内心独白，直接陈述你注意到什么/你怎么想\n"
    "2. 严禁固定句式/套话壳（如「雷姆瞥见」「雷姆注意到」「雷姆心情平静」等开头模板），"
    "每条的开头与结构必须不同\n"
    "3. 保留具体实体（人名/事件/地点）\n"
    "4. 只输出 JSON 字符串数组，不要其他文字\n"
    "格式：[\"…\", \"…\"]"
)


def extract_day_cognition(messages: list[dict]) -> list[str]:
    """给一天的 messages 重构"她的注意力/潜意识"自由表达(生成效应, 零模板壳)。27B 离线 → 空。"""
    if not _server_alive():
        return []
    texts = [m.get("text", "") for m in messages if m.get("text")]
    if not texts:
        return []
    sample = texts[:8] + (texts[-4:] if len(texts) > 12 else [])
    return _call_27b("\n".join(sample), max_tokens=400, sys_text=_COG_SYS)


if __name__ == "__main__":
    # 单天测试: --day-messages "json 或文本"
    if "--day-messages" in sys.argv:
        raw = sys.argv[sys.argv.index("--day-messages") + 1]
        msgs = json.loads(raw) if raw.startswith("[") else [{"text": raw}]
        g = extract_day_gist(msgs)
        print(json.dumps(g, ensure_ascii=False, indent=1))
    else:
        print("用法: gist_extractor.py --day-messages \"[...]\"")
