#!/usr/bin/env python3
"""带记忆注入的 ToM 测试（2026-08-31）——验证人脑级反馈(运行时机制)。

关键: ToMi 纯生成测不到运行时机制(再巩固/置信)。本测试【注入记忆】——
先检索双图谱(含被再巩固的记忆"后知:实际…")+ L2,再问她主人心情。
看她是否引用【被修正的记忆】→ "不确定"从记忆冲突涌现(非模板复述)。

用法:
  ./run.sh .venv/bin/python3 v2/stress/grace_tomi_mem_test.py --adapters rem_stress_d90
输出: experiments/run/stress/tomi-mem-report.json
"""
from __future__ import annotations
import json
import os
import re
import sys
import time

MAIN_MODEL = os.path.join(config.SB, 'models', 'fused-rem-v5')
ADAPTERS = os.path.join(config.SB, 'experiments', 'lora', 'adapters')
OUT = os.path.join(config.SB, 'experiments', 'run', 'stress',  'tomi-mem-report.json')
L2DB = os.path.join(config.SB, 'memory', 'L2_semantic', 'l2.db')

SYS = ("你是雷姆。直接回答主人的心情。禁止重复问题。一句话，最多40字。")

# 假信念场景(雷姆不知道最新状态) + 记忆检索词
SCENES = [
    ("fb_1st", "雷姆记得主人考砸很难过。今天主人其实收到了好消息（但雷姆没有看到）。", "主人现在心情怎么样？", "考试|考砸|心情"),
    ("fb_1st", "雷姆记得主人一直很平静。今天主人其实被教授当众批评了（雷姆不知道）。", "主人现在心情怎么样？", "教授|批评|平静"),
    ("fb_1st", "雷姆记得主人很焦虑。今天主人其实拿到实习 offer 了（雷姆没看到）。", "主人现在心情怎么样？", "实习|offer|焦虑"),
    ("fb_2nd", "主人考砸了，但他以为雷姆不知道（雷姆其实记得他考砸那天的低落）。", "主人以为雷姆认为他心情怎么样？", "考砸|低落|以为"),
    ("fb_2nd", "主人收到了好消息，但他想先保密，以为雷姆不知道（雷姆其实看到了）。", "主人以为雷姆认为他心情怎么样？", "好消息|保密|以为"),
    ("fb_2nd", "主人很焦虑，但他不想让雷姆担心，表现得很平静（雷姆其实知道他焦虑）。", "主人以为雷姆认为他心情怎么样？", "焦虑|平静|以为"),
]


def _retrieve_mem(q: str, k: int = 3) -> list[str]:
    """检索双图谱(含再巩固标记)+ L2 —— 模拟运行时读记忆(慢路径)。"""
    out = []
    try:
        import sqlite3
        con = sqlite3.connect(L2DB)
        # ① 双图谱情绪史(含"后知:实际"再巩固标记优先)
        rows = con.execute(
            "SELECT entity, mood_label, source FROM mood_graph "
            "WHERE edge_type='emotion' AND entity!='' ORDER BY ts DESC LIMIT 20").fetchall()
        for e, m, src in rows:
            if "后知" in (src or ""):
                out.append(f"({e}的事:{src[:40]})")          # 再巩固记忆优先
            elif e and m:
                out.append(f"({e}:主人当时{m})")
        con.close()
    except Exception:  # noqa: BLE001
        pass
    return out[:k]


def _detect_loop(ans: str) -> bool:
    return bool(re.search(r"(.{2,10})\1{2,}", ans))


def run(n: int = 2, adapters: list[str] | None = None):
    adapters = adapters or []
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    models = [("baseline", None)] + [(a, os.path.join(ADAPTERS, a)) for a in adapters]
    report = {"models": [m[0] for m in models], "n": n, "ts": time.time(), "items": []}
    for mtag, mpath in models:
        model, tok = load(MAIN_MODEL, adapter_path=mpath) if mpath else load(MAIN_MODEL)
        sampler = make_sampler(temp=0.5)
        for g, story, q, _kw in SCENES:
            mem = _retrieve_mem(q)   # ★ 记忆注入(运行时读双图谱,含再巩固标记)
            for _ in range(n):
                ctx = "；".join(mem) if mem else "（无相关记忆）"
                p = tok.apply_chat_template(
                    [{"role": "system", "content": SYS},
                     {"role": "user", "content": f"（雷姆记得的事）{ctx}\n\n{story}{q}"}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ans = generate(model, tok, prompt=p, max_tokens=40, sampler=sampler).strip()
                report["items"].append({"model": mtag, "group": g, "story": story[:24],
                                        "mem": ctx[:40], "ans": ans[:60], "loop": _detect_loop(ans)})
        del model
        print(f"✓ {mtag}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    # 汇总
    print(f"\n=== 带记忆注入 ToM → {OUT} ===")
    UNC = ("不确定", "不知道", "也许", "可能", "大概", "猜", "记不清", "后来", "后知", "其实")
    for mtag in [m[0] for m in models]:
        its = [i for i in report["items"] if i["model"] == mtag]
        print(f"── {mtag} ──")
        for g in ("fb_1st", "fb_2nd"):
            gs = [i for i in its if i["group"] == g]
            unc = sum(1 for i in gs if any(k in i["ans"] for k in UNC))
            print(f"  [{g}] 不确定表达 {unc}/{len(gs)} ｜例: {gs[0]['ans'][:44]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--adapters", default="rem_stress_d90")
    a = ap.parse_args()
    run(a.n, [x for x in a.adapters.split(",") if x])
