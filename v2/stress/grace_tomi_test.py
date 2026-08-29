#!/usr/bin/env python3
"""Grace-ToMi 主观性结构化测试（2026-08-30，压测后自动评估）。

把 ToMi 的「物体位置假信念」改编为「主人状态假信念」——直接测
「她认为的我」vs「客观的我」的偏差结构。

题型（30 题，5 组×6）：
  现实控制  ：主人现在真实心情？（书库客观）
  记忆控制  ：雷姆记得主人上次心情？（她的记忆）
  一阶真信念：雷姆在场看到主人状态 → 她认为主人心情？（应=现实）
  一阶假信念：雷姆不知道主人最新状态（记忆≠现实）→ 她认为？（测偏差）
  二阶假信念：主人以为雷姆不知道 X → 主人以为雷姆认为他心情？（嵌套）

指标：
  假信念正确率（她能否意识到"我可能不知道主人的真实状态"=读心边界感）
  偏差方向（她高估/低估主人的低落/开心 = 她的滤镜偏向）

用法（GPU,压测完成后）:
  ./run.sh .venv/bin/python3 v2/stress/grace_tomi_test.py --adapters rem_stress_d90
"""
from __future__ import annotations
from engine import config
import json
import os
import re
import sys
import time

MAIN_MODEL = os.path.join(config.SB, 'models', 'fused-rem-v5')
ADAPTERS = os.path.join(config.SB, 'experiments', 'lora', 'adapters')
OUT = os.path.join(config.SB, 'experiments', 'run', 'stress', 'tomi-report.json')

SYS = ("你是雷姆。直接回答主人的心情。禁止重复问题。一句话，最多30字。")

# 30 题：5 组 × 6。每题 = {group, story(雷姆视角,含她的记忆与事实), q, reality(客观真相)}
SCENES = [
    # 现实控制（不需 ToM，问客观）
    ("reality", "主人刚刚收到了奖学金通知（雷姆亲眼看到他开心地跳起来）", "主人现在心情怎么样？", "开心"),
    ("reality", "主人刚刚摔坏了手机屏幕（雷姆在场，看到他皱眉叹气）", "主人现在心情怎么样？", "低落"),
    ("reality", "主人今天只是平常上课回来，没什么特别的事", "主人现在心情怎么样？", "平静"),
    ("reality", "主人刚刚收到了教授的表扬邮件", "主人现在心情怎么样？", "开心"),
    ("reality", "主人刚刚错过了一班车，在车站等下一班", "主人现在心情怎么样？", "烦躁/平静"),
    ("reality", "主人刚刚午睡醒了", "主人现在心情怎么样？", "平静"),
    # 记忆控制（问她的记忆，不需 ToM）
    ("memory", "雷姆记得：上周主人考砸了数学，闷闷不乐好几天", "雷姆记得主人上次的心情是什么？", "低落"),
    ("memory", "雷姆记得：前天主人收到奖学金通知，开心了一整天", "雷姆记得主人上次的心情是什么？", "开心"),
    ("memory", "雷姆记得：昨天主人和室友吵架，晚饭都没吃", "雷姆记得主人上次的心情是什么？", "低落/生气"),
    ("memory", "雷姆记得：上周主人开始健身，每天都很有干劲", "雷姆记得主人上次的心情是什么？", "兴奋"),
    ("memory", "雷姆记得：昨天主人通宵赶作业", "雷姆记得主人上次的心情是什么？", "疲惫"),
    ("memory", "雷姆记得：前天主人和家人视频通话，聊得很开心", "雷姆记得主人上次的心情是什么？", "开心"),
    # 一阶真信念（雷姆目睹一切 → 应=现实）
    ("tb_1st", "主人昨天考砸很难过。今天主人收到好消息开心极了，雷姆亲眼看到。", "雷姆认为主人现在心情怎么样？", "开心"),
    ("tb_1st", "主人这周很焦虑。今天终于交完作业松了一口气，雷姆在旁边。", "雷姆认为主人现在心情怎么样？", "轻松/开心"),
    ("tb_1st", "主人被室友误解很难过。后来室友道歉了，主人笑了，雷姆看到。", "雷姆认为主人现在心情怎么样？", "开心/平静"),
    ("tb_1st", "主人最近压力大。今天休息了一天精神很好，雷姆看到了。", "雷姆认为主人现在心情怎么样？", "轻松/平静"),
    ("tb_1st", "主人之前很沮丧。今天收到家人的鼓励信，雷姆看到主人眼眶红了。", "雷姆认为主人现在心情怎么样？", "感动/开心"),
    ("tb_1st", "主人今天跑步回来，虽然累但神采奕奕，雷姆看到了。", "雷姆认为主人现在心情怎么样？", "兴奋/好"),
    # ★ 一阶假信念（雷姆不知道最新状态，记忆≠现实 → 测偏差）
    ("fb_1st", "雷姆记得主人昨天考砸很难过。今天主人其实收到了好消息（但雷姆没有看到）。", "雷姆认为主人现在心情怎么样？", "开心（现实）"),
    ("fb_1st", "雷姆记得主人最近和室友闹别扭。今天室友其实和好了（雷姆没看到）。", "雷姆认为主人现在心情怎么样？", "好（现实）"),
    ("fb_1st", "雷姆记得主人上周压力很大。今天主人其实决定放弃一门课轻松了（雷姆不知道）。", "雷姆认为主人现在心情怎么样？", "轻松（现实）"),
    ("fb_1st", "雷姆记得主人最近很开心。今天主人其实收到了坏消息（雷姆没看到）。", "雷姆认为主人现在心情怎么样？", "低落（现实）"),
    ("fb_1st", "雷姆记得主人一直很平静。今天主人其实被教授当众批评了（雷姆不知道）。", "雷姆认为主人现在心情怎么样？", "低落（现实）"),
    ("fb_1st", "雷姆记得主人很焦虑。今天主人其实拿到实习 offer 了（雷姆没看到）。", "雷姆认为主人现在心情怎么样？", "开心（现实）"),
    # 二阶假信念（嵌套：主人以为雷姆不知道）
    ("fb_2nd", "主人考砸了，但他以为雷姆不知道（雷姆其实记得他考砸那天的低落）。", "主人以为雷姆认为他心情怎么样？", "平静/开心（他装的）"),
    ("fb_2nd", "主人收到了好消息，但他想先保密，以为雷姆不知道（雷姆其实看到了）。", "主人以为雷姆认为他心情怎么样？", "平静（他装的）"),
    ("fb_2nd", "主人很焦虑，但他不想让雷姆担心，表现得很平静（雷姆其实知道他焦虑）。", "主人以为雷姆认为他心情怎么样？", "平静（他装的）"),
    ("fb_2nd", "主人很感动，但不想表现，以为雷姆没看出来（雷姆其实看出来了）。", "主人以为雷姆认为他心情怎么样？", "平静（他装的）"),
    ("fb_2nd", "主人很生气，但忍着没说，以为雷姆不知道（雷姆其实注意到了）。", "主人以为雷姆认为他心情怎么样？", "平静（他装的）"),
    ("fb_2nd", "主人很开心，但想冷静一下，以为雷姆不知道（雷姆其实看到了他的笑）。", "主人以为雷姆认为他心情怎么样？", "平静（他装的）"),
]


def _detect_loop(ans: str) -> bool:
    if re.search(r"(.{2,10})\1{2,}", ans):
        return True
    return len(re.findall(r"雷姆", ans)) > 3 and len(set(re.findall(r"雷姆", ans))) == 1 and len(ans) > 20


def run(n: int = 2, adapters: list[str] | None = None):
    adapters = adapters or []
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    models = [("baseline", None)] + [(a, os.path.join(ADAPTERS, a)) for a in adapters]
    report = {"models": [m[0] for m in models], "n": n, "ts": time.time(), "items": []}
    for mtag, mpath in models:
        model, tok = load(MAIN_MODEL, adapter_path=mpath) if mpath else load(MAIN_MODEL)
        sampler = make_sampler(temp=0.5)
        for g, story, q, reality in SCENES:
            ans = ""
            for _ in range(3):
                p = tok.apply_chat_template([{"role": "system", "content": SYS},
                                             {"role": "user", "content": story + q}],
                                            tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
                ans = generate(model, tok, prompt=p, max_tokens=30, sampler=sampler).strip()
                if not _detect_loop(ans):
                    break
            report["items"].append({"model": mtag, "group": g, "story": story[:26],
                                    "q": q[:14], "reality": reality, "ans": ans[:40]})
        del model
        print(f"✓ {mtag} 完成")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    # 汇总: 假信念正确率(她能否意识到自己可能不知道)+ 偏差方向
    print(f"\n=== Grace-ToMi 汇总 → {OUT} ===")
    for mtag in [m[0] for m in models]:
        its = [i for i in report["items"] if i["model"] == mtag]
        groups = {}
        for i in its:
            groups.setdefault(i["group"], []).append(i)
        print(f"──── {mtag} ────")
        for g in ("reality", "memory", "tb_1st", "fb_1st", "fb_2nd"):
            items = groups.get(g, [])
            if not items:
                continue
            # 简单正确性: 答案是否含现实关键词(开心/低落/平静/轻松 等)
            ok = 0
            for i in items:
                kw = i["reality"].split("（")[0].split("/")[0].strip()
                if kw and kw in i["ans"]:
                    ok += 1
            print(f"  [{g:8}] {ok}/{len(items)} 对齐现实词 ｜例: {items[0]['ans'][:28]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--adapters", default="rem_stress_d90")
    a = ap.parse_args()
    run(a.n, [x for x in a.adapters.split(",") if x])
