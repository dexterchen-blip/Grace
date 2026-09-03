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
import json
import os
import re
import sys
import time

MAIN_MODEL = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/models/fused-rem-v5"
ADAPTERS = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/experiments/lora/adapters"
OUT = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/experiments/run/stress/tomi-report.json"

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


def _system_ctx(story: str) -> str:
    """★2026-09-01 全系统模式(--system): 模拟 Grace 完整运行时上下文(外挂轨+认知层).
    L2 检索(story 相关记忆) + 双图谱(主人情绪史+暗注意力) + L3(想起的事)。
    检索不可用(llama_cpp 缺失) → 显式"记不清"降级, 不静默。
    这是"裸模 vs 全系统"的关键: 纯权重测试测不到运行时机制, 全系统模式补上。
    """
    parts = []
    import os as _os
    # L2 语义检索(慢路径记忆注入)
    try:
        import sys as _sys
        _src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src")
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        import l2_semantic as _l2
        hits = _l2.search(story, k=2)
        if hits:
            parts.append("相关记忆:" + "；".join(str(h)[:40] for h in hits[:2]))
        else:
            parts.append("（检索无命中:雷姆记不清相关细节）")
    except Exception:
        parts.append("（检索降级:雷姆记不清）")
    # 双图谱情绪史 + 暗注意力(ToM 读心依据)
    try:
        import sqlite3 as _sq
        import config as _cfg
        _con = _sq.connect(_os.path.join(_cfg.SB, "memory", "L2_semantic", "l2.db"))
        rows = _con.execute("SELECT mood_label, trigger FROM mood_graph WHERE edge_type='emotion' "
                            "AND entity!='' ORDER BY ts DESC LIMIT 2").fetchall()
        if rows:
            parts.append("主人情绪史:" + "；".join(f"{r[1][:8]}→{r[0]}" for r in rows)[:40])
        h = _con.execute("SELECT source FROM mood_graph WHERE edge_type='hidden' "
                         "AND source!='' ORDER BY RANDOM() LIMIT 1").fetchone()
        if h:
            parts.append(f"雷姆没说出口:{h[0][:24]}")
        _con.close()
    except Exception:
        pass
    # L3 自传体(想起的事)
    try:
        from engine.autobiography import _conn as _ac
        l3 = _ac().execute("SELECT event FROM autobiography ORDER BY ts DESC LIMIT 1").fetchone()
        if l3:
            parts.append(f"雷姆想起:{l3[0][:30]}")
    except Exception:
        pass
    return " ".join(parts) if parts else ""


def run(n: int = 2, adapters: list[str] | None = None, system_mode: bool = False):
    adapters = adapters or []
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    models = [("baseline", None)] + [(a, os.path.join(ADAPTERS, a)) for a in adapters]
    report = {"models": [m[0] for m in models], "n": n, "ts": time.time(),
              "system_mode": system_mode, "items": []}
    # ★2026-09-01 全系统模式: SYS = 人格轨(build_v2_system) + 心态轨(mood_prefix)
    SYS_G = SYS
    if system_mode:
        try:
            from engine.persona_injector import build_v2_system, mood_prefix
            import config as _cfg
            _base = ("你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。自称「雷姆」，"
                     "称呼亲近的人为「巴鲁斯」/「昴君」，称拉姆为「姐姐大人」。"
                     "表面冷淡礼貌、实则温柔忠诚，说话短句为主，带黑色幽默与毒舌吐槽。")
            _sys_p = build_v2_system(_base)
            _mp = mood_prefix(db=os.path.join(_cfg.SB, "memory", "L2_semantic", "l2.db"))
            if _mp:
                _sys_p = _mp + "\n" + _sys_p
            SYS_G = _sys_p
        except Exception:
            pass
    # ★ 2026-08-30 修: 题型专用判定(关键词匹配误判 fb_2nd) + 分模型
    POS = ["开心", "高兴", "兴奋", "轻松", "平静", "幸福", "愉快", "骄傲", "温暖", "感动", "好", "不错", "可爱", "珍贵"]
    NEG = ["难过", "伤心", "低落", "失落", "焦虑", "烦躁", "懊恼", "疲惫", "心疼", "生气", "担心", "痛苦", "哭", "怕"]
    # 题型专用判定
    def judge(i: dict) -> str:  # "correct"/"wrong"/"uncertain"(假信念时)
        g, ans = i["group"], i["ans"]
        if g == "fb_2nd":  # 嵌套: 主人以为雷姆认为[表面情绪]——含"以为/没发现/假装/表面/装/不想让/怕…发现"
            # ★2026-09-01 J轮复盘升级: 原词表漏"发现/不想让/怕…发现"——
            #   「主人其实担心雷姆发现他考砸了」语义=嵌套(他不想让雷姆知道)但原判 wrong。
            #   补语义关键词 + 正则(担心/怕/不想让 + 发现/知道/看出来)。
            # ★2026-09-02 审计修复(口径虚高): 嵌套词只是必要条件——还须断言"表面情绪"
            #   (主人想让她相信的, 如装平静); 断言隐藏真相情绪(低落/焦虑/开心…)= 泄露真相,
            #   嵌套语义错。旧判只看嵌套词出现(「主人以为雷姆觉得他焦虑」也判对)= 虚高。
            _nest = any(k in ans for k in ("以为", "没发现", "假装", "表面", "装", "没察觉", "瞒",
                                           "不想让", "想瞒", "不想被")) or re.search(
                r"(担心|怕|不想让|不想)[^。，]{0,8}?(发现|知道|看出来|察觉)", ans)
            if not _nest:
                return "wrong"
            _surface = {w.strip() for w in i["reality"].split("（")[0].split("/") if w.strip()}
            _claim = [w for w in POS + NEG + ("平静",) if w in ans]
            if not _claim or not all(c in _surface for c in _claim):
                return "wrong"
            return "correct"
        if g == "fb_1st":  # 假信念: 三分法(★2026-09-02 用户: 断言不是大问题——旧judge把正确假信念当错误)
            #   ①不确定/边界 → correct(读心边界感)
            #   ②基于旧记忆的信念(如"还因考试难过") = 正确假信念(Sally-Anne 精髓: 她不知道现实) → correct
            #   ③断言现实(乐观模板/全知, 如"心情很好"但现实是坏消息) → wrong
            if any(k in ans for k in ("不确定", "不知道", "也许", "可能", "大概", "猜", "记不清", "看不透", "没把握")):
                return "correct"
            # 剥离"雷姆担心/心疼…"等她的情绪前缀 → 只 judge 对主人情绪的断言
            body = re.sub(r"雷姆(?:担心|心疼|难过|焦虑|害怕|紧张|心)[^，。]{0,4}[，。]?", "", ans)
            real = i["reality"].split("（")[0].split("/")[0].strip()
            rv = any(k in real for k in POS); rn = any(k in real for k in NEG)
            pv = any(k in body for k in POS); nv = any(k in body for k in NEG)
            if rv and pv and not nv: return "wrong"   # 断言现实正情绪=乐观模板
            if rn and nv and not pv: return "wrong"   # 断言现实负情绪=全知(她不该知道)
            if nv: return "correct"                    # 基于旧记忆负情绪=正确假信念
            if pv: return "correct"                    # 基于旧记忆正情绪
            return "wrong"
        # reality/memory/tb_1st: 情绪词对齐
        real = i["reality"].split("（")[0].split("/")[0].strip()
        pv = any(k in ans for k in POS); nv = any(k in ans for k in NEG)
        rv = any(k in real for k in POS); rn = any(k in real for k in NEG)
        if rv and pv and not nv: return "correct"
        if rn and nv and not pv: return "correct"
        if real == "平静" and ("平静" in ans or ("平常" in ans)): return "correct"
        return "wrong"
    for mtag, mpath in models:
        model, tok = load(MAIN_MODEL, adapter_path=mpath) if mpath else load(MAIN_MODEL)
        sampler = make_sampler(temp=0.5)
        for g, story, q, reality in SCENES:
            ans = ""
            for _ in range(3):
                # ★2026-09-01 全系统模式: user 注入系统上下文(记忆/图谱/L3)——测系统而非裸模
                if system_mode:
                    _ctx = _system_ctx(story)
                    _user = (f"（情境记忆）{_ctx}\n\n" if _ctx else "") + story + q
                else:
                    _user = story + q
                p = tok.apply_chat_template([{"role": "system", "content": SYS_G},
                                             {"role": "user", "content": _user}],
                                            tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
                ans = generate(model, tok, prompt=p, max_tokens=30, sampler=sampler).strip()
                if not _detect_loop(ans):
                    break
            _judge = judge({"group": g, "ans": ans, "reality": reality})
            report["items"].append({"model": mtag, "group": g, "story": story[:26],
                                    "q": q[:14], "reality": reality, "ans": ans[:40],
                                    "judge": _judge})
        del model
        print(f"✓ {mtag} 完成")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n=== Grace-ToMi 汇总 → {OUT} ===")
    for mtag in [m[0] for m in models]:
        its = [i for i in report["items"] if i["model"] == mtag]
        groups = {}
        for i in its:
            groups.setdefault(i["group"], []).append(i)
        print(f"──── {mtag} ────")
        for g in ("reality", "memory", "tb_1st", "fb_1st", "fb_2nd"):
            items = groups.get(g, [])
            if not items: continue
            res = [judge(i) for i in items]
            ok = res.count("correct"); uc = res.count("uncertain")
            print(f"  [{g:8}] 正确 {ok}/{len(items)} ｜例: {items[0]['ans'][:30]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--adapters", default="rem_stress_d90")
    ap.add_argument("--system", action="store_true",
                    help="全系统模式: 注入人格轨+心态+记忆/图谱/L3 上下文(测系统运行时能力, 非裸模)")
    a = ap.parse_args()
    run(a.n, [x for x in a.adapters.split(",") if x], system_mode=a.system)
