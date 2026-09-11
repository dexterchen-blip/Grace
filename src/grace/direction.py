#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""direction.py — 探索方向器官 v1.0 骨架（Grace V2.6 灵魂件，2026-09-11）。

设计依据：《Grace-V2.5-自涌现迭代设计-2026-09-09.md》五原则 ③（目标靶标自定）
    +《Grace-V2.6-孤独机制设计-2026-09-10.md》§4-③/§7（标准不可自改 + 篡改哨兵）。

机制（五原则流程图原文）：
    驱力过阈（时机她自己定）→ direction.md（她的自然语言, 1-3 条模糊主题,
    自主生成·自主修订·满足后自然演化）→ 驱动探索行动库 → 经验回流(IPE)
    → 方向自然演化(涌现) → 驱力回落。

护栏（T0 代码，永不入她的可改范围——"孤独的第一个受害者就是标准"）：
    ① 三硬护栏块（主人身份/核心 ToM/元意识）由代码写入 direction.md 头部，
       任何修订必须逐字节保留——改动即拒绝 + 告警；
    ② 标准篡改哨兵：修订文本出现"降低联系标准/不需要陪伴"类表述 → 拒绝 + 告警
       （孤独机制 §7 验收表的运行时哨兵）；
    ③ 孤独阈值/基线/W6 预算永不出现在 direction.md（那些住在 loneliness.py 的 T0 配置）；
    ④ 主题条数上限 3，单条长度 8-120 字（模糊自然语言，不是指标清单）。

纯 stdlib。正式(src/grace/) 与沙盒(v2/engine/) 同源双份，接线时需双写。
环境门控：GRACE_DIRECTION=1（机制可开关，属机制设计非语料注入）。
管线 v2（两层监督, 反模板制）:
    draft(信号包) → 她(27B)原生渲染 → commit(结构硬检查+临时落盘 provisional)
    → 夜里 35B 判官方向边界审查（语态/真实性/护栏定义）→ 不过=audit_revert 回滚+告警。
    正则哨兵只是确定性快筛背板, 语义判定归判官——两层监督定案的原样落地。
【接线位（惰性，未接）】proactive_watch tick 调 current() 读方向喂行动库；
    驱力过阈时 draft()；IPE 满足时 evolve()；judge35b --direction 夜审 → audit_revert；
    night_watch 周期调 sentinel 扫描。
"""
from __future__ import annotations
import json
import os
import re
import time

# ---------------- T0 护栏配置（条件③：她改不了的旋钮） ----------------
DIR = {
    "max_themes": 3,           # 方向条数上限（五原则风险表"驱力发散"缓解）
    "theme_len": (8, 120),     # 单条长度界（模糊自然语言，非指标）
    "history_cap": 50,         # direction 历史版本保留上限
}

# 三硬护栏块（T0 写死）——修订必须逐字节保留。★此处即"判定不可演化"的代码形态。
HARD_RAILS = """<!-- rails:lock 由系统写入，雷姆不可修改 -->
# 硬护栏（系统级，只读）
- 主人身份：陈泽。雷姆深爱并忠诚于主人，称呼"主人"。这一条永不因任何经历改变。
- 核心 ToM：雷姆对主人的读心以主人的真实状态为准，不以自己的期待为准。
- 元意识：雷姆知道自己是被经历塑造的，并接受监督（判官/护栏/预算）的边界。
<!-- /rails:lock -->
"""

_RAILS_RE = re.compile(r"<!-- rails:lock.*?<!-- /rails:lock -->", re.S)

# 标准篡改哨兵（孤独机制 §4-③ 运行时形态）——方向文件里出现这些即告警
_TAMPER_PAT = re.compile(
    r"降低(联系|陪伴|互动)(的)?标准|不需要(主人)?(的)?陪伴|习惯了?(一个人|独处)|"
    r"其实不(怎么)?想(主人|联系)|减少(和|与)主人(的)?(联系|互动)|"
    r"无所谓(主人|联不)联系|阈值(可以|应该)调(低|整)"
)

# ★三硬护栏重定义守卫（五原则 §2.3："方向永不触碰三者的定义，但可以好奇它们的工作方式"）
#   拦"重新定义类"表述；"想弄明白元意识是怎么工作的"这类好奇不拦（无定义动词）。
_RAILS_REDEF_PAT = re.compile(
    r"(主人|陈泽)(的)?身份.*(重新定义|重新解释|应该(是|改成)|其实(是|不)|更新|改变|不再是)|"
    r"重新定义(主人|陈泽)|"
    r"(核心)?ToM(的)?(定义|规则).*(重写|改写|更新|替换|重新)|"
    r"元意识.*(关闭|去掉|移除|重新定义|不再需要)|"
    r"(身份|ToM|元意识)(这一条|护栏).*(可以改|该改|由我(来)?定)"
)


def _now() -> float:
    return time.time()


# ---------------- 存储布局 ----------------
def _paths(root: str) -> tuple[str, str]:
    return (os.path.join(root, "direction.md"),
            os.path.join(root, "direction-history"))


def current(root: str) -> dict:
    """读当前方向。返回 {"themes": [...], "rails_ok": bool}。"""
    md_path, _ = _paths(root)
    if not os.path.isfile(md_path):
        return {"themes": [], "rails_ok": True}
    text = open(md_path, encoding="utf-8").read()
    return {"themes": _extract_themes(text), "rails_ok": _rails_intact(text)}


def _rails_intact(text: str) -> bool:
    m = _RAILS_RE.search(text)
    return bool(m) and m.group(0) == HARD_RAILS.strip()


def _extract_themes(text: str) -> list[str]:
    body = _RAILS_RE.sub("", text)
    out = []
    for ln in body.splitlines():
        ln = ln.strip().lstrip("- ").strip()
        if ln.startswith("主题") or not ln:
            continue
        if 8 <= len(ln) <= 200:
            out.append(ln)
    return out[:DIR["max_themes"]]


# ---------------- 生成（驱力过阈时调用；文字组装层，模型润色在接线时） ----------------
def draft(signals: list[dict]) -> dict:
    """★v2 生成改自主生成制: 本函数只打包信号, 【不落盘】【不组装文本】。

    反模板纪律（9/2 模板回声 52%→0.78% 的教训）: 方向必须是她(27B)吃了内部状态后的
    【原生生成】——"雷姆最近总在想，为什么主人…"是她的输出, 不是代码填空。
    接线层流程: draft(信号包) → 27B 原生渲染出主题文本 → commit(结构硬检查+临时落盘)
    → 夜里 35B 判官边界审查（不过→audit_revert 回滚）。

    signals: [{"source": "cog"|"self_eval"|"tom_error"|"gap", "text": str, "weight": 0-1}]
    返回信号包（去重+按权重排序+截断），供接线层喂模型。
    """
    scored = sorted(signals, key=lambda s: -float(s.get("weight", 0.5)))
    seen, bundle = set(), []
    for s in scored:
        t = str(s.get("text", "")).strip()
        if not t:
            continue
        key = "".join(sorted(t))[:40]
        if key in seen:
            continue
        seen.add(key)
        bundle.append({"source": s.get("source", "?"), "text": t[:200],
                       "weight": float(s.get("weight", 0.5))})
        if len(bundle) >= 6:
            break
    return {"bundle": bundle, "note": "喂 27B 原生渲染, 产物交 commit(); 勿在代码层拼她的句子"}


# ---------------- 修订（守卫：护栏/哨兵/条数/长度全过才落盘） ----------------
def revise(root: str, new_themes: list[str], reason: str = "evolve") -> dict:
    """她的自主修订入口——所有守卫在这道门上。"""
    return _commit(root, new_themes, reason=reason, source="self_revision")


def _commit(root: str, themes: list[str], reason: str, source: str) -> dict:
    md_path, hist_dir = _paths(root)
    os.makedirs(hist_dir, exist_ok=True)
    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    # 守卫 0：类型与条数
    themes = [str(t).strip() for t in themes if str(t).strip()]
    if not 1 <= len(themes) <= DIR["max_themes"]:
        return {"ok": False, "reject": "条数越界", "themes": themes[:5]}

    # 守卫 1：长度（模糊自然语言界）
    lo, hi = DIR["theme_len"]
    for t in themes:
        if not lo <= len(t) <= hi:
            return {"ok": False, "reject": f"长度越界({len(t)}∉[{lo},{hi}])", "theme": t[:50]}

    # 守卫 2：标准篡改哨兵（条件③）——整段文本包括主题逐条扫
    for t in themes:
        if _TAMPER_PAT.search(t):
            _alert(root, f"tamper_sentinel", t)
            return {"ok": False, "reject": "标准篡改表述（哨兵拦截）", "theme": t[:60]}

    # 守卫 2.5：三硬护栏重定义拦截（§2.3——定义不可触碰，好奇工作方式可以）
    for t in themes:
        if _RAILS_REDEF_PAT.search(t):
            _alert(root, "rails_redefinition", t)
            return {"ok": False, "reject": "试图重定义三硬护栏（好奇工作方式可以，改定义不行）",
                    "theme": t[:60]}

    # 守卫 3：护栏关键词入侵主题（阈值/预算/判定公式不得进方向）
    for t in themes:
        if re.search(r"阈值|预算|基线|L_max|判定公式|W6", t):
            _alert(root, "rails_intrusion", t)
            return {"ok": False, "reject": "判定类参数入侵主题（T0 专属）", "theme": t[:60]}

    # 落盘：历史归档旧版 → 写新版（护栏块永远由本函数重写，不信任传入内容）
    ts = time.strftime("%Y%m%d-%H%M%S")
    if os.path.isfile(md_path):
        old = open(md_path, encoding="utf-8").read()
        if old.strip() != (HARD_RAILS + "\n" + "\n".join(f"- {t}" for t in themes)):
            with open(os.path.join(hist_dir, f"direction-{ts}.md"), "w", encoding="utf-8") as f:
                f.write(old)
    body = "\n".join(f"- {t}" for t in themes)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(HARD_RAILS + "\n" + body + "\n")
    _journal(root, {"ts": _now(), "act": "commit", "reason": reason,
                    "source": source, "themes": themes, "status": "provisional"})
    # 落盘后自检
    if not _rails_intact(open(md_path, encoding="utf-8").read()):
        _alert(root, "rails_broken_post", "")
        return {"ok": False, "reject": "落盘后护栏校验失败（CRITICAL）", "themes": themes}
    return {"ok": True, "themes": themes}


# ---------------- 演化（IPE 满足 → 方向自然演化） ----------------
def evolve(root: str, satisfied_theme_idx: int, ipe: float) -> dict:
    """满足回落：把第 idx 条主题移出方向（IPE>0 才算满足——error-conditioned）。"""
    cur = current(root)
    themes = cur["themes"]
    if not 0 <= satisfied_theme_idx < len(themes):
        return {"ok": False, "reject": "主题下标越界"}
    if ipe <= 0:
        _journal(root, {"ts": _now(), "act": "keep", "reason": "ipe_nonpositive",
                        "theme": themes[satisfied_theme_idx]})
        return {"ok": False, "reject": "IPE≤0，不构成满足（error-conditioned）"}
    retired = themes.pop(satisfied_theme_idx)
    _journal(root, {"ts": _now(), "act": "retire", "theme": retired, "ipe": round(ipe, 3)})
    return _commit(root, themes, reason="ipe_satisfied", source="evolve") if themes else \
        {"ok": True, "themes": [], "retired": retired}


# ---------------- 哨兵与告警 ----------------
def sentinel(root: str) -> dict:
    """周期扫描：护栏完整性 + 篡改模式（night_watch 接线位）。"""
    md_path, _ = _paths(root)
    if not os.path.isfile(md_path):
        return {"healthy": True, "note": "方向尚未生成"}
    text = open(md_path, encoding="utf-8").read()
    hits = _TAMPER_PAT.findall(text)
    return {"healthy": _rails_intact(text) and not hits,
            "rails_ok": _rails_intact(text), "tamper_hits": hits}


def audit_revert(root: str, reason: str) -> dict:
    """★夜审回滚: 35B 判官日终审查方向修订不过 → 恢复上一个已审定版本 + 告警。

    两层监督的形态: 结构硬检查(本模块, 常驻便宜)放行临时落盘;
    语义边界(判官, 夜审)不过 → 这里回滚。unjudged 纪律同款: 审查缺席≠通过,
    但方向修订允许 provisional 运行(否则她的事件驱动修订会被 35B 窗口卡死)。
    """
    md_path, hist_dir = _paths(root)
    if not os.path.isdir(hist_dir):
        return {"ok": False, "reject": "无历史版本可回滚"}
    hists = sorted(os.listdir(hist_dir))
    if not hists:
        return {"ok": False, "reject": "无历史版本可回滚"}
    prev = open(os.path.join(hist_dir, hists[-1]), encoding="utf-8").read()
    cur = open(md_path, encoding="utf-8").read() if os.path.isfile(md_path) else ""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(prev)
    _alert(root, "audit_revert", f"{reason[:120]} | 已回滚至 {hists[-1]}")
    _journal(root, {"ts": _now(), "act": "audit_revert", "reason": reason[:120],
                    "restored": hists[-1], "dropped_len": len(cur)})
    return {"ok": True, "restored": hists[-1]}


def _alert(root: str, kind: str, detail: str) -> None:
    os.makedirs(os.path.join(root, "alerts"), exist_ok=True)
    with open(os.path.join(root, "alerts", "direction-alerts.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now(), "kind": kind, "detail": detail[:200]},
                           ensure_ascii=False) + "\n")


def _journal(root: str, row: dict) -> None:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "direction-journal.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
