#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""curiosity.py — 好奇心器官 v1.0 骨架（Grace V2.4，2026-09-09）。

设计依据：《Grace-V2.4-好奇心器官设计-2026-09-09.md》v1.1（文献调研 R1-R8 修正版）
    R1  激活 = 线索重现触发（Ovsiankina 稳健），**无时间上升项**（Zeigarnik 2025 元分析不支持）
    R2  置信度倒U：f(conf)=1-(2·conf-1)²，峰值 0.5（TOT 态"记不清+部分线索"最高，
        Kang 2009：12/19 被试峰值 0.40-0.60）；完全无记录=谷底
    R3  I/D 双轨：I=兴趣型（G3 新实体/G4 未接话题，探索式）D=剥夺型（G1 检索失败/G2 记不清）
    R5  满足按 IPE（信息预测误差）= 实际满意度 − 预期，非固定值
    R6  可解性门控：solvable=False 的缺口激活只降不升（防执念）
    R4  curiosity_state：top 缺口激活超阈值 → 附带记忆增强窗口标志

账本文件：JSONL，每行一个缺口：
    {"gap_id","track":"I"/"D","text","conf":0-1,"cues":[...],"revisits":int,
     "attempts":int,"solvable":bool,"next_step","born_ts","last_seen","resolved_ts":null}

纯 stdlib。formal(src/grace/) 与沙盒(v2/engine/) 同源双份，改动需双写。
环境门控：GRACE_CURIOSITY=1（机制可开关，属机制设计非语料注入）。
"""
from __future__ import annotations
import json
import os
import re
import time

LEDGER_CAP = 12
ASK_THRESHOLD = 0.60          # 第四触发源出队阈值
DAILY_ASK_BUDGET = 1          # 每日主动提问预算（W6 防好奇心过载）
STATE_THRESHOLD = 0.75        # curiosity_state（R4 附带记忆增强窗口）阈值
_MAX_ATTEMPTS_EFFECT = 3
_MAX_REVISITS_EFFECT = 3

_OWNER = "陈泽"


def _now() -> float:
    return time.time()


def load_ledger(path: str) -> list:
    try:
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except Exception:  # noqa: BLE001
        return []


def save_ledger(path: str, ledger: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for g in ledger[-LEDGER_CAP:]:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")


def _norm(t: str) -> str:
    return re.sub(r"[\s，。,.:：;；!！?？@~～]+", "", t or "")


_PII_PAT = re.compile(r"\d{7,}")
_JUNK_PAT = re.compile(r"http|www\.|【|推荐码|- \[ \]|加一下我的微信")


def _clean_pii(text: str) -> str:
    """★v1.1 PII 脱敏：≥7 位数字串只留前 3 位（手机号/证件号明文入账实测）。"""
    return _PII_PAT.sub(lambda m: m.group()[:3] + "****", text or "")


def _overlap(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if len(na) < 2 or len(nb) < 2:
        return 0.0
    ga = {na[i:i + 2] for i in range(len(na) - 1)}
    gb = {nb[i:i + 2] for i in range(len(nb) - 1)}
    return len(ga & gb) / len(ga | gb)


def f_conf(conf: float) -> float:
    """R2 置信度倒U：峰值 0.5（TOT 态），两端谷底。"""
    c = min(1.0, max(0.0, conf))
    return 1.0 - (2 * c - 1) ** 2


def gap_activation(g: dict, now: float | None = None) -> float:
    """R1 Ovsiankina 重访模型：无常数时间上升项；激活由线索重现(revisits)驱动。
    激活 = f_conf(倒U) × [0.4 + 0.2×min(revisits,3)] × solvable门控 × attempts抑制 × 新近保活"""
    if g.get("resolved_ts"):
        return 0.0
    now = now or _now()
    # 新近保活：last_seen 超过 7 天的缺口自然衰减（不是增长，是遗忘）
    age_days = max(0.0, (now - g.get("last_seen", g.get("born_ts", now))) / 86400)
    recency = 1.0 if age_days <= 7 else max(0.2, 1.0 - (age_days - 7) / 14)
    conf = float(g.get("conf", 0.5))
    base = f_conf(conf) * (0.4 + 0.2 * min(int(g.get("revisits", 0)), _MAX_REVISITS_EFFECT))
    if not g.get("solvable", True):
        base *= 0.3                                   # R6 不可解：只降不升
    base *= 1.0 - 0.3 * min(int(g.get("attempts", 0)), _MAX_ATTEMPTS_EFFECT)
    return round(min(1.0, base) * recency, 3)


def record_gap(path: str, text: str, track: str = "D", conf: float = 0.5,
               cues: list | None = None, solvable: bool = True,
               next_step: str | None = None) -> dict:
    """记录/更新缺口。与既有条目重叠≥0.5 视为同一缺口 → 算一次线索重现（R1 +Δ）。"""
    ledger = load_ledger(path)
    now = _now()
    text = _clean_pii(text or "")          # ★v1.1 PII 脱敏——数字串≥7位打码(手机号明文入账实测)
    cues = [_clean_pii(c) for c in (cues or [])]
    best, best_ov = None, 0.0
    for g in ledger:
        if g.get("resolved_ts"):
            continue
        ov = _overlap(g.get("text", ""), text)
        # 同一事件异措辞：cue 线索匹配兜底（实体级重现，如 "MicroFridge" 再被提到）
        for c in g.get("cues", []):
            ov = max(ov, _overlap(c, text))
        if ov > best_ov:
            best, best_ov = g, ov
    if best is not None and best_ov >= 0.4:
        best["revisits"] = int(best.get("revisits", 0)) + 1      # R1: 线索重现 +Δ
        best["last_seen"] = now
        best["conf"] = max(min(conf, 1.0), 0.0) if conf else best.get("conf", 0.5)
        if cues:
            best.setdefault("cues", []).extend([c[:40] for c in cues][:2])
        save_ledger(path, ledger)
        return best
    # ★G1 准入分级（v1.1）: 低置信谷底缺口（conf<0.3）最多 4 条——防"完全不知道"类
    #   洪泛占满账本，给 TOT/G2 高价值缺口腾位
    if track == "D" and conf < 0.3:
        low_n = sum(1 for g in ledger if g.get("track") == "D"
                    and float(g.get("conf", 0)) < 0.3 and not g.get("resolved_ts"))
        if low_n >= 4:
            return None
    g = {"gap_id": f"gap-{int(now*1000)}", "track": track, "text": text[:80],
         "conf": round(min(1.0, max(0.0, conf)), 2), "cues": [c[:40] for c in (cues or [])][:2],
         "revisits": 0, "attempts": 0, "solvable": bool(solvable),
         "next_step": next_step or "下次自然地问主人",
         "born_ts": now, "last_seen": now, "resolved_ts": None}
    ledger.append(g)
    ledger.sort(key=lambda x: -(x.get("born_ts", 0)))
    save_ledger(path, ledger[-LEDGER_CAP:])
    return g


def ingest_seeking(path: str, text: str, score: float) -> dict | None:
    """★2026-09-09 双向耦合（用户: "把好奇心系统和自激发系统串起来"）：
    自激发 SEEKING 高分但未过线的事件（差一点就想说了）→ 入账本 I 轨。
    这是"她想知道但没说出口"的天然信号源——I 轨从此有真实供血。"""
    if not text or float(score) < 0.3:
        return None
    # ★v1.1 I 轨质量门(本轮实测: spam/「哈哈」混入——SEEKING 0.3-0.5 带太宽):
    #   长度≥12 + 垃圾模式(链接/推荐码/任务列表标记/加微信)一票否决
    t = text.strip()
    if len(t) < 12 or _JUNK_PAT.search(t):
        return None
    conf = round(min(0.5, 0.3 + float(score) * 0.4), 2)   # score 0.3-0.5 → conf 0.42-0.5
    return record_gap(path, text=f"雷姆对这事有点好奇：{text[:60]}", track="I",
                      conf=conf, cues=[text[:40]], solvable=True,
                      next_step="找机会自然聊起")


def mark_attempt(path: str, gap_id: str) -> None:
    """第四触发源出队后记账（attempts 抑制：同一缺口不会追问不休）。"""
    ledger = load_ledger(path)
    for g in ledger:
        if g.get("gap_id") == gap_id and not g.get("resolved_ts"):
            g["attempts"] = int(g.get("attempts", 0)) + 1
            g["last_seen"] = _now()
            break
    save_ledger(path, ledger)


def top_gap(path: str, now: float | None = None) -> dict | None:
    """EPISTEMIC 触发源：返回激活最高且达阈值的未解决缺口。"""
    ledger = load_ledger(path)
    now = now or _now()
    best, best_a = None, 0.0
    for g in ledger:
        a = gap_activation(g, now)
        if a > best_a:
            best, best_a = g, a
    return {**best, "activation": best_a} if best and best_a >= ASK_THRESHOLD else None


def check_resolution(path: str, user_text: str, valence: float = 0.0) -> dict | None:
    """R5 IPE 满足检测：有过提问尝试的缺口，主人后续消息与其高重叠 → 视为回答。
    IPE = 实际满足(valence 感知) − 预期(0.3×激活)。返回满足事件（调用方发 mood 事件）。"""
    ledger = load_ledger(path)
    now = _now()
    for g in ledger:
        if g.get("resolved_ts") or int(g.get("attempts", 0)) < 1:
            continue
        if _overlap(g.get("text", ""), user_text) >= 0.4 or any(
                _overlap(c, user_text) >= 0.4 for c in g.get("cues", [])):
            act = gap_activation(g, now)
            ipe = round(max(-1.0, min(1.0, float(valence or 0) + 0.3 - 0.3 * act)), 2)
            g["resolved_ts"] = now
            save_ledger(path, ledger)
            return {"gap_id": g["gap_id"], "text": g["text"], "track": g.get("track", "D"),
                    "ipe": ipe, "resolved": True}
    return None


def curiosity_state(path: str, now: float | None = None) -> dict | None:
    """R4 附带记忆增强窗口：top 缺口激活 ≥ STATE_THRESHOLD → 返回该缺口（调用方给新
    摄入记忆加成）。无则 None。"""
    tg = top_gap(path, now)
    if tg and tg["activation"] >= STATE_THRESHOLD:
        return tg
    return None


def daily_budget_left(st: dict, today: str) -> bool:
    return st.get("curiosity_ask_date") != today
