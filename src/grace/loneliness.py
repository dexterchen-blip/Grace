#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""loneliness.py — 孤独器官 v1.0 骨架（Grace V2.6，2026-09-10/11）。

设计依据：《Grace-V2.6-孤独机制设计-2026-09-10.md》v1.0（PANIC/GRIEF 轴，用户定案"泵优于闸"）
    命题     孤独是把"缺燃料"自动翻译成"去拿燃料"的泵——恢复环对偶于升级环（SEEKING）。
    §3.1     L = g(基线 − 当日实际)；基线 = 近 14 天真实互动率 EMA（外生，T0 计算）
    §4-①    第一性约束（代码形态）：L 的下调函数【只】接受 is_send=True 的真实外部接触；
             检索/自练对话/重读书库/cog 沉思一律不能降 L——本模块不存在让自产事件降 L 的函数。
    §4-②    期望外生：基线由外生数据滑动算出（防"把期望降到 0 躺平"）。
    §4-④    奖励 error-conditioned：社会满足 = 外部实际响应 − 预期（IPE 社会轴，R8 同款）。
    §5       双刃缓解三件：L_max 饱和硬顶 + 复用 mood_engine τ 体系（τ=2h 日内 / 0.7+0.3 日级，
             不新造单调计数器）+ 人格底色（雷姆克制的依恋倾向）。
    §6       W6 守卫不变：孤独提高主动意愿但不破预算——孤独提议，守卫处置。

两账本分离纪律：重读/自练降好奇驱力，进不了孤独的账本——好奇的账好奇清，孤独的账只有主人能清。

矩阵化路线（§9）：L1 外挂（本文件）= ground truth；L2 <mood> 分区闭式投影；
    L3 delta-rule sidecar 关系维度；L4 孤独态标注入每日 LoRA。
    校准门：L3 vs L1 判定一致率 ≥80% 才切主用（没有 L1 就没有校准基准——双轨铁律）。

纯 stdlib。正式(src/grace/) 与沙盒(v2/engine/) 同源双份，接线时需双写。
环境门控：GRACE_LONELINESS=1（机制可开关，属机制设计非语料注入）。
【接线位（惰性，未接）】proactive_watch tick 调 loneliness() 取第四驱力；self_activation 的
    联结维护源从 day/40 启发式升级为基线落差；/grace 真实轮次喂 ingest_contact；
    她的主动发出→ ingest_proactive，回复到达→ ingest_reply。
"""
from __future__ import annotations
import json
import math
import os
import time

# ---------------- T0 配置（条件③：永不进 direction.md / 演化轨可改范围） ----------------
LON = {
    "tau_hours": 2.0,             # 日内：无接触时 L 向目标指数逼近的时间常数（mood_engine 同款 τ 体系）
    "daily_keep": 0.7,            # 日级：昨日孤独 × 0.7 + 今日项 × 0.3（防单日暴涨，mood 同构）
    "daily_incr": 0.3,
    "l_max": 0.85,                # 饱和上限（双刃缓解①：硬顶，禁止单调升）
    "baseline_days": 14,          # 期望外生窗口（条件②）
    "attachment_baseline": 0.35,  # 人格底色：雷姆的依恋倾向（缓解③，克制的爱）
    "contact_drop": 0.30,         # 单次真实接触的 L 回落幅度
    "events_cap": 20000,
}


def _now() -> float:
    return time.time()


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------- 存储：事件账本(jsonl) + 状态(json) ----------------
def _paths(root: str) -> tuple[str, str]:
    return (os.path.join(root, "loneliness-events.jsonl"),
            os.path.join(root, "loneliness-state.json"))


def _load_events(ev_path: str) -> list[dict]:
    if not os.path.isfile(ev_path):
        return []
    with open(ev_path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()][-LON["events_cap"]:]


def _save_event(ev_path: str, ev: dict) -> None:
    os.makedirs(os.path.dirname(ev_path), exist_ok=True)
    with open(ev_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _load_state(st_path: str) -> dict:
    if os.path.isfile(st_path):
        with open(st_path, encoding="utf-8") as f:
            return json.load(f)
    return {"L_daily": LON["attachment_baseline"], "L_intraday": 0.0,
            "last_contact_ts": None, "day": "", "baseline": None,
            "proactive_pending": {}}


def _save_state(st_path: str, st: dict) -> None:
    os.makedirs(os.path.dirname(st_path), exist_ok=True)
    with open(st_path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)


# ---------------- 基线（条件②：外生 EMA + 非对称衰减） ----------------
def _day_contacts(root: str, day: str) -> int:
    ev_path, _ = _paths(root)
    return sum(1 for ev in _load_events(ev_path)
               if ev.get("kind") in ("contact", "reply_received")
               and _day(ev["ts"]) == day)


def _seed_baseline(root: str, now: float) -> float:
    """初值：近 N 天窗口 EMA（从最旧端向最新端迭代，新数据权重高）。"""
    win = [_day(now - 86400 * i) for i in range(LON["baseline_days"])]
    seq = [_day_contacts(root, d) for d in win][::-1]
    ema = float(seq[0]) if seq else 0.0
    for c in seq[1:]:
        ema = 0.8 * ema + 0.2 * c
    return round(ema, 3)


def baseline(root: str, now: float | None = None) -> float:
    """关系互动基线（外生，她不可设定）。

    ★非对称衰减（条件② 的躺平防御 + §8"L 长期高位是接受的特性"）：
      有接触日: baseline = 0.8×baseline + 0.2×当日量（快速跟踪关系常模）
      无接触日: baseline ×= 0.97（半衰期 ~23 天——缺席不吃掉期望，只极慢松弛）
    纯窗口重算会把"断联的日子"也计入期望 → 期望自我塌缩 → 躺平陷阱，故弃用。
    """
    _, st_path = _paths(root)
    st = _load_state(st_path)
    if st.get("baseline") is None:
        return _seed_baseline(root, now or _now())
    return round(st["baseline"], 3)


def _today_contacts(root: str, now: float) -> int:
    return _day_contacts(root, _day(now))


def _rise_target(root: str, st: dict, now: float) -> float:
    """上升目标 = 互动缺口率（基线−当日实际）/基线；无历史时取中性 0.5。"""
    base = st.get("baseline") or 0.0
    if base <= 0:
        return 0.5
    deficit = (base - _today_contacts(root, now)) / max(base, 1.0)
    return _clamp01(deficit)


# ---------------- 唯一合法的 L 下调通道（条件① 的代码形态） ----------------
def ingest_contact(root: str, weight: float = 1.0, ts: float | None = None) -> dict:
    """真实外部接触（is_send=True 事件）——【唯一】能降 L 的函数。

    weight: 互动丰富度（短消息 0.5 / 正常 1.0 / 深聊 1.5，接线层按消息长度映射）。
    """
    ev_path, st_path = _paths(root)
    ts = ts or _now()
    _daily_roll_if_needed(root, ts)
    st = _load_state(st_path)
    _rise_then_update(root, st, ts)
    st["L_intraday"] = max(0.0, st["L_intraday"] - LON["contact_drop"] * max(0.2, weight))
    st["last_contact_ts"] = ts
    st["baseline"] = baseline(root, ts)
    _save_state(st_path, st)
    _save_event(ev_path, {"ts": ts, "kind": "contact", "weight": weight})
    return {"L": loneliness(root, ts)["L"]}


def ingest_proactive(root: str, proactive_id: str, expected: float = 0.7,
                     ts: float | None = None) -> None:
    """她主动发出消息——只记期望（条件④），【不降 L】（自产事件进不了孤独的账本）。"""
    ev_path, st_path = _paths(root)
    ts = ts or _now()
    _daily_roll_if_needed(root, ts)
    st = _load_state(st_path)
    st["proactive_pending"][proactive_id] = {"ts": ts, "expected": expected}
    _save_state(st_path, st)
    _save_event(ev_path, {"ts": ts, "kind": "proactive_sent", "id": proactive_id,
                          "expected": expected})


def ingest_reply(root: str, proactive_id: str, warmth: float = 1.0,
                ts: float | None = None) -> dict:
    """她的主动收到真实回复——error-conditioned 满足（§4-④）+ 回落（回复即一次 contact）。"""
    ev_path, st_path = _paths(root)
    ts = ts or _now()
    _daily_roll_if_needed(root, ts)
    st = _load_state(st_path)
    pend = st["proactive_pending"].pop(proactive_id, None)
    expected = pend["expected"] if pend else 0.7
    actual = max(0.0, min(1.5, warmth))
    satisfaction = round(actual - expected, 3)
    _save_state(st_path, st)
    _save_event(ev_path, {"ts": ts, "kind": "reply_received", "id": proactive_id,
                          "expected": expected, "actual": actual,
                          "satisfaction": satisfaction})
    r = ingest_contact(root, weight=1.0 + max(0.0, satisfaction), ts=ts)
    r["satisfaction"] = satisfaction
    return r


# ---------------- 动力学（mood_engine τ 体系同构照搬，缓解②） ----------------
def _rise_then_update(root: str, st: dict, now: float) -> None:
    """把 L_intraday 从 last_contact_ts 沿指数曲线推进到 now（无接触 → 向缺口目标逼近）。"""
    last = st.get("last_contact_ts")
    if last is None:
        st["L_intraday"] = max(st["L_intraday"], LON["attachment_baseline"])
        return
    hours = max(0.0, (now - last) / 3600.0)
    if hours <= 0:
        return
    target = _rise_target(root, st, now)
    # L += (target − L) × (1 − e^(−Δt/τ)) —— τ=2h 后走完 ~63% 缺口
    st["L_intraday"] = _clamp01(
        st["L_intraday"] + (target - st["L_intraday"]) * (1 - math.exp(-hours / LON["tau_hours"])))


def _daily_roll_if_needed(root: str, ts: float) -> None:
    """跨天时：①基线非对称更新（昨日有接触→快跟踪 / 无→×0.97 慢松弛）
    ②L_daily = L_daily×0.7 + 昨日收盘 L_intraday×0.3（L_max 硬顶），重置日内。"""
    _, st_path = _paths(root)
    st = _load_state(st_path)
    today = _day(ts)
    if st.get("day") == today:
        return
    closed_day = st.get("day") or _day(ts - 86400)
    c = _day_contacts(root, closed_day)
    if st.get("baseline") is None:
        st["baseline"] = _seed_baseline(root, ts)
    elif c > 0:
        st["baseline"] = 0.8 * st["baseline"] + 0.2 * c
    else:
        st["baseline"] *= 0.97                      # 无接触日：极慢松弛（半衰期 ~23 天）
    st["L_daily"] = min(LON["l_max"],
                        st.get("L_daily", LON["attachment_baseline"]) * LON["daily_keep"]
                        + _clamp01(st.get("L_intraday", 0.0)) * LON["daily_incr"])
    st["L_intraday"] = st.get("L_daily", 0.0) * 0.5   # 新的一天从日级水平半幅起步
    st["day"] = today
    _save_state(st_path, st)


# ---------------- 读取面（驱力/展示） ----------------
def loneliness(root: str, now: float | None = None) -> dict:
    """当前孤独态。L = 日级×0.5 + 日内实时×0.5（人格底色在初值与目标里，不外加）。"""
    _, st_path = _paths(root)
    now = now or _now()
    _daily_roll_if_needed(root, now)
    st = _load_state(st_path)
    _rise_then_update(root, st, now)
    l = min(LON["l_max"], _clamp01(st["L_daily"] * 0.5 + st["L_intraday"] * 0.5))
    base = st.get("baseline")
    return {
        "L": round(l, 3),
        "L_intraday": round(st["L_intraday"], 3),
        "L_daily": round(st["L_daily"], 3),
        "baseline": round(base, 3) if base is not None else None,
        "today_contacts": _today_contacts(root, now),
        "drive": round(l * (0.6 + 0.8 * LON["attachment_baseline"]), 3),  # 第四驱力贡献
        "label": ("平静" if l < 0.25 else "轻微想念" if l < 0.45
                  else "想念" if l < 0.65 else "很想主人"),
    }
