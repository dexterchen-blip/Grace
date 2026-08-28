#!/usr/bin/env python3
"""心态轨引擎 —— 日级状态推演 + 日内变化机制（Grace V2 第三轨）。

核心洞察：心态既不是记忆也不是人格，是中速（日级）变量。
  日级：心态 = 昨日心态 × 0.7 + 今日事件增量 × 0.3（防"精神分裂"，Grace_v2 设计 §8.4）
  日内：当天内事件即时拨动"当前心态"，并随时间**指数衰减回归当日 base**——
        早上考砸很糟 → 中午被拉回专注 → 晚上逐步平复（像真人，而不是钉死一整天）。

存储：
  mood_states  表：日级基准（date/mood_label/intensity/...）
  mood_intraday 表：日内事件记录（date/ts/intensity/triggers/base_intensity/...）
铁律：心态永不改写 L0–L3；L3 高风险自发永不受心态影响。
"""
from __future__ import annotations
import json
import math
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
import config  # noqa: E402


SCHEMA = """
CREATE TABLE IF NOT EXISTS mood_states(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date TEXT NOT NULL,
  mood_label TEXT NOT NULL,
  intensity REAL NOT NULL DEFAULT 0.5,
  triggers TEXT DEFAULT '',
  summary TEXT DEFAULT '',
  source TEXT DEFAULT 'mood_engine',
  approved INTEGER DEFAULT 0,
  reset INTEGER DEFAULT 0,
  created REAL, updated REAL)
"""

INTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS mood_intraday(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date TEXT NOT NULL,
  ts REAL NOT NULL,
  mood_label TEXT NOT NULL,
  intensity REAL NOT NULL,
  triggers TEXT DEFAULT '',
  base_intensity REAL NOT NULL,
  event_text TEXT DEFAULT '',
  source TEXT DEFAULT 'intraday_engine',
  created REAL)
"""

# 日内衰减参数（情绪半衰期 τ：2 小时后事件影响剩 ~37%）
TAU_HOURS = 2.0
# 事件即时影响幅度（常规 0.4；weight>0.7 的大事 0.6——单件大事可拉高）
IMPACT = 0.4
IMPACT_BIG = 0.6


def _conn(db: str = None):
    db = db or config.L2_DB
    os.makedirs(os.path.dirname(db), exist_ok=True)
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    con.execute(INTRA_SCHEMA)
    con.commit()
    return con


def latest(db: str = None) -> dict | None:
    con = _conn(db)
    row = con.execute(
        "SELECT date,mood_label,intensity,triggers,summary,source,approved,reset "
        "FROM mood_states ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    if not row:
        return None
    return {"date": row[0], "mood_label": row[1], "intensity": row[2],
            "triggers": row[3], "summary": row[4], "source": row[5],
            "approved": row[6], "reset": row[7]}


def _label_from(intensity: float, delta: float) -> str:
    """强度 + 情绪方向 → 词表标签（日级/日内共用）。"""
    if intensity >= 0.8:
        return "兴奋" if delta > 0 else "焦虑"
    if intensity >= 0.6:
        return "轻微兴奋" if delta >= 0 else "专注"
    if intensity <= 0.42:
        return "低落"
    return "平静"


def derive(events: list[dict], db: str = None, ts: float | None = None) -> dict:
    """从今日事件流推演心态（规则引擎）。

    events: [{"text": ..., "sentiment": -1..1, "weight": 0..1}, ...]
    ts 可注入（夜班推演时刻/测试模拟时间线）；默认现在。
    完整 mood_engine（35B 推演）是开放问题，本版先给可跑的规则引擎骨架。
    """
    ts = ts or time.time()
    prev = latest(db)
    base = prev["intensity"] if prev else config.MOOD["default_intensity"]
    delta = 0.0
    triggers = []
    for e in events:
        delta += (e.get("sentiment", 0.0) * e.get("weight", 0.5))
        triggers.append(e.get("text", "")[:80])
    # 情绪增量映射到 0..1（负=坏事趋 0，正=好事趋 1）
    delta_mapped = max(0.0, min(1.0, (delta + 1) / 2))
    # 心态 = 昨日 × 0.7 + 今日项 × 0.3
    #   今日项 = 0.5(中性基线) + (delta_mapped-0.5)×2 × 0.3 → 拆开即：
    #   intensity = prev×0.7 + 0.15 + (delta_mapped-0.5)×0.6（clamp 0-1）
    #   中性稳态=0.5；好事 +0.24/日；坏事 -0.24/日；跳变<0.4 平滑
    intensity = round(max(0.0, min(1.0,
                     base * config.MOOD["decay"] + 0.15 + (delta_mapped - 0.5) * 0.6)), 3)

    # 词表映射（按强度 + 情绪方向选标签）
    label = _label_from(intensity, delta)

    rec = {"date": time.strftime("%Y-%m-%d", time.localtime(ts)), "mood_label": label, "intensity": intensity,
           "triggers": json.dumps(triggers[:5], ensure_ascii=False),
           "summary": f"规则引擎推演（锚点{config.PERSONA['name']}）：事件增量 Δ={delta:+.2f}",
           "source": "mood_engine", "approved": 0, "reset": 0}
    con = _conn(db)
    con.execute(
        "INSERT INTO mood_states(date,mood_label,intensity,triggers,summary,source,approved,reset,created,updated) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (rec["date"], rec["mood_label"], rec["intensity"], rec["triggers"], rec["summary"],
         rec["source"], rec["approved"], rec["reset"], ts, ts))
    con.commit()
    con.close()
    print(f"[v2-mood] 心态推演 → {label}（强度 {intensity}）")
    return rec


def reset(db: str = None) -> None:
    """一键重置心态（人审闸门允许手动覆盖）。"""
    con = _conn(db)
    con.execute("UPDATE mood_states SET reset=1, updated=? WHERE id=(SELECT MAX(id) FROM mood_states)", (time.time(),))
    con.commit()
    con.close()
    print("[v2-mood] 心态已重置（当日态归位）")


def derive_with_35b(events: list[dict], api_url: str = None, db: str = None) -> dict:
    """35B 推演版（M3-① 骨架，白天不跑——35B 错峰铁律）。

    完整版：夜班把当日事件流喂给 35B（沙盒端口 18200），让它产出 mood_label/intensity/triggers，
    再走平滑衰减 + 人审闸门。当前骨架：无 35B 服务时回退规则引擎。
    """
    import urllib.request
    api_url = api_url or "http://127.0.0.1:18200/v1/chat/completions"
    try:
        prompt = ("你是心态分析师。根据以下今日事件，判断 AI 助手的明日心态"
                  "（mood_label ∈ " + str(config.MOOD["labels"]) +
                  "，intensity 0-1，并列出触发事件）。事件：\n" +
                  "\n".join(f"- {e.get('text','')} (sentiment={e.get('sentiment',0)})" for e in events))
        body = json.dumps({"model": "Qwen3.5-35B-A3B", "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 200}).encode()
        req = urllib.request.Request(api_url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read().decode())
        text = out["choices"][0]["message"]["content"]
        # 粗解析（完整版用 json 输出协议）
        label = next((l for l in config.MOOD["labels"] if l in text), "平静")
        return derive([{"text": f"35B推演: {text[:100]}", "sentiment": 0.5, "weight": 0.5}], db=db)
    except Exception as e:  # noqa: BLE001
        print(f"[v2-mood] 35B 推演不可用（{e}），回退规则引擎")
        return derive(events, db=db)


def timeline(db: str = None, limit: int = 7) -> list[dict]:
    con = _conn(db)
    rows = con.execute(
        "SELECT date,mood_label,intensity,triggers,approved,reset FROM mood_states "
        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [{"date": r[0], "mood_label": r[1], "intensity": r[2], "triggers": r[3],
             "approved": r[4], "reset": r[5]} for r in rows]


# ================= 日内变化机制（M3 增强） =================

def _day_base(date: str, db: str = None) -> float:
    """当日日级基准强度（mood_states 中当天的记录；无则默认 0.5）。"""
    con = _conn(db)
    row = con.execute("SELECT intensity FROM mood_states WHERE date=? ORDER BY id DESC LIMIT 1",
                      (date,)).fetchone()
    con.close()
    return row[0] if row else config.MOOD["default_intensity"]


def _decay_factor(ts: float, now: float) -> float:
    """事件影响的时间衰减：exp(-Δt/τ)。τ=TAU_HOURS 小时后影响剩 ~37%。"""
    hours = max(0.0, (now - ts) / 3600.0)
    return math.exp(-hours / TAU_HOURS)


def apply_intraday_event(event: dict, db: str = None, ts: float = None,
                         base: float | None = None) -> dict:
    """日内事件即时拨动心态（写 mood_intraday）。

    event: {"text", "sentiment": -1..1, "weight": 0..1}
    机制：强度 = clamp(当日base + sentiment×impact×event_weight, 0, 1)
         impact：常规 0.4，weight>0.7 的大事 0.6（单件大事可拉高，琐事只微调）。
    ts 可注入（测试模拟一天内不同时刻）。
    """
    ts = ts or time.time()
    date = time.strftime("%Y-%m-%d", time.localtime(ts))
    base = base if base is not None else _day_base(date, db)
    sentiment = max(-1.0, min(1.0, event.get("sentiment", 0.0)))
    weight = max(0.0, min(1.0, event.get("weight", 0.5)))
    impact = IMPACT_BIG if weight > 0.7 else IMPACT
    intensity = round(max(0.0, min(1.0, base + sentiment * impact * weight)), 3)
    label = _label_from(intensity, sentiment)

    rec = {"date": date, "ts": ts, "mood_label": label, "intensity": intensity,
           "triggers": event.get("text", "")[:80],
           "base_intensity": base, "event_text": event.get("text", "")[:80],
           "source": "intraday_engine", "created": time.time()}
    con = _conn(db)
    con.execute(
        "INSERT INTO mood_intraday(date,ts,mood_label,intensity,triggers,base_intensity,event_text,source,created) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (rec["date"], rec["ts"], rec["mood_label"], rec["intensity"], rec["triggers"],
         rec["base_intensity"], rec["event_text"], rec["source"], rec["created"]))
    con.commit()
    con.close()
    print(f"[v2-mood] 日内事件 → {label}（强度 {intensity}，base={base}，{event.get('text','')[:20]}）")
    return rec


def current_intraday(db: str = None, now: float = None) -> dict | None:
    """当前时刻的日内心态（最近一次日内事件 + 衰减回归当日 base）。

    无日内事件 → 返回 None（调用方回退日级心态）。
    """
    now = now or time.time()
    date = time.strftime("%Y-%m-%d", time.localtime(now))
    con = _conn(db)
    row = con.execute(
        "SELECT date,ts,mood_label,intensity,triggers,base_intensity,event_text "
        "FROM mood_intraday WHERE date=? ORDER BY id DESC LIMIT 1", (date,)).fetchone()
    con.close()
    if not row:
        return None
    base = row[5]
    event_intensity = row[3]
    drift = event_intensity - base                      # 事件造成的偏移
    cur = round(max(0.0, min(1.0, base + drift * _decay_factor(row[1], now))), 3)
    label = _label_from(cur, drift)
    return {"date": row[0], "ts": row[1], "mood_label": label, "intensity": cur,
            "triggers": row[4], "base_intensity": base, "event_text": row[6],
            "decayed": abs(drift) > 1e-9 and _decay_factor(row[1], now) < 0.999}


def intraday_timeline(db: str = None, date: str = None, limit: int = 20) -> list[dict]:
    """当天日内曲线（按时间升序）。"""
    date = date or time.strftime("%Y-%m-%d")
    con = _conn(db)
    rows = con.execute(
        "SELECT date,ts,mood_label,intensity,triggers,base_intensity,event_text "
        "FROM mood_intraday WHERE date=? ORDER BY ts DESC LIMIT ?", (date, limit)).fetchall()
    con.close()
    return [{"date": r[0], "ts": r[1], "mood_label": r[2], "intensity": r[3],
             "triggers": r[4], "base_intensity": r[5], "event_text": r[6]} for r in rows]


# ================= 三层情绪融合（人格底色 × 长期趋势 × 短期日内） =================

def persona_baseline(persona: str | None = None) -> float:
    """LoRA 人格情绪底色（config.PERSONA.mood_baseline，默认 0.5）。

    雷姆（rem）：外冷内热、忠诚奉献，底色中性偏内敛 → 0.5（不改变她的深情底色，
    但作为"情绪基线锚点"参与融合，让同事件下她的表达更克制）。
    """
    persona = persona or config.PERSONA["name"]
    return float(config.PERSONA.get("mood_baseline", 0.5))


def long_term_trend(db: str = None, days: int = 7, now: float = None) -> dict:
    """长期情绪趋势（历史 mood_states 时间线，跨窗口对比）。

    来源 = 情绪记忆（mood_states 历史），反映"这段时间整体心态"（慢变量）。
    direction = 当前 7 天均值 vs 上一 7 天均值：回升/下行/平稳（跨窗口，对单边持续敏感）。
    返回 {mean, slope, direction, n}：mean=当前窗口水平，slope=(当前-上一)/days。
    """
    now = now or time.time()
    con = _conn(db)
    cutoff = now - 2 * days * 86400
    rows = con.execute(
        "SELECT created, intensity FROM mood_states WHERE created>=? AND created<=? ORDER BY created ASC",
        (cutoff, now)).fetchall()
    con.close()
    if len(rows) < 2:
        return {"mean": persona_baseline(), "slope": 0.0, "direction": "平稳", "n": len(rows)}
    # 按时间切两半：上一窗口(旧) vs 当前窗口(新)
    mid = len(rows) // 2
    first = sum(r[1] for r in rows[:mid]) / mid
    second = sum(r[1] for r in rows[mid:]) / (len(rows) - mid)
    mean = round(second, 3)
    slope = round((second - first) / days, 4)
    direction = "回升" if second - first > 0.01 else ("下行" if first - second > 0.01 else "平稳")
    return {"mean": mean, "slope": slope, "direction": direction, "n": len(rows),
            "prev_mean": round(first, 3)}


def combined_emotion(db: str = None, persona: str | None = None, now: float = None) -> dict:
    """三层情绪融合 —— 最终驱动注入与自发的当前情绪。

      长期锚点 anchor = 0.6×历史趋势均值 + 0.4×人格底色（慢变量，LoRA 模型 + 情绪记忆共同决定）
      有日内事件: combined = 0.3×anchor + 0.7×日内当前（当下感觉最强烈）
      无日内事件: combined = 0.5×anchor + 0.5×日级 base

    返回分层分解（baseline/trend/daily/intraday/combined/label），便于审计与实验。
    """
    now = now or time.time()
    date = time.strftime("%Y-%m-%d", time.localtime(now))
    base_p = persona_baseline(persona)
    trend = long_term_trend(db, days=7, now=now)
    anchor = round(0.6 * trend["mean"] + 0.4 * base_p, 3)
    daily = _day_base(date, db)
    intra = current_intraday(db=db, now=now)

    if intra:
        combined = round(0.3 * anchor + 0.7 * intra["intensity"], 3)
        layer = "intraday"
    else:
        combined = round(0.5 * anchor + 0.5 * daily, 3)
        layer = "daily"
    drift = combined - anchor
    label = _label_from(combined, drift)
    return {
        "persona": persona or config.PERSONA["name"],
        "baseline": base_p, "trend_mean": trend["mean"], "trend_direction": trend["direction"],
        "anchor": anchor, "daily": daily,
        "intraday": intra["intensity"] if intra else None,
        "intraday_trigger": intra["event_text"] if intra else None,
        "combined": combined, "label": label, "layer": layer, "now": now,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--timeline":
        for m in timeline():
            print(m)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset()
        sys.exit(0)
    print("CLI 冒烟：推演一条示例心态")
    r = derive([{"text": "LongMemEval R@5=0.96，很顺", "sentiment": 0.8, "weight": 0.7}])
    print("→", r)
