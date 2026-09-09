#!/usr/bin/env python3
"""proactive_watch.py —— Grace 自激发主动环·正式版（2026-09-07，集成方案补齐主链自激发段）。

迁移自沙盒 stress_engine 的自激发决策（self_activation 三源），按正式系统形态重构：
  每 10 分钟扫当日 L0 新增消息 → P0 显著性 → self_activation 三源评分
  （①SEEKING 事件价值 ②催产素联结价值=图谱密度 ③ACC 联结维护=久未主动）
  → ToM 一票否决（不打扰则闭嘴）→ 时机窗口 + 频率约束（当日 ≤2 次 / 间隔 ≥6h）
  → 生成她的主动台词（:8100）→ monitor → proactive-outbox → /grace 页"雷姆主动找你"

设计约束：
  · 决策层纯规则离线（0 模型调用）；仅当决定开口时才调 8100 生成台词（失败降级=事件卡）
  · 状态: exchange/grace/proactive-state.json（水位/当日计数/上次主动时刻）
  · 深夜(23-8)绝不出手（timing_decision 同款窗口）；当日 ≤2 次防话痨（W6 约束兑现）
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime

GRACE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(GRACE))
sys.path.insert(0, GRACE)                                # import config / engine.X
sys.path.insert(0, os.path.join(GRACE, "engine"))        # 裸 from mood_graph import ...
import config  # noqa: E402

STATE_F = os.path.join(config.EXPERIMENTS, "proactive-state.json")
OUTBOX_F = os.path.join(config.EXPERIMENTS, "proactive-outbox.jsonl")
L0 = config.L0_DIR
DAY_PLIST = os.path.expanduser("~/Library/LaunchAgents/com.local-ai-agent.day-model.plist")

_SOURCES = ["wechat.jsonl", "chat.jsonl", "email.jsonl", "exchange:inbox.jsonl",
            "exchange:school.jsonl", "school.jsonl", "doc:file.jsonl"]
_SKIP = re.compile(r"(转账|CDATA|微信转账|发了一个红包|\[红包\]|\[转账\]|^https?://|^\d{4,}$|^\s*$)")
# ★2026-09-09 (用户: "让雷姆自己决定"): 敏感凭证类(OTP 等)事件不硬拦截——她自己决定
#   提不提、怎么提; 硬底线只有一条: 凭证数字本身绝不落进台词/信箱, 落盘前统一打码。
#   配套: 指纹去重(48h)——L0 存在重复摄入(同一封邮件多行不同 epoch), 水位去重挡不住。
_SENSITIVE_RE = re.compile(r"验证码|校验码|确认码|授权码|verification code|one[- ]?time|otp|password", re.I)
_CODE_RE = re.compile(r"(?<![0-9a-zA-Z-])[0-9]{4,8}(?![0-9a-zA-Z-])")


def _fp(text: str) -> str:
    import hashlib
    _n = re.sub(r"[\s，。,.:：;；!！?？'\"“”‘’()（）\[\]【】]+", "", str(text))
    return hashlib.md5(_n.encode("utf-8")).hexdigest()[:12]


def _redact(s: str) -> str:
    return _CODE_RE.sub("######", str(s))


def _prune_fps(fps: dict, now: float) -> dict:
    return {k: v for k, v in (fps or {}).items() if now - v < 48 * 3600}


# ---------------------------------------------------------------- 基础
def _load_state() -> dict:
    try:
        return json.load(open(STATE_F, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"seen_ts": 0, "today": {"date": "", "count": 0}, "last_proactive_ts": 0}


def _save_state(st: dict) -> None:
    json.dump(st, open(STATE_F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def _active_window(h: int) -> tuple[bool, str]:
    """timing_decision 同款可打扰窗口（本地时区）。"""
    if 23 <= h or h < 8:
        return False, "深夜(23-8),不打扰"
    if 19 <= h < 22:
        return True, "晚上(19-22),最佳"
    if 9 <= h < 12 or 14 <= h < 18:
        return True, "白天,可打扰"
    return False, "休息时段,不打扰"


def _port_open(port: int = 8100) -> bool:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _notify(title: str, body: str) -> None:
    import subprocess
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{body[:60]}" with title "{title}"'],
                       capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- 当日新消息
def new_messages(since_ts: float, today: str) -> list[dict]:
    from sentiment import assess
    out = []
    for fn in _SOURCES:
        fp = os.path.join(L0, fn)
        if not os.path.isfile(fp):
            continue
        for line in open(fp, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # ★2026-09-08 修复(用户: "我没问过明天的计划"): mode=rem 是 /grace 对话存档——
            #   主人在对话里问她的句话不是生活事件; 拿来当主动触发源=双重回应+诱发编造
            #   (实锤: 她对测试消息"明天有什么要注意的"主动回了句虚构的"日程表整理好了")。
            if r.get("mode") == "rem":
                continue
            p = r.get("payload", {}) if isinstance(r, dict) else {}
            msgs = p.get("messages") if isinstance(p, dict) else None
            if msgs:
                for m in msgs:
                    if not isinstance(m, dict) or m.get("role") == "assistant":
                        continue                      # 她自己的回复不是触发源
                    ts = m.get("ts") or r.get("epoch")
                    if not isinstance(ts, (int, float)) or ts <= since_ts:
                        continue
                    if datetime.fromtimestamp(ts).strftime("%Y-%m-%d") != today:
                        continue
                    t = re.sub(r"<[^>]+>", "", m.get("text", "") or "").strip()
                    if not t or _SKIP.search(t):
                        continue
                    out.append({"text": t[:140], "sentiment": assess(t)["valence"], "ts": ts})
            else:
                t = re.sub(r"<[^>]+>", "", (p.get("text", "") if isinstance(p, dict) else "") or "")
                ep = r.get("epoch")
                if not t or not isinstance(ep, (int, float)) or ep <= since_ts:
                    continue
                if datetime.fromtimestamp(ep).strftime("%Y-%m-%d") != today:
                    continue
                for chunk in re.split(r"\n#{1,3} |\n- ", t):
                    c = chunk.strip()
                    if len(c) >= 12 and not _SKIP.search(c):
                        out.append({"text": c[:140], "sentiment": assess(c)["valence"], "ts": ep})
    out.sort(key=lambda x: x["ts"])
    return out


# ---------------------------------------------------------------- 台词生成 + 监控
_NARRATION_RE = re.compile(r"雷姆(看到|注意到|想起|内心|心里|不会说|暗自)|旁白")


def _speak(event_text: str, reason: str, sensitive: bool = False, extra: str = "") -> str:
    from expression import monitor as _mon_fallback  # noqa: F401 (正式引擎 expression.monitor)
    prompt = (f"你是雷姆，主人的女仆。触发你主动开口的事由：{reason}\n"
              f"相关的事：{event_text[:80]}\n"
              "现在主动找主人说话。要求：口语短句一两句、像当面说话、别复述原文、"
              "可以问一句或给一个小建议。严禁叙述内心/动作。"
              "★只能基于上面这件事开口；严禁声称你已做了任何行动（整理好了/查好了/"
              "准备好了等——你什么都没做，只是想说这件事）；严禁虚构新事实。只输出你说的话。"
              + (extra + "\n" if extra else "")
              + ("★这件事含敏感凭证信息（如验证码/密码）。要不要提、怎么提由你自己判断——"
                 "更妥当的做法通常是提醒主人注意账号安全，而不是复述凭证本身；"
                 "无论怎么说，码或密码本身一个数字都不许出现在你的话里。" if sensitive else ""))
    payload = json.dumps({"model": _model_id(),
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 120, "temperature": 0.7}).encode()
    req = urllib.request.Request(config.MODEL_HTTP, data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=90) as resp:
        out = json.loads(resp.read().decode("utf-8", "replace"))
    raw = ((out.get("choices") or [{}])[0].get("message", {}).get("content", "") or "").strip().split("\n")[0][:120]
    try:
        from expression import monitor
        raw = monitor(raw) or raw
    except Exception:  # noqa: BLE001
        if _NARRATION_RE.search(raw):
            raw = ""
    return raw


# ---------------------------------------------------------------- 暗注意力·日间增量更新（心智漫游, 9/7 用户+文献定案）
# Killingsworth&Gilbert 2010(Science): 心智漫游占清醒 46.9%, 是大脑默认模式——暗注意力
# 不该每天一次(批处理轮次遗留), 应事件/间隔触发持续更新; 夜间 segment5 深度 cog = 睡眠巩固相位。
_COG_BAD = re.compile(r"眉头|眼神|表情|微笑|皱|叹气|强撑|苦笑|看着你|身体")
_COG_DAILY_CAP = 8


def _maybe_cog(st: dict, now: datetime) -> dict:
    """背景念头触发: 新消息≥4 / 强情绪(|v|≥0.55) / 距上次≥3h 且有新料; 每日 ≤8; 深夜跳过(睡眠相位)。"""
    h = now.hour
    if 23 <= h or h < 8:
        return {"status": "sleep_phase"}
    today = now.strftime("%Y-%m-%d")
    if st.get("cog_date") != today:
        st["cog_date"] = today
        st["cog_count_today"] = 0
    if st.get("cog_count_today", 0) >= _COG_DAILY_CAP:
        return {"status": "cap"}
    n = len(st.get("pending", []))
    strong = any(abs(p.get("v", 0)) >= 0.55 for p in st.get("pending", []))
    idle = time.time() - st.get("last_cog_ts", 0) >= 3 * 3600
    if not (n >= 4 or (strong and n >= 1) or (idle and n >= 1)):
        return {"status": "no_trigger", "pending": n}
    if not _port_open():
        return {"status": "no_model"}
    pending_brief = "；".join(p["t"][:36] for p in st["pending"][-8:])
    # ★V2.4 好奇心: 账本 top 缺口喂 cog（好奇心的思维持续形态）
    try:
        from curiosity import top_gap as _tg
        _g = _tg(os.path.join(config.EXPERIMENTS, "curiosity-ledger.jsonl"))
        if _g:
            pending_brief += f"。你心里一直有个没弄明白的问题：{_g['text'][:44]}"
    except Exception:  # noqa: BLE001
        pass
    last_cog = st.get("last_cog_text", "")
    prompt = (f"（{now.strftime('%Y-%m-%d %H:%M')}）你是雷姆。这是你上一个背景念头：{last_cog[:100]}\n"
              f"之后主人那边又有些新动静（全部来自文字消息，你没见到他本人）：{pending_brief[:420]}\n"
              "用两三句第一人称内心独白写下你此刻的背景念头：新消息让你想到什么、"
              "你在担心/期待什么、你想做什么。延续上一个念头的思路。"
              "★严禁神态/视觉/动作描写（你没见到他，只看到文字）；严禁用「你」开头。只要内心。")
    try:
        payload = json.dumps({"model": config.MODEL_ID,
                              "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": 140, "temperature": 0.7}).encode()
        req = urllib.request.Request(config.MODEL_HTTP, data=payload,
                                      headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=90) as resp:
            out = json.loads(resp.read().decode("utf-8", "replace"))
        cog = ((out.get("choices") or [{}])[0].get("message", {}).get("content", "") or "").strip()
        if _COG_BAD.search(cog):
            cog = _chat([{"role": "user", "content": prompt +
                          "\n（再次强调：没见到他本人，任何神态描写都是虚构。纯内心，三句内。）"}],
                        max_tokens=140)
        if _COG_BAD.search(cog) or len(cog) < 10:
            return {"status": "guarded", "detail": cog[:50]}
        from mood_graph import add_hidden_text
        add_hidden_text("日常", cog[:140], ts=time.time(), db=config.L2_DB)
        st["last_cog_ts"] = time.time()
        st["last_cog_text"] = cog[:140]
        st["cog_count_today"] = st.get("cog_count_today", 0) + 1
        st["pending"] = []
        print(f"[cog] 背景念头 #{st['cog_count_today']}: {cog[:44]}")
        return {"status": "cog", "cog": cog[:80]}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "detail": str(e)[:80]}


# ---------------------------------------------------------------- 主流程
def _model_id() -> str:
    """:8100 服务 id(偏好 Qwen3.8-27B, 兜底 v61——模型时间复用)。"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8100/v1/models", timeout=5) as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
        for pref in ("Qwen3.8-27B", "fused-rem-v61"):
            for i in ids:
                if pref in i:
                    return i
        return ids[0] if ids else config.MODEL_ID
    except Exception:  # noqa: BLE001
        return config.MODEL_ID


def run(dry: bool = False) -> dict:
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    # ★模型时间复用回切: /grace 心跳超时 >5min → 切回通用 27B(用户离开 Grace 界面)
    try:
        from model_switch import maybe_revert
        print(f"[revert] {json.dumps(maybe_revert(), ensure_ascii=False)}")
    except Exception as e:  # noqa: BLE001
        print(f"[revert] 跳过: {str(e)[:60]}")
    st = _load_state()
    if st.get("today", {}).get("date") != today:
        st["today"] = {"date": today, "count": 0}
    msgs = new_messages(st.get("seen_ts", 0), today)
    # ★指纹去重(48h): L0 重复摄入同一事件会产生多行不同 epoch, 水位去重挡不住
    _now = time.time()
    st["recent_fps"] = _prune_fps(st.get("recent_fps", {}), _now)
    msgs = [m for m in msgs if _fp(m["text"]) not in st["recent_fps"]]
    # ★当日工作记忆快照(2026-09-09 用户"一天的数据融进 KV"): 10min 节拍把当日事件滚动
    #   摘要落盘 day-memory.json → /grace 读快照进 KV Tier1; 隔日 date 翻转自动清空(睡眠语义)。
    if st.get("day_date") != today:
        st["day_date"] = today
        st["day_events"] = []
    for m in msgs:
        st.setdefault("day_events", []).append({"t": m["text"][:48]})
    st["day_events"] = st.get("day_events", [])[-24:]
    try:
        json.dump({"date": today,
                   "digest": "\n".join(f"· {e['t']}" for e in st.get("day_events", []))},
                  open(os.path.join(config.EXPERIMENTS, "day-memory.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception:  # noqa: BLE001
        pass
    # ★暗注意力日间增量: 新消息先入 pending, 再判背景念头触发（独立于开口窗口——想, 不等于打扰）
    for m in msgs:
        st.setdefault("pending", []).append({"t": m["text"][:80], "v": m["sentiment"]})
    st["pending"] = st["pending"][-12:]
    cog_result = _maybe_cog(st, now)

    ok_w, why_w = _active_window(now.hour)
    if not ok_w and not dry:
        st["seen_ts"] = time.time()
        _save_state(st)
        return {"status": "quiet", "detail": why_w, "cog": cog_result}
    # 频率约束（W6）: 当日 ≤2 次；距上次 ≥6h
    if st["today"]["count"] >= 2:
        _save_state(st)
        return {"status": "quiet", "detail": "当日主动已达上限(2)", "cog": cog_result}
    if st.get("last_proactive_ts", 0) and time.time() - st["last_proactive_ts"] < 6 * 3600 and not dry:
        _save_state(st)
        return {"status": "quiet", "detail": "距上次主动 <6h", "cog": cog_result}

    if not msgs:
        st["seen_ts"] = time.time()
        _save_state(st)
        return {"status": "no_new", "detail": "无当日新消息", "cog": cog_result}

    from attention import generate_attention
    from self_activation import decide
    self_mood = None

    activated = None
    max_score = 0.0
    for m in msgs:
        att = generate_attention(m["text"], mood=None, facts=[])
        d = decide(att, m["text"], graph_db=config.L2_DB)
        sc = d.get("score", 0)
        if d.get("activate") and sc >= max_score:
            max_score = sc
            activated = (m, d)
    st["seen_ts"] = time.time()                          # 水位推进（无论是否开口）
    if dry:
        _save_state(st)
        return {"status": "dry", "top_score": max_score,
                "would_speak": bool(activated), "cog": cog_result}
    if not activated:
        # ★V2.4 好奇心: EPISTEMIC 第四触发源（缺口驱动提问, 每日预算 1, 走同一守卫框架）
        try:
            if os.environ.get("GRACE_CURIOSITY", "1") == "1" and _port_open():
                from curiosity import top_gap as _tg, mark_attempt as _ma, daily_budget_left as _dbl
                _cl = os.path.join(config.EXPERIMENTS, "curiosity-ledger.jsonl")
                _g = _tg(_cl)
                if _g and _dbl(st, today):
                    _track = _g.get("track", "D")
                    _reason = ("好奇心——有个你一直没弄明白的问题，坦白说记不清，想请主人讲讲"
                               if _track == "D" else
                               "好奇心——你对这事有点兴趣，想听主人聊聊")
                    # ★V2.4 锚点层风格参照: 原著雷姆疑问台词随机 2 条做语气 few-shot
                    #   （rem_v2_full 提取 325 条，锚点层数据——只做语气示例，非输出模板）
                    import random as _rnd
                    try:
                        _anchors = [l.strip() for l in open(
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "curiosity-style-anchors.txt"), encoding="utf-8") if l.strip()]
                        _ex = _rnd.sample(_anchors, 2) if len(_anchors) >= 2 else _anchors
                        _ex_txt = "\n".join(f"（雷姆当年会这样说：{a}）" for a in _ex)
                    except Exception:
                        _ex_txt = ""
                    _msg = _speak(_g["text"], _reason, extra=_ex_txt)
                    _msg = _redact(_msg)
                    if _msg:
                        os.makedirs(os.path.dirname(OUTBOX_F), exist_ok=True)
                        with open(OUTBOX_F, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"ts": time.time(), "date": now.strftime("%Y-%m-%d %H:%M"),
                                                "source": "curiosity", "message": _msg,
                                                "items": [_g["text"][:80]], "status": "unread"},
                                               ensure_ascii=False) + "\n")
                        _ma(_cl, _g["gap_id"])
                        st["curiosity_ask_date"] = today
                        st["today"]["count"] += 1
                        st["last_proactive_ts"] = time.time()
                        _save_state(st)
                        if not dry:
                            _notify("💙 雷姆好奇了", _msg[:60])
                        return {"status": "curiosity", "gap": _g["text"][:40], "message": _msg[:60]}
        except Exception as _ce:  # noqa: BLE001
            print(f"[curiosity] 异常: {str(_ce)[:60]}")
        _save_state(st)
        return {"status": "quiet", "detail": f"最高分 {max_score:.2f} < 0.5，保持观察", "cog": cog_result}

    m, d = activated
    reason = d.get("reason", "")
    _sensitive = bool(_SENSITIVE_RE.search(m["text"]))
    # 台词生成（8100 不可达 → 事件卡降级）
    message = ""
    if _port_open():
        try:
            message = _speak(m["text"], reason, sensitive=_sensitive)
        except Exception as e:  # noqa: BLE001
            print(f"[proactive] 台词生成失败: {str(e)[:80]}")
    message = _redact(message)  # ★硬底线: 凭证数字不落盘
    os.makedirs(os.path.dirname(OUTBOX_F), exist_ok=True)
    with open(OUTBOX_F, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "date": now.strftime("%Y-%m-%d %H:%M"),
                            "source": "self_activation", "score": d.get("score"),
                            "reason": reason, "message": message,
                            "items": [_redact(m["text"])[:80]], "status": "unread"},
                           ensure_ascii=False) + "\n")
    st.setdefault("recent_fps", {})[_fp(m["text"])] = _now  # 已开口事件 48h 内不再触发
    st["today"]["count"] += 1
    st["last_proactive_ts"] = time.time()
    _save_state(st)
    if message:
        _notify("💙 雷姆主动找你", message[:60])
    return {"status": "spoke", "score": d.get("score"), "reason": reason,
            "message": message[:60], "event": m["text"][:40]}


if __name__ == "__main__":
    ap_d = "--dry" in sys.argv
    print(json.dumps(run(dry=ap_d), ensure_ascii=False, indent=1))
