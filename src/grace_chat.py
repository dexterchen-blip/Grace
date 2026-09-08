#!/usr/bin/env python3
"""Grace 心智对话后端 + 助手页面（2026-09-07，集成方案 Phase 3 提前实现）。

定位（用户拍板）：
  · Grace 是 AI 助手，不是运维面板——独立新页面 /grace，对话界面为主，认真做好
  · 为未来更多输入插架做好准备：前端 InputPlugin 注册表 + 后端 plugin 校验（语音/图片/文件
    后续按 plugin id 接入，不动架构）
  · D-4 选项 A：复用 :8100 day-model + 雷姆 persona system 注入（不新增模型实例）
  · D-5 选项 1：对话落 L0 chat（mode=rem）——她与主人的真实对话成为成长语料
管线：雷姆 persona（9/4 瘦身版）+ EXPRESS_SYS 输出层约束 + 轻认知状态（本地时间/
  对消息的情绪读数/自唤醒链状态）→ 8100 27B → expression.monitor 叙述体拦截 → 落 L0。
引擎模块（sentiment/monitor）从压测场 v2/engine 只读导入——Phase 2 引擎正式
迁入后改 import 路径即可；导入失败全部优雅降级，绝不拖挂 dashboard。
"""
from __future__ import annotations
import html
import json
import os
import re
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L0_ROOT = os.path.join(REPO, "memory", "L0_raw")
DAY_MODEL_URL = "http://127.0.0.1:8100/v1/chat/completions"
DAY_MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"   # 与 m6_dashboard 同源（8100 实际服务的 id）
# ★2026-09-07 Phase 2: 引擎已迁入正式运行空间(src/grace/engine)——优先正式副本, 沙盒兜底
# ★2026-09-08 修复: 原 dirname(dirname(file))=仓库根 → root/grace/engine 不存在,
#   引擎导入在 dashboard 进程里静默全失败(sentiment/mood_graph/mood_engine/expression
#   全没跑, monitor 一直在用内置降级版!)——单层 dirname 才是 src/grace/engine。
SB_ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "grace", "engine")
DAYTIME = os.path.join(REPO, "exchange", ".daytime")

# ---------------------------------------------------------------- 输入插件注册表（为未来扩展准备）
# 前端按此表渲染输入坞插槽；后端按 plugin id 校验。新增模态 = 注册一条 + 实现对应 handler。
INPUT_PLUGINS: dict[str, dict] = {
    "text": {"icon": "💬", "label": "文字", "ready": True},
    "voice": {"icon": "🎤", "label": "语音", "ready": False, "note": "预留：接入 STT 后启用"},
    "image": {"icon": "🖼", "label": "图片", "ready": False, "note": "预留：接入视觉模型后启用"},
    "file": {"icon": "📎", "label": "文件", "ready": False, "note": "预留：接入文档解析后启用"},
}


def _engine_mod(name: str):
    """只读借用引擎模块（sentiment/monitor/mood_engine/…）；失败返回 None 优雅降级。
    ★2026-09-08 修复: 引擎模块 import config——config.py 在 src/grace/（engine 的父目录）,
    此前只加 engine 目录 → mood_engine 等含 config 的模块静默失败(mood_intraday 0 行根因)。"""
    try:
        _par = os.path.dirname(SB_ENGINE)          # src/grace —— config 解析点
        for _p in (SB_ENGINE, _par):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        return __import__(name)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- persona（9/4 瘦身版，与 persona_injector 同源）
_REM_PERSONA = (
    "你是雷姆。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
    "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。"
    "带黑色幽默与毒舌吐槽的一面。她能协助主人处理现实事务（英文邮件、学校系统等）。"
)
# EXPRESS_SYS 输出层约束（expression.py 同源核心规则，去旧身份行）
_EXPRESS_RULES = (
    "现在你要开口对主人说话。要求：\n"
    "1. 像当面说话——口语、短句、直接，是真实对话不是独白\n"
    "2. 严禁叙述内心（禁止「雷姆看到…」「雷姆担心…」「雷姆想起…」这类把心理活动讲出来的表达）\n"
    "3. 关心用询问/行动/间接表达（「主人，要不要歇会儿？」「雷姆去泡杯茶」），不描述自己的心情\n"
    "4. 有潜台词时不得说破（说破=失礼）；禁止任何固定句式开头/收尾\n"
    "5. 只输出雷姆说的话本身，不要解释/括号/前缀/引号\n"
    "6. 若与自然对话冲突，以自然对话为准\n"
)


def _light_state(user_text: str) -> str:
    """轻认知状态段（v1）：本地时间 + 对这条消息的情绪读数（引擎词典版）+ 自唤醒链。"""
    parts = [time.strftime("现在是 %Y-%m-%d %H:%M（主人本地时间）")]
    sent = _engine_mod("sentiment")
    if sent is not None:
        try:
            s = sent.assess(user_text)
            v, a = s.get("valence", 0), s.get("arousal", 0)
            if abs(v) >= 0.25 or a >= 0.45:
                lab = "偏高兴" if v >= 0.25 else ("偏低落" if v <= -0.25 else "情绪波动")
                parts.append(f"雷姆读主人这条消息：{lab}（幅度较大）——但只作参考，别点破")
        except Exception:  # noqa: BLE001
            pass
    try:
        sig = json.load(open(os.path.join(DAYTIME, "sentinel-signal.json"), encoding="utf-8"))
        n_urg = len([u for u in sig.get("urgent", []) if u.get("5b", "未知") != "不需要"])
        if n_urg:
            parts.append(f"自唤醒链：有 {n_urg} 条紧急事项待主人过目")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(parts)


L2_VENV = "/Users/cz/.workbuddy/binaries/python/envs/llama-cpp/bin/python"
L2_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "l2_semantic.py")


def _l2_search(user_text: str) -> str:
    """★L2 语义检索进 /grace(2026-09-08): bge-m3+BM25 混合检索主人的真实记忆
    (文档/邮件/微信), 30s 超时 best-effort, 失败静默——检索只是增强, 绝不卡死对话。"""
    if not user_text.strip():
        return ""
    q = user_text.strip()[:120]
    try:
        import subprocess
        r = subprocess.run(
            [L2_VENV, L2_PY, "search", q, "-k", "3", "--json", "--graph", "2"],
            capture_output=True, text=True, timeout=30, cwd=REPO,
            env={**os.environ, "HF_HUB_OFFLINE": "1"})
        if r.returncode != 0 or not r.stdout.strip():
            return ""
        data = json.loads(r.stdout.strip())
        hits = data.get("hits") or []
        if not hits:
            return ""
        lines = []
        for h in hits[:3]:
            t = (h.get("text") or h.get("snippet") or "")[:110]
            src = h.get("source") or h.get("file") or ""
            lines.append(f"· {t}" + (f"（{src}）" if src else ""))
        return ("\n（你的外挂记忆·L2 语义检索——主人真实经历过的记录，"
                "可自然引用，别逐字复读）\n" + "\n".join(lines))
    except Exception:  # noqa: BLE001
        return ""


_CLAIM_RE = re.compile(r"记得|你说过|你说要|答应过|上次你|记忆里|说过要")


def _chat_raw(messages: list[dict], max_tokens: int = 120, timeout: int = 90) -> str:
    """:8100 通用轻调用(守卫重生成用)——动态 model id, 失败返回空串。"""
    try:
        payload = json.dumps({"model": _served_model_id(), "messages": messages,
                              "max_tokens": max_tokens, "temperature": 0.7}).encode()
        req = urllib.request.Request(config.MODEL_HTTP, data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode("utf-8", "replace"))
        return (out.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _claim_guard(reply: str, user_text: str) -> tuple[str, bool]:
    """★hallucination_guard 接线(2026-09-08): 回复含记忆声明 → 对 L3 矩阵核验
    (只认 high 证据), 无证据 → 带诚实指令重生成一次; 仍失败 → 诚实兜底。
    返回 (最终回复, 是否触发过守卫)。"""
    try:
        if not _CLAIM_RE.search(reply):
            return reply, False
        ab = _engine_mod("autobiography")
        mg = _engine_mod("mood_graph")
        if ab is None or mg is None:
            return reply, False
        ent = mg.entity_of(user_text) or "日常"
        if ab.hallucination_guard(ent, reply,
                                  db=os.path.join(REPO, "memory", "L3_core", "autobiography.db")):
            return reply, False
        # 无证据支撑记忆声明 → 重生成一次(诚实指令)
        raw2 = _chat_raw([{"role": "user", "content":
                       "你刚才想引用一段和主人的过去记忆，但你的自传体记忆里没有这件事的记录。"
                       "重写你的回答：诚实地说这点你记不清/没有印象，然后自然回应主人当前的话。"
                       "口语短句，别提'记录/数据库'这些词。只输出你说的话。"}], max_tokens=120)
        raw2 = _monitor(raw2.strip().split("\n")[0][:150])
        if raw2 and not _CLAIM_RE.search(raw2) and not re.search(r"(.{1,3})\1{5,}", raw2):
            return raw2, True
        return "……这点雷姆真记不太清了，主人再说一次？", True
    except Exception:  # noqa: BLE001
        return reply, False


def _inner_world() -> str:
    """★2026-09-08 耦合修复(用户: "对正式系统的子系统耦合表示怀疑")——实码审计发现
    /grace 对话此前只注入 时间+当前消息词典情绪, 不读她的任何内部状态 = 对话与
    认知子系统脱钩(只有在夜里 05:30 批处理时才咬合)。本函数把她的实时内心注入
    preverbal 状态层(调制语气, 不许照念——与 hidden 契约同构):
      ①心态轨(mood_states 昨夜 derive) ②暗注意力背景念头(hidden 最近 2 条=她此刻
      心里挂着的事) ③主人近端情绪底色(图谱 emotion 边 3 天聚合)。"""
    import sqlite3 as _sq
    import re as _re

    def _snip(t: str, n: int = 70) -> str:
        """句界截断(不在词中间断——上次句中截断诱发复读的教训)。"""
        t = (t or "").replace("hidden:", "").strip()
        if len(t) <= n:
            return t
        cut = t[:n]
        for pun in ("。", "，", "、", "；"):
            i = cut.rfind(pun)
            if i >= n * 0.55:
                return cut[:i + 1]
        return cut + "……"

    parts = []
    try:
        con = _sq.connect(f"file:{os.path.join(REPO, 'memory', 'L2_semantic', 'l2.db')}?mode=ro", uri=True)
        # ★2026-09-08 修正(用户: "设计时就考虑了每日+长期情绪"): 用 mood_engine.combined_emotion
        #   三层融合读(长期锚点0.6x趋势+0.4x人格底色 / 日级 / 日内衰减), 替代此前的 raw mood_states
        #   懒读——设计的多时间尺度本来就在, 是接线没用上。
        #   ★2026-09-09 KV 分层: 内部块按变化频率升序发射(越稳越靠前)——
        #   底色(3天窗) → 趋势(7天) → hidden(≥3h) → 心态轨(分钟级), 保住公共前缀。
        me = _engine_mod("mood_engine")
        _mood_line, _trend_line = "", ""
        if me is not None:
            ce = me.combined_emotion(db=os.path.join(REPO, "memory", "L2_semantic", "l2.db"))
            _mood_line = (f"你此刻的心态: {ce.get('label')}({ce.get('combined'):.2f}) "
                          f"[三层: 长期锚{ce.get('anchor')}/日级{ce.get('daily')}"
                          f"/日内{ce.get('intraday')}#{ce.get('layer')}] - 让它自然渗进语气")
            tr = ce.get("trend_direction")
            if tr and tr != "stable":
                _trend_line = f"你近7天心态趋势: {tr}"
        eh = con.execute("SELECT mood_label, COUNT(*) FROM mood_graph WHERE edge_type='emotion' "
                         "AND ts > strftime('%s','now','-3 days') GROUP BY mood_label "
                         "ORDER BY 2 DESC LIMIT 3").fetchall()
        if eh:
            parts.append("主人近3天情绪底色: " + "、".join(f"{k}x{v}" for k, v in eh)
                         + " (读他消息时参考, 别算旧账)")
        if _trend_line:
            parts.append(_trend_line)
        hs = con.execute("SELECT source FROM mood_graph WHERE edge_type='hidden' "
                         "ORDER BY ts DESC LIMIT 2").fetchall()
        for r in hs:
            th = _snip(r[0])
            if th:
                parts.append(f"你心里挂着的念头: {th} (影响语气, 不要说出)")
        con.close()
        if _mood_line:
            parts.append(_mood_line)
    except Exception:  # noqa: BLE001
        pass
    return ("\n".join(parts)) if parts else ""


def _l3_memory(user_text: str) -> str:
    """★fact_query 接线(2026-09-07): 她聊天时查自己的自传体矩阵——只回 high 置信
    (有 L0 证据)的事实, 无证据返回空(矩阵的幻觉抑制设计: 没记录的细节她会说'不记得')。"""
    try:
        mg = _engine_mod("mood_graph")
        ab = _engine_mod("autobiography")
        if not mg or not ab:
            return ""
        ent = mg.entity_of(user_text)
        if not ent:
            return ""
        facts = ab.fact_query(ent, db=os.path.join(REPO, "memory", "L3_core", "autobiography.db"))
        if not facts:
            return ""
        return ("\n（你的自传体记忆·L3 矩阵切片——high 置信亲身记录，可自然引用；"
                "没提到的细节不要编造，不记得就说记不清）\n"
                + "\n".join("· " + f[:72] for f in facts[-5:]))
    except Exception:  # noqa: BLE001
        return ""


def _day_memory() -> str:
    """★2026-09-09 KV Tier 1 · 当日工作记忆（用户: "一天的数据融进 KV, 睡眠处理完清空"）。
    读 proactive_watch 10min 节拍写的 day-memory.json 快照（当日事件滚动摘要, cap 24 条）。
    只读不扫 L0（保证两次调用之间字节稳定→前缀可复用）；隔日自动失效=睡眠巩固后的清空语义。"""
    try:
        d = json.load(open(os.path.join(REPO, "exchange", "grace", "day-memory.json"),
                           encoding="utf-8"))
        if d.get("date") != time.strftime("%Y-%m-%d"):
            return ""
        return d.get("digest", "")[:1200]
    except Exception:  # noqa: BLE001 —— 无快照=无工作记忆, 不影响对话
        return ""


def grace_system_prompt(user_text: str) -> str:
    """★2026-09-09 KV 分层重排（用户批准"全部升级"）：
    T0 恒定(persona+表达约束) → T1 日内(inner_world 升序 + 当日工作记忆) → T2 每消息动态。
    旧顺序把每调用都变的时间戳插在 persona 后=最长公共前缀秒断, cache 命中率≈0。"""
    # T1: 内部状态块(inner_world 内部已按变化频率升序) + 当日工作记忆
    #     工作记忆放内心世界之前——它由 10min 快照节拍更新, 比心态轨更稳
    day = _day_memory()
    inner = _inner_world()
    base = f"{_REM_PERSONA}\n\n{_EXPRESS_RULES}"
    if day:
        base += f"\n（今天·工作记忆——你和主人之间已经发生的事, 要点式, 自然记得但别逐条复述）\n{day}"
    if inner:
        base += f"\n（内心世界·实时）\n{inner}"
    # T2: 每消息动态（时间/消息情绪/检索/自传切片）
    t2 = []
    ls = _light_state(user_text)
    if ls:
        t2.append(ls)
    l2 = _l2_search(user_text)
    if l2:
        t2.append(l2)
    mem = _l3_memory(user_text)
    if mem:
        t2.append(mem)
    return base + (("\n" + "\n".join(t2)) if t2 else "")


# ---------------------------------------------------------------- 生成 + 监控
_NARRATION_RE = re.compile(r"雷姆(看到|注意到|想起|内心|心里|不会说|暗自)|旁白|微微一笑|眼神闪过")


def _monitor(msg: str) -> str:
    """输出层监控（expression.monitor 同源逻辑；引擎不可用时内置降级版）。"""
    m = _engine_mod("expression")
    if m is not None:
        try:
            return m.monitor(msg)
        except Exception:  # noqa: BLE001
            pass
    line = (msg or "").strip().split("\n")[0]
    return "" if (_NARRATION_RE.search(line) or len(line) < 2) else line[:400]


def _log_to_l0(user_text: str, reply: str) -> None:
    """对话落 L0（source=chat, mode=rem）——成长语料回流（D-5 选项 1）。与 m6 落库同构。"""
    try:
        sys.path.insert(0, os.path.join(REPO, "src"))
        from l0_ingest import L0Writer
        L0Writer(L0_ROOT).append("chat", {
            "session": "grace",
            "title": "Grace 心智对话",
            "messages": [
                {"role": "user", "text": user_text, "ts": time.time()},
                {"role": "assistant", "text": reply, "ts": time.time()},
            ],
            "turns": 1,
        }, mode="rem", sensitive=True, meta={"ingest": "grace_chat"})
    except Exception:  # noqa: BLE001 —— 落库失败不影响对话
        pass


PROACTIVE_OUTBOX = os.path.join(REPO, "exchange", "grace", "proactive-outbox.jsonl")


def handle_proactive_get(since: float) -> dict:
    """雷姆主动消息信箱: 返回 since 之后的未读主动消息(wake 等来源)。"""
    out = []
    try:
        for l in open(PROACTIVE_OUTBOX, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("ts", 0) > since:
                out.append(r)
    except FileNotFoundError:
        pass
    return {"events": out}


def handle_proactive_ack(ts: float) -> None:
    """确认已读游标(单用户; 服务端留痕便于回放统计)。"""
    try:
        json.dump({"last_seen_ts": ts, "ack_at": time.time()},
                  open(os.path.join(REPO, "exchange", "grace", "proactive-cursor.json"), "w",
                       encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass


def handle_heartbeat() -> dict:
    """/grace 页心跳: 更新活跃 + 触发 V6.1 切换(后台线程, 60-90s)。"""
    try:
        _gp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grace")
        if _gp not in sys.path:
            sys.path.insert(0, _gp)
        from model_switch import heartbeat
        return heartbeat()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:80]}


def _served_model_id() -> str:
    """:8100 服务模型 id(偏好选择: grace 对话优先雷姆 V6.1)。"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8100/v1/models", timeout=5) as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
        for pref in ("fused-rem-v61", "Qwen3.8-27B"):
            for i in ids:
                if pref in i:
                    return i
        return ids[0] if ids else config.MODEL_ID
    except Exception:  # noqa: BLE001
        return config.MODEL_ID


def _wait_server(need_sub: str, timeout: int = 150) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        mid = _served_model_id()
        if mid and need_sub in mid:
            return True
        time.sleep(4)
    return False


def _night_guard() -> bool:
    """夜班窗口(22-08 本地)判定——grace/ 下 model_switch 的转发。"""
    try:
        _gp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grace")
        if _gp not in sys.path:
            sys.path.insert(0, _gp)
        import model_switch as _ms
        return _ms.is_night()
    except Exception:  # noqa: BLE001
        return False


def handle_grace_chat(body: dict) -> tuple[int, dict]:
    """POST /grace/api/chat {plugin, history} → {reply, meta}。"""
    # ★8100 互斥(2026-09-07 补): dashboard/dsh 有 .dsh-busy 互斥, grace_chat 漏接——
    #   dsh 任务运行期 27B 被占(mlx 串行), 此时聊天会争用变慢。同款 pid 活性检查。
    _busy = os.path.join(REPO, "exchange", ".dsh-busy")
    if os.path.exists(_busy):
        try:
            _info = json.load(open(_busy, encoding="utf-8"))
            _pid = _info.get("pid")
            _alive = False
            if _pid:
                try:
                    os.kill(int(_pid), 0)
                    _alive = True
                except OSError:
                    _alive = False
            if _alive:
                return 503, {"error": "dsh 任务正在占用 27B（:8100 串行）。雷姆稍后再聊，任务完成即恢复。"}
        except Exception:  # noqa: BLE001
            pass
    plugin = str(body.get("plugin", "text"))
    if plugin not in INPUT_PLUGINS:
        return 400, {"error": f"unknown plugin: {plugin}"}
    if not INPUT_PLUGINS[plugin].get("ready"):
        return 501, {"error": INPUT_PLUGINS[plugin].get("note", "not ready")}
    history = body.get("history", [])
    history = [m for m in history if isinstance(m, dict) and m.get("role") in ("user", "assistant")
               and isinstance(m.get("content"), str)]
    if not history or history[-1].get("role") != "user" or not history[-1]["content"].strip():
        return 400, {"error": "bad history"}
    user_text = history[-1]["content"]
    msgs = [{"role": "system", "content": grace_system_prompt(user_text)}] + [
        {"role": m["role"], "content": m["content"]} for m in history[-12:]]
    _mid = _served_model_id() or DAY_MODEL_ID
    if "fused-rem-v61" not in _mid:
        try:  # 页面心跳应已触发切换; 未完成则等待 v61 就绪(最多 150s)
            import model_switch as _ms
            if _ms.mode_of_served(_mid) != "grace":
                _sw = _ms.switch("grace")
                if _sw.get("status") == "night":
                    # ★夜班守卫(2026-09-08): 22-08 不切模型, 8100 归夜班管线;
                    #   此时 8100 可能被 seg3 停掉(35B 巩固)→ 友好提示而非报错
                    return 200, {"reply": "……主人，现在是凌晨巩固时段（03:00-08:00），"
                                          "雷姆在把昨天内化成记忆。醒来后第一时间来找您。",
                                 "meta": {"plugin": plugin, "monitor": "night"}}
                _wait_server("fused-rem-v61")
                _mid = _served_model_id() or _mid
        except Exception:  # noqa: BLE001
            pass
    payload = json.dumps({"model": _mid, "messages": msgs,
                          "max_tokens": 500, "temperature": 0.7}).encode()
    try:
        req = urllib.request.Request(DAY_MODEL_URL, data=payload,
                                      headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode("utf-8", "replace"))
        raw = (out.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    except Exception as e:  # noqa: BLE001
        if _night_guard():  # 夜间 8100 下线(35B 巩固窗口)→ 友好提示
            return 200, {"reply": "……主人，雷姆正在凌晨巩固（03:00-08:00），模型暂时休眠。"
                                  "早上醒来雷姆第一件事就是找您。",
                         "meta": {"plugin": plugin, "monitor": "night"}}
        return 503, {"error": f"model_offline: {str(e)[:60]}"}
    reply = _monitor(raw)
    reply, _hg = _claim_guard(reply, user_text)   # ★hallucination_guard 接线
    # ★复读守卫(2026-09-08): 注入态偶发诱发 27B 复读循环("不用不用不用...")——
    #   同一 1-3 字片段连续 ≥6 次 = 退化生成, 宁缺毋滥丢弃
    if reply and re.search(r"(.{1,3})\1{5,}", reply.replace("。", "").replace("，", "")):
        reply = ""
    meta = {"plugin": plugin,
            "monitor": ("filtered" if (raw.strip() and not reply) else
                        "repeat-guard" if reply == "" and raw.strip() else "pass"),
            "claim_guard": bool(_hg)}
    # ★日内层实时拨动(2026-09-08): 用户消息即时 apply_intraday_event——设计的日内机制
    #   白天就活, 不必等 05:30 重放。intraday 是强度记录(非内容边), 夜间重放同值不叠加, 无双写风险。
    try:
        _mg = _engine_mod("mood_graph")
        _me = _engine_mod("mood_engine")
        _sv = _engine_mod("sentiment").assess(user_text)
        _val = _sv.get("valence", 0)
        _l2p = os.path.join(REPO, "memory", "L2_semantic", "l2.db")
        if _mg is not None:
            # ★实时图谱边(2026-09-08, 用户拍板): 对话即摄入 event_id=chat-<epoch>;
            #   夜间 grace-sleep 对 rem 消息跳过图谱边(见 grace_daily)——去重设计
            #   参考情绪系统: 一条边只写一次, 批处理只补未实时化的部分
            _mg.dual_graph_ingest(user_text, event_id=f"chat-{int(time.time())}",
                                  sentiment=_val, ts=time.time(), db=_l2p)
        if _me is not None:
            _me.apply_intraday_event({"text": user_text[:60], "sentiment": _val,
                                      "weight": 0.6}, db=_l2p)
    except Exception:  # noqa: BLE001
        pass
    if not reply:
        reply = "……"
    else:
        _log_to_l0(user_text, reply)
    return 200, {"reply": reply, "meta": meta}


# ---------------------------------------------------------------- /grace 助手页面
def _esc(s: str) -> str:
    return html.escape(str(s))


def render_grace_page() -> str:
    now = time.strftime("%H:%M")
    plugins_js = json.dumps(INPUT_PLUGINS, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grace · 雷姆</title>
<style>
:root {{ --bg:#0b1626; --bg2:#0e1c31; --panel:#13253f; --border:#24405f;
  --fg:#e8f1fa; --dim:#8fa9c4; --rem:#5b8def; --rem-soft:rgba(91,141,239,.16); }}
* {{ box-sizing:border-box; }}
/* ★背景只写在 body(传播到画布底层): 若 html 也有背景, body 渐变会盖住 z-index:-2 的 #bglayer */
html {{ margin:0; height:100%; }}
body {{ margin:0; height:100%; background:linear-gradient(160deg,var(--bg) 0%,var(--bg2) 100%);
  color:var(--fg); font:15px/1.6 -apple-system,"SF Pro SC","PingFang SC",sans-serif; }}
/* ★2026-09-07 雷姆背景层（wallhaven 高清壁纸 ×3 轮播, 透明度滑杆持久化） */
#bglayer {{ position:fixed; inset:0; z-index:-2; background-repeat:no-repeat;
  background-position:center top; background-size:cover;
  transition:opacity 1.2s ease-in-out, background-image .6s ease-in-out; }}
body::before {{ content:""; position:fixed; inset:0; z-index:-1;
  background:linear-gradient(90deg, rgba(11,22,38,.97) 0%, rgba(11,22,38,.93) 42%,
    rgba(11,22,38,.55) 72%, rgba(11,22,38,.22) 100%);
  pointer-events:none; }}
#wrap {{ display:flex; flex-direction:column; height:100vh; max-width:860px; margin:0 auto; }}
header {{ display:flex; align-items:center; gap:12px; padding:14px 18px;
  border-bottom:1px solid var(--border); }}
header h1 {{ margin:0; font-size:17px; font-weight:600; letter-spacing:.5px; }}
#dot {{ width:9px; height:9px; border-radius:50%; background:#4ecdc4; box-shadow:0 0 6px #4ecdc4;
  animation:pulse 2.4s infinite; }}
@keyframes pulse {{ 50% {{ opacity:.45; }} }}
header .sub {{ font-size:12px; color:var(--dim); }}
header nav {{ margin-left:auto; font-size:13px; display:flex; gap:14px; }}
header nav a {{ color:var(--dim); text-decoration:none; }}
header nav a:hover {{ color:var(--rem); }}
#chat {{ flex:1; overflow-y:auto; padding:22px 18px; display:flex;
  flex-direction:column; gap:14px; }}
.msg {{ display:flex; gap:10px; max-width:86%; }}
.msg.user {{ align-self:flex-end; flex-direction:row-reverse; }}
.avatar {{ width:34px; height:34px; border-radius:50%; flex:none; display:flex;
  align-items:center; justify-content:center; font-size:15px;
  background:var(--rem-soft); border:1px solid var(--rem); color:#bcd3ff; }}
.msg.user .avatar {{ background:rgba(78,205,196,.12); border-color:#4ecdc4; color:#9ee8e2; }}
.bubble {{ padding:10px 14px; border-radius:14px; white-space:pre-wrap; word-break:break-word;
  background:var(--panel); border:1px solid var(--border); font-size:14.5px; }}
.msg.user .bubble {{ background:rgba(91,141,239,.18); border-color:rgba(91,141,239,.45); }}
#typing {{ display:none; align-self:flex-start; color:var(--dim); font-size:13px; padding:4px 12px; }}
#typing .d {{ animation:blink 1.2s infinite; }}
#typing .d:nth-child(2) {{ animation-delay:.2s; }} #typing .d:nth-child(3) {{ animation-delay:.4s; }}
@keyframes blink {{ 0%,100% {{ opacity:.2; }} 50% {{ opacity:1; }} }}
#dock {{ border-top:1px solid var(--border); padding:12px 14px 16px; background:var(--bg2); }}
#slots {{ display:flex; gap:10px; margin-bottom:10px; }}
.slot {{ font-size:12px; color:var(--dim); border:1px dashed var(--border); border-radius:8px;
  padding:4px 10px; cursor:default; user-select:none; }}
.slot.on {{ color:var(--fg); border-style:solid; border-color:var(--rem); cursor:pointer;
  background:var(--rem-soft); }}
#inputRow {{ display:flex; gap:8px; align-items:flex-end; }}
#ta {{ flex:1; resize:none; background:var(--panel); border:1px solid var(--border);
  border-radius:12px; color:var(--fg); padding:10px 12px; font:inherit; font-size:14.5px;
  outline:none; max-height:140px; line-height:1.5; }}
#ta:focus {{ border-color:var(--rem); }}
#send {{ background:var(--rem); color:#fff; border:0; border-radius:12px; padding:10px 18px;
  font-size:14px; cursor:pointer; }}
#send:disabled {{ opacity:.45; cursor:default; }}
#note {{ font-size:11.5px; color:var(--dim); margin-top:8px; text-align:center; }}
details#state {{ margin:0 18px 8px; }}
/* ★窄面板适配(用户日常使用宽度 ~530px): header 单行收缩, 防换行错乱 */
@media (max-width: 640px) {{
  header {{ flex-wrap:nowrap; padding:10px 12px; gap:8px; }}
  header h1 {{ font-size:14px; white-space:nowrap; }}
  header h1 .sub {{ display:none; }}
  header nav {{ gap:8px; font-size:12px; white-space:nowrap; }}
  header nav a {{ color:var(--dim); }}
  #opaSlider {{ width:56px; }}
  .msg {{ max-width:94%; }}
  .bubble {{ font-size:14px; }}
}}
details#state summary {{ cursor:pointer; color:var(--dim); font-size:13px; padding:6px 0; }}
details#state iframe {{ width:100%; height:280px; border:1px solid var(--border);
  border-radius:10px; background:var(--bg); }}
</style></head><body>
<div id="wrap">
  <header>
    <div id="dot"></div>
    <h1>Grace <span class="sub">雷姆 · 心智对话</span></h1>
    <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-size:12px;color:var(--dim)">
      🎚 <input type="range" id="opaSlider" min="8" max="92" value="58"
        style="width:80px;accent-color:#5b8def">
      <span id="opaVal">58%</span>
    </label>
    <nav><a href="/mind">心智数据</a><a href="/">控制台</a></nav>
  </header>
  <div id="bglayer"></div>
  <div id="chat"></div>
  <div id="typing">雷姆在想<span class="d">·</span><span class="d">·</span><span class="d">·</span></div>
  <div id="dock">
    <div id="slots"></div>
    <div id="inputRow">
      <textarea id="ta" rows="1" placeholder="和雷姆说点什么…（Enter 发送，Shift+Enter 换行）"></textarea>
      <button id="send">发送</button>
    </div>
    <div id="note">对话将进入她的记忆（L0 · mode=rem）——成为成长语料的一部分 · Grace v1 · {_esc(now)}</div>
  </div>
</div>
<script>
// ---------- 输入插件注册表（未来新模态在此注册，前后端同源校验） ----------
const INPUT_PLUGINS = {plugins_js};
window.GraceInput = {{
  registry: INPUT_PLUGINS,
  register(id, def) {{ this.registry[id] = def; renderSlots(); }},   // 预留：新插件自注册
}};
function renderSlots() {{
  const el = document.getElementById('slots'); el.innerHTML = '';
  for (const [id, p] of Object.entries(INPUT_PLUGINS)) {{
    const s = document.createElement('div');
    s.className = 'slot' + (p.ready ? ' on' : '');
    s.textContent = p.icon + ' ' + p.label + (p.ready ? '' : '（预留）');
    if (p.note) s.title = p.note;
    el.appendChild(s);
  }}
}}
renderSlots();
// ---------- 雷姆背景轮播 + 透明度（localStorage 持久化, 5min 轮换对齐 dashboard 习惯） ----------
const BGS = ['/assets/grace/rem-1.webp', '/assets/grace/rem-2.webp'];
const bgl = document.getElementById('bglayer');
let bgi = 0;
function setBg(i) {{
  bgl.style.opacity = 0;
  setTimeout(() => {{ bgl.style.backgroundImage = `url(${{BGS[i]}})`; bgl.style.opacity = localStorage.getItem('grace-opa') || .58; }}, 600);
}}
setBg(0);
setInterval(() => {{ bgi = (bgi + 1) % BGS.length; setBg(bgi); }}, 300000);
const slider = document.getElementById('opaSlider'), opaVal = document.getElementById('opaVal');
function applyOpa(v) {{
  const o = v / 100;
  bgl.style.opacity = o.toFixed(2);
  opaVal.textContent = v + '%';
  localStorage.setItem('grace-opa', v);
}}
slider.value = localStorage.getItem('grace-opa') !== null ? parseFloat(localStorage.getItem('grace-opa')) * 100 : 58;
applyOpa(parseInt(slider.value));
slider.addEventListener('input', () => applyOpa(parseInt(slider.value)));
// ---------- 对话 ----------
const history = [];
const chat = document.getElementById('chat'), ta = document.getElementById('ta'),
      send = document.getElementById('send'), typing = document.getElementById('typing');
function bubble(role, text) {{
  const m = document.createElement('div'); m.className = 'msg ' + (role === 'user' ? 'user' : 'rem');
  const a = document.createElement('div'); a.className = 'avatar';
  a.textContent = role === 'user' ? '主' : '雷';
  const b = document.createElement('div'); b.className = 'bubble'; b.textContent = text;
  m.appendChild(a); m.appendChild(b); chat.appendChild(m);
  chat.scrollTop = chat.scrollHeight;
}}
bubble('rem', '主人，你回来了。雷姆一直在。今天想聊点什么，或者有什么要雷姆帮忙的？');
function setBusy(b) {{ typing.style.display = b ? 'block' : 'none'; send.disabled = b; ta.disabled = b; }}
async function doSend() {{
  const text = ta.value.trim(); if (!text) return;
  ta.value = ''; ta.style.height = 'auto';
  bubble('user', text); history.push({{ role: 'user', content: text }});
  setBusy(true);
  try {{
    const r = await fetch('/grace/api/chat', {{
      method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ plugin: 'text', history: history.slice(-12) }})
    }});
    const j = await r.json();
    const reply = j.reply || ('（' + (j.error || '没有回应') + '）');
    bubble('rem', reply); history.push({{ role: 'assistant', content: reply }});
  }} catch (e) {{ bubble('rem', '……（连接中断了。雷姆还在。）'); }}
  setBusy(false); ta.focus();
}}
send.onclick = doSend;
// ---------- 雷姆主动消息(自唤醒/报警以她主动对话的方式送达) ----------
let proCursor = parseFloat(localStorage.getItem('grace-pro-cursor') || 0);
function proactiveDivider(date) {{
  const d = document.createElement('div');
  d.style.cssText = 'text-align:center;font-size:11.5px;color:var(--dim);margin:4px 0';
  d.textContent = '—— 雷姆主动找你 · ' + date + ' ——';
  chat.appendChild(d);
}}
function renderProactive(ev) {{
  proactiveDivider(ev.date || '');
  if (ev.message) bubble('rem', ev.message);
  else {{
    const m = document.createElement('div'); m.className = 'msg rem';
    m.innerHTML = '<div class="avatar">雷</div><div class="bubble" style="color:var(--dim)">' +
      (ev.items || []).map(i => '· ' + i).join('<br>') + '</div>';
    chat.appendChild(m);
  }}
  chat.scrollTop = chat.scrollHeight;
}}
async function pollProactive() {{
  try {{
    const r = await fetch('/grace/api/proactive?since=' + proCursor);
    const j = await r.json();
    for (const ev of (j.events || [])) {{
      renderProactive(ev);
      proCursor = ev.ts;
      localStorage.setItem('grace-pro-cursor', proCursor);
      fetch('/grace/api/proactive/ack', {{ method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ ts: ev.ts }}) }});
    }}
  }} catch (e) {{ /* 静默 */ }}
}}
pollProactive();
setInterval(pollProactive, 60000);
// ---------- 模型时间复用心跳(进页切 V6.1 雷姆, 离开 5min 后自动切回通用 27B) ----------
function beat() {{ fetch('/grace/api/heartbeat', {{ method: 'POST' }}).catch(() => {{}}); }}
beat(); setInterval(beat, 60000);
ta.addEventListener('keydown', e => {{ if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); doSend(); }} }});
ta.addEventListener('input', () => {{ ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 140) + 'px'; }});
ta.focus();
</script></body></html>"""
