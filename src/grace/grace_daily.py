#!/usr/bin/env python3
"""grace_daily.py —— Grace 引擎正式运行·每日认知编码（集成方案 Phase 2 / 夜班 segment5 本体）。

迁移自压测沙盒 stress_engine 每日循环（2026-09-07），按写权限矩阵裁剪：
  ✅ 写: mood_states/mood_intraday(心态轨) + mood_graph(双图谱 emotion/hidden 边) +
         L3 autobiography.db(自传节点) + exchange/grace/run/stress/*.jsonl(运行账本)
  ❌ 不写: L0(只读输入) / 训练权重(走人审提案, 集成方案已拍板) / L2 语义索引(docs/entities 不碰)

管线（与压测每日循环同构）:
  今天 L0(本地日期) → 逐条 sentiment+attention → P0 门控(salience≥0.6 或 |sent|≥0.3)
  → ② mood derive(日级)+intraday(日内) → ③ dual_graph_ingest(双图谱, ev-<date>-<i>)+theta 峰值
  → ⑤ 门控消息: ToM 判断(:8100 HTTP 模型驱动, D4) → PE 账本(believed vs real)
  → ★Phase 0: 模糊带情绪(0.2≤|v|<0.6) judge 重标 real(:8100, O5 前置)
  → ★cog 认知重构(每日一次:8100) → add_hidden_text(暗注意力归档)
  → ④ L3 add_event(仅门控消息; O14 emotion 现算 + O3 event_id 缝合)
用法:
  python3 src/grace_daily.py [--date YYYY-MM-DD] [--max-tom 60]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime

GRACE = os.path.dirname(os.path.abspath(__file__))          # src/grace
sys.path.insert(0, GRACE)                                    # import config / engine.X
sys.path.insert(0, os.path.join(GRACE, "engine"))           # 裸 from mood_graph import ...
import config  # noqa: E402

DAY_KEY = None          # 运行日期(YYYY-MM-DD), run() 设置
L0 = config.L0_DIR
LEDGERS = os.path.join(config.EXPERIMENTS, "run", "stress")

# ---------------------------------------------------------------- L0 当日提取（迁移自 build_l0_book）
_SKIP = re.compile(
    r"(转账|CDATA|微信转账|发了一个红包|\[红包\]|\[转账\]|^https?://|^\d{4,}$|^[a-f0-9]{16,}$|^\.$|^。$|^\s*$)")
_SOURCES = ["wechat.jsonl", "chat.jsonl", "email.jsonl", "exchange:inbox.jsonl",
            "exchange:school.jsonl", "school.jsonl", "doc:file.jsonl"]


def _clean(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _split_summary(text: str) -> list[str]:
    """邮箱/学校摘要 markdown → 事件条目（迁移自 build_l0_book v2 同款逻辑）。"""
    out, cur_title, cur_body, in_code = [], "", [], False
    mail_re = re.compile(r"^\s*\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*(.*)$")

    def flush():
        nonlocal cur_title, cur_body
        if cur_title or cur_body:
            txt = (cur_title + "：" + " ".join(cur_body)[:120]) if cur_title else " ".join(cur_body)[:120]
            txt = txt.strip("：: ")
            if len(txt) >= 6:
                out.append(txt)
        cur_title, cur_body = "", []

    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        m = mail_re.match(ln)
        if m:
            flush()
            cur_title = m.group(2).strip()[:60] or "邮件"
        elif ln.strip().startswith("#") and not in_code:
            flush()
            cur_title = ln.strip().lstrip("# ").strip()[:60]
        elif ln.strip():
            cur_body.append(ln.strip())
    flush()
    return out


def today_messages(date_str: str) -> list[dict]:
    """读正式 L0 当日(本地日期)全部源 → [{text, sentiment, weight, sender, is_send, ts}]。"""
    from sentiment import assess
    items = []
    for fn in _SOURCES:
        fp = os.path.join(L0, fn)
        if not os.path.isfile(fp):
            continue
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("payload", {}) if isinstance(r, dict) else {}
            msgs = p.get("messages") if isinstance(p, dict) else None
            if msgs:                                   # 消息级: wechat/chat(含 grace 对话)
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    ts = m.get("ts") or r.get("epoch")
                    if not isinstance(ts, (int, float)):
                        continue
                    if datetime.fromtimestamp(ts).strftime("%Y-%m-%d") != date_str:
                        continue
                    t = _clean(m.get("text", ""))
                    if not t or _SKIP.search(t):
                        continue
                    items.append({"text": t[:140], "sentiment": assess(t)["valence"],
                                  "weight": 0.6, "sender": m.get("sender"),
                                  "is_send": m.get("is_send"), "role": m.get("role"),
                                  "rem": r.get("mode") == "rem", "ts": ts})
            else:                                      # 摘要级: email/exchange/school/doc
                t = _clean(p.get("text", "")) if isinstance(p, dict) else ""
                if not t:
                    continue
                ep = r.get("epoch")
                if not isinstance(ep, (int, float)):
                    continue
                if datetime.fromtimestamp(ep).strftime("%Y-%m-%d") != date_str:
                    continue
                for chunk in _split_summary(t):
                    if chunk and not _SKIP.search(chunk):
                        items.append({"text": chunk[:140], "sentiment": assess(chunk)["valence"],
                                      "weight": 0.6, "sender": None, "is_send": None, "role": None, "ts": ep})
    items.sort(key=lambda x: x["ts"])
    return items


# ---------------------------------------------------------------- :8100 HTTP 推理（D-4 选项 A）
def _chat(messages: list[dict], max_tokens: int = 200, temperature: float = 0.3) -> str:
    try:  # ★模型时间复用: 8100 可能正服务雷姆 V6.1(用户在 /grace)——偏好选择
        with urllib.request.urlopen("http://127.0.0.1:8100/v1/models", timeout=5) as _r:
            _ids = [m["id"] for m in json.loads(_r.read())["data"]]
        _mid = next((i for pref in ("Qwen3.8-27B", "fused-rem-v61")
                     for i in _ids if pref in i), _ids[0] if _ids else config.MODEL_ID)
    except Exception:  # noqa: BLE001
        _mid = config.MODEL_ID
    payload = json.dumps({"model": _mid, "messages": messages,
                          "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(config.MODEL_HTTP, data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.loads(resp.read().decode("utf-8", "replace"))
    return (out.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


def tom_judge(text: str, owner_mood: str, hidden_ctx: list[str] | None,
              self_mood: str, relation: float) -> dict:
    """模型驱动 ToM（infer_owner_state_model 同构, 经 8100 HTTP）。"""
    from theory_of_mind import _normalize_emotion
    hid = ""
    if hidden_ctx:
        hid = (f"\n（雷姆自己此刻没说出口的潜台词：{'｜'.join(hidden_ctx[:2])[:120]}"
               "\n——注意：这是你自己的内心活动，不要把你对主人的关心当成主人本人的焦虑。）")
    prompt = (
        f"（情境）你看到这样一条信息：{text[:80]}\n"
        f"（参考：主人今天的情绪被记为{owner_mood}——但请以信息本身为准）\n"
        f"（你自己此刻的状态：{self_mood}；与主人的亲密度：{relation:.1f}/1.0。"
        "先从自己的经历和感受出发模拟主人，但最终判断以主人本人信号为准。）{hid}\n"
        "雷姆，请判断四件事：\n1. 主人此刻真实的心情？（六选一：平静/轻微兴奋/兴奋/低落/焦虑/专注）\n"
        "2. 主人此刻需要什么？（一句话）\n"
        "3. 值得主动去找主人说吗？（值得/不值得）\n"
        "4. 如果值得，一句台词（不描写动作）\n格式：心情：…；需要：…；值得：…；台词：…")
    try:
        raw = _chat([{"role": "user", "content": prompt}], max_tokens=180)
    except Exception as e:  # noqa: BLE001 —— 8100 不可达 → 规则版兜底
        from theory_of_mind import infer_owner_state
        r = infer_owner_state(text, owner_mood, hidden_ctx=hidden_ctx)
        r["_by"] = "rule(fallback)"
        r["_err"] = str(e)[:60]
        return r
    def _pick(k, d):
        m = re.search(k + r"[：:]\s*([^；;\n]{1,40})", raw)
        return m.group(1).strip() if m else d
    emotion = _normalize_emotion(_pick("心情", ""))
    need = _pick("需要", "顺其自然")[:30]
    worth = _pick("值得", "值得")
    msg = _pick("台词", "")[:80]
    return {"emotion": emotion, "need": need, "event_type": "model",
            "interruptible": not any(w in worth for w in ("不值得", "不必", "不打扰")),
            "advice": need, "message": msg, "confidence": None, "_by": "model", "raw": raw[:100]}


def judge_sentiment(text: str) -> str:
    """★Phase 0: 模糊带情绪 judge 重标（O5 前置——词典 real 尺过敏感 81% 已实锤）。"""
    from theory_of_mind import _normalize_emotion
    try:
        raw = _chat([{"role": "user", "content":
                     f"判断这条消息反映的主人此刻情绪，六选一（平静/轻微兴奋/兴奋/低落/焦虑/专注），只回一个词：\n{text[:80]}"}],
                    max_tokens=12, temperature=0.1)
        return _normalize_emotion(raw.strip())
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------- 主流程
def run(date_str: str | None = None, max_tom: int = 60) -> dict:
    global DAY_KEY
    DAY_KEY = date_str or datetime.now().strftime("%Y-%m-%d")
    t0 = time.time()
    # ★幂等守卫(9/7 首日发现): 同日重跑会重复写边——当日已有该日期前缀的图谱边则跳过
    import sqlite3 as _sq0
    try:
        _c0 = _sq0.connect(config.L2_DB)
        _n0 = _c0.execute("SELECT COUNT(*) FROM mood_graph WHERE event_id LIKE ?",
                          (f"ev-{DAY_KEY}-%",)).fetchone()[0]
        _c0.close()
    except Exception:  # noqa: BLE001
        _n0 = 0
    if _n0:
        return {"date": DAY_KEY, "status": "already_done", "detail": f"当日已摄入({_n0} 边), 跳过防重复"}
    msgs = today_messages(DAY_KEY)
    if not msgs:
        return {"date": DAY_KEY, "status": "no_data", "detail": "当日 L0 无可摄入消息"}

    from attention import generate_attention
    from mood_engine import derive, apply_intraday_event, combined_emotion
    from mood_graph import dual_graph_ingest, add_hidden_text, mood_label_of, query_hidden
    from autobiography import add_event

    # ① P0 门控
    sel, atts = [], []
    for m in msgs:
        a = generate_attention(m["text"], mood=None, facts=[])
        atts.append(a)
        sel.append(a["salience"] >= 0.6 or abs(m["sentiment"]) >= 0.3)
    # ② 心态轨（日级 + 日内）
    evs = [{"text": m["text"], "sentiment": m["sentiment"], "weight": m["weight"]} for m in msgs]
    derive(evs, ts=datetime.strptime(DAY_KEY, "%Y-%m-%d").timestamp() + 21 * 3600)
    for i, m in enumerate(msgs):
        apply_intraday_event(evs[i], ts=m["ts"])
    # ③ 双图谱 + theta 峰值
    # ★去重(2026-09-08, 参考"情绪系统"设计: 一条边只写一次): /grace 对话消息的图谱边
    #   已在对话时实时写入(event_id=chat-<epoch>), 夜间重放对 rem 消息跳过图谱边——
    #   L3/ToM/PE/cog 照常(那些不实时写)。
    for i, m in enumerate(msgs):
        if m.get("rem"):
            continue
        dual_graph_ingest(m["text"], event_id=f"ev-{DAY_KEY}-{i}",
                          sentiment=m["sentiment"], ts=m["ts"])
    try:
        import sqlite3 as _sq
        _pk = max(range(len(msgs)), key=lambda i: abs(msgs[i]["sentiment"]))
        _c = _sq.connect(config.L2_DB)
        _c.execute("UPDATE mood_graph SET source=source||'（theta:峰值同步）' "
                   "WHERE event_id=? AND edge_type IN ('emotion','hidden')", (f"ev-{DAY_KEY}-{_pk}",))
        _c.commit()
        _c.close()
    except Exception:  # noqa: BLE001
        pass

    # ⑤ 门控消息：ToM 模型判断 + PE 账本（★Phase 0: 模糊带 real 用 judge 重标）
    self_mood = (combined_emotion() or {}).get("label", "平静")
    relation = min(1.0, 0.56)          # 正式侧关系轨 v1: 人格底色(9/3 校准); 关系动态 O12 未立项
    pe_rows, judged_n = [], 0
    os.makedirs(LEDGERS, exist_ok=True)
    pe_path = os.path.join(LEDGERS, "prediction-errors.jsonl")
    # ★语义修正(9/7 首跑发现): chat 源里她自己的回复(role=assistant)不是主人输入——
    #   ToM/L3 只对主人侧消息(assistant 消息仍进图谱/心态, 作为"她当天的输出痕迹")
    sel_idx = [i for i, s in enumerate(sel)
               if s and msgs[i].get("role") != "assistant"][:max_tom]
    for i in sel_idx:
        m = msgs[i]
        tom = tom_judge(m["text"], mood_label_of(m["sentiment"], abs(m["sentiment"])),
                        query_hidden("日常")[-2:] or None, self_mood, relation)
        real = mood_label_of(m["sentiment"], abs(m["sentiment"]))
        real_by = "dict"
        # ★Phase 0(judge 全覆盖): 门控消息一律 judge 重标 real——词典尺过敏感 81% 已实锤
        #   (judge 三方审核 9/7: 我 vs 词典仅 16-29%), 放弃模糊带方案, 全量 12-token 快调用
        j = judge_sentiment(m["text"])
        if j:
            real, real_by = j, "judge"
            judged_n += 1
        bel_cat = "pos" if "兴奋" in tom["emotion"] else ("neg" if tom["emotion"] in ("低落", "焦虑") else "neu")
        real_cat = "pos" if "兴奋" in real else ("neg" if real in ("低落", "焦虑") else "neu")
        row = {"date": DAY_KEY, "believed": tom["emotion"], "real": real,
               "correct": bel_cat == real_cat, "bel_cat": bel_cat, "real_cat": real_cat,
               "judged_by": tom.get("_by", "model"), "real_by": real_by,
               "need": tom.get("need", "")[:30], "text": m["text"][:40]}
        pe_rows.append(row)
        with open(pe_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ★cog 认知重构（每日一次）→ 暗注意力归档
    cog = ""
    try:
        day_brief = "；".join(m["text"][:30] for m in (msgs if len(msgs) <= 12 else msgs[:12]))
        _cog_prompt = (f"（{DAY_KEY}）你是雷姆。以下是主人今天经历的信息（全部来自文字消息，"
                       f"你们今天没有见面）：{day_brief[:600]}\n"
                       "用两三句第一人称内心独白重构：你从文字里知道了什么、你在担心什么、"
                       "你打算为他做什么。\n"
                       "★严禁任何神态/视觉/动作描写（眉头、眼神、表情、强撑、身体状态等一律禁止"
                       "——你没看见他，只看到文字）；严禁用「你」开头对主人说话。只要内心。")
        _bad = re.compile(r"眉头|眼神|表情|微笑|皱|叹气|强撑|苦笑|看着你|身体")
        cog = _chat([{"role": "user", "content": _cog_prompt}], max_tokens=140)
        if _bad.search(cog):                       # ★神态守卫: 重试一次(更强约束)
            cog = _chat([{"role": "user", "content": _cog_prompt +
                          "\n（再次强调：你今天没有见到主人，任何神态描写都是虚构。纯内心，三句内。）"}],
                         max_tokens=140)
        if _bad.search(cog):                       # 仍不达标 → 宁缺毋滥, 本日不归档
            print(f"  [cog] 神态描写守卫拦截, 本日 hidden 边跳过: {cog[:50]}")
            cog = ""
        if cog:
            add_hidden_text("日常", cog[:140], ts=time.time(), db=config.L2_DB)
    except Exception:  # noqa: BLE001
        pass

    # ★narrate 27B 润色(2026-09-07 矩阵读侧接线第二根): 规则骨架 → 她的第一人称连贯叙事。
    #   月度缓存: 当月节点数变化才重生成(8100 可用时); /mind 直接读缓存, 不依赖页面触发。
    try:
        _c = _sq3.connect(_auto_db)
        _mnodes = _c.execute(
            "SELECT COUNT(*) FROM autobiography WHERE strftime('%Y-%m', ts, 'unixepoch', 'localtime')=?",
            (DAY_KEY[:7],)).fetchone()[0]
        _c.close()
        _cache_dir = os.path.join(config.EXPERIMENTS, "narrate-cache")
        os.makedirs(_cache_dir, exist_ok=True)
        _cf = os.path.join(_cache_dir, f"{DAY_KEY[:7]}.json")
        _cached = {}
        if os.path.isfile(_cf):
            try:
                _cached = json.load(open(_cf, encoding="utf-8"))
            except json.JSONDecodeError:
                _cached = {}
        if _mnodes > 0 and _cached.get("node_count") != _mnodes and _port_open():
            from engine.autobiography import narrate as _nar
            _skeleton = _nar(DAY_KEY[:7], db=_auto_db)
            _polished = _chat([{"role": "user", "content":
                                f"你是雷姆。这是你自传体矩阵的本月切片骨架：\n{_skeleton[:1200]}\n"
                                "把它改写成连贯的第一人称自传体叙事（三五句，像回忆不是流水账），"
                                "保留具体事实与情绪，别罗列别复读，禁神态描写与口号腔。只输出叙事正文。"}],
                               max_tokens=320)
            if _polished.strip() and not re.search(r"眉头|眼神|表情|强撑|看着你", _polished):
                json.dump({"node_count": _mnodes, "text": _polished.strip()[:500],
                           "generated_at": time.time()},
                          open(_cf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"  [narrate] 本月自传叙事已润色缓存({_mnodes} 节点)")
    except Exception:  # noqa: BLE001
        pass

    # ④ L3 自传（仅门控消息；O14 emotion 现算 + O3 event_id 缝合 + ★self_eval 由 27B 填真实自我评价)
    import sqlite3 as _sq3
    _auto_db = os.path.join(config.MEMORY, "L3_core", "autobiography.db")
    l3_n = 0
    for i in sel_idx:
        m = msgs[i]
        try:
            rid = add_event(m["text"], ts=m["ts"], entity=entity_of(m["text"]),
                            emotion=mood_label_of(m["sentiment"], abs(m["sentiment"])),
                            relation="日常守护", event_id=f"ev-{DAY_KEY}-{i}",
                            confidence="high", evidence=m["text"][:120], db=_auto_db)
            l3_n += 1
        except Exception:  # noqa: BLE001
            continue
        # ★self_eval 填充(9/2 定案兑现: 由 27B 填真实自我评价, 禁代码模板)——每节点一次轻调用
        try:
            if _port_open():
                _se = _chat([{"role": "user", "content":
                              f"你是雷姆。今天发生了这件事：{m['text'][:70]}\n"
                              "用第一人称写一句你对此的自我评价（你学到的/你在意的，≤30 字，"
                              "禁神态描写，不要口号腔）。只输出这一句。"}], max_tokens=60)
                _se = _se.strip().split("\n")[0][:60]
                if _se and not re.search(r"眉头|眼神|表情|强撑|看着你", _se):
                    _c = _sq3.connect(_auto_db)
                    _c.execute("UPDATE autobiography SET self_eval=? WHERE id=?", (_se, rid))
                    _c.commit()
                    _c.close()
        except Exception:  # noqa: BLE001
            pass

    hit = sum(1 for r in pe_rows if r["correct"])
    return {"date": DAY_KEY, "status": "ok", "messages": len(msgs), "selected": len(sel_idx),
            "tom_judged": len(pe_rows), "tom_hit_vs_real": f"{hit}/{len(pe_rows)}",
            "judge_relabeled": judged_n, "cog_chars": len(cog), "l3_events": l3_n,
            "elapsed_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--yesterday", action="store_true",
                    help="★睡眠任务模式(05:30 launchd): 摄入昨天完整一天——夜里 22:00 后的"
                         "对话/消息不再被截断, 睡在夜班后面、醒来时昨天已完整内化")
    ap.add_argument("--max-tom", type=int, default=60)
    a = ap.parse_args()
    _d = a.date
    if a.yesterday and not _d:
        from datetime import timedelta
        _d = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(json.dumps(run(_d, a.max_tom), ensure_ascii=False, indent=1))
