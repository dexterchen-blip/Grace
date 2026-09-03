#!/usr/bin/env python3
"""build_l0_book.py — V2 真实书库构建器(2026-09-02 用户: 模拟本地 AI 系统真实运行).

放弃虚构 90 天书库(26 句池, 唯一率 19%), 改吃正式系统完整 L0(全部源):
  微信(消息级) + chat(对话) + email/exchange(邮箱摘要→邮件事件) + school(学校) + doc(文档)

→ 按真实日期归一 → inputs-v2/day-NNN.json(与 stress_engine 兼容格式:
  {"day", "date", "messages": [{"text","sentiment","weight"}], "events": [...]})

用法: ./run.sh .venv/bin/python3 v2/stress/build_l0_book.py [--l0 <正式L0目录>] [--out <输出目录>]
默认输出 experiments/run/stress/inputs-v2/ (不覆盖 K 轮在用的 inputs/)
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

TZ = timezone(timedelta(hours=8))  # Asia/Shanghai

# 清洗: 转账/CDATA/碎片/纯数字/链接
_SKIP = re.compile(
    r"(转账|CDATA|微信转账|发了一个红包|\[红包\]|\[转账\]|^https?://|^\d{4,}$|^[a-f0-9]{16,}$|^\.$|^。$|^\s*$)")


def _sentiment(text: str) -> float:
    """书库情绪标注 —— ★2026-09-03 Phase 0: 走统一 engine/sentiment.assess()（分层效价词表）。

    原 _POS/_NEG 词表仅 24+23 词且计次(0.3+0.15p) → 88.8% 消息 sentiment=0（"谢谢/催办/
    面签"都漏）→ derive 当日心态/图谱情绪边失真。统一模块含强/中/弱分层 + 否定削弱。
    """
    from engine.sentiment import assess
    return assess(text)["valence"]


def _clean(text: str) -> str:
    t = re.sub(r"<[^>]+>", "", text or "").strip()
    return t


def _iter_raw(l0_dir: str):
    """读全部 L0 源 → 产出 (ts_epoch, text, source)。"""
    sources = ["wechat.jsonl", "chat.jsonl", "email.jsonl",
               "exchange:inbox.jsonl", "exchange:school.jsonl", "school.jsonl", "doc:file.jsonl"]
    for fn in sources:
        fp = os.path.join(l0_dir, fn)
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
            if msgs:  # 消息级: wechat/chat
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    t = _clean(m.get("text", ""))
                    if not t or _SKIP.search(t):
                        continue
                    ts = m.get("ts") or r.get("epoch")
                    if not isinstance(ts, (int, float)):
                        continue
                    yield ts, t, fn.replace(".jsonl", "")
            else:  # 摘要级: email/exchange/school/doc → 解析成条目
                t = _clean(p.get("text", "")) if isinstance(p, dict) else ""
                if not t:
                    continue
                ep = r.get("epoch")
                if not isinstance(ep, (int, float)):
                    continue
                # 摘要 markdown → 拆成条目(## 标题 + 正文行)
                for chunk in _split_summary(t):
                    if chunk and not _SKIP.search(chunk):
                        yield ep, chunk, fn.replace(".jsonl", "")


def _split_summary(text: str) -> list[str]:
    """邮箱/学校摘要 markdown → 事件条目列表。

    ★2026-09-02 v2: 识别 ``` 代码块内的邮件条目 `[YYYY-MM-DD HH:MM] 标题` + 正文段落
    → 拆成独立邮件事件(细粒度); 其余 markdown 标题/段落也拆。
    """
    out = []
    lines = text.splitlines()
    mail_re = re.compile(r"^\s*\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*(.*)$")
    cur_title = ""
    cur_body = []
    in_code = False

    def flush():
        nonlocal cur_title, cur_body
        if cur_title or cur_body:
            txt = (cur_title + "：" + " ".join(cur_body)[:120]) if cur_title else " ".join(cur_body)[:120]
            txt = txt.strip("：: ")
            if len(txt) >= 6:
                out.append(txt)
        cur_title, cur_body = "", []

    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("```"):
            in_code = not in_code
            continue
        m = mail_re.match(s)
        if m and m.group(2):  # [日期] 邮件标题 → 新邮件条目
            flush()
            cur_title = m.group(2).strip()
            cur_body = []
            continue
        if m and not m.group(2):  # [日期] 空标题(续上文)
            continue
        if not in_code and (s.startswith("###") or s.startswith("##") or s.startswith("#")):
            flush()
            cur_title = re.sub(r"^#+\s*", "", s)
            cur_body = []
        else:
            cur_body.append(s)
    flush()
    return out


def main() -> None:
    l0_dir = "/Users/cz/WorkBuddy/skills find and make/local-ai-agent/memory/L0_raw"
    out_dir = os.path.join(config.EXPERIMENTS, "run", "stress", "inputs-v2")
    if "--l0" in sys.argv:
        l0_dir = sys.argv[sys.argv.index("--l0") + 1]
    if "--out" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--out") + 1]

    # 1. 收集全部条目 → 按天分组
    from collections import defaultdict
    by_day: dict[str, list] = defaultdict(list)
    total = 0
    for ts, text, src in _iter_raw(l0_dir):
        d = datetime.fromtimestamp(ts, tz=TZ).strftime("%Y-%m-%d")
        by_day[d].append({"text": text[:140], "sentiment": _sentiment(text),
                          "weight": 0.6, "source": src, "ts": ts})
        total += 1

    days = sorted(by_day)
    if not days:
        print("❌ 未解析到任何消息")
        return

    # 2. 写入 day-NNN.json(按真实日期排序, 从第 1 天编号)
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"days": len(days), "start": days[0], "end": days[-1],
                "total_raw": total, "files": []}
    for i, d in enumerate(days, 1):
        items = sorted(by_day[d], key=lambda x: x["ts"])
        msgs = [{"text": it["text"], "sentiment": it["sentiment"], "weight": it["weight"]}
                for it in items]
        rec = {"day": i, "date": d, "messages": msgs,
               "events": [{"text": m["text"], "sentiment": m["sentiment"], "weight": m["weight"]}
                          for m in msgs]}
        fp = os.path.join(out_dir, f"day-{i:03d}.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        manifest["files"].append(os.path.basename(fp))

    # 3. 统计验证
    all_text = [it["text"] for lst in by_day.values() for it in lst]
    uniq = len(set(all_text))
    daily = sorted(len(v) for v in by_day.values())
    print(f"=== V2 真实书库构建完成 ===")
    print(f"来源: {l0_dir}")
    print(f"总条目: {total} | 唯一: {uniq} | 唯一率 {uniq/total*100:.1f}%")
    print(f"覆盖: {len(days)} 天 ({days[0]} → {days[-1]})")
    print(f"每天条数: 最少 {daily[0]} / 中位 {daily[len(daily)//2]} / 最多 {daily[-1]} | 零消息天 {sum(1 for v in by_day.values() if not v)}")
    print(f"→ {out_dir}/day-001.json … day-{len(days):03d}.json")
    # 来源构成
    from collections import Counter
    srcs = Counter(it["source"] for lst in by_day.values() for it in lst)
    print(f"来源构成: {dict(srcs)}")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
