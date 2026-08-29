#!/usr/bin/env python3
"""5B 哨兵突发检测器（2026-08-29 双模型分层唤醒架构 v0）。

设计：
  · 常驻轻量（Llama-3.2-3B 4bit ~2GB）—— 实时巡检 .daytime/（L-1 实时抓取）
  · 只读 —— 不写记忆/不训练（无改动权限）
  · ★ 监测关键词由完整 Grace V2.1 决定：读 sentinel_keywords.json（Grace 夜班可更新，
    哨兵只读该配置，无写权限）
  · 发现突发 → 紧急度分级 → 写唤醒信号（sentinel-signal.json）→ 完整系统按需唤醒

用法:
  ./run.sh .venv/bin/python3 v2/engine/sentinel.py [--scan] [--model]
  --scan    只跑规则扫描(关键词分级,不加载 5B)
  --model   加载 5B 做兜底理解(慢)
"""
from __future__ import annotations
from engine import config
import json
import os
import re
import sys
import time
from datetime import datetime

SENTINEL_DIR = os.path.dirname(os.path.abspath(__file__))
KEYWORDS_FILE = os.path.join(SENTINEL_DIR, "sentinel_keywords.json")   # ★ Grace 决定
SIGNAL_FILE = "os.path.join(config.EXCHANGE, '.daytime')/sentinel-signal.json"
DAYTIME = "os.path.join(config.EXCHANGE, '.daytime')"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentinel_state.json")   # 哨兵水位(只扫新增)


def load_keywords() -> dict:
    """读关键词配置（Grace V2.1 决定；哨兵只读）。"""
    if os.path.isfile(KEYWORDS_FILE):
        return json.load(open(KEYWORDS_FILE, encoding="utf-8"))
    return {"urgent": [], "important": [], "routine": []}


def _since() -> float:
    """读哨兵水位（上次扫描时间）；无水位 = 当天 00:00（5B 只看当天/新增）。"""
    if os.path.isfile(STATE_FILE):
        try:
            return json.load(open(STATE_FILE, encoding="utf-8")).get("last_scan", 0.0)
        except (OSError, ValueError):
            pass
    now = datetime.now()
    return datetime(now.year, now.month, now.day).timestamp()


def scan_daytime(keywords: dict, daytime: str = DAYTIME, since: float | None = None) -> list[dict]:
    """★ 5B 无完整注意力 —— 只扫水位之后新增的信息（当天/更新），不扫历史。"""
    since = since if since is not None else _since()
    hits = []
    if not os.path.isdir(daytime):
        return hits
    pats = {lvl: [k.lower() for k in kw] for lvl, kw in keywords.items()}
    for root, _, files in os.walk(daytime):
        for fn in sorted(files):
            if fn.endswith(".json") and "signal" in fn:
                continue
            fp = os.path.join(root, fn)
            if os.path.getmtime(fp) < since:      # ★ 只处理水位后的新文件
                continue
            fp = os.path.join(root, fn)
            try:
                txt = open(fp, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            low = txt.lower()
            for line in txt.splitlines():
                if not line.strip():
                    continue
                ll = line.lower()
                level = None
                for lvl in ("urgent", "important", "routine"):
                    if any(k and k in ll for k in pats[lvl]):
                        level = lvl
                        break
                if level:
                    hits.append({"file": fn, "line": line.strip()[:120],
                                 "level": level, "ts": time.time()})
    return hits


def judge_with_5b(hits: list[dict]) -> list[dict]:
    """5B 兜底：对命中条目做紧急度确认（理解上下文,不过滤真突发）。"""
    try:
        import os as _os
        _os.environ["HF_HUB_OFFLINE"] = "1"        # 离线铁律(沙箱死代理 502)
        for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            _os.environ.pop(_k, None)
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        model, tok = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
        sampler = make_sampler(temp=0.2)
        for h in hits:
            if h["level"] == "routine":
                continue
            q = (f"以下是主人邮箱/微信的一条消息。判断是否需要立即通知主人(仅答:需要/不需要)。\n"
                 f"消息:{h['line']}")
            p = tok.apply_chat_template([{"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True)
            ans = generate(model, tok, prompt=p, max_tokens=12, sampler=sampler).strip()
            h["5b"] = "需要" if ("需要" in ans and "不需要" not in ans[:4]) else ("不需要" if "不需要" in ans else "未知")
        del model
    except Exception as e:  # noqa: BLE001
        for h in hits:
            h["5b"] = f"err:{str(e)[:40]}"
    return hits


def emit_signal(hits: list[dict]) -> dict:
    """写唤醒信号（Grace 完整系统读取）。"""
    urgent = [h for h in hits if h["level"] == "urgent"]
    important = [h for h in hits if h["level"] == "important"]
    routine = [h for h in hits if h["level"] == "routine"]
    # 推进水位（下次只扫更新）
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_scan": time.time()}, f)
    sig = {"ts": time.time(), "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "urgent": urgent[:10], "important": important[:10], "routine_count": len(routine),
           "wake": bool(urgent), "note": "🔴紧急→立即唤醒;🟡重要→等合适时机;🟢日常→夜班汇总"}
    os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
    with open(SIGNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(sig, f, ensure_ascii=False, indent=1)
    return sig


def main():
    use_model = "--model" in sys.argv
    kw = load_keywords()
    print(f"=== 5B 哨兵巡检 {datetime.now().strftime('%H:%M')} ===")
    print(f"关键词配置(Grace 决定): urgent={len(kw.get('urgent',[]))} important={len(kw.get('important',[]))}")
    hits = scan_daytime(kw)
    print(f"扫描 .daytime: 命中 {len(hits)} 条")
    if hits and use_model:
        hits = judge_with_5b(hits)
        for h in hits[:5]:
            print(f"  [{h['level']}][5b:{h.get('5b','')}] {h['line'][:50]}")
    else:
        for h in hits[:8]:
            print(f"  [{h['level']}] {h['line'][:50]}")
    sig = emit_signal(hits)
    print(f"\n信号: {'🔴 唤醒' if sig['wake'] else '🟢 不唤醒'} ｜ 紧急 {len(sig['urgent'])} ｜ 重要 {len(sig['important'])}")
    print(f"→ {SIGNAL_FILE}")


if __name__ == "__main__":
    main()
