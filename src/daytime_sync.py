#!/usr/bin/env python3
"""L-1 瞬时记忆实时同步（2026-08-21 用户需求：每 30 分钟微信 + 邮件 → .daytime/）。

白天模型（27B 代理 + 惠惠 persona）对话时注入 L-1 段（见 m6_dashboard._l1_context），
实时看到最近 30 分钟的微信/邮件动态。
L-1 = exchange/.daytime/（瞬时层，夜班 03:00 清空，永不进 L0-L3 语义索引）。

launchd：com.local-ai-agent.daytime-sync（StartInterval 1800）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCHANGE = os.path.join(REPO, "exchange")
DAYTIME = os.path.join(EXCHANGE, ".daytime")
def _local_tz():
    """★2026-09-09 时区统一修正：机器已迁 PDT，UTC+8 硬编码全部改本地时区
    （AIAGENT_TZ 可覆盖，与 night_watch/grace-sleep 同约定）。"""
    import os as _os
    name = _os.environ.get("AIAGENT_TZ")
    if name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(name)
        except Exception:
            pass
    return datetime.now().astimezone().tzinfo


TZ_CN = _local_tz()

WECHAT_VENV = "/Users/cz/.workbuddy/binaries/python/envs/wechat-bot/bin/python"
WECHAT_CONFIG = "/Users/cz/WorkBuddy/skills find and make/wechat-summary-bot/config.toml"
WECHAT_DB = "/Users/cz/WorkBuddy/skills find and make/wechat-summary-bot/summary.db"
WECHAT_STATE = os.path.join(DAYTIME, ".wechat_state.json")

EMAIL_CMD = [sys.executable, os.path.join(REPO, "src", "scrape.py"), "email", "--manual"]


def today() -> str:
    return datetime.now(TZ_CN).strftime("%Y-%m-%d")


def sync_email() -> bool:
    """邮件 → .daytime/邮箱摘要-<today>.md（scrape.py email --manual，已有 L-1 落点）。"""
    try:
        r = subprocess.run(EMAIL_CMD, capture_output=True, text=True, timeout=240, cwd=REPO)
        return r.returncode == 0
    except Exception as e:
        print(f"[daytime-sync] email 异常: {e}")
        return False


def sync_wechat() -> tuple[int, int]:
    """刷新 summary.db + 提取【当天】全部消息到 .daytime/wechat-<today>.md。
    只留当天（L-1 瞬时语义）：since = 今天 00:00（恒定）——每次重写整份当天摘要，
    历史消息不进 L-1。2026-08-21 修：原 max(水位,当天零点) 让 md 只含本次增量，
    覆盖写后把当天更早消息冲掉（白天模型看不到完整当天动态）。
    返回 (消息数, 会话数)。"""
    # 1) 刷新 summary.db（wechat_summary.run 增量读微信本地库，实测 ~2s）
    #    2026-08-23 修：微信 app 活跃读写时解密等 DB 锁会卡 → 300s 超时（间歇）；
    #    超时后隔 60s 重试一次（等 app 忙完）；仍失败则降级继续用已有库。
    for _attempt in (1, 2):
        try:
            subprocess.run([WECHAT_VENV, "-m", "wechat_summary.run", "--config", WECHAT_CONFIG],
                           capture_output=True, text=True, timeout=300,
                           cwd=os.path.dirname(WECHAT_CONFIG))
            break
        except Exception as e:
            if _attempt == 1:
                print(f"[daytime-sync] wechat 同步超时（第 1 次），60s 后重试…: {e}")
                time.sleep(60)
            else:
                print(f"[daytime-sync] wechat 同步异常: {e}")  # 不致命：继续用已有库
    if not os.path.exists(WECHAT_DB):
        return 0, 0

    # 当天 00:00（本地）的 unix 时间戳 —— 恒定 since：摘要 = 当天全部消息
    now_dt = datetime.now(TZ_CN)
    day_start = datetime(now_dt.year, now_dt.month, now_dt.day, tzinfo=TZ_CN).timestamp()
    since = day_start

    con = sqlite3.connect(f"file:{WECHAT_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT conversation, display_name, is_send, ts, content FROM messages"
        " WHERE ts > ? ORDER BY conversation, ts",
        (since,),
    ).fetchall()
    con.close()
    if not rows:
        return 0, 0

    by_conv: dict[str, list[tuple[int, bool, str]]] = {}
    for conv, name, is_send, ts, content in rows:
        by_conv.setdefault(name or conv, []).append((int(ts), bool(is_send), content or ""))
    max_ts = max(r[3] for r in rows)

    # 2) 组 md：每会话取最近 8 条，整体截断 ~2500 字
    lines = [f"# 微信最新动态 {today()}（L-1 瞬时，30 分钟同步）\n"]
    for name, msgs in sorted(by_conv.items(), key=lambda kv: -max(m[0] for m in kv[1])):
        lines.append(f"\n## {name}")
        for ts, is_send, content in msgs[-8:]:
            hhmm = datetime.fromtimestamp(ts, TZ_CN).strftime("%H:%M")
            who = "发送" if is_send else "收到"
            lines.append(f"- [{hhmm}] {who}: {content[:60]}")
        if sum(len(l) for l in lines) > 2500:
            lines.append("\n…（更多会话略，见 summary.db）")
            break

    with open(os.path.join(DAYTIME, f"wechat-{today()}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(WECHAT_STATE, "w", encoding="utf-8") as f:
        json.dump({"max_ts": max_ts}, f)
    return len(rows), len(by_conv)


def _clean_old_files() -> int:
    """清理 .daytime/ 里非当天的数据文件（L-1 只留当天，历史不背）。
    点前缀文件（水位等）与当天文件保留。返回清理数。"""
    n = 0
    t = today()
    try:
        for fn in os.listdir(DAYTIME):
            if fn.startswith("."):
                continue
            if t in fn:  # 文件名含今天日期 → 保留
                continue
            try:
                os.remove(os.path.join(DAYTIME, fn))
                n += 1
            except OSError:
                pass
    except OSError:
        pass
    return n


def main() -> None:
    # 夜班时段（02:00-05:00）跳过：03:00 会清空 L-1，且 35B 在跑，避免白跑/抢资源
    h = datetime.now(TZ_CN).hour
    if 2 <= h < 5:
        print(f"[daytime-sync] {h}:00 夜班时段，跳过")
        return
    os.makedirs(DAYTIME, exist_ok=True)
    cleaned = _clean_old_files()
    e = sync_email()
    w_new, w_conv = sync_wechat()
    ts = datetime.now(TZ_CN).strftime("%H:%M")
    print(f"[daytime-sync] {ts} email={'✓' if e else '✗'} wechat=+{w_new}条/{w_conv}会话"
          f"（清旧 {cleaned} 个）→ {DAYTIME}")


if __name__ == "__main__":
    main()
