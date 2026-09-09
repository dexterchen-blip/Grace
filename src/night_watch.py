#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""night_watch.py — 夜班值守自动监控（2026-09-05 dsh 集成升级配套）。

职责：升级完成后的第一个夜班（D4 轮收尾 + night_pipeline 全链）持续跟踪系统状态
与关键指标，重点检验升级（sender 修复 / Michelle MCP / dashboard 升级路由）是否
引入异常，自动记录错误日志供次日排查。

形态：纯 stdlib，launchd StartInterval 300 每 5 分钟一轮；窗口外（09:00-22:00）秒退。
输出：exchange/shared/night-watch/
  watch-YYYY-MM-DD.jsonl   每轮一行摘要（全量事实）
  err-YYYY-MM-DD.md        仅异常事件追加（次日排查入口）
  day-report-YYYY-MM-DD.md 08:30 后第一轮生成的全夜汇总，之后当天静默

铁律：监控自身绝不拖系统——零模型、零重 IO；任何自身异常吞掉并记 stderr。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/ -> local-ai-agent/
WATCH_DIR = os.path.join(REPO, "exchange", "shared", "night-watch")
def _local_tz():
    """★2026-09-09 时区迁移修正: 用户已移居美国(系统时区=PDT)。
    值守窗口/时间戳一律用本地时区（AIAGENT_TZ 可覆盖, 与 grace-sleep plist 同约定）,
    不再硬编码 UTC+8——旧逻辑导致值守跑在用户白天、真实夜间全盲。"""
    name = os.environ.get("AIAGENT_TZ")
    if name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(name)
        except Exception:
            pass
    return datetime.now().astimezone().tzinfo


TZ = _local_tz()

WINDOW_START = 22  # 22:00 起值守
WINDOW_END = 9     # 09:00 止（跨午夜）
REPORT_AT = 8      # 08:30 后写全天汇总

# 增量扫描的日志与告警关键词
LOG_FILES = {
    "night-pipeline": os.path.join(REPO, "exchange", "shared", "night-pipeline.log"),
    "night-pipeline.err": os.path.join(REPO, "exchange", "shared", "night-pipeline.err"),
    "serve-night": os.path.join(REPO, "exchange", "shared", "serve-night.log"),
    "serve-day": os.path.join(REPO, "exchange", "shared", "serve-day.log"),
    "dsh-web.err": os.path.join(REPO, "exchange", "shared", "dsh-web.err"),
    "dsh-web.log": os.path.join(REPO, "exchange", "shared", "dsh-web.log"),
    "dashboard.err": os.path.join(REPO, "exchange", "shared", "dashboard.err"),
    "michelle-mcp": os.path.join(REPO, "exchange", "shared", "michelle-mcp.log"),
}
ERROR_RE = re.compile(
    r"Traceback|ERROR|CRITICAL|panic|MemoryError|OOM|watchdogd|"
    r"ModuleNotFoundError|PermissionError| refused|timeout(?!.* benign)", re.IGNORECASE)

OFFSET_FILE = os.path.join(WATCH_DIR, ".offsets.json")


def _now() -> datetime:
    return datetime.now(TZ)


def _in_window(d: datetime) -> bool:
    return d.hour >= WINDOW_START or d.hour < WINDOW_END


def _probe_http(url: str, timeout: float = 3.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _mem_free_pct() -> float:
    """memory_pressure 的 System-wide memory free percentage。失败返回 -1（不误报）。"""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True,
                             text=True, timeout=10).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return float(m.group(1)) if m else -1.0
    except Exception:
        return -1.0


def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True,
                             text=True, timeout=10).stdout
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def _load_offsets() -> dict:
    try:
        with open(OFFSET_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_offsets(off: dict) -> None:
    try:
        with open(OFFSET_FILE, "w", encoding="utf-8") as f:
            json.dump(off, f)
    except OSError:
        pass


def _scan_logs() -> tuple[list[dict], dict]:
    """增量扫描各日志，返回 (异常事件列表, 新 offsets)。首见日志从头扫但只取尾部 500 行。"""
    events: list[dict] = []
    off = _load_offsets()
    for name, path in LOG_FILES.items():
        if not os.path.exists(path):
            continue
        try:
            size = os.path.getsize(path)
            seen = int(off.get(name, 0))
            if seen == 0:
                seen = max(0, size - 60_000)  # 首见只看尾部 60KB，避免首夜全量重扫
            if size < seen:  # 日志被轮转/清空
                seen = 0
            if size == seen:
                continue
            with open(path, "rb") as f:
                f.seek(seen)
                chunk = f.read().decode("utf-8", errors="replace")
            off[name] = size
            for ln in chunk.splitlines():
                if ERROR_RE.search(ln):
                    events.append({"log": name, "line": ln.strip()[:300]})
        except Exception:
            continue
    return events, off


def _probe_round() -> dict:
    d = _now()
    r: dict = {"ts": d.isoformat(timespec="seconds"), "issues": []}

    # 进程存活
    r["night_pipeline_pids"] = _pgrep(r"night_pipeline\.py")
    r["stress_pids"] = _pgrep(r"ai-sandbox-stress|stress_engine")
    r["dashboard_alive"] = _probe_http("http://127.0.0.1:3091/")
    r["dsh_web_alive"] = _probe_http("http://127.0.0.1:3090/")

    # 模型端口（OpenAI 兼容 /v1/models）
    import urllib.request
    def port_up(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False
    r["p8100"] = port_up(8100)
    r["p8200"] = port_up(8200)

    # 双模型铁律：8100 与 8200 同时在线 = CRITICAL
    if r["p8100"] and r["p8200"]:
        r["issues"].append({"level": "CRITICAL",
                            "msg": "8100 与 8200 同时在线——双模型同驻，watchdog panic 风险！"})
    elif not r["p8100"]:
        # ★2026-09-09 补检查(4 小时静默掉线教训): 8100 单独掉线且夜班管线不在跑
        #   (35B 窗口内 8100 本就该停) → 故障告警。9/9 07:35-13:39 掉线 4h 无人觉察,
        #   两轮压测 PE 96% 静默回退 rule → 11% 假读数。
        try:
            _np_up = subprocess.run(["pgrep", "-f", "night_pipeline"],
                                    capture_output=True, timeout=5).returncode == 0
        except Exception:  # noqa: BLE001
            _np_up = False
        if not _np_up:
            r["issues"].append({"level": "WARN",
                                "msg": "8100 掉线且夜班管线未运行——day-model 意外停机"
                                       "（bootout/崩溃未恢复），白天系统与 judge 均不可用"})

    # 内存
    r["mem_free_pct"] = _mem_free_pct()
    if 0 <= r["mem_free_pct"] < 8:
        r["issues"].append({"level": "CRITICAL",
                            "msg": f"内存 free {r['mem_free_pct']:.0f}%（<8%，爆机边缘）"})
    elif 8 <= r["mem_free_pct"] < 15:
        r["issues"].append({"level": "WARN",
                            "msg": f"内存 free {r['mem_free_pct']:.0f}%（<15%）"})

    # 核心服务死活（03:00-09:00 夜班窗口内 dashboard/dsh-web 应在线）
    if 3 <= d.hour < 9:
        if not r["dashboard_alive"]:
            r["issues"].append({"level": "WARN", "msg": "dashboard :3091 探活失败"})
        if not r["dsh_web_alive"]:
            r["issues"].append({"level": "WARN", "msg": "dsh-web :3090 探活失败"})

    # 日志增量扫描
    events, off = _scan_logs()
    r["log_events"] = len(events)
    r["_events"] = events
    _save_offsets(off)

    # 夜班产物检查：05:00 后若 night-pipeline.log 今日无增长 → WARN（夜班可能没跑）
    np_log = LOG_FILES["night-pipeline"]
    if 5 <= d.hour < 9 and os.path.exists(np_log):
        mtime = datetime.fromtimestamp(os.path.getmtime(np_log), TZ)
        if mtime.date() == d.date() and (d - mtime) > timedelta(hours=2):
            # 2026-09-06 误报修复：night_pipeline 是一次性任务（实测 ~21 分钟跑完 exit=0），
            # 完成后日志理应不再增长。当日日志含「管线完成 … exit=0」→ idle 属预期，不告警。
            try:
                with open(np_log, encoding="utf-8") as f:
                    if f"管线完成 {d.strftime('%Y-%m-%d')}" in f.read()[-50_000:]:
                        pass  # 今日管线已正常收尾
                    else:
                        r["issues"].append({"level": "WARN",
                                            "msg": f"night-pipeline.log 已 {int((d-mtime).total_seconds()//60)} 分钟无增长"})
            except OSError:
                r["issues"].append({"level": "WARN",
                                    "msg": f"night-pipeline.log 已 {int((d-mtime).total_seconds()//60)} 分钟无增长"})

    # 升级验证点：sender 标注生效（05:00 后夜班巩固若已跑，日志应出现 [我]/[对方·）
    if 5 <= d.hour < 9:
        try:
            with open(np_log, encoding="utf-8") as f:
                tail = f.read()[-100_000:]
            if "[我]" in tail or "[对方·" in tail:
                r["sender_fix_seen"] = True
        except OSError:
            pass

    return r


def _append(day: str, obj: dict) -> None:
    os.makedirs(WATCH_DIR, exist_ok=True)
    with open(os.path.join(WATCH_DIR, f"watch-{day}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _append_err(day: str, events: list[dict], ctx: dict) -> None:
    if not events and not ctx.get("issues"):
        return
    os.makedirs(WATCH_DIR, exist_ok=True)
    with open(os.path.join(WATCH_DIR, f"err-{day}.md"), "a", encoding="utf-8") as f:
        f.write(f"\n## {_now().isoformat(timespec='seconds')}\n")
        for i in ctx.get("issues", []):
            f.write(f"- **[{i['level']}]** {i['msg']}\n")
        for e in events[:20]:
            f.write(f"- [{e['log']}] `{e['line']}`\n")
        if len(events) > 20:
            f.write(f"- …另有 {len(events)-20} 条日志异常行\n")


def _write_day_report(day: str) -> None:
    path = os.path.join(WATCH_DIR, f"watch-{day}.jsonl")
    rounds, err_total, issue_total, sender_seen = 0, 0, 0, False
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                rounds += 1
                err_total += o.get("log_events", 0)
                issue_total += len(o.get("issues", []))
                sender_seen = sender_seen or bool(o.get("sender_fix_seen"))
    except OSError:
        pass
    out = os.path.join(WATCH_DIR, f"day-report-{day}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# 夜班值守汇总 {day}\n\n"
                f"- 值守轮数: {rounds}（5 分钟/轮）\n"
                f"- 日志异常行累计: {err_total}\n"
                f"- 主动告警累计: {issue_total}\n"
                f"- sender 修复生效抽查: {'✅ 见到 [我]/[对方·] 标注' if sender_seen else '⚠️ 未观察到（夜班巩固可能未跑或未到 05:00）'}\n"
                f"- 详情: watch-{day}.jsonl / err-{day}.md\n")
    print(f"[night_watch] day report -> {out}", file=sys.stderr)


def main() -> int:
    try:
        d = _now()
        if not _in_window(d):
            return 0  # 窗口外秒退
        day = d.strftime("%Y-%m-%d")
        r = _probe_round()
        events = r.pop("_events", [])
        _append(day, r)
        _append_err(day, events, r)
        # 08:30 后第一轮写全天汇总（report 存在则跳过）
        if (d.hour >= REPORT_AT and d.minute >= 30) or d.hour > REPORT_AT:
            rep = os.path.join(WATCH_DIR, f"day-report-{day}.md")
            if not os.path.exists(rep):
                _write_day_report(day)
    except Exception as e:  # 监控自身绝不崩出噪音
        print(f"[night_watch] internal error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
