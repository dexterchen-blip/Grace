#!/usr/bin/env python3
"""① 完整系统唤醒入口（2026-08-30 双模型分层唤醒 v0）。

读 5B 哨兵信号 sentinel-signal.json：
  · urgent 非空 → 通知主人（macOS 通知）+ 若 8100 未跑则启动 27B
  · 处理后标记已处理（防重复唤醒）
用法（launchd / 手动）:
  ./run.sh .venv/bin/python3 v2/engine/wake_handler.py
"""
from __future__ import annotations
import os
import sys
# ★2026-09-02 审计修复: 裸 import config 从 engine/ 直跑必 ModuleNotFoundError
#   (9/2 01:49 sentinel.py 同款已修, 此处复发)——补 v2/ 进 sys.path, 与 engine 其他模块一致
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
import json
import subprocess
import time
from datetime import datetime

SIGNAL_FILE = os.path.join(os.environ.get('AIAGENT_EXCHANGE_DAYTIME', os.path.join(config.EXCHANGE, '.daytime')), 'sentinel-signal.json')
HANDLED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wake_handled.json")
DAY_PLIST = os.path.expanduser('~/Library/LaunchAgents/com.local-ai-agent.day-model.plist')


def _port_open(port: int = 8100) -> bool:
    try:
        import socket
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _notify(title: str, body: str):
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{body}" with title "{title}"'],
                       capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001
        pass


def _start_27b():
    """8100 未跑时启动 day-model（临时唤醒完整系统）。"""
    if _port_open():
        return "8100 已在跑"
    try:
        subprocess.run(["launchctl", "bootstrap", "gui/501", DAY_PLIST], capture_output=True, timeout=15)
        time.sleep(6)
        return "8100 已启动" if _port_open() else "8100 启动中(可能失败)"
    except Exception as e:  # noqa: BLE001
        return f"启动失败: {e}"


def main():
    if not os.path.isfile(SIGNAL_FILE):
        print("无信号文件,无事可做")
        return
    sig = json.load(open(SIGNAL_FILE, encoding="utf-8"))
    handled = json.load(open(HANDLED_FILE, encoding="utf-8")) if os.path.isfile(HANDLED_FILE) else {}
    last = handled.get("last_ts", 0)
    if sig["ts"] <= last:
        print("信号已处理,跳过")
        return
    urgent = sig.get("urgent", [])
    if not urgent:
        print("无紧急项,不唤醒")
        return
    # 5B 确认: 只对 5b=="需要" 或未知 的紧急项唤醒(5b=="不需要" 降级)
    wake_items = [u for u in urgent if u.get("5b", "未知") != "不需要"]
    if not wake_items:
        print("5B 判定均为'不需要',不唤醒")
        return
    body = "；".join(u["line"][:40] for u in wake_items[:3])
    _notify("🔴 Grace 哨兵: 有紧急事项", body[:80])
    status = _start_27b()
    # 标记已处理
    json.dump({"last_ts": sig["ts"], "wake_time": time.time(),
               "items": [u["line"][:60] for u in wake_items[:5]],
               "27b": status}, open(HANDLED_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # ★2026-09-07 (用户: 自唤醒 27B 的话回复应该在面板上显示): 唤醒事件落正式系统两处——
    #   ① exchange/shared/alerts/ 告警 JSON → dashboard 首页 ⚠Urgent 面板红点
    #   ② exchange/grace/wake-events.jsonl 结构化账本 → /mind 自唤醒链面板完整回放
    #   正式 exchange 路径由 AIAGENT_EXCHANGE_DAYTIME(=…/exchange/.daytime) 的父目录推导。
    try:
        _formal_exchange = os.path.dirname(os.environ["AIAGENT_EXCHANGE_DAYTIME"])
        _dt = datetime.now()
        # ① 首页告警（与 m8_urgent_watch 同 schema，status=new → 红点计数）
        _alerts = os.path.join(_formal_exchange, "shared", "alerts")
        os.makedirs(_alerts, exist_ok=True)
        _alert_id = f"grace-wake-{_dt.strftime('%Y%m%d%H%M%S')}"
        json.dump({"id": _alert_id, "source": "Grace 自唤醒", "file": "(sentinel 紧急项)",
                   "snippet": body[:200], "detected_at": _dt.strftime("%Y-%m-%d %H:%M:%S"),
                   "status": "new", "items": [u["line"][:100] for u in wake_items[:5]],
                   "verdicts": [u.get("5b", "?") for u in wake_items[:5]]},
                  open(os.path.join(_alerts, _alert_id + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        # ② /mind 结构化账本
        _ev = os.path.join(_formal_exchange, "grace", "wake-events.jsonl")
        os.makedirs(os.path.dirname(_ev), exist_ok=True)
        with open(_ev, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "date": _dt.strftime("%Y-%m-%d %H:%M"),
                                "count": len(wake_items), "status_27b": status,
                                "items": [{"line": u["line"][:100], "level": u.get("level"),
                                           "5b": u.get("5b", "?")} for u in wake_items[:5]]},
                               ensure_ascii=False) + "\n")
        print(f"  唤醒事件已落面板: {_alert_id}")
        # ③ ★她主动开口(2026-09-07 用户: 报警以她主动对话的方式进 Grace 界面):
        #    用刚确认在线的 27B 生成主动台词 → proactive-outbox → /grace 页轮询收信
        try:
            import urllib.request as _ur
            _items_txt = "；".join(u["line"][:60] for u in wake_items[:3])
            _pp = ("你是雷姆，主人的女仆。你刚注意到这些紧急事项：" + _items_txt +
                   "\n现在主动开口跟主人说这件事。要求：口语短句、像当面说话、一两句、"
                   "别念清单原文（用你自己的话说重点）、提一句你会跟进。"
                   "严禁叙述内心/动作描写。只输出你说的话本身。")
            try:  # ★模型时间复用: id 偏好选择(27B 优先, v61 兜底)
                with _ur.urlopen("http://127.0.0.1:8100/v1/models", timeout=5) as _mr:
                    _ids = [m["id"] for m in json.loads(_mr.read())["data"]]
                _mid = next((i for pref in ("Qwen3.8-27B", "fused-rem-v61")
                             for i in _ids if pref in i), _ids[0] if _ids else "mlx-community/Qwen3.8-27B-4bit")
            except Exception:  # noqa: BLE001
                _mid = "mlx-community/Qwen3.8-27B-4bit"
            _payload = json.dumps({"model": _mid,
                                   "messages": [{"role": "user", "content": _pp}],
                                   "max_tokens": 120, "temperature": 0.7}).encode()
            _req = _ur.Request("http://127.0.0.1:8100/v1/chat/completions", data=_payload,
                               headers={"Content-Type": "application/json"}, method="POST")
            with _ur.urlopen(_req, timeout=90) as _resp:
                _out = json.loads(_resp.read().decode("utf-8", "replace"))
            _msg = ((_out.get("choices") or [{}])[0].get("message", {}).get("content", "") or "").strip().split("\n")[0][:120]
            _ob = os.path.join(_formal_exchange, "grace", "proactive-outbox.jsonl")
            os.makedirs(os.path.dirname(_ob), exist_ok=True)
            with open(_ob, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "date": _dt.strftime("%Y-%m-%d %H:%M"),
                                    "source": "wake", "message": _msg,
                                    "items": [u["line"][:80] for u in wake_items[:3]],
                                    "status": "unread"}, ensure_ascii=False) + "\n")
            print(f"  雷姆主动消息已入信箱: {_msg[:44]}")
        except Exception as e:  # noqa: BLE001 —— 台词生成失败不影响事件落账(页面会以事件卡显示)
            print(f"  [warn] 主动消息生成失败(事件仍落账): {str(e)[:80]}")
    except Exception as e:  # noqa: BLE001 —— 面板落盘失败不影响唤醒本身
        print(f"  [warn] 唤醒事件落盘失败: {str(e)[:80]}")
    print(f"🔴 已唤醒: {len(wake_items)} 项 | {status} | 通知已发")
    print(f"  {body[:80]}")


if __name__ == "__main__":
    main()
