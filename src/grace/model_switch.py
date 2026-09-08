#!/usr/bin/env python3
"""model_switch.py —— 8100 单实例·模型时间复用切换器（2026-09-07，用户拍板方案）。

进 /grace → 切雷姆 V6.1（fused-rem-v61 完整融合权重）；离开（心跳超时）→ 切回通用 27B。
单模型铁律兼容：同一时刻 :8100 只驻留一个模型；切换 = 停旧起新（~60-90s）。

机制：
  · grace 模式: launchd day-model bootout（防 KeepAlive 拉起打架）→ serve_day.sh grace 8100
  · normal 模式: kill grace 服务 → day-model bootstrap 恢复（launchd 自管）
  · 状态: exchange/grace/model-mode.json {mode, switched_at}；心跳: grace-active 文件
  · 切换互斥锁: 防并发重启
"""
from __future__ import annotations
import json
import os
import subprocess
import time
import urllib.request

GRACE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(GRACE))
STATE_F = os.path.join(REPO, "exchange", "grace", "model-mode.json")
LOCK_F = os.path.join(REPO, "exchange", "grace", "model-switch.lock")
ACTIVE_F = os.path.join(REPO, "exchange", "grace", "grace-active")
DAY_PLIST = os.path.expanduser("~/Library/LaunchAgents/com.local-ai-agent.day-model.plist")
SERVE = os.path.join(REPO, "src", "serve_day.sh")
LOG = os.path.join(REPO, "exchange", "shared", "serve-grace.log")
V61_ID_SUB = "fused-rem-v61"
DEFAULT_ID_SUB = "Qwen3.8-27B"
GRACE_MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"


def served_model_ids(timeout: int = 5) -> list[str]:
    """/v1/models 全部 id(mlx 服务器列表顺序不定, Qwen3-8B 可能排 data[0])。"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8100/v1/models", timeout=timeout) as r:
            return [m["id"] for m in json.loads(r.read())["data"]]
    except Exception:  # noqa: BLE001
        return []


def served_model_id(timeout: int = 5) -> str | None:
    ids = served_model_ids(timeout)
    return ids[0] if ids else None


def mode_of_served(ids: list[str] | str | None) -> str:
    """★按全部 id 判定(9/7 实测: mlx 列表把 Qwen3-8B 排 data[0], 只看首个必错)。"""
    if not ids:
        return "down"
    ids = [ids] if isinstance(ids, str) else ids
    if any(V61_ID_SUB in i for i in ids):
        return "grace"
    if any(DEFAULT_ID_SUB in i for i in ids):
        return "normal"
    return "unknown"


def _kill_8100() -> None:
    subprocess.run(["bash", "-c",
                    "lsof -nP -iTCP:8100 -sTCP:LISTEN -t 2>/dev/null | xargs kill 2>/dev/null"],
                   capture_output=True, timeout=10)


def _wait_ready(want: str, timeout: int = 240) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(want in i for i in served_model_ids()):
            return True
        time.sleep(4)
    return False


def is_night() -> bool:
    """★2026-09-08 夜班守卫 v2(用户: "Grace 的睡眠放到夜班后面"): 守卫窗口 = 夜班管线
    实际运行窗 03:00-08:00(night-pipeline 03:00 触发; night-compare 用 :8100 到 07:55)。
    此窗口内 ①/grace 心跳不切 grace(:8100 归夜班管线/compare) ②35B 驻留时绝不 bootstrap
    day-model(双驻留 panic)。晚间 22:00-03:00 她醒着——V6.1 心跳切换照常, 陪聊到凌晨。
    Grace 自己的睡眠 = 05:30 独立任务(摄入昨天完整一天), 睡在夜班后面。"""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        h = datetime.now(ZoneInfo(os.environ.get("AIAGENT_TZ", "Asia/Los_Angeles"))).hour
    except Exception:  # noqa: BLE001
        from datetime import datetime as _dt
        h = _dt.now().hour
    return 3 <= h < 8


def _night_35b_alive() -> bool:
    """:8200 是否有 35B 驻留(夜间单模型铁律的另一半)。"""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", 8200), timeout=1):
            return True
    except OSError:
        return False


def switch(mode: str) -> dict:
    """切换 :8100 服务模型。mode: grace | normal。返回 {status, mode, detail}。"""
    # ★夜班守卫: 22-08 切 grace 直接拒绝(8100 归夜班管线管)
    if mode == "grace" and is_night():
        return {"status": "night", "detail": "夜间(22-08)雷姆在夜班巩固, 不切换模型"}
    # ★压测守卫(2026-09-08): 沙盒压测轮在跑时切 grace = 压测进程内模型(15.5G)+8100 服务
    #   (15.5G)+训练 subprocess(20.7G) 多重驻留 → 20:48 爆内存同款风险。压测期间 8100 保持停机。
    if mode == "grace":
        try:
            _r = subprocess.run(["pgrep", "-f", "stress_engine"], capture_output=True, timeout=5)
            if _r.returncode == 0:
                return {"status": "busy", "detail": "压测轮进行中(单模型铁律), 不切换模型"}
        except Exception:  # noqa: BLE001
            pass
    # 互斥锁（防并发切换；陈旧锁 >10min 视为僵尸）
    if os.path.isfile(LOCK_F):
        try:
            if time.time() - os.path.getmtime(LOCK_F) < 600:
                return {"status": "busy", "detail": "另一个切换正在进行"}
        except OSError:
            pass
    open(LOCK_F, "w").close()
    try:
        t0 = time.time()
        want_id = V61_ID_SUB if mode == "grace" else DEFAULT_ID_SUB
        if mode_of_served(served_model_ids()) == mode:
            _write_state(mode)
            return {"status": "ok", "mode": mode, "detail": "已是目标模式"}
        if mode == "grace":
            # day-model(KeepAlive) 先 bootout, 防Normal自动拉起打架
            subprocess.run(["launchctl", "bootout", "gui/501/com.local-ai-agent.day-model"],
                           capture_output=True, timeout=10)
            _kill_8100()
            logf = open(LOG, "ab")
            subprocess.Popen(["bash", SERVE, "grace", "8100"], stdout=logf, stderr=logf,
                             cwd=REPO, start_new_session=True)
            ok = _wait_ready(V61_ID_SUB)
        else:
            _kill_8100()                          # 停 grace 服务
            if _night_35b_alive():
                # ★夜间 35B 驻留: 绝不 bootstrap day-model(27B+35B 双驻留=panic)。
                #   只杀 v61 清场, 账本记 normal; 8100 由夜班管线按错峰流程自行恢复。
                _write_state("normal")
                return {"status": "ok", "mode": "normal",
                        "detail": "夜间 35B 驻留, 已清 v61, 8100 交还夜班管线"}
            subprocess.run(["launchctl", "bootstrap", "gui/501", DAY_PLIST],
                           capture_output=True, timeout=10)   # day-model 恢复(自管 normal)
            ok = _wait_ready(DEFAULT_ID_SUB)
        _write_state(mode if ok else "unknown")
        return {"status": "ok" if ok else "timeout", "mode": mode if ok else None,
                "elapsed_s": round(time.time() - t0, 1)}
    finally:
        try:
            os.remove(LOCK_F)
        except OSError:
            pass


def _write_state(mode: str) -> None:
    os.makedirs(os.path.dirname(STATE_F), exist_ok=True)
    json.dump({"mode": mode, "switched_at": time.time()},
              open(STATE_F, "w", encoding="utf-8"), indent=1)


def heartbeat() -> dict:
    """/grace 页心跳: 更新活跃时刻; 若当前不是 grace 模式 → 后台切过去。"""
    os.makedirs(os.path.dirname(ACTIVE_F), exist_ok=True)
    json.dump({"ts": time.time()}, open(ACTIVE_F, "w"))
    # ★夜班守卫: 22-08 心跳只记活跃, 不触发切换(8100 归夜班管线)
    if is_night():
        return {"mode": "night", "switching": False,
                "message": "夜班进行中(03:00-08:00), 模型切换暂停; 08:00 后她自动醒来"}
    cur = mode_of_served(served_model_ids())
    st = _read_state()
    if cur != "grace" and st.get("mode") != "switching":
        import threading
        threading.Thread(target=switch, args=("grace",), daemon=True).start()
        return {"mode": cur, "switching": True}
    return {"mode": cur if cur != "down" else st.get("mode", "?"), "switching": False}


def maybe_revert(max_idle_s: int = 300) -> dict:
    """被 proactive_watch 每 10min 调: /grace 心跳超时 → 切回 normal。"""
    st = _read_state()
    if st.get("mode") != "grace":
        return {"status": "skip", "detail": "非 grace 模式"}
    try:
        idle = time.time() - json.load(open(ACTIVE_F))["ts"]
    except Exception:  # noqa: BLE001
        idle = 10 ** 9
    if idle < max_idle_s:
        return {"status": "skip", "detail": f"grace 活跃中({int(idle)}s 前)"}
    # ★夜间对账: 35B 驻留时只清账本不启 27B(单模型铁律); 无 35B 才走正常回切
    if is_night() and _night_35b_alive():
        cur = mode_of_served(served_model_ids())
        if cur == "grace":
            _kill_8100()                          # 清掉挂在 8100 的 v61(35B 独占内存)
        _write_state("normal")
        return {"status": "ok", "detail": "夜间对账: 已清 grace, 8100 交还夜班管线(不启 27B)"}
    return switch("normal")


def _read_state() -> dict:
    try:
        return json.load(open(STATE_F, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else ""
    if m in ("grace", "normal"):
        print(json.dumps(switch(m), ensure_ascii=False))
    else:
        mid = served_model_id()
        print(json.dumps({"served": mid, "mode": mode_of_served(mid), "state": _read_state()},
                         ensure_ascii=False))
