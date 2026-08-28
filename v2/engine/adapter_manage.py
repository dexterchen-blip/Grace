#!/usr/bin/env python3
"""适配器轮换管理 —— M2-③ 隔天生效 + 24h 反悔窗口（Grace_v2 设计 §6/§4）。

人审驯服铁律：LoRA 训练产出后**隔天生效**，留 24 小时反悔窗口。
本模块管理适配器版本：
  - 训练产物：experiments/lora/adapters/rem_v1_YYYYMMDD/（带日期）
  - 生效记录：experiments/lora/active.json  {persona: {active: dir, history: [{dir, promoted_at}]}}
  - promote：把某日期适配器设为生效（记录 promoted_at）
  - rollback：回滚到 history 中上一个生效版本（24h 内反悔）
  - list：当前版本 + 历史

用法（沙盒内）:
  ./run.sh python3 v2/engine/adapter_manage.py list
  ./run.sh python3 v2/engine/adapter_manage.py promote rem_v1_20260827
  ./run.sh python3 v2/engine/adapter_manage.py rollback
"""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
import config  # noqa: E402

ACTIVE_FILE = os.path.join(config.ADAPTERS, "active.json")


def _load_active() -> dict:
    if os.path.isfile(ACTIVE_FILE):
        with open(ACTIVE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"personas": {}}


def _save_active(d: dict) -> None:
    os.makedirs(config.ADAPTERS, exist_ok=True)
    with open(ACTIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def list_adapters(persona: str = None) -> dict:
    """列出所有适配器目录 + 当前生效版本。"""
    persona = persona or config.PERSONA["name"]
    dirs = []
    if os.path.isdir(config.ADAPTERS):
        for name in sorted(os.listdir(config.ADAPTERS)):
            p = os.path.join(config.ADAPTERS, name)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "adapters.safetensors")):
                dirs.append({"name": name,
                             "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))})
    active = _load_active().get("personas", {}).get(persona, {})
    return {"persona": persona, "adapters": dirs,
            "active": active.get("active"), "history": active.get("history", [])}


def promote(name: str, persona: str = None, decided_by: str = "user") -> bool:
    """把 name 适配器设为生效（记录 promoted_at = 生效时刻，即隔天生效起点）。"""
    persona = persona or config.PERSONA["name"]
    p = os.path.join(config.ADAPTERS, name)
    if not os.path.isfile(os.path.join(p, "adapters.safetensors")):
        print(f"[adapter] ❌ {name} 无适配器文件")
        return False
    d = _load_active()
    st = d.setdefault("personas", {}).setdefault(persona, {"active": None, "history": []})
    if st["active"] and st["active"] != name:
        st["history"].insert(0, {"dir": st["active"], "promoted_at": st.get("promoted_at")})
    st["active"] = name
    st["promoted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    st["decided_by"] = decided_by
    _save_active(d)
    print(f"[adapter] ✅ {persona} 生效版本 → {name}（{st['promoted_at']}，by {decided_by}）")
    return True


def rollback(persona: str = None) -> bool:
    """回滚到上一个生效版本（24h 反悔窗口）。"""
    persona = persona or config.PERSONA["name"]
    d = _load_active()
    st = d.get("personas", {}).get(persona)
    if not st or not st["history"]:
        print(f"[adapter] ❌ {persona} 无历史版本可回滚")
        return False
    prev = st["history"].pop(0)
    st["active"] = prev["dir"]
    st["promoted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    st["decided_by"] = "rollback"
    _save_active(d)
    print(f"[adapter] ↩️ {persona} 已回滚 → {prev['dir']}")
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        print(json.dumps(list_adapters(), ensure_ascii=False, indent=2))
    elif cmd == "promote" and len(sys.argv) > 2:
        promote(sys.argv[2])
    elif cmd == "rollback":
        rollback()
    else:
        print(__doc__)
