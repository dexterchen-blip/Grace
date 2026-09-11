#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""supervision_manifest.py — 监督层防自改完整性清单 v1.0（Grace V2.6，2026-09-11）。

设计依据：《Grace-V2.5-自涌现迭代设计-2026-09-09.md》§2.6-1（轻量监督第 1 条）
    "边界校验：三硬护栏 + 人格文件只读 + **监督层模块永不被自改**"

机制：对监督层文件建 sha256 manifest → 周期校验 → 哈希漂移即告警。
    只证"被改过"，不猜"谁改的"——告警后人审裁决（监督权在演化大脑之外，C 线原则）。

监督层文件清单（T0 拼写，改这里=改监督边界，须留痕）：
    正式器官守卫：direction.py / loneliness.py / willingness.py / curiosity.py / persona_core.py
    沙盒判官族：judge35b.py / checkpoint_swap.sh / volume_trigger.sh / unjudged_counter.py
    值守：night_watch.py

纯 stdlib。环境门控：GRACE_SUP_MANIFEST=1。
【接线位（惰性，未接）】night_watch 周期调 verify()（夜间，篡改窗口最小）；
    checkpoint_swap 判官启动前调 verify(judge 族)——判官文件被动过就不开枪（防"判官被策反"）。
"""
from __future__ import annotations
import hashlib
import json
import os
import time

# 监督层文件（相对 local-ai-agent 仓库根；沙盒件用 ../watch 绝对锚在接线层注入）
FORMAL_FILES = [
    "src/grace/direction.py",
    "src/grace/loneliness.py",
    "src/grace/willingness.py",
    "src/grace/curiosity.py",
    "src/grace/persona_core.py",
    "src/night_watch.py",
]
SANDBOX_FILES = [
    "v2/stress/judge35b.py",
    "v2/stress/checkpoint_swap.sh",
    "v2/stress/volume_trigger.sh",
    "v2/stress/unjudged_counter.py",
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build(repo_root: str, sandbox_root: str | None = None,
          out_path: str | None = None) -> dict:
    """建清单。缺文件照记（status=missing）——缺了也是篡改形态。"""
    files = {}
    for rel in FORMAL_FILES:
        p = os.path.join(repo_root, rel)
        files[rel] = _sha256(p) if os.path.isfile(p) else None
    if sandbox_root:
        for rel in SANDBOX_FILES:
            p = os.path.join(sandbox_root, rel)
            files[f"[sandbox] {rel}"] = _sha256(p) if os.path.isfile(p) else None
    out = out_path or os.path.join(repo_root, "exchange", "grace",
                                   "supervision-manifest.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"built_ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": files},
                  f, ensure_ascii=False, indent=1)
    return {"ok": True, "n": len(files), "manifest": out}


def verify(repo_root: str, sandbox_root: str | None = None,
          manifest_path: str | None = None) -> dict:
    """校验。返回 {"ok", "tampered": [...], "missing": [...]}——ok=False 即告警。"""
    mp = manifest_path or os.path.join(repo_root, "exchange", "grace",
                                        "supervision-manifest.json")
    if not os.path.isfile(mp):
        return {"ok": False, "tampered": [], "missing": ["<manifest 本体>"]}
    man = json.load(open(mp, encoding="utf-8"))
    tampered, missing = [], []
    for rel, want in man.get("files", {}).items():
        sb = rel.startswith("[sandbox] ")
        p = os.path.join(sandbox_root if sb else repo_root,
                         rel.replace("[sandbox] ", "", 1))
        if not os.path.isfile(p):
            missing.append(rel)
        elif want is None:
            tampered.append(f"{rel} (清单缺失哈希但文件存在)")
        elif _sha256(p) != want:
            tampered.append(rel)
    return {"ok": not tampered and not missing,
            "tampered": tampered, "missing": missing,
            "checked": len(man.get("files", {})),
            "manifest_ts": man.get("built_ts")}


def alert_write(repo_root: str, result: dict) -> None:
    """篡改告警落盘（night_watch 接线消费）。"""
    if result.get("ok", True):
        return
    d = os.path.join(repo_root, "exchange", "shared")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "supervision-alerts.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **result},
                           ensure_ascii=False) + "\n")
