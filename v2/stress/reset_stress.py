#!/usr/bin/env python3
"""压力副本重置(带归档) —— 2026-08-29 用户约定：跑新轮次前，旧产物先移入 archive/ 而非删除。

归档内容（按时间戳目录）:
  adapters/  rem_stress_*（LoRA 成长轨迹）
  l0/        chat.jsonl / official.jsonl / grace_vault.jsonl
  snapshots/ day-*.json 断点
  mood/      mood_states + mood_intraday（SQLite 快照）
  l3/        autobiography.db（如存在）
  datasets/  stress-* 训练数据集
  logs/      stress.log / analysis.md

用法（压力副本内）:
  ./run.sh .venv/bin/python3 v2/stress/reset_stress.py [--keep-log]
"""
from __future__ import annotations
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime

SB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 压力副本根
STRESS = os.path.join(SB, "experiments", "run", "stress")
ADAPTERS = os.path.join(SB, "experiments", "lora", "adapters")
DATASETS = os.path.join(SB, "experiments", "lora", "datasets")
L0 = os.path.join(SB, "memory", "L0_raw")
L2 = os.path.join(SB, "memory", "L2_semantic", "l2.db")
L3 = os.path.join(SB, "memory", "L3_core")


def mv(src: str, dst_dir: str, label: str):
    if os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, os.path.join(dst_dir, os.path.basename(src)))
        print(f"  ✓ {label}: {os.path.basename(src)} → archive/")


def main():
    keep_log = "--keep-log" in sys.argv
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    arch = os.path.join(STRESS, "archive", stamp)
    print(f"=== 压力副本重置（归档到 {arch}）===")

    # adapters: rem_stress_*
    for d in sorted(glob("rem_stress_*", ADAPTERS)):
        os.makedirs(os.path.join(arch, "adapters"), exist_ok=True)
        shutil.move(os.path.join(ADAPTERS, d), os.path.join(arch, "adapters", d))
        print(f"  ✓ adapter: {d}")
    # l0
    for f in ("chat.jsonl", "official.jsonl", "grace_vault.jsonl"):
        mv(os.path.join(L0, f), os.path.join(arch, "l0"), "l0")
    # snapshots
    for f in sorted(glob("day-*.json", STRESS)):
        os.makedirs(os.path.join(arch, "snapshots"), exist_ok=True)
        shutil.move(os.path.join(STRESS, f), os.path.join(arch, "snapshots", os.path.basename(f)))
        print(f"  ✓ snapshot: {os.path.basename(f)}")
    # mood 快照
    if os.path.isfile(L2):
        dst = os.path.join(arch, "mood")
        os.makedirs(dst, exist_ok=True)
        con = sqlite3.connect(L2)
        bak = sqlite3.connect(os.path.join(dst, "l2-mood.db"))
        con.backup(bak)
        bak.close()
        con.execute("DELETE FROM mood_states")
        con.execute("DELETE FROM mood_intraday")
        con.commit()
        con.close()
        print("  ✓ mood: 快照 + 清空")
    # L3
    if os.path.isdir(L3):
        shutil.move(L3, os.path.join(arch, "l3"))
        print("  ✓ l3: autobiography.db")
    # datasets
    for d in sorted(glob("stress-*", DATASETS)):
        os.makedirs(os.path.join(arch, "datasets"), exist_ok=True)
        shutil.move(os.path.join(DATASETS, d), os.path.join(arch, "datasets", os.path.basename(d)))
        print(f"  ✓ dataset: {os.path.basename(d)}")
    # logs
    if not keep_log:
        for f in ("stress.log", "analysis.md", "proactive-live.jsonl"):
            mv(os.path.join(STRESS, f), os.path.join(arch, "logs"), "log")
    print(f"\n✅ 重置完成 → archive/{stamp}/ ｜ 输入 inputs/ 保留")


def glob(pat: str, base: str) -> list[str]:
    import glob as g
    return [os.path.basename(p) for p in g.glob(os.path.join(base, pat))]


if __name__ == "__main__":
    main()
