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
sys.path.insert(0, os.path.join(SB, "v2", "engine"))
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
        # ★2026-09-02 W轮排查修复: 漏清 mood_graph —— 它和 mood_states 同库但从未被 reset 清过,
        #   每轮 +7000 边全累积(K+V2+W = 17012 边, ts 跨 89 天):
        #   ①ToM 读情绪史/联结价值/图谱训练样本(extract_graph_samples)被旧轮污染
        #   ②event_id 每轮重复 ev-d{day}-{i}, 再巩固 UPDATE 会误改旧轮同 id 边
        con.execute("DELETE FROM mood_graph")
        for _t in ("entities", "relations", "doc_entities", "docs", "meta_kv"):
            try:
                con.execute(f"DELETE FROM {_t}")
            except Exception:  # noqa: BLE001
                pass
        try:
            con.execute("DELETE FROM sqlite_sequence")
        except Exception:  # noqa: BLE001
            pass
        con.commit()
        con.close()
        print("  ✓ mood: 快照 + 清空(mood_states/intraday/**mood_graph**)")
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
    # ★2026-09-03 修复(--keep-log 语义反了): 原 `if not keep_log:` 一档包住全部 6 个文件 →
    #   带 --keep-log 的轮次反而【保留】4 个 append 累积的 live 文件(净轮实锤污染:
    #   prediction-errors 778 条里 day1 有 75 条(正常 20-30)、8 个 day 超标;
    #   训练集吃旧轮 proactive「雷姆会准备好的」×33; proactive 159 条含 9 个旧轮重复组)。
    #   正确语义: --keep-log = 只保留 stress.log/analysis.md 供排查;
    #   4 个 append 累积文件【无论如何都归档】(否则新轮数据不可单独归因)。
    for f in ("proactive-live.jsonl", "feedback-live.jsonl",
              "cognition-live.jsonl", "prediction-errors.jsonl"):
        mv(os.path.join(STRESS, f), os.path.join(arch, "logs"), "log(强制归档)")
    if not keep_log:
        for f in ("stress.log", "analysis.md"):
            mv(os.path.join(STRESS, f), os.path.join(arch, "logs"), "log")
    # ★2026-09-03 存储治理: 轮次元数据归档 + 权重 registry 保留(live) + 中间 adapter 清理建议
    import os as _os
    _rm = _os.path.join(STRESS, "round-meta.json")
    if _os.path.isfile(_rm):
        mv(_rm, _os.path.join(arch, "logs"), "round-meta(轮次元数据归档)")
    try:
        sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "engine"))
        import storage as _st
        _stale = _st.stale_files()
        for s in _stale:
            print(f"  ⚠ 僵尸存储: {s['file']} ({s['issue']}) → {s['action']}")
        _rep = _st.prune_middle_adapters()
        _tot = sum(len(v.get("to_delete", [])) for v in _rep.values())
        if _tot:
            print(f"  [storage] 中间 adapter 建议清理 {_tot} 个(保留轮末/断点权重, registry 在 lora/weights-registry.json)")
        else:
            print("  [storage] 权重 registry 现无中间 adapter 待清")
    except Exception as _se:  # noqa: BLE001
        print(f"  [storage] 清理建议失败(不影响重置): {_se}")
    print(f"\n✅ 重置完成 → archive/{stamp}/ ｜ 输入 inputs/ 保留")


def glob(pat: str, base: str) -> list[str]:
    import glob as g
    return [os.path.basename(p) for p in g.glob(os.path.join(base, pat))]


if __name__ == "__main__":
    main()
