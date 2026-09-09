#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""instinct_distill.py — S1 本能蒸馏管线骨架（Grace V2.5，2026-09-09）。

目标：把"器官中介轨迹"蒸馏成模型本能（R-SFT，AgentArk 策略1）。
三段式：
  ① collect —— 运行时轨迹日志（grace_chat 写 trajectory-journal.jsonl，本模块提供读取/过滤）
  ② build   —— 轨迹 → SFT 数据集（messages 格式，器官标注作 metadata）
  ③ train   —— 夜班窗口调用 mlx_lm.lora（包装器，dry-run 默认）

数据规矩（铁律兼容）：只收真实运行轨迹（formal /grace 交互），压测/合成数据不进。
"""
from __future__ import annotations
import argparse
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOURNAL = os.path.join(REPO, "exchange", "grace", "trajectory-journal.jsonl")
OUT_DIR = os.path.join(REPO, "experiments", "instinct")


def collect_stats(journal: str = JOURNAL) -> dict:
    """轨迹库存盘点：S1 蒸馏的就绪度仪表。"""
    n, with_wm, with_anns = 0, 0, 0
    if os.path.isfile(journal):
        for l in open(journal, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            n += 1
            if r.get("wm_entries"):
                with_wm += 1
            if r.get("annotations"):
                with_anns += 1
    return {"trajectories": n, "with_wm": with_wm, "with_annotations": with_anns,
            "s1_ready": n >= 1000,          # R-SFT 最低门槛（策略级 SFT 经验值）
            "note": "s1_ready 阈值 1000 条为 v1 估计，实测后调整"}


def build_dataset(out_dir: str = OUT_DIR, min_len: int = 4, journal: str = JOURNAL) -> dict:
    """轨迹 → SFT 数据集。

    每条轨迹（grace_chat 写入）结构：
      {"ts", "user_text", "reply", "wm_entries": [{t,tag,nov}], "inner_world",
       "annotations": [...], "l2_hits": ...}
    SFT messages：system=当时注入的 T0+T1 重建，user=user_text，assistant=reply。
    只收 reply 非空、回声防线通过的轨迹（collect 端已保证）。
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"instinct-{time.strftime('%Y%m%d')}.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for l in open(journal, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if not r.get("reply") or len(r.get("user_text", "")) < min_len:
                continue
            wm = r.get("wm_entries") or []
            wm_txt = "\n".join(
                f"· [{e.get('tag','')}] {e.get('t','')} (nov {e.get('nov', 0):.2f})"
                if isinstance(e, dict) else f"· {e}" for e in wm[-12:])
            sys_prompt = (f"你是雷姆。主人名叫陈泽。\n\n"
                          f"（今天·工作记忆）\n{wm_txt}\n"
                          + (f"（注意力提示）\n" + "\n".join(r["annotations"]) + "\n"
                             if r.get("annotations") else ""))
            f.write(json.dumps({"messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": r["user_text"]},
                {"role": "assistant", "content": r["reply"]}],
                "meta": {"ts": r.get("ts"), "source": "organ-trajectory"}},
                ensure_ascii=False) + "\n")
            n += 1
    return {"dataset": out_path, "samples": n}


def train(dataset: str, dry: bool = True) -> dict:
    """夜班窗口训练包装器（默认 dry-run——实跑需用户确认+夜班窗）。"""
    cmd = ["python3", "-m", "mlx_lm", "lora",
           "--model", "mlx-community/Qwen3.8-27B-4bit",
           "--train", "--data", dataset,
           "--batch-size", "1", "--iters", "150",
           "--num-layers", "16", "--grad-checkpoint",
           "--learning-rate", "1e-5"]
    return {"cmd": " ".join(cmd), "dry": dry,
            "note": "实跑前确认：①夜班窗（22:00 后）②35B 未驻留 ③数据 ≥1000 条"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "build", "train"])
    ap.add_argument("--dataset")
    ap.add_argument("--dry", action="store_true", default=True)
    a = ap.parse_args()
    if a.cmd == "stats":
        print(json.dumps(collect_stats(), ensure_ascii=False, indent=1))
    elif a.cmd == "build":
        print(json.dumps(build_dataset(), ensure_ascii=False, indent=1))
    elif a.cmd == "train":
        ds = a.dataset or os.path.join(OUT_DIR, f"instinct-{time.strftime('%Y%m%d')}.jsonl")
        print(json.dumps(train(ds, dry=a.dry), ensure_ascii=False, indent=1))
