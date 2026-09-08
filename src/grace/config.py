#!/usr/bin/env python3
"""Grace V2.3 引擎 —— 正式运行空间配置（2026-09-07 迁移自压测沙盒 v2/config.py）。

★ 本文件是引擎的"正式系统身份证"：与沙盒版唯一差异 = 路径解析。
  沙盒版: SB = AIAGENT_SANDBOX(沙盒根, run.sh env -i 注入), 所有记忆/产物隔离在沙盒内。
  正式版: SB = local-ai-agent 仓库根(按 __file__ 解析), 记忆=正式 memory/, 产物=exchange/grace/。

三轨分工铁律（迁移不改）:
  外挂轨(慢)  管事实 → memory/ L0·L2·L3        —— 谁改: 夜班 segment5 引擎写(图谱/L3)
  权重轨(极慢) 管风格/人格 → (正式暂不训练, 训练走人审提案——见集成方案)
  心态轨(日级) 管当日心情 → l2.db mood_states   —— 谁改: 夜班 segment5 引擎写
"""
from __future__ import annotations
import os

# ---------- 根路径（正式系统：src/grace/config.py → 仓库根；不读 AIAGENT_SANDBOX） ----------
SB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # local-ai-agent/
SRC = os.path.join(SB, "src")                       # 正式系统源码
MEMORY = os.path.join(SB, "memory")                 # ★ 正式四层记忆（与 m4/夜班共用的库）
EXCHANGE = os.path.join(SB, "exchange")             # 正式提案/信箱
PROPOSALS = os.path.join(EXCHANGE, "proposals")     # 人审闸门（正式现有）
# ★ 引擎运行账本落点（正式产物区；引擎模块按 EXPERIMENTS/run/stress/... 读 PE——零改动兼容）
EXPERIMENTS = os.path.join(EXCHANGE, "grace")
LORA_ROOT = os.path.join(EXPERIMENTS, "lora")
REPORTS = os.path.join(EXPERIMENTS, "run")
MOOD_ROOT = os.path.join(EXPERIMENTS, "mood")

# ---------- 三轨存储（全部指向正式库） ----------
L0_DIR = os.path.join(MEMORY, "L0_raw")
L2_DB = os.path.join(MEMORY, "L2_semantic", "l2.db")     # ★ 正式生产库（docs/entities/... 共存, 引擎建 mood_graph/mood_states 表）
L3_FILE = os.path.join(MEMORY, "L3_core", "core.md")
DATASETS = os.path.join(LORA_ROOT, "datasets")
ADAPTERS = os.path.join(LORA_ROOT, "adapters")
SNAPSHOTS = os.path.join(LORA_ROOT, "snapshots")

# ---------- 模型服务（正式：day-model :8100, D-4 选项 A；夜班 segment5 bootstrap 后经 HTTP 调用） ----------
MODEL_HTTP = "http://127.0.0.1:8100/v1/chat/completions"
MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"

# ---------- 权重轨（正式暂不训练；超参保留与沙盒一致，供未来人审后启用） ----------
LORA = {
    "model": MODEL_ID,
    "rank": 8, "scale": 20,
    "learning_rate": 1e-5, "incr_lr": 1e-6,
    "iters": 150, "batch_size": 1,
    "anchor_ratio": 0.05, "grad_checkpoint": True, "save_every": 50,
}
LORA_LIFECYCLE = {"daily_keep": 7, "weekly_merge": True, "monthly_full_retrain": True,
                  "snapshot_before_train": True}

MOOD = {
    "decay": 0.7,
    "labels": ["平静", "轻微兴奋", "兴奋", "低落", "焦虑", "专注"],
    "default_intensity": 0.5,
}

PERSONA = {
    "name": "rem",
    "display": "雷姆（Re:Zero 罗兹瓦尔宅邸女仆）",
    "mood_baseline": 0.56,             # 原作校准（9/3 X 轮）
    "anchor_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "rem.md"),
    "dataset_dir": os.path.join(DATASETS, "rem"),
    "adapter_dir": os.path.join(ADAPTERS, "rem_v1"),
}

TRAIN_CANDIDATE_TYPE = "lora_train"
CANDIDATE_STATUSES = ("pending", "approved", "rejected", "expired")

if __name__ == "__main__":
    print(f"SB        = {SB}")
    print(f"L2_DB     = {L2_DB}")
    print(f"EXPERIMENTS = {EXPERIMENTS}")
    print(f"PERSONA   = {PERSONA['name']} (anchor: {os.path.basename(PERSONA['anchor_file'])})")
