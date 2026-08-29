#!/usr/bin/env python3
"""Grace V2 三轨框架配置。

三轨分工铁律（见 Grace_v2_融合设计.md §2）：
  外挂轨(慢)  管事实/真相/可审计 → memory/ L0·L2·L3        —— 谁改: 人审后才写
  权重轨(极慢) 管风格/语气/人格底色 → experiments/lora/     —— 谁改: 夜班训练 + 人审
  心态轨(日级) 管当日心情着色 → l2.db mood_states 表        —— 谁改: 夜班推演 + 可覆盖

本模块只做路径/常量解析，不产生任何副作用；所有路径在 AIAGENT_SANDBOX 之下，
运行入口必须经过沙盒 run.sh（env -i 白名单），保证零污染正式系统。
"""
from __future__ import annotations
import os

# ---------- 根路径（run.sh 注入 AIAGENT_SANDBOX；外部直接 import 时兜底解析） ----------
SB = os.environ.get("AIAGENT_SANDBOX", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(SB, "src")                      # 运行时基线（local-ai-agent 快照，已隔离 patch）
MEMORY = os.path.join(SB, "memory")                # 四层记忆（沙盒内全新空结构）
EXCHANGE = os.path.join(SB, "exchange")            # 提案/信箱（沙盒内）
PROPOSALS = os.path.join(EXCHANGE, "proposals")    # 人审闸门落点（pending/approved/rejected/...）
EXPERIMENTS = os.path.join(SB, "experiments")      # 三轨实验落点
LORA_ROOT = os.path.join(EXPERIMENTS, "lora")
REPORTS = os.path.join(EXPERIMENTS, "run")         # 每次训练只出一个报告 → 落这里
MOOD_ROOT = os.path.join(EXPERIMENTS, "mood")

# ---------- 三轨存储 ----------
L0_DIR = os.path.join(MEMORY, "L0_raw")
L2_DB = os.path.join(MEMORY, "L2_semantic", "l2.db")
L3_FILE = os.path.join(MEMORY, "L3_core", "core.md")
DATASETS = os.path.join(LORA_ROOT, "datasets")
ADAPTERS = os.path.join(LORA_ROOT, "adapters")
SNAPSHOTS = os.path.join(LORA_ROOT, "snapshots")

# ---------- 人审驯服自训练 API ----------
API_HOST = "127.0.0.1"
API_PORT = 18300                     # 沙箱专用端口段（18000+），永不碰正式 3091/8100/8200

# ---------- 权重轨（LoRA）默认超参（Grace_v2 设计 §3/§4） ----------
LORA = {
    "model": "mlx-community/Qwen3.8-27B-4bit",   # 沙箱 HF_HOME 只读引用宿主 27B；小模型实验用 setup_model.sh
    "rank": 8,                      # 2026-08-30 对齐 rem_v5 定稿(mlx 默认 rank8/scale20)
    "scale": 20,
    "learning_rate": 1e-5,          # 冷启动 lr；增量续训用 incr_lr=1e-6(15 iters)
    "incr_lr": 1e-6,
    "iters": 150,                   # 冷启动上限；实际动态: 冷启动 clamp(样本×20,60,150)/增量 15
    "batch_size": 1,                # 48GB 硬约束
    "anchor_ratio": 0.05,           # 锚点回放 5%（抗灾难性遗忘）
    "grad_checkpoint": True,        # gradient checkpointing（省显存）
    "save_every": 50,
}
# LoRA 累积管理：7 天滑动窗口 + 周 merge + 月全量重训 + git 快照
LORA_LIFECYCLE = {"daily_keep": 7, "weekly_merge": True, "monthly_full_retrain": True,
                  "snapshot_before_train": True}

# ---------- 心态轨默认参数（Grace_v2 设计 §8.4） ----------
MOOD = {
    "decay": 0.7,                   # 心态 = 昨日 × 0.7 + 今日增量 × 0.3（防"精神分裂"）
    "labels": ["平静", "轻微兴奋", "兴奋", "低落", "焦虑", "专注"],  # 固定词表（开放问题 4 的暂定解）
    "default_intensity": 0.5,
}

# ---------- 权重轨默认人格：雷姆（Re:Zero） ----------
PERSONA = {
    "name": "rem",
    "display": "雷姆（Re:Zero 罗兹瓦尔宅邸女仆）",
    "mood_baseline": 0.5,              # 人格情绪底色（LoRA 驱动）：雷姆外冷内热、中性偏内敛
    "anchor_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "rem.md"),
    "dataset_dir": os.path.join(DATASETS, "rem"),
    "adapter_dir": os.path.join(ADAPTERS, "rem_v1"),
}

# ---------- 训练候选（人审闸门）类型 ----------
TRAIN_CANDIDATE_TYPE = "lora_train"          # 独立于 L3 记忆提案的"训练用样本集"提案
CANDIDATE_STATUSES = ("pending", "approved", "rejected", "expired")


def ensure_dirs() -> None:
    """建齐 V2 需要的目录（幂等）。"""
    for d in (REPORTS, MOOD_ROOT, DATASETS, ADAPTERS, SNAPSHOTS,
              os.path.join(PROPOSALS, "pending"), os.path.join(PROPOSALS, "approved"),
              os.path.join(PROPOSALS, "rejected")):
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"SB         = {SB}")
    print(f"API        = {API_HOST}:{API_PORT}")
    print(f"LORA       = {LORA}")
    print(f"PERSONA    = {PERSONA}")
    print(f"PROPOSALS  = {PROPOSALS}")
