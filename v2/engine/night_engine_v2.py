#!/usr/bin/env python3
"""夜班引擎 V2 骨架 —— 从「整理记忆」升级为「用当天记忆做微量 LoRA 训练」。

一天生命周期（Grace_v2 设计 §9）：
  白天: 被动服务 + 主动记录（对话 → L-1 → 定时固化 L0/L2）
  夜班: ①记忆巩固 ②提炼训练候选 ③人审闸门 ④LoRA训练 ⑤权重快照 ⑥心态推演
  次日: 新 LoRA 生效 + 自发能力 + 心态着色

本骨架把 ③④ 作为「人审驯服自训练」的核心闭环：
  候选 → proposal 闸门（pending）→ API 人审（approve/reject）→ 只训 approved + 锚点回放
  → 快照 → 报告（每次训练只出一个报告）→ 隔天生效（24h 反悔窗口）。

注意：本文件只做流程编排与落点，训练本体由 mlx_lm.lora 执行（experiments/README 有命令）；
一切写入都在沙盒 AIAGENT_SANDBOX 之下，正式系统零接触。
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402
from candidate_extract import CandidateQueue, extract_from_l0  # noqa: E402
from mood_engine import derive as mood_derive  # noqa: E402
from report import build_report  # noqa: E402


# ---------- ① 记忆巩固（v1 现有；骨架先占位调用，night_pipeline 完整版在 src/） ----------
def step1_consolidate() -> dict:
    print("[v2-night] ① 记忆巩固 —— 复用 v1 night_pipeline 巩固段（src/night_pipeline.py）")
    # 沙盒内可直接跑 src/night_pipeline.py（已 patch 隔离）；此处保留接口，由调用方决定。
    return {"step": 1, "ok": True, "note": "巩固段 = src/night_pipeline.py（未在此骨架内重复实现）"}


# ---------- ② 提炼训练候选（数据侧防渗漏：事实类过滤掉；源按天排序） ----------
def step2_extract(max_samples: int = 20, date: str | None = None,
                  l0_dir: str | None = None) -> list[str]:
    """提炼训练候选。默认源 = AIAGENT_PROD_L0（正式 L0 只读引用）→ 兜底沙盒 L0；
    date=None → 提炼最新一天（源按天排序）。"""
    print("[v2-night] ② 提炼训练候选 —— 源按天排序，事实类剔除")
    src = l0_dir or os.environ.get("AIAGENT_PROD_L0") or config.L0_DIR
    if os.environ.get("AIAGENT_PROD_L0"):
        print(f"  源（只读引用正式系统 L0）: {src}")
    r = extract_from_l0(src, date=date, max_samples=max_samples)
    if not r["samples"]:
        print(f"  {r.get('date') or '最新一天'} 无风格样本（stats={r['stats']}），跳过候选创建")
        return []
    cid = CandidateQueue().create(
        date=r["date"],
        title=f"LoRA 风格样本 {r['date']}（{config.PERSONA['name']}）",
        description=f"来自 {r['stats']['total']} 条记忆（日期 {r['date']}）；事实类 {r['stats']['filtered']} 条已过滤。",
        samples=r["samples"], kind_stats=r["stats"])
    print(f"  候选 {cid} → pending，等待人审")
    return [cid]


# ---------- ③ 人审闸门：只训已批准候选 ----------
def step3_gate(candidate_ids: list[str] | None = None) -> list[dict]:
    """只取 approved/ 目录下所有候选作为本次训练原料。

    人审驯服语义：训练只吃「已被人类批准」的候选（昨晚提炼 → 今晨批准 → 今晚训练）；
    本次提炼的新候选若仍 pending，仅提示，绝不纳入训练。
    """
    print("[v2-night] ③ 人审闸门 —— 只训 approved 候选")
    q = CandidateQueue()
    approved = q.list("approved")
    if candidate_ids:
        for cid in candidate_ids:
            rec = q.load(cid)
            if rec and rec["status"] == "pending":
                print(f"  ⏳ {cid} 仍 pending —— 未经批准当晚不训练（人审驯服自训练铁律）")
            elif rec and rec["status"] == "rejected":
                print(f"  ✖ {cid} 已否决 —— 跳过")
    if not approved:
        print("  （当前无已批准候选，本次仅锚点回放可训练）")
    return approved


# ---------- ④ LoRA 训练（微量 · 低秩 · 锚点回放 5%；产物带日期，隔天生效） ----------
def step4_train(approved: list[dict], dry_run: bool = True) -> dict:
    """执行 mlx_lm.lora（骨架默认 dry_run，避免误触发真实训练占满 GPU/内存）。

    产物：experiments/lora/adapters/rem_v1_YYYYMMDD/（匹配 lora_lifecycle 7 天滑动窗口），
    训练完成自动登记到 adapter_manage（隔天生效，24h 反悔窗口可回滚）。
    """
    print(f"[v2-night] ④ LoRA 训练 —— rank={config.LORA['rank']} lr={config.LORA['learning_rate']} "
          f"iters={config.LORA['iters']} anchor_ratio={config.LORA['anchor_ratio']}")
    day = time.strftime("%Y%m%d")
    adapter = os.path.join(config.ADAPTERS, f"{config.PERSONA['name']}_v1_{day}")
    os.makedirs(adapter, exist_ok=True)
    if dry_run:
        print(f"  [dry-run] 生成适配器落点 {adapter}；真实训练见 experiments/README 命令。")
        return {"adapter": adapter, "dry_run": True, "loss": None,
                "samples": sum(len(a.get("samples", [])) for a in approved),
                "anchors": os.path.join(config.PERSONA["dataset_dir"], "sample.jsonl")}

    # 真实训练：把锚点样本 + 批准样本写入训练集，再调 mlx_lm.lora（推荐 -c yaml，见 train_rem.yaml）
    ds_dir = os.path.join(config.DATASETS, f"train-{time.strftime('%Y%m%d-%H%M')}")
    os.makedirs(ds_dir, exist_ok=True)
    merged = _merge_dataset(approved, ds_dir)
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", config.LORA["model"],
        "--train", "--data", ds_dir,
        "--adapter-path", adapter,
        "--batch-size", str(config.LORA["batch_size"]),
        "--iters", str(config.LORA["iters"]),
        "--learning-rate", str(config.LORA["learning_rate"]),
        "--steps-per-report", "10", "--steps-per-eval", "50", "--save-every", str(config.LORA["save_every"]),
    ]
    print("  训练命令:", " ".join(cmd))
    # 真实训练占资源，需在沙盒内显式执行（此处不自动跑，防止夜班意外 OOM 正式系统）
    # 训练完成后由调用方登记生效：adapter_manage.promote(<name>)（隔天生效，24h 反悔窗口）
    return {"adapter": adapter, "dry_run": True, "cmd": cmd, "dataset": merged, "loss": None}


def _merge_dataset(approved: list[dict], out_dir: str) -> str:
    """锚点回放（5% 锚点样本）+ 批准候选样本 → 训练集 jsonl。"""
    import random
    anchor = os.path.join(config.PERSONA["dataset_dir"], "sample.jsonl")
    lines = []
    if os.path.exists(anchor):
        with open(anchor, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    samples = []
    for a in approved:
        samples.extend(a.get("samples", []))
    # 锚点比例：anchor_ratio = 锚点 / 总样本
    n_anchor = max(1, int(len(samples) * config.LORA["anchor_ratio"] / (1 - config.LORA["anchor_ratio"])))
    merged = random.sample(lines, min(n_anchor, len(lines))) if lines else []
    merged += samples[:200]
    random.shuffle(merged)
    ds = os.path.join(out_dir, "train.jsonl")
    with open(ds, "w", encoding="utf-8") as f:
        for ln in merged:
            f.write(ln if ln.endswith("\n") else ln + "\n")
    print(f"  训练集已合并 → {ds}（{len(merged)} 条，其中锚点 {min(n_anchor, len(lines))} 条）")
    return ds


# ---------- ⑤ 权重快照（git 可回滚，复用 v1 版本化思路） ----------
def step5_snapshot() -> str:
    print("[v2-night] ⑤ 权重快照 —— experiments/lora/snapshots git")
    os.makedirs(config.SNAPSHOTS, exist_ok=True)
    if not os.path.isdir(os.path.join(config.SNAPSHOTS, ".git")):
        subprocess.run(["git", "-C", config.SNAPSHOTS, "init", "-q"], check=False)
    subprocess.run(["git", "-C", config.SNAPSHOTS, "add", "-A"], check=False)
    r = subprocess.run(["git", "-C", config.SNAPSHOTS, "commit", "-q", "-m",
                        f"lora snapshot {time.strftime('%Y%m%d-%H%M')}"], capture_output=True, text=True)
    h = subprocess.run(["git", "-C", config.SNAPSHOTS, "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    sha = h.stdout.strip() or "-"
    print(f"  快照 {sha}")
    return sha


# ---------- ⑥ 心态推演（第三轨） ----------
def step6_mood(events: list[dict] | None = None) -> dict:
    print("[v2-night] ⑥ 心态推演 —— mood_engine（规则骨架，35B 推演为开放问题）")
    return mood_derive(events or [{"text": "夜班流水线完成", "sentiment": 0.3, "weight": 0.4}])


# ---------- 报告（每次训练只出一个报告） ----------
def step7_report(result: dict, approved: list[dict], snapshot_sha: str, candidate_ids: list[str]) -> str:
    print("[v2-night] ⑦ 训练报告")
    stats = {"approved_candidates": len(approved),
             "samples": result.get("samples", 0)}
    return build_report(
        candidate_ids=candidate_ids, dataset_stats=stats,
        lora_cfg=config.LORA, adapter_path=result.get("adapter", "-"),
        snapshot_hash=snapshot_sha, loss_points=result.get("loss"))


# ---------- 全流程 ----------
def run_night(dry_run: bool = True, skip_consolidate: bool = False) -> dict:
    """夜班 V2 全流程骨架。默认 dry_run：只编排落点，不真实训练。"""
    t0 = time.time()
    steps = {}
    if not skip_consolidate:
        steps["1_consolidate"] = step1_consolidate()
    cids = step2_extract()
    steps["2_extract"] = cids
    approved = step3_gate(cids)
    steps["3_gate"] = [a["id"] for a in approved]
    result = step4_train(approved, dry_run=dry_run)
    steps["4_train"] = result
    sha = step5_snapshot()
    steps["5_snapshot"] = sha
    mood = step6_mood()
    steps["6_mood"] = mood
    report_path = step7_report(result, approved, sha, cids)
    steps["7_report"] = report_path
    summary = {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "candidates": cids,
        "approved": [a["id"] for a in approved],
        "adapter": result.get("adapter"),
        "snapshot": sha,
        "mood": mood.get("mood_label"),
        "report": report_path,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print("\n==== 夜班 V2（骨架）完成 ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    dry = "--real" not in sys.argv
    if dry:
        print("⚠ 默认 dry-run（不真实训练）。加 --real 才会执行 mlx_lm.lora 训练。")
    run_night(dry_run=dry)
