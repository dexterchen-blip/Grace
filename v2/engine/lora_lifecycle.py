#!/usr/bin/env python3
"""LoRA 累积管理 —— M2-④ 7 天滑动窗口 + 周 merge + 月全量重训（Grace_v2 设计 §4）。

策略：
  1. 保留最近 7 天每日 LoRA（推理时全部叠加/或按 active 生效）
  2. 每周日 merge 成「周 LoRA」，清每日（保留 snapshots git 可回滚）
  3. 每月一次全量重训提醒
  4. 每次训练前 git 快照权重（night_engine_v2 step5 已做）

用法（沙盒内）:
  ./run.sh python3 v2/engine/lora_lifecycle.py prune         # 清理 7 天前的每日 adapter
  ./run.sh python3 v2/engine/lora_lifecycle.py weekly_merge  # 周 merge（调 mlx_lm.fuse，需停 8100 错峰）
  ./run.sh python3 v2/engine/lora_lifecycle.py status        # 查看累积状态
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
import config  # noqa: E402

DAILY_KEEP = 7            # 7 天滑动窗口
WEEKLY_MERGE = True       # 周 merge 启用
MONTHLY_FULL_RETRAIN = True


def _daily_adapters(persona: str = None) -> list[dict]:
    """按日期命名的适配器目录 rem_v1_YYYYMMDD（每日训练产物）。"""
    persona = persona or config.PERSONA["name"]
    pat = re.compile(rf"^{persona}_v1_(\d{{8}})$")
    out = []
    if os.path.isdir(config.ADAPTERS):
        for name in os.listdir(config.ADAPTERS):
            m = pat.match(name)
            if m and os.path.isfile(os.path.join(config.ADAPTERS, name, "adapters.safetensors")):
                out.append({"name": name, "date": m.group(1),
                            "mtime": os.path.getmtime(os.path.join(config.ADAPTERS, name))})
    return sorted(out, key=lambda x: x["date"])


def status() -> dict:
    dailies = _daily_adapters()
    return {"persona": config.PERSONA["name"],
            "daily_keep": DAILY_KEEP,
            "dailies": [{"name": d["name"], "date": d["date"],
                         "age_days": (time.time() - d["mtime"]) / 86400} for d in dailies],
            "weekly_merge": WEEKLY_MERGE, "monthly_full_retrain": MONTHLY_FULL_RETRAIN}


def prune(dry_run: bool = True) -> list[str]:
    """删除 7 天前的每日 adapter 目录（快照 git 中仍有，可回滚）。"""
    dailies = _daily_adapters()
    cutoff = time.time() - DAILY_KEEP * 86400
    doomed = [d for d in dailies if d["mtime"] < cutoff]
    for d in doomed:
        if dry_run:
            print(f"[lifecycle] (dry) 将删除 {d['name']}（{time.strftime('%m-%d', time.localtime(d['mtime']))}）")
        else:
            import shutil
            shutil.rmtree(os.path.join(config.ADAPTERS, d["name"]), ignore_errors=True)
            print(f"[lifecycle] 已删除 {d['name']}")
    return [d["name"] for d in doomed]


def weekly_merge() -> str | None:
    """把最近 7 天每日 adapter merge 成周 LoRA（mlx_lm.fuse）。

    注意：fuse 需要加载 27B + 所有 adapter，必须与 8100 day-model 错峰（停→merge→恢复）。
    """
    dailies = _daily_adapters()[-DAILY_KEEP:]
    if len(dailies) < 2:
        print(f"[lifecycle] 不足 2 个每日 adapter（{len(dailies)}），跳过周 merge")
        return None
    if any(os.path.isfile(p) for p in
           [f"/tmp/fuse{time.strftime('%Y%m%d')}"]) or True:
        pass
    week_dir = os.path.join(config.ADAPTERS, f"{config.PERSONA['name']}_weekly_{time.strftime('%Y%m%d')}")
    os.makedirs(week_dir, exist_ok=True)
    # mlx_lm.fuse 需要 adapter 列表 + 输出目录；权重文件逐个拷入（fuse 语义见设计文档开放问题 2）
    import shutil
    for i, d in enumerate(dailies):
        src = os.path.join(config.ADAPTERS, d["name"], "adapters.safetensors")
        dst = os.path.join(week_dir, f"daily_{i+1}_{d['date']}.safetensors")
        shutil.copy(src, dst)
    print(f"[lifecycle] 周 merge（v1：聚合拷贝 + 记录）→ {week_dir}")
    print("  ⚠ 真正的权重 merge（mlx_lm.fuse --adapter-path ...）是开放问题 2（merge 失真控制），"
          "当前为聚合快照，供推理时多 adapter 叠加或后续 fuse。")
    return week_dir


# ---------- M5-② 训练健康检查 + 自动回滚 ----------
HEALTH_OK = "ok"
HEALTH_OVERFIT = "overfit"
HEALTH_BAD = "bad"


def check_training_health(log_text: str) -> str:
    """从训练日志判断健康度（M5-②，防漂移/过拟合固化）。

    规则：
      val 发散（最后 val > 2.0 或持续上升）→ bad
      过拟合（train < 0.15 且 val > 1.0，train/val 差距巨大）→ overfit（rem_v1 症状）
      否则 → ok
    """
    import re as _re
    vals = [float(m) for m in _re.findall(r"Val loss ([\d.]+)", log_text)]
    trains = [float(m) for m in _re.findall(r"Train loss ([\d.]+)", log_text)]
    if not vals:
        return HEALTH_BAD
    val_final = vals[-1]
    val_first = vals[0]
    train_final = trains[-1] if trains else 1.0
    if val_final > 2.0 or val_final > val_first + 1.0:
        return HEALTH_BAD                                   # 发散
    if train_final < 0.15 and val_final > 1.0:
        return HEALTH_OVERFIT                               # 过拟合（复读机风险）
    return HEALTH_OK


def auto_rollback_if_needed(log_path: str, dry_run: bool = True) -> str | None:
    """训练后健康检查；不健康 → 建议/执行回滚到上一版 adapter（24h 反悔窗口）。"""
    if not os.path.isfile(log_path):
        return None
    with open(log_path, encoding="utf-8") as f:
        log = f.read()
    verdict = check_training_health(log)
    if verdict == HEALTH_OK:
        print(f"[lifecycle] 训练健康 ✅（{log_path}）")
        return None
    if dry_run:
        print(f"[lifecycle] ⚠ 训练不健康（{verdict}）→ 建议回滚上一版 adapter（dry-run 未执行）")
        return verdict
    from adapter_manage import rollback
    ok = rollback()
    print(f"[lifecycle] 已自动回滚：{ok}")
    return verdict


def monthly_full_retrain(dry_run: bool = True) -> str | None:
    """月全量重训流程（M5-③）—— 整合本月全部每日 adapter 的样本 + 锚点，全量重训。

    触发：每月一次（设计文档 §4：月全量重训防累积失真）。
    产出建议：整合 datasets/rem 全部历史 jsonl + 锚点 → 重训 rem_monthly。
    """
    import glob as _g
    dss = sorted(_g.glob(os.path.join(config.DATASETS, config.PERSONA["name"], "*.jsonl")))
    if dry_run:
        print(f"[lifecycle] 月全量重训（dry-run）：候选数据 {len(dss)} 个文件")
        for d in dss[-5:]:
            print(f"    {os.path.basename(d)}")
        print("    建议命令：整合后 ./run.sh python3 -m mlx_lm.lora -c v2/tools/train_rem.yaml（iters 调大）")
        return None
    # 真实执行：合并全部数据 → 临时 train 集
    out = os.path.join(config.DATASETS, config.PERSONA["name"], f"monthly-{time.strftime('%Y%m')}.jsonl")
    n = 0
    with open(out, "w", encoding="utf-8") as fo:
        for d in dss:
            if "monthly" in d:
                continue
            for line in open(d, encoding="utf-8"):
                fo.write(line)
                n += 1
    print(f"[lifecycle] 月全量数据已整合 → {out}（{n} 条）；随后跑 mlx_lm.lora（iters 适当调大）")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif cmd == "prune":
        prune(dry_run="--yes" not in sys.argv)
    elif cmd == "weekly_merge":
        weekly_merge()
    elif cmd in ("monthly_retrain", "monthly"):
        monthly_full_retrain(dry_run="--yes" not in sys.argv)
    else:
        print(__doc__)
