#!/usr/bin/env python3
"""storage.py — ★2026-09-03 存储层管理（单一事实源 + 权重/轮次治理）。

背景（用户：改进存储设计，权重也改进进去）：
  现状问题：
    ① db 分散：l2.db(图谱+心态) / mood.db(空僵尸) / autobiography.db(L3) —— 无统一路径管理
    ② 权重 2.5G + 归档 6.3G 堆积：rem_stress_d{1..33} 每轮 33 个 adapter 无 registry 无清理
    ③ live jsonl 无 schema 版本声明（feedback v3 / proactive v4 字段演进靠注释）
    ④ 无轮次元数据（每轮参数/产物无法溯源）
  本模块：集中读写 轮次元数据 + 权重登记 + 存储分层路径声明（代码内单一事实源）。
  设计原则：只新增不破坏——现有各模块路径照旧可用；新能力（registry/round-meta）逐步接入。
"""
from __future__ import annotations

import json
import os
import time

# ---------------------------------------------------------------- 路径单一事实源
_THIS = os.path.dirname(os.path.abspath(__file__))          # v2/engine/
V2 = os.path.dirname(_THIS)                                  # v2/
# ★SB 尊重 AIAGENT_SANDBOX env（同 config.py）——压测场/test-sandbox 隔离全靠它
SB = os.environ.get("AIAGENT_SANDBOX") or os.path.dirname(V2)  # 压测场根（默认）

MEMORY = os.path.join(SB, "memory")
L2_DB = os.path.join(MEMORY, "L2_semantic", "l2.db")         # 双图谱 + 心态（主情感存储）
L3_DB = os.path.join(MEMORY, "L3_core", "autobiography.db")  # 自传叙事矩阵
EXPERIMENTS = os.path.join(SB, "experiments")
STRESS_ROOT = os.path.join(EXPERIMENTS, "run", "stress")     # live 观测 + 断点 + 报告
ADAPTERS = os.path.join(EXPERIMENTS, "lora", "adapters")
WEIGHTS_REG = os.path.join(EXPERIMENTS, "lora", "weights-registry.json")
ROUND_META = os.path.join(STRESS_ROOT, "round-meta.json")

# ---------------------------------------------------------------- live 文件 schema 版本声明
# 字段演进只追加不改写；读取方按版本兼容（已有兼容逻辑，本表作权威文档）
LIVE_SCHEMA = {
    "feedback-live.jsonl": {
        "ver": 3, "date": "2026-09-02",
        "fields": ["situation", "believed", "real", "w"],
        "history": [
            "v1(8/30): {text,w} 句式整句",
            "v2(9/2): {situation,real,w} 零句式数据对",
            "v3(9/2): +believed 判断冲突对(我以为X后来知道Y)",
        ]},
    "proactive-live.jsonl": {
        "ver": 4, "date": "2026-09-03",
        "fields": ["day", "situation", "message", "pending", "generated", "suppressed",
                   "think", "intent", "emotion", "confidence", "attention", "owner_mood",
                   "mood", "relation", "dmn_spontaneous"],
        "history": [
            "v1: rule 规则句",
            "v2(9/1): 断点模型生成 + rule_generated 标记",
            "v3(9/2): pending(message 空) + intent",
            "v4(9/3): +think 暗注意力层(评估双轨)",
        ]},
    "prediction-errors.jsonl": {
        "ver": 2, "date": "2026-09-01",
        "fields": ["day", "believed", "real", "correct", "pe", "w"],
        "history": ["v1(8/31): 只记错", "v2(9/1): 全量(对+错) + correct 字段"]},
    "cognition-live.jsonl": {"ver": 2, "date": "2026-09-02",
                             "fields": ["text"], "history": []},
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _load(path: str, default):
    if not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


# ================================================================ 轮次元数据
def create_round(**meta) -> str:
    """轮启动登记（round-meta.json）。返回 round_id（时间戳）。"""
    rounds = _load(ROUND_META, {"rounds": {}})
    rid = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    entry = {"id": rid, "started_at": now_iso(), "created_by": "stress_engine",
             **meta}
    rounds["rounds"][rid] = entry
    _save(ROUND_META, rounds)
    return rid


def resume_round(rid: str, note: str = "") -> None:
    """续跑标记（detect 到已摄入 → 追加 resume 记录，不覆盖原元数据）。"""
    rounds = _load(ROUND_META, {"rounds": {}})
    if rid in rounds.get("rounds", {}):
        rounds["rounds"][rid]["resumed_at"] = now_iso()
        if note:
            rounds["rounds"][rid].setdefault("resume_notes", []).append(note)
        _save(ROUND_META, rounds)


def finalize_round(rid: str, summary: dict) -> None:
    """轮完成登记（训练数/断点数/耗时）。"""
    rounds = _load(ROUND_META, {"rounds": {}})
    if rid in rounds.get("rounds", {}):
        rounds["rounds"][rid]["finished_at"] = now_iso()
        rounds["rounds"][rid]["summary"] = summary
        _save(ROUND_META, rounds)


# ================================================================ 权重登记（registry）
def register_weight(round_id: str, day: int, adapter: str, *,
                    samples: int = 0, ok: bool = True, extra: dict | None = None) -> None:
    """训练成功登记一条权重记录（可追溯：轮次/天/样本数/参数）。"""
    reg = _load(WEIGHTS_REG, {"weights": []})
    reg.setdefault("weights", []).append({
        "round": round_id, "day": day, "adapter": adapter,
        "samples": samples, "ok": ok, "ts": now_iso(),
        **(extra or {}),
    })
    _save(WEIGHTS_REG, reg)


def weights_since(ts_iso: str | None = None) -> list[dict]:
    """权重清单（可选按时间过滤）。"""
    reg = _load(WEIGHTS_REG, {"weights": []})
    ws = reg.get("weights", [])
    if ts_iso:
        ws = [w for w in ws if w.get("ts", "") >= ts_iso]
    return ws


def prune_middle_adapters(round_id: str = "", keep_days: tuple = ()) -> dict:
    """中间 adapter 清理建议（★只报告不删除——数据安全优先）。

    keep_days 为空 = 只保留末 adapter；报告建议删除的中间 adapter 清单，
    调用方（reset/运维）确认后自行删除。
    """
    import glob as _glob
    reg = _load(WEIGHTS_REG, {"weights": []})
    ws = [w for w in reg.get("weights", []) if not round_id or w.get("round") == round_id]
    by_round = {}
    for w in ws:
        by_round.setdefault(w["round"], []).append(w)
    report = {}
    for rid, wlist in by_round.items():
        days = sorted({w["day"] for w in wlist if w.get("ok")})
        if not days:
            continue
        keep = set(keep_days) or {days[-1]}                     # 默认只留末 adapter
        mids = [d for d in days if d not in keep]
        adapters = [f"rem_stress_d{d}" for d in mids]
        exist = [a for a in adapters if os.path.isdir(os.path.join(ADAPTERS, a))]
        report[rid] = {"middle_days": mids, "to_delete": exist,
                       "keep_days": sorted(keep)}
    return report


def prune_now(round_id: str = "", keep_days: tuple = (), dry: bool = True) -> list[str]:
    """执行清理（dry=True 默认只打印）。返回实际删除的 adapter 名。"""
    import shutil as _sh
    rep = prune_middle_adapters(round_id, keep_days)
    deleted = []
    for rid, info in rep.items():
        for a in info.get("to_delete", []):
            path = os.path.join(ADAPTERS, a)
            if dry:
                print(f"  [storage] 建议清理(中间adapter): {a} ({round(os.path.getsize(path)/1e6,1)}MB)")
            else:
                _sh.rmtree(path, ignore_errors=True)
                deleted.append(a)
    if not dry and deleted:
        print(f"  [storage] 已清理 {len(deleted)} 个中间 adapter")
    return deleted


# ================================================================ 僵尸文件治理
def stale_files() -> list[dict]:
    """扫描已知僵尸/重复存储（如空 mood.db），返回建议。"""
    out = []
    mood_db = os.path.join(MEMORY, "L2_semantic", "mood.db")
    if os.path.isfile(mood_db):
        try:
            import sqlite3
            con = sqlite3.connect(mood_db)
            n = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
            con.close()
            if n == 0:
                out.append({"file": mood_db, "issue": "空库僵尸(0 表)", "action": "可删除"})
        except Exception:  # noqa: BLE001
            pass
    return out


if __name__ == "__main__":
    import sys
    print("storage.py 冒烟:")
    print("  L2_DB:", L2_DB)
    print("  L3_DB:", L3_DB)
    print("  WEIGHTS_REG:", WEIGHTS_REG)
    print("  live schema:", {k: v["ver"] for k, v in LIVE_SCHEMA.items()})
    print("  僵尸:", stale_files())
    if "--prune-report" in sys.argv:
        print("  prune 报告:", json.dumps(prune_middle_adapters(), ensure_ascii=False)[:300])
