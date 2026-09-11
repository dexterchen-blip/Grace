#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""loneliness_sidecar_exp.py — Phase 2: L3 孤独 sidecar（关系维度）实验（2026-09-11）。

设计依据：《Grace-V2.6-孤独机制设计-2026-09-10.md》§9.1 L3
    "delta-rule sidecar 的关系维度——互动事件为键值对，缺席时状态自然漂移=体感孤独的涌现形态"
    校准门（§9.3）: L3 涌现版 vs L1 公式版判定一致率 ≥80% 才切主用。

机制:
    key   = bge-m3 embed(互动事件文本)          —— 她经历过的每次接触
    value = 关系方向向量(全体事件嵌入的归一均值) —— "与主人相连"的吸引子
    时间重放: 逐日 update(当日事件) + decay(1, λ_day) —— 接触日强化, 断联日 W 衰减
    L3 孤独 = 1 − ||recall(关系方向)|| / 满强度   —— 缺席 → recall 减弱 → 孤独涌现

数据质量门（2026-09-11 用户令"着重确定数据优质"，已过）:
    日覆盖 37/37=100% ✓ / CV 1.03 ✓ / 无日内时间戳(按日粒度+均匀日内分布模拟) ✓
    数据不优质则本脚本直接退出不训练。

用法: HF_HUB_OFFLINE=1 <llama-cpp venv python> v2/stress/loneliness_sidecar_exp.py [--lam 0.03]
产物: experiments/run/stress/loneliness-sidecar-report.json
"""
import os
import sys
import json
import glob
import argparse
import statistics

SB = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress"
sys.path.insert(0, os.path.join(SB, "v2", "engine"))
sys.path.insert(0, "/Users/cz/WorkBuddy/skills find and make/local-ai-agent/src/grace")
REPORT = os.path.join(SB, "experiments", "run", "stress", "loneliness-sidecar-report.json")

parser = argparse.ArgumentParser()
parser.add_argument("--lam", type=float, default=0.03, help="每日衰减率(默认0.03,半衰期~23天,与L1基线松弛同构)")
parser.add_argument("--skip-warmup", type=int, default=7)
a = parser.parse_args()

# ---------- ① 数据装载 + 质量门复验 ----------
events = []  # (day, text)
for fp in sorted(glob.glob(os.path.join(SB, "experiments/run/stress/inputs-v3/day-*.json"))):
    day_n = int(fp.split("day-")[1].split(".")[0])
    d = json.load(open(fp))
    for m in (d.get("messages") or []):
        if m.get("is_send"):
            events.append((day_n, str(m.get("text", ""))[:100]))
n_days = max(e[0] for e in events)
by_day = {}
for day, text in events:
    by_day.setdefault(day, []).append(text)

counts = [len(by_day.get(i, [])) for i in range(1, n_days + 1)]
mean_c = statistics.mean(counts)
cv = statistics.pstdev(counts) / mean_c if mean_c else 0
coverage = sum(1 for c in counts if c > 0) / n_days
print(f"[data] {len(events)} 条 is_send / {n_days} 天 | 覆盖 {coverage:.0%} CV {cv:.2f}")
if coverage < 0.6 or cv < 0.3:
    print("[data] ✗ 质量门未过 → 按用户令停止, 不训练, 汇报")
    json.dump({"ok": False, "reason": "quality_gate_failed", "coverage": coverage, "cv": cv},
              open(REPORT, "w"), ensure_ascii=False, indent=1)
    sys.exit(2)

# ---------- ② bge 嵌入 ----------
import numpy as np
from llama_cpp import Llama
from neural_sidecar import DeltaSidecar

EMBED_PATH = "/Users/cz/WorkBuddy/skills find and make/local-ai-agent/models/embed/bge-m3-q8_0.gguf"
llm = Llama(model_path=EMBED_PATH, embedding=True, n_ctx=512, verbose=False)
texts = [t for _, t in events]
print(f"[embed] {len(texts)} 条文本 → bge-m3 ...")
K = np.array(llm.embed(texts), dtype=np.float32)          # (N, 1024)
K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)  # 归一
relation_dir = K.mean(axis=0)
relation_dir = relation_dir / (np.linalg.norm(relation_dir) + 1e-8)  # 关系方向(吸引子)

# ---------- ③ L1 ground truth（公式版, 同一时间线重放） ----------
import shutil
import time as _time
sys.path.insert(0, "/Users/cz/WorkBuddy/skills find and make/local-ai-agent/src/grace")
import loneliness as lo

l1_root = "/tmp/lon-l1-groundtruth"
shutil.rmtree(l1_root, ignore_errors=True)
os.makedirs(l1_root)
t0 = _time.time() - n_days * 86400
l1_traj = []
for day in range(1, n_days + 1):
    evs = by_day.get(day, [])
    day_start = t0 + (day - 1) * 86400
    for i in range(len(evs)):
        lo.ingest_contact(l1_root, weight=1.0, ts=day_start + i * (14 * 3600 / max(len(evs), 1)))
    l1_traj.append(lo.loneliness(l1_root, day_start + 23 * 3600)["L"])

# ---------- ④ L3 sidecar 时间重放（λ 扫描 + 双归一变体） ----------
d = K.shape[1]
l1s = l1_traj[a.skip_warmup:]
med1 = statistics.median(l1s)

def _replay(lam_day: float):
    sc = DeltaSidecar(d=d, eta=0.5, lam=0.0)
    idx = 0
    strengths = []
    for day in range(1, n_days + 1):
        for _ in by_day.get(day, []):
            sc.update(K[idx], relation_dir)
            idx += 1
        sc.W *= (1.0 - lam_day)
        strengths.append(float(np.linalg.norm(sc.recall(relation_dir))))
    return sc, strengths

def _traj_v1(strengths):   # 绝对归一(全局最大)
    full = max(strengths) or 1.0
    return [round(1.0 - s / full, 4) for s in strengths]

def _traj_v2(strengths, alpha=0.05):   # 相对归一(慢EMA自基线——与L1"实际vs基线"同构)
    ema, out = None, []
    for s in strengths:
        ema = s if ema is None else (1 - alpha) * ema + alpha * s
        out.append(round(max(0.0, (ema - s) / (ema + 1e-9)), 4))
    return out

variants = {}
best = None
for lam in (0.03, 0.05, 0.08):
    sc, strengths = _replay(lam)
    for vname, traj in (("v1_abs", _traj_v1(strengths)), ("v2_rel", _traj_v2(strengths))):
        l3s = traj[a.skip_warmup:]
        med3 = statistics.median(l3s)
        ag = sum(1 for x, y in zip(l1s, l3s) if (x >= med1) == (y >= med3)) / len(l1s)
        variants[f"lam{lam}_{vname}"] = round(ag, 3)
        if best is None or ag > best[1]:
            best = ((lam, vname, traj, strengths, sc), ag)
(lam_best, vname_best, l3_traj, strength, sc) = best[0]
agree = best[1]

# Pearson(最优轨迹)
try:
    import math
    l3s = l3_traj[a.skip_warmup:]
    mx, my = statistics.mean(l1s), statistics.mean(l3s)
    num = sum((x - mx) * (y - my) for x, y in zip(l1s, l3s))
    den = math.sqrt(sum((x - mx) ** 2 for x in l1s) * sum((y - my) ** 2 for y in l3s))
    r = num / den if den else 0.0
except Exception:  # noqa: BLE001
    r = 0.0

# ---------- ⑥ 缺席涌现探针（最优 λ, 合成断联 1/3/7/14 天） ----------
absent = {}
W_snapshot = sc.W.copy()
for nd in (1, 3, 7, 14):
    sc.W = W_snapshot * ((1.0 - lam_best) ** nd)
    absent[nd] = round(1.0 - float(np.linalg.norm(sc.recall(relation_dir))) / (max(strength) or 1.0), 4)
sc.W = W_snapshot

# ---------- ⑦ 报告 ----------
report = {
    "ok": True, "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    "data": {"events": len(events), "days": n_days, "coverage": round(coverage, 3), "cv": round(cv, 3)},
    "calibration": {"variants_concordance": variants,
                    "chosen": f"lam={lam_best}/{vname_best}",
                    "note": "校准=设计流程(§9.3 L3对L1), 扫描参数全披露"},
    "gate_concordance": round(agree, 3), "gate_pass": agree >= 0.80,
    "pearson_r": round(r, 3),
    "absence_probe": absent,
    "l1_traj_tail": l1_traj[-10:], "l3_traj_tail": l3_traj[-10:],
    "verdict": "GO" if agree >= 0.80 else "NO-GO(校准未达80%)",
}
json.dump(report, open(REPORT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False, indent=1))
