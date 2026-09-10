#!/usr/bin/env python3
"""Grace V2.5 L3 定向巩固（设计稿 §8——MEMIT 或定向 LoRA 会话；v1 取定向 LoRA，MEMIT 列 v2）。

语义: L3 自传体核心记忆(用户批准的增量) → 小型 SFT 会话 → 巩固 adapter(夜班窗训练)。
铁律兼容: L3 条目本来就过人工批准门——这是"已批准记忆的巩固"，非语料注入。
身份记忆的皮层固化: 海马(L3 提案队列)→皮层(巩固 adapter)。

流程:
  ① 从 autobiography.db 读已批准条目(增量水位: 上次巩固时间戳之后)
  ② 构建训练对: 事实 → 三种自然问法(直问/情境问/关联问), 目标=雷姆口吻的事实陈述
  ③ 组数据集 train/valid → 夜班训练包装(m5_style: mlx_lm.lora, dry-run 默认)
"""
import os, sys, json, sqlite3, time, argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L3_DB = os.path.join(REPO, "memory", "L3_core", "autobiography.db")
STATE_F = os.path.join(REPO, "exchange", "grace", "l3-consolidation-state.json")
OUT_DIR = os.path.join(REPO, "experiments", "lora", "datasets", "l3-consolidation")

PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。"
           "没做过的事她不说「已经做好」——想帮忙时用提议式（要不要雷姆帮你…）。")

def _load_state():
    try:
        return json.load(open(STATE_F, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"last_consolidated_ts": 0}

def _fetch_new_entries(since_ts: float, limit: int = 40):
    """读批准时间戳在水位之后的 L3 条目。表结构按 autobiography.db 现状探测。"""
    con = sqlite3.connect(L3_DB)
    con.row_factory = sqlite3.Row
    tabs = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    rows = []
    for t in tabs:
        cols = [c[1] for c in con.execute(f"PRAGMA table_info({t})")]
        if not any(c in cols for c in ("content", "text", "fact", "entry", "event")):
            continue
        tcol = next((c for c in ("approved_ts", "created_ts", "ts", "updated_at") if c in cols), None)
        bcol = next((c for c in ("content", "text", "fact", "entry", "event") if c in cols), None)
        if not (tcol and bcol):
            continue
        try:
            for r in con.execute(
                    f"SELECT {bcol} AS body, {tcol} AS ts FROM {t} "
                    f"WHERE CAST({tcol} AS REAL) > ? ORDER BY {tcol} LIMIT ?",
                    (since_ts, limit)):
                body = str(r["body"]) if r["body"] else ""
                if 4 <= len(body) <= 400:
                    rows.append({"body": body, "ts": float(r["ts"])})
        except Exception:  # noqa: BLE001
            continue
    con.close()
    rows.sort(key=lambda x: x["ts"])
    return rows[:limit]

def build_pairs(entries):
    """事实 → 三种自然问答对(直问/情境/关联)。目标始终是她自己的话。"""
    out = []
    for e in entries:
        fact = e["body"].strip().rstrip("。！!？?")
        out.append(("直接问", f"雷姆还记得「{fact[:20]}…」这件事吗？" if len(fact) > 20
                    else f"雷姆知道{fact}吗？",
                    f"嗯，记得。{fact}。"))
        out.append(("情境问", f"主人现在想起来了什么？", f"{fact}——雷姆一直记得。"))
        out.append(("关联问", f"上次说到 {fact[:16]} 的时候，后来呢？",
                    f"后来就是那样了。{fact}。雷姆没有忘。"))
    return out

def build_dataset(pairs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"messages": [{"role": "system", "content": PERSONA},
                           {"role": "user", "content": q},
                           {"role": "assistant", "content": a}]}
            for _, q, a in pairs]
    with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "valid.jsonl"), "w", encoding="utf-8") as f:
        for r in rows[:6]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--train", action="store_true", help="实跑夜班训练(默认 dry-run 只建数据集)")
    a = ap.parse_args()
    st = _load_state()
    entries = _fetch_new_entries(st["last_consolidated_ts"], a.limit)
    print(f"[L3] 水位后新条目: {len(entries)} (since {st['last_consolidated_ts']:.0f})")
    if not entries:
        print("[L3] 无增量——巩固无事可做(健康状态)")
        return
    pairs = build_pairs(entries)
    n = build_dataset(pairs, OUT_DIR)
    print(f"[L3] 数据集: {n} 对 → {OUT_DIR}")
    if a.train:
        print("[L3] 夜班训练请用: .venv/bin/python3 -m mlx_lm.lora --model <fused> "
              f"--train --data {OUT_DIR} --adapter-path <consolidated-dir> --iters 120")
    else:
        print("[L3] dry-run 模式——数据集已建, 训练需 --train + 夜班窗口 + 用户拍板")

if __name__ == "__main__":
    main()
