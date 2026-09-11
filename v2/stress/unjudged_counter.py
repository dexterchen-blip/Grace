#!/usr/bin/env python3
"""体量触发器计数器 v1（2026-09-10 判官首战标定：阈值 600 / 吞吐 28 条每分钟 / swap ~5min）。

职责: 扫描沙盒训练数据集里的 assistant 输出, 对照已判哈希集, 提供
  count / build(未筛输入文件) / seed(首判种子) / mark(检查点后记账) 四个模式。

用法:
  unjudged_counter.py count
  unjudged_counter.py build <out.jsonl> [cap=800]
  unjudged_counter.py seed  <input.jsonl> <verdicts.jsonl>   # 初始化(覆盖)已判集合
  unjudged_counter.py mark  <input.jsonl> <verdicts.jsonl>   # 追加 pass/drop 的哈希

纪律: unjudged 的条目绝不 mark（重筛纪律）——mark 只收 verdict ∈ {pass, drop}。
"""
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DS_GLOB = os.path.join(ROOT, "experiments", "lora", "datasets",
                       "stress-rem_stress_d*", "train.jsonl")
JUDGED = os.path.join(ROOT, "experiments", "run", "stress", "judgments",
                      ".judged-hashes.txt")


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _assistant_items():
    """[(text, ...)] 全量 assistant 输出（>=4 字）。"""
    out = []
    for fp in sorted(glob.glob(DS_GLOB)):
        day = os.path.basename(os.path.dirname(fp)).split("_stress_")[-1]
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for ln, line in enumerate(f):
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                for m in r.get("messages", []):
                    if m.get("role") == "assistant":
                        t = str(m.get("content", "")).strip()
                        if len(t) >= 4:
                            out.append((f"{day}-{ln}", t))
    return out


def _load_judged() -> set:
    if not os.path.isfile(JUDGED):
        return set()
    with open(JUDGED, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}


def _verdict_map(verdicts_path: str) -> dict:
    """id → verdict。"""
    vm = {}
    with open(verdicts_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                vm[str(r.get("id"))] = r.get("verdict")
            except Exception:  # noqa: BLE001
                continue
    return vm


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "count":
        judged = _load_judged()
        n, seen = 0, set()
        for _, t in _assistant_items():
            h = _hash(t)
            if h in judged or h in seen:
                continue
            seen.add(h)
            n += 1
        print(n)
        return 0

    if mode == "build":
        out_path = sys.argv[2]
        cap = int(sys.argv[3]) if len(sys.argv) > 3 else 800
        judged = _load_judged()
        seen, rows = set(), []
        for _id, t in _assistant_items():
            h = _hash(t)
            if h in judged or h in seen:
                continue
            seen.add(h)
            rows.append({"id": _id, "kind": "rem_output", "text": t})
            if len(rows) >= cap:
                break
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(len(rows))
        return 0

    if mode in ("seed", "mark"):
        input_path, verdicts_path = sys.argv[2], sys.argv[3]
        vm = _verdict_map(verdicts_path)
        judged = _load_judged()
        added = 0
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                if vm.get(str(s.get("id"))) not in ("pass", "drop"):
                    continue   # unjudged/缺失 → 不记账（重筛纪律）
                h = _hash(str(s.get("text", "")))
                if h not in judged:
                    judged.add(h)
                    added += 1
        os.makedirs(os.path.dirname(JUDGED), exist_ok=True)
        with open(JUDGED, "a" if mode == "mark" else "w", encoding="utf-8") as f:
            for h in judged:
                f.write(h + "\n")
        print(added)
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
