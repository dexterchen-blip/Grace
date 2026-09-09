#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""neural_memory.py — 在线压缩记忆 v1.1（Grace V2.4，2026-09-09）。

工作记忆器官第三代：新奇度检测 + 惊奇门控语义压缩。

v1.1 实现决策（诚实记录）：在线自编码器方案已实验并搁置——重建误差被自身
拟合状态污染（小样本下 med/P90 校准无判别力），改用**记忆型新奇度（kNN）**：
    nov_i = 1 − max_j cos(emb_i, emb_j)   （j = 已见过的全部事件嵌入）
    · 重复事件 → sim≈1 → nov≈0（习惯化，跨日持续——嵌入库持久）
    · 新话题   → sim 低 → nov 高
    · 认知同构：海马 match/mismatch 检测；AE 惊奇门控留作 v2 实验（校准陷阱已记录）

运行环境：llama-cpp venv（llama_cpp + numpy）。嵌入库状态：memory/neural-wm/。
CLI：
    update --date YYYY-MM-DD --events '<json: [{"t","tag"}...]>' --out <day-memory.json>
写入：out {"date","digest","entries":[{"t","tag","nov"}]}
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMBED_PATH = os.path.join(REPO, "models", "embed", "bge-m3-q8_0.gguf")
WM_DIR = os.path.join(REPO, "memory", "neural-wm")
SEEN_CAP = 2000          # 嵌入库上限（FIFO）——足够数月的事件流


def _embedder():
    from llama_cpp import Llama
    return Llama(model_path=EMBED_PATH, embedding=True,
                 n_gpu_layers=-1, n_ctx=2048, verbose=False)


def _ehash(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]


class MemoryBank:
    """已见事件嵌入库 + 新奇度检测。"""

    def __init__(self, wm_dir: str = WM_DIR):
        self.dir = wm_dir
        os.makedirs(wm_dir, exist_ok=True)
        self.state_path = os.path.join(wm_dir, "bank.npz")
        if os.path.isfile(self.state_path):
            z = np.load(self.state_path)
            self.E = z["E"]                                   # (N, 1024)
            self.texts = list(z["texts"])
        else:
            self.E = np.zeros((0, 1024), dtype=np.float32)
            self.texts = []

    def _save(self):
        np.savez(self.state_path, E=self.E, texts=np.array(self.texts))

    def _embed(self, texts):
        todo = [(t, _ehash(t)) for t in texts if _ehash(t) not in getattr(self, "_cache", {})]
        if todo:
            llm = _embedder()
            vecs = llm.embed([t for t, _ in todo])
            if not hasattr(self, "_cache"):
                self._cache = {}
            for (t, h), v in zip(todo, vecs):
                self._cache[h] = np.array(v[:1024], dtype=np.float32)
        if not hasattr(self, "_cache"):
            self._cache = {}
        return [self._cache[_ehash(t)] for t in texts]

    def update_day(self, date: str, events: list) -> dict:
        """事件流 → 逐条新奇度（kNN）→ 写嵌入库（FIFO cap）。"""
        texts = [e["t"] for e in events]
        vecs = self._embed(texts)
        entries = []
        for e, v in zip(events, vecs):
            if self.E.shape[0] > 0:
                sims = self.E @ v / (np.linalg.norm(self.E, axis=1) * np.linalg.norm(v) + 1e-8)
                nov = round(float(1.0 - float(np.max(sims))), 3)
            else:
                nov = 1.0                                     # 第一个事件：全新
            entries.append({"t": e["t"], "tag": e.get("tag", ""), "nov": max(0.0, nov)})
        # 入库（FIFO cap）
        self.E = np.vstack([self.E, np.array(vecs)])[-SEEN_CAP:]
        for e in events:
            self.texts.append(e["t"][:60])
        self.texts = self.texts[-SEEN_CAP:]
        self._save()
        entries.sort(key=lambda x: -x["nov"])
        digest = "\n".join(f"· [{e.get('tag','')}] {e['t']} (nov {e['nov']:.2f})" for e in entries)
        return {"date": date, "entries": entries, "digest": digest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["update"])
    ap.add_argument("--date", required=True)
    ap.add_argument("--events", required=True, help="JSON: [{t,tag}]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "update":
        events = json.loads(a.events)
        m = MemoryBank()
        out = m.update_day(a.date, events)
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps({"ok": True, "n": len(out["entries"]),
                          "top_nov": out["entries"][0]["nov"] if out["entries"] else None},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
