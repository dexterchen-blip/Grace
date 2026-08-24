# Grace — Local AI Memory Agent System

A fully offline local AI memory & automation agent system (macOS / Apple Silicon).

**Design goal**: Personal data sovereignty — email, WeChat, school info, and conversations all stay
local, orchestrated by two models running in staggered shifts to close the loop of
**ingest → understand → remember → auto-remind**, with zero cloud API dependency.

> ⚠️ This repository is a **sanitized public release**: it contains no crawler-collection layer
> code or approach, no personal data, no model weights, and no secrets.
> Privacy boundaries (memory stores L0-L3, exchange data, test-sandbox, models/, face features,
> key inventory) are intentionally excluded.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Data ingress (crawler collection layer, maintained separately, │
│  not in this repo): email / wechat / canvas / docs / images /   │
│  conversations → structured local files                         │
└──────────────┬──────────────────────────────┬──────────────┘
               ▼                              ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│  L-1 transient memory    │   │  Night pipeline               │
│  exchange/.daytime/      │   │  gate → ingest → embed →      │
│  30-min sync, cleared     │   │  consolidate → knowledge     │
│  next day                 │   │  graph → review → proposals  │
└──────────────┬───────────┘   └──────────────┬───────────────┘
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│  L0 raw memory (JSONL) → L2 semantic index (sqlite-vec + FTS5) │
│  → L3 core memory (core.md, facts/todos/rules, git-versioned)  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Local dashboard (:3091) + secret store (macOS Keychain)      │
│  Night report / today's priorities / proposal approval /      │
│  ask-secrets-in-chat (leak-proof)                             │
└─────────────────────────────────────────────────────────────┘
```

## Four-layer memory (core idea)

| Layer | Store | Lifetime | Notes |
|---|---|---|---|
| **L-1 transient** | `exchange/.daytime/` | 30-min sync, cleared overnight | day-time scrape temp files; never enters formal memory |
| **L0 raw** | `memory/L0_raw/*.jsonl` | long-term | ingested raw facts (sole writer = night pipeline) || **L2 semantic** | `memory/L2_semantic/l2.db` | long-term | vector (bge-m3) + BM25 dual index, RRF fusion; KG tables |
| **L3 core** | `memory/L3_core/core.md` | long-term (git-versioned) | facts/todos/rules/preferences; written by 35B consolidation, corrected by 27B review |


## Dual-model staggered-shift rule

- **Day 27B** (Qwen3.8-27B-4bit, MLX): resident, dashboard chat
- **Night 35B** (Qwen3.5-35B-A3B Q6_K, llama.cpp): consolidation + deep analysis (thinking kept)
- **Only one model resident at any time** (48GB machine can't hold both — lesson from 2026-08-22 double panic)
- Night: suspend 27B → 35B consolidate/graph → stop 35B → boot 27B review → restore day model

## Night pipeline (night_pipeline.py)

```
segment1 gate+ingest (freshness check → L0 delta)
segment2 embed (L2 build: vector + FTS5)
segment3 consolidate (35B → L3 facts/priorities/proposals) + knowledge graph (27B extract) + deep review (27B)
segment4 watchdog (L3 git snapshot → expire proposals → decay scan)
Output: night report (changelog + today's priority list) + modification proposals (require user approval)
```

## Modules (src/)

| Module | Responsibility |
|---|---|
| `m4_ingest.py` / `l0_ingest.py` | L0 ingestion (md5 watermarks, idempotent) |
| `l2_semantic.py` | L2 semantic index (bge-m3 vectors + BM25, RRF) |
| `knowledge_graph.py` | entity/relation extraction (27B, no-thinking clean JSON) |
| `night_pipeline.py` | 4-segment orchestration + gates + report + watchdog |
| `proposal_queue.py` | modification proposals (pending/approved/expired; sleeping agent never waits on humans) |
| `m6_dashboard.py` / `m6_sidebar.py` | local dashboard (reports/proposals/chat/tool mode) |
| `vault.py` | secret store (macOS Keychain; ask/store secrets in chat, leak-proof redaction) |
| `writeback.py` | L3 write-back executor (applies approved changes) |
| `reminder_check.py` / `m8_urgent_watch.py` | reminder rules (remind_after: don't flag high before date) |
| `chat_capture.py` | conversations → L0 |
| `vision_ingest.py` / `doc_ingest.py` | image/document ingestion |
| `verify_night.py` | night-run artifact verification |

## Directory layout

```
src/            core code (this repo)
docs/           design documents
memory/         memory stores (L0-L3, private, not in this repo)
exchange/       data exchange area (private, not in this repo)
models/         local model weights (40GB+, not in this repo)
test-sandbox/   sandbox test env (isolated, not in this repo)
```

## Runtime environment

- macOS (Apple Silicon), 48GB+ recommended
- Python venvs: llama-cpp (night), mlx-lm (day), doc-parse, l2 semantic
- Models: Qwen3.5-35B-A3B (night) / Qwen3.8-27B-4bit (day) / bge-m3 (embedding)
- All model services listen on 127.0.0.1 only

## Design docs

- `docs/本地AI代理系统设计文档.md` — full system design (memory layers/night pipeline/proposals/security)
- `docs/本地AI代理系统-dsh架构反转版-草案.md` — dsh architecture migration draft

---
*Architecture & implementation: 2026-08; sanitized public release of a personal local AI system.*
