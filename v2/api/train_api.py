#!/usr/bin/env python3
"""人审驯服自训练 API —— Grace V2 最值钱的创新点（Grace_v2 设计 §6）。

把 v1 的「人审管记忆」升级成「人审管模型自己改自己」：
  夜班提炼训练候选 → 进闸门（pending）→ 人类经本 API 批准/否决/编辑
  → 只训 approved + 锚点回放 → 每次训练只出一个报告 → 隔天生效（24h 反悔窗口）。

技术：纯标准库 http.server（与 m6_dashboard 同风格，零第三方依赖）。
端口：18300（沙箱专用段，永不碰正式 3091/8100/8200）。
一切数据落沙盒（exchange/proposals + experiments/run），正式系统零接触。

端点：
  GET  /api/v2/status                         三轨状态
  GET  /api/v2/persona                        当前人格锚点（雷姆）
  GET  /api/v2/candidates?status=pending      候选列表
  GET  /api/v2/candidates/{id}                候选详情
  POST /api/v2/candidates/{id}/approve        批准   {decided_by?}
  POST /api/v2/candidates/{id}/reject         否决   {reason?, decided_by?}
  POST /api/v2/candidates/{id}/edit           编辑   {samples?[], title?, description?} → 回到 pending
  POST /api/v2/train                          发起一次训练（只含 approved 候选；dry_run 默认 true）
  GET  /api/v2/reports                        报告列表
  GET  /api/v2/reports/{name}                 报告内容
  POST /api/v2/mood/derive                    心态推演 {events: [{text,sentiment,weight}]}
  GET  /api/v2/mood/timeline                  心态时间线
"""
from __future__ import annotations
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import config  # noqa: E402
from candidate_extract import CandidateQueue  # noqa: E402
from report import list_reports  # noqa: E402
from mood_engine import derive as mood_derive, timeline as mood_timeline  # noqa: E402
from night_engine_v2 import run_night  # noqa: E402

QUEUE = CandidateQueue()
_lock = threading.Lock()


def _json(data, status=200):
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    return (status, {"Content-Type": "application/json; charset=utf-8"}, body)


def _txt(text, status=200):
    body = text.encode("utf-8")
    return (status, {"Content-Type": "text/plain; charset=utf-8"}, body)


def _read_body(h: BaseHTTPRequestHandler) -> dict:
    try:
        n = int(h.headers.get("Content-Length", 0))
        if n:
            return json.loads(h.rfile.read(n).decode("utf-8") or "{}")
    except Exception:
        pass
    return {}


def handle_get(path: str, query: dict | None = None) -> tuple:
    query = query or {}
    if path == "/api/v2/status":
        c = QUEUE.list("pending")
        return _json({
            "sandbox": config.SB,
            "tracks": {
                "外挂轨": {"L0": config.L0_DIR, "L2": config.L2_DB, "L3": config.L3_FILE},
                "权重轨": {"persona": config.PERSONA["name"], "adapter_dir": config.PERSONA["adapter_dir"],
                           "lora": config.LORA, "lifecycle": config.LORA_LIFECYCLE},
                "心态轨": {"mood_states": "l2.db 表", "decay": config.MOOD["decay"]},
            },
            "gate": {"pending_candidates": len(c),
                     "api_port": config.API_PORT,
                     "rule": "未经批准当晚不训练；LoRA 隔天生效；快照可回滚"},
        })
    if path == "/api/v2/persona":
        try:
            with open(config.PERSONA["anchor_file"], encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            content = "(anchor 文件缺失)"
        return _json({"persona": config.PERSONA, "anchor_md": content[:4000]})
    if path == "/api/v2/candidates":
        status = query.get("status", ["pending"])[0]
        return _json({"candidates": QUEUE.list(status)})
    if path == "/api/v2/reports":
        return _json({"reports": list_reports()})
    if path == "/api/v2/mood/timeline":
        return _json({"timeline": mood_timeline()})

    # /api/v2/candidates/{id}
    parts = path.strip("/").split("/")
    if len(parts) == 4 and parts[:3] == ["api", "v2", "candidates"]:
        rec = QUEUE.load(parts[3])
        return _json({"candidate": rec}) if rec else _json({"error": "not found"}, 404)
    if len(parts) == 4 and parts[:3] == ["api", "v2", "reports"] and parts[3].endswith(".md"):
        p = os.path.join(config.REPORTS, parts[3])
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return _txt(f.read())
        return _json({"error": "report not found"}, 404)
    return _json({"error": "not found", "path": path}, 404)


def handle_post(path: str, body: dict) -> tuple:
    parts = path.strip("/").split("/")
    # POST /api/v2/candidates/{id}/approve|reject|edit
    if len(parts) == 5 and parts[:3] == ["api", "v2", "candidates"]:
        cid, action = parts[3], parts[4]
        with _lock:
            rec = QUEUE.load(cid)
            if not rec:
                return _json({"error": f"candidate {cid} not found"}, 404)
            if action == "approve":
                ok = QUEUE.approve(cid, decided_by=body.get("decided_by", "api-user"))
                return _json({"ok": ok, "id": cid, "status": "approved"}) if ok else \
                    _json({"error": f"{cid} 非 pending（status={rec['status']}）"}, 409)
            if action == "reject":
                ok = QUEUE.reject(cid, reason=body.get("reason", ""), decided_by=body.get("decided_by", "api-user"))
                return _json({"ok": ok, "id": cid, "status": "rejected"}) if ok else \
                    _json({"error": f"{cid} 非 pending（status={rec['status']}）"}, 409)
            if action == "edit":
                if "samples" in body:
                    rec["samples"] = body["samples"]
                if "title" in body:
                    rec["title"] = body["title"]
                if "description" in body:
                    rec["description"] = body["description"]
                rec["status"] = "pending"
                rec["decided_at"] = rec["decided_by"] = rec["reject_reason"] = None
                os.remove(QUEUE._find(cid))
                with open(QUEUE._path("pending", cid), "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                return _json({"ok": True, "id": cid, "status": "pending", "samples": len(rec["samples"])})
    if len(parts) == 3 and parts == ["api", "v2", "train"]:
        dry = bool(body.get("dry_run", True))
        # 发起训练（人审驯服：只训 approved；骨架默认 dry-run 防误触发真实训练）
        with _lock:
            summary = run_night(dry_run=dry, skip_consolidate=True)
        return _json({"ok": True, "run": summary}, 200)
    if len(parts) == 5 and parts[:4] == ["api", "v2", "mood", "derive"]:
        rec = mood_derive(body.get("events", []))
        return _json({"ok": True, "mood": rec})
    return _json({"error": "not found", "path": path}, 404)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    def _respond(self):
        try:
            u = urlparse(self.path)
            from urllib.parse import parse_qs
            query = parse_qs(u.query)
            if self.command == "GET":
                status, hdrs, body = handle_get(u.path, query)
            elif self.command == "POST":
                body = _read_body(self)
                status, hdrs, body = handle_post(u.path, body)
            else:
                status, hdrs, body = _json({"error": "method not allowed"}, 405)
        except Exception as e:  # noqa: BLE001
            status, hdrs, body = _json({"error": str(e)}, 500)
        self.send_response(status)
        for k, v in hdrs.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _respond


def main():
    config.ensure_dirs()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else config.API_PORT
    srv = ThreadingHTTPServer((config.API_HOST, port), Handler)
    print(f"[v2-api] 人审驯服自训练 API 起于 http://{config.API_HOST}:{port}/api/v2/status")
    print(f"  - 训练候选闸门: exchange/proposals/pending（type=lora_train）")
    print(f"  - 训练报告: experiments/run/*-lora-report.md（每次训练只出一个报告）")
    print(f"  - 隔离: 仅 127.0.0.1，端口 {port} 属沙箱段，正式系统零接触")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[v2-api] 退出")


if __name__ == "__main__":
    main()
