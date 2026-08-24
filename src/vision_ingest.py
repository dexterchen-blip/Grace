#!/usr/bin/env python3
"""图像识别摄入（2026-08-22 用户方案：夜班发现图片 → Qwen3-VL-4B 识别 → L0）。

流程：
  1. 扫描 exchange/inbox/** 下新图片（jpg/jpeg/png/webp/gif），按水位幂等
  2. 按需拉起 Qwen3-VL-4B（serve_vision.sh 8081，探测无则起，处理完停）
  3. 每张图：VL-4B 识别（描述 + OCR 提取文字）→ 写 L0（source=vision:image）
  4. 文本结果进 L2（l2_semantic build 扫 L0 自动索引）

沙盒：AIAGENT_SANDBOX 时 SCAN_ROOT/L0/水位跟随沙盒（与正式隔离）。

用法：
  python3 vision_ingest.py           # 扫 + 识别 + 落 L0（夜班 seg1 后调用）
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l0_ingest import L0Writer  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCHANGE = os.path.join(REPO, "exchange")
L0_ROOT = os.path.join(REPO, "memory", "L0_raw")
STATE_PATH = os.path.join(REPO, "memory", "L1_working", "vision_ingest_state.json")

# 测试沙盒（AIAGENT_SANDBOX 重定向，与正式隔离）
SANDBOX = os.environ.get("AIAGENT_SANDBOX", "")
if SANDBOX:
    EXCHANGE = os.path.join(SANDBOX, "exchange")
    L0_ROOT = os.path.join(SANDBOX, "memory", "L0_raw")
    STATE_PATH = os.path.join(SANDBOX, "memory", "L1_working", "vision_ingest_state.json")

VL_PORT = 8081
VL_URL = f"http://127.0.0.1:{VL_PORT}/v1/chat/completions"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
PROMPT = ("识别这张图片：1) 用 1-2 句话描述内容；2) 提取图中所有可见文字。"
          "只输出 JSON：{\"description\":\"...\",\"text\":\"...\"}，不要其他文字。")


def _ensure_vl() -> bool:
    """探测 VL-4B :8081，无则拉起 serve_vision.sh。返回是否就绪。"""
    try:
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{VL_PORT}/v1/models"), timeout=5)
        return True
    except Exception:
        pass
    script = os.path.join(REPO, "src", "serve_vision.sh")
    if not os.path.exists(script):
        return False
    logf = open(os.path.join(EXCHANGE, "shared", "serve-vision.log"), "ab")
    subprocess.Popen(["bash", script, str(VL_PORT)], stdout=logf, stderr=logf,
                     cwd=REPO, start_new_session=True)
    for _ in range(60):  # 最多 3 分钟等模型加载
        time.sleep(3)
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{VL_PORT}/v1/models"), timeout=3)
            return True
        except Exception:
            pass
    return False


def _stop_vl() -> None:
    subprocess.run(["bash", "-c", f"lsof -nP -iTCP:{VL_PORT} -sTCP:LISTEN -t 2>/dev/null | xargs kill 2>/dev/null"],
                   timeout=5)


def _recognize(img_path: str) -> dict | None:
    """VL-4B 识别单图，返回 {description, text} 或 None。"""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = "image/png" if img_path.endswith(".png") else "image/jpeg"
    payload = json.dumps({
        "model": "qwen3-vl",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": PROMPT}]}],
        "max_tokens": 500, "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(VL_URL, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    content = resp["choices"][0]["message"].get("content") or ""
    start, end = content.find("{"), content.rfind("}")
    if start >= 0 and end > start:
        try:
            d = json.loads(content[start:end + 1])
            if isinstance(d, dict):
                return {"description": str(d.get("description", "")).strip(),
                        "text": str(d.get("text", "")).strip()}
        except Exception:
            pass
    return {"description": content[:300], "text": ""}


def run() -> int:
    """扫新图片 → VL 识别 → L0。返回处理张数。"""
    if not _ensure_vl():
        print("[vision] VL-4B 不可用，跳过")
        return 0
    state = {"files": {}}
    if os.path.exists(STATE_PATH):
        try:
            state = json.load(open(STATE_PATH, encoding="utf-8"))
        except Exception:
            pass
    w = L0Writer(L0_ROOT)
    n = 0
    for dirpath, _dirs, files in os.walk(os.path.join(EXCHANGE, "inbox")):
        for fn in sorted(files):
            if not fn.lower().endswith(IMG_EXTS) or fn.startswith("."):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, REPO)
            try:
                sig = hashlib.md5(open(p, "rb").read()).hexdigest()
            except OSError:
                continue
            if state["files"].get(rel) == sig:
                continue
            try:
                result = _recognize(p)
            except Exception as e:
                print(f"[vision] 识别失败 {rel}: {e}")
                continue
            if not result or not (result.get("description") or result.get("text")):
                print(f"[vision] 无识别结果 {rel}")
                continue
            desc = result.get("description", "")
            text = result.get("text", "")
            w.append("vision:image", {"path": rel, "description": desc, "ocr_text": text},
                     sensitive=False, meta={"ingest": "vision_ingest"})
            state["files"][rel] = sig
            n += 1
            print(f"[vision] {rel} → L0（desc={desc[:40]}… ocr={text[:30]}…）")
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    _stop_vl()
    print(f"[vision] 完成 {n} 张（VL-4B 已停）")
    return n


if __name__ == "__main__":
    run()
