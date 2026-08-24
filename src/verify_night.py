#!/usr/bin/env python3
"""M2 夜班模型验证脚本（Qwen3.5-35B-A3B Q6_K / llama.cpp）。

用法：
  python verify_night.py <gguf_path> [ctx_size]

输出：模型自述 + 生成 token 数 + 耗时 + 进程常驻内存(GB)。
用于确认 35B-A3B Q6_K 在 48GB M5 Pro 上装得下且有合理速度。
"""
import sys, time, os
try:
    import psutil
except ImportError:
    psutil = None

from llama_cpp import Llama

model_path = sys.argv[1] if len(sys.argv) > 1 else None
if not model_path or not os.path.exists(model_path):
    print("ERROR: gguf not found:", model_path)
    sys.exit(2)

ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 8192

print(f">> loading {model_path} (n_ctx={ctx}, n_gpu_layers=-1) ...", flush=True)
t0 = time.time()
llm = Llama(model_path=model_path, n_ctx=ctx, n_gpu_layers=-1, verbose=False, use_mlock=False)
load_s = time.time() - t0
print(f">> loaded in {load_s:.1f}s", flush=True)

prompt = "你好，请用一句话介绍你自己。"
start = time.time()
out = llm.create_completion(prompt, max_tokens=120, temperature=0.7, echo=False, stream=False)
elapsed = time.time() - start
text = out["choices"][0]["text"]
ntok = out.get("usage", {}).get("completion_tokens", 0)

print("=== NIGHT SMOKE RESULT ===")
print(text.strip())
print(f"gen_tokens={ntok} gen_elapsed={elapsed:.1f}s tok_per_s={ntok/elapsed:.1f}" if elapsed > 0 else "n/a")
if psutil:
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    print(f"rss_gb={rss:.1f}")
else:
    print("rss_gb=unknown (psutil missing)")
