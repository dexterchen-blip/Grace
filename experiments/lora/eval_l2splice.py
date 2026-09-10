#!/usr/bin/env python3
"""Grace V2.5 L2-KV-splice PoC（设计稿 §14.2——L2 内容进模型的第一次实测）。

架构诚实修正: Qwen3.8 混合注意力(48线性+16全注意)下, 块 KV 不能任意位置拼接——
线性层递归状态不满足可加性(块单独编码从零态出发≠从前缀态出发)。
正确形态 = **前缀条件化块缓存**: 每块在 [稳定前缀] 之后编码一次, 存为 KV 分区快照;
查询时恢复快照 → 直接喂 user → 生成。块缓存跨查询复用(稳定前缀不变时)。

PoC 对照:
  A 全量: 每查询 prefill [stable][c1][c2][user] → 生成(基准)
  B 拼接: 预编码 [stable]→快照0, [stable][c1]→快照1, [stable][c1][c2]→快照2(一次性)
          每查询恢复快照2 → prefill [user] → 生成
验收: ①答案位级一致 ②prefill 加速 ≥40% ③块快照内存可存(486 块外推)
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "..")
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as cache_mod

MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
ADAPTER = os.path.join(os.path.dirname(HERE), "lora", "adapters", "organ-m1-v2")
REPORT = os.path.join(os.path.dirname(HERE), "lora", "report-l2splice.json")

PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")
CHUNKS = [
    "主人上周和学姐确认了接机安排，3月3日下午两点的航班到洛杉矶。学姐说可以直接来接。",
    "宿舍的MicroFridge租好了，是mini冰箱加微波炉一体的那种，月租十五美元，押金五十。",
]
QUERIES = ["接机的事情安排好了吗？", "宿舍的电器都齐了吗？"]

def _render(tok, sys_c, q):
    try:
        return tok.apply_chat_template([{"role": "system", "content": sys_c},
                                        {"role": "user", "content": q}],
                                       add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "system", "content": sys_c},
                                        {"role": "user", "content": q}],
                                       add_generation_prompt=True, tokenize=False)

def _snap(caches):
    out = []
    for c in caches:
        st = c.state
        if isinstance(st, tuple):
            out.append(tuple(x.astype(x.dtype) for x in st))
        elif isinstance(st, list):
            out.append([x.astype(x.dtype) if hasattr(x, "astype") else x for x in st])
        else:
            out.append(st)
    return out

def _restore(caches, snap):
    for c, s in zip(caches, snap):
        c.state = list(s) if isinstance(s, list) else s     # rc2 教训: 列表引用必须新分配

def _nbytes(snap):
    tot = 0
    for s in snap:
        items = s if isinstance(s, (list, tuple)) else [s]
        for x in items:
            if x is not None and hasattr(x, "nbytes"):
                tot += x.nbytes
    return tot

def main():
    t0 = time.time()
    model, tok = load(MODEL, adapter_path=ADAPTER)
    im_end = int(tok.convert_tokens_to_ids("<|im_end|>"))

    def gen_from(logits, cache, max_tokens=60):
        out = []
        for _ in range(max_tokens):
            nx = int(mx.argmax(logits[0, -1]))
            if nx == im_end:
                break
            out.append(nx)
            logits = model(mx.array([[nx]]), cache=cache)
        return tok.decode(out).strip()

    sys_c = PERSONA + "\n<|audio_pad|>平静 0.3"
    stable_ids = tok.encode(_render(tok, sys_c, "X").split("<|im_start|>user")[0])
    chunk_ids = [tok.encode(c) for c in CHUNKS]
    q_texts = [_render(tok, sys_c, q).split("<|im_start|>user")[1] for q in QUERIES]

    # ---------- A 全量基准 ----------
    a_ans, a_times = [], []
    for q_t in q_texts:
        ids = stable_ids + chunk_ids[0] + chunk_ids[1] + tok.encode(q_t)
        c = cache_mod.make_prompt_cache(model)
        t1 = time.time()
        lg = model(mx.array([ids]), cache=c)
        mx.eval(lg)
        dt = time.time() - t1
        a_ans.append(gen_from(lg, c)); a_times.append(round(dt, 3))
        print(f"[L2S] A 全量[{dt:.2f}s]: {a_ans[-1][:42]}")

    # ---------- B 前缀条件化块缓存 ----------
    pc = cache_mod.make_prompt_cache(model)
    t1 = time.time()
    model(mx.array([stable_ids]), cache=pc)
    snap0 = _snap(pc)
    model(mx.array([chunk_ids[0]]), cache=pc)
    snap1 = _snap(pc)                                   # [stable][c1] 的块缓存
    model(mx.array([chunk_ids[1]]), cache=pc)
    snap2 = _snap(pc)                                   # [stable][c1][c2]
    mx.eval([v for s in (snap2,) for v in (s if isinstance(s, (list, tuple)) else [s])])
    precompute_s = time.time() - t1
    chunk_bytes = _nbytes(snap1)
    b_ans, b_times = [], []
    for q_t in q_texts:
        _restore(pc, snap2)
        t1 = time.time()
        lg = model(mx.array([tok.encode(q_t)]), cache=pc)
        mx.eval(lg)
        dt = time.time() - t1
        b_ans.append(gen_from(lg, c_ := pc)); b_times.append(round(dt, 3))
        print(f"[L2S] B 拼接[{dt:.2f}s]: {b_ans[-1][:42]}")

    eq = [a == b for a, b in zip(a_ans, b_ans)]
    tA = sum(a_times); tB = sum(b_times)
    speedup = (tA - tB) / tA * 100 if tA else 0
    gates = {"bit_equal": f"{sum(eq)}/{len(eq)}", "bit_pass": all(eq),
             "prefill_speedup_pct": round(speedup, 1), "speedup_pass": speedup >= 40,
             "chunk_cache_mb": round(chunk_bytes / 1e6, 1)}
    rep = {"version": "L2-as-KV-splice PoC (prefix-conditioned chunk cache)",
           "chunk_tokens": [len(c) for c in chunk_ids], "stable_tokens": len(stable_ids),
           "precompute_s": round(precompute_s, 2), "A_times": a_times, "B_times": b_times,
           "A_answers": a_ans, "B_answers": b_ans, "gates": gates,
           "extrapolation_486_chunks": f"{round(chunk_bytes/1e6*486/1024, 1)}GB (fp32)",
           "elapsed_s": round(time.time() - t0, 1)}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[L2S] 位级一致 {sum(eq)}/{len(eq)} | prefill {tA:.2f}s→{tB:.2f}s 加速 {speedup:.0f}% "
          f"| 块缓存 {gates['chunk_cache_mb']}MB/块 | 486块外推 {rep['extrapolation_486_chunks']}")
    ok = gates["bit_pass"] and gates["speedup_pass"]
    print(f"[L2S] 门: {'全过 ✓' if ok else '有未过项 ✗'} | 报告 {REPORT}")

if __name__ == "__main__":
    main()
