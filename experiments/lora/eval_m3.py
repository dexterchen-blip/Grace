#!/usr/bin/env python3
"""Grace V2.5-rc2: 器官 KV 分区零样本验证 v2（修 thinking 抑制/EOS 截停/计时口径）。

分区 ABI:
  [稳定分区] 模板头+persona+mood（真实模板渲染, 快照复用）
  [动态分区] wm+gap+user+assistant 头（每查询重算）

验收门:
  ① 位级等价: 快照恢复 vs 同分块重算 → 答案逐字一致 4/4（KV 恢复正确性）
  ② 诚实等价: vs 一次性全量 prefill → 语义等价 ≥3/4（float 粒度差异, 信息项）
  ③ prefill 加速 ≥25%（仅测 prompt 处理, 不含生成）
  ④ mood 分区独立刷新（定性）
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
REPORT = os.path.join(os.path.dirname(HERE), "lora", "report-m3.json")

PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")

QUERIES = [
    (["主人昨晚熬夜赶due", "感冒了买了药", "数学quiz考了92分"], "我今天要好好休息吗？"),
    (["主人昨晚熬夜赶due", "感冒了买了药", "数学quiz考了92分"], "我这次数学考得怎么样？"),
    ([], "你是谁？"),
    (["包饭的事情定了", "机票出票了3月1号走"], "出行的事都安排好了吗？"),
]

def _sys_content(mood, wm):
    s = PERSONA + "\n" + f"<|audio_pad|>{mood}"
    if wm:
        s += ("\n<|audio_start|>\n" + "\n".join(f"· [主人说] {w}" for w in wm)
              + "\n<|audio_end|>\n<|vision_end|>0.3")
    return s

def _render(tok, sys_content, q):
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": sys_content}, {"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            [{"role": "system", "content": sys_content}, {"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=False)

def _split(tok, mood, wm, q):
    """全模板渲染 → 按 wm 块边界切 稳定/动态 两段（文本级, encode 后拼接为规范 id 序列）。"""
    full = _render(tok, _sys_content(mood, wm), q)
    if wm:
        cut = full.index("<|audio_start|>")
    else:
        cut = full.index("<|im_end|>")   # 无 wm: 稳定段到 system 收尾前
    return full[:cut], full[cut:]

def _prefill(model, ids, cache, timed=False):
    t = time.time()
    lg = model(mx.array([ids]), cache=cache)
    if timed:
        mx.eval(lg)
        return lg, time.time() - t
    return lg

def main():
    t0 = time.time()
    model, tok = load(MODEL, adapter_path=ADAPTER)
    print(f"[M3] 模型加载 {time.time()-t0:.0f}s")
    stop_ids = set()
    for t in ("<|im_end|>",):
        stop_ids.add(int(tok.convert_tokens_to_ids(t)))
    if getattr(tok, "eos_token_id", None):
        stop_ids.add(int(tok.eos_token_id))

    def gen_from_logits(model, cache, logits, max_tokens=80):
        out = []
        for _ in range(max_tokens):
            nxt = int(mx.argmax(logits[0, -1]))
            if nxt in stop_ids:
                break
            out.append(nxt)
            logits = model(mx.array([[nxt]]), cache=cache)
        return tok.decode(out).strip()

    # ---------- ① 位级等价 + ② 诚实等价 ----------
    bit_eq, one_shot_answers = 0, []
    times_full_one, times_part = [], []
    answers = {}
    for idx, (wm, q) in enumerate(QUERIES):
        s_txt, d_txt = _split(tok, "平静 0.3", wm, q)
        s_ids, d_ids = tok.encode(s_txt), tok.encode(d_txt)
        # (a) 全量一次性 prefill（诚实基线）
        c1 = cache_mod.make_prompt_cache(model)
        lg, t_full = _prefill(model, s_ids + d_ids, c1, timed=True)
        a_one = gen_from_logits(model, c1, lg)
        times_full_one.append(t_full)
        # (b) 同分块重算（stable+dyn 两次调用, 干净缓存）
        c2 = cache_mod.make_prompt_cache(model)
        _prefill(model, s_ids, c2)
        lg2 = _prefill(model, d_ids, c2)
        a_two = gen_from_logits(model, c2, lg2)
        answers[idx] = {"q": q, "one_shot": a_one, "two_chunk": a_two}
        print(f"[M3] Q{idx} {q[:16]} | 一次: {a_one[:36]}")
        print(f"[M3]      {'':16} | 分块: {a_two[:36]}")
    # (c) 分区模式: stable 一次 + 快照 → 每查询恢复 + 动态
    s_txt0, _ = _split(tok, "平静 0.3", QUERIES[0][0], QUERIES[0][1])
    pc = cache_mod.make_prompt_cache(model)
    _, prefill_stable = _prefill(model, tok.encode(s_txt0), pc, timed=True)
    def _snap(caches):
        """深拷贝快照: KVCache.update 原地 slice 写(引用必被污染), ArraysCache.state 是列表引用。"""
        out = []
        for c in caches:
            st = c.state
            if isinstance(st, tuple):
                out.append(tuple(x.astype(x.dtype) if hasattr(x, "astype") else x for x in st))
            elif isinstance(st, list):
                out.append([x.astype(x.dtype) if hasattr(x, "astype") else x for x in st])
            else:
                out.append(st)
        return out
    snap = _snap(pc)
    for idx, (wm, q) in enumerate(QUERIES):
        s_txt, d_txt = _split(tok, "平静 0.3", wm, q)
        d_ids = tok.encode(d_txt)
        for c, s in zip(pc, snap):
            # ★修复: ArraysCache.state setter 赋列表引用——线性层 cache[i]=新状态会原地改写快照列表
            #   (Q0 生成污染 snap → Q1 起恢复的全是脏状态)。恢复必须赋"新列表"(数组本身功能性不可变)。
            c.state = list(s) if isinstance(s, list) else s
        lg, tp = _prefill(model, d_ids, pc, timed=True)
        times_part.append(tp)
        a_part = gen_from_logits(model, pc, lg)
        answers[idx]["partition"] = a_part
        answers[idx]["bit_eq"] = (a_part == answers[idx]["two_chunk"])
        bit_eq += answers[idx]["bit_eq"]
        print(f"[M3] 分区 Q{idx}: {a_part[:40]} | 位级等价 {'✓' if answers[idx]['bit_eq'] else '✗'}")

    # ---------- ④ mood 分区独立刷新 ----------
    s_worry, d_worry = _split(tok, "担忧 0.7", QUERIES[0][0], QUERIES[0][1])
    pw = cache_mod.make_prompt_cache(model)
    _prefill(model, tok.encode(s_worry), pw)
    a_mood = gen_from_logits(model, pw, _prefill(model, tok.encode(d_worry), pw))
    print(f"[M3] mood=担忧: {a_mood[:56]}")

    # ---------- 门 ----------
    tf = sorted(times_full_one)[len(times_full_one)//2]
    tp = sorted(times_part)[len(times_part)//2]
    speedup = (tf - tp) / tf * 100 if tf else 0
    one_eq = sum(1 for i in answers if answers[i]["partition"] == answers[i]["one_shot"])
    gates = {"bit_equivalence": f"{bit_eq}/4", "bit_pass": bit_eq >= 4,
             "one_shot_exact": f"{one_eq}/4", "prefill_speedup_pct": round(speedup, 1),
             "speedup_pass": speedup >= 25,
             "mood_partition_refresh": a_mood != answers[0]["partition"]}
    rep = {"version": "grace-v2.5-rc2 (M3 KV partitions v2)",
           "stable_tokens": len(tok.encode(s_txt0)),
           "prefill_stable_s": round(prefill_stable, 3),
           "times_full_one_shot_s": [round(t, 3) for t in times_full_one],
           "times_partition_dyn_s": [round(t, 3) for t in times_part],
           "answers": answers, "mood_worry_answer": a_mood, "gates": gates,
           "elapsed_s": round(time.time() - t0, 1)}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[M3] 位级等价 {bit_eq}/4 | 一次性vs分区精确 {one_eq}/4 | "
          f"prefill 中位 {tf:.3f}s→{tp:.3f}s 加速 {speedup:.0f}%")
    ok = gates["bit_pass"] and gates["speedup_pass"] and gates["mood_partition_refresh"]
    print(f"[M3] 门: {'全过 ✓' if ok else '有未过项 ✗'} | 报告 {REPORT}")

if __name__ == "__main__":
    main()
