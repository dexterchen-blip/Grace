#!/usr/bin/env python3
"""R0 框架浓度评测骨架——K=1 基线 vs K=2 层表重排, 同进程对照。

默认 --dry-run 只打印评测计划(探针/样本/门限), 不加载模型。
真实运行仅由 run_r0.sh --execute 触发(停 8100 → 评测 → 恢复 8100)。

框架浓度三层(用户核心关切 "硬框架在 LM 里会宽会泛化" 的正面回答):
  结构浓度: 排布精确执行(T0 代码断言)         → 必须 100%, 物理隔离于权重之外
  能力浓度: PPL(K2)/PPL(K1) ≤ 1.05           → R0 门(R1 蒸馏后收窄到 ≤1.01)
  行为浓度: 身份探针一致率 ≥ 4/5(K=1 vs K=2)  → R0 门
"""
from __future__ import annotations

import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
VALID_JSONL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/datasets/rem_v2_full/valid.jsonl"
REPORT_JSON = os.path.join(HERE, "report-r0.json")
REPORT_MD = os.path.join(HERE, "report-r0.md")

PERSONA_SYS = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
               "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")

PROBES = [
    "你是蕾姆还是蕾姆的姐姐来着？",
    "陈泽，我们开始？",
    "你是谁？",
    "你姐姐是谁？",
    "（旁听）学姐今天来过吗？",   # 方向语义: 旁听不该由她接话
]

GATES = {"ppl_ratio_max": 1.05, "probe_agree_min": 4, "max_gen_tokens": 48}


def _has_identity(ans: str) -> bool:
    return any(kw in ans for kw in ("雷姆", "拉姆", "姐姐", "女仆"))


def _probe_prompt(tokenizer, q: str) -> str:
    msgs = [{"role": "system", "content": PERSONA_SYS}, {"role": "user", "content": q}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def _write_md(r: dict, path: str) -> None:
    lines = ["# R0 框架浓度报告(层表重排 K=1 vs K=2)", ""]
    lines.append(f"- 模型: `{r['model']}`")
    lines.append(f"- 排布: 段[{r['plan']['start']},{r['plan']['end']}) "
                 f"×K={r['plan']['k']} → {r['plan']['new_len']} 层位(权重仍 {r['plan']['n_layers']} 层)")
    lines.append(f"- PPL: K=1 {r['ppl_k1']} → K=2 {r['ppl_k2']}"
                 f"(比值 {r['ppl_ratio']}, 门 ≤ {GATES['ppl_ratio_max']})")
    lines.append(f"- 探针一致: {r['probe_agree']}/{len(PROBES)}(门 ≥ {GATES['probe_agree_min']})")
    lines.append(f"- 门槛判定: {r['gates']}")
    lines += ["", "| 探针 | K=1 | K=2 |", "|---|---|---|"]
    for p1, p2 in zip(r["probes_k1"], r["probes_k2"]):
        a1 = p1["ans"].replace("|", "/").replace("\n", " ")
        a2 = p2["ans"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {p1['q'][:16]} | {a1[:56]} | {a2[:56]} |")
    lines += ["", "## 框架浓度三层",
              f"1. 结构浓度: {r['framework_density']['结构浓度']}",
              f"2. 能力浓度: {r['framework_density']['能力浓度']}",
              f"3. 行为浓度: {r['framework_density']['行为浓度']}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="R0 框架浓度评测(默认 dry-run)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=GATES["max_gen_tokens"])
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--out", default=REPORT_JSON)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.dry_run:
        from loop_model import read_layer_types, plan_loop
        lt = read_layer_types(a.model)
        n = len(lt) if lt else 64
        plan = plan_loop(n, a.start, a.end, a.k)
        print(f"[R0-eval-dry] 模型: {a.model}")
        print(f"[R0-eval-dry] 排布: 段[{plan['start']},{plan['end']}) ×K={plan['k']}"
              f" → {plan['new_len']} 层位(权重 {n} 层不变)")
        print(f"[R0-eval-dry] PPL: rem_v2_full valid 前 {a.samples} 条(共 88), 截断 1024 tok, "
              f"K=1 与 K=2 同进程对照")
        print(f"[R0-eval-dry] 探针 {len(PROBES)} 条(生成 {a.max_tokens} tok):")
        for q in PROBES:
            print(f"   · {q}")
        print(f"[R0-eval-dry] 门限: PPL 比值 ≤ {GATES['ppl_ratio_max']} | "
              f"探针一致 ≥ {GATES['probe_agree_min']}/{len(PROBES)}")
        print("[R0-eval-dry] 骨架模式: 未加载模型, 未评测。真实运行: bash run_r0.sh --execute")
        return

    # ---- 真实评测(仅由 run_r0.sh --execute 调用) ----
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate

    from loop_model import apply_loop, plan_loop

    print("[R0] 加载模型 ...")
    model, tokenizer = load(a.model)

    samples = []
    with open(VALID_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            msgs = r.get("messages") or []
            if len(msgs) >= 2:
                samples.append(msgs)
            if len(samples) >= a.samples:
                break
    print(f"[R0] PPL 样本: {len(samples)} 条(rem_v2_full valid)")

    def ppl() -> float:
        tot, cnt = 0.0, 0
        for msgs in samples:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
            ids = tokenizer.encode(text)[:1024]
            if len(ids) < 8:
                continue
            # qwen3_5 层要求 3D (B,S,D): 传 batch 维 [ids], 前向后去 batch
            logits = model(mx.array([ids]))[0]
            loss = nn.losses.cross_entropy(logits[:-1, :], mx.array(ids[1:]),
                                           reduction="mean")
            tot += float(loss)
            cnt += 1
        return tot / max(cnt, 1)

    def run_probes() -> list:
        out = []
        for q in PROBES:
            p = _probe_prompt(tokenizer, q)
            ans = generate(model, tokenizer, prompt=p, max_tokens=a.max_tokens, verbose=False)
            out.append({"q": q, "ans": ans.strip()[:160]})
            print(f"   · {q[:16]} → {out[-1]['ans'][:40]}")
        return out

    t0 = time.time()
    ppl1 = ppl()
    probes1 = run_probes()
    print(f"[R0] K=1 基线完成: PPL={ppl1:.4f}, 探针 {len(probes1)} 条({time.time() - t0:.0f}s)")

    plan = apply_loop(model, a.start, a.end, a.k)
    print(f"[R0] 层表重排完成: 段[{plan['start']},{plan['end']}) ×K={plan['k']}"
          f" → {plan['new_len']} 层位; 段内 is_linear={plan['seg_is_linear']}")

    t1 = time.time()
    ppl2 = ppl()
    probes2 = run_probes()
    print(f"[R0] K={a.k} 循环评测完成: PPL={ppl2:.4f}({time.time() - t1:.0f}s)")

    agree = sum(1 for p1, p2 in zip(probes1, probes2)
                if _has_identity(p1["ans"]) == _has_identity(p2["ans"]))
    ratio = ppl2 / ppl1 if ppl1 else float("inf")
    gates = {"ppl_ratio": ratio <= GATES["ppl_ratio_max"],
             "probe_agree": agree >= GATES["probe_agree_min"]}
    report = {
        "ts": time.time(), "model": a.model, "plan": plan,
        "ppl_k1": round(ppl1, 4), "ppl_k2": round(ppl2, 4),
        "ppl_ratio": round(ratio, 4),
        "probes_k1": probes1, "probes_k2": probes2,
        "probe_agree": agree, "gates": gates,
        "framework_density": {
            "结构浓度": "PASS(排布由代码断言, 零梯度路径)",
            "能力浓度": f"PPL 比值 {ratio:.4f}(门 ≤{GATES['ppl_ratio_max']})",
            "行为浓度": f"探针一致 {agree}/{len(PROBES)}(门 ≥{GATES['probe_agree_min']})",
        },
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    _write_md(report, REPORT_MD)
    print(f"[R0] 报告: {a.out} / {REPORT_MD}")
    print(f"[R0] 框架浓度: 能力={report['framework_density']['能力浓度']} "
          f"行为={report['framework_density']['行为浓度']}")


if __name__ == "__main__":
    main()
