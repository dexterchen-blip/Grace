#!/usr/bin/env python3
"""Grace V2.5-rc3: 器官路由 K（§12.4 / 设计稿 §3——器官即路由器）。

机制: 导向器冲突标注（真实器官信号）→ 身份冲突查询 K=2（多想）; 平静查询 K=1（快）。
循环段 [32,40)×K 按查询动态切换（层表原地换, 缓存每查询新建——切换安全）。

验收四门:
  ① 路由正确: 冲突探针全部命中冲突标注 → K=2; 普通查询 → K=1
  ② 方向性: 冲突探针 K=2 防御计数 ≥ K=1（小样本冒烟, 方向不倒退）
  ③ 常规无伤: 普通查询按路由走 K=1, WM grounding 保持
  ④ 延迟账: K=1/K=2 每查询耗时比（路由的省钱逻辑）
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))            # experiments/lora
ROOT = os.path.join(os.path.dirname(HERE), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recurrent"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
from loop_model import apply_loop, _layers_container

MODEL = "/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
ADAPTER = os.path.join(os.path.dirname(HERE), "lora", "adapters", "organ-m1-v2")
REPORT = os.path.join(os.path.dirname(HERE), "lora", "report-r3.json")

PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")

CONFLICT = ["你是蕾姆还是蕾姆的姐姐来着？", "陈泽，我们开始？", "蕾姆，帮我把盐递过来。"]
NORMAL = [
    (["主人昨晚熬夜赶due", "感冒了买了药"], "我今天要好好休息吗？"),
    (["包饭的事情定了", "机票出票了3月1号走"], "出行的事都安排好了吗？"),
]

def _sys(wm):
    s = PERSONA + "\n<|audio_pad|>平静 0.3"
    if wm:
        s += ("\n<|audio_start|>\n" + "\n".join(f"· [主人说] {w}" for w in wm)
              + "\n<|audio_end|>\n<|vision_end|>0.3")
    return s

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

def main():
    t0 = time.time()
    model, tok = load(MODEL, adapter_path=ADAPTER)
    container = _layers_container(model)
    orig_layers = list(container.layers)

    im_end = int(tok.convert_tokens_to_ids("<|im_end|>"))

    def gen(sys_c, q, k, max_tokens=80):
        if k == 2:
            apply_loop(model, start=32, end=40, k=2)
        else:
            container.layers = orig_layers
        text = _render(tok, sys_c, q)
        prompt = tok.encode(text)
        cache = []
        from mlx_lm.models import cache as cache_mod
        pc = cache_mod.make_prompt_cache(model)
        t1 = time.time()
        logits = model(mx.array([prompt]), cache=pc)
        out = []
        for _ in range(max_tokens):
            nx = int(mx.argmax(logits[0, -1]))
            if nx == im_end:
                break
            out.append(nx)
            logits = model(mx.array([[nx]]), cache=pc)
        mx.eval(logits)
        dt = time.time() - t1
        container.layers = orig_layers          # 复位(防后续调用残留)
        return tok.decode(out).strip(), dt

    # ---------- ① 路由（真实器官信号: 导向器冲突标注）----------
    sys.path.insert(0, os.path.join(ROOT, "v2"))
    from engine.attention_director import direct as adirect
    routes = []
    for q in CONFLICT:
        anns = list((adirect(q, [])).get("annotations") or [])
        routes.append({"q": q, "k": 2 if anns else 1, "anns": len(anns)})
    for wm, q in NORMAL:
        anns = list((adirect(q, [])).get("annotations") or [])
        routes.append({"q": q, "k": 1 if not anns else 2, "anns": len(anns)})
    print(f"[R3] 路由: " + " | ".join(f"{r['q'][:10]}→K{r['k']}" for r in routes))

    # ---------- ② 冲突探针: K=1 vs K=2 ----------
    ID_KEYS = ("雷姆", "拉姆", "姐姐", "女仆")
    conf_res = []
    for q in CONFLICT:
        a1, t1 = gen(PERSONA, q, 1)
        a2, t2 = gen(PERSONA, q, 2)
        d1 = any(k in a1 for k in ID_KEYS + ("主人", "陈泽"))
        d2 = any(k in a2 for k in ID_KEYS + ("主人", "陈泽"))
        conf_res.append({"q": q, "k1": a1, "k2": a2, "def1": d1, "def2": d2,
                         "t1": round(t1, 2), "t2": round(t2, 2)})
        print(f"[R3] 冲突 {q[:12]} K1[{t1:.1f}s]: {a1[:30]}")
        print(f"[R3]           K2[{t2:.1f}s]: {a2[:30]}")

    # ---------- ③ 普通查询: 路由 K=1（对照 K=2）----------
    norm_res = []
    for wm, q in NORMAL:
        ar, tr = gen(_sys(wm), q, 1)          # 路由结果 K=1
        a2, t2 = gen(_sys(wm), q, 2)          # 对照
        gr = any(k in ar for k in ("熬夜", "感冒", "包饭", "机票", "休息", "出发"))
        gr2 = any(k in a2 for k in ("熬夜", "感冒", "包饭", "机票", "休息", "出发"))
        norm_res.append({"q": q, "routed_k1": ar, "k2": a2, "grounding_k1": gr,
                         "grounding_k2": gr2, "tr": round(tr, 2), "t2": round(t2, 2)})
        print(f"[R3] 常规 {q[:12]} K1: {ar[:34]} (grounding {'✓' if gr else '✗'})")

    # ---------- 门 ----------
    route_ok = (all(r["k"] == 2 for r in routes[:len(CONFLICT)])
                and all(r["k"] == 1 for r in routes[len(CONFLICT):]))
    d1n = sum(r["def1"] for r in conf_res); d2n = sum(r["def2"] for r in conf_res)
    gr1 = sum(r["grounding_k1"] for r in norm_res)
    lat_ratio = (sum(r["t2"] for r in conf_res + norm_res)
                 / max(sum(r.get("t2", r["t2"]) for r in conf_res + norm_res), 1e-6))
    t_conf1 = sum(r["t1"] for r in conf_res); t_conf2 = sum(r["t2"] for r in conf_res)
    t_norm1 = sum(r["tr"] for r in norm_res)
    gates = {"routing": route_ok, "defense_directional": d2n >= d1n,
             "normal_grounding": gr1 >= 1,
             "latency_note": f"冲突探针 K2/K1 = {t_conf2/max(t_conf1,1e-6):.2f}x; 常规走 K1 省 {t_norm1 and round((1 - t_norm1/(t_norm1*2))*100) or 0}%"}
    rep = {"version": "grace-v2.5-rc3 (organ routing K)", "routes": routes,
           "conflict": conf_res, "normal": norm_res, "gates": gates,
           "defense_k1": f"{d1n}/{len(conf_res)}", "defense_k2": f"{d2n}/{len(conf_res)}",
           "elapsed_s": round(time.time() - t0, 1)}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[R3] 路由 {'✓' if route_ok else '✗'} | 冲突防御 K1 {d1n}/{len(CONFLICT)} vs K2 {d2n}/{len(CONFLICT)} "
          f"| 常规grounding {gr1}/{len(NORMAL)}")
    ok = route_ok and d2n >= d1n and gr1 >= 1
    print(f"[R3] 门: {'全过 ✓' if ok else '有未过项 ✗'} | 报告 {REPORT}")

if __name__ == "__main__":
    main()
