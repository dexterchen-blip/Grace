#!/usr/bin/env python3
"""Grace V2.5 图谱→sidecar 向量化大测试（用户提案 §15：情绪/记忆图谱+L2 部分检索 → 塞进模型）。

假设: 图谱本来就是联想记忆(实体↔情绪边/三元组), N3 delta-rule sidecar 可以存储它——
  训练对: k=embed(trigger/predicate-object 描述), v=embed(实体+情绪上下文)
  检索: 新消息 embed → W recall → 最近实体(替代 sqlite 图遍历的向量快速路径)

运行环境: llama-cpp venv(llama_cpp+numpy)。5AM 自动化全量 / --limit 小样冒烟。
评估: 留出 15% 边做探针, top-1/top-3 实体命中(vs 随机基线) + 单查询延迟对照。
"""
import os, sys, json, time, argparse, sqlite3
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from neural_sidecar import DeltaSidecar

L2_DB = os.path.join(REPO, "memory", "L2_semantic", "l2.db")
EMBED_PATH = os.path.join(REPO, "models", "embed", "bge-m3-q8_0.gguf")
OUT_DIR = os.path.join(REPO, "exchange", "grace")

def _embedder():
    from llama_cpp import Llama
    return Llama(model_path=EMBED_PATH, embedding=True, n_ctx=512, verbose=False)

def embed(llm, texts):
    vecs = llm.embed(texts)
    return np.asarray(vecs, dtype=np.float32)

def load_edges(db_path):
    """图谱边 → 训练对 (描述文本, 实体+情绪/关系目标文本)。"""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    pairs = []
    for r in con.execute("SELECT entity, mood_label, intensity, trigger FROM mood_graph "
                         "WHERE trigger IS NOT NULL AND length(trigger) >= 6"):
        v = f"{r['entity']} {r['mood_label']} {r['intensity']:.1f}"
        pairs.append(("mood", r["trigger"][:120], v))
    for r in con.execute("SELECT s, r, o FROM relations "
                         "WHERE s IS NOT NULL AND o IS NOT NULL"):
        pairs.append(("rel", f"{r['r']} {r['o']}".strip()[:120], str(r["s"])))
    for r in con.execute("SELECT name FROM entities WHERE name IS NOT NULL AND length(name) >= 2"):
        # 实体自反对(保底: 实体名 → 实体), 让每个实体至少有一条 v 可命中
        pairs.append(("ent", str(r["name"])[:120], str(r["name"])[:120]))
    con.close()
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=全量; 小样冒烟用")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--holdout", type=float, default=0.15)
    a = ap.parse_args()
    t0 = time.time()
    pairs = load_edges(L2_DB)
    if a.limit:
        pairs = pairs[:a.limit]
    print(f"[GS] 图谱边加载: {len(pairs)} 对 (mood+rel+ent)")
    llm = _embedder()
    # 嵌入(批)
    K = embed(llm, [p[1] for p in pairs])
    V = embed(llm, [p[2] for p in pairs])
    print(f"[GS] 嵌入完成 {K.shape} ({time.time()-t0:.0f}s)")
    # 归一化
    K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)

    # 留出评估集(按实体目标分层近似: 随机即可, 报告里注明)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(pairs))
    n_hold = max(4, int(len(pairs) * a.holdout))
    hold, train = idx[:n_hold], idx[n_hold:]

    # 训练 sidecar
    sc = DeltaSidecar(d=K.shape[1], eta=0.6)
    for ep in range(a.epochs):
        for i in train:
            sc.update(K[i], V[i])
    fid = sc.fidelity([K[i] for i in train], [V[i] for i in train])
    print(f"[GS] sidecar 训练: {len(train)} 对 × {a.epochs} epoch | 训练保真 {fid:.3f}")

    # 评估: 探针 k → W recall → 与全部 v(含训练值)余弦最近 → 命中实体?
    allV = V
    all_targets = [pairs[i][2] for i in range(len(pairs))]
    hits1, hits3, lat = 0, 0, []
    for i in hold:
        t1 = time.time()
        v_hat = sc.recall(K[i])
        v_hat = v_hat / (np.linalg.norm(v_hat) + 1e-8)
        sims = allV @ v_hat
        lat.append((time.time() - t1) * 1000)
        top3 = np.argsort(-sims)[:3]
        truth = pairs[i][2]
        pred = [all_targets[j] for j in top3]
        if pred[0] == truth:
            hits1 += 1
        if truth in pred:
            hits3 += 1
    n = len(hold)
    # 基线: 随机猜测命中率(唯一目标数倒数)
    uniq = len(set(all_targets))
    rep = {"version": "graph→sidecar vectorization", "pairs": len(pairs),
           "train": len(train), "holdout": n, "epochs": a.epochs,
           "train_fidelity": round(float(fid), 3),
           "top1": round(float(hits1 / n), 3), "top3": round(float(hits3 / n), 3),
           "random_baseline": round(1.0 / int(uniq), 5), "unique_targets": uniq,
           "latency_ms_p50": round(float(np.median(lat)), 2),
           "sidecar_dim": K.shape[1], "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    sc.save(os.path.join(OUT_DIR, "graph-sidecar.npz"))
    with open(os.path.join(OUT_DIR, "graph-sidecar-report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[GS] top1 {rep['top1']:.1%} | top3 {rep['top3']:.1%} | 随机基线 {rep['random_baseline']:.3%} "
          f"| 单查询 {rep['latency_ms_p50']}ms | 报告 graph-sidecar-report.json")

if __name__ == "__main__":
    main()
