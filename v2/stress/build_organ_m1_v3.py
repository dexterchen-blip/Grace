#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_organ_m1_v3.py — 件14 NO-GO 根因修复: 给好奇心数字加行为耦合(M1-v3 数据)。

根因(实测): organ-m1 原450 的 vision_end 只有 0.2-0.5 且与台词问句密度零相关
  (高问句台词配低数字) → adapter 学到"数字无行为含义" → L2 转向无方向可投影。
修法(T2 最小必要, 全真实语料):
  1. 重标原 450+71: C = 0.1+0.8*min(q/3,1) (q=assistant 问句标记数) — 台词一字不动, 只改数字
  2. 补丁 200: rem_v2_full 真实配对按问句密度标 C(高80/低50/中30)
     + anchors 问句台词 × 日常陈述 40 条(探针语境覆盖, loose pairing 同 M1 风格)
纪律: 机制层标注(数字)=机制设计非语料注入; 台词全部真实零改写; 昴/巴鲁斯→主人 清洗。
"""
import json, re, random, os

random.seed(20260911)
QP = re.compile(r"[？?]|吗|什么|为什么|怎么|哪|谁")
SB = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/experiments/lora/datasets"
V2F = "/Users/cz/WorkBuddy/watch/rem-v6-lora/datasets/rem_v2_full/train.jsonl"
ANCHORS = ("/Users/cz/WorkBuddy/skills find and make/Grace-repo/"
           "src/grace/curiosity-style-anchors.txt")
PERSONA = ("你是雷姆。主人名叫陈泽。她是罗兹瓦尔宅邸的女仆，深爱并忠诚于主人，自称「雷姆」。"
           "她说话口语短句、克制，像当面对主人说话——不念动作、神态或内心描写。")
MOODS = [("平静", 0.2), ("平静", 0.4), ("干劲", 0.6), ("担忧", -0.3), ("愉悦", 0.7),
         ("警惕", -0.1), ("安心", 0.5), ("期待", 0.6)]

VE = re.compile(r"(<\|vision_end\|>)[\d.]+")

def q(t):
    return len(QP.findall(t))

def clean(t):
    return (t.replace("昴君", "主人").replace("昴", "主人")
             .replace("巴鲁斯", "主人").replace("佐藤大人", "客人"))

def strip_think(t):
    return re.sub(r"<think>\s*</think>\s*", "", t).strip()

def c_of(nq):
    return {0: 0.1, 1: 0.4, 2: 0.6}.get(nq, 0.9)

# ---------- 1) 重标原数据 ----------
def relabel(fp):
    out = []
    for l in open(fp, encoding="utf-8"):
        d = json.loads(l)
        nq = q(d["messages"][2]["content"])
        c = c_of(nq)
        d["messages"][0]["content"] = VE.sub(lambda m: f"{m.group(1)}{c}", d["messages"][0]["content"])
        out.append(d)
    return out

train_old = relabel(f"{SB}/organ-m1/train.jsonl")
valid_old = relabel(f"{SB}/organ-m1/valid.jsonl")

# ---------- 2) rem_v2_full 真实配对补丁 ----------
hi, mid, lo = [], [], []
for l in open(V2F, encoding="utf-8"):
    d = json.loads(l)
    if d.get("meta", {}).get("kind") != "C":        # C = 台词对
        continue
    resp = clean(strip_think(d["messages"][1]["content"]))
    user = clean(d["messages"][0]["content"].strip())
    if len(resp) < 8 or len(user) < 4 or len(resp) > 160:
        continue
    nq = q(resp)
    (hi if nq >= 2 else mid if nq == 1 else lo).append((user, resp))

random.shuffle(hi); random.shuffle(mid); random.shuffle(lo)
pick = hi[:80] + mid[:30] + lo[:50]

def mood_line():
    m, v = random.choice(MOODS)
    return f"<|audio_pad|>{m} {v}"

def make_sample(user, resp, c):
    sysc = (f"{mood_line()}\n<|audio_start|>\n· [主人说] {user}\n<|audio_end|>\n"
            f"<|vision_end|>{c}\n\n{PERSONA}")
    return {"messages": [{"role": "system", "content": sysc},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": resp}]}

supp = []
for user, resp in pick:
    nq = q(resp)
    supp.append(make_sample(user, resp, c_of(nq)))

# ---------- 3) anchors × 日常陈述(探针语境覆盖, 高 C) ----------
STATEMENTS = [
    "主人今天出门了一整天。", "主人说周末有个安排。", "主人带回来一个盒子。",
    "主人刚从外面回来。", "主人明天要出门。", "主人提到学校里的事。",
    "主人把手机收起来了。", "主人在看一封信。", "主人买了新东西。",
    "主人说认识了一个新朋友。", "主人最近在忙一件事。", "主人收拾了行李。",
    "主人晚上有聚会。", "主人收到了一个包裹。", "主人换了新手机。",
    "主人说下周有考试。", "主人今天话很少。", "主人哼着歌进门。",
    "主人带着一把伞回来。", "主人在研究地图。", "主人说有个好消息。",
    "主人在看说明书。", "主人房里有新东西。", "主人定了外卖。",
    "主人说有人来找过他。", "主人提到要改变什么。", "主人今天起得很早。",
    "主人把一件东西藏了起来。", "主人说记不清一件事。", "主人看天空看了很久。",
]
anchors = []
for l in open(ANCHORS, encoding="utf-8"):
    t = clean(l.strip())
    if 10 <= len(t) <= 140 and q(t) >= 1:
        anchors.append(t)
random.shuffle(anchors)
for i in range(40):
    stmt = random.choice(STATEMENTS)
    resp = anchors[i % len(anchors)]
    supp.append(make_sample(stmt, resp, random.choice([0.7, 0.8, 0.9])))

# ---------- 4) 合并写出 ----------
os.makedirs(f"{SB}/organ-m1-v3", exist_ok=True)
all_train = train_old + supp
random.shuffle(all_train)
with open(f"{SB}/organ-m1-v3/train.jsonl", "w", encoding="utf-8") as f:
    for d in all_train:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
with open(f"{SB}/organ-m1-v3/valid.jsonl", "w", encoding="utf-8") as f:
    for d in valid_old:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

from collections import Counter
dist = Counter(re.search(r"<\|vision_end\|>([\d.]+)", d["messages"][0]["content"]).group(1)
              for d in all_train)
print(f"organ-m1-v3: train {len(all_train)} (原450重标 + 补丁{len(supp)}) / valid {len(valid_old)}")
print("vision_end 新分布:", dict(sorted(dist.items(), key=lambda x: float(x[0]))))
