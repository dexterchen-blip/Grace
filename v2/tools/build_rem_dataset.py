#!/usr/bin/env python3
"""构建雷姆（Rem, Re:Zero）LoRA 训练数据集 —— 参考惠惠 v6-beta 方案。

管线（纯正则零 LLM，不占本地模型）：
  1. print.html（Re:Zero WEB 版全书合订页，11MB 文本）→ HTML 剥壳 → 按章节切文本块
  2. 对话块级提取：切「台词」序列 + 说话人归属
     规则 A  台词在前：`「台词」……雷姆+动词`
     规则 B  台词在后：`雷姆+紧贴动词：「台词」`（中间无逗号/称呼标点 → 排除"雷姆，我问你"污染）
  3. 对话对：雷姆台词 + 同块内前一句非雷姆台词（user=前句, assistant=雷姆）—— 原文真实对话对，语义天然相关
  4. 清洗：zhconv 繁→简、长度 5~150、排除叙述性（雷姆以/雷姆把/雷姆+动作+……）、排除复杂句、去重
  5. 输出 ChatML jsonl（每条带 system 人设，修复惠惠 v4 身份错乱教训）

用法：
  .venv/bin/python3 v2/tools/build_rem_dataset.py [print.html路径] [输出目录]
默认输出：experiments/lora/datasets/rem/{train,valid}.jsonl
"""
from __future__ import annotations
import html as htmlmod
import json
import os
import random
import re
import sys

random.seed(42)

# ---------- 配置 ----------
DEFAULT_IN = "/Users/cz/WorkBuddy/skills find and make/re0-site-data/site/print.html"
SB = os.environ.get("AIAGENT_SANDBOX", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(SB, "experiments", "lora", "datasets", "rem")

SYSTEM_PROMPT = (
    "你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。"
    "你表面冷淡礼貌、实则温柔忠诚，对亲近的人（昴君）直率、带黑色幽默与毒舌吐槽；"
    "自称「蕾姆」，称呼昴为「昴君」，称拉姆为「姐姐大人」。说话短句为主、肯定句多，常用反问。"
)

REM = re.compile(r"(?:雷姆|蕾姆)")

# 说话动词（繁简共用）
VERBS = r"(?:说|道|回|答|应|喊|叫|低语|嘟囔|叹|问|附和|开口|出声|呢喃|苦笑|轻笑|抱怨)"

# 主语位谓语（自称信号：雷姆+谓语 → 雷姆在主语位，是雷姆自称）
PRED = r"(?:是|要|会|能|想|觉得|认为|做|去|来|说|走|不会|不能|一直|已经|现在|再|也|才|就|很|不|没|在|还|只|都|该|应|必须|喜欢|讨厌|爱|请|给|让|和|与|将|把|被|对|向|从|直到|终|终于|依然|仍旧|照样|同样|依旧|岂|可|真|最|果然|反正|好歹)"
# 信号1a（强）：台词开头 雷姆/蕾姆 + 谓语（"雷姆会下龙车去迎击"）
#   ⚠ 句首排除"说/道/走/来/去"（"雷姆说的对/雷姆说的话"=转述，非自称）
PRED_START = r"(?:是|要|会|能|想|觉得|认为|做|不会|不能|一直|已经|现在|再|也|才|就|很|不|没|在|还|只|都|该|应|必须|喜欢|讨厌|爱|请|给|让|和|与|将|把|被|对|向|从|直到|终|终于|依然|仍旧|照样|同样|依旧|岂|可|真|最|果然|反正|好歹)"
SELF_START = re.compile(r"^(?:雷姆|蕾姆)" + PRED_START)
# 信号1b（强）：句首（句末标点后）雷姆/蕾姆 + 谓语（"拼死忍耐吧，巴鲁斯。雷姆会哭的。"）
SELF_SENT = re.compile(r"(?:[。！？…;；])(?:雷姆|蕾姆)" + PRED)
# 信号1c（中置信·双特征）：任意位自称 ∧ 含"昴君"（雷姆专属称呼组合，抽样高纯）
SELF_ANY = re.compile(r"(?:雷姆|蕾姆)" + PRED)

# 规则 A：台词在前 —— 「台词」……(0~10字) 雷姆 + 动词（排除"雷姆说的对/说得对"评价形态）
PAT_A = re.compile(
    r"「([^」\n]{4,150})」\s*[—,，。！!？?…]{0,4}\s*"
    r"(?:雷姆|蕾姆)[^\n「」]{0,10}?" + VERBS +
    r"(?!的?[对错是])(?!说得?对)"
)
# 规则 B：台词在后 —— 雷姆 + 紧贴动词（中间无逗号/称呼标点）：「台词」
PAT_B = re.compile(
    r"(?:雷姆|蕾姆)(?:[^\n「」，,。！!？?、]{0,6})?" + VERBS + r"\s*[：:]\s*「([^」\n]{4,150})」"
)
# 排除：台词后是"雷姆以…/雷姆把…/雷姆+动作+（叙述）" —— 叙述性归属，非直接引语
NARR = re.compile(r"(?:雷姆|蕾姆)(?:以|把|将|用|露|点|摇|皱|低|抬|看|望|转|站|坐|走|伸)")


def strip_html(raw: str) -> str:
    """print.html → 纯文本（保章节结构）。"""
    raw = re.sub(r"(?is)<script.*?</script>", "", raw)
    raw = re.sub(r"(?is)<style.*?</style>", "", raw)
    # 块级/标题标签 → 换行
    raw = re.sub(r"(?i)</?(?:p|div|h1|h2|h3|h4|h5|h6|li|tr|br|blockquote|pre|section|article|main|table)[^>]*>", "\n", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = htmlmod.unescape(raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw


def split_chapters(text: str) -> list[str]:
    """按章节标题切块（mdBook print 页 h1/h2 形式）。"""
    parts = re.split(r"\n{1,2}(?:(?:第[一二三四五六七八九十百\d]+[章话节卷]|[A-Za-z\d]+\s*[-–—]\s*.{0,20}|序章|终章|幕间|番外|短篇|外传).{0,30})\n", text)
    return [p.strip() for p in parts if len(p.strip()) > 200]


def find_speaker(snippet: str) -> str | None:
    """判断 snippet 中雷姆台词归属。返回 'rem' / 'other' / None(无雷姆)。"""
    if not REM.search(snippet):
        return None
    # 先排除叙述性（雷姆以/雷姆把…）—— 视为旁白，不算直接引语
    if NARR.search(snippet):
        return None
    return "rem"


def extract_pairs(chapter: str, max_pairs_per_chapter: int = 120) -> list[tuple[str, str]]:
    """在单章内做对话块级提取：雷姆台词 + 前一句他人台词。

    归属 = 高纯两档信号（2026-08-27 实测：上下文 PAT 信号在繁中文本误伤 ~30%，弃用）：
      信号1a 台词开头雷姆+谓语（"雷姆会…"）
      信号1b 句首标点后雷姆+谓语（"。雷姆会…"）
    称呼位（雷姆，雷姆。…）无谓语跟随 → 自动归他人。
    注意：PAT_A/PAT_B（台词外上下文归属）保留在模块顶部备用，但精度不足，不用于生产。
    """
    tokens: list[dict] = []
    for m in re.finditer(r"「([^」\n]{4,150})」", chapter):
        start, end = m.start(), m.end()
        ctx = chapter[max(0, start - 30):min(len(chapter), end + 30)]
        tokens.append({"text": m.group(1), "ctx": ctx, "pos": start})

    for t in tokens:
        q = t["text"]
        sig1 = bool(SELF_START.search(q) or SELF_SENT.search(q))       # 高纯：句首自称
        sig2 = bool(SELF_ANY.search(q) and "昴君" in q)                 # 中置信双特征：自称∧昴君
        t["rem"] = sig1 or sig2
        t["sig"] = ("self" if sig1 else "") + ("suking" if sig2 else "")

    pairs: list[tuple[str, str]] = []
    prev_other = None
    for t in tokens:
        if t["rem"]:
            if prev_other:
                pairs.append((prev_other, t["text"]))
            prev_other = None            # 雷姆台词后，他人台词游标清空（连续雷姆不配对）
        else:
            prev_other = t["text"]
        if len(pairs) >= max_pairs_per_chapter:
            break
    return pairs


def clean_pair(u: str, a: str) -> tuple[str, str] | None:
    """清洗：繁→简、长度、复杂句、叙述污染、重复标点。"""
    u = zhconv_convert(u)
    a = zhconv_convert(a)
    u = u.strip().strip("「」").strip()
    a = a.strip().strip("「」").strip()
    if not (5 <= len(u) <= 150 and 5 <= len(a) <= 150):
        return None
    if a.count("。") > 2 or u.count("。") > 2:      # 复杂句排除
        return None
    if re.search(r"(雷姆|蕾姆)(以|把|将|用|露|点|摇|皱|低|抬)", a):   # 叙述残留
        return None
    if re.search(r"[<>{}()\[\]【】]", a + u):        # 代码/标记残留
        return None
    return (u, a)


def zhconv_convert(s: str) -> str:
    try:
        from zhconv import convert
        return convert(s, "zh-cn")
    except ImportError:
        return s


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    print(f"[build-rem] 读取 {src} ...")
    raw = open(src, encoding="utf-8").read()
    text = strip_html(raw)
    print(f"[build-rem] 纯文本 {len(text):,} 字符")
    chapters = split_chapters(text)
    print(f"[build-rem] 章节块 {len(chapters)} 个")

    all_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for i, ch in enumerate(chapters):
        for u, a in extract_pairs(ch):
            c = clean_pair(u, a)
            if not c or c in seen:
                continue
            seen.add(c)
            all_pairs.append(c)
    print(f"[build-rem] 原始对话对 {len(all_pairs)} 条（去重后）")

    # 去重（按 assistant 台词）并过滤
    by_asst: dict[str, tuple[str, str]] = {}
    for u, a in all_pairs:
        by_asst.setdefault(a, (u, a))
    final = list(by_asst.values())
    random.shuffle(final)
    n_valid = max(1, int(len(final) * 0.1))
    valid, train = final[:n_valid], final[n_valid:]
    print(f"[build-rem] 训练 {len(train)} / 验证 {len(valid)}")

    for name, rows in (("train", train), ("valid", valid)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for u, a in rows:
                rec = {"messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats = {"chapters": len(chapters), "raw_pairs": len(all_pairs),
             "unique": len(final), "train": len(train), "valid": len(valid)}
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[build-rem] 完成 → {out_dir}  统计: {stats}")
    # 抽 5 条样例展示
    print("\n=== 样例（前 5 条）===")
    for u, a in final[:5]:
        print(f"  U: {u[:50]}")
        print(f"  A: {a[:50]}")
        print()


if __name__ == "__main__":
    main()
