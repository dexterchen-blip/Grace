#!/usr/bin/env python3
"""主人对话注入·全量扩展生成器（2026-09-10 用户："生成一大坨对话，500+"）。

背景: build_dialogue_inputs.py 只给断点天(11/22/33)生成 3 条, 且目标目录 inputs-v2 已过时
(现役轮输入 = inputs-v3, 其 day 文件尚无 dialogue_inputs 字段)。本脚本对 inputs-v3 全部天
× 每天 5 批 × 3 条 ≈ 570 条, 一次到位。

铁律对齐: 合成输入 = 实验层(测完即弃, 只做轮内变量), 雷姆轮内输出才是成长层——与
build_dialogue_inputs.py 同款纪律。

用法: /Users/cz/WorkBuddy/watch/ai-sandbox-stress/run.sh .venv/bin/python3 \
        v2/stress/build_dialogue_scale.py [--batches 5] [--days 0(=全部)]
8100 离线则直接退出(不阻塞, 留旧逻辑回退)。
输出: inputs-v3/day-NNN.json["dialogue_inputs"] = [{"situation": ...}] 15 条/天
"""
import json
import os
import re
import sys
import time
import urllib.request as _ur

HERE = os.path.dirname(os.path.abspath(__file__))
STRESS_ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                           "experiments", "run", "stress")
INPUTS = os.path.join(STRESS_ROOT, "inputs-v3")
SERVER = "http://127.0.0.1:8100/v1/chat/completions"

# 我的真实口语锚(书库抽取, 2026-09-04, 与 build_dialogue_inputs.py 同源)
_STYLE = [
    "出来拿来了吗", "你cpu gpu超频只要不疯就不会罢工[坏笑]",
    "我这个完全找不到了….@送餐", "校外的保险倒是会比校内的便宜点",
]
_SYS = (
    "你是陈泽，一名即将赴美 UCSB 的新生，正在忙签证/选课/生活杂事。你在跟你的女仆助手雷姆说话。\n"
    "要求：\n"
    "1. 口语、简短（10-40 字），像发微信，别书面\n"
    "2. 可以吐槽、调侃、交代事、随口问一句——就是你会对身边人说的话\n"
    "3. 结合下方你今天的经历（提到相关的事）\n"
    "4. 不要解释、不要称呼堆砌、不要每次带'雷姆'——自然就好\n"
    f"你的说话风格参考：{(' / '.join(_STYLE))}\n"
    "只输出 3 句，每句一行。"
)


def _alive() -> bool:
    try:
        with _ur.urlopen("http://127.0.0.1:8100/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _gen_owner_lines(day_text: str, temp: float) -> list[str]:
    body = json.dumps({
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "messages": [{"role": "system", "content": _SYS},
                     {"role": "user", "content": f"我今天(片段)：\n{day_text[:900]}"}],
        "max_tokens": 120, "temperature": temp,
    }).encode("utf-8")
    req = _ur.Request(SERVER, data=body, headers={"Content-Type": "application/json"})
    with _ur.urlopen(req, timeout=90) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    txt = out["choices"][0]["message"]["content"]
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    return [l[:60] for l in lines[:3]]


def _fragments(rec: dict) -> list[str]:
    msgs = [m.get("text", "") for m in rec.get("messages", []) if m.get("text")]
    frag = []
    for t in msgs:
        if re.search(r"@|noreply|threads shown|UCSB \w+@|http|^\[", t):
            continue
        frag.append(t)
    return frag or msgs[:10]


def main() -> None:
    batches, day_limit = 5, 0
    args = sys.argv[1:]
    if "--batches" in args:
        batches = int(args[args.index("--batches") + 1])
    if "--days" in args:
        day_limit = int(args[args.index("--days") + 1])
    if not _alive():
        print("[owner-scale] 8100 离线, 退出(轮将回退旧 dialogue 逻辑)")
        return
    days = sorted(d for d in os.listdir(INPUTS) if re.fullmatch(r"day-\d{3}\.json", d))
    if day_limit:
        days = days[:day_limit]
    total, fail = 0, 0
    t0 = time.time()
    for fp_name in days:
        fp = os.path.join(INPUTS, fp_name)
        try:
            rec = json.load(open(fp, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  [owner-scale] {fp_name} 读取失败: {e}")
            continue
        frag = _fragments(rec)
        all_lines: list[str] = []
        for b in range(batches):
            window = frag[max(0, len(frag) - 14 - b * 4):max(0, len(frag) - b * 4)] or frag[-14:]
            try:
                lines = _gen_owner_lines("\n".join(window), 0.8 + 0.05 * b)
            except Exception:  # noqa: BLE001
                fail += 1
                continue
            all_lines.extend(lines)
            time.sleep(0.5)
        # 天内去重保序
        seen, uniq = set(), []
        for s in all_lines:
            if s and s not in seen:
                seen.add(s)
                uniq.append(s)
        if not uniq:
            continue
        rec["dialogue_inputs"] = [{"situation": s} for s in uniq]
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        total += len(uniq)
        print(f"  ✓ {fp_name}: {len(uniq)} 条 ({total} 累计, {time.time()-t0:.0f}s)", flush=True)
    print(f"[owner-scale] 完成: {total} 条主人话 / {len(days)} 天 / 失败批次 {fail} / {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
