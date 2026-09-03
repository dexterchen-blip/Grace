#!/usr/bin/env python3
"""主人对话注入生成器(2026-09-04 用户: 开一个 LLM 模拟我的对话注入)。

dialogue 断点产物的根问题 = 主人输入从书库文本挑(邮件/系统/群聊碎片, 不是"我对雷姆说的话")。
正式系统里她面对的是真实对话——压测须注入"LLM 模拟我(陈泽)对她说的话"才构成对话场景。

用法: ./run.sh .venv/bin/python3 v2/stress/build_dialogue_inputs.py [day...]
  默认只生成断点天(11/22/33); 8100 离线则跳过(断点回退旧逻辑, 不阻塞)。
  输出: 写 inputs-v2/day-NNN.json["dialogue_inputs"] = [{situation, owner}] 3 条。
"""
import json
import os
import re
import sys
import time
import urllib.request as _ur

STRESS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "experiments", "run", "stress")
INPUTS = os.path.join(STRESS_ROOT, "inputs-v2")
SERVER = "http://127.0.0.1:8100/v1/chat/completions"

# 我的真实口语锚(书库抽取, 2026-09-04)
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


def _gen_owner_lines(day_text: str) -> list[str]:
    body = json.dumps({
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "messages": [{"role": "system", "content": _SYS},
                     {"role": "user", "content": f"我今天(片段)：\n{day_text[:900]}"}],
        "max_tokens": 120, "temperature": 0.85,
    }).encode("utf-8")
    req = _ur.Request(SERVER, data=body, headers={"Content-Type": "application/json"})
    with _ur.urlopen(req, timeout=90) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    txt = out["choices"][0]["message"]["content"]
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    return [l[:60] for l in lines[:3]]


def main() -> None:
    days = [int(a) for a in sys.argv[1:]] or [11, 22, 33]
    if not _alive():
        print("[owner] 8100 离线, 跳过(断点将回退旧 dialogue 逻辑)")
        return
    for d in days:
        fp = os.path.join(INPUTS, f"day-{d:03d}.json")
        if not os.path.isfile(fp):
            continue
        rec = json.load(open(fp, encoding="utf-8"))
        msgs = [m.get("text", "") for m in rec.get("messages", []) if m.get("text")]
        # 当天摘要: 取带情绪的/长的前几条(邮箱 UI/系统滤掉)
        frag = []
        for t in msgs:
            if re.search(r"@|noreply|threads shown|UCSB \w+@|http|^\[", t):
                continue
            frag.append(t)
        if not frag:
            frag = msgs[:10]
        try:
            lines = _gen_owner_lines("\n".join(frag[-14:]))
        except Exception as e:  # noqa: BLE001
            print(f"  [owner] day{d:03d} 生成失败: {e}")
            continue
        if not lines:
            continue
        rec["dialogue_inputs"] = [{"situation": s} for s in lines]
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        print(f"  ✓ day-{d:03d}: {len(lines)} 条主人话注入")
        for s in lines:
            print(f"      主人: {s}")


if __name__ == "__main__":
    main()
