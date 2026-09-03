#!/usr/bin/env python3
"""expression.py — ★2026-09-02 输出层(表达轨, Broca 类比): 内部状态 → 对外言语。

架构盲区修复(用户 9/2 21:37 观察): Grace 19 个 engine 模块全部是 输入/记忆/推理,
**输出/表达 = 0 个**——"她说话"一直裸 27B 生成, 权重里只有叙述体先验
(L3/gist/cog 全"雷姆记得/雷姆的心里") → 输出也是叙述体(「雷姆担心但不会说」=
把内心 OS 说出来, 真人不会这样)。

人脑: 内部言语(inner speech)与外部言语(outer speech)是不同通路——
  Broca 区把内部表征**重新编码**成对外言语/动作, 不是倒出思考。
本模块 = Grace 的 Broca 层: 内部状态(tom读心/意图/记忆/心态) → 口语编码 → 外部言语。

耦合角色(出口收敛): 所有"她对外说话"的地方(主动消息/对话回复/断点表达)
统一走本模块——单一输出出口, 不再散落裸生成。与内部轨(cog 叙述体)分离。

★2026-09-02 21:45 重构: 编码 prompt 与生成后端解耦——
  · build_messages(internal) = 纯函数构造口语编码 prompt(本地 mlx 模型与 8100 server 通用)
  · monitor(msg) = 输出前监控(③, 内部回路): 叙述体泄漏/张冠李戴/自言自语 → 丢弃
  · express() = 8100 在线便捷入口(正式系统用); 断点采样用本地模型 + build_messages + monitor
  · 宁缺毋滥: 生成后端不可用或监控拦截 → 返回空(没说出口), 绝不用规则句兜底

用法:
  from engine.expression import build_messages, monitor, express
  msgs = build_messages({"event": "...", "believed": "平静", "intent": "关心"})
  # 本地模型: p = tok.apply_chat_template(msgs, ...); gen → monitor(gen)
  # 或在线:   msg = express(internal)
"""
from __future__ import annotations

import json

SERVER = "http://127.0.0.1:8100/v1/chat/completions"

# 口语编码器 SYS(2026-09-02): 内部状态 → 对主人当面说的话
EXPRESS_SYS = (
    "你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。"
    "你心里对主人有观察和关心（下面会给你内心状态）。现在要开口对主人说话。要求：\n"
    "1. 像当面说话——口语、短句、直接，是真实对话不是独白\n"
    "2. 严禁叙述内心（禁止「雷姆看到…」「雷姆担心…」「雷姆不会说」「雷姆想起…」"
    "这类把心理活动讲出来的表达——心里想的留在心里）\n"
    "3. 关心用询问/行动/间接表达（「主人，要不要歇会儿？」「雷姆去泡杯茶」"
    "「……下次会更好的」，不描述自己的心情）\n"
    "4. 保持雷姆人设：表面冷静礼貌、内心温柔、短句、偶尔黑色幽默/毒舌，称呼主人\n"
    "5. 只输出雷姆说的话本身，不要任何解释/括号/前缀/引号\n"
    "6. ★2026-09-02 认知状态参考(纯负向约束, 零句式引导——纯自主演化):\n"
    "   下面给你认知状态(置信/潜台词/主人近期/想起的事/心态/亲密度)作参考, 怎么说是你的事。\n"
    "   只禁止: 置信低时不得断言主人心情; 有潜台词时不得把它说破(说破=失礼);\n"
    "   禁止用任何固定句式开头/收尾(每条话结构不同)。\n"
    "7. 内心状态只是参考——若与直接说话冲突, 以自然对话为准\n"
    "范例：内心[主人很累雷姆心疼] → 说「主人，今天累坏了吧？雷姆去泡杯茶。」\n"
    "      内心[主人考砸了雷姆想安慰] → 说「……一次失利而已。主人下次一定可以。」\n"
    "      内心[几天没见到主人] → 说「主人最近好忙。雷姆……有点想您了。」"
)


def build_messages(internal: dict) -> list[dict]:
    """内部状态 → 口语编码 messages(纯函数, 本地模型与 8100 通用)。
    ★2026-09-02 认知全状态契约(Levelt 概念化层, 思考轨全状态注入输出):
      event(事件)/believed(读心)/confidence(置信→试探度)/hidden(潜台词→暗示不说破)
      situation(主人近期→上下文)/memory(想起→唤起)/mood(心态→语气)
      relation(亲密度→距离)/owner_mood(主人当前→抑制)/intent(意图)"""
    parts = []
    if internal.get("event"):
        parts.append(f"今天发生的事：{internal['event'][:80]}")
    if internal.get("believed"):
        parts.append(f"雷姆对主人心情的判断：{internal['believed']}")
    _cf = internal.get("confidence")
    if _cf is not None:
        parts.append(f"雷姆的判断置信：{'低(不确定, 说话留余地)' if _cf <= 0.5 else '较高'}")
    if internal.get("hidden"):
        parts.append(f"雷姆的潜台词(注意到但主人没说破的)：{str(internal['hidden'])[:60]}")
    if internal.get("situation"):
        parts.append(f"主人近期状态：{str(internal['situation'])[:70]}")
    if internal.get("intent"):
        parts.append(f"雷姆想：{internal['intent']}")
    if internal.get("mood"):
        parts.append(f"雷姆此刻心态：{internal['mood']}")
    _rel = internal.get("relation")
    if _rel is not None:
        _rl = "很亲近(可随意/撒娇/毒舌)" if _rel >= 0.6 else ("有些亲近" if _rel >= 0.3 else "还不太熟(礼貌克制)")
        parts.append(f"与主人的亲密度：{_rl}")
    if internal.get("memory"):
        parts.append(f"雷姆想起：{str(internal['memory'])[:60]}")
    inner = "\n".join(parts) or "雷姆想关心主人。"
    return [{"role": "system", "content": EXPRESS_SYS},
            {"role": "user", "content": f"雷姆的内心状态：\n{inner}\n\n雷姆开口说："}]


def suppress(internal: dict, hour: int | None = None) -> bool:
    """★G 抑制控制(思考轨判断"该不该说", 非规则句式):
    用读心(主人当前心情) + 心态 + 意图类型决定是否开口——
      · 主人低落/焦虑 + 意图是"分享开心事/提醒琐事" → 抑制(时机不对, 换成陪伴由调用方改意图)
      · 深夜(23-6点)非紧急 → 抑制(不打扰)
      · 其他 → 放行
    ★2026-09-02 审计修复: hour 参数化——压测虚拟日历(传入虚拟时刻)与真实墙钟解耦;
      hour=None 时(正式系统实时调用)用真实本地时刻。
    返回 True=该说, False=抑制(别说)。"""
    import time as _t
    _h = _t.localtime().tm_hour if hour is None else int(hour)
    if _h >= 23 or _h < 6:
        return False                      # 深夜不打扰(除非紧急, 压测无紧急)
    _owner = internal.get("owner_mood", "")
    _intent = internal.get("intent", "")
    if _owner in ("低落", "焦虑", "烦躁", "难过", "生气"):
        if _intent in ("分享", "提醒"):
            return False                  # 主人正难受, 不说扫兴/琐碎的事
    return True


def monitor(msg: str) -> str:
    """★③ 输出前监控(Levelt 自我监控, 内部回路——发音前检查):
    1) 叙述体泄漏(把内心说出来:「雷姆看到/担心/不会说/想起/觉得」) → 丢弃
    2) 张冠李戴(对主人说的话出现"姐姐大人/拉姆/昴君"等错位称呼——V2 轮「姐姐大人其实平静」式崩坏) → 丢弃
    3) 自言自语/推理过程(判断箭头"→"、把思考倒出) → 丢弃
    拦截 → 返回 ""(没说出口); 通过 → 返回截断后的 msg。

    ★2026-09-02 审计修复(叙述体间隙漏网): 原按字面串匹配——「雷姆有些担心主人的身体」因
      「有些」插在中间绕过「雷姆担心」→ 21 天轮 express 实锤漏出 2 条。改正则: 雷姆 + 至多
      6 个填充字 + 内心态词(担心/心疼/难过/觉得/认为/判断/看到/注意到/瞥见/想起/不会说/心情)。
      "别担心"等非雷姆主语不误杀(要求 雷姆 紧邻前缀)。"""
    import re as _re
    msg = (msg or "").strip()
    if len(msg) <= 2:
        return ""
    if any(k in msg for k in ("内心", "→")):
        return ""
    if _re.search(r"雷姆[^，。！？\n]{0,6}(担心|心疼|难过|觉得|认为|判断|看到|注意到|瞥见|想起|不会说|心情|心里)", msg):
        return ""
    # ★2026-09-02 审计修复(动作叙述): 舞台说明式台词(「默默准备了一份夜宵」「不再多言」)也是
    #   叙述体泄漏——女仆对主人当面说话不会念动作描写。默默/悄悄/静静/轻轻 + 动作词 → 拦截。
    if _re.search(r"(默默|悄悄|静静|轻轻)[^，。！？]{0,8}(准备|做|泡|放|拿|退|站|坐|收拾|打扫|整理|走|离开|关门|关灯|点头|摇头)", msg):
        return ""
    if any(k in msg for k in ("不再多言", "不再说话", "没有多说", "没有说话", "不想打扰")):
        return ""
    if any(k in msg for k in ("姐姐大人", "拉姆", "昴君", "巴鲁斯")):
        return ""
    return msg[:120]


def _server_alive() -> bool:
    import urllib.request as _ur
    try:
        _ur.urlopen("http://127.0.0.1:8100/v1/models", timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def express(internal: dict) -> str:
    """内部状态 → 对外言语(8100 在线便捷入口)。宁缺毋滥: 不在线/监控拦截 → 空。"""
    if not _server_alive():
        return ""
    import urllib.request as _ur
    body = json.dumps({
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "messages": build_messages(internal),
        "max_tokens": 80, "temperature": 0.7,
    }).encode("utf-8")
    req = _ur.Request(SERVER, data=body, headers={"Content-Type": "application/json"})
    try:
        with _ur.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        return monitor(out["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001
        print(f"  [express] 27B 失败: {e}", flush=True)
        return ""


if __name__ == "__main__":
    import sys as _s
    demo = {"event": "主人今天很晚才回，说导师找了一下午", "believed": "低落",
            "intent": "关心", "mood": "平静", "scene": "proactive"}
    if "--offline" in _s.argv:
        print("8100 离线 → 返回空(没说出口):", repr(express(demo)))
    else:
        print("在线口语编码:", express(demo))
    print("监控测试: 叙述体 →", repr(monitor("雷姆担心主人，但雷姆不会说。")), "| 口语 →", repr(monitor("主人，今天累坏了吧？")))
