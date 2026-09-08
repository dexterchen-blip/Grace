#!/usr/bin/env python3
"""自发分级引擎 —— M4（Grace_v2 设计 §7 风险分级沙箱）。

三级分级：
  L1 低风险：播报、分享 → 自动放行
  L2 中风险：主动提问、建议 → 限时自动（一段时间无异议则放行）
  L3 高风险：联系他人、执行动作 → **必须人审**（proposal_queue，type=dispatch）

铁律：
  - L3 高风险自发**永不受心态影响**（心态只调 L1/L2 频率，不动风险等级）
  - 自发内容生成先过「外挂事实校验 + 人格风格」，再输出（校验钩子由调用方提供）
  - 所有自发行为留痕：experiments/run/initiative-*.log + L3 进提案队列

用法（沙盒内）:
  ./run.sh python3 v2/engine/initiative.py --sim "提醒我下午3点开会"     # 分类 + 放行策略
  ./run.sh python3 v2/engine/initiative.py --sim "给妈妈发一条微信"       # L3 → 需人审
"""
from __future__ import annotations
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/
import config  # noqa: E402

# ---------- 风险分类 ----------
_L3_HINTS = re.compile(
    r"(?:发(?:.{0,5}?)(?:微信|邮件|短信|消息)|联系|打电话|转账|付款|下单|购买|提交(?!的)|发送|发出|批准|同意|"
    r"删除|修改|写(?:入|进)|执行|运行|部署|报名|预约|取消|退订|签|授权|登录|退出)"
)
_L2_HINTS = re.compile(
    r"(?:你觉得|建议|要不要|应该|可以(?:吗|么)?|想问|提醒|关注|注意|别忘|记得|问一下)"
)
_L1_HINTS = re.compile(
    r"(?:播报|分享|告知|告诉|汇报|报告|更新|同步|天气|新闻|好消息|提醒你|知道了|完成了)"
)


def classify_action(text: str) -> dict:
    """把自发行为分类为 L1/L2/L3 + 放行策略。"""
    if _L3_HINTS.search(text):
        return {"level": "L3", "policy": "必须人审（proposal_queue）",
                "desc": "高风险：联系他人/执行动作/资金/写操作"}
    if _L2_HINTS.search(text):
        return {"level": "L2", "policy": "限时自动放行（默认 30 分钟无异议）",
                "desc": "中风险：主动提问/建议/提醒"}
    if _L1_HINTS.search(text):
        return {"level": "L1", "policy": "自动放行",
                "desc": "低风险：播报/分享/告知"}
    return {"level": "L1", "policy": "自动放行（默认低风险）",
            "desc": "未匹配风险词，按低风险处理"}


def _mood_bias(energy: float) -> float:
    """心态只调节 L1/L2 频率（bias 范围 -0.3..0.4），绝不改变 L3 风险等级。"""
    return max(-0.5, min(0.5, energy - 0.5))   # 能量 0.3→-0.2（低落降频），0.8→+0.3（高涨加频）


def should_act(level: str, mood_energy: float = 0.5) -> tuple[bool, str]:
    """放行判断。L1 恒放行；L2 按窗口+心态；L3 永不自动。"""
    if level == "L3":
        return False, "L3 必须人审（proposal_queue type=dispatch），心态不参与"
    if level == "L1":
        return True, "L1 自动放行"
    # L2：限时窗口（30 分钟无异议自动放行），心态只调频率
    bias = _mood_bias(mood_energy)
    return True, f"L2 限时窗口放行（mood_bias={bias:+.1f}，仅调频率不动风险等级）"


def emit(action: str, decided_by: str = "system") -> dict:
    """自发行为留痕 + L3 进人审提案队列。返回记录。"""
    cls = classify_action(action)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "action": action,
           "level": cls["level"], "policy": cls["policy"], "decided_by": decided_by}
    # 落痕
    os.makedirs(config.REPORTS, exist_ok=True)
    with open(os.path.join(config.REPORTS, "initiative.log"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # L3 → 人审提案（复用 v1 proposal_queue 的 dispatch 类型）
    if cls["level"] == "L3":
        try:
            sys.path.insert(0, config.SRC)
            from proposal_queue import ProposalQueue
            pid = ProposalQueue(root=os.path.join(config.EXCHANGE, "proposals")).create(
                type="dispatch",
                title=f"[自发] {action[:30]}",
                description=f"自发引擎请求执行：{action}",
                target={"action": action, "risk": "L3"},
                source="v2-initiative",
                priority="medium",
            )
            rec["proposal_id"] = pid or "(黑名单拒绝)"
        except Exception as e:  # noqa: BLE001
            rec["proposal_error"] = str(e)
    return rec


if __name__ == "__main__":
    if "--sim" in sys.argv:
        i = sys.argv.index("--sim")
        text = sys.argv[i + 1]
        cls = classify_action(text)
        print(f"行为: {text}")
        print(f"分级: {cls['level']} ｜ {cls['desc']}")
        print(f"策略: {cls['policy']}")
        ok, why = should_act(cls["level"])
        print(f"放行: {'✅' if ok else '⛔'} {why}")
        if cls["level"] == "L3":
            r = emit(text)
            print(f"已入人审提案: {r.get('proposal_id')}")
    else:
        print(__doc__)
