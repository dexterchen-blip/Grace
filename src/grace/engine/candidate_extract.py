#!/usr/bin/env python3
"""训练候选提炼 —— 从记忆源筛出"拟用于今晚 LoRA"的风格样本集（按天排序）。

双轨一致性铁律（Grace_v2 设计 §5）：
  事实性内容（谁/什么/何时/偏好）只进 L0/L2/L3，绝不进 LoRA；
  本模块在数据侧防渗漏——把事实类样本过滤掉，从源头不让事实进训练集。

源信息按天排序（M2 修正，2026-08-27 用户拍板）：
  - 记忆记录带 epoch/ts 时间戳 → 按本地日期(Asia/Shanghai)分组、升序排列
  - CLI `--days` 看按天统计；`--date YYYY-MM-DD` 只提炼那一天
  - 多源支持：沙盒 L0（默认）或正式系统 L0 只读引用（--l0 <path>，读 OK 写零）

产出：type="lora_train" 的候选提案（pending），落沙盒 exchange/proposals/pending/，
复用 v1 人审闸门状态机语义（pending → approved / rejected）。
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # v2/（config.py）
import config  # noqa: E402

_LOCAL_TZ = timezone(timedelta(hours=8))   # Asia/Shanghai（机器时区，与 launchd 任务一致）


# ---------- 时间归一化：把记录归到本地日期 ----------
def _local_date(rec: dict) -> str | None:
    """从记录提取本地日期 YYYY-MM-DD（优先 epoch 秒 → ts ISO 字符串）。"""
    ep = rec.get("epoch")
    if isinstance(ep, (int, float)) and ep > 0:
        try:
            return datetime.fromtimestamp(ep, tz=timezone.utc).astimezone(_LOCAL_TZ).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            pass
    ts = rec.get("ts", "")
    if ts:
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(_LOCAL_TZ).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


# ---------- 风格 / 事实启发式分类 ----------
_FACT_HINTS = re.compile(
    r"(?:\d{4}[-/年]\d{1,2}|周[一二三四五六日天]|今天|明天|昨天|"
    r"(?:约|预订|定了|改到|在|于)\s*[\d:点]|"
    r"(?:邮箱|邮件|课程|作业|截止|ddl|due|会议|预约|航班|车次|密码|账号|地址|电话|"
    r"医生|体检|疫苗|签证|I-20|学费|汇率|成绩|GPA|选课|宿舍|租房|"
    r"张三|李四|王五|陈泽|Ze|Chen|"
    r"(?:¥|￥|\$|USD|RMB|CNY)\s?\d|\d+\s?(?:美金|美元|人民币|块|元)|"
    r"学期|学年|月份|季度|开学|放假|期中|期末|毕业|截止日期|有效期)"
    r")",
    re.IGNORECASE,
)
_STYLE_HINTS = re.compile(
    r"(蕾姆|雷姆|巴鲁斯|昴君|昴|姐姐大人|鬼族|女仆|呜呣|——|？！|…|是的是的|不行不行|哼|呐|嘛|"
    r"(?:这不是|难道|难道说|话说|话说回来|说起来)"
    r")",
)


def classify(text: str) -> dict:
    """把一条样本分类为 style / fact / mixed / neutral，并给风险分级。"""
    fact_hits = _FACT_HINTS.findall(text)
    style_hits = _STYLE_HINTS.findall(text)
    if fact_hits and not style_hits:
        kind, risk = "fact", "high"
    elif fact_hits and style_hits:
        kind, risk = "mixed", "medium"
    elif style_hits:
        kind, risk = "style", "low"
    else:
        kind, risk = "neutral", "low"
    return {"kind": kind, "risk": risk, "fact_hits": len(fact_hits), "style_hits": len(style_hits)}


class CandidateQueue:
    """训练候选提案队列（人审驯服自训练的第一步闸门）。

    复用 v1 proposal 目录布局（exchange/proposals/{status}/{id}.json）与
    pending → approved/rejected 状态机；type=lora_train，id 带日期：lora-YYYYMMDD-<seq>。
    """

    def __init__(self, root: str = None):
        self.root = root or config.PROPOSALS

    def _path(self, status: str, cid: str) -> str:
        return os.path.join(self.root, status, f"{cid}.json")

    def _find(self, cid: str) -> str | None:
        for s in config.CANDIDATE_STATUSES:
            p = self._path(s, cid)
            if os.path.exists(p):
                return p
        return None

    def load(self, cid: str) -> dict | None:
        p = self._find(cid)
        if not p:
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def list(self, status: str = "pending") -> list[dict]:
        d = os.path.join(self.root, status)
        out = []
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d), reverse=True):
                if fn.endswith(".json"):
                    with open(os.path.join(d, fn), encoding="utf-8") as f:
                        out.append(json.load(f))
        return out

    def create(self, *, date: str, title: str, description: str, samples: list[str],
               kind_stats: dict, source: str = "night-candidate-extract",
               priority: str = "medium", expires_in_hours: float = 24.0) -> str:
        """创建候选。date = 源记忆的本地日期（YYYY-MM-DD），id 带日期便于按天排序。"""
        seq = int(time.time() * 1000) % 100000
        cid = f"lora-{date.replace('-', '')}-{seq:05d}"
        rec = {
            "id": cid,
            "type": config.TRAIN_CANDIDATE_TYPE,
            "date": date,                          # 源记忆日期（按天排序/7 天滑动窗口的依据）
            "title": title,
            "description": description,
            "samples": samples,
            "kind_stats": kind_stats,
            "persona": config.PERSONA["name"],
            "source": source,
            "priority": priority,
            "status": "pending",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "decided_at": None, "decided_by": None, "reject_reason": None,
        }
        os.makedirs(os.path.join(self.root, "pending"), exist_ok=True)
        with open(self._path("pending", cid), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        print(f"[v2-candidate] 创建 {cid} → pending/（{title}，样本 {len(samples)} 条）")
        return cid

    def approve(self, cid: str, decided_by: str = "user") -> bool:
        rec = self.load(cid)
        if not rec or rec["status"] != "pending":
            return False
        rec["status"] = "approved"
        rec["decided_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["decided_by"] = decided_by
        os.remove(self._find(cid))
        with open(self._path("approved", cid), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        print(f"[v2-candidate] {cid} → approved（by {decided_by}）")
        return True

    def reject(self, cid: str, reason: str = "", decided_by: str = "user") -> bool:
        rec = self.load(cid)
        if not rec or rec["status"] != "pending":
            return False
        rec["status"] = "rejected"
        rec["decided_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["decided_by"] = decided_by
        rec["reject_reason"] = reason
        os.remove(self._find(cid))
        with open(self._path("rejected", cid), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        print(f"[v2-candidate] {cid} → rejected（{reason or 'no reason'}）")
        return True


# ---------- 文本提取：按 L0 记录结构解析（不同 source 文本路径不同） ----------
def _extract_texts(rec: dict) -> list[str]:
    """从一条 L0 记录提取 0..N 条文本。

    结构（实测 2026-08-27）：
      exchange:*  → payload.text（markdown 摘要）
      wechat      → payload.messages[].text（逐条消息）
      chat        → payload.messages[].text（逐轮对话，user/assistant 都算，交人审判断）
      兜底        → payload 为 str 直接用；否则 json 序列化截断
    """
    p = rec.get("payload")
    if isinstance(p, str):
        return [p] if p.strip() else []
    if isinstance(p, dict):
        if isinstance(p.get("text"), str) and p["text"].strip():
            return [p["text"]]
        msgs = p.get("messages")
        if isinstance(msgs, list):
            out = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                t = m.get("text") or m.get("content") or ""
                if isinstance(t, str) and t.strip():
                    out.append(t)
            return out
        try:
            return [json.dumps(p, ensure_ascii=False)[:800]]
        except TypeError:
            return []
    return []


# ---------- L0 源读取：按天排序 ----------
def _iter_records(l0_dir: str):
    """遍历 L0 目录所有 jsonl 的记录（保持文件内顺序）。"""
    if os.path.isdir(l0_dir):
        for fn in sorted(os.listdir(l0_dir)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(l0_dir, fn), encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            yield json.loads(ln)
                        except json.JSONDecodeError:
                            continue


def list_days(l0_dir: str = None) -> list[dict]:
    """按本地日期统计源记录数，升序返回（[{"date": "2026-08-18", "count": 382}, ...]）。"""
    l0_dir = l0_dir or config.L0_DIR
    c: Counter = Counter()
    for rec in _iter_records(l0_dir):
        d = _local_date(rec)
        if d:
            c[d] += 1
    return [{"date": d, "count": c[d]} for d in sorted(c)]


def extract_from_l0(l0_dir: str = None, date: str | None = None,
                    max_samples: int = 20) -> dict:
    """从源提炼训练候选。

    date=None → 提炼**最新一天**（默认，符合"每天一次"语义）；
    date=YYYY-MM-DD → 只提炼那一天。
    返回 {"date", "samples", "stats"}；stats 含按天分布。
    """
    l0_dir = l0_dir or config.L0_DIR
    days = list_days(l0_dir)
    if not days:
        return {"date": None, "samples": [], "stats": {"total": 0}}
    target = date or days[-1]["date"]          # 默认最新一天

    stats = {"total": 0, "style": 0, "mixed": 0, "fact": 0, "neutral": 0, "filtered": 0}
    style_samples: list[str] = []
    day_counts: Counter = Counter()
    for rec in _iter_records(l0_dir):
        d = _local_date(rec)
        if d != target:
            continue
        day_counts[d] += 1
        for text in _extract_texts(rec):
            if len(text) < 8:
                continue
            stats["total"] += 1
            c = classify(text)
            if c["kind"] == "fact":
                stats["fact"] += 1
                stats["filtered"] += 1
                continue
            stats[c["kind"]] += 1
            if c["kind"] in ("style", "mixed") and len(style_samples) < max_samples:
                style_samples.append(text[:500])
    return {"date": target, "samples": style_samples,
            "stats": stats, "days": days}


# ---------- CLI ----------
def main() -> None:
    args = sys.argv[1:]
    l0 = None
    if "--l0" in args:
        i = args.index("--l0")
        l0 = args[i + 1]
        args = args[:i] + args[i + 2:]

    if "--list" in args:
        for status in config.CANDIDATE_STATUSES:
            for c in CandidateQueue().list(status):
                print(f"[{status}] {c['id']} date={c.get('date','-')} 样本={len(c.get('samples', []))} "
                      f"by={c.get('decided_by') or '-'}")
        return

    if "--days" in args or not any(a.startswith("--date") for a in args):
        src = l0 or config.L0_DIR
        print(f"源（只读）: {src}")
        print("按天记录数（升序）:")
        for d in list_days(l0):
            print(f"  {d['date']}  {d['count']} 条")
        if "--days" in args:
            return
        # 无 --date 且非 --days：默认提炼最新一天并建候选

    date = None
    if "--date" in args:
        i = args.index("--date")
        date = args[i + 1]

    r = extract_from_l0(l0, date=date)
    if not r["samples"]:
        print(f"[v2-candidate] {r.get('date') or '最新一天'} 无风格样本，跳过创建候选。stats={r['stats']}")
        return
    cid = CandidateQueue().create(
        date=r["date"],
        title=f"LoRA 风格样本 {r['date']}（{config.PERSONA['name']}）",
        description=f"来自 {r['stats']['total']} 条记忆（日期 {r['date']}）；"
                    f"事实类 {r['stats']['filtered']} 条已过滤不进训练集。",
        samples=r["samples"],
        kind_stats=r["stats"],
    )
    print(f"→ 候选 {cid} 已入闸门，等待人审（POST /api/v2/candidates/{cid}/approve）")


if __name__ == "__main__":
    main()
