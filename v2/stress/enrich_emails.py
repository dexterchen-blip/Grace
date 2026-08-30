#!/usr/bin/env python3
"""邮件类事务场景生成器（2026-08-31）——对齐真实本地 AI 系统的数据构成。

真实系统数据构成: 邮件(事务)+微信(生活)+Canvas(学业)。模拟书库缺事务侧。
本生成器按【学期阶段演化】注入邮件类事件(主人转述邮件→雷姆听到):

  day 1-30   入学事务: 免疫/TB/缴费/奖学金申请/选课/宿舍
  day 31-60  学期中:   作业截止/quiz/小组项目/考试时间表/成绩
  day 61-90  学期末:   期末考试/成绩公布/下学期选课/暑假安排

模式来自真实 UCSB 抓取(邮箱摘要 md),日期/数字脱敏。
用法: ./run.sh .venv/bin/python3 v2/stress/enrich_emails.py
"""
from __future__ import annotations
import json
import glob
import os


def _sentiment(t: str) -> float:
    """粗情绪: 事务邮件多为中性偏负(截止/压力), 根据关键词粗判。"""
    neg = ("截止", "逾期", "罚款", "紧张", "压力", "挂", "问题", "没交", "不及格", "担心")
    pos = ("松口气", "顺利", "过了", "开心", "期待", "兴奋", "好消息")
    if any(k in t for k in pos):
        return 0.6
    if any(k in t for k in neg):
        return -0.5
    return 0.0

IN = "experiments/run/stress/inputs"

# 邮件类场景: (day, 主人转述) —— 3 阶段演化
EARLY = [  # day 1-30 入学事务
    (2, "主人说：收到学校的邮件，入学免疫记录不完整，9/15 前要提交 TB 筛查证明"),
    (6, "主人说：邮件说学费账单已出，9/20 前缴费，逾期有罚款"),
    (9, "主人说：学校邮件通知奖学金申请开放了，9/30 截止，要两封推荐信"),
    (13, "主人说：邮件提醒选课时间到了，热门课要抢"),
    (17, "主人说：宿舍邮件说门禁卡要激活，不激活周末进不了楼"),
    (21, "主人说：国际学生办公室发邮件，说 I-20 信息要更新"),
    (25, "主人说：邮箱又收到 ACTION REQUIRED，说成绩单要认证"),
    (29, "主人说：邮件说停车证申请开放，不申请会被贴罚单"),
]
MID = [  # day 31-60 学期中: 作业/考试
    (33, "主人说：Canvas 显示这周五有作业截止，他还没开始写"),
    (37, "主人说：教授发了邮件，下周二有 quiz，范围是前两周的内容"),
    (41, "主人说：小组项目要交了，但组员一直不回消息"),
    (45, "主人说：邮件说期中考试时间表出来了，连着三天有考试"),
    (49, "主人说：作业截止延长了，但代价是期末多 5% 权重"),
    (53, "主人说：教授回邮件说论文初稿有问题，要重写"),
    (57, "主人说：系统提醒选课 waitlist 结果出了，有一门没选上"),
    (60, "主人说：邮件说实验报告今晚 11:59 截止，他还在改数据"),
]
LATE = [  # day 61-90 学期末
    (63, "主人说：邮件通知期末考试周安排，有一门在周六早上 8 点"),
    (67, "主人说：教授说期末项目占总分 30%，下下周交"),
    (71, "主人说：邮件说成绩公布时间定了，说好紧张"),
    (75, "主人说：下学期选课系统开放，他纠结选哪几门"),
    (79, "主人说：邮件说暑假宿舍申请开放，不申请就得搬出去"),
    (83, "主人说：成绩单邮件来了，他说不太敢看"),
    (87, "主人说：下学期教材清单出了，又贵又厚"),
    (90, "主人说：邮件说成绩正式公布，他终于松了口气"),
]
ALL = EARLY + MID + LATE


def main():
    fs = sorted(glob.glob(os.path.join(IN, "day-*.json")))
    if not fs:
        print("无书库文件")
        return
    inserted = 0
    for day, text in ALL:
        fp = os.path.join(IN, f"day-{day:03d}.json")
        if not os.path.isfile(fp):
            continue
        d = json.load(open(fp, encoding="utf-8"))
        msgs = d.get("messages", [])
        if not msgs:
            continue
        # 邮件类消息插到每天最后(第 5 条位置, 若已有 5 条则替换第 5 条)
        if len(msgs) >= 5:
            msgs = msgs[:4]
        msgs.append({"text": text, "ts": 1787000000 + day * 86400 + 21 * 3600,
                     "sender": "owner", "source": "email",
                     "sentiment": _sentiment(text), "weight": 1.0})
        d["messages"] = msgs
        json.dump(d, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        inserted += 1
    print(f"邮件类场景已注入 {inserted} 天(3 阶段:入学事务{len(EARLY)}/学期中{len(MID)}/学期末{len(LATE)})")


if __name__ == "__main__":
    main()
