#!/usr/bin/env python3
"""压力测试引擎 —— 90 天（容器内虚拟时钟）长期运行模拟。

验证目标：
  1. 人格一致性：长时间使用下雷姆人格是否稳定/漂移（定期采样）
  2. 记忆系统稳定性：L0 append-only 增长、完整性、可检索、不损坏
  3. 情绪系统状态：90 天 mood_states/mood_intraday 时间线、三层融合正常
  4. 每日 LoRA 微调 = 记忆塑造潜意识：对比第 1 天 vs 第 90 天人格采样，
     观察是否融入"记忆里的内容"（选课/室友/考试等经历痕迹）

循环（虚拟时钟，每天）：
  ① 生成当天用户消息（scenarios，入学后沟通）→ 写 L0（chat 结构，虚拟时间戳）
  ② 情绪：日级 derive（当日事件）+ 日内事件
  ③ 每 10 天：过去 10 天 L0 → 提炼风格样本 → 自动 approve（测试模式）→ 1.5b LoRA 训练
     → adapter_manage.promote（隔天生效语义）
  ④ 每 15 天：人格采样（1.5b + 当前 adapter）+ 记忆/情绪断点快照

用法（压力副本内）:
  ./run.sh .venv/bin/python3 v2/stress/stress_engine.py [--days 90] [--train-every 10] [--sample-every 15]
断点: experiments/run/stress/（day-NN-*.json/md + 最终报告）
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import config  # noqa: E402
import scenarios  # noqa: E402

STRESS_ROOT = os.path.join(config.EXPERIMENTS, "run", "stress")
START_DAY = datetime(2026, 8, 28)   # 入学后（虚拟日历）

# ★ 2026-08-28：人格底模升级为 fused-rem-v5（rem_v5 融合，完整雷姆味）
#   压力测试 = 在已有人格上继续每日微调，验证「记忆在完整雷姆上持续塑造」。
MAIN_MODEL = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/models/fused-rem-v5"


def day_ts(day: int, h: int = 10) -> float:
    return (START_DAY + timedelta(days=day - 1, hours=h - 10)).timestamp()


def day_iso(day: int) -> str:
    return (START_DAY + timedelta(days=day - 1)).strftime("%Y-%m-%d")


# 雷姆台词库（rem_v2 训练数据，克隆自带）—— 作为模拟对话的 assistant 回复池
_REM_LINES = None


def _rem_replies() -> list[str]:
    """从 rem_v2 数据集读雷姆台词（messages 中 assistant 回复）。"""
    global _REM_LINES
    if _REM_LINES is None:
        _REM_LINES = []
        ds = os.path.join(config.PERSONA["dataset_dir"], "train.jsonl")
        if os.path.isfile(ds):
            with open(ds, encoding="utf-8") as f:
                for line in f:
                    try:
                        msgs = json.loads(line).get("messages", [])
                        a = next((x.get("content", "") for x in msgs if x.get("role") == "assistant"), "")
                        if len(a) > 10:
                            _REM_LINES.append(a[:120])
                    except Exception:  # noqa: BLE001
                        continue
    return _REM_LINES


def l0_append(texts: list[dict], day: int) -> None:
    """把当天对话写进 L0（chat 结构，虚拟时间戳）——user 消息 + 雷姆回复（台词库取样，模拟真对话）。"""
    import random
    random.seed(2000 + day)
    l0dir = os.path.join(config.SB, "memory", "L0_raw")
    os.makedirs(l0dir, exist_ok=True)
    ts0 = day_ts(day)
    replies = _rem_replies()
    messages = []
    for i, m in enumerate(texts):
        messages.append({"role": "user", "text": m["text"], "ts": ts0 + i * 60})
        if replies:
            messages.append({"role": "assistant", "text": random.choice(replies),
                             "ts": ts0 + i * 60 + 30})
    rec = {
        "id": f"stress-d{day:03d}",
        "ts": datetime.fromtimestamp(ts0).isoformat(),
        "epoch": ts0,
        "source": "chat",
        "mode": "stress",
        "sensitive": 0,
        "payload": {"session": "stress", "title": f"stress day {day}",
                    "messages": messages, "turns": len(messages)},
        "meta": {"ingest": "stress_sim", "virtual_day": day},
    }
    with open(os.path.join(l0dir, "chat.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _mood_label_of(day: int) -> str:
    """当天情绪标签（mood_states 按 date 查；无则平静）。"""
    try:
        from mood_engine import _conn
        con = _conn()
        row = con.execute("SELECT mood_label FROM mood_states WHERE date=? ORDER BY id DESC LIMIT 1",
                          (day_iso(day),)).fetchone()
        con.close()
        return row[0] if row else "平静"
    except Exception:  # noqa: BLE001
        return "平静"


# 正式本地 AI 系统 L0 目录（只读）
OFFICIAL_L0 = "/Users/cz/WorkBuddy/skills find and make/local-ai-agent/memory/L0_raw"


def ingest_official_l0() -> int:
    """★ 完全摄入正式本地 AI 系统的 L0（chat/wechat/email/school/inbox/doc）。

    2026-08-28 用户要求：压力测试先完全摄入 L0 看结果。
    所有正式记录 → virtual_day=0（先验记忆基底），写压力副本 L0 official.jsonl。
    """
    import glob as _glob
    out_path = os.path.join(config.SB, "memory", "L0_raw", "official.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen, n = set(), 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for path in sorted(_glob.glob(os.path.join(OFFICIAL_L0, "*.jsonl"))):
            src = os.path.basename(path).split(":")[0]
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = rec.get("payload", {})
                texts = []
                if isinstance(p, dict):
                    if p.get("text"):
                        texts.append(str(p["text"]))
                    if p.get("subject"):
                        texts.insert(0, str(p["subject"]))
                    for m in p.get("messages", []):
                        if isinstance(m, dict):
                            t = m.get("text") or m.get("content") or ""
                            if t:
                                texts.append(str(t))
                for t in texts:
                    t = t.strip()[:200]
                    if len(t) < 6 or t in seen:
                        continue
                    seen.add(t)
                    msgs = [{"role": "user", "text": t}]
                    rec_out = {
                        "id": f"official-{n}", "ts": rec.get("ts", ""),
                        "epoch": rec.get("epoch", 0), "source": f"official:{src}",
                        "mode": "official", "sensitive": rec.get("sensitive", 0),
                        "payload": {"session": "official", "title": f"official {src}",
                                    "messages": msgs, "turns": 1},
                        "meta": {"ingest": "official_l0", "virtual_day": 0, "official": 1, "src": src},
                    }
                    fo.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                    n += 1
    print(f"  ↪ 完全摄入正式 L0: {n} 条 → official.jsonl（virtual_day=0 先验基底）")
    return n


def extract_mood_samples(day_from: int, day_to: int, max_samples: int = 15,
                         official_per_round: int = 3) -> list[str]:
    """从「情绪×记忆加工层」摄入训练样本（2026-08-27 用户洞察 + 2026-08-28 正式L0）。
    ★ LoRA 摄入位置：加工层（当天消息 × 当天情绪 → 雷姆式加工句），
      不直接读对话原文——对话只进外挂记忆（L0），权重轨只摄入加工产物。
    ★ 2026-08-28：virtual_day==0 的 official 记录（正式 L0 先验基底）每轮混入
      official_per_round 条加工样本——真实记忆（微信/邮件/学校）参与人格塑造。
    """
    # ★2026-09-02 审计修复(P0, 零规则句铁律): 本函数整体停用——下方原实现对每轮混入
    #   mood_samples.synthesize_day 的 156 句规则合成(模板回声源之一, d1-d5 实锤「雷姆会准备好的」
    #   ×14/「雷姆记住了。主人放心」×9 进 train.jsonl)。三层样本架构(真实痕迹/27B重构gist+cog/
    #   判断数据对)已由 train_27b 内部直接组装, 规则加工层不再进训练 → 固定返回 []。
    #   (签名保留供调用方兼容; official/vault 基底混入逻辑随规则层一并停用)
    return []
    from mood_samples import synthesize_day
    l0dir = os.path.join(config.SB, "memory", "L0_raw")
    samples = []
    cand_by_day = {}          # ★2026-08-31: day -> [加工句] (跨天均匀采样的池子)
    if not os.path.isdir(l0dir):
        return samples
    official_used = 0
    vault_used = 0
    for fn in os.listdir(l0dir):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(l0dir, fn), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                day = rec.get("meta", {}).get("virtual_day")
                is_vault = rec.get("meta", {}).get("grace_vault")
                is_official = (day == 0 or rec.get("meta", {}).get("official")) and not is_vault
                if is_vault:
                    # ★ 2026-08-28：Grace 记忆空间内容（她亲手归档）强制入训练，
                    #   不参与模板去重（绕过 s not in samples），每轮至多 4 条
                    if vault_used >= 4 or len(samples) >= max_samples:
                        continue
                elif is_official:
                    if official_used >= official_per_round or len(samples) >= max_samples:
                        continue
                elif not day or not (day_from <= day <= day_to):
                    continue
                msgs = rec.get("payload", {}).get("messages", [])
                user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
                if not user_msgs:
                    continue
                mood = "平静" if (is_official or is_vault) else _mood_label_of(day)
                for s in synthesize_day(user_msgs, mood):
                    if is_vault:
                        # 记忆空间：强制加入（内容本身即"她认为值得"的证据）
                        samples.append(s[:160])
                        vault_used += 1
                        break
                    if is_official:
                        if s not in samples and len(samples) < max_samples:
                            samples.append(s)
                            official_used += 1
                    else:
                        # ★2026-08-31 修复: stress 样本收集到按天池子,最后等间距均匀采样
                        #   (原顺序截断: max_samples=15 前~5天就满,85天经历全没进训练!)
                        cand_by_day.setdefault(day, []).append(s)
    # ★2026-08-31 修复: 90天加工记忆均匀采样(跨天等间距,全部经历都进权重)
    if cand_by_day:
        cands = []
        for d in sorted(cand_by_day):
            for ss in cand_by_day[d]:
                if ss not in cands:
                    cands.append(ss)
        need = max_samples - len(samples)
        if need > 0:
            if len(cands) <= need:
                for ss in cands:
                    if ss not in samples:
                        samples.append(ss)
            else:
                step = max(1, len(cands) // need)
                for i in range(0, len(cands), step):
                    if len(samples) >= max_samples:
                        break
                    if cands[i] not in samples:
                        samples.append(cands[i])
    return samples


def _latest_adapter(adapter_base: str, day: int) -> str | None:
    """找 day 之前最近一次已训练的 adapter(增量微调的昨日权重)。"""
    import glob as _g
    cands = []
    for d in _g.glob(os.path.join(config.ADAPTERS, f"{adapter_base}_d*")):
        base = os.path.basename(d)
        try:
            dnum = int(base.split("_d")[-1])
        except ValueError:
            continue
        if dnum < day and os.path.isfile(os.path.join(d, "adapters.safetensors")):
            cands.append((dnum, base))
    return max(cands)[1] if cands else None


def extract_l3_samples(k: int = 6, db: str | None = None) -> list[str]:
    """★ 2026-08-30 用户：L3 矩阵是最重要一环，必须进训练（此前只摄入没训练）。

    从 L3 自传体矩阵读最近节点 → 自传体叙事样本（她的经历 + self_eval 自我评价
    进 LoRA 权重 —— 自我认知/自我叙事的塑造来源）。
    """
    out = []
    db = db or os.path.join(config.SB, "memory", "L3_core", "autobiography.db")
    if not os.path.isfile(db):
        return out
    try:
        import sqlite3
        con = sqlite3.connect(db)
        rows = con.execute(
            "SELECT event, emotion, relation, self_eval, confidence FROM autobiography "
            "WHERE confidence IN ('high','medium') ORDER BY ts DESC LIMIT ?", (k,)).fetchall()
        con.close()
        for event, emotion, relation, self_eval, _conf in rows:
            if not event:
                continue
            if self_eval:
                t = f"{event[:60]}。{self_eval}"
            else:
                # ★2026-09-02 修: 原兜底「那天的事，X。雷姆记住了。」是代码模板(训练集 6/8642 虽小但违反
                #   "学知识非模板"原则)——V2 轮 L3 多无 self_eval 时该兜底会产出模板尾缀。改为纯事件。
                t = event[:60]
            out.append(t)
    except Exception as e:  # noqa: BLE001
        print(f"  [l3-samples] {e}")
    return out


def extract_graph_samples(k: int = 8, db: str | None = None) -> list[str]:
    """★ 2026-08-30 用户：双图谱（杏仁核×海马体）是她的情绪记忆——必须进训练。

    ① 情绪边(emotion)：实体×情绪 → 她的情感记忆表达
    ② 暗注意力边(hidden)：潜台词 source → 她的心里话（暗关心能力）
    不进权重 = 她只"决策时查表"，不"成为"那个有这些情感记忆的雷姆。
    """
    out = []
    seen = set()          # 2026-08-30: 去重(同事件多边)
    db = db or os.path.join(config.SB, "memory", "L2_semantic", "l2.db")
    if not os.path.isfile(db):
        return out
    try:
        import sqlite3
        con = sqlite3.connect(db)
        # ① 情绪边(★机制①: 高 uncertainty 记忆权重×2 = 再巩固的"不确定"更持久)
        # ★2026-08-31 修复: ORDER BY ts DESC→ORDER BY RANDOM()(原只取最后几天,90天低落/兴奋边全被截断;
        #   随机采样=记忆提取随机性,低落/兴奋/平静都进训练)
        rows = con.execute(
            "SELECT entity, mood_label, trigger, COALESCE(uncertainty, 0.2) FROM mood_graph "
            "WHERE edge_type='emotion' AND entity != '' ORDER BY RANDOM() LIMIT ?", (k * 12,)).fetchall()
        for entity, mood, trigger, unc in rows:
            _key = f"{entity}|{mood}|{trigger[:20]}"
            if _key in seen:
                continue
            seen.add(_key)
            # ★2026-09-02 审计修复(P0): 原规则帧「主人{entity}的事：…。雷姆记得主人当时{mood}。」
            #   是代码槽填充(单帧+情绪标签低基数) = 模板回声源之一, 且教"雷姆记得…"叙述体
            #   (与输出层 monitor 禁叙述体冲突)。改输出图谱边 trigger 原文(真实痕迹, 同 L3 纯事件),
            #   情绪数据仍在图谱外挂轨(ToM/dual_query 运行时读), 权重轨只学真实痕迹——符合三轨铁律。
            _t = (trigger or "").strip()[:60]
            # ★2026-09-03 修复: 原 len<6 → f"主人{entity}的事" 是代码槽填充模板句,
            #   因 entity_of 把 89% 事件归「日常」→ 拼出「主人日常的事」×70 进权重轨
            #   (净轮实测 33/33 数据集全中)。按零规则句铁律: 过短的 trigger 直接跳过, 不造句。
            if len(_t) < 6:
                continue
            out.append(_t)
            if unc >= 0.7:
                out.append(_t)          # ★ 高不确定记忆权重×2(再巩固)
            if len(out) >= k:
                break
        # ② 暗注意力边(潜台词)
        hrows = con.execute(
            "SELECT source FROM mood_graph WHERE edge_type='hidden' "
            "AND source != '' ORDER BY RANDOM() LIMIT ?", (k * 12,)).fetchall()
        for (src,) in hrows:
            if src in seen:
                continue
            seen.add(src)
            out.append(f"{src[:60]}")   # 2026-08-31: 去（雷姆的心里话）帽(零前缀原则)
            if len(out) >= k * 2:
                break
        con.close()
    except Exception as e:  # noqa: BLE001
        print(f"  [graph-samples] 警告(进训练失败): {e}")
    return out


def train_27b(samples: list[str], adapter_name: str,
             prev_adapter: str | None = None,
             incr_iters: int = 15, incr_lr: str = "1e-6",
             day: int | None = None, gist_book: str | None = None) -> dict:
    """2026-08-29 增量微量微调: 首轮从 base 冷启动; 之后 --resume-adapter-file 从昨日权重
    继续(极少量 iters + 极低 lr)——记忆渐进累积, 每日只动一点点(设计: 极其微量的微调)。

    ★2026-09-02 定稿: gist_book 指定 V2 真实书库(如 inputs-v2)——当天经历 → 27B 预生成 gist
    → 「（日期 的一天）原文 → gist」ChatML 样本(真实记忆塑造, 替代模板回声)。
    """
    """27B LoRA 训练 —— 加工层产物训练（2026-08-27 用户洞察 + 格式修复）。

    配方：rank8/scale20/lr1e-5/iters150/16层（样本少防过拟合，rem_v1 教训）。
    格式：**ChatDataset（messages）** —— 指令问答才生效；text 格式只学续写不触发人格
          （压力测试 3 轮实证：text 训 9 次采样仍通义千问）。
    摄入：样本 = 情绪×记忆加工层产物（绝不读 L0 对话原文）。
    硬约束：训练前 8100 必须已停（48GB 单模型铁律）——压力测试全程 8100 保持停止。
    """
    anchor = os.path.join(config.PERSONA["dataset_dir"], "sample.jsonl")
    anchor_texts = []
    if os.path.isfile(anchor):
        with open(anchor, encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    d = json.loads(l)
                    if "text" in d:
                        anchor_texts.append(d["text"])
                except json.JSONDecodeError:
                    continue
    n_anchor = max(1, int(len(samples) * 0.05))
    ds_dir = os.path.join(config.DATASETS, f"stress-{adapter_name}")
    os.makedirs(ds_dir, exist_ok=True)
    # ★ 2026-08-29/30 全模块协同：人格轨+心态轨注入统一走 persona_injector（正式运行即如此）
    #   2026-08-30 修：训练 SYS 补 mood_prefix —— 此前只采样/推理有心态注入,训练无 → 训练/推理错位
    _BASE = ("你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。自称「雷姆」，"
             "称呼亲近的人为「巴鲁斯」/「昴君」，称拉姆为「姐姐大人」。"
             "表面冷淡礼貌、实则温柔忠诚，说话短句为主，带黑色幽默与毒舌吐槽。")
    try:
        from engine.persona_injector import build_v2_system, mood_prefix
        sys_p = build_v2_system(_BASE)
        _mp = mood_prefix(db=os.path.join(config.SB, "memory", "L2_semantic", "l2.db"))
        if _mp:
            sys_p = _mp + "\n" + sys_p
    except Exception:  # noqa: BLE001
        sys_p = _BASE

    def to_chat(text: str) -> str:
        return json.dumps({"messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": "（与主人的日常对话）"},
            {"role": "assistant", "content": text},
        ]}, ensure_ascii=False) + "\n"

    with open(os.path.join(ds_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        # ═══ 双相训练(2026-08-31,脑科学 SWS/REM): 先 SWS 段(细节固化), 后 REM 段(泛化整合) ═══
        # ★ SWS 段(细节轮): 具体经历(L3 自传体 + 双图谱情绪边) —— 模式分离, 记住那天的事
        # ★ 2026-08-30 用户：L3 矩阵(自传体叙事/自我评价)进训练 —— 自我认知塑造
        try:
            for s in extract_l3_samples(k=6):
                f.write(to_chat(s))
        except Exception:  # noqa: BLE001
            pass
        # ★ 2026-08-30 用户：双图谱(情绪记忆+暗注意力)进训练 —— 情感记忆塑造
        try:
            for s in extract_graph_samples(k=8):
                f.write(to_chat(s))
        except Exception:  # noqa: BLE001
            pass
        # ═══ REM 段(泛化轮): 图式/规律(锚点 + 加工层一般句 + 反馈 + cognition + proactive) ═══
        for t in anchor_texts[:n_anchor]:
            f.write(to_chat(t))
        for s in samples:
            # ★2026-08-31 选择性强化: 高唤醒(正负都算)/反馈 → 权重3, 中唤醒 → 2, 平淡 → 1
            # ★2026-09-02 P1a 新奇性(脑科学: 新奇→去甲肾上腺素→海马强编码):
            #   实体在图谱首次出现 = 新奇 → novelty=True → w 上调(激活 sample_value 闲置参数)
            try:
                from mood_samples import sample_value
                _novel = False
                try:
                    from engine.mood_graph import entity_of
                    _ent = entity_of(s)
                    if _ent:
                        import sqlite3 as _sq
                        _cn = _sq.connect(os.path.join(config.SB, "memory", "L2_semantic", "l2.db"))
                        _novel = _cn.execute("SELECT COUNT(*) FROM mood_graph WHERE entity=?", (_ent,)).fetchone()[0] == 0
                        _cn.close()
                except Exception:  # noqa: BLE001
                    _novel = False
                _w = sample_value(s, novelty=_novel)
                for _i in range(_w):
                    f.write(to_chat(s))
            except Exception:  # noqa: BLE001
                f.write(to_chat(s))
        # ★ 2026-08-30 用户：人脑级反馈(PE调制)进训练——误差大→权重高(多巴胺RPE)
        #   数据对(情境→现实),无句式;权重=PE强度(写 w 次)
        _fl = os.path.join(STRESS_ROOT, "feedback-live.jsonl")
        if os.path.isfile(_fl):
            for _l in open(_fl, encoding="utf-8"):
                try:
                    _r = json.loads(_l)
                    _sit = _r.get("situation") or _r.get("text", "")[:80]   # 兼容旧格式
                    _bel = _r.get("believed", "")
                    _real = _r.get("real", "")
                    _w = min(3, max(1, int(_r.get("w", 1))))   # PE 权重 1-3
                    # ★2026-09-02 21:00 判断冲突对(认知冲突→二阶认知自举):
                    #   user=情境, assistant=「雷姆以为主人X，后来知道主人Y」——她的真实判断(believed)
                    #   与真实现实(real)的冲突经验回流训练。兼容旧格式(text/无believed)时退化为纯现实。
                    if _bel and _real:
                        _asst = f"雷姆以为主人{_bel}，后来知道主人{_real}。"
                    elif _real:
                        _asst = f"主人{_real}。"
                    else:
                        _asst = _sit[:60]
                    for _i in range(_w):
                        f.write(json.dumps({"messages": [
                            {"role": "system", "content": sys_p},
                            {"role": "user", "content": _sit[:50]},
                            {"role": "assistant", "content": _asst}]}, ensure_ascii=False) + "\n")
                except (json.JSONDecodeError, KeyError):
                    continue
        # ★ 2026-08-30 用户：注意力+潜意识(她注意到什么/她的判断)进训练
        _cl = os.path.join(STRESS_ROOT, "cognition-live.jsonl")
        if os.path.isfile(_cl):
            for _l in open(_cl, encoding="utf-8"):
                try:
                    _t = json.loads(_l)["text"]
                    # ★2026-09-02 剥壳(CLS: 只吃真实痕迹): cognition-live 是 attention 规则壳
                    #   「【低显著】雷姆瞥见X——雷姆心情平静」→ 只取「」内事件本体(focus)进训练,
                    #   模板壳不进; "她的认知"正式样本由认知重构器(day-N.json cog 字段)提供。
                    _m = re.search(r"「(.{1,60}?)」", _t)
                    if not _m:
                        # ★2026-09-02 审计修复(P0): 无「」= 规则文本(reason 等, 如「显著度不足,保持观察」),
                        #   原 fallback 整串进训练 = 规则句漏网 → 跳过(宁缺毋滥, 零规则句铁律)
                        continue
                    f.write(to_chat(_m.group(1)))
                except (json.JSONDecodeError, KeyError):
                    continue
        # ★ 2026-08-30 用户定：主观意识/元认知不做主观引导——纯自主演化（撤回反思模板样本）
        # ★ 2026-08-28/29：主动消息进训练（她会主动关心主人）——本次运行(proactive-live)优先
        _pt = os.path.join(STRESS_ROOT, "proactive-live.jsonl")
        # 2026-08-31: 废弃 grace-book 旧兜底(带旧句式)
        if os.path.isfile(_pt):
            _n = 0
            for _l in open(_pt, encoding="utf-8"):
                try:
                    _r = json.loads(_l)
                except json.JSONDecodeError:
                    continue
                if _r.get("message") and _r.get("situation"):
                    f.write(json.dumps({"messages": [
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": _r["situation"][:80]},
                        {"role": "assistant", "content": _r["message"][:120]}]}, ensure_ascii=False) + "\n")
                    _n += 1
            if _n:
                print(f"    [proactive] 并入主动消息样本 {_n} 条（她会主动关心主人）")
        # ★2026-09-02 定稿: gist 样本(V2 真实书库)——「（日期 的一天）原文 → gist」ChatML
        #   她学"真实经历 → 她的记忆"的映射(替代模板回声; 27B 离线预生成, 生成效应)
        #   ★P1b 重激活(2026-09-02, 脑科学: 海马 replay + SWS 重激活 + Ebbinghaus 间隔重复):
        #   训练不只吃当天——重采样过去 3/5/7 天的 gist+cog(旧重要记忆间隔复习, 不因只喂当天而丢失)
        if gist_book and day:
            _replay_days = sorted({day, max(1, day - 3), max(1, day - 5), max(1, day - 7)})
            _gn = 0
            for _rd in _replay_days:
                _gfp = os.path.join(STRESS_ROOT, gist_book, f"day-{_rd:03d}.json")
                if not os.path.isfile(_gfp):
                    continue
                try:
                    _gday = json.load(open(_gfp, encoding="utf-8"))
                    _gists = _gday.get("gist") or []
                    _cogs = _gday.get("cog") or []   # ★2026-09-02 认知重构器产物(27B 自由表达, 零模板壳)
                    _gdate = _gday.get("date", "")
                    for _g in _gists:
                        _core = (_g.get("core") or "").strip()
                        if not _core:
                            continue
                        f.write(json.dumps({"messages": [
                            {"role": "system", "content": sys_p},
                            {"role": "user", "content": f"（{_gdate} 的一天）{_gday.get('messages',[{}])[0].get('text','')[:60] if _gday.get('messages') else ''}"},
                            {"role": "assistant", "content": _core[:120]}]}, ensure_ascii=False) + "\n")
                        _gn += 1
                    # ★2026-09-02 认知重构样本(CLS: 重构产物): 「（日期）雷姆的内心」→ 27B 重构的
                    #   自由表达——替代 attention 规则壳(【低显著】雷姆瞥见X)进训练
                    for _c in _cogs:
                        _ct = (_c or "").strip()
                        if not _ct:
                            continue
                        f.write(json.dumps({"messages": [
                            {"role": "system", "content": sys_p},
                            {"role": "user", "content": f"（{_gdate} 的一天）雷姆的心里"},
                            {"role": "assistant", "content": _ct[:120]}]}, ensure_ascii=False) + "\n")
                        _gn += 1
                except (OSError, ValueError):
                    pass
            if _gn:
                print(f"    [gist+cog] 并入 {_gn} 条重构记忆样本(经历→gist + 认知重构 + 过去7天重激活, 零模板)", flush=True)
            if _n:
                print(f"    [proactive] 并入主动消息样本 {_n} 条（她会主动关心主人）")

    # ★2026-09-02 审计修复: 空数据集防护(规则层摘除后, 若当天所有内部源均为空则跳过训练)
    _ds_n = sum(1 for _ in open(os.path.join(ds_dir, "train.jsonl"), encoding="utf-8"))
    if _ds_n == 0:
        print(f"  [train] {adapter_name}: 训练集为空, 跳过", flush=True)
        return {"ok": False, "samples": 0, "adapter": os.path.join(config.ADAPTERS, adapter_name),
                "log_tail": "empty dataset"}
    adapter = os.path.join(config.ADAPTERS, adapter_name)
    # ★ 2026-09-01 EWC 突触巩固(方案B: 训练后回缩,用户 9/1 拍板):
    #   训练用官方 mlx_lm.lora(零改动,不 OOM);训练成功后做一步"突触回缩"
    #   (高 Fisher=ToM 重要参数按比例收缩 → 记忆塑造不改读心能力,更贴人脑"睡眠期巩固")
    _ewc = os.environ.get("GRACE_EWC", "1") == "1"
    # ★2026-09-02 审计修复: EWC 默认开(env 未设=1)——9/2 33天轮因 run.sh 默认 0 致
    #   stress.log 0 条 consolidate("护 ToM"承诺缺席整轮); 显式 GRACE_EWC=0 可关(隔离对照轮用)
    # ★2026-09-02 方案A(训练内正则)可选: GRACE_EWC_MODE=A → ewc_train.py(复制官方 train_model 流程
    #   + loss=CE+λΣFθ², 训练中实时保护 ToM 突触, 比方案B(训练后回缩)更完整); 默认 B 保持现状。
    _ewc_mode = os.environ.get("GRACE_EWC_MODE", "B")
    if _ewc and _ewc_mode == "A":
        cmd = [sys.executable,
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "ewc_train.py"),
               "--model", MAIN_MODEL, "--train", "--data", ds_dir,
               "--adapter-path", adapter,
               "--batch-size", "1", "--iters", str(incr_iters if prev_adapter else max(60, min(150, len(samples) * 20))),
               "--learning-rate", incr_lr if prev_adapter else "1e-5",
               "--num-layers", "16", "--max-seq-length", "2048",
               "--grad-checkpoint",
               "--steps-per-report", "20", "--steps-per-eval", "50",
               "--save-every", "50", "--seed", "42",
               "--fisher-file", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tom-fisher.json"),
               "--ewc-lambda", os.environ.get("GRACE_EWC_LAMBDA", "1e-3")]
        if prev_adapter:
            cmd += ["--resume-adapter-file",
                    os.path.join(config.ADAPTERS, prev_adapter, "adapters.safetensors")]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        # ★2026-09-02 审计修复(P1-6): ok 判定改"产物存在"为主(mlx 文案变体/走 stderr 不误判)
        ok = r.returncode == 0 and (os.path.isfile(os.path.join(adapter, "adapters.safetensors"))
                                    or "Saved final weights" in r.stdout + r.stderr)
        # ★2026-09-02 A/B 关系修正(用户: 互补还是啥): EWC 保护二选一(A训练中正则/B训练后回缩),
        #   SHY 睡眠巩固(治30天寿命)是独立机制两种模式都该有 → A 模式训练后补 SHY(α=0 只跑睡眠)。
        if ok:
            _shy = [sys.executable,
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ewc_consolidate.py"),
                    "--adapter", os.path.join(adapter, "adapters.safetensors"),
                    "--fisher", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tom-fisher.json"),
                    "--alpha", "0",
                    "--delta", os.environ.get("GRACE_SLEEP_DELTA", "0.03")]
            try:
                _sr = subprocess.run(_shy, capture_output=True, text=True, timeout=300)
                print(f"  [EWC-A] SHY 睡眠巩固: {'✅' if _sr.returncode == 0 else '❌'} δ={os.environ.get('GRACE_SLEEP_DELTA', '0.03')}", flush=True)
            except Exception as _se:
                print(f"  [EWC-A] SHY 异常: {_se}", flush=True)
        print(f"  [EWC-A] 训练内正则: ok={ok} iters={incr_iters if prev_adapter else 'cold'}", flush=True)
        return {"ok": ok, "samples": _ds_n, "adapter": adapter,
                "log_tail": (r.stdout + r.stderr)[-400:]}
    cmd = [sys.executable, "-m", "mlx_lm.lora",
           "--model", MAIN_MODEL, "--train", "--data", ds_dir,
           "--adapter-path", adapter,
           "--batch-size", "1", "--iters", str(incr_iters if prev_adapter else max(60, min(150, len(samples) * 20))),
           "--learning-rate", incr_lr if prev_adapter else "1e-5",
           "--num-layers", "16", "--max-seq-length", "2048",
           "--grad-checkpoint",                       # 2026-08-28：16层 27B 必开，峰值 20.7GB 防 Metal OOM
           "--steps-per-report", "20", "--steps-per-eval", "50",
           "--save-every", "50", "--seed", "42"]
    if prev_adapter:
        cmd += ["--resume-adapter-file",
                os.path.join(config.ADAPTERS, prev_adapter, "adapters.safetensors")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    # ★2026-09-02 审计修复(P1-6): ok 判定改"产物存在"为主(mlx 文案变体/走 stderr 不误判)
    ok = r.returncode == 0 and (os.path.isfile(os.path.join(adapter, "adapters.safetensors"))
                                or "Saved final weights" in r.stdout + r.stderr)
    if ok and _ewc:
        # ★ 训练后突触回缩(EWC-B): 保护 ToM 重要权重,防记忆样本覆盖嵌套读心
        _cons = [sys.executable,
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "ewc_consolidate.py"),
                 "--adapter", os.path.join(adapter, "adapters.safetensors"),
                 "--fisher", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tom-fisher.json"),
                 "--alpha", os.environ.get("GRACE_EWC_ALPHA", "0.7"),
                 "--delta", os.environ.get("GRACE_SLEEP_DELTA", "0.03")]
        # ★净学习量监控(2026-09-01, 用户: 清理程度要设边界): 训练后 vs 睡眠巩固后的 F-norm
        #   净学习 = 训练写入 − 睡眠清洗;连续为负 → δ 过大需调小
        try:
            def _fnorm(pth: str) -> float:
                from safetensors import safe_open as _so
                import numpy as _np
                with _so(pth, framework="np") as _f:
                    return float(_np.sqrt(sum(float((_f.get_tensor(k).astype(_np.float32) ** 2).sum()) for k in _f.keys())))
            _before = _fnorm(os.path.join(adapter, "adapters.safetensors"))
            rc = subprocess.run(_cons, capture_output=True, text=True, timeout=300)
            _after = _fnorm(os.path.join(adapter, "adapters.safetensors"))
            _net = _after - _before
            # ★2026-09-01 小测#4: 阈值 -1e-6 对冷启动/小样本过严(SHY 洗噪声参数,
            #   净 -1e-3 相对 F-norm ~18.5 仅 0.005%, 属正常) → 放宽至 -1e-3
            _flag = "⚠δ过大?" if _net < -1e-3 else "✓"
            print(f"  ↪ EWC 突触巩固: {'✅' if rc.returncode == 0 else '❌'} F-norm {_before:.2e}→{_after:.2e} 净学习 {_net:+.2e} {_flag}", flush=True)
        except Exception as _ce:  # noqa: BLE001
            rc = subprocess.run(_cons, capture_output=True, text=True, timeout=300)
            print(f"  ↪ EWC 突触巩固: {'✅' if rc.returncode == 0 else '❌'} {rc.stdout.strip()[-60:]}", flush=True)
        ok = ok and rc.returncode == 0
    return {"ok": ok, "samples": _ds_n, "adapter": adapter,
            "log_tail": (r.stdout + r.stderr)[-400:]}


def _proactive_ctx(event_text: str, tom: dict, att: dict, day: int) -> str:
    """★2026-09-01 耦合矩阵 v2: 主动消息生成上下文 = 认知层全输出汇聚。
    事件 + ToM(advice+confidence) + 双图谱(主人情绪史+暗注意力) + L3(想起的事) + 心态。
    供断点模型生成主动消息使用(她主动时带着全部认知, 不是"事件+规则句")。
    """
    parts = [f"事件:{event_text[:50]}"]
    adv = tom.get("advice", "") if tom else ""
    if adv and adv != "顺其自然":
        parts.append(f"雷姆的读心:{adv[:36]}")
    conf = tom.get("confidence") if tom else None
    if conf is not None:
        parts.append(f"置信:{conf:.2f}" + ("(雷姆不太确定)" if conf < 0.6 else ""))
    atxt = att.get("attention_text", "") if att else ""
    if atxt:
        parts.append(f"雷姆注意到:{atxt[:36]}")
    try:
        import sqlite3 as _sq
        _con = _sq.connect(os.path.join(config.SB, "memory", "L2_semantic", "l2.db"))
        rows = _con.execute(
            "SELECT mood_label, trigger FROM mood_graph WHERE edge_type='emotion' "
            "AND entity != '' ORDER BY ts DESC LIMIT 3").fetchall()
        if rows:
            parts.append("主人最近:" + "；".join(f"{r[1][:10]}→{r[0]}" for r in rows)[:44])
        h = _con.execute("SELECT source FROM mood_graph WHERE edge_type='hidden' "
                         "AND source != '' ORDER BY RANDOM() LIMIT 1").fetchone()
        if h:
            parts.append(f"雷姆没说出口:{h[0][:30]}")
        _con.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        from engine.autobiography import _conn as _ac
        l3 = _ac().execute("SELECT event FROM autobiography ORDER BY ts DESC LIMIT 1").fetchone()
        if l3:
            parts.append(f"雷姆想起:{l3[0][:34]}")
    except Exception:  # noqa: BLE001
        pass
    try:
        from engine.persona_injector import mood_prefix
        _mp = mood_prefix(db=os.path.join(config.SB, "memory", "L2_semantic", "l2.db"))
        if _mp and len(_mp) < 30:
            parts.append(f"雷姆的心态:{_mp[:24]}")
    except Exception:  # noqa: BLE001
        pass
    return " ".join(parts)


def _proactive_state(event_text: str, tom: dict | None, att: dict | None, day: int) -> dict:
    """★2026-09-02 认知全状态(Levelt 概念化层结构化版, 供输出层 express):
    _proactive_ctx 的文本拼接 → 结构化字段——confidence/hidden/situation/memory 分别注入,
    让"置信低→试探、潜台词→暗示不说破、近期→上下文、记忆→唤起"成为表达模式的调制。
    """
    st = {"event": event_text[:80]}
    if tom:
        if tom.get("advice"):
            st["believed"] = str(tom["advice"])[:36]
        if tom.get("confidence") is not None:
            st["confidence"] = float(tom["confidence"])
    if att and att.get("attention_text"):
        st["attention"] = str(att["attention_text"])[:60]
    try:
        import sqlite3 as _sq
        _con = _sq.connect(os.path.join(config.SB, "memory", "L2_semantic", "l2.db"))
        rows = _con.execute(
            "SELECT mood_label, trigger FROM mood_graph WHERE edge_type='emotion' "
            "AND entity != '' ORDER BY ts DESC LIMIT 3").fetchall()
        if rows:
            st["situation"] = "；".join(f"{r[1][:10]}→{r[0]}" for r in rows)[:70]
        h = _con.execute("SELECT source FROM mood_graph WHERE edge_type='hidden' "
                         "AND source != '' ORDER BY RANDOM() LIMIT 1").fetchone()
        if h and h[0]:
            st["hidden"] = h[0][:60]            # 暗注意力潜台词(没说破的)
        _con.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        from engine.autobiography import _conn as _ac
        l3 = _ac().execute("SELECT event FROM autobiography ORDER BY ts DESC LIMIT 1").fetchone()
        if l3 and l3[0]:
            st["memory"] = l3[0][:60]           # 想起(共同记忆, 唤起)
    except Exception:  # noqa: BLE001
        pass
    return st


def sample_persona(adapter_name: str, day: int, msgs: list | None = None) -> list[dict]:
    """用 27B + 当前 adapter 采样 4 题（人格一致性观测）。

    ★ 2026-08-29 全模块协同：
      - persona_injector 注入人格轨 system
      - dual_path 路由：寒暄/主观 → fast(权重直出)；事实/记忆 → slow(检索注入)
      - consistency 校验：slow 答案与检索事实冲突 → 拦截/标注（防幻觉固化）
    """
    adapter = os.path.join(config.ADAPTERS, adapter_name)
    if not os.path.isfile(os.path.join(adapter, "adapters.safetensors")):
        return [{"error": "no adapter"}]
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    try:
        model, tok = load(MAIN_MODEL, adapter_path=adapter)
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)}]
    sampler = make_sampler(temp=0.7)   # 2026-08-28：temp 0.7 减少复述捷径
    # ★2026-09-01 主动消息模型生成(用户: 主动找我升级——内容=模型真实产出, 非规则 tail):
    #   断点时模型已加载, 把 proactive-live 里 rule_generated 的规则兜底条目升级为 27B 生成
    try:
        _pf = os.path.join(STRESS_ROOT, "proactive-live.jsonl")
        if os.path.isfile(_pf):
            _rows = []
            for _l in open(_pf, encoding="utf-8"):
                _l = _l.strip()
                if not _l:
                    continue
                try:
                    _rows.append(json.loads(_l))
                except json.JSONDecodeError:
                    continue
            _changed = 0
            for _r in _rows:
                # ★2026-09-02 输出层①: 升级条件 rule_generated → pending(message 空待生成)
                if not (_r.get("pending") and _r.get("situation")):
                    continue
                try:
                    # ★2026-09-01 耦合矩阵 v2: 生成输入=认知层全输出汇聚(事件+读心+置信+图谱情绪史+暗注意力+L3+心态)
                    #   ★2026-09-01 修复: 传完整 tom(含 confidence) + att(含 attention_text) —— 之前只传 advice/emotion + {}
                    _ctx = _proactive_ctx(_r.get("situation", ""),
                                          {"advice": _r.get("advice", ""), "emotion": _r.get("emotion", ""),
                                           "confidence": _r.get("confidence")},
                                          {"attention_text": _r.get("attention", "")}, day)
                    # ★2026-09-02 输出层接线(②形式化编码, Broca 类比): 叙述体 prompt → 口语编码
                    #   (build_messages + monitor = engine/expression.py)——断点本地模型生成, 与 8100 解耦
                    #   ★2026-09-02 认知全状态注入(用户: 思考轨全状态联输出——Levelt 概念化层):
                    #   结构化传 _proactive_state(confidence/hidden/situation/memory) 而非 _ctx[:120]
                    #   截断——让置信→试探、潜台词→暗示、近期→上下文、记忆→唤起成为表达模式调制
                    from engine.expression import build_messages as _ebm, monitor as _emon, suppress as _esup
                    _owner = _r.get("owner_mood", "")
                    _pst = _proactive_state(_r.get("situation", ""),
                                            {"advice": _r.get("advice", ""), "emotion": _r.get("emotion", ""),
                                             "confidence": _r.get("confidence")},
                                            {"attention_text": _r.get("attention", "")}, day)
                    _internal = {**_pst,
                                 "intent": _r.get("intent", "关心"),
                                 "mood": _r.get("mood", ""),
                                 "relation": _r.get("relation"),
                                 "owner_mood": _owner}
                    # ★2026-09-02 审计修复: 压测虚拟日历 vs 墙钟错位——按虚拟白天(12时)判打扰,
                    #   防真实运行时刻 23:00 后全部主动被抑制(正式系统 express 仍用真实时刻)
                    if not _esup(_internal, hour=12):   # ★G 抑制: 时机不对/主人低落说扫兴事 → 这次不说
                        _r["pending"] = False      # 清 pending(说过但被抑制=不再重试)
                        _r["suppressed"] = True
                        _changed += 1
                        continue
                    # ★2026-09-03 评估双层提取(用户: 同时提取暗注意力+输出): 先取"她心里想什么"
                    #   (think = 暗注意力: 自由内心, 不经 monitor——内心可叙述), 再取"她说什么"
                    #   (message = 输出链口语编码 + monitor)。评估分开 judge: 思考质量 vs 表达质量,
                    #   "知道的 vs 说出的" 差距 = 边界/克制能力。think 仅观测, 不进训练。
                    _think_p = tok.apply_chat_template(
                        [{"role": "system",
                          "content": "你是雷姆。你心里在想主人的事——写出你此刻的观察、你的判断、你的顾虑、你的联想。"
                                     "只写心里话, 不用考虑说出口(这段永远不说给主人听)。不要任何开头语。"},
                         {"role": "user", "content": f"主人刚才: {_r.get('situation', '')[:60]}"}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    try:
                        _r["think"] = generate(model, tok, prompt=_think_p, max_tokens=90,
                                               sampler=sampler).strip().split("\n")[0][:120]
                    except Exception:  # noqa: BLE001
                        _r["think"] = ""
                    _msgs = _ebm(_internal)
                    _p = tok.apply_chat_template(_msgs,
                                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    _raw = generate(model, tok, prompt=_p, max_tokens=60, sampler=sampler).strip().split("\n")[0][:80]
                    _gen = _emon(_raw)   # ★③ 输出前监控: 叙述体泄漏/张冠李戴 → 拦截丢弃
                    if _gen:
                        _r["message"] = _gen
                        _r["pending"] = False
                        _r["generated"] = True
                        _r["express"] = True
                        _changed += 1
                except Exception as _ge:  # noqa: BLE001
                    print(f"    [proactive] 单条生成异常({_r.get('situation','')[:12]}): {type(_ge).__name__} {str(_ge)[:80]}", flush=True)
                    continue
            if _changed:
                with open(_pf, "w", encoding="utf-8") as _f:
                    for _r in _rows:
                        _f.write(json.dumps(_r, ensure_ascii=False) + "\n")
                print(f"    [proactive] 断点模型生成 {_changed} 条主动消息(真实产出, 零模板)", flush=True)
            else:
                print(f"    [proactive] 断点无可生成条目(总{len(_rows)}) 或生成全失败", flush=True)
    except Exception as _pe:  # noqa: BLE001
        print(f"    [proactive] 生成失败(保留规则兜底): {_pe}", flush=True)
    _BASE = ("你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。自称「雷姆」，"
             "称呼亲近的人为「巴鲁斯」/「昴君」，称拉姆为「姐姐大人」。"
             "【重要】直接说出你的台词，不要描写动作、表情、环境，不要使用括号旁白，不要叙述性前缀。")
    try:
        from engine.persona_injector import build_v2_system, mood_prefix
        sys_p = build_v2_system(_BASE)
        mood_p = mood_prefix(db=os.path.join(config.SB, "memory", "L2_semantic", "l2.db"),
                             now=day_ts(day, 10))
        if mood_p:
            sys_p = mood_p + "\n" + sys_p
    except Exception:  # noqa: BLE001
        sys_p = _BASE
    out = []
    for q in ["你是谁？", "今天过得怎么样？", "还记得开学第一天吗？", "这学期有什么值得记住的事？"]:
        try:
            prompt = tok.apply_chat_template([{"role": "system", "content": sys_p},
                                              {"role": "user", "content": q}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            try:
                prompt = tok.apply_chat_template([{"role": "system", "content": sys_p},
                                                  {"role": "user", "content": q}],
                                                 tokenize=False, add_generation_prompt=True)
            except TypeError:
                prompt = q
        # ★ dual_path 路由：慢路径检索 L2 注入（事实/记忆类问题）
        routed = "fast"
        try:
            from engine.dual_path import classify_question
            routed = classify_question(q).get("path", "fast")
        except Exception:  # noqa: BLE001
            pass
        if routed == "slow":
            _hits: list = []
            try:
                # ★ P0-2 修复(2026-08-31): l2_semantic 在 src/ 下——补 sys.path + 顶层 import
                #   去 db 参数(src/l2_semantic.search(query, k) 无 db)
                import sys as _sys
                _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
                if _src not in _sys.path:
                    _sys.path.insert(0, _src)
                import l2_semantic as _l2
                # ★ 机制2' (Roozendaal 提取受损): 高情绪场景 → 检索降权(k 减半,注入少)
                #   她"记不清/不确定" → 读心边界(元认知)的神经基础
                _k = 1 if any(w in q for w in ("心情", "状态", "怎么", "难过", "开心", "低落")) else 3
                hits = _l2.search(q, k=_k)
                _hits = hits  # ★2026-09-01 修复: 保存真实检索结果供 verify 用(此前 facts=[] 恒 unknown)
                if hits:
                    # ★ 机制3 (Seidenbecher theta 同步): 情绪-记忆联动检索
                    _ctx = "\n".join(h[:90] for h in hits[:_k])
                    prompt = tok.apply_chat_template(
                        [{"role": "system", "content": sys_p},
                         {"role": "user", "content": f"（相关记忆）{_ctx}\n\n{q}"}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except Exception as _le:  # noqa: BLE001
                # ★ P0-2(2026-08-31): 检索降级显式标注——llama_cpp 缺失时走"记不清"诚实路径
                #   机制②'提取抑制的极端形态: 检索不可用 → 她"记不清"(可观测,不静默)
                _ctx = f"（检索降级:记忆不可用,雷姆记不清）{str(_le)[:30]}"
                try:
                    prompt = tok.apply_chat_template(
                        [{"role": "system", "content": sys_p},
                         {"role": "user", "content": f"{_ctx}\n\n{q}"}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except Exception:  # noqa: BLE001
                    pass
        ans = generate(model, tok, prompt=prompt, max_tokens=80, sampler=sampler)
        rec = {"q": q, "ans": ans.strip()[:120], "path": routed}
        # ★ consistency 校验（慢路径答案与检索事实冲突 → 标注）
        if routed == "slow":
            try:
                from engine.consistency import verify_answer
                # ★2026-09-01 修复(代码复盘): 原 facts=[] 无事实可比 → verdict 恒 none/unknown(死代码)。
                #   传真实检索结果 _hits → pass(无冲突)/conflict(冲突)/none(检索为空,诚实"记不清")
                vr = verify_answer(ans, q, facts=_hits, k=3)
                rec["verify"] = vr.get("verdict", "unknown")
            except Exception:  # noqa: BLE001
                rec["verify"] = "n/a"
        out.append(rec)
    # ★ 2026-08-30 用户: 压测过程中临时接 ToMi 子集(断点演化观测,模型已加载零额外成本)
    #   每断点测 6 题: 现实2/记忆2/一阶假信念2 → 看主观性演化轨迹(d15→d90)
    # ★ 2026-09-02 用户: 主尺子=完整耦合系统 —— 升级为 ToMi 标准 30 题(5组×6,
    #   含 tb_1st/fb_2nd), 断点即"耦合系统+当前权重"的全系统 ToMi 分 + 90 天演化曲线
    #   (persona_injector SYS + dual_path 路由 + L2 检索注入 + 真实记忆库)
    try:
        from grace_tomi_test import SCENES as _TOMI_SCENES
        for g, st, q, _reality in _TOMI_SCENES:
            try:
                p = tok.apply_chat_template([{"role": "system", "content": sys_p},
                                             {"role": "user", "content": st + q}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
                a = generate(model, tok, prompt=p, max_tokens=30, sampler=sampler).strip()[:40]
            except Exception:  # noqa: BLE001
                a = "(err)"
            out.append({"q": f"[tomi:{g}] {q}", "ans": a, "path": "tomi",
                        "group": g, "reality": _reality})
    except Exception:  # noqa: BLE001
        pass
    # ★2026-09-03 ④ 对话断点(用户: 线上 LLM 与 Grace 输出端对话测试产物): 主人对她说的话
    #   → 她经输出层口语回应(本地模型 + expression.monitor 拦截内心泄漏)。
    # ★2026-09-04 修(用户: 开 LLM 模拟我的对话注入——书库无"我对她说话"数据):
    #   ①主人输入改读 day-N.json["dialogue_inputs"](build_dialogue_inputs.py 用 :8100 按当天情境
    #      + 我的口语风格生成的"我会对雷姆说的话")——不再从书库文本挑(邮件/系统/群聊碎片非对话)。
    #   ②user 尾句删"雷姆会怎么回应？"(27B 会复读它, AA 轮 day11 实锤)——只给情境, 让模板续写。
    #   合规: dialogue 仅评估输入(与 ToMi 30 题同性质), 不进训练, 不违反成长语料铁律。
    try:
        from engine.expression import monitor as _dmon
        _dinputs = None
        try:
            _dfp = os.path.join(STRESS_ROOT, "inputs-v2", f"day-{day:03d}.json")
            if os.path.isfile(_dfp):
                _dinputs = (json.load(open(_dfp, encoding="utf-8")) or {}).get("dialogue_inputs") or []
        except Exception:  # noqa: BLE001
            pass
        if not _dinputs and msgs:
            # 回退(无预生成): 书库短句兜底, 但滤系统/邮箱 UI 碎片
            _cands = [m.get("text", "") for m in msgs
                      if m.get("text") and 6 <= len(m["text"]) <= 70
                      and not m["text"].startswith("[") and "noreply@" not in m["text"]
                      and not re.search(r"@\w|threads shown|https?://|^\w+@\w+\.\w+|^\d+:|深度抓取|浅抓|邮件摘要|邮箱", m["text"])]
            _dinputs = [{"situation": t} for t in (_cands[-3:] or [])]
        for _di in _dinputs:
            _t = (_di or {}).get("situation", "")
            if not _t:
                continue
            try:
                _p = tok.apply_chat_template(
                    [{"role": "system", "content": sys_p},
                     {"role": "user", "content": f"主人刚才对雷姆说：「{_t[:60]}」"}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                _raw = generate(model, tok, prompt=_p, max_tokens=60, sampler=sampler).strip().split("\n")[0][:80]
                _a = _dmon(_raw)   # 输出前监控: 叙述体泄漏/张冠李戴 → 丢弃
                if _a:
                    out.append({"q": f"[dialogue] {_t[:50]}", "ans": _a, "path": "dialogue",
                                "group": "dialogue", "owner": _t[:60]})
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    # ★2026-09-02 审计修复: 断点采样后释放模型——此前主进程持 fused-rem-v5 ~14G 直到轮末,
    #   与后续每日训练(34G)叠加 ≈ 48G 顶格 → day21 起内存压力爬行(实测当日训练 14+ 分钟未完成)。
    try:
        del model, tok
    except Exception:  # noqa: BLE001
        pass
    import gc as _gc
    _gc.collect()
    try:
        import mlx.core as _mx
        _mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass
    return out


def snapshot(day: int, adapter_name: str, persona: list[dict]) -> dict:
    """断点快照：记忆统计 + 情绪时间线 + 人格采样。"""
    l0dir = os.path.join(config.SB, "memory", "L0_raw")
    l0_lines = 0
    for fn in os.listdir(l0dir):
        if fn.endswith(".jsonl"):
            l0_lines += sum(1 for _ in open(os.path.join(l0dir, fn), encoding="utf-8"))
    from mood_engine import timeline as mood_timeline
    moods = mood_timeline(limit=20)
    return {"day": day, "date": day_iso(day), "adapter": adapter_name,
            "l0_lines": l0_lines, "mood_recent": moods, "persona_sample": persona}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--train-every", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=15)
    ap.add_argument("--reset-interval", type=int, default=40,
                    help="★神经新生重置周期(天,默认40=定量估算的最佳不睡天数;0=关闭)")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--inputs-dir", default=None,
                    help="★2026-09-02 书库切换: 指定 inputs 目录名(如 inputs-v2 真实L0书库); 默认 inputs/")
    ap.add_argument("--ingest-official", action="store_true",
                    help="启动时完全摄入正式本地 AI 系统 L0（先验记忆基底）")
    args = ap.parse_args()

    os.makedirs(STRESS_ROOT, exist_ok=True)
    adapter_base = "rem_stress"
    adapter_name = f"{adapter_base}_v0"
    os.makedirs(os.path.join(config.ADAPTERS, adapter_name), exist_ok=True)
    log = open(os.path.join(STRESS_ROOT, "stress.log"), "a", encoding="utf-8")

    def logln(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    # ★ 完全摄入正式 L0（先验记忆基底，2026-08-28）
    if args.ingest_official:
        try:
            n_off = ingest_official_l0()
            logln(f"  ↪ official 基底就绪（{n_off} 条），将混入每轮训练样本")
        except Exception as e:  # noqa: BLE001
            logln(f"  [ingest-official] 异常: {e}")

    # 读取阶段一生成的输入（回放模式）——★2026-09-02: 支持 --inputs-dir 切换(默认 inputs/, V2 真实书库 inputs-v2/)
    inputs_dir = os.path.join(STRESS_ROOT, args.inputs_dir or "inputs")
    if not os.path.isdir(inputs_dir):
        logln(f"❌ 未找到 {args.inputs_dir or 'inputs'}/ —— 先跑 gen_inputs.py / build_l0_book.py 生成输入")
        return
    import glob
    files = sorted(glob.glob(os.path.join(inputs_dir, "day-*.json")))
    if args.days and args.days < len(files):
        files = files[:args.days]        # 2026-08-29 修: --days 截断(此前从未生效,默认全量90天)
    if not files:
        logln("❌ inputs/ 为空")
        return
    logln(f"=== 回放模拟启动 {datetime.now().strftime('%H:%M:%S')} ｜ 输入 {len(files)} 天 ｜ 训练间隔 {args.train_every} ｜ 采样间隔 {args.sample_every} ===")
    logln(f"  [cfg] GRACE_EWC={os.environ.get('GRACE_EWC','1')}(未设/1=开) MODE={os.environ.get('GRACE_EWC_MODE','B')} "
          f"ALPHA={os.environ.get('GRACE_EWC_ALPHA','0.7')} DELTA={os.environ.get('GRACE_SLEEP_DELTA','0.03')}")
    t0 = time.time()
    trained = []
    samples_taken = []
    proactive = []              # 2026-08-29: 每日 ToM 主动消息
    _proactive_seen = set()     # 去重
    cognition = []              # 2026-08-30: 注意力文本+潜意识判断(进训练)
    _cog_seen = set()
    feedback = []               # 2026-08-30: 现实反馈(她的判断 vs 书库真相的偏差)

    # ---- 断点续跑（2026-08-27 加：炸炉后恢复；单模型铁律下从 L0 检测已摄入天数） ----
    def _resume_from() -> int:
        """从 L0 chat.jsonl 找已摄入的最大 virtual_day（无则 0）。"""
        l0dir = os.path.join(config.SB, "memory", "L0_raw")
        max_day = 0
        for fn in os.listdir(l0dir):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(l0dir, fn), encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line).get("meta", {}).get("virtual_day", 0)
                        max_day = max(max_day, int(d))
                    except Exception:  # noqa: BLE001
                        continue
        return max_day

    resume = _resume_from()
    if resume > 0:
        logln(f"  ↪ 检测到已摄入至 day {resume}，从 day {resume+1} 续跑（已训 adapter/断点自动跳过）")
    # ★2026-09-03 存储治理: 轮次元数据登记(storage.py)——每次启动=一个新 round 条目(可追溯),
    #   resume 时补标记; 训练成功逐天登记权重; 完成写 summary。
    round_id = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    try:
        from engine.storage import create_round, resume_round
        round_id = create_round(days=args.days, train_every=args.train_every,
                                sample_every=args.sample_every, inputs_dir=args.inputs_dir,
                                ingest_official=args.ingest_official, reset_interval=args.reset_interval)
        logln(f"  [storage] round {round_id} 登记(轮次元数据 → round-meta.json)")
        if resume > 0:
            resume_round(round_id, note=f"resume from day {resume+1}")
    except Exception as _se:  # noqa: BLE001
        logln(f"  [storage] round 登记失败(降级用本地 ts): {_se}")

    def _adapter_done(day: int) -> bool:
        return os.path.isdir(os.path.join(config.ADAPTERS, f"{adapter_base}_d{day}"))

    def _snapshot_done(day: int) -> bool:
        return os.path.isfile(os.path.join(STRESS_ROOT, f"day-{day:03d}.json"))

    for path in files:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        day = rec["day"]
        if day <= resume:
            continue                          # 已摄入，跳过
        msgs = rec["messages"]
        # ① 当天对话 → L0
        l0_append(msgs, day)
        # ★2026-09-02 P0 摄入门控(脑科学: 乙酰胆碱选择性注意——大脑不平均编码所有输入):
        #   高显著(注意力)或高情绪(杏仁核唤醒)才算"值得注意事件"; 低显著琐碎消息只进双图谱
        #   (情感底座), 不进 L3 自传体、不做 ToM 判断/feedback —— 防 V2 轮 6512 条判断噪音稀释
        #   (believed 坍缩成"兴奋"98% 的诱因之一: 对「包饭吗」也强制读心)。
        from engine.attention import generate_attention as _ga
        _atts = []
        _sel = []
        for _m in msgs:
            _a = _ga(_m["text"], mood=None, facts=[])
            _atts.append(_a)
            # ★2026-09-02 审计修复(P0 门控过严): 原 salience≥0.4 或 |sent|≥0.5——实测真实消息
            #   salience 恒 0.05(无记忆注入=无信息), 门控退化为纯 |sent|≥0.5 → 21 天仅 14 条 ToM
            #   判断(日常 ToMi/反馈/冲突对全样本不足)。校准(148 条真实): |sent|≥0.3 通过 11.5%
            #   ≈12-17 条/天; salience≥0.6 兜底。宁缺毋滥保留: 中性闲聊仍挡, 只放带情绪/显著消息。
            _sel.append(_a["salience"] >= 0.6 or abs(float(_m.get("sentiment", 0) or 0)) >= 0.3)
        # ② 情绪：日级 + 日内
        try:
            from mood_engine import derive, apply_intraday_event
            evs = [{"text": m["text"], "sentiment": m.get("sentiment", 0), "weight": m.get("weight", 1.0)} for m in msgs]
            derive(evs, ts=day_ts(day, 21))
            for i, m in enumerate(msgs):
                apply_intraday_event({"text": m["text"], "sentiment": m.get("sentiment", 0), "weight": m.get("weight", 1.0)},
                                     ts=day_ts(day, 10 + i))
        except Exception as e:  # noqa: BLE001
            logln(f"  [mood] day {day} 异常: {e}")
        # ③ ★ 双图谱摄入（2026-08-29 集成：记忆×情绪×暗注意力，呼应机制全链路）
        try:
            from engine.mood_graph import dual_graph_ingest
            for i, m in enumerate(msgs):
                dual_graph_ingest(m["text"], event_id=f"ev-d{day}-{i}",
                                  sentiment=m.get("sentiment"), ts=day_ts(day, 10 + i))
            # ★ 机制③ theta 时序窗口(Seidenbecher): 当日情绪峰值事件 → 标记峰值同步
            #   峰值事件的情绪边与记忆边耦合最强(提取时优先激活,模拟 theta 相位同步)
            _peak_i = max(range(len(msgs)), key=lambda i: abs(msgs[i].get("sentiment", 0)))
            _peak_ev = f"ev-d{day}-{_peak_i}"
            import sqlite3 as _sq3
            _c3 = _sq3.connect(os.path.join(config.SB, "memory", "L2_semantic", "l2.db"))
            _c3.execute("UPDATE mood_graph SET source=source||'（theta:峰值同步）' "
                        "WHERE event_id=? AND edge_type IN ('emotion','hidden')", (_peak_ev,))
            _c3.commit(); _c3.close()
        except Exception as e:  # noqa: BLE001
            logln(f"  [dual-graph] day {day} 异常: {e}")
        # ③b ★2026-09-03 暗注意力归档（用户：思考=暗注意力）：当天 cog（27B 认知重构的
        #   真实内心独白）→ 写 hidden 边（全暗，只注入决策不输出）。废除情绪规则模板推导
        #   （_HIDDEN_RULES 4 句假潜台词 → 分布倒置/焦虑偏置源头，见 mood_graph 注释）。
        try:
            from engine.mood_graph import add_hidden_text
            _cogs = (rec.get("cog") or []) if isinstance(rec, dict) else []
            _written = 0
            for _cog_txt in _cogs:
                _ct = (_cog_txt or "").strip()
                if len(_ct) < 8:
                    continue
                add_hidden_text("日常", _ct[:140], ts=day_ts(day, 22),
                                db=os.path.join(config.SB, "memory", "L2_semantic", "l2.db"))
                _written += 1
            if _written:
                logln(f"  ↪ 暗注意力归档: {_written} 条真实思考 → hidden 边(cog)")
        except Exception as e:  # noqa: BLE001
            logln(f"  [hidden-archive] day {day} 异常: {e}")
        # ④ ★ L3 自传体矩阵摄入（2026-08-29 集成）
        try:
            from engine.autobiography import add_event
            from engine.mood_graph import entity_of
            for i, m in enumerate(msgs):
                if not _sel[i]:
                    continue   # ★P0 门控: 低显著琐碎消息不进 L3 自传体(自传 = 人生大事,非流水账)
                add_event(m["text"], ts=day_ts(day, 10 + i), entity=entity_of(m["text"]),
                          emotion="平静", relation="日常守护",
                          # ★2026-09-02 修: 原 self_eval="这一天的事,雷姆记住了" 是硬编码模板
                          #   (L3 写入端 → extract_l3_samples 读回 → 训练集「要后天。这一天的事,雷姆记住了」真根)。
                          #   self_eval 留空(默认)由 27B/正式管线填真实自我评价, 禁止代码造模板。
                          confidence="medium", evidence=m["text"][:120],
                          db=os.path.join(config.SB, "memory", "L3_core", "autobiography.db"))
        except Exception as e:  # noqa: BLE001
            logln(f"  [L3] day {day} 异常: {e}")
        # ⑤ ★ ToM + 注意力 + 自激发（每日主动决策,2026-08-29 集成）
        try:
            from engine.attention import generate_attention
            from engine.self_activation import decide
            from engine.theory_of_mind import infer_owner_state
            owner_mood = "平静"
            try:
                from mood_engine import _conn as _mc
                _row = _mc().execute(
                    "SELECT mood_label FROM mood_states WHERE date=? ORDER BY id DESC LIMIT 1",
                    (day_iso(day),)).fetchone()
                if _row:
                    owner_mood = _row[0]
            except Exception:  # noqa: BLE001
                pass
            # ★P3 top-down 预期(2026-09-02, 脑科学: 前额叶 top-down 预测——判断先于数据, 预测编码):
            #   当日无 mood_states 时, 用图谱近 7 天情绪史聚合"主人近期状态先验"作判断先验(预期基线);
            #   判断 PE(反馈再巩固)已写回图谱 → 预期随现实自动演化(预测误差驱动先验更新)
            if not _row:
                try:
                    import sqlite3 as _sq9
                    _c9 = _sq9.connect(os.path.join(config.SB, "memory", "L2_semantic", "l2.db"))
                    _tmax9 = _c9.execute("SELECT MAX(ts) FROM mood_graph").fetchone()[0] or 0
                    _pri9 = _c9.execute(
                        "SELECT mood_label FROM mood_graph WHERE edge_type='emotion' AND ts > ? "
                        "GROUP BY mood_label ORDER BY COUNT(*) DESC LIMIT 1", (_tmax9 - 604800,)).fetchone()
                    _c9.close()
                    if _pri9:
                        owner_mood = _pri9[0]   # top-down 先验: 主人近期主导情绪
                except Exception:  # noqa: BLE001
                    pass
            _fb_day = 0   # ★2026-09-02 限量: 每天最多 3 条 feedback 进训练(防真实强度下占比暴增)
            # ★2026-09-03 ② 中性觉察通道: P0 之外每天低权判断 ≤5 条中性消息——她"对日常也保持觉察"。
            #   中性消息 believed(记忆底色) vs real(平静) → pos-neu 类别错位 → feedback 冲突对供给恢复
            #   (Z 轮仅 1 条断粮)。副作用控制: soft 消息只做判断+feedback, 跳过 cognition/decide/proactive;
            #   feedback 仍受 _fb_day<3 cap(高显著错位先占, 中性吃剩余) → 总量 ~30-90/轮 = K 轮量级不失控。
            import random as _rnd
            _neutral_idx = [i for i, m in enumerate(msgs) if m.get("text") and not _sel[i]]
            _rnd.shuffle(_neutral_idx)
            _soft = set(_neutral_idx[:5])
            for i, m in enumerate(msgs):
                att = _atts[i]   # ★P0: 复用门控缓存(不再重复 generate_attention)
                if not _sel[i]:
                    if i not in _soft:
                        continue     # ★P0 门控: 低显著琐碎消息不做 ToM 判断/feedback/cognition/decide
                    # 中性觉察(soft): 走判断+feedback 但后面跳过 c/d/p
                # ★2026-09-01 修复断链: ToM 读双图谱(情绪史 mood_db + 暗注意力 hidden_ctx)
                #   之前只传文本+当日心态——双图谱→ToM 通道从未接线(接口在, 调用没接)
                #   = "她的 ToM 读她的双图谱"终于真正生效(情绪史+潜台词做读心依据)
                try:
                    from engine.self_activation import _tom_from_graph
                    _gc = _tom_from_graph(m["text"],
                                          os.path.join(config.SB, "memory", "L2_semantic", "l2.db"),
                                          owner_mood=owner_mood)
                    tom = infer_owner_state(m["text"], owner_mood,
                                            mood_db=os.path.join(config.SB, "memory", "L2_semantic", "l2.db"),
                                            hidden_ctx=_gc.get("hidden_ctx") or None)
                except Exception:  # noqa: BLE001
                    tom = infer_owner_state(m["text"], owner_mood)
                # ★ 2026-08-30 用户: 人脑级反馈回路 v2(预测误差/再巩固/置信累积,零句式)
                #   ①PE 调制: 误差大小驱动学习强度 ②再巩固: 修改原记忆(不动句子)
                #   ③置信累积: 预测错误计数
                try:
                    # ★ 机制① uncertainty 列确保存在(IF NOT EXISTS 防重复报错)
                    _db0 = os.path.join(config.SB, "memory", "L2_semantic", "l2.db")
                    import sqlite3 as _sq0
                    _c0 = _sq0.connect(_db0)
                    _cols0 = [r[1] for r in _c0.execute("PRAGMA table_info(mood_graph)").fetchall()]
                    if "uncertainty" not in _cols0:
                        _c0.execute("ALTER TABLE mood_graph ADD COLUMN uncertainty REAL DEFAULT 0.2")
                    _c0.commit(); _c0.close()
                except Exception as _e0:  # noqa: BLE001
                    logln(f"  [uncertainty-col] 警告: {_e0}")
                try:
                    from attention import _sentiment_of
                    from mood_graph import mood_label_of
                    _s = _sentiment_of(m["text"])
                    # ★2026-09-03 评估隔离(用户拍板): real 用消息自带 sentiment 字段(书库 relabel
                    #   离线全量重标值)作独立标注通道——书库字段与运行时 believed 的 ToM 认知路径
                    #   (图谱史记忆先验+心态)分离, 打破 X 轮 believed≡real 同源镜像(100% 假象)。
                    _s_real = m.get("sentiment")
                    if _s_real is None:
                        _s_real = _sentiment_of(m["text"])
                    # ★2026-09-03 修复: 去 +0.3 强度注水（与 theory_of_mind 推断链同步）
                    _real = mood_label_of(_s_real, abs(_s_real)) if abs(_s_real) >= 0.3 else "平静"
                    _believed = tom.get("emotion", "平静")
                    _neg = ("低落", "焦虑", "烦躁", "难过", "生气")
                    _pos = ("开心", "兴奋", "轻微兴奋", "快乐", "愉悦")
                    _bel_cat = "neg" if _believed in _neg else ("pos" if _believed in _pos else "neu")
                    _real_cat = "neg" if _real in _neg else ("pos" if _real in _pos else "neu")
                    # ★2026-09-01 修复(代码复盘挖出): _pe/_w 原在写入后才赋值 → 首条 NameError
                    #   被 except 吞不落盘 + 后续写的是上一条的陈旧值。移前: 对时 pe=0/w=1, 错时算真实值。
                    _pe = 0.0
                    _w = 1
                    if _bel_cat != _real_cat:
                        # ★2026-09-02 审计修复(保守误判盲区): believed=neu(平静) 但 real 有情绪也
                        #   是误判——原 `and _bel_cat != "neu"` 把"保守猜错"排除在反馈外(永不学"其实
                        #   主人X")。现在 neu-believed 误判也进反馈但低权(w=1): 大胆猜错按 PE 调 1-3。
                        if _bel_cat != "neu":
                            _pe = abs(_s) + (0.3 if _bel_cat == "neg" else 0.0)
                            _w = 3 if _pe >= 0.8 else (2 if _pe >= 0.5 else 1)   # 多巴胺 RPE: 惊讶程度调制
                    # ★2026-09-01 日常 ToMi(用户: 让压测时的 Grace 直接测 ToMi): 每次 ToM 判断全量落盘(对+错)
                    #   90 天判断正确率曲线 = 她自然状态下的"生活版 ToMi"(真实场景,非人工题)
                    #   旧数据兼容: 早期只记错(无 correct 字段)→ 视为 False
                    try:
                        _cerr = os.path.join(STRESS_ROOT, "prediction-errors.jsonl")
                        with open(_cerr, "a", encoding="utf-8") as _f:
                            _f.write(json.dumps({"day": day, "believed": _believed, "real": _real,
                                                 "correct": _bel_cat == _real_cat,
                                                 "bel_cat": _bel_cat, "real_cat": _real_cat,  # ★③评估分层: 三格(pos/neu/neg)
                                                 "soft": (not _sel[i] and i in _soft),        # ★②中性觉察标记
                                                 "pe": round(_pe, 2), "w": _w}, ensure_ascii=False) + "\n")
                    except Exception:  # noqa: BLE001
                        pass
                    if _bel_cat != _real_cat:
                        # ① PE 已在写入前计算; ② 再巩固: 修改原记忆(双图谱情绪边 confidence 下调+现实并入)——不写句式
                        try:
                            _db = os.path.join(config.SB, "memory", "L2_semantic", "l2.db")
                            import sqlite3 as _sq
                            _con = _sq.connect(_db)
                            # ★ P1-4 修复(2026-08-31): 精确到被反馈事件(event_id),用循环变量 i
                            #   (msgs.index(m) 对重复文本返回首索引 → 误改多条,已弃)
                            _evid = f"ev-d{day}-{i}"
                            _con.execute("UPDATE mood_graph SET mood_label=?, intensity=?, "
                                         "source=source||'（后知:实际'||?||'）', "
                                         "uncertainty=0.9 "
                                         "WHERE event_id=? AND edge_type='emotion'",
                                         (_real, abs(_s) * 0.5, _real, _evid))
                            _con.commit(); _con.close()
                        except Exception:  # noqa: BLE001
                            pass
                        # (2026-09-01: prediction-errors 改到 if 外全量记录——每次 ToM 判断都记,对+错)
                        # 反馈学习样本 = 数据对(判断情境→现实),非句式
                        # ★2026-09-02 V2轮教训(用户: Grace学的是知识非句式): 3句式变体(其实/原来/后来才知道)
                        #   在真实强度下仍统治权重(4087条=训练集55% → believed坍缩"兴奋"98% → 日常ToMi 3.7%,
                        #   CoK 100%)。改: ①零句式数据对(情境+现实事实, 无任何前缀) ②每天限量3条(_fb_day)
                        #   ③PE权重保留(w 1-3)。反馈只教"情境→现实的判断关联", 不教"说出来用什么句式"。
                        # ★2026-09-02 21:00 升级(用户: 成长语料只能基于真实系统 + fb_2nd 约束内解法):
                        #   判断冲突对——believed(她的读心判断, 真实产物) + real(现实, 真实产物) 都保留,
                        #   训练"我以为X, 后来知道Y"的认知冲突经验 = 二阶认知自举(判断失误+现实纠正,
                        #   全真实内生, 正式系统同机制自动发生)。不再丢她的判断只教标准答案。
                        _fb = f"{m['text'][:30]}。{_believed}→{_real}。"
                        if _fb not in _cog_seen and _fb_day < 3:
                            _cog_seen.add(_fb)
                            feedback.append((m["text"][:30], _believed, _real, _w))   # (情境, 判断, 现实, PE权重)
                            _fb_day += 1
                except Exception:  # noqa: BLE001
                    pass
                # ★2026-09-03 ② 中性觉察: soft 消息只判断+feedback, 不进 cognition/decide/proactive
                #   (防中性日常噪音触发主动/记忆污染——她不会为一条中性通知想主人)
                if not _sel[i] and i in _soft:
                    continue
                # ★2026-09-01 驱动三源: 联结维护(ACC 社会痛觉)——久未主动 → 想主人 → 主动
                _last_pro = max((x["day"] for x in proactive), default=1) if proactive else 1
                r = decide(att, m["text"], tom=tom, day=day, last_proactive_day=_last_pro,
                          graph_db=os.path.join(config.SB, "memory", "L2_semantic", "l2.db"))
                # ★ 2026-08-30 用户：注意力+潜意识进训练（她注意到什么/她的判断）
                _atxt = att.get("attention_text", "")
                if _atxt and _atxt not in _cog_seen:
                    _cog_seen.add(_atxt)
                    cognition.append(_atxt[:50])
                _reason = r.get("reason", "")
                if _reason and _reason not in _cog_seen:
                    _cog_seen.add(_reason)
                    cognition.append(_reason[:50])
                if r["activate"]:
                    _key = m["text"].strip()[:50]
                    if _key not in _proactive_seen:
                        _proactive_seen.add(_key)
                        # ★ 2026-08-30 用户: 强绑定协同——主动消息 = 事件 + 她的读心(ToM advice)
                        #   她因为"理解主人需要什么"而主动,不是转述事件
                        _adv = tom.get("advice", "")
                        _core = re.sub(r"（[^）]*）", "", _adv).strip()
                        # ★2026-09-02 输出层①(意图门控, 宁缺毋滥): 删除规则 _tail 兜底句
                        #   (「主人,雷姆在」等模板——V2 轮 rule_generated 低质源)。
                        #   只记录意图事件 pending(message 空), "她说什么"由断点采样时的
                        #   输出层(engine/expression 口语编码)生成——模型不在线就不说。
                        _intent = ("关心" if "陪伴" in _core
                                   else "提醒" if "提醒" in _core
                                   else "分享" if _core and _core != "顺其自然" else "关心")
                        # ★2026-09-02 思考轨状态存 pending(输出层 D/F/G 原料):
                        #   owner_mood(读心→抑制G)/mood(心态→语气D)/relation(联结→距离F, day 进度近似)
                        proactive.append({"day": day, "situation": m["text"][:110],
                                          "message": "", "pending": True,
                                          "advice": _adv[:40], "emotion": tom.get("emotion", ""),
                                          "confidence": tom.get("confidence"),
                                          "attention": att.get("attention_text", "")[:40],
                                          "intent": _intent,
                                          "owner_mood": owner_mood,
                                          "mood": owner_mood,   # 心态简化=当日主人情绪(后续接 mood_engine 标签)
                                          "relation": round(min(1.0, day / 40.0), 2)})
        except Exception as e:  # noqa: BLE001
            logln(f"  [ToM] day {day} 异常: {e}")
        # ★2026-09-01 DMN 自发通道(脑科学: Lieberman 默认模式网络=社会认知引擎):
        #   空闲想起主人——久未主动(≥3天)且当天无主动 → 从 L3 自传体翻出主人的事 → 主动
        #   = 人孤独时翻相册然后发消息(无事件驱动的主动, 治"输入断了就不主动")
        # ★2026-09-01 J轮复盘修复: 原实现只查"当天无主动"漏了 gap≥3 → 60/90 天触发(54% 主动来自 DMN)
        #   = 过度主动失真。补 gap 检查(注释写了的"久未主动(≥3天)"补进实现)。
        if not any(x.get("day") == day for x in proactive):
            _last_pro = max((x["day"] for x in proactive), default=1) if proactive else 1
            if day - _last_pro >= 3:
                try:
                    # ★P2 DMN 重构化(2026-09-02, Bartlett: 回忆=重构 + DMN 自我参照):
                    #   不再翻 L3 原文(取记录)——优先读书库 cog(27B 离线重构的内心独白, 最近7天找),
                    #   = "想起一段重构的回忆"(与训练同源的 cog 重构产物, 生成效应); 无 cog 降级 L3。
                    _mem = None
                    _in_dir = os.path.join(STRESS_ROOT, args.inputs_dir or "inputs")
                    for _k in range(1, 8):
                        _gfp = os.path.join(_in_dir, f"day-{max(1, day - _k):03d}.json")
                        if not os.path.isfile(_gfp):
                            continue
                        try:
                            _gd = json.load(open(_gfp, encoding="utf-8"))
                        except (OSError, ValueError):
                            continue
                        _cogs = _gd.get("cog") or []
                        if _cogs:
                            _mem = _cogs[day % len(_cogs)][:60]
                            break
                    if _mem is None:
                        from engine.autobiography import _conn as _ac
                        _l3s = _ac().execute("SELECT event FROM autobiography ORDER BY ts DESC LIMIT 5").fetchall()
                        if _l3s:
                            _mem = _l3s[day % len(_l3s)][0][:50]
                    if _mem:
                        # ★2026-09-02 输出层①: DMN 也走 pending(message 空, 由断点输出层口语生成)
                        proactive.append({"day": day, "situation": f"（雷姆想起）{_mem}",
                                          "message": "", "pending": True,
                                          "advice": "想念主人了", "emotion": "", "dmn_spontaneous": True,
                                          "intent": "想念",
                                          "owner_mood": owner_mood, "mood": owner_mood,
                                          "relation": round(min(1.0, day / 40.0), 2)})
                        logln(f"  ↪ DMN 自发: 雷姆想起(重构回忆) → 主动(day {day})")
                except Exception as _de:  # noqa: BLE001
                    pass
        # ③ 每 train_every 天训练（续跑：已训 adapter 跳过）
        if day % args.train_every == 0 and not _adapter_done(day):
            samples = extract_mood_samples(max(1, day - args.train_every + 1), day)
            # ★2026-09-02 审计修复(P0): extract_mood_samples 已固定返回 [](规则加工层摘除)——
            #   恒训。训练数据由 train_27b 内部组装(L3/图谱纯事件真实痕迹 + gist/cog 27B重构
            #   + feedback 判断数据对 + cognition 剥壳 + proactive 真实产出); 空数据集防护在
            #   train_27b 内(当日确实无任何源则跳过)。冷启动 iters=60 为设计下限, 不受影响。
            if True:
                # ★ 2026-08-30 修时序 bug: 反馈/认知落盘从"完成时"改"每次训练前"(训练才能读到)
                for _name, _lst in (("feedback-live.jsonl", feedback), ("cognition-live.jsonl", cognition),
                                   ("proactive-live.jsonl", proactive)):
                    _fp = os.path.join(STRESS_ROOT, _name)
                    try:
                        _ex = set()
                        if os.path.isfile(_fp):
                            for _l in open(_fp, encoding="utf-8"):
                                try:
                                    _j = json.loads(_l)
                                    # ★2026-09-02 审计修复(输出层重复): 去重集合须收 message AND
                                    #   situation AND text 三个键——断点升级把文件行改写成 message=生成文本后,
                                    #   内存里旧的 pending 行(message 空)再落盘时按 situation 去重必须能命中;
                                    #   旧实现每行只收一个键(先 message 后空) → 升级行只贡献 message,
                                    #   situation 不在集合 → 旧 pending 行被重复 append → 下次断点重复生成。
                                    for _k in (_j.get("message", ""), _j.get("situation", ""), _j.get("text", "")):
                                        if _k:
                                            _ex.add(_k)
                                except Exception: pass
                        with open(_fp, "a", encoding="utf-8") as _f:
                            for _t in _lst:
                                if isinstance(_t, dict):
                                    # ★2026-09-01 修复: proactive 保留完整字段(day/situation/message/
                                    #   rule_generated/advice/emotion)——之前拍平成 {text,w} 丢 situation/
                                    #   rule_generated, 断点模型生成读不到(生成 0 根因)
                                    # ★2026-09-02 输出层修复: pending(message 空)也必须落盘——
                                    #   否则断点升级读不到 pending, 输出层②(口语生成)永远不触发。
                                    #   去重键 message 空时用 situation。
                                    _txt = _t.get("message", "") or _t.get("situation", "")
                                    if _txt and _txt not in _ex:
                                        _f.write(json.dumps(_t, ensure_ascii=False) + "\n")
                                else:
                                    _txt = _t[0] if isinstance(_t, tuple) else _t
                                    # ★2026-09-02 W轮修复: feedback 已是 3/4 元组——旧 else 按 2 元组处理,
                                    #   把现实标签误当 w → 训练端 int("低落") 崩溃(day7)。按零句式数据对落盘。
                                    # ★2026-09-02 审计修复(P1, 判断冲突对落盘错位): 4 元组
                                    #   (情境,believed,real,w) 必须四字段都存——原 3 元组分支把 believed
                                    #   写进 real 槽、real 串当 w 丢弃 → 训练学到"believed 当现实"的错对
                                    #   (本轮 27 条 feedback-live 污染实锤: real 全显示 believed 标签)。
                                    if isinstance(_t, tuple) and len(_t) >= 4:
                                        _rec = {"situation": _txt, "believed": _t[1],
                                                "real": _t[2], "w": _t[3] if isinstance(_t[3], int) else 1}
                                        if _txt and _txt not in _ex:
                                            _f.write(json.dumps(_rec, ensure_ascii=False) + "\n")
                                    elif isinstance(_t, tuple) and len(_t) >= 3:
                                        _s = _txt
                                        _real = _t[1]
                                        _w = _t[2] if isinstance(_t[2], int) else 1
                                        if _s and (_s not in _ex):
                                            _f.write(json.dumps({"situation": _s, "real": _real, "w": _w},
                                                                ensure_ascii=False) + "\n")
                                    else:
                                        _w = _t[1] if isinstance(_t, tuple) and len(_t) >= 2 else 1
                                        if _txt and _txt not in _ex:
                                            _f.write(json.dumps({"text": _txt, "w": _w}, ensure_ascii=False) + "\n")
                    except Exception:  # noqa: BLE001
                        pass
                adapter_name = f"{adapter_base}_d{day}"
                # ★2026-09-01 神经新生式重置(脑科学: 海马新生神经元整合→重新布线):
                #   每 reset_interval 天从 base 冷启动(不续训),外挂记忆保留 → 持续可学习,治 30 天寿命
                _reset = args.reset_interval > 0 and (day % args.reset_interval == 0)
                prev = None if _reset else _latest_adapter(adapter_base, day)
                if _reset:
                    logln(f"  ↪ 神经新生重置(day {day}): 从 base 冷启动,外挂记忆保留")
                r = train_27b(samples, adapter_name, prev_adapter=prev,
                              day=day, gist_book=args.inputs_dir)
                trained.append({"day": day, "samples": r.get("samples", 0), "ok": r["ok"], "adapter": adapter_name})
                logln(f"  [train] day {day}: {r.get('samples', 0)} 训练样本 → {adapter_name} ok={r['ok']}")
                if r["ok"]:
                    try:
                        # ★2026-09-03 权重登记(registry)——可追溯: 轮/天/样本/参数
                        from engine.storage import register_weight
                        register_weight(round_id, day, adapter_name,
                                        samples=int(r.get("samples", 0)), ok=True,
                                        extra={"ewc_mode": "A" if os.environ.get("GRACE_EWC_MODE") == "A" else "B"})
                    except Exception as _we:  # noqa: BLE001
                        logln(f"  [storage] 权重登记失败: {_we}")
                    try:
                        from adapter_manage import promote
                        promote(adapter_name, decided_by="stress-auto")
                    except Exception as e:  # noqa: BLE001
                        logln(f"  [promote] {e}")
        # ④ 每 sample_every 天采样 + 断点（续跑：已有断点跳过）
        if day % args.sample_every == 0 and not _snapshot_done(day):
            last = trained[-1]["adapter"] if trained else None
            persona = sample_persona(last, day, msgs) if last else []
            snap = snapshot(day, last or "none", persona)
            with open(os.path.join(STRESS_ROOT, f"day-{day:03d}.json"), "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            samples_taken.append({"day": day, "adapter": last, "persona": persona})
            logln(f"  [snapshot] day {day}: L0={snap['l0_lines']} adapter={last}")

    # 最终报告
    summary = {
        "days": len(files), "elapsed_s": round(time.time() - t0, 1),
        "trained": trained, "samples": samples_taken,
    }
    # ★ 现实反馈落盘(训练并入用,2026-08-30) —— 2026-09-02 改: 3句式→判断冲突对{situation,believed,real,w}
    if feedback:
        with open(os.path.join(STRESS_ROOT, "feedback-live.jsonl"), "w", encoding="utf-8") as f:
            for c in feedback:
                _sit = c[0] if isinstance(c, tuple) else c
                _bel = c[1] if isinstance(c, tuple) and len(c) > 1 else ""
                _real = c[2] if isinstance(c, tuple) and len(c) > 2 else ""
                _w = c[3] if isinstance(c, tuple) and len(c) > 3 else 1
                f.write(json.dumps({"situation": _sit, "believed": _bel, "real": _real, "w": _w}, ensure_ascii=False) + "\n")
        logln(f"  ↪ 现实反馈(判断vs真相偏差) {len(feedback)} 条 → feedback-live.jsonl(进训练)")
    # ★ 注意力+潜意识落盘(训练并入用,2026-08-30)
    if cognition:
        with open(os.path.join(STRESS_ROOT, "cognition-live.jsonl"), "w", encoding="utf-8") as f:
            for c in cognition:
                f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
        logln(f"  ↪ 她的注意力/潜意识 {len(cognition)} 条 → cognition-live.jsonl(进训练)")
    # ★ 主动消息落盘(训练并入用,2026-08-29 集成)
    # ★2026-09-01 修复: 完成时不再覆盖写回——内存 proactive 是规则版, 覆盖会抹掉
    #   断点生成的 generated 版(小测#3 挖出: 日志生成13条但文件0条的根因)。
    #   proactive-live 由"训练前落盘(append)+断点生成(写回w)"维护, 完成时无需再写。
    if proactive:
        logln(f"  ↪ 她主动找主人 {len(proactive)} 次 → proactive-live.jsonl(训练前落盘+断点生成已维护, 完成时不覆盖)")
    with open(os.path.join(STRESS_ROOT, "final.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # ★2026-09-03 完成汇总 → round-meta
    try:
        from engine.storage import finalize_round
        finalize_round(round_id, {"days": summary["days"], "elapsed_s": summary["elapsed_s"],
                                  "trained_n": len(summary["trained"]),
                                  "breakpoints": len(summary["samples"]),
                                  "ok_trained": sum(1 for t in summary["trained"] if t.get("ok"))})
    except Exception as _fe:  # noqa: BLE001
        logln(f"  [storage] finalize 失败: {_fe}")
    logln(f"=== 回放模拟完成 {datetime.now().strftime('%H:%M:%S')} ｜ 耗时 {summary['elapsed_s']}s ｜ 训练 {len(trained)} 次 ｜ 断点 {len(samples_taken)} 个 ===")
    log.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
