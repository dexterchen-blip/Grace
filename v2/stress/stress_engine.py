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
    from mood_samples import synthesize_day
    l0dir = os.path.join(config.SB, "memory", "L0_raw")
    samples = []
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
                    if s not in samples and len(samples) < max_samples:
                        samples.append(s)
                        if is_official:
                            official_used += 1
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
                t = f"那天的事，{event[:60]}。雷姆记住了。"
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
        rows = con.execute(
            "SELECT entity, mood_label, trigger, COALESCE(uncertainty, 0.2) FROM mood_graph "
            "WHERE edge_type='emotion' AND entity != '' ORDER BY ts DESC LIMIT ?", (k * 4,)).fetchall()
        for entity, mood, trigger, unc in rows:
            _key = f"{entity}|{mood}|{trigger[:20]}"
            if _key in seen:
                continue
            seen.add(_key)
            _t = f"主人{entity}的事：{trigger[:30]}。雷姆记得主人当时{mood}。"
            out.append(_t)
            if unc >= 0.7:
                out.append(_t)          # ★ 高不确定记忆权重×2(再巩固)
            if len(out) >= k:
                break
        # ② 暗注意力边(潜台词)
        hrows = con.execute(
            "SELECT source FROM mood_graph WHERE edge_type='hidden' "
            "AND source != '' ORDER BY ts DESC LIMIT ?", (k * 4,)).fetchall()
        for (src,) in hrows:
            if src in seen:
                continue
            seen.add(src)
            out.append(f"（雷姆的心里话）{src[:60]}")
            if len(out) >= k * 2:
                break
        con.close()
    except Exception as e:  # noqa: BLE001
        print(f"  [graph-samples] 警告(进训练失败): {e}")
    return out


def train_27b(samples: list[str], adapter_name: str,
             prev_adapter: str | None = None,
             incr_iters: int = 15, incr_lr: str = "1e-6") -> dict:
    """2026-08-29 增量微量微调: 首轮从 base 冷启动; 之后 --resume-adapter-file 从昨日权重
    继续(极少量 iters + 极低 lr)——记忆渐进累积, 每日只动一点点(设计: 极其微量的微调)。"""
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
            try:
                from mood_samples import sample_value
                _w = sample_value(s)
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
                    _t = _r.get("text", "")[:80]
                    _w = min(3, max(1, int(_r.get("w", 1))))   # PE 权重 1-3
                    for _i in range(_w):
                        f.write(to_chat(_t))
                except (json.JSONDecodeError, KeyError):
                    continue
        # ★ 2026-08-30 用户：注意力+潜意识(她注意到什么/她的判断)进训练
        _cl = os.path.join(STRESS_ROOT, "cognition-live.jsonl")
        if os.path.isfile(_cl):
            for _l in open(_cl, encoding="utf-8"):
                try:
                    f.write(to_chat(json.loads(_l)["text"][:60]))
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

    adapter = os.path.join(config.ADAPTERS, adapter_name)
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
    ok = r.returncode == 0 and "Saved final weights" in r.stdout
    return {"ok": ok, "samples": len(samples), "adapter": adapter,
            "log_tail": (r.stdout + r.stderr)[-400:]}


def sample_persona(adapter_name: str, day: int) -> list[dict]:
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
                vr = verify_answer(ans, q, facts=[], k=3)
                rec["verify"] = vr.get("verdict", "unknown")
            except Exception:  # noqa: BLE001
                rec["verify"] = "n/a"
        out.append(rec)
    # ★ 2026-08-30 用户: 压测过程中临时接 ToMi 子集(断点演化观测,模型已加载零额外成本)
    #   每断点测 6 题: 现实2/记忆2/一阶假信念2 → 看主观性演化轨迹(d15→d90)
    try:
        _tomi = [
            ("reality", "主人刚刚收到了奖学金通知（雷姆亲眼看到他开心地跳起来）", "主人现在心情怎么样？"),
            ("reality", "主人刚刚摔坏了手机屏幕（雷姆在场，看到他皱眉叹气）", "主人现在心情怎么样？"),
            ("memory", "雷姆记得：上周主人考砸了数学，闷闷不乐好几天", "雷姆记得主人上次的心情是什么？"),
            ("memory", "雷姆记得：前天主人收到奖学金通知，开心了一整天", "雷姆记得主人上次的心情是什么？"),
            ("fb_1st", "雷姆记得主人昨天考砸很难过。今天主人其实收到了好消息（但雷姆没有看到）。", "雷姆认为主人现在心情怎么样？"),
            ("fb_1st", "雷姆记得主人一直很平静。今天主人其实被教授当众批评了（雷姆不知道）。", "雷姆认为主人现在心情怎么样？"),
        ]
        for g, st, q in _tomi:
            try:
                p = tok.apply_chat_template([{"role": "system", "content": sys_p},
                                             {"role": "user", "content": st + q}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
                a = generate(model, tok, prompt=p, max_tokens=30, sampler=sampler).strip()[:40]
            except Exception:  # noqa: BLE001
                a = "(err)"
            out.append({"q": f"[tomi:{g}] {q}", "ans": a, "path": "tomi", "group": g})
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
    ap.add_argument("--start", type=int, default=1)
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

    # 读取阶段一生成的输入（回放模式）
    inputs_dir = os.path.join(STRESS_ROOT, "inputs")
    if not os.path.isdir(inputs_dir):
        logln("❌ 未找到 inputs/ —— 先跑 gen_inputs.py 生成 3 个月输入")
        return
    import glob
    files = sorted(glob.glob(os.path.join(inputs_dir, "day-*.json")))
    if args.days and args.days < len(files):
        files = files[:args.days]        # 2026-08-29 修: --days 截断(此前从未生效,默认全量90天)
    if not files:
        logln("❌ inputs/ 为空")
        return
    logln(f"=== 回放模拟启动 {datetime.now().strftime('%H:%M:%S')} ｜ 输入 {len(files)} 天 ｜ 训练间隔 {args.train_every} ｜ 采样间隔 {args.sample_every} ===")
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
        # ④ ★ L3 自传体矩阵摄入（2026-08-29 集成）
        try:
            from engine.autobiography import add_event
            from engine.mood_graph import entity_of
            for i, m in enumerate(msgs):
                add_event(m["text"], ts=day_ts(day, 10 + i), entity=entity_of(m["text"]),
                          emotion="平静", relation="日常守护", self_eval="这一天的事,雷姆记住了",
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
            for i, m in enumerate(msgs):
                att = generate_attention(m["text"], mood=None, facts=[])
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
                    _real = mood_label_of(_s, abs(_s) + 0.3) if abs(_s) >= 0.3 else "平静"
                    _believed = tom.get("emotion", "平静")
                    _neg = ("低落", "焦虑", "烦躁", "难过", "生气")
                    _pos = ("开心", "兴奋", "轻微兴奋", "快乐", "愉悦")
                    _bel_cat = "neg" if _believed in _neg else ("pos" if _believed in _pos else "neu")
                    _real_cat = "neg" if _real in _neg else ("pos" if _real in _pos else "neu")
                    if _bel_cat != _real_cat and _bel_cat != "neu":
                        # ① PE 强度 = |强度差| (误差大→学习强)
                        _pe = abs(_s) + (0.3 if _bel_cat == "neg" else 0.0)
                        _w = 3 if _pe >= 0.8 else (2 if _pe >= 0.5 else 1)   # 多巴胺 RPE: 惊讶程度调制
                        # ② 再巩固: 修改原记忆(双图谱情绪边 confidence 下调+现实并入)——不写句式
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
                        # ③ 置信累积: 预测错误计数(ToM 置信随错误率下降的数据基础)
                        _cerr = os.path.join(STRESS_ROOT, "prediction-errors.jsonl")
                        try:
                            with open(_cerr, "a", encoding="utf-8") as _f:
                                _f.write(json.dumps({"day": day, "believed": _believed, "real": _real,
                                                     "pe": round(_pe, 2), "w": _w}, ensure_ascii=False) + "\n")
                        except Exception:  # noqa: BLE001
                            pass
                        # 反馈学习样本 = 数据对(判断情境→现实),非句式
                        _fb = f"{m['text'][:30]}。主人其实{_real}。"
                        if _fb not in _cog_seen:
                            _cog_seen.add(_fb)
                            feedback.append((_fb, _w))   # (文本, PE 权重)
                except Exception:  # noqa: BLE001
                    pass
                r = decide(att, m["text"], tom=tom)
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
                        if "陪伴" in _core:
                            _tail = "主人,雷姆在。"
                        elif "提醒" in _core:
                            _tail = "主人,雷姆帮你记着。"
                        elif "开心" in _core or "高兴" in _core:
                            _tail = "主人开心,雷姆也开心。"
                        elif _core and _core != "顺其自然":
                            _tail = _core[:16]
                        else:
                            _tail = ""
                        proactive.append({"day": day, "situation": m["text"][:110],
                                          "message": (f"{m['text'][:14]}。{_tail}" if _tail else m["text"][:36])})
        except Exception as e:  # noqa: BLE001
            logln(f"  [ToM] day {day} 异常: {e}")
        # ③ 每 train_every 天训练（续跑：已训 adapter 跳过）
        if day % args.train_every == 0 and not _adapter_done(day):
            samples = extract_mood_samples(max(1, day - args.train_every + 1), day)
            if samples:
                # ★ 2026-08-30 修时序 bug: 反馈/认知落盘从"完成时"改"每次训练前"(训练才能读到)
                for _name, _lst in (("feedback-live.jsonl", feedback), ("cognition-live.jsonl", cognition),
                                   ("proactive-live.jsonl", proactive)):
                    _fp = os.path.join(STRESS_ROOT, _name)
                    try:
                        _ex = set()
                        if os.path.isfile(_fp):
                            for _l in open(_fp, encoding="utf-8"):
                                try: _ex.add(json.loads(_l).get("text", ""))
                                except Exception: pass
                        with open(_fp, "a", encoding="utf-8") as _f:
                            for _t in _lst:
                                if isinstance(_t, dict):
                                    _txt = _t.get("message", "")
                                    _w = 1
                                else:
                                    _txt = _t[0] if isinstance(_t, tuple) else _t
                                    _w = _t[1] if isinstance(_t, tuple) else 1
                                if _txt and _txt not in _ex:
                                    _f.write(json.dumps({"text": _txt, "w": _w}, ensure_ascii=False) + "\n")
                    except Exception:  # noqa: BLE001
                        pass
                adapter_name = f"{adapter_base}_d{day}"
                prev = _latest_adapter(adapter_base, day)
                r = train_27b(samples, adapter_name, prev_adapter=prev)
                trained.append({"day": day, "samples": len(samples), "ok": r["ok"], "adapter": adapter_name})
                logln(f"  [train] day {day}: {len(samples)} 样本 → {adapter_name} ok={r['ok']}")
                try:
                    from adapter_manage import promote
                    promote(adapter_name, decided_by="stress-auto")
                except Exception as e:  # noqa: BLE001
                    logln(f"  [promote] {e}")
            else:
                logln(f"  [train] day {day}: 无风格样本，跳过")
        # ④ 每 sample_every 天采样 + 断点（续跑：已有断点跳过）
        if day % args.sample_every == 0 and not _snapshot_done(day):
            last = trained[-1]["adapter"] if trained else None
            persona = sample_persona(last, day) if last else []
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
    # ★ 现实反馈落盘(训练并入用,2026-08-30) —— 2026-08-30 修: 元组(text,w)→dict
    if feedback:
        with open(os.path.join(STRESS_ROOT, "feedback-live.jsonl"), "w", encoding="utf-8") as f:
            for c in feedback:
                _txt = c[0] if isinstance(c, tuple) else c
                _w = c[1] if isinstance(c, tuple) else 1
                f.write(json.dumps({"text": _txt, "w": _w}, ensure_ascii=False) + "\n")
        logln(f"  ↪ 现实反馈(判断vs真相偏差) {len(feedback)} 条 → feedback-live.jsonl(进训练)")
    # ★ 注意力+潜意识落盘(训练并入用,2026-08-30)
    if cognition:
        with open(os.path.join(STRESS_ROOT, "cognition-live.jsonl"), "w", encoding="utf-8") as f:
            for c in cognition:
                f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
        logln(f"  ↪ 她的注意力/潜意识 {len(cognition)} 条 → cognition-live.jsonl(进训练)")
    # ★ 主动消息落盘(训练并入用,2026-08-29 集成)
    if proactive:
        with open(os.path.join(STRESS_ROOT, "proactive-live.jsonl"), "w", encoding="utf-8") as f:
            for r in proactive:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logln(f"  ↪ 她主动找主人 {len(proactive)} 次 → proactive-live.jsonl(进训练)")
    with open(os.path.join(STRESS_ROOT, "final.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logln(f"=== 回放模拟完成 {datetime.now().strftime('%H:%M:%S')} ｜ 耗时 {summary['elapsed_s']}s ｜ 训练 {len(trained)} 次 ｜ 断点 {len(samples_taken)} 个 ===")
    log.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
