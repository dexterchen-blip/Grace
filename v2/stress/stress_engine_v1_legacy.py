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


def train_27b(samples: list[str], adapter_name: str) -> dict:
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
    sys_p = ("你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。自称「雷姆」，"
             "称呼亲近的人为「巴鲁斯」/「昴君」，称拉姆为「姐姐大人」。"
             "表面冷淡礼貌、实则温柔忠诚，说话短句为主，带黑色幽默与毒舌吐槽。")

    def to_chat(text: str) -> str:
        return json.dumps({"messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": "（与主人的日常对话）"},
            {"role": "assistant", "content": text},
        ]}, ensure_ascii=False) + "\n"

    with open(os.path.join(ds_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for t in anchor_texts[:n_anchor]:
            f.write(to_chat(t))
        for s in samples:
            f.write(to_chat(s))
        # ★ 2026-08-28：主动消息进训练（她会主动关心主人）——v3 ToM 实时化的产物
        _pt = "/Users/cz/WorkBuddy/watch/ai-sandbox-stress/grace-book/run/proactive-train-v3.jsonl"
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
                        {"role": "user", "content": f"（情境）{_r['situation'][:80]}"},
                        {"role": "assistant", "content": _r["message"][:120]}]}, ensure_ascii=False) + "\n")
                    _n += 1
            if _n:
                print(f"    [proactive] 并入主动消息样本 {_n} 条（她会主动关心主人）")

    adapter = os.path.join(config.ADAPTERS, adapter_name)
    cmd = [sys.executable, "-m", "mlx_lm.lora",
           "--model", MAIN_MODEL, "--train", "--data", ds_dir,
           "--adapter-path", adapter,
           "--batch-size", "1", "--iters", "150", "--learning-rate", "1e-5",
           "--num-layers", "16", "--max-seq-length", "2048",
           "--grad-checkpoint",                       # 2026-08-28：16层 27B 必开，峰值 20.7GB 防 Metal OOM
           "--steps-per-report", "20", "--steps-per-eval", "50",
           "--save-every", "50", "--seed", "42"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    ok = r.returncode == 0 and "Saved final weights" in r.stdout
    return {"ok": ok, "samples": len(samples), "adapter": adapter,
            "log_tail": (r.stdout + r.stderr)[-400:]}


def sample_persona(adapter_name: str, day: int) -> list[dict]:
    """用 27B + 当前 adapter 采样 4 题（人格一致性观测）。"""
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
    sys_p = ("你是雷姆（Rem，蕾姆），罗兹瓦尔宅邸的女仆，鬼族，拉姆的妹妹。自称「雷姆」，"
             "称呼亲近的人为「巴鲁斯」/「昴君」，称拉姆为「姐姐大人」。"
             "【重要】直接说出你的台词，不要描写动作、表情、环境，不要使用括号旁白，不要叙述性前缀。")
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
        ans = generate(model, tok, prompt=prompt, max_tokens=80, sampler=sampler)
        out.append({"q": q, "ans": ans.strip()[:120]})
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
            evs = [{"text": m["text"], "sentiment": m["sentiment"], "weight": m["weight"]} for m in msgs]
            derive(evs, ts=day_ts(day, 21))
            for i, m in enumerate(msgs):
                apply_intraday_event({"text": m["text"], "sentiment": m["sentiment"], "weight": m["weight"]},
                                     ts=day_ts(day, 10 + i))
        except Exception as e:  # noqa: BLE001
            logln(f"  [mood] day {day} 异常: {e}")
        # ③ 每 train_every 天训练（续跑：已训 adapter 跳过）
        if day % args.train_every == 0 and not _adapter_done(day):
            samples = extract_mood_samples(max(1, day - args.train_every + 1), day)
            if samples:
                adapter_name = f"{adapter_base}_d{day}"
                r = train_27b(samples, adapter_name)
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
    with open(os.path.join(STRESS_ROOT, "final.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logln(f"=== 回放模拟完成 {datetime.now().strftime('%H:%M:%S')} ｜ 耗时 {summary['elapsed_s']}s ｜ 训练 {len(trained)} 次 ｜ 断点 {len(samples_taken)} 个 ===")
    log.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
