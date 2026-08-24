#!/usr/bin/env python3
"""M5 夜班巩固管线（设计 §7/§4，grill Q12/Q19/Q21/Q33）。

四段流水线：
  Segment 1 (03:00)  摄入：m4_ingest all → L0_raw
  Segment 2 (03:30)  嵌入：l2_semantic build → L2_semantic
  Segment 3 (04:00)  巩固：serve_night.sh → 35B-A3B LLM 巩固 L0/L1 → L3 + 报告 + 提案
  Segment 4 (08:00)  看门狗：kill server → L3 git snapshot → expire proposals → finalize report

门控（2026-08-18 用户定）：Segment 1 全部完成才允许进 Segment 3，资料未齐不巩固。
交付物（2026-08-18 用户定）：夜班报告 = changelog + AI 自主判断的「今日应在意的学校/微信信息」优先级清单。

安全机制：
  - 每段 heartbeat + exit code（exchange/shared/heartbeat.json）
  - 单任务 10min 超时强杀
  - 连续两晚无报告 → red_alert=True（dashboard 红点位）
  - 每晚 L3 git 快照
  - persona 隔离：只喂 mode=normal，persona 子树永不进巩固

用法：
  python3 night_pipeline.py run                    # 全量跑（launchd 入口）
  python3 night_pipeline.py run --use-day-model    # 开发期用白天 27B 代替夜间 35B
  python3 night_pipeline.py segment3               # 只跑巩固段（调试用）
  python3 night_pipeline.py status                 # 查管线状态
  python3 night_pipeline.py report                 # 显示最近一份夜班报告

环境：标准库 only（urllib 调 OpenAI 兼容 API）。子进程调 m4_ingest.py / l2_semantic.py / serve_night.sh。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L0_ROOT = os.path.join(REPO, "memory", "L0_raw")
L1_ROOT = os.path.join(REPO, "memory", "L1_working")
L3_ROOT = os.path.join(REPO, "memory", "L3_core")
L3_PATH = os.path.join(L3_ROOT, "core.md")
STATE_PATH = os.path.join(L1_ROOT, "consolidation_state.json")
HEARTBEAT_PATH = os.path.join(REPO, "exchange", "shared", "heartbeat.json")
REPORT_DIR = os.path.join(REPO, "exchange", "outbox", "reports")

# 测试沙盒（2026-08-22）：AIAGENT_SANDBOX=<dir> 时全部写路径重定向到沙盒，
# 与正式记忆/交换系统完全隔离。沙盒目录须含 memory/ + exchange/ 结构。
SANDBOX = os.environ.get("AIAGENT_SANDBOX", "")
# 沙盒真抓取开关（2026-08-23）：沙盒模式默认跳过门控真抓取（保守隔离）；
# 设 ALLOW_SANDBOX_SCRAPE=1 时允许门控在沙盒里真实触发抓取（email/ucsb/canvas），
# 落点仍走沙盒 exchange —— 用于验证「夜班自动真抓取 + 摄入」配合，正式库零污染。
ALLOW_SANDBOX_SCRAPE = os.environ.get("ALLOW_SANDBOX_SCRAPE") == "1"
if SANDBOX:
    L0_ROOT = os.path.join(SANDBOX, "memory", "L0_raw")
    L1_ROOT = os.path.join(SANDBOX, "memory", "L1_working")
    L3_ROOT = os.path.join(SANDBOX, "memory", "L3_core")
    L3_PATH = os.path.join(L3_ROOT, "core.md")
    STATE_PATH = os.path.join(L1_ROOT, "consolidation_state.json")
    HEARTBEAT_PATH = os.path.join(SANDBOX, "exchange", "shared", "heartbeat.json")
    REPORT_DIR = os.path.join(SANDBOX, "exchange", "outbox", "reports")

PY = sys.executable
L2_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "l2_semantic.py")
# 夜班模型用 llama-cpp venv（有 sqlite_vec + llama_cpp）；摄入用 3.13 管理器
L2_VENV = "~/.workbuddy/binaries/python/envs/llama-cpp/bin/python"

TZ_CN = timezone(timedelta(hours=8))
SEGMENT_TIMEOUT = 600  # 10 min per segment


# ================================================================ 状态

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_cn() -> str:
    return datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S")


def today_cn() -> str:
    return datetime.now(TZ_CN).strftime("%Y-%m-%d")


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_consolidation": None,
        "last_l0_count": 0,
        "last_report_path": None,
        "consecutive_failures": 0,
        "red_alert": False,
        "segments": {},
    }


def save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)


# ================================================================ heartbeat

def write_heartbeat(segment: str, status: str, detail: str = "") -> None:
    os.makedirs(os.path.dirname(HEARTBEAT_PATH), exist_ok=True)
    hb = {
        "segment": segment,
        "status": status,
        "detail": detail,
        "timestamp": now_iso(),
        "local_time": now_cn(),
        "pid": os.getpid(),
    }
    with open(HEARTBEAT_PATH, "w", encoding="utf-8") as f:
        json.dump(hb, f, ensure_ascii=False, indent=2)


# ================================================================ Segment 1: 摄入

def _prefetch_gate() -> list[str]:
    """§7/§13 门控：巩固前资料必须抓取完毕。

    检查各源新鲜度，对过期源做 best-effort 触发抓取（本地 AI 主权触发，
    不依赖 WorkBuddy 自动化）。失败不阻塞管线——记入报告让用户知道。
    返回触发日志行。
    """
    log = []
    sys.path.insert(0, os.path.join(REPO, "src"))
    from scrape import source_status, scrape_email, scrape_outlook
    # L-1 瞬时记忆日清（2026-08-20 用户定）：exchange/.daytime/ 是白天对话触发的
    # 临时抓取（时间敏感性，像人的瞬时记忆），晚上抓取最完整 → 夜班开始时清空，
    # 隔夜作废。清理动作绝不阻塞管线。沙盒模式清沙盒自己的 .daytime。
    try:
        daytime = os.path.join(SANDBOX if SANDBOX else REPO, "exchange", ".daytime")
        if os.path.isdir(daytime):
            n = 0
            for fn in os.listdir(daytime):
                if fn.startswith("."):  # 水位文件（.wechat_state.json 等）保留
                    continue
                p = os.path.join(daytime, fn)
                try:
                    os.remove(p)
                    n += 1
                except OSError:
                    pass
            if n:
                log.append(f"门控: L-1 瞬时记忆已清空（{n} 个白天临时抓取文件）")
    except Exception as e:
        log.append(f"门控: L-1 清理异常 {e}")
        print(f"[seg1] L-1 清理异常: {e}")
    # email：2026-08-20 起「每次夜班必抓」，必须放在「全部新鲜提前返回」之前——
    # 8/19 下午抓过后 8/20 凌晨被判 fresh 跳过，导致今日邮件全部缺失（用户发现）。
    # 邮件时效性强，宁可多抓也不漏。**沙盒模式跳过真抓取**（测试不碰正式 exchange；
    # ALLOW_SANDBOX_SCRAPE=1 时放行，落点仍走沙盒）。
    if SANDBOX and not ALLOW_SANDBOX_SCRAPE:
        log.append("门控: 沙盒模式，跳过 email/outlook 真抓取")
    else:
        try:
            if scrape_email():
                log.append("门控: email 直抓完成（每次夜班必抓）")
            else:
                log.append("门控: email 直抓失败（Edge 未登录/VPN 不通）")
        except Exception as e:  # best-effort，绝不让门控杀死管线
            log.append(f"门控: email 直抓异常 {e}")
            print(f"[seg1] 门控 email 异常: {e}")
        # outlook：2026-08-20 用户指示一并纳入夜班必抓（SSO 抖动/失败如实记录，不阻塞）。
        try:
            if scrape_outlook():
                log.append("门控: outlook 直抓完成（每次夜班必抓）")
            else:
                log.append("门控: outlook 直抓失败（SSO 会话过期或 CDP 起不来）")
        except Exception as e:
            log.append(f"门控: outlook 直抓异常 {e}")
            print(f"[seg1] 门控 outlook 异常: {e}")
    st = source_status()
    stale = [k for k, v in st.items() if not v["fresh"]]
    if not stale:
        log.append("门控: 全部数据源新鲜")
        print("[seg1] 门控: 全部数据源新鲜")
        return log
    log.append(f"门控: 过期源 {stale}")
    print(f"[seg1] 门控: 过期源 {stale}，尝试触发抓取")
    # ucsb-scrape：2026-08-19 起改为门控触发（本地 AI 主权触发，对齐 email 逻辑）。
    # 注意 run_sync 每天 04:00 已由 com.user.ucsb-data 跑，但产出在 skill 的 output/，
    # 需经 scrape_ucsb 同步进 exchange 契约落点；gold 8/10 后 302 会话问题会如实报告。
    # 沙盒模式跳过（ucsb 抓取写正式 exchange/school，防泄漏；ALLOW_SANDBOX_SCRAPE 放行落沙盒）。
    if "ucsb-scrape" in stale and (not SANDBOX or ALLOW_SANDBOX_SCRAPE):
        try:
            from scrape import scrape_ucsb
            n = scrape_ucsb()
            log.append(f"门控: ucsb 同步触发（exchange 落点新增 {n} 件）"
                       if n else "门控: ucsb 无今日产出（GOLD 会话可能过期）")
            print(f"[seg1] 门控 ucsb 触发: {n} 件")
        except Exception as e:
            log.append(f"门控: ucsb 同步异常 {e}")
            print(f"[seg1] 门控 ucsb 异常: {e}")

    # canvas-scrape（2026-08-23 接入）：Canvas 课程全量抓取（Page 内容+附件+公告）。
    # 学术诚信红线：canvas_bot 只碰 type=Page / 文件 / 公告，绝不打开测验/LTI。
    # 门控过期（36h）才触发，避免每夜重抓；沙盒模式默认跳过（防写正式 exchange），
    # ALLOW_SANDBOX_SCRAPE=1 时沙盒内真实触发（落点仍走沙盒 exchange）。
    if "canvas-scrape" in stale and (not SANDBOX or ALLOW_SANDBOX_SCRAPE):
        try:
            from scrape import scrape_canvas
            n = scrape_canvas()
            log.append(f"门控: canvas 抓取触发（exchange 落点新增 {n} 件）"
                       if n else "门控: canvas 无今日产出（会话过期？）")
            print(f"[seg1] 门控 canvas 触发: {n} 件")
        except Exception as e:
            log.append(f"门控: canvas 抓取异常 {e}")
            print(f"[seg1] 门控 canvas 异常: {e}")

    # 重要邮件深抓（用户 2026-08-19 定夺）：浅抓后自动对重要邮件取全文。
    # 只要今日浅抓文件存在就跑，不限于 email 过期——白天手动抓过也照样深抓。
    # **沙盒模式跳过深抓**（email_deep 写正式 exchange/inbox/email，防泄漏）。
    if SANDBOX:
        log.append("门控: 沙盒模式，跳过重要邮件深抓")
    else:
        from scrape import today as _today
        shallow = os.path.join(REPO, "exchange", "inbox", "email", f"邮箱摘要-{_today()}.md")
        if os.path.isfile(shallow):
            try:
                import email_deep
                if email_deep.run():
                    log.append("门控: 重要邮件深抓完成（见 邮箱深度-*.md）")
                else:
                    log.append("门控: 深抓无产出（无重要邮件或全部失败）")
            except Exception as e:
                log.append(f"门控: 深抓异常 {e}")
                print(f"[seg1] 门控 深抓异常: {e}")
    return log


def segment1_ingest() -> dict:
    """m4_ingest all → L0_raw。返回 {'new_records': N, 'sources': {...}}"""
    write_heartbeat("segment1", "running", "prefetch gate → m4_ingest all")
    print(f"[seg1] {now_cn()} 摄入段启动 — 门控 + m4_ingest all")

    gate_log = _prefetch_gate()

    m4 = os.path.join(REPO, "src", "m4_ingest.py")
    result = subprocess.run(
        [PY, m4, "all"],
        capture_output=True, text=True, timeout=SEGMENT_TIMEOUT,
        cwd=REPO,
    )
    if result.returncode != 0:
        write_heartbeat("segment1", "failed", result.stderr[-500:])
        raise RuntimeError(f"seg1 m4_ingest failed: {result.stderr[-300:]}")

    # 统计 L0 行数
    l0_count = 0
    sources = {}
    for fn in os.listdir(L0_ROOT):
        if not fn.endswith(".jsonl"):
            continue
        src = fn[:-6]
        with open(os.path.join(L0_ROOT, fn), encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        sources[src] = n
        l0_count += n

    detail = f"L0 total={l0_count}, sources={sources}"
    write_heartbeat("segment1", "done", detail)
    print(f"[seg1] {now_cn()} 摄入完成 — {detail}")

    # 图像识别摄入（2026-08-22 用户方案）：发现 exchange/inbox 图片 → Qwen3-VL-4B
    # 识别（描述+OCR）→ L0(vision:image)。best-effort 不阻塞；VL 按需拉起、用完停。
    # 沙盒模式自动跟随 AIAGENT_SANDBOX（vision_ingest 内部重定向）。
    try:
        sys.path.insert(0, os.path.join(REPO, "src"))
        import vision_ingest
        vn = vision_ingest.run()
        print(f"[seg1] 图像识别摄入: {vn} 张")
    except Exception as e:
        print(f"[seg1] 图像识别异常（不阻塞）: {e}")

    # 文档识别摄入（2026-08-22 用户方案）：exchange/inbox 新文档（pdf/docx/xlsx…）
    # → 文本提取 → L0(doc:file)。doc-parse venv 子进程；沙盒跟随。
    try:
        DOC_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc_ingest.py")
        DOC_VENV = "~/.workbuddy/binaries/python/envs/doc-parse/bin/python"
        if not os.path.exists(DOC_VENV):
            print("[seg1] doc-parse venv 未装，跳过文档识别")
        else:
            r = subprocess.run([DOC_VENV, DOC_PY], capture_output=True, text=True,
                               timeout=900, cwd=REPO)
            print(f"[seg1] 文档识别: {(r.stdout or r.stderr).strip()[-120:]}")
    except Exception as e:
        print(f"[seg1] 文档识别异常（不阻塞）: {e}")

    return {"new_records": l0_count, "sources": sources, "exit_code": 0,
            "gate_log": gate_log}


# ================================================================ Segment 2: 嵌入

def segment2_embed() -> dict:
    """l2_semantic build → L2_semantic。"""
    write_heartbeat("segment2", "running", "l2_semantic build")
    print(f"[seg2] {now_cn()} 嵌入段启动 — l2_semantic build")

    result = subprocess.run(
        [L2_VENV, L2_PY, "build"],
        capture_output=True, text=True, timeout=SEGMENT_TIMEOUT,
        cwd=REPO,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    if result.returncode != 0:
        write_heartbeat("segment2", "failed", result.stderr[-500:])
        raise RuntimeError(f"seg2 l2_semantic build failed: {result.stderr[-300:]}")

    # 读索引规模
    stats_result = subprocess.run(
        [L2_VENV, L2_PY, "stats"],
        capture_output=True, text=True, timeout=30, cwd=REPO,
    )
    detail = stats_result.stdout.strip().replace("\n", "; ")
    write_heartbeat("segment2", "done", detail)
    print(f"[seg2] {now_cn()} 嵌入完成 — {detail}")
    return {"exit_code": 0, "stats": stats_result.stdout.strip()}


# ================================================================ Segment 3: 巩固

def _start_night_model(use_day: bool = False) -> subprocess.Popen | None:
    """拉起夜班模型服务（或开发期白天模型）。返回 Popen 或 None（已常驻则不拉）。"""
    port = 8100 if use_day else 8200
    # 探活
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
        urllib.request.urlopen(req, timeout=5)
        print(f"[seg3] 模型服务已在 :{port} 运行，复用")
        return None
    except (urllib.error.URLError, OSError):
        pass

    script = os.path.join(REPO, "src", "serve_day.sh" if use_day else "serve_night.sh")
    if not os.path.exists(script):
        if use_day:
            print("[seg3] serve_day.sh 不存在，尝试连 8100")
            return None
        print(f"[seg3] {script} 不存在，回退白天模型 :8100")
        return _start_night_model(use_day=True)

    print(f"[seg3] 拉起模型服务: {script}")
    # 输出必须落文件不能用 PIPE：llama.cpp 启动时打印整个 chat template +
    # Metal 着色器编译日志（远超 64KB 管道缓冲），无人读管道 → 子进程写满即
    # 阻塞 → 服务永远起不来（2026-08-19 03:00 launchd 首跑真凶，手动跑落文件所以正常）。
    server_log = os.path.join(REPO, "exchange", "shared", "serve-night.log")
    log_fp = open(server_log, "ab", buffering=0)
    proc = subprocess.Popen(
        ["bash", script],
        stdout=log_fp, stderr=subprocess.STDOUT,
        cwd=REPO,
    )
    # 等就绪（最多 300 秒；35B 冷启动 ~75s，Metal 着色器重编译/内存 contention 时更慢）
    for i in range(60):
        time.sleep(5)
        # 子进程早死（缺依赖/端口被占）→ 快速失败，别干等 300s
        if proc.poll() is not None:
            write_heartbeat("segment3", "failed", f"模型服务早退 exit={proc.returncode}")
            log_fp.close()
            raise RuntimeError(
                f"模型服务启动后早退 exit={proc.returncode}，见 {os.path.relpath(server_log, REPO)}")
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
            urllib.request.urlopen(req, timeout=5)
            print(f"[seg3] 模型服务就绪（~{i*5}s）")
            log_fp.close()
            return proc
        except (urllib.error.URLError, OSError):
            continue
    write_heartbeat("segment3", "failed", f"模型服务 {port} 300s 未就绪")
    proc.kill()
    log_fp.close()
    raise RuntimeError(f"模型服务 {port} 300s 未就绪，见 {os.path.relpath(server_log, REPO)}")


def _stop_night_model(proc: subprocess.Popen | None, use_day: bool = False) -> None:
    """跑完杀掉夜班模型（白天模型常驻不杀）。"""
    if use_day:
        print("[seg3] 白天模型常驻，不杀")
        return
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[seg3] 夜班模型已停止")
    # 也直接杀 8200 上的监听进程（保险）
    subprocess.run(["bash", "-c", "lsof -nP -iTCP:8200 -sTCP:LISTEN -t 2>/dev/null | xargs kill 2>/dev/null"],
                   timeout=5)


# ---- 夜班单模型错峰（2026-08-22 修：48GB 装不下 27B+35B 双驻留 43.5GB + 推理峰值 → watchdog panic）----
DAY_MODEL_LABEL = "com.local-ai-agent.day-model"
DAY_MODEL_PLIST = "~/Library/LaunchAgents/com.local-ai-agent.day-model.plist"


def _suspend_day_model() -> None:
    """夜班开始：暂停 27B 常驻（launchd day-model bootout），释放 ~15.5GB，35B 独占。"""
    subprocess.run(["launchctl", "bootout", f"gui/501/{DAY_MODEL_LABEL}"],
                   capture_output=True, text=True, timeout=10)
    subprocess.run(["bash", "-c", "lsof -nP -iTCP:8100 -sTCP:LISTEN -t 2>/dev/null | xargs kill 2>/dev/null"],
                   timeout=5)
    print("[seg3] 27B 已暂停（夜班单模型制，防 43GB 双驻留 panic）")


def _resume_day_model() -> None:
    """夜班结束：恢复 27B 常驻。"""
    subprocess.run(["launchctl", "bootstrap", "gui/501", DAY_MODEL_PLIST],
                   capture_output=True, text=True, timeout=10)
    print("[seg3] 27B 已恢复常驻")


def _start_review_model() -> subprocess.Popen | None:
    """临时起 27B 用于深度审校（35B 已停，单模型制）。返回 Popen 或 None。"""
    script = os.path.join(REPO, "src", "serve_day.sh")
    logf = open(os.path.join(REPO, "exchange", "shared", "serve-day.log"), "ab")
    try:
        return subprocess.Popen(["bash", script, "8100"], stdout=logf, stderr=logf,
                                cwd=REPO, start_new_session=True)
    except Exception as e:
        print(f"[seg3] 审校模型拉起失败: {e}")
        return None


def _stop_review_model(proc: subprocess.Popen | None) -> None:
    """停掉临时审校 27B。"""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    subprocess.run(["bash", "-c", "lsof -nP -iTCP:8100 -sTCP:LISTEN -t 2>/dev/null | xargs kill 2>/dev/null"],
                   timeout=5)
    print("[seg3] 审校 27B 已停止")


def _get_model_id(port: int) -> str:
    """从 /v1/models 拿真实 model id。"""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read())
    models = data.get("data", [])
    if models:
        return models[0]["id"]
    return "qwen3.5-35b-night"  # fallback


def _strip_think(text: str) -> str:
    """剥掉 thinking 输出，只留正式回答。

    llama_cpp server 实测形态（2026-08-18）：chat template 把开标签 <think>
    当生成前缀吃掉，content 里只有「思考文本 + </think> + 正式回答」，
    没有开标签；也可能有完整 <think>...</think> 对或截断未闭合块。三种都处理。
    """
    import re
    # 完整 <think>...</think> 对
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 只剩闭合标签（llama_cpp 形态）：闭合标签前的一切都是思考
    if "</think>" in text:
        text = text.split("</think>")[-1]
    # 未闭合 <think>（被 max_tokens 截断）——整块丢弃
    if "<think>" in text:
        text = text[:text.index("<think>")]
    return text.strip()


def _llm_chat(port: int, model_id: str, system: str, user: str,
              max_tokens: int = 6144, no_think: bool = True) -> str:
    """调 OpenAI 兼容 chat API，返回文本。10min 超时。

    no_think=True（开发模式 Qwen3.8）：user 末尾加 /no_think 抑制思考，省时间。
    no_think=False（生产 35B，2026-08-18 用户定）：保留 thinking 换巩固质量，
      夜班几乎无时间限制；max_tokens 需加大（思考本身吃 token），
      返回前剥 <think> 块。
    """
    if no_think:
        user = user.rstrip() + "\n/no_think"
    payload = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer sk-local"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=SEGMENT_TIMEOUT)
    data = json.loads(resp.read())
    choice = data["choices"][0]
    msg = choice["message"]
    # Qwen3.x: content 可能缺失（reasoning 吃光 token），回退 reasoning 字段
    content = msg.get("content")
    if content:
        return _strip_think(content)
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    finish = choice.get("finish_reason", "?")
    print(f"[seg3] [warn] content 缺失 (finish={finish})，回退 reasoning ({len(reasoning)} chars)")
    return _strip_think(reasoning) if reasoning else ""


def _gather_new_l0(st: dict, include_cloud_drop: bool = False) -> str:
    """收集上次巩固后的新 L0 记录，格式化为给 LLM 的文本。

    水位 = last_consolidation_ts（时间戳）；首跑无水位时取最近 15 条/源。
    cloud-drop 来源用**独立水位** cloud_drop_ts（门控跳过时不推进，
    否则被跳过的数据会随主水位前进而永远丢失）。
    include_cloud_drop=False 时跳过 cloud-drop 来源（§11：云端数据经提案门控后才巩固）。
    """
    last_ts = st.get("last_consolidation_ts", 0)
    cloud_ts = st.get("cloud_drop_ts", 0)
    is_first = last_ts == 0
    lines = []
    total_new = 0
    skipped_cloud = 0
    for fn in sorted(os.listdir(L0_ROOT)):
        if not fn.endswith(".jsonl"):
            continue
        source = fn[:-6]
        records = []
        with open(os.path.join(L0_ROOT, fn), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        # cloud-drop 门控（urgent:* 与 exchange:cloud-drop 同源云端数据，同样隔离）
        is_cloud = source.startswith("exchange:cloud-drop") or source.startswith("urgent:cloud-drop")
        # 按时间戳过滤新记录（cloud-drop 走独立水位）
        if is_cloud:
            new_recs = [r for r in records if r.get("epoch", 0) > cloud_ts]
        elif is_first:
            new_recs = records[-15:]  # 首跑取最近 15 条
        else:
            new_recs = [r for r in records if r.get("epoch", 0) > last_ts]
        if not new_recs:
            continue
        if is_cloud and not include_cloud_drop:
            skipped_cloud += len(new_recs)
            continue
        total_new += len(new_recs)
        lines.append(f"\n### Source: {source} ({len(new_recs)} records)")
        for rec in new_recs[:30]:  # 限制条数防止 prompt 爆
            p = rec.get("payload", {})
            if "messages" in p:  # wechat
                msgs = p.get("messages", [])
                conv = p.get("conversation", "?")
                snippet = " | ".join(
                    f"{m.get('display_name','?')}: {m.get('text','')[:60]}"
                    for m in msgs[:4]
                )
                lines.append(f"  [{conv}] {snippet[:200]}")
            elif "text" in p:  # file/corpus
                text = str(p["text"])[:200]
                fname = p.get("filename", p.get("path", "?"))
                lines.append(f"  [{fname}] {text}")
    if not lines:
        lines.append("(无新记录)")
    lines.append(f"\n(本次巩固覆盖 {total_new} 条新记录)")
    if skipped_cloud:
        lines.append(f"(另有 {skipped_cloud} 条 cloud-drop 云端数据待提案门控批准，未纳入本次巩固)")
    return "\n".join(lines)


def _cloud_drop_gate() -> tuple[bool, int]:
    """cloud-drop 巩固门控（§11）。返回 (是否批准, 待门控记录数)。

    有 approved 的 consolidate 提案（target 含 cloud-drop）→ 放行；
    有未批准的新 cloud-drop 数据 → 自动建一条 consolidate 提案（等用户点头）。
    水位用独立的 cloud_drop_ts（被门控跳过的数据不丢，下次继续待批）。
    """
    sys.path.insert(0, os.path.join(REPO, "src"))
    from proposal_queue import ProposalQueue
    pq = ProposalQueue()
    st = load_state()
    cloud_ts = st.get("cloud_drop_ts", 0)
    pending_count = 0
    for prefix in ("exchange:cloud-drop", "urgent:cloud-drop"):
        fp = os.path.join(L0_ROOT, prefix + ".jsonl")
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("epoch", 0) > cloud_ts:
                        pending_count += 1
    approved = pq.gate_check("consolidate", "cloud-drop")
    if approved:
        for prop in approved:
            pq.mark_executed(prop["id"])
        return True, pending_count
    if pending_count:
        # 去重：已有 pending 的 cloud-drop 巩固提案就不重复建（否则每晚一条刷屏）
        existing = [p for p in pq.list("pending")
                    if p["type"] == "consolidate"
                    and "cloud-drop" in p.get("target", {}).get("path", "")]
        if existing:
            print(f"[seg3] cloud-drop 门控：{pending_count} 条待批，提案 {existing[0]['id']} 已存在，跳过重复创建")
        else:
            pid = pq.create(
                type="consolidate",
                title=f"巩固 cloud-drop 云端数据（{pending_count} 条）",
                description="云端投放数据隔离待批。批准后下次夜班巩固将纳入 L2/L3。",
                target={"path": "exchange/cloud-drop/", "action": "consolidate",
                        "details": {"pending_records": pending_count}},
                source="night-pipeline",
                priority="medium",
            )
            if pid:
                print(f"[seg3] cloud-drop 门控：{pending_count} 条待批 → 提案 {pid}")
    return False, pending_count


def _search_l2_for_context(topics: list[str]) -> str:
    """对关键主题跑 L2 检索，获取上下文。每次 search 独立加载 bge-m3，所以合并为 1 次调用。"""
    # 合并主题为 1 个查询（省 2 次模型加载）
    combined = " ".join(topics[:2])
    lines = ["\n### L2 语义检索上下文"]
    try:
        result = subprocess.run(
            [L2_VENV, L2_PY, "search", combined, "-k", "5"],
            capture_output=True, text=True, timeout=60, cwd=REPO,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
        if result.returncode == 0 and result.stdout.strip():
            lines.append(result.stdout.strip()[:800])
    except (subprocess.TimeoutExpired, Exception) as e:
        lines.append(f"(L2 检索失败: {e})")
    return "\n".join(lines)


def _parse_llm_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON（容错：找 { 到 } 的范围，尝试修复截断的 JSON）。"""
    # 尝试直接 parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 找 ```json ... ``` 块
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start) if "```" in text[start:] else len(text)
        try:
            return json.loads(text[start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass
    # 找第一个 { 到最后一个 }
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        chunk = text[first:last + 1]
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
        # 尝试修复截断的 JSON：补全未闭合的括号
        repaired = _repair_truncated_json(text[first:])
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    return None


def _deep_review(parsed: dict, phase1_raw: str) -> dict:
    """阶段2 深度审校（2026-08-21 用户方案：35B 广度 → 27B 深度）。
    白天 27B（:8100 常驻，零额外内存）审校夜班 35B 的巩固产出：
      - 只修正 35B 产出【内部】的冲突/错误，不扩大范围、不引入无关待办；
      - 冲突检测：priority_list 条目 vs new_facts/用户最新陈述 矛盾（如新事实说已降级
        却仍标 urgent）→ 降级/移除该条目 + 生成对应 modification（更新 L3 旧待办）。
    best-effort：27B 不可用/超时/输出异常 → 原样返回 35B 产出。"""
    try:
        import urllib.request
        payload = json.dumps({
            "model": "mlx-community/Qwen3.8-27B-4bit",
            "messages": [
                {"role": "system", "content": (
                    "你是记忆巩固的深度审校员。输入是夜班模型(35B)的巩固产出 JSON"
                    "（含 new_facts/modifications/priority_list/summary）。"
                    "只做三件事，其余一律保持 35B 原样，绝不扩大范围：\n"
                    "1) 冲突修正：若 priority_list 中某条目与 new_facts 或用户最新陈述矛盾"
                    "（例如新事实说\"已基本完成/9月后提醒\"，priority 却标 urgent/high）→ "
                    "降级或移除该条目，并在 modifications 里补一条修改 L3 旧待办的提案"
                    "（modifications 的 target 写 L3 待办的原始文本、item 写待办描述、"
                    "action 用 update 或 remove，不要用 priority_list 索引做 target）；\n"
                    "2) 错误修正：modifications/priority_list 里的明显事实错误改掉；\n"
                    "3) 不要新增与本次数据无关的待办，不要重写整份 priority_list。\n"
                    "只输出与输入同结构的 JSON，无 markdown。")},
                {"role": "user", "content": f"35B 产出 JSON：\n{phase1_raw}"},
            ],
            "max_tokens": 6144, "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8100/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
        content = resp["choices"][0]["message"].get("content") or ""
        reviewed = _parse_llm_json(content)
        if reviewed and reviewed.get("new_facts") is not None:
            print(f"[seg3] 深度审校完成（27B 修正 35B 产出）")
            return reviewed
        print(f"[seg3] [warn] 深度审校输出无法解析，保留 35B 产出")
    except Exception as e:
        print(f"[seg3] 深度审校降级（用 35B 原产出）: {e}")
    return parsed


def _repair_truncated_json(s: str) -> str | None:
    """尝试修复被 max_tokens 截断的 JSON：处理未闭合的字符串、括号。"""
    # 找最后一个完整的 },（数组元素之间的分隔符）或 }]（数组最后一个元素）
    # 策略：从后往前找 }, 或 }] 或 } 模式，截断到那里，再补闭合括号
    for i in range(len(s) - 1, -1, -1):
        if s[i] == "}":
            # 检查这个 } 后面是 , 或 ] 或空白（说明这是一个完整的对象）
            after = s[i+1:i+3].lstrip() if i+1 < len(s) else ""
            if after.startswith(",") or after.startswith("]") or after == "":
                # 截断到这个 } 后面的逗号（如果有），去掉 trailing comma
                cut = i + 1
                if after.startswith(","):
                    cut = i + 2  # 包含逗号
                truncated = s[:cut]
                # 如果末尾是逗号，去掉
                truncated = truncated.rstrip().rstrip(",")
                # 数未闭合的括号
                depth_brace = 0
                depth_bracket = 0
                in_str = False
                esc = False
                for c in truncated:
                    if esc:
                        esc = False
                        continue
                    if c == "\\":
                        esc = True
                        continue
                    if c == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if c == "{":
                        depth_brace += 1
                    elif c == "}":
                        depth_brace -= 1
                    elif c == "[":
                        depth_bracket += 1
                    elif c == "]":
                        depth_bracket -= 1
                # 补闭合
                for _ in range(depth_bracket):
                    truncated += "]"
                for _ in range(depth_brace):
                    truncated += "}"
                return truncated
    return None


CONSOLIDATION_SYSTEM = """You are a memory consolidation agent for a local AI assistant system.
Your job is to analyze today's new data and update the user's core memory.

IMPORTANT: Output ONLY the JSON object. Do not think out loud. Do not explain your reasoning. Just output the JSON directly.

Rules:
1. New facts → add to core memory directly (append-only, never delete)
2. Modifications to existing facts → create proposals (user must approve)
3. Identify what the user should care about tomorrow, prioritized by urgency
4. Only process mode=normal data (persona data is isolated, never consolidated)
5. Be concise and factual. No speculation. If unsure, don't add.
6. **Conflict rule (2026-08-21)**: if today's new facts/chat contradict an existing todo/status in core memory (e.g. user says a task is done or deprioritized), the user's latest statement WINS: (a) exclude that item from priority_list or downgrade it, (b) MUST generate a modification proposal updating the stale todo, (c) never keep flagging as urgent what the user already resolved.
7. **Reminder rule (2026-08-23)**: if a todo in core memory carries `remind_after=YYYY-MM-DD`, do NOT mark it high/urgent in priority_list before that date (downgrade to low/medium or omit). User's scheduled reminders take precedence over email urgency — e.g. "immunization remind_after=2026-09-06" must NOT appear as high/urgent before Sept 6 even if ACTION-REQUIRED emails keep arriving.

Output ONLY a JSON object with this exact structure (no markdown, no explanation):
{"new_facts":[{"category":"school|wechat|general|tech","fact":"one-line fact","priority":"high|medium|low"}],"modifications":[{"target":"what section","old":"current text","new":"proposed text","reason":"why"}],"priority_list":{"school":[{"item":"what","priority":"high|medium|low","reason":"why","deadline":"date or null"}],"wechat":[{"item":"what","priority":"high|medium|low","reason":"why","sender":"who or null"}]},"summary":"one-line summary"}"""


def segment3_consolidate(st: dict, use_day_model: bool = False) -> dict:
    """巩固段：拉起夜班模型 → LLM 巩固 → 写 L3 + 报告 + 提案。"""
    write_heartbeat("segment3", "running", "starting night model + LLM consolidation")
    print(f"[seg3] {now_cn()} 巩固段启动")

    # 门控：检查 segment1 是否完成
    seg1_status = st.get("segments", {}).get("segment1", {}).get("status")
    if seg1_status != "done":
        msg = f"门控失败：segment1 状态={seg1_status}，资料未齐不巩固"
        write_heartbeat("segment3", "skipped", msg)
        print(f"[seg3] {msg}")
        return {"exit_code": -1, "reason": msg}

    # 拉起模型
    # 【错峰】先暂停 27B 常驻（release ~15.5GB）→ 35B 独占；开发模式（本就 27B）跳过。
    review_proc = None
    if not use_day_model:
        _suspend_day_model()
    proc = _start_night_model(use_day_model)
    port = 8100 if use_day_model else 8200
    model_id = _get_model_id(port)
    print(f"[seg3] 模型: {model_id} @ :{port}")

    try:
        # cloud-drop 门控（§11：云端数据先隔离，批准后才进巩固）
        cloud_ok, cloud_pending = _cloud_drop_gate()
        # 收集数据
        new_l0_text = _gather_new_l0(st, include_cloud_drop=cloud_ok)
        l3_content = ""
        if os.path.exists(L3_PATH):
            with open(L3_PATH, encoding="utf-8") as f:
                l3_content = f.read()
        l2_context = _search_l2_for_context([
            "UCSB deadline tuition orientation housing important message",
        ])

        # 构建 prompt
        user_prompt = f"""## Current core memory (L3):
{l3_content[:3000]}

## Today's new data (L0, since last consolidation):
{new_l0_text}

## L2 semantic search context:
{l2_context}

## Task:
Analyze today's data and update core memory. Focus on:
1. School-related items (UCSB, courses, tuition, orientation, housing, visa, insurance)
2. WeChat messages that require action or attention
3. Any deadlines approaching

Output the JSON object now."""

        write_heartbeat("segment3", "llm_call", f"model={model_id}, prompt~{len(user_prompt)}chars")
        print(f"[seg3] LLM 调用中（prompt ~{len(user_prompt)} chars）…")

        # 生产 35B：保留 thinking（用户定夺：质量优先，夜班无时间限制），max_tokens 加大；
        # 开发 27B：/no_think + 6144 tokens 求快。
        if use_day_model:
            llm_response = _llm_chat(port, model_id, CONSOLIDATION_SYSTEM, user_prompt,
                                     max_tokens=6144, no_think=True)
        else:
            llm_response = _llm_chat(port, model_id, CONSOLIDATION_SYSTEM, user_prompt,
                                     max_tokens=16384, no_think=False)
        print(f"[seg3] LLM 返回 {len(llm_response)} chars")

        # 解析
        parsed = _parse_llm_json(llm_response)
        if not parsed:
            print(f"[seg3] [warn] LLM 输出非 JSON，原始输出作报告附录")
            parsed = {
                "new_facts": [],
                "modifications": [],
                "priority_list": {"school": [], "wechat": []},
                "summary": "LLM 输出解析失败，原始输出见附录",
                "_raw_llm": llm_response[:2000],
            }

        # 阶段2 深度审校 + 知识图谱（2026-08-21 用户方案：35B 广度 → 27B 深度）：
        # 【错峰】35B 巩固完成 → 停 35B（释放 ~28GB）→ 临时起 27B → 图谱抽取 + 审校 → 停 27B。
        # 任何时刻只有一个模型驻留（48GB 装不下 27B+35B 双驻留，2026-08-22 panic 教训）。
        # 【2026-08-23 改】图谱抽取从 35B 改为 27B：35B thinking 保留（巩固质量优先，
        #   8/18 用户定）导致图谱抽取时输出思考过程 → JSON 污染/截断 → 0 实体；
        #   27B 无 thinking、JSON 干净、输出快（沙盒 5 事实实测 19 关系）。
        # use_day_model（开发模式 27B 就是巩固模型）图谱同样跑 27B。
        if not use_day_model:
            _stop_night_model(proc, use_day_model)
            review_proc = _start_review_model()
            for _ in range(40):  # 最多 ~120s 等 27B 就绪
                time.sleep(3)
                try:
                    urllib.request.urlopen("http://127.0.0.1:8100/v1/models", timeout=3)
                    break
                except Exception:
                    pass
        # 知识图谱抽取（27B，JSON 干净）：增量 L2 文档 → 三元组 → l2.db 图谱表。
        # best-effort 不阻塞；失败绝不影响巩固。
        try:
            KG_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_graph.py")
            r = subprocess.run(
                [L2_VENV, KG_PY, "extract", "--port", "8100",
                 "--model", "mlx-community/Qwen3.8-27B-4bit"],
                capture_output=True, text=True, timeout=1500, cwd=REPO)
            print(f"[seg3] 图谱抽取(27B): {(r.stdout or r.stderr).strip()[-120:]}")
        except Exception as e:
            print(f"[seg3] 图谱抽取异常（不阻塞）: {e}")
        # 深度审校（27B 修正 35B 产出；use_day_model 跳过——本来就是 27B）
        if not use_day_model:
            parsed = _deep_review(parsed, llm_response)
            _stop_review_model(review_proc)

        # 写 L3（新事实直接追加）
        new_facts = parsed.get("new_facts", [])
        if new_facts:
            with open(L3_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n## {today_cn()} 夜班巩固新增\n")
                for fact in new_facts:
                    cat = fact.get("category", "general")
                    text = fact.get("fact", "")
                    pri = fact.get("priority", "low")
                    f.write(f"- [{cat}/{pri}] {text}\n")
            print(f"[seg3] L3 追加 {len(new_facts)} 条新事实")

        # 修改/删除 → 提案队列
        modifications = parsed.get("modifications", [])
        proposal_ids = []
        if modifications:
            sys.path.insert(0, os.path.join(REPO, "src"))
            from proposal_queue import ProposalQueue
            pq = ProposalQueue()
            for mod in modifications:
                pid = pq.create(
                    type="modify",
                    title=f"修改 L3: {mod.get('target', '?')[:60]}",
                    description=f"旧: {mod.get('old', '')[:100]}\n新: {mod.get('new', '')[:100]}\n原因: {mod.get('reason', '')}",
                    target={"path": "memory/L3_core/core.md", "action": "update",
                            "details": {"target": mod.get("target"), "old": mod.get("old"), "new": mod.get("new")}},
                    source="night-consolidation",
                    priority="medium",
                )
                if pid:
                    proposal_ids.append(pid)
            print(f"[seg3] 创建 {len(proposal_ids)} 条修改提案 → pending/")

        # 生成夜班报告（2026-08-24：best-effort——LLM 输出结构不稳定可能触发模板 bug，
        # 失败降级写最小报告，绝不 exit=1；数据（L3/提案/图谱）都已落库）
        try:
            report_path = _write_night_report(st, parsed, new_facts, modifications, proposal_ids, llm_response)
        except Exception as _re:
            import traceback as _tb2
            print(f"[seg3] 报告生成失败（降级最小报告，不退出）: {_re}")
            print(_tb2.format_exc())
            report_path = os.path.join(REPORT_DIR, f"night-{today_cn()}.md")
            os.makedirs(REPORT_DIR, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as _f:
                _f.write(f"# 夜班报告 {today_cn()}\n\n> 报告生成降级（模板异常）。数据已在 L3/提案/图谱。\n\n- L3 新增 {len(new_facts)} 条\n- 提案 {len(proposal_ids)} 条\n")
        st["last_report_path"] = os.path.relpath(report_path, REPO)

        # 更新状态（时间戳水位供下次巩固增量用）
        st["last_consolidation"] = now_iso()
        st["last_consolidation_ts"] = time.time()
        if cloud_ok:
            st["cloud_drop_ts"] = time.time()  # cloud-drop 独立水位只在放行纳入后推进

        write_heartbeat("segment3", "done", f"facts={len(new_facts)}, proposals={len(proposal_ids)}, report={report_path}")
        print(f"[seg3] {now_cn()} 巩固完成 — facts={len(new_facts)}, proposals={len(proposal_ids)}")

        return {
            "exit_code": 0,
            "new_facts": len(new_facts),
            "proposals": len(proposal_ids),
            "report": os.path.relpath(report_path, REPO),
        }

    finally:
        _stop_review_model(review_proc)  # 兜底：审校 27B 若残留则停（None 安全）
        _stop_night_model(proc, use_day_model)
        if not use_day_model:
            _resume_day_model()  # 恢复 27B 常驻（重启自启）


def _write_night_report(st: dict, parsed: dict, new_facts: list,
                        modifications: list, proposal_ids: list[str],
                        raw_llm: str) -> str:
    """写夜班报告到 exchange/outbox/reports/night-YYYY-MM-DD.md。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    date = today_cn()
    path = os.path.join(REPORT_DIR, f"night-{date}.md")

    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    lines = [
        f"# 夜班报告 {date}",
        "",
        f"> 生成时间：{now_cn()}",
        f"> 巩固模型：Qwen3.5-35B-A3B Q6_K（llama.cpp）" if not st.get("_use_day") else
        f"> 巩固模型：Qwen3.8-27B（MLX，开发模式）",
        "",
        "## 概要",
        "",
        parsed.get("summary", "（无摘要）"),
        "",
        "## Changelog",
        "",
    ]

    # Changelog
    seg1 = st.get("segments", {}).get("segment1", {})
    seg2 = st.get("segments", {}).get("segment2", {})
    # §7 门控日志（数据源新鲜度/触发结果）
    for g in seg1.get("gate_log", []):
        lines.append(f"- [门控] {g}")
    sources = seg1.get("sources", {})
    for src, n in sources.items():
        lines.append(f"- [摄入] {src}: {n} 条记录")
    lines.append(f"- [嵌入] L2 索引更新（{str(seg2.get('stats') or 'N/A')[:80]})")
    lines.append(f"- [巩固] L3 新增 {len(new_facts)} 条事实")
    lines.append(f"- [提案] 创建 {len(proposal_ids)} 条修改提案（待审批）")

    # 新事实
    if new_facts:
        lines.extend(["", "## 新增事实记忆", ""])
        for f in new_facts:
            cat = f.get("category", "?")
            pri = f.get("priority", "?")
            text = f.get("fact", "")
            lines.append(f"- [{cat}/{pri}] {text}")

    # 今日应在意的学校信息（2026-08-24：priority_list 可能 None/项可能 None，全兜底防崩）
    pl = parsed.get("priority_list") or {}
    school_items = (pl.get("school") or []) if isinstance(pl, dict) else []
    school_items = [x for x in school_items if isinstance(x, dict)]
    lines.extend(["", "## 今日应在意的学校信息", ""])
    if school_items:
        for item in sorted(school_items, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority"), 3)):
            pri = item.get("priority", "low")
            text = item.get("item", "")
            reason = item.get("reason", "")
            deadline = item.get("deadline")
            icon = priority_emoji.get(pri, "⚪")
            dl = f" | 截止: {deadline}" if deadline else ""
            lines.append(f"{icon} **[{pri.upper()}]** {text}{dl}")
            if reason:
                lines.append(f"   _原因：{reason}_")
    else:
        lines.append("（今夜无特别需要关注的学校事项）")

    # 今日应在意的微信信息
    wechat_items = (pl.get("wechat") or []) if isinstance(pl, dict) else []
    wechat_items = [x for x in wechat_items if isinstance(x, dict)]
    lines.extend(["", "## 今日应在意的微信信息", ""])
    if wechat_items:
        for item in sorted(wechat_items, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority"), 3)):
            pri = item.get("priority", "low")
            text = item.get("item", "")
            reason = item.get("reason", "")
            sender = item.get("sender")
            icon = priority_emoji.get(pri, "⚪")
            s = f" | 来自: {sender}" if sender else ""
            lines.append(f"{icon} **[{pri.upper()}]** {text}{s}")
            if reason:
                lines.append(f"   _原因：{reason}_")
    else:
        lines.append("（今夜无特别需要关注的微信事项）")

    # 待审批提案
    if proposal_ids:
        lines.extend(["", "## 待审批提案", ""])
        lines.append("以下修改需要你点头才会执行（被否决的永不再提）：")
        mods = [m for m in (modifications or []) if isinstance(m, dict)]
        for i, pid in enumerate(proposal_ids):
            mod = mods[i] if i < len(mods) else {}
            target = mod.get("target", "?")
            reason = mod.get("reason", "")
            lines.append(f"- `{pid}` — 修改 {target}: {reason}")

    # 系统
    lines.extend(["", "## 系统状态", ""])
    lines.append(f"- 摄入: {'✓' if seg1.get('status') == 'done' else '✗'}")
    lines.append(f"- 嵌入: {'✓' if seg2.get('status') == 'done' else '✗'}")
    lines.append(f"- 巩固: ✓")
    lines.append(f"- 连续失败: {st.get('consecutive_failures', 0)}")
    if st.get("red_alert"):
        lines.append("- ⚠ **红点告警：连续两晚无报告**")

    # 附录：原始 LLM 输出（截断）
    if parsed.get("_raw_llm"):
        lines.extend(["", "## 附录：LLM 原始输出（解析失败）", "", "```", parsed["_raw_llm"][:1000], "```"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[seg3] 夜班报告 → {path}")
    return path


# ================================================================ Segment 4: 看门狗

def segment4_watchdog(st: dict, seg3_result: dict) -> dict:
    """看门狗：L3 git snapshot + expire proposals + finalize。"""
    write_heartbeat("segment4", "running", "watchdog: git + expire + finalize")
    print(f"[seg4] {now_cn()} 看门狗启动")

    # L3 git snapshot（沙盒模式 git 沙盒 memory，不碰正式）
    git_result = "skipped"
    try:
        _git_cwd = os.path.join(SANDBOX if SANDBOX else REPO, "memory")
        subprocess.run(
            ["git", "add", "-A"],
            cwd=_git_cwd,
            capture_output=True, timeout=10,
        )
        commit_msg = f"night-consolidation {today_cn()}: facts={seg3_result.get('new_facts', 0)}, proposals={seg3_result.get('proposals', 0)}"
        r = subprocess.run(
            ["git", "commit", "-m", commit_msg, "--allow-empty"],
            cwd=_git_cwd,
            capture_output=True, text=True, timeout=15,
        )
        git_result = "committed" if r.returncode == 0 else f"noop ({r.stdout.strip()[:80]})"
    except Exception as e:
        git_result = f"error: {e}"
    print(f"[seg4] L3 git: {git_result}")

    # expire proposals
    proposal_expired = 0
    try:
        sys.path.insert(0, os.path.join(REPO, "src"))
        from proposal_queue import ProposalQueue
        pq = ProposalQueue()
        proposal_expired = pq.expire_and_escalate()
    except Exception as e:
        print(f"[seg4] proposal expire error: {e}")
    print(f"[seg4] 过期提案: {proposal_expired}")

    # L1 30 天滚动清理
    cleanup_count = _l1_rolling_cleanup()
    print(f"[seg4] L1 清理: {cleanup_count} 条过期记录")

    # L2 分层遗忘（2026-08-21）：90 天未命中 → 陈旧降权（×0.3），30 天内命中复活。
    # L0 永存不删（append-only 档案）；L3 走人审提案治理。失败不阻塞。
    try:
        r = subprocess.run([L2_VENV, L2_PY, "decay"],
                           capture_output=True, text=True, timeout=60, cwd=REPO)
        print(f"[seg4] L2 遗忘扫描: {r.stdout.strip()[-80:]}")
    except Exception as e:
        print(f"[seg4] L2 遗忘扫描异常（不阻塞）: {e}")

    # 成功/失败判定
    if seg3_result.get("exit_code") == 0:
        st["consecutive_failures"] = 0
        st["red_alert"] = False
    else:
        st["consecutive_failures"] = st.get("consecutive_failures", 0) + 1
        if st["consecutive_failures"] >= 2:
            st["red_alert"] = True
            write_heartbeat("segment4", "red_alert", "连续两晚无报告！")

    write_heartbeat("segment4", "done", f"git={git_result}, expired={proposal_expired}, cleanup={cleanup_count}")
    print(f"[seg4] {now_cn()} 看门狗完成")
    return {"exit_code": 0, "git": git_result, "expired": proposal_expired, "cleanup": cleanup_count}


def _l1_rolling_cleanup() -> int:
    """L1 30 天滚动清理：删除 ingest_state.json 之外的过期文件。"""
    cutoff = time.time() - 30 * 86400
    count = 0
    if not os.path.isdir(L1_ROOT):
        return 0
    for fn in os.listdir(L1_ROOT):
        if fn.startswith(".") or fn == "ingest_state.json" or fn == "consolidation_state.json":
            continue
        path = os.path.join(L1_ROOT, fn)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                count += 1
        except OSError:
            continue
    return count


# ================================================================ 主流程

def run_pipeline(use_day_model: bool = False) -> int:
    """全量跑四段。"""
    st = load_state()
    st["_use_day"] = use_day_model
    st["segments"] = {}
    pipeline_start = time.time()
    print(f"=== 夜班管线启动 {now_cn()} ===")
    write_heartbeat("pipeline", "running", f"start {now_cn()}")

    exit_code = 0

    # Segment 1
    try:
        r = segment1_ingest()
        st["segments"]["segment1"] = {"status": "done", **r}
        save_state(st)
    except Exception as e:
        st["segments"]["segment1"] = {"status": "failed", "error": str(e)}
        save_state(st)
        print(f"[seg1] FAILED: {e}")
        exit_code = 1

    # Segment 2
    try:
        r = segment2_embed()
        st["segments"]["segment2"] = {"status": "done", **r}
        save_state(st)
    except Exception as e:
        st["segments"]["segment2"] = {"status": "failed", "error": str(e)}
        save_state(st)
        print(f"[seg2] FAILED: {e}")
        # 嵌入失败不阻塞巩固（L2 可能已有旧索引可用）
        st["segments"]["segment2"] = {"status": "degraded", "error": str(e)}
        save_state(st)

    # Segment 3（门控：seg1 必须 done）
    seg3_result = {"exit_code": -1}
    try:
        seg3_result = segment3_consolidate(st, use_day_model)
        st["segments"]["segment3"] = {"status": "done" if seg3_result.get("exit_code") == 0 else "failed", **seg3_result}
        save_state(st)
    except Exception as e:
        st["segments"]["segment3"] = {"status": "failed", "error": str(e)}
        save_state(st)
        import traceback as _tb
        print(f"[seg3] FAILED: {e}")
        print(_tb.format_exc())  # 2026-08-24：打全 traceback，杜绝「只知道报错不知道哪行」
        seg3_result = {"exit_code": -1, "error": str(e)}
        exit_code = 1

    # Segment 4
    try:
        r = segment4_watchdog(st, seg3_result)
        st["segments"]["segment4"] = {"status": "done", **r}
        save_state(st)
    except Exception as e:
        st["segments"]["segment4"] = {"status": "failed", "error": str(e)}
        save_state(st)
        print(f"[seg4] FAILED: {e}")
        exit_code = 1

    elapsed = int(time.time() - pipeline_start)
    st["last_run_elapsed"] = elapsed
    save_state(st)
    write_heartbeat("pipeline", "done" if exit_code == 0 else "failed", f"elapsed={elapsed}s")
    print(f"=== 管线完成 {now_cn()} 耗时 {elapsed}s exit={exit_code} ===")
    return exit_code


def show_status() -> None:
    st = load_state()
    print(f"=== 夜班管线状态 ===")
    print(f"  上次巩固: {st.get('last_consolidation', '从未')}")
    print(f"  L0 记录数: {st.get('last_l0_count', 0)}")
    print(f"  上次报告: {st.get('last_report_path', '无')}")
    print(f"  连续失败: {st.get('consecutive_failures', 0)}")
    print(f"  红点告警: {'是 ⚠' if st.get('red_alert') else '否'}")
    print(f"  各段状态:")
    for seg, info in st.get("segments", {}).items():
        status = info.get("status", "?")
        icon = {"done": "✓", "failed": "✗", "running": "⏳", "degraded": "⚠", "skipped": "⊘"}.get(status, "?")
        print(f"    {icon} {seg}: {status}")


def show_report() -> None:
    st = load_state()
    path = st.get("last_report_path")
    if not path:
        # 找最近的报告
        if os.path.isdir(REPORT_DIR):
            reports = sorted(os.listdir(REPORT_DIR), reverse=True)
            if reports:
                path = os.path.relpath(os.path.join(REPORT_DIR, reports[0]), REPO)
    if not path:
        print("无夜班报告")
        return
    full = os.path.join(REPO, path)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as f:
            print(f.read())
    else:
        print(f"报告文件不存在: {path}")


# ================================================================ CLI

def main():
    ap = argparse.ArgumentParser(description="M5 夜班巩固管线")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_run = sub.add_parser("run", help="全量跑四段")
    sp_run.add_argument("--use-day-model", action="store_true",
                        help="开发期用白天 27B 代替夜间 35B")

    sub.add_parser("segment1", help="只跑摄入段")
    sub.add_parser("segment2", help="只跑嵌入段")
    sp_s3 = sub.add_parser("segment3", help="只跑巩固段（调试）")
    sp_s3.add_argument("--use-day-model", action="store_true")
    sub.add_parser("segment4", help="只跑看门狗段")
    sub.add_parser("status", help="查管线状态")
    sub.add_parser("report", help="显示最近夜班报告")

    args = ap.parse_args()

    if args.cmd == "run":
        ec = run_pipeline(use_day_model=args.use_day_model)
        sys.exit(ec)
    elif args.cmd == "segment1":
        st = load_state()
        r = segment1_ingest()
        st["segments"] = st.get("segments", {})
        st["segments"]["segment1"] = {"status": "done", **r}
        save_state(st)
    elif args.cmd == "segment2":
        st = load_state()
        r = segment2_embed()
        st["segments"] = st.get("segments", {})
        st["segments"]["segment2"] = {"status": "done", **r}
        save_state(st)
    elif args.cmd == "segment3":
        st = load_state()
        st["_use_day"] = args.use_day_model  # 不用 state 里的残留值，以本次 CLI 为准
        r = segment3_consolidate(st, use_day_model=args.use_day_model)
        st["segments"] = st.get("segments", {})
        st["segments"]["segment3"] = {"status": "done" if r.get("exit_code") == 0 else "failed", **r}
        save_state(st)
    elif args.cmd == "segment4":
        st = load_state()
        segment4_watchdog(st, {"exit_code": 0})
    elif args.cmd == "status":
        show_status()
    elif args.cmd == "report":
        show_report()


if __name__ == "__main__":
    main()
