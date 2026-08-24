#!/usr/bin/env python3
"""M6 本地 AI dashboard（设计 §16，2026-08-18 用户定：首屏＝夜班报告）。

内容主权自有：直接读 exchange/ + memory/ 状态文件渲染，不依赖 dsh 运行。
渲染底座借社区：agent 对话走 dsh web UI（:3080，另行启动），本 dashboard 只管状态面。

首屏四区：
  1. 夜班报告（最新 night-*.md：changelog + 今日应在意的学校/微信优先级清单）
  2. urgent 告警（exchange/shared/alerts/，M8 高频扫描产出 → 红点）
  3. 待审批提案（exchange/proposals/pending/，批准/否决按钮 —— 写回安全边界 §10：
     只有用户点按钮才动提案状态，AI 推断永不写回）
  4. 系统状态（heartbeat.json + 连续失败/红点告警 + 睡眠态）

用法：
  python3 m6_dashboard.py [--port 3091]
然后浏览器开 http://127.0.0.1:3091（仅监听本机，绝不出境）。
常驻：launchd plist = dsh/com.local-ai-agent.dashboard.plist（KeepAlive）。
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import threading
import urllib.parse
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCHANGE = os.path.join(REPO, "exchange")
REPORT_DIR = os.path.join(EXCHANGE, "outbox", "reports")
ALERTS_DIR = os.path.join(EXCHANGE, "shared", "alerts")
HEARTBEAT_PATH = os.path.join(EXCHANGE, "shared", "heartbeat.json")
PROPOSALS_DIR = os.path.join(EXCHANGE, "proposals")
L3_PATH = os.path.join(REPO, "memory", "L3_core", "core.md")
CONSOLIDATION_STATE = os.path.join(REPO, "memory", "L1_working", "consolidation_state.json")
L0_ROOT = os.path.join(REPO, "memory", "L0_raw")

# #17 内置 chat：直连白天 27B（mlx_lm OpenAI 兼容服务）
DAY_MODEL_URL = "http://127.0.0.1:8100/v1/chat/completions"
DAY_MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"

# M7 Megumin persona（草案 §5 Q34）：惠惠实例 :8101（serve_day.sh persona 挂 500 checkpoint），
# persona 对话 mode=persona 隔离到 L0_raw/persona/ 子树，永不进事实记忆/语义索引。
PERSONA_MODEL_URL = "http://127.0.0.1:8101/v1/chat/completions"
PERSONA_SYSTEM = ("你是惠惠（Megumin），红魔族天才魔法师，傲娇、毒舌、自恋、中二，但内心善良。"
                  "口头禅是「爆裂魔法」。你说话直接、有点小暴躁，但其实很在意同伴。"
                  "自称「本小姐」，称呼同伴时常用名字。")
PERSONA_SERVE = os.path.join(REPO, "src", "serve_day.sh")
PERSONA_LOG = os.path.join(REPO, "models", "megumin-lora", "logs", "serve-persona.log")

# 抓取触发并发锁（§13：抓取器挂靠本地 AI，dashboard 是它的触发入口之一）
_scrape_lock = threading.Lock()

TZ_CN = timezone(timedelta(hours=8))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def now_cn() -> str:
    return datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S")


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------- 数据读取

def latest_report() -> tuple[str, str]:
    """返回 (文件名, 渲染后的 HTML 片段)。无报告返回提示。"""
    if not os.path.isdir(REPORT_DIR):
        return "", "（暂无夜班报告）"
    reports = sorted((f for f in os.listdir(REPORT_DIR) if f.endswith(".md")), reverse=True)
    if not reports:
        return "", "（暂无夜班报告）"
    with open(os.path.join(REPORT_DIR, reports[0]), encoding="utf-8") as f:
        return reports[0], md_to_html(f.read())


def md_to_html(md: str) -> str:
    """极简 md→html：标题/列表/粗体/斜体/代码/引用。够夜班报告用。"""
    out = []
    in_code = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append("<pre>" if in_code else "</pre>")
            continue
        if in_code:
            out.append(esc(line))
            continue
        line_esc = esc(line)
        line_esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line_esc)
        line_esc = re.sub(r"`(.+?)`", r"<code>\1</code>", line_esc)
        line_esc = re.sub(r"_(.+?)_", r"<em>\1</em>", line_esc)
        if line.startswith("# "):
            out.append(f"<h2>{line_esc[2:]}</h2>")
        elif line.startswith("## "):
            out.append(f"<h3>{line_esc[3:]}</h3>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{line_esc[2:]}</blockquote>")
        elif line.startswith("- "):
            out.append(f"<li>{line_esc[2:]}</li>")
        elif line.strip():
            out.append(f"<p>{line_esc}</p>")
    return "\n".join(out)


def list_alerts() -> list[dict]:
    if not os.path.isdir(ALERTS_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(ALERTS_DIR), reverse=True):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(ALERTS_DIR, fn), encoding="utf-8") as f:
                a = json.load(f)
            if a.get("status") == "new":
                out.append(a)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def list_pending_proposals() -> list[dict]:
    d = os.path.join(PROPOSALS_DIR, "pending")
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def system_status() -> dict:
    hb = {}
    if os.path.exists(HEARTBEAT_PATH):
        try:
            with open(HEARTBEAT_PATH, encoding="utf-8") as f:
                hb = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    st = {}
    if os.path.exists(CONSOLIDATION_STATE):
        try:
            with open(CONSOLIDATION_STATE, encoding="utf-8") as f:
                st = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "heartbeat": hb,
        "consecutive_failures": st.get("consecutive_failures", 0),
        "red_alert": st.get("red_alert", False),
        "last_consolidation": st.get("last_consolidation"),
    }


def sleep_state() -> str:
    """睡眠态：当前时间在 03:00–09:00 之外 = 醒着；之内 = 睡眠/巩固窗口。"""
    h = datetime.now(TZ_CN).hour
    return "😴 睡眠窗口（夜班管线 03:00–08:00）" if 3 <= h < 9 else "☀️ 清醒（交互模式）"


# ---------------------------------------------------------------- #17 内置 chat（白天 27B 直连 + 对话落 L0）

# 记忆注入（2026-08-19 #18）：L3 常驻 + L2 按 query 检索，白天对话不再是"失忆"状态。
# 设计依据 §3：L3 = 常驻上下文（~2000 token 预算），core.md 3.5KB ≈ 1.2K tokens 直接全塞。
L2_VENV = "~/.workbuddy/binaries/python/envs/llama-cpp/bin/python"
L2_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "l2_semantic.py")


def _load_l3_context() -> str:
    """读 L3 核心记忆（best-effort，文件缺失/异常返回空串）。"""
    try:
        if os.path.exists(L3_PATH):
            with open(L3_PATH, encoding="utf-8") as f:
                return f.read()[:4000]
    except Exception:
        pass
    return ""


def _l2_search_context(query: str) -> str:
    """按用户问题跑 L2 语义检索（bge-m3 + BM25 混合），best-effort 30s 超时。
    失败静默降级为仅 L3 —— 检索只是增强，绝不能卡死对话。"""
    if not query.strip():
        return ""
    try:
        import subprocess
        r = subprocess.run(
            [L2_VENV, L2_PY, "search", query.strip()[:120], "-k", "3"],
            capture_output=True, text=True, timeout=30, cwd=REPO,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()[:1500]
    except Exception:
        pass
    return ""


def _l1_context() -> str:
    """读 L-1 瞬时层（exchange/.daytime/）最新微信/邮件动态（best-effort，截断）。
    L-1 每 30 分钟由 daytime_sync.py 同步（用户 2026-08-21 需求），供白天模型实时参考。"""
    parts: list[str] = []
    try:
        d = os.path.join(EXCHANGE, ".daytime")
        if not os.path.isdir(d):
            return ""
        # 最新邮件摘要：只取标题列表（正文太长，且晚上海报会覆盖）
        for fn in sorted(os.listdir(d), reverse=True):
            if fn.startswith("邮箱摘要") and fn.endswith(".md"):
                titles = [l.strip() for l in open(os.path.join(d, fn), encoding="utf-8")
                          if l.startswith("[")][:25]
                if titles:
                    parts.append("## 邮件（L-1 实时，最新标题）\n" + "\n".join(titles))
                break
        # 微信最新动态
        for fn in sorted(os.listdir(d), reverse=True):
            if fn.startswith("wechat-") and fn.endswith(".md"):
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    parts.append("## 微信最新动态（L-1 实时）\n" + f.read()[:1500])
                break
    except Exception:
        pass
    return "\n\n".join(parts)[:3500]


def _build_chat_system(query: str) -> str:
    """白天对话 system prompt：身份 + L3 常驻记忆 + L2 检索上下文 + L-1 实时动态。"""
    parts = [
        "你是本地 AI 代理（白天 Qwen3.8-27B，全本地运行，绝不出境）。",
        "基于下方记忆回答；记忆里有的信息直接引用，记忆里没有的如实说不知道，绝不编造。",
        "涉及用户个人 / 学校 / 微信 / 邮件信息时一律以记忆为准。",
        f"当前时间：{now_cn()}（中国标准时间 UTC+8）。",
    ]
    l3 = _load_l3_context()
    if l3:
        parts.append("\n## L3 核心记忆（常驻，权威）\n" + l3)
    l2 = _l2_search_context(query)
    if l2:
        parts.append("\n## L2 语义检索（按你刚才的问题查的）\n" + l2)
    l1 = _l1_context()
    if l1:
        parts.append("\n## L-1 最新动态（近 30 分钟实时同步，瞬时参考）\n" + l1)
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 工具调用（§13 本地 AI 主动触发抓取）

# 用户消息含这些词 → 启用工具模式（27B 可主动请求 email_scrape 触发抓取）
EMAIL_HINT_RE = re.compile(r"邮箱|邮件|抓取|收件箱|来信|信件|gmail|outlook|email|mail", re.IGNORECASE)

# 2026-08-23：问密码/账号/凭据 → 工具模式（vault_list/vault_get）
VAULT_HINT_RE = re.compile(r"密码|凭据|账号密码|登录信息|cred:|vault|口令", re.IGNORECASE)

TOOL_EMAIL_SCRAPE = {
    "type": "function",
    "function": {
        "name": "email_scrape",
        "description": "抓取本地邮箱（Gmail UCSB+个人）生成今日摘要到 exchange/.daytime/ "
                       "（L-1 瞬时记忆，不进正式记忆，隔夜即清）。用户问邮箱内容/邮件/抓取/收件箱时调用。",
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取本地 exchange/ 目录下的文件内容（如邮箱摘要），返回前 8000 字。",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "相对 exchange/ 的文件路径，如 inbox/email/邮箱摘要-2026-08-20.md"}},
            "required": ["file_path"]},
    },
}

# 2026-08-23：vault 密钥工具（用户要求「本地 AI 可以告知我的密码」）。
# 安全：vault_get 返回的密码用 ⟦secret⟧ 标记包裹，log_chat_to_l0 对标记区间打码，
# 密码绝不进 L0/L3/记忆；只在本地对话里展示给用户本人。
VAULT_PY = ["~/.workbuddy/binaries/python/versions/3.13.12/bin/python3",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault.py")]
SECRET_OPEN, SECRET_CLOSE = "⟦secret⟧", "⟦/secret⟧"

# 2026-08-23：vault_get 工具执行过 → 本次对话含密码，落 L0 时整段打码不记录。
_vault_touched = False

TOOL_VAULT_LIST = {
    "type": "function",
    "function": {
        "name": "vault_list",
        "description": "列出本地密钥库（vault）中保存的密码条目（站点名，如 cred:collegeboard）。"
                       "用户问密码/账号/登录信息/凭据时，先调用它找到对应条目名。",
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_VAULT_GET = {
    "type": "function",
    "function": {
        "name": "vault_get",
        "description": "从本地密钥库取出某条凭据（cred:站点名 或 wechat_db_passphrase）。"
                       "返回 {\"url\",\"username\",\"password\"}。拿到后直接告诉用户账号密码。"
                       "重要：不要把密码写进任何摘要/记录，只口头告知。",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "凭据键名，如 cred:prod.idp.collegeboard.org"}},
            "required": ["key"]},
    },
}

# 2026-08-24：对话存密码（vault_set）。仅允许 cred:*（用户凭据），
# 拒绝系统键（wechat_db_passphrase / edge:*）——系统键只读防误覆盖。
TOOL_VAULT_SET = {
    "type": "function",
    "function": {
        "name": "vault_set",
        "description": "把用户提供的新密码/凭据存入本地密钥库（Keychain）。"
                       "key 必须是 cred:站点名（如 cred:epfl），value 是密码或 JSON"
                       "{\"url\",\"username\",\"password\"}。存完告诉用户已保存。"
                       "重要：不要把密码写进任何摘要/记录。",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "凭据键名，如 cred:epfl（必须 cred: 开头）"},
            "value": {"type": "string", "description": "密码明文，或 JSON {\"url\":\"\",\"username\":\"\",\"password\":\"\"}"}},
            "required": ["key", "value"]},
    },
}


def _exec_tool(name: str, args_str: str) -> str:
    """执行工具，返回结果文本。安全：read_file 仅允许 exchange/ 内文件。"""
    global _vault_touched
    if name == "email_scrape":
        return _run_email_scrape()
    if name == "read_file":
        try:
            fp = json.loads(args_str or "{}").get("file_path", "")
        except Exception:
            return "参数解析失败"
        abs_fp = os.path.abspath(os.path.join(EXCHANGE, str(fp)))
        if not abs_fp.startswith(os.path.abspath(EXCHANGE) + os.sep) or not os.path.isfile(abs_fp):
            return "路径不合法（仅允许 exchange/ 内文件）"
        try:
            with open(abs_fp, encoding="utf-8") as f:
                return f.read()[:8000]
        except Exception as e:
            return f"读取失败: {e}"
    if name == "vault_list":
        try:
            r = subprocess.run([*VAULT_PY, "list"], capture_output=True, text=True, timeout=30)
            keys = [k for k in (r.stdout or "").splitlines() if k.startswith("cred:")]
            return "vault 密码条目:\n" + "\n".join(keys) if keys else "vault 暂无密码条目"
        except Exception as e:
            return f"vault list 失败: {e}"
    if name == "vault_get":
        try:
            key = json.loads(args_str or "{}").get("key", "")
        except Exception:
            return "参数解析失败"
        if not key.startswith("cred:") and key != "wechat_db_passphrase":
            return "仅允许取 cred:* 或 wechat_db_passphrase"
        try:
            r = subprocess.run([*VAULT_PY, "get", key], capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                return f"未找到: {key}"
            _vault_touched = True
            return f"{SECRET_OPEN}{r.stdout.strip()}{SECRET_CLOSE}"
        except Exception as e:
            return f"vault get 失败: {e}"
    if name == "vault_set":
        try:
            a = json.loads(args_str or "{}")
            key = str(a.get("key", "")).strip()
            value = str(a.get("value", "")).strip()
        except Exception:
            return "参数解析失败"
        if not key.startswith("cred:"):
            return "仅允许存 cred:*（系统键只读，防误覆盖）"
        if not value:
            return "密码值不能为空"
        try:
            r = subprocess.run([*VAULT_PY, "set", key], input=value, text=True,
                               capture_output=True, timeout=30)
            if r.returncode != 0:
                return f"vault set 失败: {(r.stderr or '')[:100]}"
            _vault_touched = True
            return f"已保存 {key} 到本地密钥库（Keychain）"
        except Exception as e:
            return f"vault set 异常: {e}"
    return f"未知工具: {name}"


def _run_email_scrape() -> str:
    """执行邮箱抓取（复用 /scrape/email 逻辑），返回「抓取结果 + 邮件标题列表」供 27B 引用。
    附标题列表是为了让 27B 拿到足够信息直接总结，避免它再请求读文件（工具循环只实现了本工具）。
    2026-08-20：写 exchange/.daytime/（L-1 瞬时记忆）——白天触发的原始抓取不进正式目录、
    不被 scan 摄入、隔夜即清，与夜班必抓零冲突；记忆只进 chat 提炼结果（log_chat_to_l0）。"""
    import subprocess
    if not _scrape_lock.acquire(blocking=False):
        return "抓取正在进行中，请稍后再问。"
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "src", "scrape.py"), "email", "--manual"],
            capture_output=True, text=True, timeout=240, cwd=REPO)
        if r.returncode != 0:
            return f"抓取失败: {(r.stderr or r.stdout or '')[-200:]}"
        out_file = os.path.join(EXCHANGE, ".daytime",
                                f"邮箱摘要-{datetime.now(TZ_CN):%Y-%m-%d}.md")
        titles = []
        if os.path.exists(out_file):
            for line in open(out_file, encoding="utf-8"):
                if line.startswith("["):
                    titles.append(line.strip())
        return (f"抓取完成 → {os.path.relpath(out_file, REPO)}（L-1 瞬时，共 {len(titles)} 封邮件）\n"
                f"邮件标题列表（前 60 条）：\n" + "\n".join(titles[:60]))
    except Exception as e:
        return f"抓取异常: {str(e)[:150]}"
    finally:
        _scrape_lock.release()


def _chat_with_tools(history: list[dict], stream: bool):
    """工具模式：带 email_scrape/read_file 工具做多轮循环（最多 4 轮）。
    27B 可主动请求抓取邮箱、读摘要文件，执行后回填再继续，直到给出最终回答。
    返回：字符串（stream=False）或一次性生成器（stream=True，最终回答整段 yield）。"""
    import urllib.request
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
    query = msgs[-1]["content"] if msgs else ""
    msgs.insert(0, {"role": "system", "content": _build_chat_system(query)})

    def _call(payload_msgs, tools=None, stream_flag=False):
        body = {"model": DAY_MODEL_ID, "messages": payload_msgs,
                "max_tokens": 8192, "temperature": 0.7}
        if tools:
            body["tools"] = tools
        if stream_flag:
            body["stream"] = True
        req = urllib.request.Request(DAY_MODEL_URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req, timeout=300)

    TOOLS = [TOOL_EMAIL_SCRAPE, TOOL_READ_FILE, TOOL_VAULT_LIST, TOOL_VAULT_GET, TOOL_VAULT_SET]
    for _rnd in range(4):
        try:
            resp = json.loads(_call(msgs, tools=TOOLS).read())
        except Exception as e:
            return f"（工具决策调用失败: {str(e)[:100]}）"
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            content = msg.get("content") or ""
            if not stream:
                return content
            def _once():
                yield content
            return _once()
        # 执行本轮所有工具调用并回填
        msgs.append({"role": "assistant", "content": msg.get("content") or None,
                     "tool_calls": tool_calls})
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            args = tc.get("function", {}).get("arguments", "")
            result = _exec_tool(name, args)
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
    # 2026-08-23 改：轮数耗尽时强制 27B 基于已收集工具结果给最终回答（不带工具），
    # 仍给不出来再回退到友好提示。
    try:
        forced = urllib.request.Request(
            DAY_MODEL_URL,
            data=json.dumps({"model": DAY_MODEL_ID, "messages": msgs,
                              "stream": False, "max_tokens": 1024,
                              "tool_choice": "none"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        r = json.loads(urllib.request.urlopen(forced, timeout=120).read())
        c = r["choices"][0]["message"].get("content") or ""
        if c.strip():
            return c.strip()
    except Exception:
        pass
    return ("（工具循环已达上限且模型未给出总结。以下是已查到的信息，请结合判断）\n"
            "如需继续查请直接告诉我更具体的关键词或时间段。")


def chat_with_day_model(history: list[dict], stream: bool = False):
    """转发到白天 27B（:8100）。history=[{role,content}...]。
    记忆注入：system prompt = 身份 + L3 core.md 全文 + 按最后一条用户消息的 L2 检索。
    2026-08-20：服务端已 enable_thinking=false（serve_day.sh），不再加 /no_think；
    用户消息含邮箱/邮件/抓取关键词 → 工具模式（27B 可主动请求 email_scrape 触发抓取）；
    stream=True 返回生成器（SSE delta 文本），False 返回完整字符串。max_tokens 8192。"""
    if history and history[-1].get("role") == "user" and (
            EMAIL_HINT_RE.search(history[-1]["content"])
            or VAULT_HINT_RE.search(history[-1]["content"])):
        return _chat_with_tools(history, stream)
    import urllib.request
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
    if msgs and msgs[-1]["role"] == "user":
        query = msgs[-1]["content"]
    else:
        query = ""
    system = _build_chat_system(query)
    msgs.insert(0, {"role": "system", "content": system})
    payload = json.dumps({
        "model": DAY_MODEL_ID, "messages": msgs,
        "max_tokens": 8192, "temperature": 0.7,
        "stream": stream,
    }).encode()
    req = urllib.request.Request(
        DAY_MODEL_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=300)

    if not stream:
        data = json.loads(resp.read())
        content = data["choices"][0]["message"].get("content") or ""
        # 剥 thinking 块（服务端已关，防御性保留）
        sys.path.insert(0, os.path.join(REPO, "src"))
        from night_pipeline import _strip_think
        return _strip_think(content)

    def _gen():
        for line in resp:
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
                delta = j["choices"][0]["delta"].get("content") or ""
            except Exception:
                delta = ""
            if delta:
                yield delta

    return _gen()


def _ensure_persona() -> bool:
    """确保惠惠实例 :8101 在跑（serve_day.sh persona，按需拉起）。返回是否就绪。"""
    import subprocess
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8101/v1/models", timeout=3)
        return True
    except Exception:
        pass
    os.makedirs(os.path.dirname(PERSONA_LOG), exist_ok=True)
    logf = open(PERSONA_LOG, "ab")
    try:
        proc = subprocess.Popen(["bash", PERSONA_SERVE, "persona", "8101"],
                                stdout=logf, stderr=logf, cwd=REPO,
                                start_new_session=True)
    except Exception as e:
        print(f"[dashboard] persona 启动失败: {e}")
        return False
    import time
    for _ in range(40):  # 最多 ~120s
        time.sleep(3)
        try:
            urllib.request.urlopen("http://127.0.0.1:8101/v1/models", timeout=3)
            return True
        except Exception:
            if proc.poll() is not None:
                print(f"[dashboard] persona 进程早退 exit={proc.returncode}")
                return False
    return False


def _build_persona_system(query: str) -> str:
    """惠惠 persona system：人设 + 只读记忆（L3 核心 + 按 query 的 L2 检索）。
    只读边界：记忆仅供了解/聊天参考，persona 对话绝不写事实记忆（草案 §5 Q34）。"""
    parts = [PERSONA_SYSTEM]
    l3 = _load_l3_context()
    if l3:
        parts.append("\n## 关于你的伙伴（只读参考，来自本地记忆）\n" + l3)
    l2 = _l2_search_context(query)
    if l2:
        parts.append("\n## 相关记忆检索（只读参考）\n" + l2)
    l1 = _l1_context()
    if l1:
        parts.append("\n## 最新动态（L-1 实时同步，只读参考）\n" + l1)
    parts.append("\n（以上记忆仅供你了解情况、陪他聊天时参考，属只读：不要修改、不要声称写入，"
                 "你的对话本身是 persona 隔离，不会进入任何记忆。）")
    return "\n\n".join(parts)


def chat_with_persona(history: list[dict], stream: bool = False):
    """惠惠 persona 对话：直连 :8101（挂 Megumin 500 checkpoint adapter）。
    只读记忆注入（L3 + L2 检索），对话仍隔离落 persona/ 子树（草案 §5 Q34）。"""
    import urllib.request
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
    query = msgs[-1]["content"] if msgs else ""
    msgs.insert(0, {"role": "system", "content": _build_persona_system(query)})
    body = {"model": DAY_MODEL_ID, "messages": msgs, "max_tokens": 8192, "temperature": 0.7}
    if stream:
        body["stream"] = True
    req = urllib.request.Request(PERSONA_MODEL_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=300)
    if not stream:
        data = json.loads(resp.read())
        return data["choices"][0]["message"].get("content") or ""

    def _gen():
        for line in resp:
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
            except Exception:
                delta = ""
            if delta:
                yield delta
    return _gen()


def log_chat_to_l0(user_text: str, assistant_text: str, mode: str = "normal") -> None:
    """dashboard 对话落 L0（source=chat，与 #16 dsh 桥同一个 source 文件，巩固统一可见）。
    mode=persona → 隔离到 L0_raw/persona/ 子树（草案 §5 Q34，惠惠对话永不进事实记忆）。
    2026-08-23：vault_get 用过的对话（_vault_touched）整段打码不落记忆；
    ⟦secret⟧...⟦/secret⟧ 区间（工具结果）也打码。密码绝不进 L0/L3。"""
    global _vault_touched
    if _vault_touched:
        assistant_text = "[密码对话：内容已告知用户，防泄漏不记录]"
        _vault_touched = False
    elif SECRET_OPEN in assistant_text:
        import re as _re
        assistant_text = _re.sub(
            f"{_re.escape(SECRET_OPEN)}.*?{_re.escape(SECRET_CLOSE)}",
            "[密码已告知用户，防泄漏不记录]", assistant_text, flags=_re.S)
    sys.path.insert(0, os.path.join(REPO, "src"))
    from l0_ingest import L0Writer
    L0Writer(L0_ROOT).append("chat", {
        "session": "dashboard",
        "title": "dashboard 内置对话",
        "messages": [
            {"role": "user", "text": user_text, "ts": datetime.now(TZ_CN).timestamp()},
            {"role": "assistant", "text": assistant_text, "ts": datetime.now(TZ_CN).timestamp()},
        ],
        "turns": 1,
    }, mode=mode, sensitive=True, meta={"ingest": "dashboard_chat"})


# ---------------------------------------------------------------- 提案动作（§10 写回安全边界：仅按钮触发）

def proposal_action(pid: str, action: str) -> tuple[bool, int]:
    """执行提案动作。批准/否决都是用户显式动作 → 记 audit；批准后立即跑 L3 写回执行器。
    返回 (成功?, 本次执行/应用的 L3 modify 条数)。"""
    from proposal_queue import ProposalQueue
    from writeback import apply_approved, log_audit
    pq = ProposalQueue(PROPOSALS_DIR)
    if action == "approve":
        ok = pq.approve(pid, decided_by="dashboard-button")
        if ok:
            log_audit({"action": "proposal.approve", "actor": "dashboard-button",
                       "target": pid, "outcome": "approved"})
            try:
                applied = apply_approved()  # §10：点头即授权，立即应用已批准的 L3 修改
                n = sum(1 for r in applied if r["ok"])
            except Exception as e:
                print(f"[dashboard] writeback 执行异常: {e}")
                n = 0
            return ok, n
        return False, 0
    if action == "reject":
        ok = pq.reject(pid, reason="dashboard 否决", decided_by="dashboard-button")
        if ok:
            log_audit({"action": "proposal.reject", "actor": "dashboard-button",
                       "target": pid, "outcome": "rejected"})
        return ok, 0
    return False, 0


# ---------------------------------------------------------------- 页面渲染

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>本地 AI Dashboard</title>
<style>
/* 鲸吟（Whale Song）配色：冰蓝→钴蓝冷色系，与 dsh 鲸鱼娘同风格 */
/* --opa：全局不透明度变量（0=全透 ~ 1=不透），由滑杆实时驱动，localStorage 持久 */
:root {{ --bg:#0a1420; --panel:#10202f; --border:#1e3a52; --fg:#dceef9; --dim:#7fa3bd;
  --red:#ff6b6b; --yellow:#ffd166; --green:#4ecdc4; --accent:#4fb3ff;
  --opa:0.45; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:-apple-system,"SF Pro SC","PingFang SC",sans-serif; }}
/* 背景轮播层：三张鲸鱼娘立绘，5 分钟淡入淡出切换（对齐 5min 自动刷新） */
#bglayer {{ position:fixed; inset:0; z-index:-2;
  background-repeat:no-repeat; background-position:right top;
  background-size:auto 100vh; background-attachment:fixed;
  transition:opacity 1.2s ease-in-out; }}
body::before {{ content:""; position:fixed; inset:0; z-index:-1;
  /* 左→右渐变压暗层，深度随 --opa 缩放 */
  background:linear-gradient(90deg,
    rgba(10,20,32,var(--shade1,.85)) 0%, rgba(10,20,32,var(--shade2,.72)) 40%,
    rgba(10,20,32,var(--shade3,.45)) 68%, rgba(10,20,32,var(--shade4,.15)) 100%);
  pointer-events:none; }}
header {{ padding:14px 20px; border-bottom:1px solid var(--border);
  display:flex; justify-content:space-between; align-items:center;
  background:linear-gradient(135deg,rgba(13,31,51,var(--opa)) 0%,rgba(18,57,92,var(--opa)) 100%);
  backdrop-filter:blur(5px); }}
header h1 {{ font-size:16px; margin:0; }}
.state {{ font-size:13px; color:var(--dim); }}
.grid {{ display:grid; grid-template-columns:2fr 1fr; gap:14px; padding:14px 20px; }}
@media (max-width:700px) {{ .grid {{ grid-template-columns:1fr; padding:10px 12px; }} }}
/* 面板不透明度由 --opa 驱动 + 毛玻璃，透出背景鲸鱼娘 */
.panel {{ background:rgba(16,32,47,var(--opa)); border:1px solid rgba(30,58,82,.5);
  border-radius:10px; padding:14px 16px; backdrop-filter:blur(6px); }}
.panel h2 {{ font-size:13px; color:var(--dim); margin:0 0 10px;
  text-transform:uppercase; letter-spacing:.05em; }}
.badge {{ display:inline-block; padding:1px 8px; border-radius:8px; font-size:12px; }}
.badge.red {{ background:rgba(255,107,107,.15); color:var(--red); }}
.badge.green {{ background:rgba(78,205,196,.15); color:var(--green); }}
.alert {{ border-left:3px solid var(--red); padding:6px 10px; margin:8px 0;
  background:rgba(255,107,107,.07); border-radius:0 6px 6px 0; font-size:13px; }}
.prop {{ border:1px solid rgba(30,58,82,.5); border-radius:8px; padding:10px;
  margin:8px 0; font-size:13px; background:rgba(10,20,32,calc(var(--opa)*0.4)); }}
.prop .title {{ font-weight:600; margin-bottom:4px; }}
.prop .desc {{ color:var(--dim); white-space:pre-wrap; font-size:12px; margin:6px 0; }}
button {{ border:0; border-radius:6px; padding:4px 12px; font-size:12px;
  cursor:pointer; margin-right:6px; }}
button.approve {{ background:var(--green); color:#062a22; }}
button.reject {{ background:var(--red); color:#2b0705; }}
.report {{ font-size:14px; line-height:1.6; }}
.report h2 {{ font-size:18px; }} .report h3 {{ font-size:15px; margin:14px 0 6px; }}
.report blockquote {{ color:var(--dim); border-left:3px solid var(--border);
  margin:4px 0; padding-left:10px; }}
.report code {{ background:#16283a; padding:1px 5px; border-radius:4px; font-size:12px; }}
.report li {{ margin:3px 0; }}
.kv {{ font-size:13px; color:var(--dim); line-height:1.8; }}
.kv b {{ color:var(--fg); }}
a {{ color:var(--accent); }}
footer {{ padding:10px 20px; color:var(--dim); font-size:12px; }}
.whale {{ position:fixed; right:14px; bottom:44px; width:120px; z-index:999;
  filter:drop-shadow(0 4px 14px rgba(79,179,255,.4));
  animation:bob 3.2s ease-in-out infinite; cursor:pointer;
  transition:transform .2s; }}
.whale:hover {{ transform:scale(1.08); }}
@keyframes bob {{ 0%,100% {{ transform:translateY(0) }} 50% {{ transform:translateY(-7px) }} }}
@media (max-width:700px) {{ .whale {{ width:84px; bottom:40px; right:8px; }} }}
</style></head><body>
<!-- 背景轮播层：三张鲸鱼娘立绘 5 分钟轮换（与 meta refresh 同周期） -->
<div id="bglayer"></div>
<script>
(function() {{
  const bgs = [
    '/assets/whale/whale-bg-long.jpg',
    '/assets/whale/whale-bg-2.jpg',
    '/assets/whale/whale-bg-3.jpg',
  ];
  const layer = document.getElementById('bglayer');
  let bi = Math.floor(Date.now() / 300000) % bgs.length;
  // 预载
  bgs.forEach(u => {{ const i = new Image(); i.src = u; }});
  layer.style.backgroundImage = 'url(' + bgs[bi] + ')';
  setInterval(() => {{
    bi = (bi + 1) % bgs.length;
    layer.style.opacity = '0';
    setTimeout(() => {{
      layer.style.backgroundImage = 'url(' + bgs[bi] + ')';
      layer.style.opacity = '1';
    }}, 1200);
  }}, 300000);
}})();
</script>
<!-- 鲸鱼娘（@linxin666/dsh-pet，Apache-2.0）：点击切换动作 -->
<img class="whale" id="whalePet" src="/assets/whale/idle.gif" alt="鲸鱼娘"
     title="点我玩～">
<script>
(function() {{
  const acts = ['idle','waving','jumping','waiting','review','running'];
  let cur = 0, timer = null;
  const img = document.getElementById('whalePet');
  img.addEventListener('click', () => {{
    cur = (cur + 1) % acts.length;
    img.src = '/assets/whale/' + acts[cur] + '.gif';
    clearTimeout(timer);
    timer = setTimeout(() => {{ img.src = '/assets/whale/idle.gif'; cur = 0; }}, 4000);
  }});
}})();
</script>
<header>
  <h1>🐳 本地 AI Dashboard</h1>
  <span class="state">
    <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer">
      🎚 <input type="range" id="opaSlider" min="10" max="95" value="45"
        style="width:90px;accent-color:#4fb3ff">
      <span id="opaVal">45%</span>
    </label>
    <span id="headInfo">· {sleep} · {now}</span>
  </span>
</header>
<script>
(function() {{
  const root = document.documentElement;
  const slider = document.getElementById('opaSlider');
  const label = document.getElementById('opaVal');
  function apply(v) {{  // v = 10..95（百分比）
    const opa = v / 100;
    root.style.setProperty('--opa', opa.toFixed(2));
    // 压暗层四档深度 = 基础深度 × 不透明度比例（越不透越压暗）
    root.style.setProperty('--shade1', (0.95 * opa + 0.10).toFixed(2));
    root.style.setProperty('--shade2', (0.80 * opa + 0.08).toFixed(2));
    root.style.setProperty('--shade3', (0.55 * opa + 0.05).toFixed(2));
    root.style.setProperty('--shade4', (0.25 * opa + 0.02).toFixed(2));
    label.textContent = v + '%';
    localStorage.setItem('dash-opa', v);
  }}
  slider.value = localStorage.getItem('dash-opa') || 45;
  apply(parseInt(slider.value));
  slider.addEventListener('input', () => apply(parseInt(slider.value)));
}})();
</script>
<div class="grid">
  <div class="panel report" id="reportPanel">
    <h2>夜班报告 {report_name}</h2>
    {report}
  </div>
  <div>
    <div class="panel" id="alertPanel">
      <h2>⚠ Urgent 告警 {alert_badge}</h2>
      {alerts}
    </div>
    <div class="panel" style="margin-top:14px" id="propPanel">
      <h2>📋 待审批提案 {prop_badge}</h2>
      {proposals}
    </div>
    <div class="panel" style="margin-top:14px">
      <h2>🩺 系统状态</h2>
      <div class="kv" id="statusPanel">{status}</div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h2>💬 本地模型对话（27B）</h2>
      <div id="chatLog" style="max-height:260px;overflow-y:auto;font-size:13px;
        display:flex;flex-direction:column;gap:8px;margin-bottom:10px"></div>
      <div style="display:flex;gap:6px">
        <input id="chatInput" placeholder="和本地 27B 聊聊…（对话会进记忆）"
          style="flex:1;background:rgba(10,20,32,.5);border:1px solid var(--border);
          border-radius:6px;padding:6px 10px;color:var(--fg);font-size:13px;outline:none">
        <button onclick="sendChat()" style="background:var(--accent);color:#06223a;
          border:0;border-radius:6px;padding:6px 14px;cursor:pointer;font-size:13px">发</button>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:8px">
        <button onclick="triggerScrape()" style="background:rgba(79,179,255,.15);
          color:var(--accent);border:1px solid rgba(79,179,255,.4);border-radius:6px;
          padding:4px 12px;cursor:pointer;font-size:12px">📮 抓取邮箱</button>
        <span id="scrapeResult" style="font-size:12px;color:var(--dim)"></span>
        <label style="font-size:12px;color:var(--dim);display:flex;align-items:center;gap:4px;cursor:pointer;margin-left:auto">
          <input type="checkbox" id="personaMode" onchange="switchMode()"> 🧙 惠惠模式（persona 隔离，不进记忆）
        </label>
      </div>
    </div>
  </div>
</div>
<script>
const chatHistory = [];
function chatBubble(role, text) {{
  const log = document.getElementById('chatLog');
  const d = document.createElement('div');
  d.style.cssText = role === 'user'
    ? 'align-self:flex-end;background:rgba(79,179,255,.18);border-radius:8px;padding:6px 10px;max-width:92%'
    : 'align-self:flex-start;background:rgba(16,32,47,.5);border:1px solid rgba(30,58,82,.5);border-radius:8px;padding:6px 10px;max-width:92%';
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}}
function switchMode() {{
  const on = document.getElementById('personaMode').checked;
  chatHistory.length = 0;
  document.getElementById('chatLog').innerHTML = '';
  const inp = document.getElementById('chatInput');
  inp.placeholder = on
    ? '和惠惠聊聊…（persona 隔离，不进记忆）'
    : '和本地 27B 聊聊…（对话会进记忆）';
}}
async function sendChat() {{
  const inp = document.getElementById('chatInput');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  chatBubble('user', text);
  chatHistory.push({{role:'user', content:text}});
  const wait = chatBubble('assistant', '…生成中');
  try {{
    const r = await fetch('/chat', {{method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{history: chatHistory, stream: true,
        persona: document.getElementById('personaMode').checked}})}});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '', reply = '';
    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {{stream:true}});
      let nl;
      while ((nl = buf.indexOf('\\n')) >= 0) {{
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') continue;
        try {{
          const j = JSON.parse(data);
          if (j.error) throw new Error(j.error);
          const delta = (j.delta || '');
          if (delta) {{ reply += delta; wait.textContent = reply; }}
        }} catch(e) {{ /* 忽略坏帧 */ }}
      }}
    }}
    wait.textContent = reply || '（空回复）';
    chatHistory.push({{role:'assistant', content: reply}});
  }} catch(e) {{
    wait.textContent = '⚠ 模型服务不可达（白天 27B 未启动？）';
  }}
}}
async function triggerScrape() {{
  const r = document.getElementById('scrapeResult');
  r.textContent = '⏳ 抓取中（约 30-60 秒）…';
  try {{
    const res = await fetch('/scrape/email', {{method:'POST'}});
    const d = await res.json();
    if (d.busy) {{ r.textContent = '⏳ 已有抓取在进行中'; return; }}
    r.textContent = d.ok
      ? '✅ 抓取完成：' + (d.stdout || '').replace(/\\s+/g, ' ').slice(0, 60)
      : '❌ 抓取失败：' + (d.stderr || d.error || '未知错误');
  }} catch(e) {{
    r.textContent = '⚠ 触发失败（服务不可达？）';
  }}
}}
document.getElementById('chatInput').addEventListener('keydown', e => {{
  if (e.key === 'Enter') sendChat();
}});
async function refreshData() {{
  try {{
    const r = await fetch('/api/data');
    const d = await r.json();
    document.getElementById('headInfo').innerHTML = '· ' + d.sleep + ' · ' + d.now;
    document.getElementById('reportPanel').innerHTML =
      '<h2>夜班报告 ' + d.report_name + '</h2>' + d.report;
    document.getElementById('alertPanel').innerHTML =
      '<h2>⚠ Urgent 告警 ' + d.alert_badge + '</h2>' + d.alerts;
    document.getElementById('propPanel').innerHTML =
      '<h2>📋 待审批提案 ' + d.prop_badge + '</h2>' + d.proposals;
    document.getElementById('statusPanel').innerHTML = d.status;
  }} catch(e) {{ /* 服务短暂不可达时保持旧数据 */ }}
}}
setInterval(refreshData, 60000);
(function () {{
  const p = new URLSearchParams(location.search);
  const acted = p.get('acted');
  if (acted === 'ok' || acted === 'fail') {{
    const n = p.get('applied');
    const msg = acted === 'ok'
      ? (n ? '✅ 提案已批准，L3 已应用 ' + n + ' 条修改' : '✅ 操作成功')
      : '❌ 操作失败';
    const d = document.createElement('div');
    d.style.cssText = 'position:fixed;top:14px;left:50%;transform:translateX(-50%);z-index:99;' +
      'background:rgba(16,32,47,.96);color:#dceef9;border:1px solid var(--border);' +
      'border-radius:8px;padding:8px 18px;font-size:13px;box-shadow:0 2px 10px rgba(0,0,0,.35);';
    d.textContent = msg;
    document.body.appendChild(d);
    setTimeout(() => d.remove(), 4000);
    history.replaceState(null, '', location.pathname);
  }}
}})();
</script>
<footer>数据主权本地 · 数据面板 60 秒自动刷新（对话不刷新不丢失）· 🐳 对话 Tab = dsh web (:3090)<br>
鲸鱼娘素材 © 2026 zhu1090093659（@linxin666/dsh-pet，Apache-2.0）· <a href="/assets/whale/LICENSE-dsh-pet" style="color:var(--dim)">许可证</a></footer>
</body></html>"""


def collect_panels() -> dict:
    """汇总各面板数据（render_page 与 /api/data 局部刷新共用）。"""
    report_name, report_html = latest_report()
    alerts = list_alerts()
    props = list_pending_proposals()
    st = system_status()

    alert_badge = f'<span class="badge red">{len(alerts)}</span>' if alerts else '<span class="badge green">0</span>'
    alerts_html = "".join(
        f'<div class="alert"><b>{esc(a.get("source",""))}</b> · {esc(a.get("detected_at",""))}<br>'
        f'{esc(a.get("file",""))}<br><span style="color:var(--dim)">{esc(a.get("snippet","")[:120])}</span></div>'
        for a in alerts
    ) or '<p style="color:var(--dim)">无未读告警</p>'

    prop_badge = f'<span class="badge red">{len(props)}</span>' if props else '<span class="badge green">0</span>'
    props_html = ""
    for p in props:
        pid = esc(p.get("id", ""))
        props_html += (
            f'<div class="prop"><div class="title">{esc(p.get("title",""))}</div>'
            f'<div class="desc">{esc(p.get("description",""))}</div>'
            f'<form method="POST" action="/proposal" style="display:inline">'
            f'<input type="hidden" name="pid" value="{pid}">'
            f'<button class="approve" name="action" value="approve">批准</button>'
            f'<button class="reject" name="action" value="reject">否决</button>'
            f'</form></div>'
        )
    if not props_html:
        props_html = '<p style="color:var(--dim)">无待审批提案</p>'

    hb = st["heartbeat"]
    red = "🔴 红点告警：连续两晚无报告" if st["red_alert"] else "🟢 正常"
    status_html = (
        f'管线心跳：<b>{esc(hb.get("status","—"))}</b>（{esc(hb.get("segment","—"))}）<br>'
        f'心跳时间：{esc(hb.get("local_time","—"))}<br>'
        f'上次巩固：{esc(st["last_consolidation"] or "从未")}<br>'
        f'连续失败：<b>{st["consecutive_failures"]}</b><br>'
        f'{red}<br>'
        f'<a href="/l3">查看 L3 核心记忆</a>'
    )
    return {
        "sleep": sleep_state(), "now": now_cn(),
        "report_name": esc(report_name), "report": report_html,
        "alert_badge": alert_badge, "alerts": alerts_html,
        "prop_badge": prop_badge, "proposals": props_html,
        "status": status_html,
    }


def render_page() -> str:
    d = collect_panels()
    return PAGE.format(
        sleep=d["sleep"], now=d["now"],
        report_name=d["report_name"], report=d["report"],
        alert_badge=d["alert_badge"], alerts=d["alerts"],
        prop_badge=d["prop_badge"], proposals=d["proposals"],
        status=d["status"],
    )


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, code: int = 200, ctype: str = "text/html; charset=utf-8"):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send(render_page())
        elif path == "/api/data":
            self._send(json.dumps(collect_panels(), ensure_ascii=False),
                       200, "application/json; charset=utf-8")
        elif path.startswith("/assets/"):
            self._send_asset(path)
        elif path == "/l3":
            content = ""
            if os.path.exists(L3_PATH):
                with open(L3_PATH, encoding="utf-8") as f:
                    content = f.read()
            self._send(
                f'<!doctype html><meta charset="utf-8"><title>L3</title>'
                f'<body style="background:#0a1420;color:#dceef9;font:14px -apple-system,sans-serif;padding:20px">'
                f'<a href="/" style="color:#4fb3ff">← 返回</a><h2>L3 核心记忆</h2>{md_to_html(content)}</body>'
            )
        elif path == "/health":
            self._send('{"ok":true}', ctype="application/json")
        else:
            self._send("not found", 404, "text/plain")

    def _send_asset(self, path: str):
        """静态素材（鲸鱼娘等）。白名单扩展名 + 防路径穿越。"""
        rel = path[len("/assets/"):]
        if ".." in rel or rel.startswith("/"):
            self._send("bad path", 400, "text/plain")
            return
        ext = os.path.splitext(rel)[1].lower()
        ctype = {
            ".gif": "image/gif", ".png": "image/png", ".webp": "image/webp",
            ".jpg": "image/jpeg", ".json": "application/json",
        }.get(ext)
        # 许可证等无扩展名文本文件
        if not ctype and "LICENSE" in rel.upper():
            ctype = "text/plain; charset=utf-8"
        if not ctype:
            self._send("forbidden", 403, "text/plain")
            return
        fp = os.path.join(REPO, "assets", rel)
        if not os.path.isfile(fp):
            self._send("not found", 404, "text/plain")
            return
        with open(fp, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/chat":
            self._handle_chat()
            return
        if path == "/scrape/email":
            self._handle_scrape_email()
            return
        if path != "/proposal":
            self._send("not found", 404, "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(length).decode())
        pid = form.get("pid", [""])[0]
        action = form.get("action", [""])[0]
        # 只允许已知提案 id 格式（防路径注入）
        if not re.fullmatch(r"prop-[0-9a-z-]+", pid):
            self._send("bad pid", 400, "text/plain")
            return
        ok, n_applied = proposal_action(pid, action)
        # 302 回首屏（批准后若应用了 L3 修改，flash 提示）
        flash = "ok" if ok else "fail"
        if ok and n_applied:
            flash += f"&applied={n_applied}"
        self.send_response(302)
        self.send_header("Location", f"/?acted={flash}")
        self.end_headers()

    def _handle_chat(self):
        """POST /chat {history, stream?, persona?} → 白天 27B / 惠惠 persona → 对话落 L0。
        persona=true：走 :8101 惠惠实例（挂 Megumin adapter），mode=persona 隔离，不进事实记忆。
        2026-08-20：新增流式（text/event-stream，delta 逐块转发），长回答边生成边显示。"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            history = body.get("history", [])
            stream = bool(body.get("stream", False))
            persona = bool(body.get("persona", False))
            if not history or history[-1].get("role") != "user":
                self._send('{"error":"bad history"}', 400, "application/json")
                return
            user_text = history[-1]["content"]

            if persona and not _ensure_persona():
                self._send('{"error":"惠惠实例未就绪（正在启动或启动失败，看 serve-persona.log）"}',
                           503, "application/json")
                return

            gen = (chat_with_persona if persona else chat_with_day_model)
            mode = "persona" if persona else "normal"

            if stream:
                # SSE 流式：边生成边转发，结束后整段落 L0
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                full: list[str] = []
                try:
                    for chunk in gen(history, stream=True):
                        full.append(chunk)
                        self.wfile.write(
                            ("data: " + json.dumps({"delta": chunk}, ensure_ascii=False) + "\n\n").encode("utf-8"))
                        self.wfile.flush()
                except Exception as e:
                    self.wfile.write(("data: " + json.dumps({"error": str(e)[:100]}, ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                reply = "".join(full)
                if reply:
                    log_chat_to_l0(user_text, reply, mode=mode)
                return

            # 非流式（保留，供测试/兼容）
            reply = gen(history)
            log_chat_to_l0(user_text, reply, mode=mode)
            self._send(json.dumps({"reply": reply}, ensure_ascii=False),
                       200, "application/json; charset=utf-8")
        except Exception as e:
            self._send(json.dumps({"error": str(e)[:200]}, ensure_ascii=False),
                       500, "application/json; charset=utf-8")

    def _handle_scrape_email(self):
        """POST /scrape/email → 触发本地 AI 的邮箱抓取（§13：抓取器挂靠本地 AI）。

        本地 AI 主动触发入口之一（dashboard 按钮 / 未来 AI 工具可复用此端点）。
        产出正常落 exchange/inbox/email/邮箱摘要-<今天>.md，夜班增量摄入（这是设计行为）。
        并发锁：抓取中再次触发返回 busy，不重复跑。仅本机监听，显式动作才触发。"""
        import subprocess
        if not _scrape_lock.acquire(blocking=False):
            self._send(json.dumps({"ok": False, "busy": True, "error": "抓取正在进行中"},
                                  ensure_ascii=False), 200, "application/json; charset=utf-8")
            return
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(REPO, "src", "scrape.py"), "email"],
                capture_output=True, text=True, timeout=240, cwd=REPO)
            self._send(json.dumps({
                "ok": r.returncode == 0,
                "stdout": (r.stdout or "")[-400:],
                "stderr": (r.stderr or "")[-200:],
            }, ensure_ascii=False), 200, "application/json; charset=utf-8")
        except Exception as e:
            self._send(json.dumps({"ok": False, "error": str(e)[:200]}, ensure_ascii=False),
                       200, "application/json; charset=utf-8")
        finally:
            _scrape_lock.release()

    def log_message(self, fmt, *args):  # 静音 access log
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3091)  # 3090 被 dsh web 占过的坑，dashboard 用 3091
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard: http://127.0.0.1:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
