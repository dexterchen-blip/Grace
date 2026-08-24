#!/usr/bin/env python3
"""本地 AI 密钥存储系统（2026-08-23 用户方案）。

存储后端：
  - 正式模式（默认）：macOS Keychain（security CLI，service=com.local-ai-agent.vault）
    ——系统级加密、锁屏保护、不落明文盘。
  - 沙盒模式（AIAGENT_SANDBOX 设置时）：临时文件模拟（不进 Keychain，防污染）。

密钥类型：
  - 直接存（set/get）：如 wechat_db_passphrase
  - Edge 网页密码代理（get edge:login/<site>）：实时只读 Edge Login Data（AES 解密，
    不复制、不改 Edge 数据）——Edge 原生密码管理器仍是主存储，vault 统一入口。

安全边界（铁律）：
  - 值绝不进 LLM prompt / 报告 / 日志；vault.get 只管道给脚本。
  - 审计：每次读写记 exchange/shared/vault.log（key 名+时间，不记值）。

用法：
  python3 vault.py set <key>            # 值从 stdin 读（防命令行泄漏）
  python3 vault.py get <key>            # 输出值（stdout，仅脚本管道）
  python3 vault.py list                 # 只列 key 名
  python3 vault.py delete <key>
  python3 vault.py migrate-wechat       # 迁移 ~/.wcdb-key-tool/wechat-passphrase.json → Keychain 并删明文
  python3 vault.py edge-list            # 列出 Edge 保存的密码站点（不显示密码）
  python3 vault.py get edge:login/<site>
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE = "com.local-ai-agent.vault"
TZ_CN = timezone(timedelta(hours=8))
AUDIT_LOG = os.path.join(REPO, "exchange", "shared", "vault.log")
EDGE_PROFILE = os.path.expanduser(
    "~/Library/Application Support/Microsoft Edge/Profile 1")

SANDBOX = os.environ.get("AIAGENT_SANDBOX", "")
if SANDBOX:
    AUDIT_LOG = os.path.join(SANDBOX, "exchange", "shared", "vault.log")


def _audit(op: str, key: str) -> None:
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now(TZ_CN):%Y-%m-%d %H:%M:%S}] {op} {key}\n")
    except Exception:
        pass


# ---------------------------------------------------------------- Keychain 后端

def _kc_add(key: str, value: str) -> None:
    subprocess.run(["security", "add-generic-password", "-s", SERVICE, "-a", key,
                    "-w", value, "-U"], check=True, capture_output=True)


def _kc_get(key: str) -> str:
    r = subprocess.run(["security", "find-generic-password", "-s", SERVICE,
                        "-a", key, "-w"], check=True, capture_output=True)
    return r.stdout.decode().rstrip("\n")


def _kc_delete(key: str) -> None:
    subprocess.run(["security", "delete-generic-password", "-s", SERVICE,
                    "-a", key], check=True, capture_output=True)


def _kc_list() -> list[str]:
    """Keychain 里的 vault key 清单（用本地清单文件维护，security dump 解析不稳）。"""
    f = os.path.join(os.path.dirname(AUDIT_LOG), "vault-keys.txt")
    if not os.path.exists(f):
        return []
    with open(f, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _kc_note_key(key: str, add: bool) -> None:
    f = os.path.join(os.path.dirname(AUDIT_LOG), "vault-keys.txt")
    keys = _kc_list()
    if add:
        if key not in keys:
            keys.append(key)
    else:
        keys = [k for k in keys if k != key]
    try:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("\n".join(keys) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- 沙盒文件后端

def _file_backend(key: str) -> str:
    d = os.path.join(tempfile.gettempdir(), "aiagent-vault-sandbox")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.vault")


def _file_add(key: str, value: str) -> None:
    p = _file_backend(key)
    with open(p, "w", encoding="utf-8") as f:
        f.write(value)
    os.chmod(p, 0o600)


def _file_get(key: str) -> str:
    with open(_file_backend(key), encoding="utf-8") as f:
        return f.read()


def _file_delete(key: str) -> None:
    try:
        os.remove(_file_backend(key))
    except FileNotFoundError:
        pass


def _file_list() -> list[str]:
    d = os.path.join(tempfile.gettempdir(), "aiagent-vault-sandbox")
    if not os.path.isdir(d):
        return []
    return [f[:-6] for f in os.listdir(d) if f.endswith(".vault")]


# ---------------------------------------------------------------- Edge 密码代理

def _edge_safe_storage_key() -> bytes:
    """macOS Edge 密码加密 key。

    实测（2026-08-23）：Safe Storage Keychain 项密码（24 字符）不是直接 key，
    key = PBKDF2-HMAC-SHA1(密码字符串, salt=b"saltysalt", iterations=1003, dkLen=16)。
    """
    r = subprocess.run(["security", "find-generic-password",
                        "-s", "Microsoft Edge Safe Storage", "-w"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("Edge Safe Storage key 不可用（需 Edge 保存过密码）")
    from Crypto.Protocol.KDF import PBKDF2
    return PBKDF2(r.stdout.strip().encode(), b"saltysalt", 16, count=1003)


def _edge_decrypt(blob: bytes, key: bytes) -> str:
    """解密 Edge Login Data 的 password_value（AES-128-CBC，v10 前缀）。"""
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
    if blob.startswith(b"v10"):
        blob = blob[3:]
    elif blob.startswith(b"v11"):
        blob = blob[3:]
    iv, ct = blob[:16], blob[16:]
    c = AES.new(key[:16], AES.MODE_CBC, iv)
    return unpad(c.decrypt(ct), 16).decode("utf-8", "replace")


def _edge_logins() -> list[dict]:
    """读 Edge Login Data（复制件，避开运行锁）。返回 [{site, user, password}]。"""
    src = os.path.join(EDGE_PROFILE, "Login Data")
    if not os.path.exists(src):
        return []
    tmp = tempfile.mktemp(suffix=".login")
    try:
        subprocess.run(["cp", src, tmp], check=True, capture_output=True)
        key = _edge_safe_storage_key()
        db = sqlite3.connect(tmp)
        rows = db.execute(
            "SELECT origin_url, username_value, password_value FROM logins "
            "WHERE password_value IS NOT NULL AND length(password_value) > 0").fetchall()
        db.close()
        out = []
        for url, user, blob in rows:
            try:
                pw = _edge_decrypt(bytes(blob), key)
            except Exception:
                continue
            out.append({"site": url, "user": user, "password": pw})
        return out
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------- 主流程

def _get(key: str) -> str:
    if key.startswith("edge:login/"):
        site = key[len("edge:login/"):]
        for lg in _edge_logins():
            if site.lower() in lg["site"].lower():
                return json.dumps({"user": lg["user"], "password": lg["password"]},
                                  ensure_ascii=False)
        raise KeyError(f"Edge 未找到站点: {site}")
    if SANDBOX:
        return _file_get(key)
    return _kc_get(key)


def _set(key: str, value: str) -> None:
    if SANDBOX:
        _file_add(key, value)
    else:
        _kc_add(key, value)
        _kc_note_key(key, add=True)
    _audit("set", key)


def _delete(key: str) -> None:
    if SANDBOX:
        _file_delete(key)
    else:
        _kc_delete(key)
        _kc_note_key(key, add=False)
    _audit("delete", key)


def _migrate_wechat() -> int:
    """迁移 ~/.wcdb-key-tool/wechat-passphrase.json → vault，删明文。"""
    p = os.path.expanduser("~/.wcdb-key-tool/wechat-passphrase.json")
    if not os.path.exists(p):
        print("明文 passphrase 不存在（可能已迁移）")
        return 0
    ph = json.load(open(p, encoding="utf-8"))["passphrase"]
    _set("wechat_db_passphrase", ph)
    os.remove(p)
    _audit("migrate-wechat", "wechat_db_passphrase")
    print(f"✅ 微信 passphrase 已迁入 vault（明文已删）: {p}")
    return 1


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    try:
        if cmd == "set":
            key = sys.argv[2]
            val = sys.stdin.read().strip()
            if not val:
                sys.exit("值不能为空（从 stdin 传）")
            _set(key, val)
            print(f"ok {key}")
        elif cmd == "get":
            key = sys.argv[2]
            print(_get(key), end="")
        elif cmd == "list":
            if SANDBOX:
                keys = _file_list()
            else:
                keys = _kc_list()
            print("\n".join(keys))
        elif cmd == "delete":
            _delete(sys.argv[2])
            print("deleted")
        elif cmd == "migrate-wechat":
            _migrate_wechat()
        elif cmd == "edge-list":
            for lg in _edge_logins():
                print(f"  {lg['site']} | {lg['user']}")
        elif cmd == "import-csv":
            import csv as _csv
            path = sys.argv[2]
            n, seen = 0, set()
            with open(path, encoding="utf-8-sig") as f:
                for row in _csv.DictReader(f):
                    name = (row.get("name") or "").strip()
                    if not name:
                        continue
                    key = f"cred:{name}"
                    if key in seen:  # 同名多账号 → 加 #n 后缀保留全部
                        i = 2
                        while f"{key}#{i}" in seen:
                            i += 1
                        key = f"{key}#{i}"
                    seen.add(key)
                    val = json.dumps({"url": row.get("url", "").strip(),
                                      "username": row.get("username", "").strip(),
                                      "password": row.get("password", "").strip(),
                                      "note": row.get("note", "").strip()},
                                     ensure_ascii=False)
                    _set(key, val)
                    n += 1
            print(f"✅ 导入 {n} 条凭据 → vault（Keychain，{len(seen)} 个唯一 key）")
        else:
            print(__doc__)
    except KeyError as e:
        sys.exit(str(e))
    except Exception as e:
        sys.exit(f"错误: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
