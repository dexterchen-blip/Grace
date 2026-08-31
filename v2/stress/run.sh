#!/usr/bin/env bash
# ai-sandbox 运行入口：在完全隔离的环境中执行命令。
#
# 用法:
#   ./run.sh <cmd...>                      # 沙箱内跑任意命令
#   ./run.sh python3 src/l0_ingest.py      # 例子
#
# 隔离保证:
#   - env -i：丢弃宿主一切环境变量（PATH/HOME/代理/密钥全不继承）
#   - AIAGENT_SANDBOX 强制指向本沙箱 → 代码所有 memory/exchange 写路径进沙箱
#   - HOME 重定向到沙箱内 home/（防写宿主 ~/.cache、~/.workbuddy 等）
#   - AIAGENT_WECHAT_DB 指向沙箱空读源 → 微信 ingest 自动跳过，不碰正式 summary.db
#   - HF_HUB_OFFLINE=1 + 不设代理 → 无网络外泄
#   - HF_HOME 默认只读引用宿主 HF cache（模型权重；AIAGENT_HF_HOME 可覆盖为沙箱内拷贝 → 完全独立）
set -euo pipefail
SB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AIAGENT_SANDBOX="$SB"
export SANDBOX_HOME="$SB/home"
export SANDBOX_PATH="$SB/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export AIAGENT_WECHAT_DB="$SB/data/sources/summary.db"
export HF_HUB_OFFLINE=1
# night_pipeline 内部调 L2 的解释器 → 沙箱 venv（依赖未装时由 setup_deps.sh 补齐）
export AIAGENT_L2_PY="$SB/.venv/bin/python"
# 沙箱 serve 脚本的模型解释器/权重回退（默认仍指向正式只读权重；完全独立可拷贝模型后覆盖）
export AIAGENT_SANDBOX_PY="${AIAGENT_SANDBOX_PY:-$SB/.venv/bin/python}"
export AIAGENT_MODEL_NIGHT="${AIAGENT_MODEL_NIGHT:-/Users/cz/WorkBuddy/skills find and make/local-ai-agent/models/night/Qwen_Qwen3.5-35B-A3B-Q6_K.gguf}"
# HF 模型缓存：默认只读引用宿主（27B/小模型都在）；AIAGENT_HF_HOME=<sandbox>/models/hf 即完全独立
export HF_HOME="${AIAGENT_HF_HOME:-/Users/cz/.cache/huggingface}"

mkdir -p "$SB/home" "$SB/logs" "$SB/data/sources"

exec env -i \
  PATH="$SANDBOX_PATH" \
  HOME="$SANDBOX_HOME" \
  AIAGENT_SANDBOX="$AIAGENT_SANDBOX" \
  AIAGENT_WECHAT_DB="$AIAGENT_WECHAT_DB" \
  AIAGENT_L2_PY="$AIAGENT_L2_PY" \
  AIAGENT_SANDBOX_PY="$AIAGENT_SANDBOX_PY" \
  AIAGENT_MODEL_NIGHT="$AIAGENT_MODEL_NIGHT" \
  HF_HOME="$HF_HOME" \
  HF_HUB_OFFLINE=1 \
  GRACE_EWC="${GRACE_EWC:-0}" \
  GRACE_EWC_LAMBDA="${GRACE_EWC_LAMBDA:-1e-3}" \
  LANG="en_US.UTF-8" LC_ALL="en_US.UTF-8" TZ="Asia/Shanghai" \
  "$@"
