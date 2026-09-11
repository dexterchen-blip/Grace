#!/bin/bash
# 体量触发·冻结判官检查点编排器（2026-09-10 用户定案：数据量到阈值 → 停 Grace → 起 35B 筛 → 恢复）。
# 驻留安全：全程任何时刻只有一份模型——停引擎/8100 → 起 35B → 筛 → 停 35B → 恢复 8100。
#
# 用法: bash v2/stress/checkpoint_swap.sh <samples.jsonl> [--no-resume]
#   --no-resume: 只筛不续跑（手动接管时用）
#
# ⚠️ 沙箱注意（macos-launchd-scheduled-task 技能坑）: WorkBuddy 沙箱 shell 里
#    launchctl 写操作可能报 error 5——本脚本带 osascript 管理员路径兜底（会弹图形密码框）。
#    生产化后此脚本应由 launchd/正式 shell 拉起（无沙箱, 直连 launchctl 即可）。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STRESS="$ROOT/experiments/run/stress"
PY="$ROOT/.venv/bin/python3"
LABEL="com.local-ai-agent.day-model"
PLIST="$HOME/Library/LaunchAgents/com.local-ai-agent.day-model.plist"
NIGHT_VENV="/Users/cz/.workbuddy/binaries/python/envs/llama-cpp/bin/python"
NIGHT_MODEL="/Users/cz/WorkBuddy/skills find and make/local-ai-agent/models/night/Qwen_Qwen3.5-35B-A3B-Q6_K.gguf"
LOG="$STRESS/checkpoint-swap.log"
SAMPLES="${1:-}"; RESUME=1
[ "${2:-}" = "--no-resume" ] && RESUME=0
mkdir -p "$STRESS/judgments"
log() { echo "[ckpt $(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

_8100_up() { curl -s --max-time 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1; }
_8200_up() { curl -s --max-time 3 http://127.0.0.1:8200/v1/models >/dev/null 2>&1; }
stop_8100() {
  _8100_up || return 0
  launchctl bootout "gui/501/$LABEL" 2>/dev/null \
    || osascript -e "do shell script \"launchctl bootout gui/501/$LABEL\" with administrator privileges" 2>/dev/null
  sleep 3; _8100_up && { log "8100 停止失败, 中止（防双驻留）"; return 1; } || log "8100 已停"
}
restore_8100() {
  _8100_up && return 0
  launchctl bootstrap "gui/501" "$PLIST" 2>/dev/null \
    || osascript -e "do shell script \"launchctl bootstrap gui/501 $PLIST\" with administrator privileges" 2>/dev/null
  for i in $(seq 1 24); do _8100_up && { log "8100 已恢复 ✓"; return 0; }; sleep 5; done
  log "⚠️ 8100 恢复失败——需人工处理"
  return 1
}
trap restore_8100 EXIT

[ -z "$SAMPLES" ] && { echo "用法: checkpoint_swap.sh <samples.jsonl> [--no-resume]"; exit 1; }
[ -f "$SAMPLES" ] || { echo "样本文件不存在: $SAMPLES"; exit 1; }
log "=== 检查点启动: $(wc -l < "$SAMPLES" | tr -d ' ') 条待筛 ==="

# ---- A. 热停（引擎每 ~15min 检查 .stop-requested, 最多等 25 分钟）----
if pgrep -f stress_engine >/dev/null; then
  touch "$STRESS/.stop-requested"
  log "已留 .stop-requested, 等引擎退出点…"
  for i in $(seq 1 150); do pgrep -f stress_engine >/dev/null || break; sleep 10; done
  if pgrep -f stress_engine >/dev/null; then
    log "引擎 25 分钟未退出, 放弃本轮检查点（引擎继续跑, 不动模型）"
    rm -f "$STRESS/.stop-requested"; exit 1
  fi
  rm -f "$STRESS/.stop-requested"
  log "引擎已热停"
fi
stop_8100 || exit 1

# ---- B. 起 35B 判官（thinking 关闭提速; serve_night.sh 注释里的官方开关）----
HF_HUB_OFFLINE=1 "$NIGHT_VENV" -m llama_cpp.server \
  --model "$NIGHT_MODEL" --host 127.0.0.1 --port 8200 \
  --n_gpu_layers -1 --n_ctx 65536 \
  --chat_template_kwargs '{"enable_thinking": false}' > /tmp/judge-35b.log 2>&1 &
JPID=$!
for i in $(seq 1 90); do _8200_up && break; sleep 5; done
_8200_up || { log "35B 判官起不来（见 /tmp/judge-35b.log）, 恢复 8100"; exit 1; }
log "35B 判官在线 (pid=$JPID, thinking off)"

# ---- C. 筛选 ----
OUT="$STRESS/judgments/$(date +%Y%m%d-%H%M%S)-verdicts.jsonl"
"$PY" "$ROOT/v2/stress/judge35b.py" "$SAMPLES" "$OUT"; JRC=$?

# ---- D. 停 35B（恢复 8100 由 trap 兜底）----
kill "$JPID" 2>/dev/null; wait "$JPID" 2>/dev/null
log "判官完成 rc=$JRC → $OUT"
[ $JRC -ne 0 ] && log "⚠️ 存在 unjudged——该批样本不得进训练集（完整性纪律）"

# ---- E. 续跑（可选）----
if [ "$RESUME" = "1" ] && [ -f "$ROOT/v2/stress/resume_round.sh" ]; then
  log "续跑轮次…"
  bash "$ROOT/v2/stress/resume_round.sh" >> "$LOG" 2>&1
fi
exit $JRC
