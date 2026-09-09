#!/bin/bash
# R0 演示入口——默认只打印计划(骨架纪律); 仅 --execute 真实运行(用户验收后使用)。
# 真实运行: 停 8100(防双驻留) → eval_r0(K=1 基线 + K=2 层表重排, 同进程对照)
#           → 恢复 8100(trap EXIT 兜底) → 报告 report-r0.json/md
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="/Users/cz/WorkBuddy/watch/ai-sandbox-stress/.venv/bin/python3"
MODEL="/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
LABEL="com.local-ai-agent.day-model"
PLIST="$HOME/Library/LaunchAgents/com.local-ai-agent.day-model.plist"

if [ "${1:-}" != "--execute" ]; then
  echo "[run_r0] 骨架模式——只打印计划(未验收, 不运行):"
  echo "=================================================="
  "$PY" "$HERE/loop_model.py" --dry-run --model "$MODEL"
  echo "--------------------------------------------------"
  "$PY" "$HERE/eval_r0.py" --dry-run --model "$MODEL"
  echo "=================================================="
  echo "[run_r0] 验收通过后执行: bash $0 --execute  (~15 分钟; 自动停/恢复 8100)"
  exit 0
fi

shift   # 丢弃 --execute, 剩余参数透传给 eval_r0.py (--k/--start/--end/--out)
echo "[run_r0] 真实模式: 停 8100 → R0 评测 → 恢复 8100  (变体参数: $*)"
RESTORE=0

stop_8100() {
  if curl -s --max-time 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then
    echo "[run_r0] 停 8100(防双驻留) ..."
    launchctl bootout gui/501/"$LABEL" 2>/dev/null || true
    sleep 3
    RESTORE=1
  else
    echo "[run_r0] 8100 本就不在线, 跳过"
  fi
}

restore_8100() {
  if [ "$RESTORE" -eq 1 ]; then
    echo "[run_r0] 恢复 8100 ..."
    launchctl bootstrap gui/501 "$PLIST" 2>/dev/null || true
    sleep 10
    if curl -s --max-time 5 http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then
      echo "[run_r0] 8100 已恢复 ✓"
    else
      echo "[run_r0] 8100 恢复中(KeepAlive 会拉起)"
    fi
  fi
}

trap restore_8100 EXIT
stop_8100

HF_HUB_OFFLINE=1 "$PY" "$HERE/eval_r0.py" --model "$MODEL" "$@"
rc=$?
echo "[run_r0] eval 退出码: $rc"
exit $rc
