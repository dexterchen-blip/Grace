#!/bin/bash
# M2 mood 闭式转向运行器——沙盒内(正式零接触): 停 8100 → 向量采集+验证 → 恢复。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="/Users/cz/WorkBuddy/watch/ai-sandbox-stress/.venv/bin/python3"
LABEL="com.local-ai-agent.day-model"
PLIST="$HOME/Library/LaunchAgents/com.local-ai-agent.day-model.plist"
RESTORE=0
stop_8100() {
  if curl -s --max-time 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then
    echo "[M2] 停 8100(防双驻留)"
    launchctl bootout gui/501/"$LABEL" 2>/dev/null || true
    sleep 3
    RESTORE=1
  fi
}
restore_8100() {
  if [ "$RESTORE" -eq 1 ]; then
    launchctl bootstrap gui/501 "$PLIST" 2>/dev/null || true
    sleep 12
    curl -s --max-time 5 http://127.0.0.1:8100/v1/models >/dev/null 2>&1 \
      && echo "[M2] 8100 已恢复 ✓" || echo "[M2] 8100 恢复中(KeepAlive 会拉起)"
  fi
}
trap restore_8100 EXIT
stop_8100
HF_HUB_OFFLINE=1 "$PY" "$HERE/../lora/m2_steering.py"
exit $?
