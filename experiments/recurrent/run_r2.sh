#!/bin/bash
# R2 器官×循环合体运行器——沙盒: 停 8100 → 合体评测 → 恢复。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="/Users/cz/WorkBuddy/watch/ai-sandbox-stress/.venv/bin/python3"
LABEL="com.local-ai-agent.day-model"
PLIST="$HOME/Library/LaunchAgents/com.local-ai-agent.day-model.plist"
RESTORE=0
stop_8100() { curl -s --max-time 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1 && { launchctl bootout gui/501/"$LABEL" 2>/dev/null; sleep 3; RESTORE=1; }; }
restore_8100() { [ "$RESTORE" -eq 1 ] && { launchctl bootstrap gui/501 "$PLIST" 2>/dev/null; sleep 12; curl -s --max-time 5 http://127.0.0.1:8100/v1/models >/dev/null && echo "[R2] 8100 已恢复 ✓"; }; }
trap restore_8100 EXIT
stop_8100
HF_HUB_OFFLINE=1 "$PY" "$HERE/../lora/eval_r2.py"
exit $?
