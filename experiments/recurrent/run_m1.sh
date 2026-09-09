#!/bin/bash
# M1 记法通顺度训练运行器——沙盒内(正式系统零接触), 白天执行模式: 停 8100 → 训练 → 评测 → 恢复。
# 用户指令(2026-09-09): "现在开始训就行不用等到晚上; 沙盒试验成功达成试验目的后再接正式系统"
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$ROOT/.venv/bin/python3"
MODEL="/Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61"
DATA="$HERE/../lora/datasets/organ-m1"
ADAPTER="$HERE/../lora/adapters/organ-m1-v1"
LABEL="com.local-ai-agent.day-model"
PLIST="$HOME/Library/LaunchAgents/com.local-ai-agent.day-model.plist"

RESTORE=0
stop_8100() {
  if curl -s --max-time 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then
    echo "[M1] 停 8100(防双驻留——训练模型也算一份)"
    launchctl bootout gui/501/"$LABEL" 2>/dev/null || true
    sleep 3
    RESTORE=1
  fi
}
restore_8100() {
  if [ "$RESTORE" -eq 1 ]; then
    echo "[M1] 恢复 8100 ..."
    launchctl bootstrap gui/501 "$PLIST" 2>/dev/null || true
    sleep 12
    curl -s --max-time 5 http://127.0.0.1:8100/v1/models >/dev/null 2>&1 \
      && echo "[M1] 8100 已恢复 ✓" || echo "[M1] 8100 恢复中(KeepAlive 会拉起)"
  fi
}
trap restore_8100 EXIT

stop_8100

echo "[M1] 训练: 450 样本 × 240 iters × 16 层(峰值 ≈20.7GB, M0 已验证量级)"
HF_HUB_OFFLINE=1 "$PY" -m mlx_lm.lora \
  --model "$MODEL" --train --data "$DATA" \
  --adapter-path "$ADAPTER" \
  --batch-size 1 --iters 240 --learning-rate 1e-5 \
  --num-layers 16 --max-seq-length 2048 --grad-checkpoint \
  --steps-per-report 20 --seed 42
tr=$?
echo "[M1] 训练退出码: $tr"
if [ $tr -ne 0 ] || [ ! -f "$ADAPTER/adapters.safetensors" ]; then
  echo "[M1] 训练失败, 跳过评测"
  exit $tr
fi

echo "[M1] 三关评测(记法阅读力/行为/能力):"
HF_HUB_OFFLINE=1 "$PY" "$HERE/eval_m1.py"
ev=$?
echo "[M1] 评测退出码: $ev"
exit $ev
