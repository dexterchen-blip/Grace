#!/usr/bin/env bash
# 雷姆 LoRA 训练 —— 参考惠惠 megumin_v6 已验证配方（adapter_config.json）
# 用法（必须在沙盒内）:
#   bash v2/tools/train_rem.sh [--iters N] [--dry-run]
#
# 硬约束:
#   - 必须在 ai-sandbox 内跑（./run.sh 白名单环境）
#   - 48GB 单模型铁律：训练前必须先停白天 27B day-model（8100），训完恢复！
set -euo pipefail
SB="/Users/cz/WorkBuddy/watch/ai-sandbox"
cd "$SB"

ITERS=1000
DRY=0
for a in "$@"; do
  case "$a" in
    --iters=*) ITERS="${a#*=}" ;;
    --dry-run) DRY=1 ;;
  esac
done

ADAPTER="$SB/experiments/lora/adapters/rem_v1"
DATA="$SB/experiments/lora/datasets/rem"
LOGFILE="$SB/logs/train-rem-$(date +%Y%m%d-%H%M).log"

echo "=== 雷姆 LoRA 训练（Grace V2 M1）==="
echo "基座   : mlx-community/Qwen3.8-27B-4bit（宿主 HF cache 只读引用）"
echo "数据   : ${DATA}（$(wc -l < "${DATA}/train.jsonl" | tr -d ' ') 训练 / $(wc -l < "${DATA}/valid.jsonl" | tr -d ' ') 验证）"
echo "超参   : rank=8 scale=20 lr=1e-5 iters=${ITERS} layers=16 max_seq=2048 batch=1"
echo "适配器 : ${ADAPTER}"

# 前置检查：8100 白天 27B 必须已停（单模型铁律）
if lsof -i :8100 -P 2>/dev/null | grep -q LISTEN; then
  echo "❌ 8100 白天 27B 仍在运行 —— 违反 48GB 单驻留铁律！请先停 day-model 再训。"
  echo "   停: launchctl bootout gui/501 com.local-ai-agent.day-model   （恢复: launchctl bootstrap gui/501 .../serve_day.plist）"
  exit 1
fi

mkdir -p "$(dirname "$ADAPTER")" logs
if [ "$DRY" = "1" ]; then
  echo "[dry-run] 检查通过，未执行训练。真实命令："
fi
echo ">>> $SB/run.sh .venv/bin/python3 -m mlx_lm.lora -c v2/tools/train_rem.yaml 2>&1 | tee $LOGFILE"
[ "$DRY" = "1" ] && exit 0

"$SB/run.sh" .venv/bin/python3 -m mlx_lm.lora \
  -c v2/tools/train_rem.yaml 2>&1 | tee "$LOGFILE"
echo "=== 训练结束，日志: $LOGFILE ==="
