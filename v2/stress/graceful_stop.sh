#!/bin/bash
# ★2026-09-09 热停止: 引擎每天边界(~10-15min)检查 .stop-requested 优雅退出。
# 用法: bash v2/stress/graceful_stop.sh [--now]
#   默认: 等引擎天边界自退(最多少跑半天, 数据零损)
#   --now: 立即 kill(天中杀, 下次启动 reset 归档重来, 也安全)
set -u
FLAG="$(cd "$(dirname "$0")/../../experiments/run/stress" 2>/dev/null && pwd)/.stop-requested"
if [ "${1:-}" = "--now" ]; then
  pkill -f stress_engine; pkill -f "auto_round" 2>/dev/null
  rm -f "$FLAG" 2>/dev/null
  echo "[stop] 已立即终止(天中杀, 下次 reset 归档重来)"
  exit 0
fi
touch "$FLAG"
echo "[stop] 停止请求已登记——引擎将在下一个天边界退出(≤15 分钟)"
echo "[stop] 观察: tail -f experiments/run/stress/stress.log | grep checkpoint"
for i in $(seq 1 100); do
  sleep 10
  pgrep -f stress_engine >/dev/null || { rm -f "$FLAG"; echo "[stop] 引擎已优雅退出 ✓"; exit 0; }
done
echo "[stop] 15分钟仍在跑(可能在长训练)——再等等或用 --now 强停"
