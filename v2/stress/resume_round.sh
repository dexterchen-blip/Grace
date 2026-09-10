#!/bin/bash
# ★2026-09-09 热停后续跑: 跳过 reset/种子(L0/L2 保留), 直启引擎按 L0 水位断点续跑。
# 用法: DENSITY=3 GRACE_SYNTHETIC=1 GRACE_CURIOSITY=1 bash v2/stress/resume_round.sh --days 37
set -u
cd "$(dirname "$0")/../.."
DAYS="${2:-37}"
DENSITY="${DENSITY:-1}"
echo "=== resume_round 续跑 $(date +%H:%M:%S) | days=$DAYS density=$DENSITY ===" >> experiments/run/stress/auto-round.log
GRACE_EWC=1 GRACE_SLEEP_DELTA=0.03 ./run.sh .venv/bin/python3 v2/stress/stress_engine.py \
    --days "$DAYS" --density "$DENSITY" --train-every 1 --sample-every 11 \
    --reset-interval 0 --inputs-dir inputs-v3 \
    >> experiments/run/stress/stress.log 2>&1
RC=$?
echo "=== resume_round 完成 rc=$RC $(date +%H:%M:%S) ===" >> experiments/run/stress/auto-round.log
exit $RC
