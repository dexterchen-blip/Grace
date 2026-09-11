#!/bin/bash
# 体量触发器 v1（2026-09-10 判官首战标定：阈值 600 / 吞吐 28 条每分钟 / swap ~5min）。
#
# 职责: 发射完整压测轮 + 监视未筛样本体量, 到阈值自动走检查点:
#   热停引擎(自身) → 等 auto_round wrapper 尾声收完(消除其 bootstrap-8100 与 swap 的竞态)
#   → checkpoint_swap.sh(停 8100 → 35B 判官 → 恢复 8100) → mark 记账 → bg-resume 续跑。
#
# 时间纪律(03:00 夜班管线互斥铁律——轮引擎进程内 15.5G + 管线 35B 28.2G 禁同驻):
#   NO_NEW_CKPT(0120) 后不开新检查点 / RESUME_UNTIL(0135) 后不再续跑
#   HARD_STOP(0155) 热停轮次, 触发器退出 — 明晨 resume_round.sh 续跑。
#
# 用法: nohup bash v2/stress/volume_trigger.sh > /tmp/volume-trigger.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STRESS="$ROOT/experiments/run/stress"
JDIR="$STRESS/judgments"
TRIGLOG="$STRESS/volume-trigger.log"
# ★2026-09-11 校准修正: 阈值口径=去重后唯一文本(实测 16% 唯一率, 3488行→589唯一)。
#   原按原始吞吐定 600 → 整轮都触不到; 300 ≈ 每 6-7 个模拟天一个检查点(判官窗 ~11min)。
THRESHOLD="${THRESHOLD:-300}"
ATTACH="${ATTACH:-0}"            # 1=接管模式: 轮已在跑, 只监视不开轮(中途修触发器用)
NO_NEW_CKPT="${NO_NEW_CKPT:-0120}"
RESUME_UNTIL="${RESUME_UNTIL:-0135}"
HARD_STOP="${HARD_STOP:-0155}"
# ★2026-09-10 修: HHMM 字符串比较跨午夜失灵("2342">="0155" 恒真→23:42 误触硬停)
#   改用"午夜起分钟数", 22:00-23:59 记为负值(=午夜前), 三条线化为分钟:
NO_NEW_M=80    # 01:20 后不开新检查点
RESUME_M=95    # 01:35 后不再续跑
HARD_M=115     # 01:55 热停(03:00 管线让路)
RESUME_MODE="${RESUME_MODE:-0}"   # 1=续跑既有轮(不 reset), 0=发新轮
# V2.5 组装态轮环境(与 run.sh 白名单对齐)
export GRACE_V25=1 DENSITY=3 GRACE_SYNTHETIC=1 GRACE_CURIOSITY=1 SAMPLE_EVERY=11
PY="$ROOT/.venv/bin/python3"
CTR="$ROOT/v2/stress/unjudged_counter.py"

log(){ echo "[trigger $(date '+%H:%M:%S')] $*" >> "$TRIGLOG"; }
mins(){ local h m; h=$(date +%H); m=$(date +%M); if [ "$h" -ge 22 ]; then echo $(( (10#$h - 24) * 60 + 10#$m )); else echo $(( 10#$h * 60 + 10#$m )); fi; }
engine_up(){ pgrep -f stress_engine.py >/dev/null 2>&1; }
wrapper_up(){ pgrep -f "auto_round|resume_round" >/dev/null 2>&1; }

# ---------- 前置 ----------
curl -s --max-time 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1 \
  || { log "中止: 8100 离线"; exit 1; }
curl -s --max-time 3 http://127.0.0.1:8200/v1/models >/dev/null 2>&1 \
  && { log "中止: 8200 在线(异常驻留)"; exit 1; }
if [ "$ATTACH" = "1" ]; then
  engine_up || { log "中止: 接管模式要求轮在跑"; exit 1; }
  ENGINE_STARTED=1
else
  engine_up && { log "中止: 已有轮在跑"; exit 1; }
fi

log "=== 体量触发器启动 | 阈值=$THRESHOLD 无新检查点=01:20 续跑线=01:35 硬停=01:55 模式=$([ "$ATTACH" = 1 ] && echo attach || ([ "$RESUME_MODE" = 1 ] && echo resume || echo fresh)) ==="

# ---------- 发射轮(fresh: auto_round 带 reset 归档+播种 / resume: 断点续跑 / attach: 不发) ----------
if [ "$ATTACH" = "1" ]; then
  log "接管模式: 挂到运行中的轮(不开新轮)"
elif [ "$RESUME_MODE" = "1" ]; then
  nohup bash "$ROOT/v2/stress/resume_round.sh" --days 37 >> "$STRESS/auto-round.log" 2>&1 &
  log "resume 模式: 续跑已发射(按 L0 水位)"
else
  bash "$ROOT/v2/stress/auto_round.sh" --days 37 --sample-every 11 >> "$TRIGLOG" 2>&1
  log "轮已发射(reset+播种 ~10 分钟后引擎起跑)"
fi

[ "$ATTACH" = "1" ] || ENGINE_STARTED=0
FAIL_BACKOFF=0     # 检查点失败后的退避时间戳(epoch)

while true; do
  sleep 60
  engine_up && ENGINE_STARTED=1
  N=$("$PY" "$CTR" count 2>>"$TRIGLOG" || echo 0)

  # ---- 硬停纪律(优先级最高, 03:00 管线让路; 22:00-24:00=负分钟, 恒不触发) ----
  if [ "$(mins)" -ge "$HARD_M" ]; then
    if engine_up; then
      log "到达硬停线 01:55 → 热停轮次(未筛 $N 条留给明晨)"
      touch "$STRESS/.stop-requested"
      for i in $(seq 1 96); do engine_up || break; sleep 10; done
      rm -f "$STRESS/.stop-requested"
    fi
    for i in $(seq 1 12); do wrapper_up || break; sleep 10; done
    log "触发器退出(硬停纪律) — 明晨: GRACE_V25=1 DENSITY=3 GRACE_SYNTHETIC=1 GRACE_CURIOSITY=1 bash v2/stress/resume_round.sh --days 37"
    exit 0
  fi

  # ---- 检查点条件 ----
  if engine_up && [ "$N" -ge "$THRESHOLD" ] && [ "$(mins)" -lt "$NO_NEW_M" ] \
     && [ "$(date +%s)" -ge "$FAIL_BACKOFF" ]; then
    log "阈值达到: 未筛 $N 条 → 热停检查点"
    touch "$STRESS/.stop-requested"
    for i in $(seq 1 96); do engine_up || break; sleep 10; done
    if engine_up; then
      rm -f "$STRESS/.stop-requested"
      FAIL_BACKOFF=$(( $(date +%s) + 900 ))
      log "引擎 16 分钟未退出, 放弃本检查点, 退避 15 分钟"
      continue
    fi
    rm -f "$STRESS/.stop-requested"
    # 等 wrapper 尾声(integrity 标记+append+幂等 bootstrap)收完再动模型——消除竞态
    for i in $(seq 1 12); do wrapper_up || break; sleep 10; done
    IN="$JDIR/$(date +%Y%m%d-%H%M%S)-input.jsonl"
    M=$("$PY" "$CTR" build "$IN" 800 2>>"$TRIGLOG" || echo 0)
    log "检查点输入 $M 条 → $IN"
    if [ "$M" -gt 0 ]; then
      bash "$ROOT/v2/stress/checkpoint_swap.sh" "$IN" --no-resume >> "$TRIGLOG" 2>&1
      RC=$?
      V=$(ls -t "$JDIR"/*-verdicts.jsonl 2>/dev/null | head -1)
      if [ -n "$V" ]; then
        AD=$("$PY" "$CTR" mark "$IN" "$V" 2>>"$TRIGLOG" || echo 0)
        log "检查点完成 rc=$RC | 记账 $AD 条 | verdicts=$V"
      else
        log "检查点 rc=$RC 但无 verdicts 产物——不记账, 下轮重试"
        FAIL_BACKOFF=$(( $(date +%s) + 900 ))
      fi
      if [ "$(mins)" -lt "$RESUME_M" ]; then
        log "bg-resume 轮次(从 L0 水位续跑)"
        nohup bash "$ROOT/v2/stress/resume_round.sh" --days 37 >> "$STRESS/auto-round.log" 2>&1 &
        sleep 90
      else
        log "已过续跑线 01:35 → 轮次保持停止, 触发器退出(明晨 resume)"
        exit 0
      fi
    fi
  fi

  # ---- 轮自然完成检测 ----
  if [ "$ENGINE_STARTED" -eq 1 ] && ! engine_up && ! wrapper_up; then
    sleep 60
    if ! engine_up && ! wrapper_up; then
      log "轮次自然完成, 触发器退出(余未筛 $N 条留给下个检查点)"
      exit 0
    fi
  fi
done
