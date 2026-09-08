#!/bin/bash
# auto_round.sh — ★2026-09-03 自动压测轮 wrapper（供 automation 调用）
#
# 设计要点（踩坑教训内建）：
#   ① automation 会话结束会杀其后台子进程（9/2 day7 崩溃根因）→ 本脚本内部 nohup 脱离会话
#   ② 轮完成后【自动恢复 8100 day-model】（9/2 教训: automation 只停不恢复）
#   ③ 每轮结果自动 append auto-results.jsonl（共情链校准的持续数据源）
#   ④ reset 在轮前执行（归档上一轮，防跨轮污染）
#
# 用法: ./v2/stress/auto_round.sh [--days 33]   （通常由 automation 以 nohup 调起）
cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
DAY_LIMIT=33
[ "$1" = "--days" ] && DAY_LIMIT="$2"
DENSITY="${DENSITY:-1}"   # ★2026-09-09 fix: L37 '"$DENSITY"' 由本 shell 展开, 必须先在此定默认值
                          #   (原默认只在 L25 env 前缀里 → 未显式传 DENSITY 时 --density 展开为空, 引擎 arg error rc=2)

# 安全等待: 若有压测在跑则退出（不叠轮）
if pgrep -f "stress_engine.py" >/dev/null 2>&1; then
  echo "[auto_round] 已有压测在跑，跳过本次" >> "$ROOT/experiments/run/stress/auto-round.log"
  exit 2
fi

# 内部轮进程（nohup 脱离会话，automation 结束不杀）
# ★2026-09-03 fix: DAY_LIMIT 需以 env 注入内层 bash（原 L41 python 段 $DAY_LIMIT 内层展开为空
#   → 轮完成 append 时 SyntaxError，auto-results.jsonl 丢行）
DAY_LIMIT="$DAY_LIMIT" DENSITY="${DENSITY:-1}" GRACE_NO_DIALOGUE="${GRACE_NO_DIALOGUE:-0}" nohup bash -c '
  ROOT="'"$ROOT"'"
  cd "$ROOT"
  echo "=== auto_round 启动 $(date +%H:%M:%S) | days="'"'"$DAY_LIMIT"'"'" ===" >> experiments/run/stress/auto-round.log
  # ① 归档上一轮
  ./run.sh .venv/bin/python3 v2/stress/reset_stress.py >> experiments/run/stress/auto-round.log 2>&1
  # ② ★2026-09-08 8100 全程在线(输入方式对齐正式系统): ToM 判断/judge 重标/cog 重构
  #    全走 8100 独立 27B(认知器官), 压测进程内 V6.1 只管对话人格与训练 subprocess。
  #    内存: 循环期 15.5+15.5=31G / 训练期(_release_model 后) 20.7+15.5=36.2G, 均 <48G。
  #    (原"停 8100"已废——那是为了训练独占, 现训练前 _release_model 已保证)
  # ③ 跑 33 天轮（EWC-B）
  GRACE_EWC=1 GRACE_SLEEP_DELTA=0.03 ./run.sh .venv/bin/python3 v2/stress/stress_engine.py \
      --days "'"$DAY_LIMIT"'" --density "'"$DENSITY"'" --train-every 1 --sample-every 11 --reset-interval 0 --inputs-dir inputs-v3 \
      >> experiments/run/stress/stress.log 2>&1
  RC=$?
  # ④ 结果汇总 append（供共情链校准自动取数）
  .venv/bin/python3 -c "
import json, os, glob, time
root = \".\"
fps = glob.glob(root + \"/experiments/run/stress/archive/*/logs/round-meta.json\")
row = {\"ts\": time.strftime(\"%Y-%m-%dT%H:%M:%S\"), \"rc\": $RC, \"days\": $DAY_LIMIT}
final = root + \"/experiments/run/stress/final.json\"
# ★2026-09-05 修(21:16 发现): 异常退出时 final.json 是上一轮旧文件 → 统计沿用旧值误导航。
#   rc!=0 不读 final(标 abnormal), 只有正常完成才带统计。
if $RC == 0 and os.path.isfile(final):
    f = json.load(open(final))
    row.update({\"elapsed_s\": f.get(\"elapsed_s\"), \"trained_n\": len(f.get(\"trained\", [])),
                \"ok\": sum(1 for t in f.get(\"trained\", []) if t.get(\"ok\")),
                \"breakpoints\": len(f.get(\"samples\", []))})
elif $RC != 0:
    row[\"note\"] = \"abnormal exit, stats skipped\"
if fps:
    rm = json.load(open(sorted(fps)[-1]))
    rids = list(rm.get(\"rounds\", {}).keys())
    if rids:
        row[\"round\"] = rids[-1]
out = root + \"/experiments/run/stress/auto-results.jsonl\"
with open(out, \"a\", encoding=\"utf-8\") as fh:
    fh.write(json.dumps(row, ensure_ascii=False) + \"\n\")
print(\"  [auto_round] 结果已 append:\", json.dumps(row, ensure_ascii=False), flush=True)
" >> experiments/run/stress/auto-round.log 2>&1
  # ⑤ 恢复 8100（白天系统要用）
  launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.local-ai-agent.day-model.plist 2>/dev/null
  echo "=== auto_round 完成 $(date +%H:%M:%S) rc=$RC（8100 已恢复）===" >> experiments/run/stress/auto-round.log
  exit $RC
' </dev/null >> "$ROOT/experiments/run/stress/auto-round.log" 2>&1 &
echo "[auto_round] 后台轮已启动 pid=$! | 日志 experiments/run/stress/auto-round.log"
exit 0
