#!/usr/bin/env bash
# 白天模型 OpenAI 兼容服务：Qwen3.8-27B-4bit (MLX)
# 由 dsh job / launchd 按需拉起；常驻占用 ~15.5GB 统一内存（48GB 机体充裕）。
# 仅监听 127.0.0.1（本地主权，不外暴露）。
# 2026-08-18 修复：HF_HUB_OFFLINE=1 + 清除代理变量 —— mlx_lm 在 chat 时会做
# snapshot_download 健康检查，若环境有代理（如本机 127.0.0.1:62846 死代理）会撞
# 502 并把整个 chat 接口卡死（这正是 M3 headless「卡死」的根因，非 dsh 问题）。
# 2026-08-20 修复（复杂问题空回复/响应慢）：Qwen3.8 是 thinking 模型，/no_think 对
# MLX 无效，复杂问题思考吃光 max_tokens → content 空 + 160s 慢响应。
# 社群标准解法：--chat-template-args '{"enable_thinking": false}' 硬关思考；
# 附带 Qwen 官方采样参数（temp 0.6/top-p 0.95/top-k 20）+ prompt cache（减长 prompt
# 预填充开销，记忆注入后每轮 ~3K tokens）。
# 2026-08-20 追加：--max-tokens 8192 —— mlx_lm server 默认仅 512，不带 max_tokens 的
# 调用方会被截断到 ~950 字（实测）；8192 覆盖所有正常长回答（实测 2533 tokens 自然结束）。
# 2026-08-20 M7 追加：persona 模式 —— 用法 serve_day.sh persona [port]：挂 Megumin LoRA
# adapter（0000500 checkpoint，盲测甜点区：300 欠拟合 / 500 人设完整 / 1000 过拟合复读）。
# 惠惠实例独立端口（默认 8101），与 :8100 本地AI代理并存；persona 对话走 persona/ 子树，
# 永不进 L0-L3 事实记忆（草案 §5 Q34，mode=persona 隔离）。
set -euo pipefail
export HF_HUB_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy 2>/dev/null || true
VENV="/Users/cz/.workbuddy/binaries/python/envs/mlx-lm/bin/python"
MODE="${1:-normal}"
PORT="${2:-8100}"
ADAPTER="/Users/cz/WorkBuddy/skills find and make/local-ai-agent/models/megumin-lora/adapters/megumin_v6/0000500_adapters.safetensors"
if [ "$MODE" = "persona" ]; then
  PORT="${2:-8101}"
  exec "$VENV" -m mlx_lm server \
    --model mlx-community/Qwen3.8-27B-4bit \
    --host 127.0.0.1 --port "$PORT" \
    --max-tokens 8192 --chat-template-args '{"enable_thinking": false}' \
    --temp 0.6 --top-p 0.95 --top-k 20 \
    --prompt-cache-size 8 --prompt-cache-bytes 4GB \
    --adapter-path "$ADAPTER"
fi
# ★2026-09-07 grace 模式（集成方案: 权重回流·单实例时间复用）——进 /grace 切雷姆 V6.1
#   完整融合权重（fused-rem-v61 = Qwen3.8-27B 底模 + 想后说 LoRA 融合），离开自动切回 normal。
if [ "$MODE" = "grace" ]; then
  exec "$VENV" -m mlx_lm server \
    --model /Users/cz/WorkBuddy/watch/rem-v6-lora/models/fused-rem-v61 \
    --host 127.0.0.1 --port "${2:-8100}" \
    --max-tokens 8192 --chat-template-args '{"enable_thinking": false}' \
    --temp 0.6 --top-p 0.95 --top-k 20 \
    --prompt-cache-size 8 --prompt-cache-bytes 4GB
fi
exec "$VENV" -m mlx_lm server \
  --model mlx-community/Qwen3.8-27B-4bit \
  --host 127.0.0.1 --port "$PORT" \
  --max-tokens 8192 --chat-template-args '{"enable_thinking": false}' \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --prompt-cache-size 8 --prompt-cache-bytes 4GB
