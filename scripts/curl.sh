#!/usr/bin/env bash
# =============================================================================
# H3-Context-IR 服务调用 Demo（T2VA 文生视频）
#
# 用法：
#   bash scripts/curl.sh                     # 默认连 http://127.0.0.1:8888
#   API_BASE=http://127.0.0.1:8888 bash scripts/curl.sh
#   TOKEN=xxx bash scripts/curl.sh           # 官方云端 API 也可用（可选）
#
# 流程：POST 创建任务 → 轮询状态 → 输出最终 Prompt
# =============================================================================
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:8888}"
TOKEN="${TOKEN:-}"

# 请求头（本地服务无需 token；设置 TOKEN 时自动带上，兼容官方 API）
HEADERS=('--header' 'Content-Type: application/json')
if [ -n "$TOKEN" ]; then
  HEADERS+=('--header' "Authorization: Bearer $TOKEN")
fi

echo "=== 1. 创建任务（POST $API_BASE/v2/h3_context_ir）==="
RESP=$(curl --silent --show-error \
    --request POST \
    --url "$API_BASE/v2/h3_context_ir" \
    "${HEADERS[@]}" \
    --data '{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as the last fleet gathers and jumps away in a blinding flash, the bridge shaking, leaving her behind."
    }
  ],
  "duration": 10,
  "ratio": "16:9"
}')
echo "$RESP"

TASK_ID=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['task_id'])")
echo ""
echo "task_id: $TASK_ID"

echo ""
echo "=== 2. 轮询任务状态（每 15 秒）==="
while true; do
  QUERY=$(curl --silent --show-error \
      --url "$API_BASE/v2/query/video_generation/$TASK_ID")
  STATUS=$(echo "$QUERY" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  echo "  status=$STATUS"
  [ "$STATUS" = "success" ] && break
  [ "$STATUS" = "failed" ] && break
  sleep 15
done

echo ""
echo "=== 3. 最终 Prompt（Context-IR 输出）==="
echo "$QUERY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if d['status'] == 'success':
    print(d['content']['prompt'])
else:
    print('任务失败:', d.get('error'))
    sys.exit(1)
"
