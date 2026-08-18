#!/usr/bin/env bash
# =============================================================================
# MiniMax-H3 T2VA 批量文生视频脚本
#
# 读取 enhanced_prompts.txt（每 5 行为一个 prompt 块：
#   "序号,integrated_multimodal_description: ..." / 空行 /
#   "overall_soundscape: ..." / 空行 / "non_diegetic_music: ..."），
# 逐条调用本机 vllm-omni MiniMax-H3 服务的 /v1/videos/sync 接口，
# 输出的 MP4 保存到 outputs/t2va/ 目录（文件名按原序号，如 t2va_000.mp4）。
#
# 用法：
#   bash scripts/minimax-h3-t2va.sh                      # 全部 54 条
#   bash scripts/minimax-h3-t2va.sh 0 5 10               # 只跑指定序号
#   LIMIT=3 bash scripts/minimax-h3-t2va.sh              # 只跑前 3 条
#   SKIP_EXISTING=0 bash scripts/minimax-h3-t2va.sh      # 覆盖已生成的文件
#
# 环境变量可覆盖：API_URL / PROMPTS_FILE / OUTPUT_DIR / WIDTH / HEIGHT /
#   FPS / STEPS / FLOW_SHIFT / DURATION / BASE_SEED / LIMIT / SKIP_EXISTING
# =============================================================================
set -uo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000/v1/videos/sync}"
PROMPTS_FILE="${PROMPTS_FILE:-/home/wjh/Reproduce-H3-Context-IR/enhanced_prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/wjh/Reproduce-H3-Context-IR/outputs/t2va}"
WIDTH="${WIDTH:-1344}"
HEIGHT="${HEIGHT:-768}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
FPS="${FPS:-24}"
STEPS="${STEPS:-50}"
FLOW_SHIFT="${FLOW_SHIFT:-12}"
DURATION="${DURATION:-5}"
BASE_SEED="${BASE_SEED:-1000}"
LIMIT="${LIMIT:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CURL_TIMEOUT="${CURL_TIMEOUT:-1800}"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run.log"

# ---- 1. 解析 prompt 文件 → TSV（序号<TAB>prompt 文本） ----
# prompt 文本 = integrated_multimodal_description + overall_soundscape +
#               non_diegetic_music 三字段（去掉行首 "序号," 前缀，字段间空行保留）
TSV_FILE="$(mktemp)"
python3 - "$PROMPTS_FILE" "$TSV_FILE" <<'PYEOF'
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as f:
    lines = f.read().splitlines()

blocks = []
cur = None
for ln in lines:
    m = re.match(r"^(\d+),", ln)
    if m:
        if cur is not None:
            blocks.append(cur)
        cur = {"idx": int(m.group(1)), "body": [ln[m.end():]]}
    elif cur is not None:
        cur["body"].append(ln)
if cur is not None:
    blocks.append(cur)

# 完整性校验：过滤空行后必须恰好三个字段，按规范格式重组
# （integrated_multimodal_description / overall_soundscape / non_diegetic_music，
#   三行之间用空行隔开，否则模型无法正确解析字段边界）
bad = []
for b in blocks:
    fields = [ln.strip() for ln in b["body"] if ln.strip()]
    if len(fields) != 3:
        bad.append((b["idx"], len(fields)))
if bad:
    sys.stderr.write(f"WARNING: blocks with unexpected field count: {bad}\n")

with open(dst, "w", encoding="utf-8") as f:
    for b in blocks:
        fields = [ln.strip() for ln in b["body"] if ln.strip()]
        if len(fields) != 3:
            sys.stderr.write(f"WARNING: idx {b['idx']} has {len(fields)} fields (expected 3), skipped\n")
            continue
        # 规范格式：三行 + 空行分隔
        text = "\n\n".join(fields).replace("\n", "\\n")
        if "\t" in text:
            sys.stderr.write(f"WARNING: idx {b['idx']} contains TAB, skipped\n")
            continue
        f.write(f"{b['idx']}\t{text}\n")
sys.stderr.write(f"parsed {len(blocks)} prompts -> {dst}\n")
PYEOF

# ---- 2. 筛选要跑的序号（命令行参数优先，其次 LIMIT） ----
SELECTED_IDX=()
if [ "$#" -gt 0 ]; then
    SELECTED_IDX=("$@")
else
    while IFS=$'\t' read -r idx _; do
        SELECTED_IDX+=("$idx")
        if [ -n "$LIMIT" ] && [ "${#SELECTED_IDX[@]}" -ge "$LIMIT" ]; then
            break
        fi
    done < "$TSV_FILE"
fi

echo "==========================================================" | tee -a "$LOG_FILE"
echo "T2VA batch start: $(date '+%F %T')  target=$API_URL  output=$OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "total to run: ${#SELECTED_IDX[@]}  (steps=$STEPS duration=${DURATION}s ${WIDTH}x${HEIGHT})" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

# ---- 3. 逐条请求 ----
OK=0; FAIL=0; SKIPPED=0
for idx in "${SELECTED_IDX[@]}"; do
    # 只取最后一个匹配行（防止文件存在重复条目时 prompt 被拼接）
    prompt="$(awk -F'\t' -v i="$idx" '$1==i {sub(/^[^\t]*\t/,""); last=$0} END {if (last != "") print last}' "$TSV_FILE" | sed 's/\\n/\n/g')"
    if [ -z "$prompt" ]; then
        echo "[$idx] SKIP: not found in prompts file" | tee -a "$LOG_FILE"
        continue
    fi

    out="$OUTPUT_DIR/t2va_$(printf '%03d' "$idx").mp4"
    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$out" ]; then
        echo "[$idx] SKIP: $out already exists" | tee -a "$LOG_FILE"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    seed=$((BASE_SEED + idx))
    echo "[$idx] $(date '+%F %T') generating -> $out (seed=$seed)" | tee -a "$LOG_FILE"

    # 同步生成，带重试
    rc=1
    for attempt in 1 2 3; do
        curl -sS --max-time "$CURL_TIMEOUT" -X POST "$API_URL" \
            --form-string "prompt=$prompt" \
            -F "width=$WIDTH" \
            -F "height=$HEIGHT" \
            -F "aspect_ratio=$ASPECT_RATIO" \
            -F "fps=$FPS" \
            -F "num_inference_steps=$STEPS" \
            -F "flow_shift=$FLOW_SHIFT" \
            -F "seed=$seed" \
            --form-string 'extra_params={"task":"t2va","duration":'"$DURATION"',"audio_flow_shift":3.0}' \
            -o "$out" \
            && [ -s "$out" ] \
            && rc=0 && break
        echo "[$idx] attempt $attempt failed (rc=$rc), retrying..." | tee -a "$LOG_FILE"
        sleep 5
    done

    if [ "$rc" -eq 0 ]; then
        OK=$((OK+1))
        echo "[$idx] DONE $(date '+%F %T') $(stat -c%s "$out" 2>/dev/null || echo 0) bytes" | tee -a "$LOG_FILE"
    else
        FAIL=$((FAIL+1))
        echo "[$idx] FAILED after 3 attempts" | tee -a "$LOG_FILE"
        rm -f "$out"
    fi
done

echo "==========================================================" | tee -a "$LOG_FILE"
echo "T2VA batch finished: $(date '+%F %T')  OK=$OK FAIL=$FAIL SKIPPED=$SKIPPED" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

rm -f "$TSV_FILE"
exit $((FAIL > 0))
