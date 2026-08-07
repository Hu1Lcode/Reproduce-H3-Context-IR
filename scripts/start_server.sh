#!/usr/bin/env bash
# =============================================================================
# H3-Context-IR 服务一键管理脚本
#
# 用法：
#   ./scripts/start_server.sh start      # 启动（后台运行）
#   ./scripts/start_server.sh stop       # 停止
#   ./scripts/start_server.sh restart    # 重启
#   ./scripts/start_server.sh status     # 查看状态
#   ./scripts/start_server.sh logs       # 实时查看日志（Ctrl+C 退出）
#   ./scripts/start_server.sh foreground # 前台启动，日志实时打印（Ctrl+C 停止）
#
# 说明：
#   - 服务运行在容器内（默认 vllm-omni-021，可用环境变量 H3C_CONTAINER 覆盖）；
#     在宿主机直接运行本脚本会自动通过 docker exec 转发到容器内执行
#   - 端口/host 自动读取 config/config.yaml 的 runtime.server_port / server_host
#   - 可用环境变量 SERVER_PORT 临时覆盖端口
#   - 日志写入 work/server.log，PID 记录在 work/server.pid
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# -----------------------------------------------------------------------------
# 容器自动适配：在宿主机运行时转发到容器内执行
# -----------------------------------------------------------------------------
CONTAINER_NAME="${H3C_CONTAINER:-vllm-omni-021}"
if [ ! -f /.dockerenv ] && ! grep -q "docker\|lxc\|kubepods" /proc/1/cgroup 2>/dev/null; then
  if command -v docker >/dev/null 2>&1; then
    echo "[h3c] 检测到宿主机环境，转发到容器 ${CONTAINER_NAME} 内执行 ..."
    exec docker exec "${CONTAINER_NAME}" bash "${SCRIPT_DIR}/start_server.sh" "$@"
  else
    echo "错误：宿主机没有 docker 命令，无法转发到容器执行" >&2
    exit 1
  fi
fi

cd "$PROJECT_ROOT"

WORK_DIR="$PROJECT_ROOT/work"
PID_FILE="$WORK_DIR/server.pid"
LOG_FILE="$WORK_DIR/server.log"

# -----------------------------------------------------------------------------
# 读取配置（config/config.yaml → 环境变量覆盖）
# -----------------------------------------------------------------------------
HOST="0.0.0.0"
PORT="8888"
if [ -f "$PROJECT_ROOT/config/config.yaml" ]; then
  read -r YAML_HOST YAML_PORT < <(
    python3 -c "
import yaml
try:
    d = yaml.safe_load(open('$PROJECT_ROOT/config/config.yaml'))
    rt = (d or {}).get('runtime', {}) or {}
    print(rt.get('server_host', '') or '', rt.get('server_port', '') or '')
except Exception:
    print('', '')
" 2>/dev/null || echo ""
  )
  [ -n "$YAML_HOST" ] && HOST="$YAML_HOST"
  [ -n "$YAML_PORT" ] && PORT="$YAML_PORT"
fi
# 环境变量最高优先级
[ -n "${SERVER_PORT:-}" ] && PORT="$SERVER_PORT"
[ -n "${SERVER_HOST:-}" ] && HOST="$SERVER_HOST"

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }

is_running() {
  if [ -f "$PID_FILE" ]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  # 兜底：按命令行查找
  pgrep -f "uvicorn server.server:app" >/dev/null 2>&1
}

get_pid() {
  if [ -f "$PID_FILE" ]; then
    cat "$PID_FILE" 2>/dev/null
  else
    pgrep -f "uvicorn server.server:app" 2>/dev/null | head -1 || true
  fi
}

get_all_pids() {
  pgrep -f "uvicorn server.server:app" 2>/dev/null || true
}

health_ok() {
  curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'
}

# -----------------------------------------------------------------------------
# 子命令
# -----------------------------------------------------------------------------
do_start() {
  if is_running; then
    yellow "服务已在运行 (PID $(get_all_pids | tr '\n' ',' | sed 's/,$//'))，无需重复启动"
    return 0
  fi
  mkdir -p "$WORK_DIR"
  echo "启动服务: uvicorn server.server:app --host $HOST --port $PORT"
  nohup uvicorn server.server:app --host "$HOST" --port "$PORT" \
    > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  # 等待就绪（最多 15 秒）
  for _ in $(seq 1 30); do
    if health_ok; then
      green "✔ 服务启动成功: http://$HOST:$PORT  (PID $(get_pid))"
      echo "  日志: $LOG_FILE"
      return 0
    fi
    sleep 0.5
  done
  red "✘ 服务启动超时，请查看日志: $LOG_FILE"
  tail -20 "$LOG_FILE" || true
  return 1
}

do_stop() {
  local pids
  pids="$(get_all_pids)"
  if [ -z "$pids" ]; then
    rm -f "$PID_FILE"
    yellow "服务未在运行"
    return 0
  fi
  echo "停止服务 (PID ${pids//$'\n'/, }) ..."
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 20); do
    if [ -z "$(get_all_pids)" ]; then
      rm -f "$PID_FILE"
      green "✔ 服务已停止"
      return 0
    fi
    sleep 0.5
  done
  yellow "进程未退出，强制终止 ..."
  pkill -9 -f "uvicorn server.server:app" 2>/dev/null || true
  rm -f "$PID_FILE"
  green "✔ 服务已强制停止"
}

do_status() {
  if is_running; then
    local pid
    pid="$(get_pid)"
    echo "● 服务运行中 (PID $pid, 端口 $PORT)"
    if health_ok; then
      echo "  健康检查: $(green OK)"
    else
      echo "  健康检查: $(red FAIL)（进程在但接口无响应）"
    fi
    echo "  日志: $LOG_FILE"
  else
    echo "○ 服务未运行"
  fi
}

do_foreground() {
  if is_running; then
    red "服务已在后台运行 (PID $(get_all_pids | tr '\n' ',' | sed 's/,$//'))，"
    red "请先执行 stop 停止，或直接使用 logs 查看实时日志"
    return 1
  fi
  mkdir -p "$WORK_DIR"
  rm -f "$PID_FILE"
  echo "前台启动: uvicorn server.server:app --host $HOST --port $PORT  (Ctrl+C 停止)"
  # exec 替换当前 shell，日志实时输出到终端
  exec uvicorn server.server:app --host "$HOST" --port "$PORT"
}

do_logs() {
  if [ ! -f "$LOG_FILE" ]; then
    yellow "日志文件不存在: $LOG_FILE"
    return 0
  fi
  tail -f "$LOG_FILE"
}

# -----------------------------------------------------------------------------
case "${1:-}" in
  start)       do_start ;;
  stop|kill|down|off) do_stop ;;
  restart)     do_stop && do_start ;;
  status)      do_status ;;
  logs)        do_logs ;;
  fg|foreground|run) do_foreground ;;
  *)
    echo "用法: $0 {start|stop|restart|status|logs|foreground}"
    echo "  start      后台启动服务"
    echo "  stop       停止服务（别名: kill / down / off）"
    echo "  restart    重启服务（改 config/config.yaml 后常用）"
    echo "  status     查看运行状态 + 健康检查"
    echo "  logs       实时跟随后台服务日志"
    echo "  foreground 前台启动，日志实时打印，Ctrl+C 停止（别名: fg）"
    exit 1
    ;;
esac
