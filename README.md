# Reproduce-H3-Context-IR

MiniMax H3-Context-IR 模块的复现实现（API 版，零 GPU 部署）。

> 依据《H3-Context-IR 复现方案》（context-ir-plan.html）与官方一手资料实现：
> - `references/base-en.txt` —— Video Prompt Writing Guide (T2VA/I2VA/FL2VA/L2VA) 官方原文
> - `references/ref-en.txt` —— Full-Reference Mode Rewrite Output Format Guide 官方原文
> - `references/SKILL.md` —— 官方 skills/h3-prompt-writing 摘要

## 架构：6 步流水线

```
Context-IR 输入 (prompt + 图像 + 视频 + 音频)
  → Step 0 任务区分（规则引擎，本地）        t2va / i2va / l2va / fl2va / ref2va
  → 本地预处理器（CPU）                    帧采样 / 场景切分 / 音色 / BPM / 立体声
  → Qwen3-Omni-Captioner API               音频语义理解（转写/情绪/环境声/音乐风格）
  → Step 1 语义提取 (qwen3.6-plus API)      prompt + 素材 → 结构化语义 JSON
  → Step 2 跨模态关联 (qwen3.6 API)         <Subject N> / <Audio N> 标签与映射
  → Step 3 Shot 分割与描述 (deepseek-v4-flash API)   镜头时间线
  → Step 4 逻辑校验 (deepseek-v4-flash API) 修正建议 + retention 分析
  → Step 5 格式化输出                      官方模板确定性渲染 + 可选 LLM 润色
  → 最终结构化 prompt（直接送入 H3-Base）
```

## 配置

**推荐方式：YAML 配置文件**（`config/config.yaml`，修改后重启服务即生效）：

```yaml
providers:
  local:
    base_url: "http://127.0.0.1:8000/v1"   # ← 本地 vLLM 服务地址
models:
  step1:     { provider: local, model: "Qwen3.6-35B-A3B" }
  step2:     { provider: local, model: "Qwen3.6-35B-A3B" }
  step3:     { provider: local, model: "Qwen3.6-35B-A3B" }
  step4:     { provider: local, model: "Qwen3.6-35B-A3B" }
  step5:     { provider: local, model: "Qwen3.6-35B-A3B" }
  captioner: { provider: local, model: "Qwen3-Omni-30B-A3B-Captioner" }
```

- 配置优先级：代码默认值 < `config/config.yaml` < 环境变量（临时覆盖）
- 可用 `H3C_CONFIG=/path/to.yaml` 指定其他配置文件
- provider 可自定义（新增条目即可，如 `myvllm: {base_url: ...}`）
- 模型名须与 vLLM 的 `--served-model-name` 一致（未指定则用模型目录名）

**环境变量方式**（临时切换，优先级高于 YAML）：

| 环境变量 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 阿里云百炼 key（Step 1/2 + Captioner） |
| `DEEPSEEK_API_KEY` | DeepSeek key（Step 3/4/5） |
| `LOCAL_OPENAI_BASE_URL` | 本地 OpenAI 兼容端点 |
| `H3C_STEP1_MODEL` ... `H3C_STEP5_MODEL` | 按步覆盖模型，格式 `provider:model` |
| `H3C_CAPTIONER_MODEL` | 音频语义模型 |
| `H3C_LLM_POLISH=1` | 开启 Step 5 LLM 润色（默认确定性拼接） |
| `H3C_CACHE=0` | 关闭磁盘缓存 |

## 一、启动服务

### 1.1 前置条件

```bash
# 1. 安装依赖（容器内已有 fastapi/uvicorn/openai 时可跳过）
pip install -r requirements.txt

# 2. 检查 config/config.yaml 中的模型端点是否正确
#    （默认指向本地 vllm-omni-021 容器：qwen3.6 → :8111，qwen3-omni → :8112）
#    云端 API 则在对应 provider 配 api_key

# 3. 确认本地推理服务已就绪
curl http://127.0.0.1:8111/v1/models   # qwen3.6
curl http://127.0.0.1:8112/v1/models   # qwen3-omni
```

### 1.2 一键脚本启动（推荐）

```bash
./scripts/start_server.sh start       # 后台启动（默认端口 8888，读 config.yaml）
./scripts/start_server.sh status      # 查看状态 + 健康检查
./scripts/start_server.sh logs        # 实时跟随后台日志（Ctrl+C 退出查看，不影响服务）
./scripts/start_server.sh foreground  # 前台启动，日志实时打印，Ctrl+C 停止服务
./scripts/start_server.sh restart     # 重启（改完 config/config.yaml 后用它）
./scripts/start_server.sh stop        # 停止（别名: kill / down / off）
```

- **宿主机 / 容器通用**：脚本会自动检测环境——在宿主机执行时自动 `docker exec` 转发到容器（默认 `vllm-omni-021`，可用环境变量 `H3C_CONTAINER` 覆盖）
- **端口**：默认 8888，来自 `config/config.yaml` 的 `runtime.server_port`；临时覆盖用 `SERVER_PORT=9000 ./scripts/start_server.sh start`
- **启动自检**：启动后自动等待 `/health` 就绪（最多 15 秒），失败会打印日志尾部

### 1.3 手动启动（容器内）

```bash
cd /home/wjh/Reproduce-H3-Context-IR
uvicorn server.server:app --host 0.0.0.0 --port 8888        # 前台
nohup uvicorn server.server:app --host 0.0.0.0 --port 8888 > work/server.log 2>&1 &  # 后台
```

### 1.4 验证启动

```bash
curl http://127.0.0.1:8888/health
# → {"status":"ok"}
```

---

## 二、调用服务

### 2.1 调用流程（两步：创建任务 → 轮询结果）

服务采用**异步任务语义**（与官方 Context-IR API 一致）：

```
POST /v2/h3_context_ir                        → {"task_id": "..."}
GET  /v2/query/video_generation/{task_id}     → 任务状态 + 最终 prompt
```

### 2.2 五种任务类型请求示例

**T2VA（文生视频）**——只有文本：

```bash
curl -X POST http://127.0.0.1:8888/v2/h3_context_ir \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-H3",
    "content": [{"type": "text", "text": "一只白猫在窗台上看雨，温馨的电影感画面"}],
    "duration": 6,
    "ratio": "16:9"
  }'
```

**I2VA（首帧生视频）**——文本 + 1 张图（`role: first_frame`）：

```bash
curl -X POST http://127.0.0.1:8888/v2/h3_context_ir \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-H3",
    "content": [
      {"type": "text", "text": "基于这张首帧图，镜头缓慢推进，画面自然运动起来"},
      {"type": "image_url", "url": "http://127.0.0.1:9080/ltx2.3-open.png", "role": "first_frame"}
    ],
    "duration": 6,
    "ratio": "16:9"
  }'
```

**L2VA（末帧生视频）**——文本 + 1 张图（`role: last_frame`）：

```bash
# 与 I2VA 相同结构，role 改为 "last_frame"
```

**FL2VA（首末帧生视频）**——文本 + 2 张图：

```bash
"content": [
  {"type": "text", "text": "从第一帧演化到最后一帧的完整动作路径"},
  {"type": "image_url", "url": ".../first.png", "role": "first_frame"},
  {"type": "image_url", "url": ".../last.png",  "role": "last_frame"}
]
```

**Ref2VA（全参考模式）**——文本 + 参考视频/音频/图（`role: reference_*`）：

```bash
curl -X POST http://127.0.0.1:8888/v2/h3_context_ir \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-H3",
    "content": [
      {"type": "text", "text": "参考视频中的角色在咖啡店里吃饼干"},
      {"type": "video_url", "url": "http://127.0.0.1:9080/ref.mp4", "role": "reference_video"},
      {"type": "audio_url", "url": "http://127.0.0.1:9080/voice.wav", "role": "reference_audio"}
    ],
    "duration": 5,
    "ratio": "adaptive"
  }'
```

### 2.3 查询结果（轮询）

```bash
TASK_ID=<上一步返回的 task_id>

curl http://127.0.0.1:8888/v2/query/video_generation/$TASK_ID
```

```json
{
  "task_id": "4fa42c53-...",
  "status": "success",
  "content": {
    "prompt": "For the target video, at 0.00 seconds...",   // ← 最终结构化 Prompt（Context-IR 输出）
    "task_type": "i2va",
    "validation": {"ok": true, "error_count": 0, "warning_count": 0, "issues": []}
  },
  "error": null
}
```

**status 流转**：`pending` → `processing` → `success` / `failed`。任务一般耗时 1-3 分钟（5 个步骤 × 本地推理），建议每 10-15 秒轮询一次。结果同时落盘到 `work/tasks/{task_id}.json`。

### 2.4 Python 调用示例

```python
import requests, time

def call_context_ir(content, duration=6, ratio="16:9", base="http://127.0.0.1:8888"):
    # 创建任务
    r = requests.post(f"{base}/v2/h3_context_ir", json={
        "model": "MiniMax-H3", "content": content, "duration": duration, "ratio": ratio,
    })
    task_id = r.json()["task_id"]
    # 轮询
    while True:
        r = requests.get(f"{base}/v2/query/video_generation/{task_id}")
        data = r.json()
        if data["status"] != "processing":
            break
        time.sleep(15)
    if data["status"] == "success":
        return data["content"]["prompt"]
    raise RuntimeError(data.get("error"))

# T2VA 示例
prompt = call_context_ir([{"type": "text", "text": "一只白猫在窗台上看雨"}])
print(prompt)
```

### 2.5 注意事项

1. **素材必须 URL 化**：`url` 字段必须是 vLLM 能访问的 HTTP 地址，不能用本地路径。
   开发时可用内置静态服务（默认 9080 端口）暴露项目目录：
   ```bash
   python3 -m http.server 9080 --directory /home/wjh/Reproduce-H3-Context-IR --bind 0.0.0.0
   # 然后 url 写 http://127.0.0.1:9080/ltx2.3-open.png
   ```
   生产环境请使用对象存储/CDN。
2. **官方输入规格**会被校验（违规返回 400）：图像 ≤ 9、视频 ≤ 3、音频 ≤ 3、总文件 ≤ 12；视频/音频每段 2-15 秒；音频必须搭配图像或视频输入。
3. **模型是思考型**（qwen3.6）：每个步骤会先"思考"再回答，单步延迟约 10-30 秒属正常；超长思考导致空输出时会自动重试。
4. 服务默认端口 8888（`config/config.yaml` → `runtime.server_port`），改完用 `./scripts/start_server.sh restart` 生效。

### 2.6 CLI 直跑（不走服务，调试用）

```bash
python scripts/run_pipeline.py --prompt "一只猫在窗台上看雨" --duration 6 --out work/result.json
# 输出完整 PipelineContext（含各步骤中间结果 + final_prompt）到 work/result.json
```

## 项目结构

```
config/settings.py         全局配置（API keys / 模型 / provider）
config/prompts/            9 个 System Prompt 模板（step1-4 + step5 五套）
references/                官方指南原文（base-en.txt / ref-en.txt / SKILL.md）
preprocessor/              Step 0 任务区分 + 本地预处理（帧采样/场景/音频特征）
pipeline/                  PipelineContext + 6 步编排 + Step 1-5 实现
api/                       OpenAI 兼容客户端 / VL / Captioner / 上传器 / 输出校验
server/                    FastAPI 服务（兼容官方 /v2/h3_context_ir 异步任务语义）
evaluation/                硬性指标（格式合规率/retention 词表/时间戳越界等）+ 对比
scripts/                   run_pipeline.py / collect_official_samples.py / smoke_test.py
```

## 设计要点

- **统一 OpenAI 兼容客户端**（`api/base_client.py`）：DashScope / DeepSeek / 本地
  vLLM 均可切换，带重试/超时/限流/磁盘缓存。
- **Step 5 确定性模板渲染**：字段名、顺序、时间戳、指令行、retention 词表由代码
  保证（`pipeline/step5_formatter.py`），`output_validator.py` 正则兜底。
- **音频分层**：语义层（Captioner API：转写/情绪/环境声/音乐风格）+ 数值层
  （本地 librosa/SpeechBrain 可选、内置 numpy/torchaudio fallback：音色/BPM/立体声）。
- **可选依赖降级**：pyscenedetect / librosa / speechbrain / demucs 未安装时自动
  使用内置轻量实现，保证在最小环境可运行。
