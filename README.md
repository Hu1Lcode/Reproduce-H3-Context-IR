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

## 使用

```bash
# 端到端运行（T2VA 示例）
python scripts/run_pipeline.py --prompt "一只猫在窗台上看雨" --duration 6 --out work/result.json

# I2VA（首帧生视频）
python scripts/run_pipeline.py --prompt "..." \
    --image ./first.jpg --image-role first_frame --duration 8

# Ref2VA（全参考）
python scripts/run_pipeline.py --prompt "..." \
    --video ./ref.mp4 --video-role reference_video \
    --audio ./voice.wav --audio-role reference_audio --duration 5

# 官方 content JSON 输入（与官方 API 请求格式一致）
python scripts/run_pipeline.py --content work/input.json

# 兼容官方异步接口的服务
uvicorn server.server:app --host 0.0.0.0 --port 8080
#   POST /v2/h3_context_ir  → {"task_id": "..."}
#   GET  /v2/query/video_generation/{task_id} → .task.content.prompt

# 收集官方示例输入（GitHub 仓库脚本中的公开素材 URL）
python scripts/collect_official_samples.py --out examples

# 评估：官方 vs 自建输出对比
python -m evaluation.compare --ours work/result.json --official official_result.json

# 回归测试
python scripts/smoke_test.py
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
