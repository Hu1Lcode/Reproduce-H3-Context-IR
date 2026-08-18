export ASCEND_RT_VISIBLE_DEVICES=12,13

# model为 Qwen3.6-27B-H3-merged（已合并 H3 Prompt Rewriter LoRA）
vllm serve /home/wjh/models/Qwen3.6-27B-H3-merged \
  --trust-remote-code \
  --port 8111 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --served-model-name h3-prompt-rewriter # 保持 API 模型名不变，与 config.yaml 中一致