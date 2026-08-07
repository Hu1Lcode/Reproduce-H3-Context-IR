export ASCEND_RT_VISIBLE_DEVICES=12,13

vllm serve /home/wjh/models/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --port 8111 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data