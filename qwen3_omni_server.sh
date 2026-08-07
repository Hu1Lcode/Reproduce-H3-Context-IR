export ASCEND_RT_VISIBLE_DEVICES=9,10,11


vllm serve /home/wjh/models/Qwen3-Omni-30B-A3B-Instruct --omni --port 8112 \
    --deploy-config /home/wjh/ltx/vllm-omni/vllm_omni/deploy/qwen3_omni_moe_multi_replicas.yaml