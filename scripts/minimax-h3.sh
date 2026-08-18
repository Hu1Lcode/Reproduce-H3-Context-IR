# 8 卡：NPU ID 0-3（Phy 0-7），全部空闲；Phy 12-13 被其他服务占用需避开
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn 
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export MINDIE_SD_FA_TYPE=ascend_laser_attention

N_NPUS=8

vllm serve /home/wjh/models/MiniMax-H3/FL2VA \
  --omni \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --num-gpus $N_NPUS \
  --init-timeout 1800 \
  --stage-init-timeout 1800 \
  --usp $N_NPUS \
  --ring 1 \
  --use-hsdp \
  --hsdp-shard-size $N_NPUS \
  --text-encoder-tp-size $N_NPUS \
  --vae-patch-parallel-size $N_NPUS \
  --vae-parallel-mode tile \
  --vae-use-tiling
