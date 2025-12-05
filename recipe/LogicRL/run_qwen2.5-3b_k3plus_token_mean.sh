#!/usr/bin bash
set -xeuo pipefail
# 环境变量
# export VLLM_ATTENTION_BACKEND=XFORMERS
export NVLS_ENABLE=1
export TORCH_CUDA_ALLOW_TF32=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# swanlab configuration
export SWANLAB_API_KEY="9l9xoXvSRy7jDqf09xQ3M"

python3 -m verl.trainer.main_ppo \
  --config-path="/root/verl/recipe/LogicRL/config" \
  --config-name=run_qwen2.5-3b_k3plus_token_mean