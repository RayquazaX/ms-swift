#!/bin/bash
# train_sft_baseline.sh
# SFT baseline: Qwen3-4B + LoRA r64a128, 100 steps on OpenThoughts-114k-math
# 用于构造 AIME25 leaderboard 的 SFT 行
# 备注：
#   - 不需要 --external_plugins（只训 base，不读 teacher_prompt）
#   - 不需要 vllm（普通 SFT）
#   - --report_to tensorboard wandb （纪律要求）
#   - 独立 WANDB_RUN_NAME，避免 tag 冲突
set -euo pipefail

: "${CONDA_ENV:=opsd}"
export BASH_ENV=/usr/local/anaconda3/etc/profile.d/conda.sh
source "$BASH_ENV" 2>/dev/null || true
conda activate "$CONDA_ENV"

: "${MODELSCOPE_CACHE:=/data/qzheng19/opsd_repro/modelscope_cache}"
export MODELSCOPE_CACHE

# --- wandb 纪律 ---
export WANDB_PROJECT=${WANDB_PROJECT:-opsd-repro}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-qwen3-4b-sft100-baseline}
: "${WANDB_DISABLE_STATS:=true}"
export WANDB_DISABLE_STATS

cleanup() { pkill -f "VLLM::EngineCore" 2>/dev/null || true; ray stop --force >/dev/null 2>&1 || true; }
trap cleanup EXIT

WS=/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd
OUT=${OUT:-$WS/eval/output/sft_baseline}
mkdir -p "$OUT"

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${NPROC_PER_NODE:=8}"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE

echo "[train_sft_baseline] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[train_sft_baseline] WANDB_PROJECT=$WANDB_PROJECT  WANDB_RUN_NAME=$WANDB_RUN_NAME"
echo "[train_sft_baseline] OUT=$OUT"

cd /home/qzheng19/ms-swift

swift sft \
    --model Qwen/Qwen3-4B \
    --dataset 'open-r1/OpenThoughts-114k-math' \
    --train_type lora \
    --lora_rank 64 --lora_alpha 128 --target_modules all-linear \
    --torch_dtype bfloat16 \
    --max_steps 100 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --max_length 8192 \
    --save_only_model true \
    --gradient_checkpointing true \
    --deepspeed zero0 \
    --attn_impl flash_attn \
    --output_dir "$OUT" \
    --report_to tensorboard wandb

echo "[train_sft_baseline] DONE. ckpt under $OUT"
echo "[train_sft_baseline] 下一步：bash eval/eval_aime25_sft.sh"
