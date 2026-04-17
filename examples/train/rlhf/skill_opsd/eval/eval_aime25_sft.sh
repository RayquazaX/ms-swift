#!/bin/bash
# eval_aime25_sft.sh
# 评测 SFT baseline 100-step LoRA ckpt 在 AIME25 上的 greedy pass@1
# 需先跑 train_sft_baseline.sh 生成 checkpoint
set -euo pipefail

: "${CONDA_ENV:=opsd}"
export BASH_ENV=/usr/local/anaconda3/etc/profile.d/conda.sh
source "$BASH_ENV" 2>/dev/null || true
conda activate "$CONDA_ENV"

: "${MODELSCOPE_CACHE:=/data/qzheng19/opsd_repro/modelscope_cache}"
export MODELSCOPE_CACHE

cleanup() { pkill -f "VLLM::EngineCore" 2>/dev/null || true; ray stop --force >/dev/null 2>&1 || true; }
trap cleanup EXIT

WS=/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd
CKPT_PARENT=${CKPT_PARENT:-$WS/eval/output/sft_baseline}

# 自动定位最新 checkpoint-100
ADAPTER=$(ls -d "$CKPT_PARENT"/v*/checkpoint-100 2>/dev/null | sort -V | tail -n1 || true)
if [ -z "$ADAPTER" ]; then
    ADAPTER=$(ls -d "$CKPT_PARENT"/checkpoint-100 2>/dev/null | tail -n1 || true)
fi
if [ -z "$ADAPTER" ] || [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: no SFT ckpt found at $CKPT_PARENT (looked for v*/checkpoint-100 and checkpoint-100)"
    echo "       请先运行 train_sft_baseline.sh 生成 checkpoint"
    exit 1
fi

OUT=$WS/eval/logs/aime25_sft_results
mkdir -p "$OUT"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "[eval_aime25_sft] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[eval_aime25_sft] ADAPTER=$ADAPTER"
echo "[eval_aime25_sft] OUT=$OUT"

swift eval \
    --model Qwen/Qwen3-4B \
    --adapters "$ADAPTER" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset aime25 \
    --vllm_max_lora_rank 64 \
    --eval_output_dir "$OUT" \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'

echo "[eval_aime25_sft] DONE. adapter=$ADAPTER. 请更新 eval/leaderboard_aime25.md"
