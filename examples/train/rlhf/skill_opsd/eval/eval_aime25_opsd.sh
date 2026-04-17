#!/bin/bash
# eval_aime25_opsd.sh
# 评测 OPSD baseline 100-step LoRA ckpt 在 AIME25 上的 greedy pass@1
# 注意：OPSD 结果已在 2026-04-08 预跑过（pass@1 = 0.2667）
#       本次不必重跑；如需 sanity check 或重复实验再跑
# 预计耗时：~5 min on 1 GPU
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

# OPSD baseline checkpoint (2026-04-08 run)
ADAPTER=${ADAPTER:-/data/qzheng19/opsd_repro/training_output/qwen3-4b/v0-20260408-062501/checkpoint-100}
if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: OPSD ckpt not found at $ADAPTER"
    exit 1
fi

OUT=$WS/eval/logs/aime25_opsd_results
mkdir -p "$OUT"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "[eval_aime25_opsd] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[eval_aime25_opsd] ADAPTER=$ADAPTER"
echo "[eval_aime25_opsd] OUT=$OUT"

swift eval \
    --model Qwen/Qwen3-4B \
    --adapters "$ADAPTER" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset aime25 \
    --vllm_max_lora_rank 64 \
    --eval_output_dir "$OUT" \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'

echo "[eval_aime25_opsd] DONE. 历史值 0.2667 (2026-04-08)"
