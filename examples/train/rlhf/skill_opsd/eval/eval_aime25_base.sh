#!/bin/bash
# eval_aime25_base.sh
# 评测 Qwen3-4B 基座（零后训练）在 AIME25 上的 greedy pass@1
# 预计耗时：~10 min on 1 GPU
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
OUT=$WS/eval/logs/aime25_base_results
mkdir -p "$OUT"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "[eval_aime25_base] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[eval_aime25_base] OUT=$OUT"

swift eval \
    --model Qwen/Qwen3-4B \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset aime25 \
    --eval_output_dir "$OUT" \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'

echo "[eval_aime25_base] DONE. 请更新 eval/leaderboard_aime25.md"
