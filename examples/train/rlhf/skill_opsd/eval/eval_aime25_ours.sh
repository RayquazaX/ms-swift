#!/bin/bash
# eval_aime25_ours.sh
# 评测 Ours (Skill-OPSD) 在 AIME25 上的 greedy pass@1
# TODO: 待 G 实验（Skill-OPSD 训练）完成后，填入 ADAPTER 路径
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

# TODO: set ADAPTER=<ours ckpt> 等 G 实验完成后填
#       e.g. ADAPTER=/data/qzheng19/opsd_repro/training_output/qwen3-4b-skill/vX-.../checkpoint-100
ADAPTER=${ADAPTER:-}
if [ -z "$ADAPTER" ]; then
    echo "ERROR: ADAPTER 未设置。请在脚本头或通过环境变量 ADAPTER=... 指定 Skill-OPSD checkpoint 路径"
    echo "       当前为占位脚本，等 G 实验完成后再跑"
    exit 1
fi
if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: Skill-OPSD ckpt not found at $ADAPTER"
    exit 1
fi

OUT=$WS/eval/logs/aime25_ours_results
mkdir -p "$OUT"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "[eval_aime25_ours] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[eval_aime25_ours] ADAPTER=$ADAPTER"
echo "[eval_aime25_ours] OUT=$OUT"

swift eval \
    --model Qwen/Qwen3-4B \
    --adapters "$ADAPTER" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset aime25 \
    --vllm_max_lora_rank 64 \
    --eval_output_dir "$OUT" \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'

echo "[eval_aime25_ours] DONE. 请更新 eval/leaderboard_aime25.md"
