#!/bin/bash
# run_phase3.sh  —  Skill-Informed In-Context Self-Distillation (Phase-3 training)
#
# Upgrades the OPSD baseline (examples/train/rlhf/opsd/opsd.sh, Task 1.3 version)
# to the Skill-Informed mode by swapping in:
#   * --external_plugins : Task 3.4's skill_opsd_plugin.py
#   * --dataset          : Task 3.3's train_with_context.jsonl  (pre-built layered
#                          teacher_prompt = skills + OPSD transition + question)
#   * --output_dir       : ckpt_phase3 (per Task 4.1 experiment isolation)
#
# All other training hyperparameters (lr, batch, lora_*, vllm, deepspeed, beta,
# temperature, top_p, top_k, jsd_token_clip, sft_alpha, flash_attn, etc.) are
# inherited verbatim from opsd.sh so the skill-informed run is a drop-in
# comparison to the baseline.
#
# ------------------------------------------------------------------
# CLI / env overrides (defaults follow config.yaml `training.*`)
# ------------------------------------------------------------------
#   CONDA_ENV=opsd                        activate this conda env
#   WORKSPACE=.../skill_opsd              skill_opsd working dir
#   MS_SWIFT_ROOT=/home/qzheng19/ms-swift ms-swift repo root
#   TRAIN_JSONL=${WORKSPACE}/train_with_context.jsonl
#   PLUGIN_PATH=${WORKSPACE}/skill_opsd_plugin.py
#   MODEL=Qwen/Qwen3-4B                   student + teacher base model
#   TEACHER_MODEL=${MODEL}                self-distillation (same model)
#   OUTPUT_DIR=${WORKSPACE}/ckpt_phase3   checkpoint / log output
#   MAX_STEPS=300                         config.yaml training.max_steps_phase3
#   TEMPERATURE=1.1  TOP_P=0.95  TOP_K=20  BETA=0  JSD_TOKEN_CLIP=0.05
#   USE_EMA=false  EMA_DECAY=0.999
#   NPROC_PER_NODE=8  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#
# Any extra args you pass on the command line are forwarded verbatim to
# `swift rlhf` (handy for `--debug`, overriding `--save_steps`, etc.):
#
#   bash run_phase3.sh --max_steps 50 --logging_steps 1
#
# The script hard-fails early when the training data or plugin is missing
# (set -euo pipefail + explicit file checks) so we never start a bogus run.
# On exit (success *or* failure) we call `ray stop --force` to avoid leaking
# ray workers — required by repo CLAUDE.md and training cleanup discipline.
# ------------------------------------------------------------------

set -euo pipefail

# ------------------------------------------------------------------
# Defaults (override via env)
# ------------------------------------------------------------------
: "${CONDA_ENV:=opsd}"
: "${WORKSPACE:=/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd}"
: "${MS_SWIFT_ROOT:=/home/qzheng19/ms-swift}"
: "${TRAIN_JSONL:=${WORKSPACE}/train_with_context.jsonl}"
: "${PLUGIN_PATH:=${WORKSPACE}/skill_opsd_plugin.py}"
: "${MODEL:=Qwen/Qwen3-4B}"
: "${TEACHER_MODEL:=${MODEL}}"
: "${OUTPUT_DIR:=${WORKSPACE}/ckpt_phase3}"

# Task 1.3 sampling / distillation hyperparameters
: "${MAX_STEPS:=300}"               # config.yaml training.max_steps_phase3
: "${TEMPERATURE:=1.1}"             # config.yaml training.temperature
: "${TOP_P:=0.95}"                  # config.yaml training.top_p
: "${TOP_K:=20}"                    # config.yaml training.top_k
: "${BETA:=0}"                      # config.yaml training.beta (forward-KL)
: "${JSD_TOKEN_CLIP:=0.05}"         # config.yaml training.jsd_token_clip
: "${USE_EMA:=false}"               # config.yaml training.use_ema_teacher
: "${EMA_DECAY:=0.999}"             # config.yaml training.ema_decay

# Distributed / hardware
: "${NPROC_PER_NODE:=8}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
# torchrun master port：并行 session 必须用不同 port 避免 EADDRINUSE
: "${MASTER_PORT:=29500}"
export MASTER_PORT

# wandb cleanup discipline (per CLAUDE.md): disable system-stats polling.
: "${WANDB_DISABLE_STATS:=true}"
export WANDB_DISABLE_STATS

# ------------------------------------------------------------------
# Cleanup: ensure ray workers never leak on exit
# ------------------------------------------------------------------
cleanup() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ------------------------------------------------------------------
# Pre-flight checks (run BEFORE conda so failures surface instantly even
# outside the training env — important for CI smoke tests)
# ------------------------------------------------------------------
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "[run_phase3] ERROR: TRAIN_JSONL='$TRAIN_JSONL' 不存在。" >&2
    echo "[run_phase3]   先跑 run_phase1.sh 收集 trajectories / 提炼 skills," >&2
    echo "[run_phase3]   再跑 context_builder.py (Task 3.3) 生成 train_with_context.jsonl。" >&2
    exit 1
fi
if [ ! -f "$PLUGIN_PATH" ]; then
    echo "[run_phase3] ERROR: PLUGIN_PATH='$PLUGIN_PATH' 不存在 (Task 3.4 产物)。" >&2
    exit 1
fi

# ------------------------------------------------------------------
# Activate conda env (idempotent; tolerates unset -u inside conda hooks)
# ------------------------------------------------------------------
# shellcheck disable=SC1090
source ~/.bashrc 2>/dev/null || true
set +u
conda activate "$CONDA_ENV"
set -u

cd "$MS_SWIFT_ROOT"

echo "[run_phase3] MS_SWIFT_ROOT = $MS_SWIFT_ROOT"
echo "[run_phase3] WORKSPACE     = $WORKSPACE"
echo "[run_phase3] MODEL         = $MODEL (teacher: $TEACHER_MODEL)"
echo "[run_phase3] PLUGIN_PATH   = $PLUGIN_PATH"
echo "[run_phase3] TRAIN_JSONL   = $TRAIN_JSONL"
echo "[run_phase3] OUTPUT_DIR    = $OUTPUT_DIR"
echo "[run_phase3] sampling      = temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K"
echo "[run_phase3] distill       = beta=$BETA jsd_token_clip=$JSD_TOKEN_CLIP lmbda=1.0 sft_alpha=0"
echo "[run_phase3] ema_teacher   = $USE_EMA (decay=$EMA_DECAY)"
echo "[run_phase3] schedule      = max_steps=$MAX_STEPS nproc=$NPROC_PER_NODE gpus=$CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------------
# Optional EMA-teacher flags (Task 1.2)
# ------------------------------------------------------------------
EMA_FLAGS=()
if [ "$USE_EMA" = "true" ] || [ "$USE_EMA" = "1" ]; then
    EMA_FLAGS=(--use_ema_teacher true --ema_decay "$EMA_DECAY")
fi

# ------------------------------------------------------------------
# Training command
# All hyperparameters below mirror examples/train/rlhf/opsd/opsd.sh (Task 1.3)
# except for {external_plugins, dataset, output_dir} — the three lines that
# flip the run from the OPSD baseline to Skill-Informed mode.
# ------------------------------------------------------------------
NPROC_PER_NODE="$NPROC_PER_NODE" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
swift rlhf \
    --rlhf_type gkd \
    --model "$MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    ${ADAPTER:+--adapters "$ADAPTER"} \
    ${TEACHER_ADAPTERS:+--teacher_adapters "$TEACHER_ADAPTERS"} \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 10240 \
    --sleep_level 1 \
    --external_plugins "$PLUGIN_PATH" \
    --dataset "$TRAIN_JSONL" \
    --lmbda 1.0 \
    --beta "$BETA" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --jsd_token_clip "$JSD_TOKEN_CLIP" \
    "${EMA_FLAGS[@]}" \
    --sft_alpha 0 \
    --torch_dtype bfloat16 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --max_length 8192 \
    --max_completion_length 2048 \
    --save_only_model true \
    --gradient_checkpointing true \
    --deepspeed zero0 \
    --attn_impl flash_attn \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard wandb \
    "$@"

echo "[run_phase3] training command returned $?; cleanup trap will run ray stop --force"
