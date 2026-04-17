#!/bin/bash
# eval_hotpotqa_evolver.sh
# 评测 EvolveR 官方 3B checkpoint 在 HotpotQA (+ NQ test set) 上的 EM/F1
#
# NOTE: 完全参考 EvolveR/scripts/test-3b.sh（已确认文件存在）
#       只是改 BASE_MODEL 指向本地 /data/qzheng19/EvolveR，改 OUT 到本 eval 目录
#
# 前置依赖（首次 bootstrap 在交互 shell 做）：
#   1. conda activate evolver 环境（本脚本用 opsd 先尝试；如报错需要切 evolver env）
#   2. 数据准备：`data/nq_hotpotqa_train/{train,test}.parquet`
#      huggingface-cli download Edaizi/EvolveR-NQ-HotpotQA --repo-type dataset \
#         --local-dir /home/qzheng19/EvolveR/data/nq_hotpotqa_train
#   3. Milvus VDB 服务（脚本自动起）
#   4. 嵌入服务：bash scripts/vllm_server.sh （port 8081）
#   5. 检索服务：bash scripts/retrieval_launch.sh （port 8000；需要 Wiki 语料 + e5 index）
#   6. 以上 3-5 需要分别在独立 tmux session 中启动
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
EVOLVER_DIR=/home/qzheng19/EvolveR
OUT=$WS/eval/logs/hotpotqa_evolver_results
mkdir -p "$OUT"

# NOTE: Edaizi/EvolveR HF repo 下有两个子目录：
#       EvolveR-3B/             （主模型，对应论文 RL 阶段 checkpoint；纯 HF safetensors）
#       EvolveR-3B-cold_start/  （cold-start SFT warm-start；也是 HF safetensors）
#       评测论文主结果 → 用 EvolveR-3B/
export BASE_MODEL=${BASE_MODEL:-/data/qzheng19/EvolveR/EvolveR-3B}
if [ ! -f "$BASE_MODEL/config.json" ] || [ ! -f "$BASE_MODEL/model.safetensors.index.json" ]; then
    echo "ERROR: EvolveR-3B weights not at $BASE_MODEL (config.json / model.safetensors.index.json missing)"
    echo "       请先 hf download Edaizi/EvolveR"
    exit 1
fi

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
GPU_NUM=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

export WANDB_PROJECT=${WANDB_PROJECT:-opsd-repro}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-eval-hotpotqa-evolver-3b}
: "${WANDB_DISABLE_STATS:=true}"
export WANDB_DISABLE_STATS
WAND_PROJECT=$WANDB_PROJECT

# EvolveR test-3b.sh 使用的路径约定
export DATA_DIR=${DATA_DIR:-data/nq_hotpotqa_train}
USE_EXPERIENCE=${USE_EXPERIENCE:-true}
export EXPERIMENT_NAME=${WANDB_RUN_NAME}
export EMBEDDING_API_URL=${EMBEDDING_API_URL:-http://127.0.0.1:8081/v1}
export RETRIEVE_URL=${RETRIEVE_URL:-http://127.0.0.1:8000/retrieve}
export EXPERIENCE_EXPORT_DIR=${EXPERIENCE_EXPORT_DIR:-data/evolver/result}
export VDB_IMPORT_DB_FILE=""
export SWANLAB_LOG_DIR="swanlog"
export VLLM_ATTENTION_BACKEND=XFORMERS
export MKL_SERVICE_FORCE_INTEL=1
export HYDRA_FULL_ERROR=1

echo "[eval_hotpotqa_evolver] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  GPU_NUM=$GPU_NUM"
echo "[eval_hotpotqa_evolver] BASE_MODEL=$BASE_MODEL"
echo "[eval_hotpotqa_evolver] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[eval_hotpotqa_evolver] OUT=$OUT"

cd "$EVOLVER_DIR"

# --- MilvusDB Server 自管（参考 scripts/test-3b.sh L30-86）---
if [ "$USE_EXPERIENCE" = "true" ]; then
    DB_SERVER_DIR="${EXPERIENCE_EXPORT_DIR}/${EXPERIMENT_NAME}/db_server"
    DB_SERVER_LOG_FILE="${DB_SERVER_DIR}/db_server-${EXPERIMENT_NAME}.log"
    DB_EXPORT_DIR="${EXPERIENCE_EXPORT_DIR}/${EXPERIMENT_NAME}/db_exports"
    export VDB_SERVER_URL="http://127.0.0.1:8080"

    cleanup_db_server() {
        echo "--- Cleaning up MilvusDB Server ---"
        curl -s -X POST "${VDB_SERVER_URL}/export/" \
          -H "Content-Type: application/json" \
          -d "{
            \"collections\": [\"principles\", \"trajectories\"],
            \"format\": \"jsonl\",
            \"output_root_dir\": \"${EXPERIENCE_EXPORT_DIR}\",
            \"experiment_name\": \"${EXPERIMENT_NAME}\"
          }" || true
    }
    trap 'cleanup; cleanup_db_server' EXIT SIGINT SIGTERM

    rm -rf "$DB_SERVER_DIR"
    mkdir -p "$DB_SERVER_DIR" "$DB_EXPORT_DIR"

    echo "--- Starting MilvusDB Server ---"
    export VDB_BASE_DIR="$DB_SERVER_DIR"
    bash evolver/experience/milvusdb/start_server.sh > "$DB_SERVER_LOG_FILE" 2>&1 &

    echo "Waiting for DB server (${VDB_SERVER_URL}) to start..."
    for i in $(seq 1 60); do
      if curl -s "${VDB_SERVER_URL}/" | grep -q '"status":"running"'; then
        echo "DB Server is running."
        break
      fi
      sleep 2
      if [ $i -eq 60 ]; then
        echo "Error: DB server failed to start within timeout. Log: $DB_SERVER_LOG_FILE"
        exit 1
      fi
    done
else
    export VDB_SERVER_URL=""
fi

# --- 主 eval 命令（完全对齐 scripts/test-3b.sh L94-166）---
python3 -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size=128 \
  data.val_batch_size=2048 \
  data.max_prompt_length=8192 \
  data.max_response_length=1024 \
  data.max_start_length=2048 \
  data.max_obs_length=2048 \
  data.shuffle_train_dataloader=true \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-8 \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=32 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.no_think_rl=false \
  actor_rollout_ref.rollout.n_agent=8 \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.actor.state_masking=true \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  +trainer.val_only=true \
  +trainer.val_before_train=true \
  trainer.val_do_sample=false \
  trainer.val_temperature=0.6 \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=${GPU_NUM} \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.project_name=$WAND_PROJECT \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir=$OUT \
  rewards.weights.format=0.1 \
  rewards.weights.outcome=1.0 \
  rewards.weights.info_gain=0 \
  rewards.weights.experience=0 \
  experience.enable=$USE_EXPERIENCE \
  experience.vdb_server_url=$VDB_SERVER_URL \
  experience.organize_interval=1 \
  experience.export_interval=50 \
  experience.experience_data_dir=${EXPERIENCE_EXPORT_DIR} \
  experience.embedding_api_url=${EMBEDDING_API_URL} \
  experience.trajectory_choice_ratio=0.25 \
  experience.retrieve_component.principle=true \
  experience.retrieve_component.structure=true \
  experience.retrieve_component.success_trajectory=false \
  experience.retrieve_component.failure_trajectory=false \
  max_turns=10 \
  retriever.url=${RETRIEVE_URL} \
  retriever.topk=3 \
  2>&1 | tee $OUT/eval_log.txt

echo "[eval_hotpotqa_evolver] DONE. OUT=$OUT"
echo "[eval_hotpotqa_evolver] 请从 log/wandb 提取 val EM/F1，更新 eval/leaderboard_hotpotqa.md"
