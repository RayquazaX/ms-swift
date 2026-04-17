#!/bin/bash
# eval_webshop_skillrl.sh
# 评测 SkillRL 官方 Webshop-7B-RL checkpoint 在 WebShop 上的 success rate
#
# NOTE: SkillRL README 没有提供独立 eval 脚本，使用 verl 框架的 val_only 模式
#       基础脚本从 SkillRL/examples/grpo_trainer/run_webshop_skills.sh 改造而来（README L146-156）
#
# 前置依赖（首次 bootstrap 在交互 shell 做）：
#   1. conda activate skillrl 环境
#   2. cd SkillRL/agent_system/environments/env_package/webshop && ./setup.sh -d all  （下载 webshop 数据）
#   3. memory_data/webshop/claude_style_skills.json 已存在（repo 自带）
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
SKILLRL_DIR=/home/qzheng19/SkillRL
OUT=$WS/eval/logs/webshop_skillrl_results
mkdir -p "$OUT"

# NOTE: 已经用 SkillRL/scripts/model_merger.py 把 verl FSDP shard merge 成 HF safetensors，MODEL_PATH 指向 hf/ 子目录
export MODEL_PATH=${MODEL_PATH:-/data/qzheng19/Webshop-7B-RL/hf}
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: Webshop-7B-RL HF ckpt not at $MODEL_PATH (config.json missing)"
    echo "       请先运行 model_merger.py 合并 FSDP shard 到 HF 格式，再设置 MODEL_PATH=/data/qzheng19/Webshop-7B-RL/hf"
    exit 1
fi

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
GPU_NUM=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

export WANDB_PROJECT=${WANDB_PROJECT:-opsd-repro}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-eval-webshop-skillrl-7b}
: "${WANDB_DISABLE_STATS:=true}"
export WANDB_DISABLE_STATS

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_BACKEND_LOG_LEVEL=debug

echo "[eval_webshop_skillrl] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  GPU_NUM=$GPU_NUM"
echo "[eval_webshop_skillrl] MODEL_PATH=$MODEL_PATH"
echo "[eval_webshop_skillrl] OUT=$OUT"

cd "$SKILLRL_DIR"

val_data_size=64
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 16 \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=6000 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/webshop/claude_style_skills.json \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.enable_dynamic_update=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    ++trainer.val_only=True \
    ++trainer.val_before_train=True \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_RUN_NAME \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$OUT \
    2>&1 | tee $OUT/eval_log.txt

echo "[eval_webshop_skillrl] DONE. OUT=$OUT"
echo "[eval_webshop_skillrl] 请从 log/wandb 提取 val success rate，更新 eval/leaderboard_webshop.md"
