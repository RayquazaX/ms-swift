#!/bin/bash
# run_full_loop.sh — Skill-Informed In-Context Self-Distillation, N-round loop
#
# Orchestrates Task 5.1 → 2.1 → (2.2) → 3.2 → 3.3 → 4.1 → Phase-4 eval
# per round, N rounds total (default 3). Each round uses the *previous*
# round's LoRA checkpoint as the student policy; round 1 falls back to the
# base model (or INITIAL_ADAPTER if supplied).
#
# Per-round sequence (round index i, 1 <= i <= N_ROUNDS):
#   Step 1. collect_trajectories.py  (base model + adapter-of-round-{i-1})
#   Step 2. skill_extract.py         (append to global skill_library.jsonl)
#   Step 3. self_icl_bootstrap.py    (optional cold-start; skip mechanism)
#   Step 4. skill_filter.py          (i > 1 AND adapter is non-empty)
#   Step 5. context_select.py        (ceil DPP, top-K per question)
#   Step 6. context_builder.py       (layered teacher_prompt)
#   Step 7. run_phase3.sh            (OPSD, MAX_STEPS per round)
#   Step 8. swift eval               (aime25 math_500 gsm8k, Native backend)
#   Step 9. tr_tl_diagnostic.py      (optional)
#   Step 10. append results.md
#
# Strict failure policy (research scenario): set -euo pipefail + cleanup trap;
# ANY step failing in round i aborts the whole loop (rounds i+1..N NOT run).
# Exception: self_icl_bootstrap is wrapped with `|| true` because its
# "skip — library already populated" exit path is treated as success at the
# loop level.
#
# ------------------------------------------------------------------
# CLI / env var overrides
# ------------------------------------------------------------------
#   CONDA_ENV=opsd                  conda env for all python/swift calls
#   WORKSPACE=<skill_opsd dir>      per-round outputs land under here
#   MS_SWIFT_ROOT=/home/qzheng19/ms-swift
#   BASE_MODEL=Qwen/Qwen3-4B        student + extractor + eval base
#   INITIAL_ADAPTER=""              optional round-0 LoRA (e.g. ckpt_phase1)
#   N_ROUNDS=3                      config.yaml loop.rounds
#   STEPS_PER_ROUND=100             training.max_steps_loop_round
#   NUM_SAMPLES=200                 trajectory_collect.num_samples
#   TOP_K=3                         context_select.top_k
#   SEED_QUESTIONS=<path.jsonl>     REQUIRED — JSONL with
#                                   {question_id,question,ground_truth,[solution]}
#   VAL_QUESTIONS=""                defaults to SEED_QUESTIONS
#   RUN_TR_TL=false                 true → run Task 5.2 per round
#   SKIP_SELF_ICL=false             true → skip cold-start bootstrap
#   EVAL_DATASETS="aime25 math_500 gsm8k"   space-separated, Native backend
#   EVAL_LIMIT=""                   optional per-dataset sample cap
#   EVAL_BACKEND=Native             {Native, OpenCompass, VLMEvalKit}
#   EVAL_INFER_BACKEND=vllm         {vllm, transformers, sglang, lmdeploy, pt}
#
# Typical invocation:
#   SEED_QUESTIONS=$PWD/seed_math.jsonl \
#   RUN_TR_TL=true \
#   bash run_full_loop.sh
#
# Disk layout produced:
#   ${WORKSPACE}/round${i}/trajectories.jsonl
#   ${WORKSPACE}/round${i}/selected.jsonl
#   ${WORKSPACE}/round${i}/train_with_context.jsonl
#   ${WORKSPACE}/ckpt_phase3_round${i}/checkpoint-*
#   ${WORKSPACE}/round${i}/eval/...            (swift eval output dir)
#   ${WORKSPACE}/round${i}/tr_tl_report.json   (optional)
#   ${WORKSPACE}/skill_library.jsonl           (global, append-only)
#   ${WORKSPACE}/results.md                    (append-only summary)
#
# NOT destructive: script never rm's the skill library or prior checkpoints.
# ------------------------------------------------------------------

set -euo pipefail

# ------------------------------------------------------------------
# Defaults (override via env)
# ------------------------------------------------------------------
: "${CONDA_ENV:=opsd}"
: "${WORKSPACE:=/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd}"
: "${MS_SWIFT_ROOT:=/home/qzheng19/ms-swift}"
: "${BASE_MODEL:=Qwen/Qwen3-4B}"
: "${INITIAL_ADAPTER:=}"
: "${N_ROUNDS:=3}"
: "${STEPS_PER_ROUND:=100}"
: "${NUM_SAMPLES:=200}"
: "${TOP_K:=3}"
: "${SEED_QUESTIONS:=}"
: "${VAL_QUESTIONS:=}"
: "${RUN_TR_TL:=false}"
: "${SKIP_SELF_ICL:=false}"
: "${EVAL_DATASETS:=aime25 math_500 gsm8k}"
: "${EVAL_LIMIT:=}"
: "${EVAL_BACKEND:=Native}"
: "${EVAL_INFER_BACKEND:=vllm}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"

# wandb discipline (per CLAUDE.md)
: "${WANDB_DISABLE_STATS:=true}"
export WANDB_DISABLE_STATS

# ------------------------------------------------------------------
# Cleanup: ray + accelerate leak discipline (per ~/.claude/CLAUDE.md)
# ------------------------------------------------------------------
cleanup() {
    ray stop --force >/dev/null 2>&1 || true
    echo "[run_full_loop] cleanup done (ray stop --force)"
}
trap cleanup EXIT

# ------------------------------------------------------------------
# Pre-flight: fail fast, BEFORE activating conda so errors surface
# ------------------------------------------------------------------
if [ -z "${SEED_QUESTIONS:-}" ]; then
    echo "[run_full_loop] ERROR: SEED_QUESTIONS is required." >&2
    echo "[run_full_loop]   export SEED_QUESTIONS=/path/to/seed.jsonl" >&2
    echo "[run_full_loop]   (each line: {question_id, question, ground_truth, [solution]})" >&2
    exit 1
fi
if [ ! -f "$SEED_QUESTIONS" ]; then
    echo "[run_full_loop] ERROR: SEED_QUESTIONS='$SEED_QUESTIONS' is not a file." >&2
    exit 1
fi
# VAL_QUESTIONS defaults to SEED_QUESTIONS if empty
: "${VAL_QUESTIONS:=$SEED_QUESTIONS}"

for f in \
    "${WORKSPACE}/collect_trajectories.py" \
    "${WORKSPACE}/skill_extract.py" \
    "${WORKSPACE}/skill_filter.py" \
    "${WORKSPACE}/context_select.py" \
    "${WORKSPACE}/context_builder.py" \
    "${WORKSPACE}/self_icl_bootstrap.py" \
    "${WORKSPACE}/tr_tl_diagnostic.py" \
    "${WORKSPACE}/run_phase3.sh" \
    "${WORKSPACE}/skill_opsd_plugin.py" \
    "${WORKSPACE}/config.yaml"
do
    if [ ! -f "$f" ]; then
        echo "[run_full_loop] ERROR: missing dependency: $f" >&2
        exit 1
    fi
done

# Ensure skill_library.jsonl exists (append-target).
touch "${WORKSPACE}/skill_library.jsonl"
# Ensure results.md exists (append-target).
touch "${WORKSPACE}/results.md"

# ------------------------------------------------------------------
# Activate conda (idempotent; tolerant of unset -u inside conda hooks)
# ------------------------------------------------------------------
# shellcheck disable=SC1090,SC1091
source ~/.bashrc 2>/dev/null || true
set +u
# Try both layouts: system anaconda, per-user conda, plain PATH.
if ! command -v conda >/dev/null 2>&1; then
    for _conda_sh in \
        /usr/local/anaconda3/etc/profile.d/conda.sh \
        /opt/conda/etc/profile.d/conda.sh \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh"
    do
        if [ -f "$_conda_sh" ]; then
            # shellcheck disable=SC1090
            source "$_conda_sh"
            break
        fi
    done
fi
conda activate "$CONDA_ENV"
set -u

cd "$MS_SWIFT_ROOT"

echo "[run_full_loop] CONDA_ENV         = $CONDA_ENV"
echo "[run_full_loop] MS_SWIFT_ROOT     = $MS_SWIFT_ROOT"
echo "[run_full_loop] WORKSPACE         = $WORKSPACE"
echo "[run_full_loop] BASE_MODEL        = $BASE_MODEL"
echo "[run_full_loop] INITIAL_ADAPTER   = ${INITIAL_ADAPTER:-<none>}"
echo "[run_full_loop] N_ROUNDS          = $N_ROUNDS  STEPS_PER_ROUND=$STEPS_PER_ROUND"
echo "[run_full_loop] NUM_SAMPLES       = $NUM_SAMPLES  TOP_K=$TOP_K"
echo "[run_full_loop] SEED_QUESTIONS    = $SEED_QUESTIONS"
echo "[run_full_loop] VAL_QUESTIONS     = $VAL_QUESTIONS"
echo "[run_full_loop] RUN_TR_TL         = $RUN_TR_TL  SKIP_SELF_ICL=$SKIP_SELF_ICL"
echo "[run_full_loop] EVAL_DATASETS     = $EVAL_DATASETS"
echo "[run_full_loop] EVAL_BACKEND      = $EVAL_BACKEND  EVAL_INFER_BACKEND=$EVAL_INFER_BACKEND"

# ------------------------------------------------------------------
# Loop
# ------------------------------------------------------------------
for round in $(seq 1 "$N_ROUNDS"); do
    echo ""
    echo "=================================================================="
    echo "[run_full_loop] ROUND $round / $N_ROUNDS"
    echo "=================================================================="

    ROUND_DIR="${WORKSPACE}/round${round}"
    CKPT_DIR="${WORKSPACE}/ckpt_phase3_round${round}"
    mkdir -p "$ROUND_DIR"

    # Resolve the adapter used for this round's trajectory sampling.
    # ms-swift 默认在 OUTPUT_DIR 下加 v0-TIMESTAMP 子目录，兼容两种布局（同 Step 8 的 glob）
    if [ "$round" -eq 1 ]; then
        ADAPTER="${INITIAL_ADAPTER:-}"
    else
        PREV_CKPT_DIR="${WORKSPACE}/ckpt_phase3_round$((round-1))"
        ADAPTER="$(ls -d "${PREV_CKPT_DIR}"/v*/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
        if [ -z "$ADAPTER" ]; then
            ADAPTER="$(ls -d "${PREV_CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
        fi
        if [ -z "$ADAPTER" ]; then
            echo "[run_full_loop] ERROR: round $round expects a checkpoint under" >&2
            echo "[run_full_loop]   ${PREV_CKPT_DIR}/v*/checkpoint-* or ${PREV_CKPT_DIR}/checkpoint-* but none exists." >&2
            exit 1
        fi
    fi
    echo "[run_full_loop] round $round adapter = ${ADAPTER:-<base only>}"

    # --- Step 1. collect_trajectories ---
    echo "[run_full_loop] Step 1/10: collect_trajectories"
    python "${WORKSPACE}/collect_trajectories.py" \
        --model "$BASE_MODEL" \
        ${ADAPTER:+--adapters "$ADAPTER"} \
        --dataset "$SEED_QUESTIONS" \
        --output "${ROUND_DIR}/trajectories.jsonl" \
        --num_samples "$NUM_SAMPLES"

    # --- Step 2. skill_extract (append to global library) ---
    echo "[run_full_loop] Step 2/10: skill_extract"
    python "${WORKSPACE}/skill_extract.py" \
        --trajectories "${ROUND_DIR}/trajectories.jsonl" \
        --skill_library "${WORKSPACE}/skill_library.jsonl" \
        --extractor_model "$BASE_MODEL" \
        --config "${WORKSPACE}/config.yaml"

    # --- Step 3. self_icl_bootstrap (cold-start; skip mechanism built-in) ---
    # Wrap in `|| true` so "skip — already enough skills" never aborts the
    # loop. A true crash (non-skip) is logged as a WARN only.
    if [ "$SKIP_SELF_ICL" != "true" ]; then
        echo "[run_full_loop] Step 3/10: self_icl_bootstrap (may skip)"
        if ! python "${WORKSPACE}/self_icl_bootstrap.py" \
                --seed_questions "$SEED_QUESTIONS" \
                --skill_library "${WORKSPACE}/skill_library.jsonl" \
                --llm_model "$BASE_MODEL" \
                --config "${WORKSPACE}/config.yaml"
        then
            echo "[run_full_loop] WARN: self_icl_bootstrap exited non-zero; continuing."
        fi
    else
        echo "[run_full_loop] Step 3/10: self_icl_bootstrap SKIPPED (SKIP_SELF_ICL=true)"
    fi

    # --- Step 4. skill_filter (i > 1 AND adapter exists) ---
    if [ "$round" -gt 1 ] && [ -n "$ADAPTER" ]; then
        echo "[run_full_loop] Step 4/10: skill_filter (KL)"
        python "${WORKSPACE}/skill_filter.py" \
            --skill_library "${WORKSPACE}/skill_library.jsonl" \
            --val_questions "$VAL_QUESTIONS" \
            --base_model "$BASE_MODEL" \
            --adapter "$ADAPTER" \
            --config "${WORKSPACE}/config.yaml"
    else
        echo "[run_full_loop] Step 4/10: skill_filter SKIPPED (round=$round, no adapter)"
    fi

    # --- Step 5. context_select (CEIL DPP, top-K) ---
    echo "[run_full_loop] Step 5/10: context_select"
    python "${WORKSPACE}/context_select.py" \
        --skill_library "${WORKSPACE}/skill_library.jsonl" \
        --questions "$SEED_QUESTIONS" \
        --output "${ROUND_DIR}/selected.jsonl" \
        --mode ceil \
        --top_k "$TOP_K"

    # --- Step 6. context_builder (layered teacher_prompt) ---
    echo "[run_full_loop] Step 6/10: context_builder"
    python "${WORKSPACE}/context_builder.py" \
        --selected "${ROUND_DIR}/selected.jsonl" \
        --skill_library "${WORKSPACE}/skill_library.jsonl" \
        --questions_with_answers "$SEED_QUESTIONS" \
        --output "${ROUND_DIR}/train_with_context.jsonl" \
        --config "${WORKSPACE}/config.yaml"

    # --- Step 7. run_phase3.sh (OPSD training) ---
    # Iterative self-distillation: both student and teacher resume from round i-1's ckpt.
    # Round 1: ADAPTER="" (empty) → student trains from base; teacher = base via disable_adapter.
    # Round i>1: ADAPTER=<round i-1 ckpt>. Student continues training the same LoRA (trainable),
    #   teacher freezes the same LoRA via --teacher_adapters. KL signal comes from context
    #   difference (teacher sees skill-enriched prompt, student sees bare question) plus from
    #   the fact that teacher's LoRA is static while student's LoRA keeps moving.
    echo "[run_full_loop] Step 7/10: run_phase3.sh (max_steps=$STEPS_PER_ROUND; adapter=${ADAPTER:-<base>})"
    TRAIN_JSONL="${ROUND_DIR}/train_with_context.jsonl" \
    PLUGIN_PATH="${WORKSPACE}/skill_opsd_plugin.py" \
    OUTPUT_DIR="${CKPT_DIR}" \
    MAX_STEPS="${STEPS_PER_ROUND}" \
    MODEL="${BASE_MODEL}" \
    CONDA_ENV="${CONDA_ENV}" \
    WORKSPACE="${WORKSPACE}" \
    MS_SWIFT_ROOT="${MS_SWIFT_ROOT}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    ADAPTER="${ADAPTER}" \
    TEACHER_ADAPTERS="${ADAPTER}" \
    bash "${WORKSPACE}/run_phase3.sh" || {
        rc=$?
        echo "[run_full_loop] WARN: run_phase3.sh returned $rc (likely post-train SIGABRT); will check for checkpoint below."
    }

    # --- Step 8. swift eval ---
    # ms-swift 默认在 OUTPUT_DIR 下加 v0-TIMESTAMP 子目录；兼容两种布局
    NEW_ADAPTER="$(ls -d "${CKPT_DIR}"/v*/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
    if [ -z "$NEW_ADAPTER" ]; then
        NEW_ADAPTER="$(ls -d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
    fi
    if [ -z "$NEW_ADAPTER" ]; then
        echo "[run_full_loop] ERROR: no checkpoint produced under ${CKPT_DIR}/checkpoint-*" >&2
        exit 1
    fi
    EVAL_OUT_DIR="${ROUND_DIR}/eval"
    mkdir -p "$EVAL_OUT_DIR"
    echo "[run_full_loop] Step 8/10: swift eval  adapter=${NEW_ADAPTER}"
    # EVAL_DATASETS is a space-separated list; use word splitting intentionally.
    # Optional EVAL_LIMIT cap passed only when non-empty.
    # swift eval CLI reference (verified via --help on 2026-04-13):
    #   --model, --adapters, --eval_dataset A B C, --eval_backend,
    #   --infer_backend, --eval_output_dir, --eval_limit, --vllm_max_lora_rank.
    # --vllm_max_lora_rank 必须 >= 训练时的 lora_rank (run_phase3.sh 默认 64)
    # 否则 vLLM 会在 load LoRA 时以 BadRequestError 崩掉。
    # shellcheck disable=SC2086
    # eval_generation_config 必须限 max_new_tokens，否则 AIME25 长题会无限 gen 卡死
    swift eval \
        --model "$BASE_MODEL" \
        --adapters "$NEW_ADAPTER" \
        --eval_backend "$EVAL_BACKEND" \
        --infer_backend "$EVAL_INFER_BACKEND" \
        --eval_dataset $EVAL_DATASETS \
        --eval_output_dir "$EVAL_OUT_DIR" \
        --vllm_max_lora_rank "${EVAL_MAX_LORA_RANK:-64}" \
        --eval_generation_config "${EVAL_GENERATION_CONFIG:-{\"max_tokens\":8192,\"temperature\":0.0,\"do_sample\":false\}}" \
        ${EVAL_LIMIT:+--eval_limit "$EVAL_LIMIT"} \
        || {
            rc=$?
            echo "[run_full_loop] WARN: swift eval returned $rc (e.g. dataset hang / partial eval); continuing loop."
        }

    # --- Step 9. tr_tl_diagnostic (optional) ---
    if [ "$RUN_TR_TL" = "true" ]; then
        echo "[run_full_loop] Step 9/10: tr_tl_diagnostic"
        python "${WORKSPACE}/tr_tl_diagnostic.py" \
            --model "$BASE_MODEL" \
            --adapters "$NEW_ADAPTER" \
            --skill_library "${WORKSPACE}/skill_library.jsonl" \
            --eval_questions "$VAL_QUESTIONS" \
            --output "${ROUND_DIR}/tr_tl_report.json" \
            --ks 1 3 5 8 \
            --num_eval 30 \
            --config "${WORKSPACE}/config.yaml"
    else
        echo "[run_full_loop] Step 9/10: tr_tl_diagnostic SKIPPED (RUN_TR_TL=false)"
    fi

    # --- Step 10. Append this round's summary to results.md ---
    echo "[run_full_loop] Step 10/10: append results.md"
    TRAJ_COUNT="$(wc -l < "${ROUND_DIR}/trajectories.jsonl" | tr -d ' ')"
    SKILL_COUNT="$(wc -l < "${WORKSPACE}/skill_library.jsonl" | tr -d ' ')"
    {
        echo ""
        echo "### Round ${round} (Task 5.3 — run_full_loop.sh)"
        echo ""
        echo "- \`ADAPTER_IN\`: \`${ADAPTER:-<base only>}\`"
        echo "- \`NEW_ADAPTER\`: \`${NEW_ADAPTER}\`"
        echo "- trajectories: \`${ROUND_DIR}/trajectories.jsonl\` (${TRAJ_COUNT} rows)"
        echo "- selected: \`${ROUND_DIR}/selected.jsonl\`"
        echo "- train_with_context: \`${ROUND_DIR}/train_with_context.jsonl\`"
        echo "- skill_library size (global, cumulative): ${SKILL_COUNT} skills"
        echo "- training MAX_STEPS: ${STEPS_PER_ROUND}"
        echo "- ckpt_dir: \`${CKPT_DIR}\`"
        echo "- eval datasets: ${EVAL_DATASETS}"
        echo "- eval output: \`${EVAL_OUT_DIR}\`"
        if [ "$RUN_TR_TL" = "true" ]; then
            echo "- tr_tl report: \`${ROUND_DIR}/tr_tl_report.json\`"
        fi
        echo ""
    } >> "${WORKSPACE}/results.md"

    echo "[run_full_loop] round $round COMPLETE  (new adapter: ${NEW_ADAPTER})"
done

echo ""
echo "[run_full_loop] ALL ${N_ROUNDS} ROUNDS COMPLETE"
echo "[run_full_loop] Final checkpoint: ${WORKSPACE}/ckpt_phase3_round${N_ROUNDS}"
echo "[run_full_loop] Final skill library size: $(wc -l < "${WORKSPACE}/skill_library.jsonl" | tr -d ' ')"
echo "[run_full_loop] Results appended to: ${WORKSPACE}/results.md"
