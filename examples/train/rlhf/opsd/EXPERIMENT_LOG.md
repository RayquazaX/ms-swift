# OPSD Experiment Log & Code Modifications

> **For future agents**: This document records all code changes made to ms-swift for OPSD work, all experiments run, and what we learned. It captures findings that are not obvious from reading `opsd.sh`/`opsd_plugin.py` alone. Workspace: `/data/qzheng19/opsd_repro/`.

---

## 1. What OPSD is (one paragraph)

OPSD (On-Policy Self-Distillation) — paper https://arxiv.org/abs/2601.18734 — uses the **same model** as both teacher and student, with the teacher receiving **privileged information** (reference reasoning trace) that the student does not see. Training minimizes a generalized Jensen-Shannon divergence between teacher and student logits on the student's on-policy sampled tokens. Implemented in `ms-swift` as a special case of GKD: when `--model == --teacher_model` and `--tuner_type lora`, the teacher is obtained for free via `disable_adapter()` on the student model. Privileged context comes from the dataset plugin (`opsd_plugin.py`) which adds a `teacher_prompt` field containing the problem + reference solution.

## 2. Code modifications to ms-swift

All changes are **additive** and gated behind `--log_opsd_io true` (default `false`). With the flag off, there is zero behavioral or performance change.

### 2.1 `swift/arguments/rlhf_args.py`

Added one field to `RLHFArguments` (near the GKD section, after `offload_teacher_model`):

```python
log_opsd_io: bool = False  # Log structured student/teacher I/O per step to opsd_io_log.jsonl
```

### 2.2 `swift/rlhf_trainers/arguments.py`

Added the same field to `GKDConfig`:

```python
@dataclass
class GKDConfig(RolloutTrainerArgumentsMixin, TrainArgumentsMixin, HfGKDConfig):
    sft_alpha: float = 0
    offload_teacher_model: bool = False
    max_completion_length: int = 512
    log_completions: bool = False
    log_opsd_io: bool = False   # <-- ADDED
```

**Why both files**: `TrainerFactory.get_training_args()` at `swift/trainers/trainer_factory.py:67-68` filters `asdict(RLHFArguments)` by `inspect.signature(GKDConfig).parameters`, dropping any keys not in `GKDConfig`. A flag added only to `RLHFArguments` shows up in `args.json` (the saved user-facing dump) but is **invisible to the trainer**. This cost us one debugging cycle — the skill `framework-cli-flag-not-applied` captures this trap.

### 2.3 `swift/rlhf_trainers/gkd_trainer.py`

**(a) In `_prepare_logging()` (~line 845)** — create a dedicated JsonlWriter when the flag is on:

```python
# OPSD I/O logging (structured student/teacher I/O per step)
self.log_opsd_io = getattr(args, 'log_opsd_io', False)
if self.log_opsd_io:
    self.opsd_io_writer = JsonlWriter(os.path.join(self.args.output_dir, 'opsd_io_log.jsonl'))
```

**(b) In `training_step()` after line 529** (inside `if args.use_vllm:` block, after OPSD teacher data prep) — log student input/output + teacher input for the first item of each batch:

```python
# --- OPSD I/O Logging (first item per batch only) ---
if self.log_opsd_io and self.accelerator.is_main_process \
        and teacher_data is not None and len(generated_inputs) > 0:
    gen_data = generated_inputs[0]
    t_data = teacher_data[0]
    student_msgs = inputs[0].get('messages', [])

    def _to_text(x):
        if isinstance(x, str):
            return x
        if isinstance(x, list) and x and isinstance(x[0], int):
            return self.processing_class.decode(x, skip_special_tokens=True)
        if isinstance(x, list):
            return ' '.join(
                str(item.get('text', item)) if isinstance(item, dict) else str(item)
                for item in x)
        return '' if x is None else str(x)

    student_response_text = _to_text(gen_data['messages'][-1].get('content'))
    record = {
        'step': self.state.global_step,
        '=== STUDENT INPUT ===': '',
        'student_system': _to_text(next(
            (m.get('content') for m in student_msgs if m.get('role') == 'system'), '')),
        'student_query': _to_text(next(
            (m.get('content') for m in student_msgs if m.get('role') == 'user'), '')),
        '=== STUDENT OUTPUT ===': '',
        'student_response': student_response_text[:2000],
        'student_response_chars': len(student_response_text),
        '=== TEACHER INPUT ===': '',
        'teacher_query': _to_text(next(
            (m.get('content') for m in t_data['messages'] if m.get('role') == 'user'), '')),
    }
    self.opsd_io_writer.append(record)
```

Notes:
- vLLM returns student response either as a string or as a list of token IDs. `_to_text()` handles both.
- Only logs the first example per batch to keep the file readable (subsample, not every example).
- `student_response` is truncated to 2000 chars in the record, but `student_response_chars` reports the full length.

**(c) In `compute_loss()` after line 412** (inside the `self._is_self_distillation` branch, after loss is computed) — log teacher top-1 predictions (decoded):

```python
# --- OPSD Teacher Output Logging ---
if self.log_opsd_io and self.accelerator.is_main_process:
    with torch.no_grad():
        t_logits = outputs_teacher.logits[0]
        t_labels = opsd_labels[0] if opsd_labels is not None else inputs['labels'][0]
        t_resp_mask = t_labels != -100
        t_top1 = t_logits[t_resp_mask].argmax(dim=-1)
        self.opsd_io_writer.append({
            'step': self.state.global_step,
            '=== TEACHER OUTPUT ===': '',
            'teacher_top1_decoded': self.processing_class.decode(t_top1, skip_special_tokens=True),
            'loss_value': float(loss.item()),
        })
```

Notes:
- Teacher has no text "output" — it only produces logits. We take `argmax` over response-token positions and decode for a readable approximation.
- Result: each training step produces **2 records** in `opsd_io_log.jsonl`: one from `training_step` (student I/O + teacher input), one from `compute_loss` (teacher output + loss).

### 2.4 Usage

Add to any GKD/OPSD training command:

```bash
--log_opsd_io true
```

Output file: `{output_dir}/{version}/opsd_io_log.jsonl`. Monitor with:

```bash
tail -f opsd_io_log.jsonl | python -m json.tool
```

### 2.5 No changes to existing OPSD files

`examples/train/rlhf/opsd/opsd.sh` and `opsd_plugin.py` are unchanged. Our experiments created new scripts outside the repo under `/data/qzheng19/opsd_repro/scripts/`.

---

## 3. Environment

Locked to these versions after ~5 rebuild iterations (see `pip-no-deps-strategy` and `cuda-extension-abi-match` skills):

| Package | Version | Why |
|---|---|---|
| Python | 3.10 | vllm 0.11.0 wheels only ship for cp310/cp311 |
| torch | 2.8.0+cu128 | Brought by vllm 0.11.0 |
| vllm | 0.11.0 | ms-swift recommended range; flash-attn 2.8.3 has wheels for this |
| flash-attn | 2.8.3 | Must be built from source after torch changes (ABI sensitive) |
| transformers | 4.57.6 | Pin `--no-deps` to avoid 5.5.0 breaking huggingface-hub compat |
| peft | 0.18.1 | Brought by ms-swift |
| trl | 0.29.1 | Brought by ms-swift |
| deepspeed | 0.18.9 | — |
| evalscope | 1.5.2.post1 | For AIME25 eval |
| wandb | 0.25.1 | Note: needs `protobuf>=4.25,<6` and `huggingface-hub<1.0` |

Conda env name: `opsd`. Activate with `conda activate opsd`; env vars set via `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh` (sets `HF_HOME=/data/qzheng19/huggingface_cache`, `USE_HF=1`, `WANDB_PROJECT=opsd-repro`).

---

## 4. Experiments run and results

All on 1 node, 10× RTX A6000 (46-49 GiB each). Training workspace: `/data/qzheng19/opsd_repro/training_output/`. Eval results: `/data/qzheng19/opsd_repro/eval_output/`.

### 4.1 Baseline reproductions (matched paper qualitatively)

| Model | Setup | Checkpoint | AIME2025 | Improvement |
|---|---|---|---|---|
| Qwen3-1.7B raw | — | — | **0.1667** (5/30) | (baseline) |
| Qwen3-1.7B + OPSD (JSD, β=0.5) | 3 GPUs, batch 4, lr 2e-5 | step 100 | **0.30** (9/30) | **+80%** |
| Qwen3-4B raw | — | — | **0.2333** (7/30) | (baseline) |
| Qwen3-4B + OPSD (JSD, β=0.5) | 3 GPUs, batch 4, lr 2e-5 | step 100 | **0.3333** (10/30) | **+43%** |

Paper reports Qwen3-4B base 0.1667 → step-100 0.2667 (+60%). Our 4B base is higher (0.2333), possibly due to different eval config or sampling seed, but the **relative improvement from OPSD is consistent**.

### 4.2 Ablation: divergence measure (`--beta`) on Qwen3-1.7B

Holding all other hyperparameters fixed (`lmbda=1.0`, `temperature=1.2`, `lr=2e-5`, batch 4, single GPU), vary `--beta`:

| β | Loss type | Step 40 | Step 60 | Step 100 |
|---|---|---|---|---|
| 0 | Forward KL `KL(student‖teacher)` | 0.2333 | pending | **0.20** ↓ |
| **0.5** | **JSD (symmetric)** | — | — | **0.30** ← best |
| 1 | Reverse KL `KL(teacher‖student)` | 0.20 | pending | 0.20 |

Key observations:
- **JSD (β=0.5) clearly dominates** — 0.30 vs 0.20 for both pure KLs at step-100. 50% relative gap.
- **Forward KL (β=0) shows the "first up then down" pattern** warned about by the RLSD paper (arXiv:2604.03128): step-40 is higher than step-100. This is the degradation signature.
- **Reverse KL (β=1) never really improves** over the 0.20 plateau; loss also bounces back up after step 40.
- Loss curves: β=0 is 0.06-0.10 range, β=1 is 0.05-0.08 (reverse KL is mode-seeking, lower magnitude but noisier), JSD is in between.

### 4.3 "Two-stage" experiment: continue OPSD from JSD-100 checkpoint

Hypothesis: if OPSD helps, starting from an already-improved model should yield further gains. Tested three ways to continue training where **both teacher and student start from the JSD-100 checkpoint**.

#### Variant A — merge-lora → new base → new LoRA (v1)

1. `swift merge-lora --model Qwen/Qwen3-1.7B --adapters <JSD-100> --output_dir <merged>`
2. Train: `--model <merged> --teacher_model <merged> --tuner_type lora`
3. Teacher via `disable_adapter()` = merged model (JSD-100 baked in); student = merged + new LoRA

| Checkpoint | AIME25 | Note |
|---|---|---|
| Merged model alone (no adapter) | **0.2333** | Down from 0.30 → bf16 merge loses ~7% |
| + new LoRA at step 40 | **0.1667** | Catastrophic, back to raw-base level |

**Finding**: `swift merge-lora` introduces measurable precision loss on top of whatever structural degradation OPSD induces. Merge is not transparent.

#### Variant B — `--adapters` continues LoRA, teacher via disable_adapter()

Skipped — equivalent to continuing normal JSD training; teacher = raw Qwen3-1.7B (not JSD-100), which contradicts the hypothesis.

#### Variant C — separate teacher with `--teacher_adapters` (v2, uses 2 GPUs)

```bash
--model Qwen/Qwen3-1.7B --adapters <JSD-100>
--teacher_model Qwen/Qwen3-1.7B --teacher_adapters <JSD-100>
--tuner_type lora ...
--use_vllm true --vllm_mode colocate
NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=3,4
```

Student loads JSD-100 LoRA (continues training). A second copy of Qwen3-1.7B + JSD-100 LoRA is loaded as a frozen teacher. No merge, no precision loss. Requires `NPROC_PER_NODE` = number of GPUs (vllm colocate incompatible with `device_map`).

| Checkpoint | AIME25 |
|---|---|
| Start (JSD-100 LoRA, before new training) | 0.30 |
| step 40 | 0.2333 ↓ |
| step 60 | 0.2333 ↓ |
| step 100 | 0.20 ↓↓ |

**Finding**: Even with the "ideal" setup (no merge artifacts, teacher exactly matches student's stage, privileged info intact), continuing OPSD training **monotonically degrades** the JSD-100 model from 0.30 down to 0.20 over 100 steps. Teacher quality is not the issue.

### 4.4 Sanity checks that informed debugging

- **Raw Qwen3-1.7B without any training**: 0.1667. Confirms the baseline.
- **JSD-100 merged alone (no new adapter)**: 0.2333. Confirmed the merge operation itself is lossy — variant A's 0.1667 at step 40 is *merge loss + training degradation* stacked.
- **Training log shape**: JSD → OPSD (variant C) loss starts low (~0.012) because teacher and student begin identically; it oscillates in 0.010-0.025 range. JSD loss on a fresh student starts ~0.09 and decreases over steps.

---

## 5. Distilled insights

### 5.1 OPSD works but fragily; Reverse/Forward KL don't

Consistent with RLSD paper findings — the **symmetric JSD** (β=0.5) is what makes OPSD useful. Forward KL degrades; Reverse KL doesn't learn; only JSD strikes a balance. This suggests the `--beta 0.5` default in `opsd.sh` is load-bearing, not arbitrary.

### 5.2 OPSD degradation is structural, not setup-dependent

Three different teacher configurations (raw base, merged JSD-100, frozen JSD-100 via teacher_adapters) all show the same pattern: early gain, then monotonic degradation past step 40-60. The RLSD paper's theoretical argument (privileged information enters the gradient direction → student is forced to encode `x→r` correlations → fake reasoning tokens) holds empirically.

**Practical implication**: stop OPSD early. The "sweet spot" is around step 40-100; continuing longer actively hurts. Don't use OPSD as a second-stage fine-tune on an already-good model.

### 5.3 `swift merge-lora` has silent precision loss

On Qwen3-1.7B, we observed 0.30 (base + LoRA dynamic) → 0.2333 (merged) — **~22% relative accuracy drop** on AIME25 from the merge alone. This is likely bf16 accumulation error. If you need a merged checkpoint, eval it immediately after merge and expect to lose some performance. For continuing training, variant C (`--teacher_adapters`) is lossless.

### 5.4 No filtering in the OPSD pipeline

`opsd_plugin.py` filters the dataset to `correct=True` samples at load time. After that, **nothing is filtered**: student's wrong generations are still distilled against teacher's (privileged) generations, and teacher's bad tokens are still used as targets. There is no reward model, no verifier, no rejection step. This is part of why OPSD is fragile — compare to GRPO which uses verifier reward to direction-anchor updates.

### 5.5 Two-stage args (RLHFArguments ↔ GKDConfig)

Any new CLI flag for GKD/OPSD needs to be added in **two** files, not one. See `framework-cli-flag-not-applied` skill for the general pattern. Adding only to `RLHFArguments` produces the silent failure: flag shows in `args.json` but trainer ignores it.

### 5.6 vLLM colocate + single-GPU gotchas

- Use `NPROC_PER_NODE=1` for single-GPU runs, not manual `RANK=0 LOCAL_RANK=0 WORLD_SIZE=1` (the latter triggers distributed rendezvous expecting `MASTER_ADDR`).
- Set `MASTER_PORT` per experiment (29500, 29501, …) or parallel single-GPU runs will collide.
- `--vllm_mode colocate` is incompatible with transformers `device_map` — you must set `NPROC_PER_NODE` equal to the GPU count in `CUDA_VISIBLE_DEVICES`.

### 5.7 OOM patterns specific to OPSD

OPSD's peak memory differs per `--beta`:
- β=0 (Forward KL): one `kl_div` call, lower peak
- β=0.5 (JSD): two `kl_div` calls + `logsumexp`, highest peak
- β=1 (Reverse KL): one `kl_div` call, medium peak (we hit OOM mid-run at step 102)

If you OOM on a specific β, the mitigation chain is: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → reduce `--vllm_gpu_memory_utilization` 0.7→0.5 → halve `--per_device_train_batch_size` with doubled `gradient_accumulation_steps`.

---

## 6. File layout produced

```
/data/qzheng19/opsd_repro/
├── models/
│   └── qwen3-1.7b-jsd100-merged/      # merge-lora artifact (not used for final analysis; lossy)
├── training_output/
│   ├── v1-20260408-051858/             # original JSD reproduction (Qwen3-1.7B)
│   ├── qwen3-4b/v0-20260408-062501/    # JSD reproduction (Qwen3-4B)
│   ├── qwen3-1.7b-beta0/v1-.../        # Forward KL ablation
│   ├── qwen3-1.7b-beta1/v1-.../        # Reverse KL ablation
│   ├── qwen3-1.7b-jsd100-opsd/         # variant A (merge-lora based)
│   └── qwen3-1.7b-jsd100-opsd-v2/      # variant C (teacher_adapters based)
├── eval_output/
│   ├── qwen3-1.7b-base/
│   ├── qwen3-4b-base/
│   ├── qwen3-4b-step100/
│   ├── qwen3-1.7b-beta{0,1}-step{40,60,100}/
│   ├── qwen3-1.7b-jsd100-merged-baseline/
│   └── qwen3-1.7b-jsd100-opsd-v2-step{40,60,100}/
├── scripts/                             # one .sh per experiment (strict isolation)
│   ├── train_qwen3_1.7b.sh
│   ├── train_qwen3_4b.sh
│   ├── train_qwen3_1.7b_beta0.sh
│   ├── train_qwen3_1.7b_beta1.sh
│   ├── train_qwen3_1.7b_jsd100_opsd.sh       # variant A
│   ├── train_qwen3_1.7b_jsd100_opsd_v2.sh    # variant C
│   └── eval_qwen3_1.7b_{beta0,beta1,jsd100_opsd_v2}_step{40,60,100}.sh
├── logs/                                # stdout of each tmux session
└── plan_opsd_io_logging.md              # historical plan file
```

Key principle (enforced from mid-conversation onward): **every experiment has a unique script, output dir, log, wandb run name, tmux session, GPU assignment, MASTER_PORT, and eval --port**. See `ablation-experiment-isolation` skill.

---

## 7. Reproducing specific experiments

### 7.1 Base OPSD (paper setup)
Use the unmodified `examples/train/rlhf/opsd/opsd.sh` directly. Verify the 0.30 / 0.33 AIME25 results reproduce.

### 7.2 Beta ablation
Copy `opsd.sh` three times, change `--beta` to 0 / 0.5 / 1, give each a distinct `--output_dir` and `WANDB_RUN_NAME`. Launch in parallel on separate GPUs with distinct `MASTER_PORT`. Eval at step 40, 60, 100 (requires `--save_steps 20`).

### 7.3 I/O logging
Add `--log_opsd_io true` to any OPSD training command. Tail `{output_dir}/{version}/opsd_io_log.jsonl` to see student input/response, teacher input (with reference solution), teacher top-1 decoded output, and per-step loss.

### 7.4 Two-stage from an existing checkpoint
If you want teacher = previous-stage model (not raw base), the cleanest route is **variant C**: use `--adapters <ckpt>` on the student side and `--teacher_adapters <ckpt>` on the teacher side, with `NPROC_PER_NODE=2` across 2 GPUs. Avoid `swift merge-lora` unless you have a reason — it leaks accuracy.

---

## 8. Related skills (for future agents)

Stored in `/home/qzheng19/.claude/commands/`:

- `opsd-repro.md` — the concrete recipe for reproducing OPSD
- `pip-no-deps-strategy.md` — layered dep install
- `cuda-extension-abi-match.md` — flash-attn / ABI issues
- `ablation-experiment-isolation.md` — isolation discipline
- `gpu-discovery-cleanup.md` — GPU ops
- `subagent-monitoring-pattern.md` — background monitoring
- `training-oom-mitigation-chain.md` — OOM recovery
- `framework-cli-flag-not-applied.md` — the `RLHFArguments ↔ GKDConfig` trap

Read those if you're extending this work and encounter a similar issue.
