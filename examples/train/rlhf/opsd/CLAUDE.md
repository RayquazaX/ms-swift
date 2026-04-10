# CLAUDE.md — OPSD Example

## Overview

OPSD (On-Policy Self-Distillation) is a special case of GKD where the same model serves as both teacher and student. The teacher receives privileged information (reference solutions) while the student sees only the problem. Paper: https://arxiv.org/abs/2601.18734

The key idea: use `teacher_prompt` to give the teacher extra context (a reference solution), then distill the teacher's output distribution into the student which only sees the original question.

## Files

| File | Purpose |
|---|---|
| `opsd.sh` | Training script |
| `opsd_plugin.py` | Dataset plugin that builds `teacher_prompt` with reference solutions |

## How to Run

```bash
# Requires 8 GPUs
bash examples/train/rlhf/opsd/opsd.sh
```

## How It Works

1. **Dataset plugin** (`opsd_plugin.py`):
   - Registers a preprocessor for `open-r1/OpenThoughts-114k-math`
   - Student sees: the math problem only
   - Teacher sees: problem + reference solution (via `teacher_prompt` field)
   - Filters to only verified-correct examples

2. **Self-distillation**:
   - `--model Qwen/Qwen3-4B` and `--teacher_model Qwen/Qwen3-4B` (same model)
   - Student gets LoRA adapters (`--tuner_type lora`)
   - Teacher uses base model weights via `disable_adapter()` — no extra model loaded
   - Training distills teacher's privileged-info output into student

3. **On-policy**: `--lmbda 1.0` means 100% on-policy — student always generates its own responses

## Configuration

```
Model:     Qwen/Qwen3-4B (both teacher and student)
Tuner:     LoRA (rank=64, alpha=128, all-linear)
Dataset:   open-r1/OpenThoughts-114k-math
vLLM:      colocate mode, gpu_utilization=0.7
lr:        2e-5
beta:      0.5 (JSD interpolation)
lmbda:     1.0 (pure on-policy)
temp:      1.2
max_completion_length: 2048
deepspeed: zero0
```

## Results

On AIME2025 benchmark:

| Checkpoint | Accuracy | Improvement |
|---|---|---|
| Base (Qwen3-4B) | 0.1667 | - |
| 100 steps | 0.2667 | +60% |

## Evaluation

```bash
swift eval --model Qwen/Qwen3-4B \
    --adapters output/Qwen3-4B/xxx/checkpoint-xxx \
    --eval_dataset aime25 --eval_backend Native --infer_backend vllm \
    --vllm_max_lora_rank 64 \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'
```

## Plugin Architecture

The plugin uses ms-swift's `register_dataset` API with a custom `RowPreprocessor`:

```python
class OpenThoughtsOPSDPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        # Filter incorrect examples
        if not row.get('correct', True):
            return None
        # Student sees: problem
        # Teacher sees: problem + solution (via teacher_prompt)
        return {'messages': [...], 'teacher_prompt': f'{problem}\n...{solution}\n...'}

register_dataset(DatasetMeta(
    ms_dataset_id='open-r1/OpenThoughts-114k-math',
    preprocess_func=OpenThoughtsOPSDPreprocessor(),
))
```

Load via: `--external_plugins examples/train/rlhf/opsd/opsd_plugin.py`

## Source Files

- GKD Trainer (shared): `swift/rlhf_trainers/gkd_trainer.py`
- OPSD teacher data building: `_build_opsd_teacher_data()` method in GKDTrainer
- Megatron version: `examples/megatron/rlhf/gkd/opsd.sh`
