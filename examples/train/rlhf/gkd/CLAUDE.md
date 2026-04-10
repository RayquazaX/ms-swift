# CLAUDE.md — GKD Examples

## Overview

GKD (Generalized Knowledge Distillation) trains a student model to match a teacher model's output distribution using Generalized Jensen-Shannon Divergence (JSD). The implementation is in `swift/rlhf_trainers/gkd_trainer.py`.

## How to Run

All scripts use `swift rlhf --rlhf_type gkd`. Requires multi-GPU setup (4-8 GPUs, 56-73 GiB VRAM per GPU depending on mode).

## Example Scripts

| Script | Mode | Description | VRAM/GPU | Speed |
|---|---|---|---|---|
| `fast.sh` | Offline (pre-sampled) | Pre-generate teacher data with vLLM, then train offline (`lmbda=0`) | 4x56G | 1.55s/it |
| `full.sh` | Online + SeqKD | Teacher generates during training (`seq_kd=true`) | 4x66G | 46s/it |
| `vllm_colocate.sh` | On-policy + vLLM colocate | Student generates with colocated vLLM (`lmbda=0.5`) | 4x73G | 11s/it |
| `vllm_server.sh` | On-policy + vLLM server | Student generates via external vLLM server (`lmbda=0.5`) | 4x54G | 5s/it |
| `teacher_server.sh` | On-policy + API teacher | Teacher logprobs from external vLLM server, top-k mode | 4xGPU | - |
| `think_model.sh` | Offline + SFT loss | Pre-sampled from reasoning model (Qwen3), `sft_alpha=0.1` for `<think>` tokens | 4x67G | 2.50s/it |

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--rlhf_type gkd` | - | Required to select GKD |
| `--teacher_model` | None | Teacher model path/ID (separate model or same as student for self-distillation) |
| `--lmbda` | 0.5 | Probability of on-policy sampling (0=offline, 1=pure on-policy) |
| `--beta` | 0.5 | JSD interpolation (0=Forward KL, 0.5=JSD, 1=Reverse KL) |
| `--seq_kd` | false | Use teacher generation for non-on-policy samples |
| `--sft_alpha` | 0 | Weight for auxiliary SFT loss (useful for think models) |
| `--temperature` | 0.9 | Generation sampling temperature |
| `--max_completion_length` | 512 | Max tokens during generation |
| `--gkd_logits_topk` | None | Top-K logits for memory-efficient KL computation (required with `--teacher_model_server`) |
| `--teacher_model_server` | None | External teacher API URL (e.g., `http://localhost:8000`) |
| `--teacher_deepspeed` | None | DeepSpeed config for teacher (e.g., `zero3_offload`) |
| `--offload_teacher_model` | false | Offload teacher to CPU to save VRAM |

## Three Training Modes (per sample)

Each training step samples one mode:
1. **On-policy** (prob=`lmbda`): Student model generates the response
2. **Sequential KD** (prob=`1-lmbda`, if `seq_kd=true`): Teacher model generates the response
3. **Offline** (prob=`1-lmbda`, if `seq_kd=false`): Use dataset response as-is

## Acceleration Strategies

1. **Pre-sampling** (`fast.sh`): Generate teacher data offline with `swift infer`, then train with `lmbda=0`. Fastest but no on-policy learning.
2. **vLLM colocate** (`vllm_colocate.sh`): `--use_vllm true --vllm_mode colocate`. Student vLLM in same process, uses `--sleep_level 1` to share GPU memory.
3. **vLLM server** (`vllm_server.sh`): `--use_vllm true --vllm_mode server`. External vLLM server for student sampling.
4. **Teacher server** (`teacher_server.sh`): `--teacher_model_server http://...`. Fetch only top-K teacher logprobs from API. Most memory-efficient for large teachers.

## Source Files

- Trainer: `swift/rlhf_trainers/gkd_trainer.py`
- Arguments: `swift/arguments/rlhf_args.py` (TeacherModelArguments, RLHFArguments)
- Tests: `tests/train/test_gkd.py`
- Docs: `docs/source_en/Instruction/GKD.md`
