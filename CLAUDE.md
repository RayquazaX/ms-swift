# CLAUDE.md

## Project Overview

ms-swift (SWIFT) is a full-pipeline framework for LLM/MLLM fine-tuning, inference, evaluation, and deployment by the ModelScope community. It supports 600+ text models and 400+ multimodal models.

**Version:** 4.2.0.dev0
**Python:** >=3.9
**License:** Apache 2.0

## Quick Reference

```bash
# Install (editable)
pip install -e .

# Core commands
swift sft      # Supervised fine-tuning
swift pt       # Pre-training
swift rlhf     # RLHF (DPO/GRPO/KTO/CPO/PPO/GKD)
swift infer    # Inference
swift deploy   # Deploy as OpenAI-compatible API
swift eval     # Evaluation
swift export   # Quantization/export
swift merge-lora  # Merge LoRA adapters
swift rollout  # Multi-turn rollout for RL
swift web-ui   # Launch Gradio Web UI

# Distributed training (auto-detected via env vars)
NPROC_PER_NODE=4 swift sft --model Qwen/Qwen2.5-7B --dataset alpaca
```

## Architecture

```
swift/
├── cli/                # CLI entry points, dispatches to pipelines
├── pipelines/          # High-level orchestrators (train/infer/eval/export)
│   ├── base.py         # SwiftPipeline base class
│   ├── train/          # SwiftSft, SwiftRLHF, SwiftPretrain
│   └── infer/          # SwiftInfer, SwiftDeploy
├── arguments/          # Dataclass hierarchy for all config
│   ├── base_args/      # ModelArgs, DataArgs, TemplateArgs
│   ├── sft_args.py     # SftArguments
│   └── rlhf_args.py    # RLHFArguments (extends SftArguments)
├── model/              # Model registry + per-vendor loaders (models/)
├── template/           # Chat template system (tokenization, formatting)
├── dataset/            # Dataset loading, preprocessing, packing
├── trainers/           # HF Trainer subclasses + TrainerFactory
├── rlhf_trainers/      # RLHF trainers (DPO, GRPO, KTO, PPO, GKD, etc.)
├── tuners/             # Custom tuner implementations (LoRA, adapter, ReFT)
├── tuner_plugin/       # Pluggable tuner interface
├── infer_engine/       # Inference backends (transformers, vllm, sglang, lmdeploy)
├── megatron/           # Megatron-LM integration (TP/PP/CP/EP)
├── ray/                # Ray distributed training
├── rewards/            # Reward functions for GRPO/RL
├── rollout/            # Multi-turn rollout for RL
├── sequence_parallel/  # Ulysses + Ring Attention
├── loss/               # Loss functions
├── config/             # Preset DeepSpeed/FSDP JSON configs
└── utils/              # Shared utilities
```

## Key Design Patterns

- **Registry pattern**: `MODEL_MAPPING`, `DATASET_MAPPING`, `TEMPLATE_MAPPING` — extend via `register_model()`, `register_dataset()`, `register_template()`
- **Factory pattern**: `TrainerFactory` selects trainer class by `task_type` or `rlhf_type`
- **Mixin composition**: `TunerMixin`, `RLHFMixin`, `RolloutMixin`, `SwiftMixin`
- **Lazy module loading**: `_LazyModule` in `__init__.py` to minimize import time

## Configuration

Three ways to configure:
1. **CLI args**: `swift sft --model X --dataset Y --lora_rank 8`
2. **YAML/JSON config**: `swift sft config.yaml` (first positional arg)
3. **Python API**: `from swift.pipelines import sft_main; sft_main(SftArguments(...))`

DeepSpeed presets: `--deepspeed zero0|zero1|zero2|zero3|zero2_offload|zero3_offload`
FSDP preset: `--fsdp fsdp2`

## Arguments Hierarchy

```
RayArguments
    └── ModelArguments + TemplateArguments + DataArguments + QuantizeArguments + GenerationArguments
        └── BaseArguments
            └── TunerArguments
                └── SftArguments (+ Seq2SeqTrainingArguments)
                    └── RLHFArguments (+ RewardModelArguments + PPOArguments + GRPOArguments + TeacherModelArguments)
```

## TrainerFactory Mapping

| rlhf_type / task_type | Trainer Class |
|---|---|
| `causal_lm` | `Seq2SeqTrainer` |
| `dpo` | `DPOTrainer` |
| `grpo` | `GRPOTrainer` |
| `gkd` | `GKDTrainer` |
| `kto` | `KTOTrainer` |
| `ppo` | `PPOTrainer` |
| `rm` | `RewardTrainer` |
| `cpo` | `CPOTrainer` |
| `orpo` | `ORPOTrainer` |

## Distributed Training

- **DDP**: native PyTorch (auto via `NPROC_PER_NODE`)
- **DeepSpeed ZeRO 0-3**: `--deepspeed zero2`
- **FSDP2**: `--fsdp fsdp2`
- **Megatron**: `megatron sft/pt/rlhf` (TP/PP/CP/EP)
- **Ray**: for multi-node orchestration
- **Sequence Parallel**: Ulysses + Ring Attention

## Inference Backends

`--infer_backend transformers|vllm|sglang|lmdeploy`

## Key Dependencies

| Package | Role |
|---|---|
| `transformers` (>=4.33) | Core LLM framework |
| `peft` (>=0.11) | LoRA / parameter-efficient tuning |
| `trl` (>=0.15) | RLHF algorithms |
| `datasets` (>=3.0) | Dataset loading |
| `accelerate` | Distributed training |
| `modelscope` (>=1.23) | Model/dataset hub |

## Build & Test

```bash
make whl      # Build wheel
make test     # Run tests (requires GPU, uses Docker CI)
make linter   # Run linter
make docs     # Build documentation
```

Tests are in `tests/` mirroring source structure. Run with:
```bash
python tests/run.py --parallel 2 --run_config tests/run_config.yaml
```

## Examples

Examples are organized in `examples/`:
- `examples/train/` — Training examples (SFT, RLHF, multimodal, etc.)
- `examples/infer/` — Inference examples
- `examples/deploy/` — Deployment (vLLM, SGLang, LMDeploy)
- `examples/eval/` — Evaluation examples
- `examples/export/` — Quantization/export
- `examples/megatron/` — Megatron parallelism
- `examples/yaml/` — YAML config examples
- `examples/models/` — Per-model scripts

## Documentation

- English: `docs/source_en/`
- Chinese: `docs/source/`
- Published: https://swift.readthedocs.io/

## CI

GitHub Actions workflows in `.github/workflows/`:
- `citest.yaml` — GPU test suite on self-hosted runners
- `citest_npu.yaml` — NPU tests
- `lint.yaml` — Linting
- `publish.yaml` — PyPI publish
