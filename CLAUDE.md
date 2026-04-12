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

## Experiment Management Rules

All experiments (training + evaluation) must follow strict isolation:

1. **独立脚本**: 每个实验组合（模型 × 超参数变体）必须有独立的 train/eval 脚本，命名格式 `{action}_{model}_{variant}.sh`，例如 `train_qwen3_1.7b_beta0.sh`、`eval_qwen3_1.7b_beta0_step100.sh`
2. **独立路径**: 每个实验的 `--output_dir` 必须唯一，例如 `training_output/qwen3-1.7b-beta0/`，不同实验绝不共享输出目录
3. **独立日志**: 每个实验的 stdout 日志独立保存，例如 `logs/train_1.7b_beta0.log`
4. **独立 wandb**: 每个实验设置独立的 `WANDB_RUN_NAME`，例如 `qwen3-1.7b-beta0-fwd-kl`
5. **独立 tmux**: 每个实验运行在独立的 tmux session 中
6. **独立 GPU**: 单卡实验使用不同 `CUDA_VISIBLE_DEVICES`
7. **独立端口**: 并行 eval 使用不同 `--port`（8001, 8002, ...）避免冲突
8. **禁止复用**: 永远不要编辑已有脚本来跑新实验，必须创建新脚本
