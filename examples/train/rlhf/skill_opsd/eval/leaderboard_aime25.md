# AIME 2025 Leaderboard

**Benchmark**：AIME 2025（math competition 30 题）
**Metric**：greedy pass@1（`temperature=0.0, do_sample=false, max_tokens=8192`）
**评测方式**：ms-swift `swift eval --eval_dataset aime25 --eval_backend Native --infer_backend vllm`
**公平性**：所有方法用同一 generation 参数；LoRA 方法 `--vllm_max_lora_rank 64`；MODELSCOPE_CACHE 指向本地 cache

| Method | Model | Params | Pass@1 | Steps | Eval Script | Date | Note |
|---|---|---|---|---|---|---|---|
| Qwen3-4B base | Qwen/Qwen3-4B | 4B | **0.2000 (6/30)** | - | `eval_aime25_base.sh` | 2026-04-13 | 零后训练 |
| SFT (100 steps) | Qwen3-4B + LoRA r64a128 | 4B | _不评测 (user decision 2026-04-13)_ | 100 | `train_sft_baseline.sh` + `eval_aime25_sft.sh` | - | OpenThoughts-114k-math |
| OPSD | Qwen3-4B + LoRA r64a128 | 4B | **0.2667** | 100 | 已预跑 | 2026-04-08 | `/data/qzheng19/opsd_repro/training_output/qwen3-4b/v0-20260408-062501/checkpoint-100/`；beta=0.5, T=1.2 |
| Ours (Skill-OPSD) | Qwen3-4B + LoRA r64a128 | 4B | _待实验_ | - | `eval_aime25_ours.sh` | - | 待训 G 实验 |

## 脚注
- 公平性 check：AIME25 generation 参数统一；LoRA rank=64 一致
- OPSD 结果引用自 2026-04-08 baseline 实验
- 2026-04-13: Qwen3-4B base 实测 0.2000 (6/30)，略高于预期 ≈ 0.1667；SFT 不测（user decision）。
