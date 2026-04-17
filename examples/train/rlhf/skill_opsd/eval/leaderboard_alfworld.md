# ALFWorld Leaderboard

**Benchmark**：ALFWorld（embodied text task）
**Metric**：Success Rate (%)
**评测方式**：SkillRL 自带 eval 脚本（优先级 1；命令见脚本）

| Method | Model | Params | Success Rate (%) | Eval Script | Date | Note |
|---|---|---|---|---|---|---|
| SkillRL | Jianwen/Alfworld-7B-RL (Qwen2.5-7B-Instruct base) | 7B | **85.94** | `eval_alfworld_skillrl.sh` | 2026-04-13 | wandb [58zcz9cd](https://wandb.ai/qzheng19-uiuc/opsd-repro/runs/58zcz9cd); val/text/test_score=5.9964；分类：look_at_obj 90.91% (10/11), pick_and_place 92.31% (12/13), pick_clean 75.00% (6/8), pick_cool 81.82% (9/11), pick_heat 71.43% (5/7), pick_two 100.00% (5/5) |
| SkillRL (paper) | 同上 | 7B | **89.9** | — | from paper | 引用 |

## 脚注

### 模型与路径
- 模型规模差异：SkillRL 7B ≠ OPSD/SFT 4B；不可跨 method 直接比较
- Checkpoint 格式：HF 上的 `Jianwen/Alfworld-7B-RL` 是 **verl FSDP shard**（`actor/model_world_size_N_rank_*.pt`），经 FSDP shard merge 后转 HF 格式
- 本地路径：`/data/qzheng19/Alfworld-7B-RL/hf`（merge 后 HF 格式；base=Qwen2.5-7B-Instruct）
- Eval 入口：SkillRL 使用 verl `trainer.val_only=True` + `val_before_train=True` 模式（参考 EvolveR `scripts/test-3b.sh`）

### 2026-04-13 Fairness Check（SkillRL 实测）
- **Run**：wandb `58zcz9cd`，2026-04-13 10:06:01 → 10:24:22（18 min 21 s）
- **Metric**：val/success_rate = **0.8594**（85.94%），val/text/test_score = 5.9964
- **评测工具**：SkillRL 自带 verl eval（优先级 1），非第三方重实现
- **关键参数（SkillRL eval 默认 val_kwargs）**：
  - `val_batch_size=64`
  - `env.rollout.n=8`（每题 8 次采样）
  - `env.max_steps=50`
  - `temperature=0.4`, `do_sample=True`
- **与论文 89.9% 差距**：3.96 pp
- **可能原因**：
  1. 数据 split 不同（我方 val batch 数量 **64** vs 论文 **134**）
  2. 采样 `temperature=0.4` 非 greedy，有随机性（SkillRL 默认 val_kwargs）
  3. Checkpoint merge 阶段 FSDP → HF 可能存在微小精度偏差
- **分类成功率明细**：

  | Task Type | Success | N | Rate |
  |---|---|---|---|
  | look_at_obj_in_light | 10 | 11 | 90.91% |
  | pick_and_place | 12 | 13 | 92.31% |
  | pick_clean_then_place_in_recep | 6 | 8 | 75.00% |
  | pick_cool_then_place_in_recep | 9 | 11 | 81.82% |
  | pick_heat_then_place_in_recep | 5 | 7 | 71.43% |
  | pick_two_obj_and_place | 5 | 5 | 100.00% |

- **跨方法比较注意**：SkillRL 7B 不与 OPSD/SFT 4B 直接比较（model scale 差异）
</content>
</invoke>