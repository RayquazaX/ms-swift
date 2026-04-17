# WebShop Leaderboard

**Benchmark**：WebShop（online shopping text env）
**Metric**：Success Rate (%) / Task Score (WebShop-specific reward, 非严格成功率)
**评测方式**：SkillRL 自带 eval 脚本（优先级 1；命令见脚本）

| Method | Model | Params | Success Rate (%) | Task Score | Eval Script | Date | Note |
|---|---|---|---|---|---|---|---|
| SkillRL | Jianwen/Webshop-7B-RL (Qwen2.5-7B-Instruct base) | 7B | **78.13** | **0.8788** | `eval_webshop_skillrl.sh` | 2026-04-13 | wandb [5m1t345e](https://wandb.ai/qzheng19-uiuc/opsd-repro/runs/5m1t345e); val/text/test_score=6.9734；实测 **高于** 论文 |
| SkillRL (paper) | 同上 | 7B | **72.7** | — | — | from paper | 引用 |

## 脚注

### 模型与路径
- 模型规模差异：SkillRL 7B ≠ OPSD/SFT 4B；不可跨 method 直接比较
- WebShop 需 `./setup.sh -d all` 准备 webshop env；若未准备 eval 会报错，需先 bootstrap env
- Checkpoint 格式：verl FSDP shard → merge 后 HF 格式
- 本地路径：`/data/qzheng19/Webshop-7B-RL/hf`（merge 后 HF 格式；7B）
- Eval 入口：SkillRL 使用 verl `trainer.val_only=True` + `val_before_train=True` 模式

### 2026-04-13 Fairness Check（SkillRL 实测）
- **Run**：wandb `5m1t345e`，2026-04-13 10:33:43 → 10:46:58（13 min 15 s）
- **Metric**：
  - `val/success_rate` = **0.7813**（78.13%，严格成功率）
  - `val/webshop_task_score` = **0.8788**（87.88%，WebShop 自带 reward，非严格成功）
  - `val/text/test_score` = 6.9734
- **与论文 72.7% 对比**：实测 success_rate 78.13% **高于** 论文 72.7%（+5.43 pp）
  - 注意：论文 72.7% 对应的是 success_rate（严格）；task_score (0.879) 是 WebShop 定义的 reward，不能直接与 72.7% 比较
- **评测工具**：SkillRL 自带 verl eval（优先级 1）
- **关键参数（SkillRL eval 默认 val_kwargs）**：
  - `val_batch_size=64`
  - `env.rollout.n=8`
  - `env.max_steps=15`
  - `temperature=0.4`, `do_sample=True`
- **可能高于论文原因**：
  1. 数据 split / seed 不同
  2. Sampling temperature + n=8 多次采样取成功
  3. 论文的 eval protocol 可能更严格（不同 n/steps）
- **跨方法比较注意**：SkillRL 7B 不与 OPSD/SFT 4B 直接比较（model scale 差异）
</content>
</invoke>