# HotpotQA Leaderboard

**Benchmark**：HotpotQA（multi-hop QA）
**Metric**：EM / F1
**评测方式**：EvolveR `scripts/test-3b.sh`（val_only=True）

| Method | Model | Params | EM | F1 | Eval Script | Date | Note |
|---|---|---|---|---|---|---|---|
| EvolveR | Edaizi/EvolveR (Qwen2.5-3B base) | 3B | _不评测 (user decision 2026-04-13)_ | _不评测 (user decision 2026-04-13)_ | `eval_hotpotqa_evolver.sh` | 2026-04-13 (skipped) | 论文 avg=0.382（跨 4 数据集） |
| EvolveR (paper) | 同上 | 3B | — | — | — | from paper | avg=**0.382**（NQ/HotpotQA/PopQA/Bamboogle 均值） |

## 脚注

### 模型与路径
- 模型规模差异：EvolveR 3B ≠ OPSD/SFT 4B
- 数据集：`Edaizi/EvolveR-NQ-HotpotQA`（train/test.parquet）
- Checkpoint 路径：`/data/qzheng19/EvolveR/EvolveR-3B/`（RL 主模型，纯 HF safetensors）；`EvolveR-3B-cold_start/` 是 SFT warm-start，不用于主评测

### 2026-04-13：Skipped（user decision）
- **用户决定**：2026-04-13 放弃 HotpotQA 评测
- **原因**：EvolveR eval 依赖栈过重，单跑成本极高
  1. 需要 **独立 conda env**（与主仓库依赖冲突）
  2. 需要 **MilvusDB** 服务
  3. 需要 **Wiki FAISS index ~30 GB** 下载 + 索引
  4. 需要同时拉起 **3 个 service**（retrieval / embedding / milvus）
- 论文值保留：**avg 0.382 是跨 4 数据集（NQ / HotpotQA / PopQA / Bamboogle）均值**，单 HotpotQA 数值需另查论文 test 脚本输出
- Fairness Check：**N/A**（未实测，无法做 fairness 对比）
</content>
</invoke>