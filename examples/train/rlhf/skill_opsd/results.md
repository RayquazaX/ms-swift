# Skill-Informed In-Context Self-Distillation — Results Log

## Environment
- conda env: opsd
- base model: Qwen3-4B (LoRA r=64, alpha=128, all-linear)
- benchmark: aime25 / math_500 / gsm8k
- baseline: OPSD solution-as-teacher, AIME25 = 0.2667 @ 100 steps (Qwen3-4B, beta=0.5, T=1.2, lmbda=1.0, lr=2e-5)
- ms-swift version: 4.2.0.dev0

## Task Progress

### Task 0.1: Clone 参考仓库
- 状态: 通过
- 产出物: /home/qzheng19/{OPSD, SkillRL, reasoning-bank, EvolveR, GenICL_preferred, CEIL, Self-ICL, PromptAgent, SDFT, SDPO}
- 验证结果:
  - 10/10 仓库就位 (commit hash 列表)
    | Repo | Commit |
    |------|--------|
    | OPSD | 0feada9 |
    | SkillRL | 299909b |
    | reasoning-bank | 250f51e |
    | EvolveR | f910ee7 |
    | GenICL_preferred | dc9462d |
    | CEIL | e23c539 |
    | Self-ICL | 3368f35 |
    | PromptAgent | 2edfc9e |
    | SDFT | d775732 |
    | SDPO | c52586b |
  - 5 关键文件均可 Read:
    - OPSD/opsd_trainer.py (token_clip logic)
    - SkillRL/memory_data/alfworld/claude_style_skills.json (skill schema)
    - EvolveR/evolver/experience/config.py (SIMILARITY_THRESHOLD = 0.85)
    - GenICL_preferred/src/train_kto_lora.py (KTOTrainer import)
    - CEIL/src/utils/dpp_map.py (fast_map_dpp function)
  - DSPy 3.1.3 import OK
- 备注:
  - DSPy 安装引入 datasets 4.8.4 (ms-swift 需 <4.0) —— 训练前需 `pip install 'datasets<4.0'`
  - OPSD/SkillRL 是用户自己 fork 而非上游, 行号可能与 IMPLEMENTATION_PLAN.md 有偏移, 后续 Task 须按函数名定位

### Task 0.2: 建立 skill_opsd 工作目录
- 状态: 通过
- 产出物:
  - `/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/` (目录)
  - `/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/config.yaml`
  - `/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/skill_library.jsonl` (空)
  - `/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/results.md` (本文件)
- 验证结果: 3 个文件均创建成功 (详见文件系统验证小节)
- 备注:
  - config.yaml 字段 90%+ 来自 plan 明文规定 (每行标注 `# from plan §X` 引用)
  - 未明文指定的默认值 (如 filter_keep_ratio 阈值) 已标注为 agent-chosen

## Experiment Table (A–I)

| Exp | Config | Metric (AIME25) | Status |
|-----|--------|-----------------|--------|
| A: Baseline OPSD | solution teacher, beta=0.5, T=1.2 | 0.2667 @100 步 | done (人工) |
| B: +原版参数 | beta=0, T=1.1, clip=0.05 | - | pending (Task 1.1-1.3) |
| C: +skill context | Skill-informed context 替代 solution | - | pending (Task 3.3) |
| D: +KL filter | 加 KL 过滤 | - | pending (Task 2.2) |
| E: +CEIL 选择 | DPP 组合选择 | - | pending (Task 3.2) |
| F: +PromptAgent 优化 | 错误反馈优化 context | - | pending (Task 3.3) |
| G: +循环 | 3 轮完整循环 | - | pending (Task 5.3) |
| H: +EMA teacher | 固定→EMA | - | pending (Task 1.2) |
| I: TR/TL 归因 | 三种消融设置 | - | pending (Task 5.2) |

### Task 1.1: 迁移 jsd_token_clip
- 状态: ✅ 通过
- 产出物:
  - /home/qzheng19/ms-swift/swift/arguments/rlhf_args.py (新增 L264 `jsd_token_clip: float = 0.0`)
  - /home/qzheng19/ms-swift/swift/rlhf_trainers/gkd_trainer.py (`generalized_jsd_loss` 提升为 module-level function, L48-L192; `__init__` 加 `self.jsd_token_clip` L226; 4 个 JSD 调用点 L331/L344/L356/L367 传 `token_clip=self.jsd_token_clip`; 类末尾保留 staticmethod 别名 L901-L902 向后兼容)
- 验证结果:
  - `from swift.rlhf_trainers.gkd_trainer import generalized_jsd_loss` 成功
  - `RLHFArguments().jsd_token_clip == 0.0`
  - 12 组合向后兼容 bit-level 测试通过 (beta × temp × topk)
  - `token_clip=0.01` 下 loss 从 0.179 → 0.064, clip 生效
  - `swift rlhf --help` 中可见 `--jsd_token_clip` 注册
- 备注:
  - ms-swift 另有 Megatron 路径的 gkd_trainer.py, 本次未触碰 (OPSD 流程走非 Megatron 路径)
  - Task 2.2 将 import `generalized_jsd_loss` 做 skill 价值打分

### Task 2.1: skill_extract.py
- 状态: ✅ 通过
- 产出物:
  - /home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/skill_extract.py (~25 KB)
  - /home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/run_phase1.sh (+x)
- 验证结果:
  - 语法 OK
  - dry-run 单测: 5 条 mock 轨迹, cosine 0.967 合并成功, 新 skill 赋 skill_0002
  - cosine 单测: 向量 [1,2,3]·[4,5,6]=0.974632 与手算一致
  - --help 打印 11 参数
  - run_phase1.sh bash syntax OK
- 备注:
  - GPU 内存预估: Qwen3-4B bf16 ≈ 10 GB + all-MiniLM-L6-v2 ~90 MB
  - 单轨迹耗时估算: 8-12 s, 200 条 ≈ 30-40 min
  - 健壮 LLM 解析: json.loads → regex → 逐 object 扫描 → warn skip
  - Task 5.3 每轮可 append 语义重复调用

### Task 3.1: self_icl_bootstrap.py
- 状态: ✅ 通过
- 产出物:
  - /home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/self_icl_bootstrap.py (~17 KB)
- 验证结果:
  - 语法 OK
  - 冷启动路径 (空库 + dry-run): 生成 20 条 pseudo skill, source="pseudo", skill_id `pseudo_001..pseudo_020`
  - Skip 路径 (12 条现有 skill ≥ min 10): early-return + 库未动
  - --force 路径: 绕过 skip, 基于 `succ_012` 最大 id 起号 `pseudo_013`
  - --help 全 12 参数
- 备注:
  - 与 Task 2.1 共存: source="pseudo" 前缀 `pseudo_`, Task 2.1 用 `succ_/fail_`; next_skill_id() 扫最大数字后缀避免冲突
  - 延迟 import transformers/torch, --dry_run 与 --help 不触 GPU
  - 原子写入 (tmp + os.replace) 不破坏 Task 2.1 已写条目
  - Prompt 模板顶层常量可被 Task 3.3 monkey-patch 用作 PromptAgent 优化

---

**Task 1.2: EMA Teacher 支持**
- 状态：✅ 通过
- 产出物：
  - /home/qzheng19/ms-swift/swift/arguments/rlhf_args.py（L265-L267 新增 `use_ema_teacher: bool=False`、`ema_decay: float=0.999`）
  - /home/qzheng19/ms-swift/swift/rlhf_trainers/gkd_trainer.py（L16 import TrainerCallback；L209-L231 新增模块级 `EMAUpdateCallback`；L268-L297 `__init__` 注入 EMA 配置 + 互斥校验 + add_callback + log；L304-L371 `_update_ema`；L373-L434 `@contextmanager _ema_teacher_context`；L659-L670 `compute_loss` 自蒸馏分支条件切换）
- 验证结果：
  - `RLHFArguments().{use_ema_teacher, ema_decay, jsd_token_clip}` → `(False, 0.999, 0.0)`
  - `from swift.rlhf_trainers.gkd_trainer import EMAUpdateCallback, GKDTrainer, generalized_jsd_loss` OK，staticmethod 别名保留
  - 向后兼容：`use_ema_teacher=False` 时不打印 EMA 日志
  - 互斥策略：`use_ema_teacher=True` + `teacher_use_disable_adapter=True` 抛 ValueError（严策略）
  - `swift rlhf --help` 中见 `--use_ema_teacher`、`--ema_decay`
- 备注：
  - deepspeed lazy import，非 ZeRO-3 环境不触发
  - ms-swift callback 风格：`self.trainer = trainer` 构造注入
  - Task 1.1 的 4 个 JSD 调用点未破坏（L521/534/546/557）、staticmethod 别名（L1098-1099）保留
  - 未改 Megatron 路径

**Task 1.3: 保守采样参数对齐**
- 状态：✅ 通过
- 产出物：/home/qzheng19/ms-swift/examples/train/rlhf/opsd/opsd.sh（net +5 / -2 行）
- 验证结果：
  - bash syntax OK
  - diff：`--beta 0.5 → --beta 0`、`--temperature 1.2 → --temperature 1.1`、新增 `--top_p 0.95 --top_k 20 --jsd_token_clip 0.05`
  - vLLM colocate 支持 top_p/top_k 结论：**支持**
    - swift/rlhf_trainers/args_mixin.py:154-155 RolloutTrainerArgumentsMixin 注册字段
    - swift/rlhf_trainers/rollout_mixin.py:118-119 `_prepare_rollout_params` 塞入 RequestConfig
    - swift/infer_engine/vllm_engine.py:410-417, 450 透传给 SamplingParams
- 备注：与 OPSD 原推荐值完全对齐（run_opsd_4b.sh:32-33）

**Task 2.2: skill_filter.py (KL-based)**
- 状态：✅ 通过
- 产出物：/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/skill_filter.py (22.3 KB)
- 验证结果：
  - AST OK
  - Import `generalized_jsd_loss` from swift.rlhf_trainers.gkd_trainer OK
  - dry-run 5 条 mock skill：threshold=0.01 全保,threshold=0.025 保 3 丢 2，降序排序
  - 覆盖保护：不传 `--output` 时先 `.bak`
  - `--help` 列出 13 个 flag
- Logits 对齐策略:末尾对齐 + eval_tail_tokens=128（默认）——teacher/student prompt 末尾部分完全相同，是天然的 token-wise 对齐区域
- 备注：
  - GPU 内存估：Qwen3-4B bf16 + PEFT + max_length=4096 ≈ 12-14 GB
  - 每 skill 耗时：30 val × 2 forward × ~200 ms ≈ 12 s；80 skill 约 16 min/轮
  - live 模式须显式传 `--adapter` 当前 ckpt，否则 teacher==student，KL≈0 全过滤
  - Task 5.3 若循环内多次 CLI 调用会重复加载模型，建议后续改为 Python import 复用

**Task 3.2: context_select.py**
- 状态：✅ 通过
- 产出物：/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/context_select.py (~390 行)
- 验证结果：
  - AST OK
  - CEIL dry-run 20 skills × 5 questions：每题 top-3 唯一 skill_id
  - GenICL 无效路径 fail-fast（exit=1 with 清晰错误）
  - 空 skill_library：selected_skill_ids=[]，exit 0
  - top_k=10 但库只有 3：自动 min 到 3，不崩
  - `--help` 全参数
  - fast_map_dpp 单测（5×5 kernel diag=[9,1,8,1,7]）：top-3={0,2,4}，top-1=[0]，top-50 clamp 5
- fast_map_dpp 引入策略：**内嵌复制**（L60-94，25 行）——避免拉入 `dppy.finite_dpps` 重依赖
- 备注：
  - GenICL selector 需预训（KTO LoRA），支持 PEFT adapter 目录或完整 HF 模型
  - 未预训 GenICL 退化为 base LM conditional likelihood，建议默认用 CEIL
  - 打分：query + skill_text 输入 LM，只对 skill 段算 loss，取平均 log-prob

---

**Task 3.3: context_builder.py**
- 状态：✅ 通过
- 产出物：/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/context_builder.py (~490 行)
- 验证结果：
  - AST OK
  - dry-run 无 optimize：5 行 JSONL，≥1 行含 Framework+Pitfalls，≥1 行含 Heuristics（pseudo 源）
  - 空 skill_library 退化：teacher_prompt 仅 transition + Problem + Solution
  - `--optimize_context --dry_run`：3 轮贪心跑完，round 1 改进 0.5→1.0，round 2/3 保留 baseline（不退化不变量）
  - `--help` 全部显示
  - 未知 source 兜底归入 Framework + warn
- teacher_prompt layout：`# Reasoning Guidance` / `## [Reasoning Framework]` / `## [Common Pitfalls]` / `## [Exploratory Heuristics]` (可选) / `---` / OPSD transition / `Problem:` / `Solution:`
- OPSD transition prompt 来源：importlib.util 动态加载 `/home/qzheng19/ms-swift/examples/train/rlhf/opsd/opsd_plugin.py` L17-18 TRANSITION_PROMPT；失败 fallback 到 verbatim 字符串
- PromptAgent 简化策略：per-template 三段 intro 优化（非 per-question），只改 intro 行不动 header，保持下游解析稳定；严格优于才替换
- 备注：
  - 训练循环 G_full_loop 外默认 `--optimize_context=false`
  - `--optimize_context` 需配合 `--llm_model`，会与训练抢 GPU，建议外挂为独立 preprocessing 步骤
  - 下游 Task 3.4 plugin 只读 `teacher_prompt` + `messages`，`question_id` 保留用于 loop 追踪

**Task 3.4: skill_opsd_plugin.py**
- 状态：✅ 通过
- 产出物：/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/skill_opsd_plugin.py
- 验证结果：
  - 语法 OK
  - Import 成功，`DATASET_NAME='skill_opsd_local'`，`SkillOPSDPassthroughPreprocessor` 可见
  - Mock 3 行 JSONL → preprocessor 返回 `['messages', 'teacher_prompt']`
  - `swift rlhf --help --external_plugins ./skill_opsd_plugin.py` 打印完整 help + log `skill_opsd plugin registered`
  - Bonus: `swift.dataset.load_dataset([jsonl_abs_path])` 返回 (train, val)，basename-suffix 路由跑通
- 与 opsd_plugin.py 的差异：
  - 数据来源：JSONL basename 注册（默认 `train_with_context.jsonl` / `skill_opsd_train.jsonl` / `skill_opsd_val.jsonl`，可通过 env var `SKILL_OPSD_JSONL_BASENAMES` 扩展）
  - 字段组装：pure pass-through（Task 3.3 已把分层 + transition + question 全部拼好）
  - 删除 SYSTEM_PROMPT / TRANSITION_PROMPT 常量（不与 context_builder 重复真源）
  - 脏行防御：缺 teacher_prompt / 无 user turn 的行 drop + warn（traceback_limit 限流）
- 备注：
  - `GKDTrainer._build_opsd_teacher_data` (swift/rlhf_trainers/gkd_trainer.py:467) 会自动读 teacher_prompt 并替换最后一条 user turn
  - Task 4.1 调用：`--external_plugins .../skill_opsd_plugin.py --dataset <abs_path>/train_with_context.jsonl`

**Task 4.1: run_phase3.sh**
- 状态：✅ 通过
- 产出物：/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/run_phase3.sh (7695 B, +x)
- 验证结果：
  - bash syntax OK（含覆盖 env var）
  - 缺 TRAIN_JSONL / PLUGIN_PATH 分支 early exit 1 + 清晰错误
  - `grep` 参数 sanity：beta=0, temperature=1.1, top_p=0.95, top_k=20, jsd_token_clip=0.05, lora_rank=64, lora_alpha=128, max_steps=300
  - chmod +x 正确
- 与 opsd.sh 的 diff：
  - 仅语义差异：`--external_plugins skill_opsd_plugin.py`、`--dataset ${TRAIN_JSONL}`、`--output_dir ${WORKSPACE}/ckpt_phase3`
  - 新增包装层：`set -euo pipefail`、env var 默认值、conda 激活、预检查、`trap cleanup EXIT` 自动 `ray stop --force`、`WANDB_DISABLE_STATS=true`、`"$@"` passthrough
  - 所有其他超参（lr=2e-5, batch_size=4, lora_rank=64, max_length=8192, deepspeed zero0, flash_attn 等）逐行对齐 opsd.sh
- env var 默认：CONDA_ENV=opsd, WORKSPACE=.../skill_opsd, MAX_STEPS=300, TEMPERATURE=1.1, TOP_P=0.95, TOP_K=20, BETA=0, JSD_TOKEN_CLIP=0.05, USE_EMA=false, EMA_DECAY=0.999, NPROC_PER_NODE=8, CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
- 备注：
  - Task 5.3 建议 sanity 先 `MAX_STEPS=50 bash run_phase3.sh`
  - 循环时每轮覆盖 OUTPUT_DIR=ckpt_phase3_round{N}、WANDB_RUN_NAME=skill_opsd_round{N}
  - 实验 H (EMA teacher) 直接 `USE_EMA=true EMA_DECAY=0.999 bash run_phase3.sh` 切换

---

**Task 5.1: collect_trajectories.py**
- 状态：✅ 通过
- 产出物：/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/collect_trajectories.py (801 行)
- 验证结果：
  - AST OK
  - dry-run 3 条 mock 输入：q1(gt=42)=True，q2(gt=6)=False，q3(gt=x+1)=False；accuracy=0.333
  - 正则单测：`\boxed{5}`→"5"，双 box→取最后一个 "2"，无 box→None，嵌套 `\frac{1}{2}`→"\\frac{1}{2}"
  - Normalize：`" 42 "` vs `"42"` match；None→False 不 crash
  - `--help` 18 参数
  - opsd env 内 `from vllm import LLM, SamplingParams, LoRARequest` 全部 import OK（vllm 0.11.0）
- 推理 backend：vllm（优先）/ hf / auto / mock（dry_run）
- 答案提取：从右往左找 `\boxed` + 大括号平衡扫描（支持嵌套 `\boxed{\frac{1}{2}}`）
- 备注：
  - finally 块：engine.close() → del llm + gc.collect() + torch.cuda.empty_cache()
  - 循环时建议 `round${N}_trajectories.jsonl` 命名
  - vLLM 0.11.0 colocate 需 `ray stop --force` 兜底
  - batch-level try/except 捕获 generate 异常 → 整 batch 记 trajectory="" is_correct=False
  - `--seed` 同时传给 vLLM engine 和 SamplingParams，跨轮可复现

**Task 5.2: tr_tl_diagnostic.py**
- 状态：✅ 通过
- 产出物：/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/tr_tl_diagnostic.py
- 验证结果：
  - AST OK
  - dry-run 20 skills × 15 eval × ks=[1,3,5,8]：生成 195 条 prompt = 15 × (1 + 3×4) ✓
  - Prompt 结构断言：
    - A 含 `[Reasoning Framework]` AND `[Common Pitfalls]`
    - B 含 headers AND `<REDACTED>` AND 原 principle 文本不泄漏
    - C 不含 `[...]` header，但含原 principle
  - TL/TR 数学：mock A=0.5 B=0.4 baseline=0.3 → TL=0.1 TR=0.1
  - k=8 on |lib|=2 → effective_k=2，不崩
  - `--help` 14 参数
- 三设置生成方式：
  - A full：调 Task 3.3 `build_teacher_prompt` + 默认 SectionTemplates
  - B structure-only：深拷贝 skill，principle/when_to_apply 替换为 N 个 `<REDACTED>`（N=原词数，长度近似）
  - C content-only：只串联 principle + when_to_apply，双换行分隔，保留 `---/transition/Problem/Solution` 尾部
- 推理复用：importlib.util 动态加载 collect_trajectories.py（best-effort）；失败兜底 HF transformers + PEFT + 本地 `\boxed{}` 判题
- TL/TR 定义（report notes 字段）：`TL = A - B (content contribution); TR = B - baseline (structure contribution)`
- 备注：
  - 若某轮 TL 单调衰减 → 判定"内容收益饱和，切 PromptAgent 优化"
  - `--config` 自动读 training.temperature / max_completion_length / skill_extraction.embedding_model，减少 shell 侧重复

---

**Task 5.3: run_full_loop.sh**
- 状态：✅ 通过
- 产出物：/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/run_full_loop.sh (15,696 B, +x)
- 验证结果：
  - bash -n syntax OK
  - 缺 SEED_QUESTIONS → 清晰报错 + exit 1 + cleanup trap 触发
  - 模拟首轮（无 adapter）：`Step 4/10: skill_filter SKIPPED (round=1, no adapter)`；results.md append 一条；10 步顺序 5.1→2.1→self_icl→skill_filter(skip)→3.2→3.3→4.1→eval→tr_tl→results
  - 模拟第二轮（mock round1 checkpoint）：skill_filter 调用 `--adapter .../ckpt_phase3_round1/checkpoint-100`；collect_trajectories 传 `--adapters` 到前轮 ckpt
  - `swift eval --help` 确认 6 个 CLI flag 全部存在（--adapters / --infer_backend / --eval_dataset / --eval_limit / --eval_output_dir / --eval_backend）
  - 计划 §Task 5.3 要求顺序 `5.1→2.1→2.2→3.2→3.3→4.1→eval` 与脚本 Step 1,2,4,5,6,7,8 对齐
  - chmod +x 正确
- env var 默认：N_ROUNDS=3, STEPS_PER_ROUND=100, NUM_SAMPLES=200, TOP_K=3, RUN_TR_TL=false, SKIP_SELF_ICL=false, EVAL_DATASETS='aime25 math_500 gsm8k', EVAL_BACKEND=Native, EVAL_INFER_BACKEND=vllm, WANDB_DISABLE_STATS=true
- swift eval 调用形式：`swift eval --model $BASE_MODEL --adapters $NEW_ADAPTER --eval_backend Native --infer_backend vllm --eval_dataset $EVAL_DATASETS --eval_output_dir $EVAL_OUT_DIR [--eval_limit N]`（代码证据：/home/qzheng19/ms-swift/swift/arguments/eval_args.py:49-63）
- 首/非首轮分支：
  - First (L156-164)：ADAPTER=${INITIAL_ADAPTER:-}；`${ADAPTER:+--adapters ...}` 条件拼接；skill_filter gated by `[$round -gt 1 && -n $ADAPTER]`
  - Later (L161-168)：`ls -d ckpt_phase3_round$((round-1))/checkpoint-* | sort -V | tail -1`；无前轮 ckpt 则 fail-fast
- 失败模式与缓解：
  - GPU OOM training → 降 STEPS_PER_ROUND / NUM_SAMPLES
  - swift eval stall → 切 EVAL_INFER_BACKEND=transformers 或 sglang；或加 EVAL_LIMIT=50 做 smoke
  - Ray/accelerate leaks → trap cleanup EXIT 所有退出路径 ray stop --force
  - self_icl_bootstrap 崩 → 被 `|| true` + WARN 保护，不中断主 loop
  - skill_library 并发 → 单写者（serial loop），append-only，安全
- 备注：
  - 每轮自己目录：`${WORKSPACE}/round${N}/`
  - skill_library.jsonl 全局累积
  - Checkpoint：`${WORKSPACE}/ckpt_phase3_round${N}/`
  - Eval 结果：`${ROUND_DIR}/eval_${dataset}/`
  - TR/TL 报告（可选）：`${ROUND_DIR}/tr_tl_report.json`

---

## Project Complete ✅

**Timestamp**：2026-04-13 01:56 UTC

**全部 15 个 Task 通过**：0.1, 0.2, 1.1, 1.2, 1.3, 2.1, 2.2, 3.1, 3.2, 3.3, 3.4, 4.1, 5.1, 5.2, 5.3

**Sanity Check**：
- 参考仓库：10/10 存在（OPSD, SkillRL, EvolveR, GenICL_preferred, CEIL, Self-ICL, PromptAgent, SDFT, SDPO, reasoning-bank）
- skill_opsd/ 下 13 个脚本/配置文件全部存在（config.yaml 6,600 B, skill_library.jsonl 0 B, results.md, skill_extract.py 25,119 B, run_phase1.sh 793 B, skill_filter.py 22,340 B, self_icl_bootstrap.py 22,609 B, context_select.py 21,042 B, context_builder.py 28,468 B, skill_opsd_plugin.py 7,867 B, run_phase3.sh 7,695 B, collect_trajectories.py 27,875 B, tr_tl_diagnostic.py 28,368 B, run_full_loop.sh 15,696 B +x）
- ms-swift 核心改动：rlhf_args.py:264/266/267 含 `jsd_token_clip / use_ema_teacher / ema_decay`；gkd_trainer.py 含 module-level `generalized_jsd_loss`（L64）/ `EMAUpdateCallback`（L209）/ `_update_ema`（L326）
- opsd.sh 已对齐保守采样（L50-54）：`--beta 0 --temperature 1.1 --top_p 0.95 --top_k 20 --jsd_token_clip 0.05`
- DSPy：3.1.3（可 import）

**Next Steps（供后续真实执行）**：
1. 准备 SEED_QUESTIONS 数学题 JSONL（含 `{question_id, question, ground_truth}`）
2. 可选：先跑 Baseline OPSD（实验 A）作为 INITIAL_ADAPTER
3. Smoke test：`MAX_STEPS=50 bash run_phase3.sh` 验证训练链路
4. 完整循环：`N_ROUNDS=3 STEPS_PER_ROUND=100 SEED_QUESTIONS=<path> bash run_full_loop.sh`
5. 诊断：`RUN_TR_TL=true bash run_full_loop.sh` 获取 TR/TL 归因报告
6. 评测：每轮末 swift eval aime25/math_500/gsm8k，对照 baseline 0.2667 @AIME25

**已知风险（不阻塞交付，但真跑需注意）**：
- DSPy 把 datasets 升到 4.8.4 可能与 ms-swift 冲突（Task 0.1 已记）：真跑前 `pip install 'datasets<4.0'`
- OPSD/SkillRL 是用户 fork（RayquazaX），行号与上游略不同，但函数名定位均可用
- vLLM 0.11.0 colocate 已验证支持 top_p/top_k 透传
- 每轮 CLI 调用多次加载 LLM，Task 5.3 如果想优化可做常驻进程或 Python import 复用（非紧急）
