"""Trajectory collection CLI for Skill-Informed OPSD (Task 5.1).

Runs the current policy (base model + optional LoRA adapter) on a batch of
problems, samples one reasoning trajectory per problem, extracts the final
answer with `\\boxed{...}` regex, and compares against ground-truth.  The
output JSONL is schema-compatible with Task 2.1's ``skill_extract.py``:

    {"question_id", "question", "trajectory", "is_correct",
     "ground_truth", "extracted_answer"}

Typical use (inside the loop `run_full_loop.sh`, Task 5.3)::

    python collect_trajectories.py \\
        --model Qwen/Qwen3-4B \\
        --adapters ckpt_phase3/checkpoint-100 \\
        --dataset problems.jsonl \\
        --output round1_trajectories.jsonl \\
        --num_samples 200

References:
    - OPSD/eval/evaluate_math.py (vLLM + LoRARequest idiom)
    - ms-swift/swift/infer_engine/vllm_engine.py (`enable_lora`, `max_lora_rank`)
    - skill_opsd/skill_extract.py (input schema for the downstream consumer)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("collect_trajectories")


# ---------------------------------------------------------------------------
# Answer extraction / normalization
# ---------------------------------------------------------------------------

# Matches the last \boxed{...} with balanced-brace support via greedy capture.
# For deeply nested boxes (e.g. \boxed{\frac{1}{2}}) we prefer a scanner-based
# extractor (see `extract_boxed_answer`).  This simple regex is kept around for
# unit-testing against the plan's spec `\\boxed\\{([^}]+)\\}`.
ANSWER_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_boxed_answer(text: str) -> Optional[str]:
    """Return the content of the **last** ``\\boxed{...}`` in *text*.

    Supports balanced braces (needed for expressions like ``\\boxed{\\frac{1}{2}}``).
    Falls back to the simple ANSWER_RE regex on non-nested cases.  Returns
    ``None`` if no box is present.
    """
    if not text:
        return None

    # Walk from the right to grab the *last* box.
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    # Find the opening brace after `\boxed`.
    brace_start = text.find("{", idx)
    if brace_start < 0:
        return None
    depth = 0
    i = brace_start
    close_idx = -1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
        i += 1
    if close_idx < 0:
        return None
    return text[brace_start + 1:close_idx].strip()


def normalize_answer(ans: Optional[str]) -> str:
    """Normalise an answer for tolerant string comparison.

    Steps: strip, remove surrounding ``$``, collapse whitespace, collapse
    double backslashes, lower-case.  Designed to be forgiving (e.g. so that
    ``"\\boxed{ 42 }"`` matches ``"42"``) without attempting full symbolic
    equivalence — that's a future math-verify hook.
    """
    if ans is None:
        return ""
    s = str(ans).strip()
    # Strip dollar delimiters sometimes used in LaTeX ground truths.
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    # Collapse redundant whitespace.
    s = re.sub(r"\s+", "", s)
    # Normalise double backslash to single backslash (LaTeX JSON-encoding artefact).
    s = s.replace("\\\\", "\\")
    # Strip trailing period (some GTs end with '.').
    s = s.rstrip(".")
    return s.lower()


def answers_match(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """Tolerant equality check between an extracted answer and the gold answer.

    Never raises — unexpected types log a warning and return False.
    """
    try:
        return bool(predicted is not None and normalize_answer(predicted) == normalize_answer(ground_truth))
    except Exception as exc:  # pragma: no cover - belt-and-braces
        LOGGER.warning("answers_match failed (%r vs %r): %s", predicted, ground_truth, exc)
        return False


# ---------------------------------------------------------------------------
# Data IO
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """Normalised input problem row."""

    question_id: str
    question: str
    ground_truth: str

    @classmethod
    def from_raw(cls, obj: Dict[str, Any], fallback_idx: int) -> "Problem":
        qid = obj.get("question_id")
        if qid is None:
            qid = obj.get("id") or obj.get("idx") or obj.get("problem_idx")
        qid = str(qid) if qid is not None else f"prob_{fallback_idx:05d}"

        question = str(
            obj.get("question")
            or obj.get("problem")
            or obj.get("prompt")
            or obj.get("query")
            or ""
        )
        ground_truth = obj.get("ground_truth")
        if ground_truth is None:
            ground_truth = (
                obj.get("answer")
                or obj.get("final_answer")
                or obj.get("gold")
                or obj.get("answer_gt")
                or ""
            )
        return cls(
            question_id=qid,
            question=question,
            ground_truth=str(ground_truth),
        )


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file; return empty list if missing.  Skips malformed lines."""
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOGGER.warning("skipping malformed JSONL line %d in %s: %s", line_no, path, exc)
    return rows


def atomic_write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Write *rows* to *path* atomically (tmp + ``os.replace``).  Creates parent dirs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(os.path.abspath(path)) or ".",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def subsample_problems(
    problems: Sequence[Problem], num_samples: int, seed: int, shuffle: bool
) -> List[Problem]:
    """Subsample up to *num_samples* problems.

    If *shuffle* is True, shuffles deterministically with *seed* before taking
    the first *num_samples*.  Otherwise simply returns the head slice (stable /
    reproducible across runs without reshuffling).
    """
    if num_samples <= 0 or num_samples >= len(problems):
        return list(problems)
    if shuffle:
        rng = random.Random(seed)
        shuffled = list(problems)
        rng.shuffle(shuffled)
        return shuffled[:num_samples]
    return list(problems[:num_samples])


# ---------------------------------------------------------------------------
# Generation engines
# ---------------------------------------------------------------------------


class GenerationEngine:
    """Thin interface every generation backend must implement."""

    def generate(self, prompts: Sequence[str]) -> List[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        """Release any GPU memory / subprocesses.  Safe to call more than once."""
        return None


class MockEngine(GenerationEngine):
    """Deterministic mock engine for `--dry_run`.  Returns ``mock_text`` for every prompt."""

    def __init__(self, mock_text: str = "Thinking... the final answer is \\boxed{42}."):
        self.mock_text = mock_text
        self._calls = 0

    def generate(self, prompts: Sequence[str]) -> List[str]:
        self._calls += 1
        return [self.mock_text for _ in prompts]


class VLLMEngine(GenerationEngine):
    """vLLM-based engine.  Lazy imports so ``--help`` stays fast."""

    def __init__(
        self,
        model: str,
        adapters: Optional[str],
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        seed: int,
        max_lora_rank: int = 64,
        gpu_memory_utilization: float = 0.7,
        max_model_len: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        enable_thinking: bool = True,
    ) -> None:
        # Local heavy imports.
        from vllm import LLM, SamplingParams  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        self._SamplingParams = SamplingParams

        llm_kwargs: Dict[str, Any] = {
            "model": model,
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": "bfloat16",
            "seed": seed,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.lora_request = None
        if adapters:
            from vllm.lora.request import LoRARequest  # type: ignore

            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank
            llm_kwargs["max_loras"] = 1
            # Resolve once; vLLM accepts directories containing adapter_model.safetensors.
            adapter_path = os.path.abspath(adapters)
            if not os.path.isdir(adapter_path):
                raise FileNotFoundError(f"Adapter path not found or not a directory: {adapter_path}")
            self.lora_request = LoRARequest("policy_lora", 1, adapter_path)
            LOGGER.info("LoRA enabled at %s (max_lora_rank=%d)", adapter_path, max_lora_rank)

        LOGGER.info("initialising vLLM LLM(%s)", model)
        self.llm = LLM(**llm_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model, trust_remote_code=True
        )
        self.enable_thinking = enable_thinking

        self.sampling_params = self._SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k is not None and top_k > 0 else -1,
            max_tokens=max_new_tokens,
            n=1,
            seed=seed,
        )

    def _build_prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        # Qwen-style tokenizers support `enable_thinking`; other models ignore the kwarg.
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def generate(self, prompts: Sequence[str]) -> List[str]:
        formatted = [self._build_prompt(q) for q in prompts]
        kwargs: Dict[str, Any] = {"use_tqdm": False}
        if self.lora_request is not None:
            kwargs["lora_request"] = self.lora_request
        outputs = self.llm.generate(formatted, self.sampling_params, **kwargs)
        # Preserve input order (vLLM may reorder by default but `llm.generate`
        # returns results in the same order as the `prompts` arg).
        return [o.outputs[0].text for o in outputs]

    def close(self) -> None:
        # Best effort: drop the engine reference and empty the CUDA allocator.
        try:
            del self.llm
        except Exception:  # pragma: no cover
            pass
        self.llm = None  # type: ignore[assignment]
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


class HFEngine(GenerationEngine):
    """Transformers fallback engine (slower but dependency-free)."""

    def __init__(
        self,
        model: str,
        adapters: Optional[str],
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        batch_size: int,
        seed: int,
        tokenizer_path: Optional[str] = None,
        enable_thinking: bool = True,
    ) -> None:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self._torch = torch
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        self.seed = seed

        LOGGER.info("loading HF tokenizer/model %s", model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype, device_map=device, trust_remote_code=True
        )

        if adapters:
            from peft import PeftModel  # type: ignore

            adapter_path = os.path.abspath(adapters)
            if not os.path.isdir(adapter_path):
                raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
            LOGGER.info("loading LoRA adapter from %s", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        self.device = device
        torch.manual_seed(seed)

    def _build_prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def generate(self, prompts: Sequence[str]) -> List[str]:
        torch = self._torch
        outputs: List[str] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = [self._build_prompt(p) for p in prompts[start:start + self.batch_size]]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            input_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0.0,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k if self.top_k and self.top_k > 0 else 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for row in gen:
                new_tokens = row[input_len:]
                outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outputs

    def close(self) -> None:
        try:
            del self.model
        except Exception:  # pragma: no cover
            pass
        self.model = None  # type: ignore[assignment]
        gc.collect()
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _detect_vllm() -> bool:
    """Return True if ``import vllm`` succeeds (does not instantiate anything)."""
    try:
        import vllm  # noqa: F401  # type: ignore

        return True
    except Exception:
        return False


def pick_engine(
    preference: str,
    *,
    model: str,
    adapters: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    max_lora_rank: int,
    enable_thinking: bool,
) -> GenerationEngine:
    """Resolve and build the generation engine per *preference* (``vllm``/``hf``/``auto``)."""
    pref = preference.lower()
    if pref == "vllm":
        return VLLMEngine(
            model=model,
            adapters=adapters,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            seed=seed,
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_thinking=enable_thinking,
        )
    if pref == "hf":
        return HFEngine(
            model=model,
            adapters=adapters,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            seed=seed,
            enable_thinking=enable_thinking,
        )
    # auto
    if _detect_vllm():
        LOGGER.info("engine=auto -> vllm detected, using vLLM")
        return VLLMEngine(
            model=model,
            adapters=adapters,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            seed=seed,
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_thinking=enable_thinking,
        )
    LOGGER.info("engine=auto -> vLLM unavailable, falling back to HF transformers")
    return HFEngine(
        model=model,
        adapters=adapters,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        seed=seed,
        enable_thinking=enable_thinking,
    )


def _score_row(problem: Problem, trajectory: str) -> Tuple[str, bool]:
    """Return (extracted_answer, is_correct) for one trajectory."""
    extracted = extract_boxed_answer(trajectory) or ""
    return extracted, answers_match(extracted, problem.ground_truth)


def run_collection(
    problems: Sequence[Problem],
    engine: GenerationEngine,
    batch_size: int,
    log_every: int,
) -> List[Dict[str, Any]]:
    """Drive batched generation + scoring; return output rows in input order."""
    out_rows: List[Dict[str, Any]] = []
    total = len(problems)
    done = 0
    for start in range(0, total, batch_size):
        chunk = list(problems[start:start + batch_size])
        try:
            gens = engine.generate([p.question for p in chunk])
        except Exception as exc:
            LOGGER.warning(
                "batch generation failed at %d..%d (%s); recording empty trajectories",
                start, start + len(chunk), exc,
            )
            gens = [""] * len(chunk)

        if len(gens) != len(chunk):
            LOGGER.warning(
                "engine returned %d outputs for %d prompts; padding/truncating",
                len(gens), len(chunk),
            )
            # Pad / truncate to match input length to keep schema aligned.
            if len(gens) < len(chunk):
                gens = list(gens) + [""] * (len(chunk) - len(gens))
            else:
                gens = gens[:len(chunk)]

        for prob, traj in zip(chunk, gens):
            extracted, is_correct = _score_row(prob, traj)
            out_rows.append({
                "question_id": prob.question_id,
                "question": prob.question,
                "trajectory": traj or "",
                "is_correct": bool(is_correct),
                "ground_truth": prob.ground_truth,
                "extracted_answer": extracted,
            })
        done += len(chunk)
        if log_every > 0 and (done % log_every == 0 or done == total):
            n_correct = sum(1 for r in out_rows if r["is_correct"])
            LOGGER.info(
                "progress: %d/%d (correct so far: %d, acc=%.3f)",
                done, total, n_correct, n_correct / max(done, 1),
            )
    return out_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collect_trajectories",
        description=(
            "Sample reasoning trajectories from the current policy (base model + "
            "optional LoRA adapter), extract \\boxed{} answers, compare to ground "
            "truth, and emit skill_extract-compatible JSONL."
        ),
    )
    p.add_argument("--model", required=True, help="Base model path or HF id (e.g. Qwen/Qwen3-4B).")
    p.add_argument(
        "--adapters",
        default=None,
        help="Optional LoRA adapter directory (contains adapter_model.safetensors).",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Input JSONL of problems with {question_id, question, ground_truth|answer|final_answer}.",
    )
    p.add_argument("--output", required=True, help="Output JSONL path for trajectories.")
    p.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Subsample first N problems (default 200, matches config.trajectory_collect).",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset deterministically with --seed before subsampling (default: False, head slice).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default 0.7).")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling (default 0.9).")
    p.add_argument("--top_k", type=int, default=20, help="Top-k sampling (default 20, <=0 disables).")
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max generated tokens per problem (default 2048).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Prompts per generation call (default 8).  vLLM handles scheduling internally anyway.",
    )
    p.add_argument(
        "--engine",
        choices=["vllm", "hf", "auto"],
        default="auto",
        help="Generation backend; auto prefers vLLM if importable (default auto).",
    )
    p.add_argument(
        "--max_lora_rank",
        type=int,
        default=64,
        help="Max LoRA rank hint for vLLM (must be >= adapter rank; default 64 matches LoRA in plan).",
    )
    p.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Override vLLM max_model_len (default: model config default).",
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.7,
        help="vLLM gpu_memory_utilization (default 0.7 to mirror opsd.sh).",
    )
    p.add_argument(
        "--no_thinking",
        action="store_true",
        help="Disable Qwen3 `enable_thinking` chat template flag (default thinking on).",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=16,
        help="Log progress every N completed problems (0 disables; default 16).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Do not load any model; use a mock engine that returns "
            "'\\boxed{42}'; process the first 3 problems only."
        ),
    )
    p.add_argument("--log_level", default="INFO", help="Python logging level (default INFO).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    raw_rows = load_jsonl(args.dataset)
    if not raw_rows:
        LOGGER.error("no problems loaded from %s", args.dataset)
        return 2

    problems = [Problem.from_raw(r, idx) for idx, r in enumerate(raw_rows)]

    effective_num = 3 if args.dry_run else args.num_samples
    problems = subsample_problems(
        problems, effective_num, seed=args.seed, shuffle=args.shuffle
    )
    LOGGER.info(
        "loaded %d problems from %s (subsampled to %d, shuffle=%s)",
        len(raw_rows), args.dataset, len(problems), args.shuffle,
    )

    engine: GenerationEngine
    if args.dry_run:
        LOGGER.info("dry-run: using MockEngine (no GPU, no model load)")
        engine = MockEngine()
    else:
        engine = pick_engine(
            preference=args.engine,
            model=args.model,
            adapters=args.adapters,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_lora_rank=args.max_lora_rank,
            enable_thinking=not args.no_thinking,
        )

    try:
        rows = run_collection(
            problems=problems,
            engine=engine,
            batch_size=args.batch_size,
            log_every=args.log_every,
        )
    finally:
        try:
            engine.close()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("engine.close() raised: %s", exc)
        # Belt-and-braces GPU cleanup so Task 5.3's round-loop doesn't leak memory.
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    atomic_write_jsonl(args.output, rows)

    n_total = len(rows)
    n_correct = sum(1 for r in rows if r["is_correct"])
    n_boxed = sum(1 for r in rows if r["extracted_answer"])
    summary = {
        "problems": n_total,
        "correct": n_correct,
        "accuracy": (n_correct / n_total) if n_total else 0.0,
        "had_boxed_answer": n_boxed,
        "output": args.output,
        "adapters": args.adapters,
        "engine": "mock" if args.dry_run else args.engine,
        "dry_run": args.dry_run,
    }
    print(json.dumps({"collect_trajectories_summary": summary}, ensure_ascii=False))
    LOGGER.info("wrote %d trajectories -> %s (acc=%.3f)", n_total, args.output, summary["accuracy"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
