#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
self_icl_bootstrap.py
=====================
Task 3.1 — Self-ICL cold-start bootstrap for the skill library.

Given a set of seed math questions and a generator LLM, this script:

  1. For each seed question, prompt the LLM to emit ``K`` style-similar
     pseudo questions.
  2. Zero-shot solve each pseudo question with the same LLM.
  3. Extract 1-2 "pseudo skills" from every (pseudo_question, pseudo_solution)
     pair via a strategy-extraction prompt.
  4. Append the new pseudo skills to ``skill_library.jsonl`` (atomic write).

The whole pipeline only runs when the existing skill library has fewer than
``min_existing_skills`` entries (aligned with ``config.self_icl.target_pseudo_skill_min``),
unless ``--force`` is passed.

Derived from:
  * ``/home/qzheng19/Self-ICL/prompt.py``      (pseudo-input / prediction prompts)
  * ``/home/qzheng19/Self-ICL/experiment.py``  (3-step flow, lines 189-319)

Schema aligned with Task 2.1 (``skill_extract.py``) except ``source="pseudo"``
and ``skill_id`` prefix ``pseudo_``, so downstream tasks (2.2 / 3.2) can
filter/down-weight cold-start skills.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Prompt templates (math-reasoning flavored, adapted from Self-ICL/prompt.py)
# -----------------------------------------------------------------------------

PSEUDO_INPUT_TEMPLATE: str = (
    "You are helping a mathematics tutor prepare additional practice problems.\n"
    "Task description: Compose novel high-school / competition level math "
    "problems that share the style, topic, and difficulty of the example below.\n\n"
    "Example problem:\n"
    "Q: {seed_question}\n\n"
    "Please come up with {k} new, diverse math problems for the task. "
    "Do NOT copy the example. Each problem should be self-contained and have a "
    "unique numeric answer. Respond in the following exact format, and nothing else:\n\n"
    "New problem 1:\nQ: <problem text>\n\n"
    "New problem 2:\nQ: <problem text>\n\n"
    "New problem 3:\nQ: <problem text>\n"
)

PSEUDO_PREDICT_TEMPLATE: str = (
    "Task description: Solve the following math problem. "
    "Let's think step by step, show each step of reasoning, "
    "and conclude with the final answer on its own line "
    "formatted as 'Final answer: \\boxed{{...}}'.\n\n"
    "Q: {pseudo_question}\nA: Let's think step by step."
)

SKILL_EXTRACT_TEMPLATE: str = (
    "You are a meta-reasoner distilling a general problem-solving skill from "
    "a single worked example. Read the problem and solution, then extract "
    "{max_skills_per_pair} reusable skill(s).\n\n"
    "Problem:\n{pseudo_question}\n\n"
    "Worked solution:\n{pseudo_solution}\n\n"
    "Respond ONLY as a JSON array of objects. Each object MUST contain:\n"
    "  - \"name\": short title of the skill (<=8 words)\n"
    "  - \"principle\": the guiding principle, 1-2 sentences\n"
    "  - \"when_to_apply\": the trigger condition (problem features)\n\n"
    "Example format:\n"
    "[{{\"name\": \"...\", \"principle\": \"...\", \"when_to_apply\": \"...\"}}]\n"
)

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class Skill:
    """Output schema, aligned with Task 2.1 ``skill_extract.py``."""

    skill_id: str
    name: str
    principle: str
    when_to_apply: str
    source: str = "pseudo"  # <-- distinguishes cold-start skills
    source_question_ids: List[str] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    metric_score: float = 0.5  # (1 + success) / (2 + usage) = 0.5 at init
    created_at: str = ""

    def to_json(self) -> Dict[str, Any]:
        d = {
            "skill_id": self.skill_id,
            "name": self.name,
            "principle": self.principle,
            "when_to_apply": self.when_to_apply,
            "source": self.source,
            "source_question_ids": list(self.source_question_ids),
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "metric_score": self.metric_score,
            "created_at": self.created_at,
        }
        return d


@dataclass
class SelfICLConfig:
    pseudo_input_count: int = 3
    target_pseudo_skill_min: int = 10
    target_pseudo_skill_max: int = 20


# -----------------------------------------------------------------------------
# Config / IO helpers
# -----------------------------------------------------------------------------


def load_config(config_path: Path) -> SelfICLConfig:
    """Minimal YAML loader: only the self_icl section matters here."""
    try:
        import yaml  # PyYAML
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("PyYAML required to read config.yaml") from e

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    section = raw.get("self_icl", {}) or {}
    return SelfICLConfig(
        pseudo_input_count=int(section.get("pseudo_input_count", 3)),
        target_pseudo_skill_min=int(section.get("target_pseudo_skill_min", 10)),
        target_pseudo_skill_max=int(section.get("target_pseudo_skill_max", 20)),
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # skip corrupt lines rather than crash the bootstrap
                continue
    return rows


def atomic_append_jsonl(path: Path, existing: List[Dict[str, Any]],
                        to_append: List[Dict[str, Any]]) -> None:
    """Write ``existing + to_append`` via tmp-file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    ) as tmp:
        tmp_path = Path(tmp.name)
        for row in existing:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in to_append:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def next_skill_id(existing: Iterable[Dict[str, Any]], start: int = 1) -> int:
    """Return 1 + max numeric suffix across any existing skill_id."""
    max_idx = start - 1
    pat = re.compile(r"(\d+)$")
    for row in existing:
        sid = str(row.get("skill_id", ""))
        m = pat.search(sid)
        if m:
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except ValueError:
                pass
    return max_idx + 1


# -----------------------------------------------------------------------------
# LLM wrapper (lazy imports; supports --dry_run monkey-patch)
# -----------------------------------------------------------------------------


class LLMGenerator:
    """Tiny wrapper around transformers.AutoModelForCausalLM."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        # Delay heavy imports until we truly need the model.
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        self._lazy_init()
        import torch

        tok = self._tokenizer
        # Use chat template when available (Qwen3 etc.)
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            input_ids = tok(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen_ids = output[0, input_ids.shape[-1]:]
        return tok.decode(gen_ids, skip_special_tokens=True).strip()


# -----------------------------------------------------------------------------
# Dry-run mock LLM (deterministic, no transformers dependency)
# -----------------------------------------------------------------------------


class MockLLMGenerator:
    """Deterministic fake LLM used by ``--dry_run``.

    Dispatches by detecting a unique signature inside the prompt so that the
    same stub handles all three prompt stages.
    """

    def __init__(self, pseudo_per_question: int, max_skills_per_pair: int = 2):
        self.pseudo_per_question = pseudo_per_question
        self.max_skills_per_pair = max_skills_per_pair

    def generate(self, prompt: str, **_: Any) -> str:
        if "New problem 1" in prompt and "come up with" in prompt:
            return self._fake_pseudo_inputs(prompt)
        if prompt.strip().endswith("Let's think step by step."):
            return self._fake_solution(prompt)
        if "JSON array" in prompt:
            return self._fake_skill_json(prompt)
        # Fallback
        return "OK"

    def _fake_pseudo_inputs(self, prompt: str) -> str:
        # Build K problems with incrementing numeric content for traceability.
        seed = _extract_seed(prompt)
        lines = []
        for i in range(1, self.pseudo_per_question + 1):
            lines.append(f"New problem {i}:")
            lines.append(
                f"Q: [dry-run] variant {i} of '{seed[:40]}' — compute f({i + 1})."
            )
            lines.append("")
        return "\n".join(lines).strip()

    def _fake_solution(self, prompt: str) -> str:
        return (
            "Step 1: Identify the target variable.\n"
            "Step 2: Apply the formula.\n"
            "Step 3: Compute the result.\n"
            "Final answer: \\boxed{42}"
        )

    def _fake_skill_json(self, prompt: str) -> str:
        payload = [
            {
                "name": f"dry-run skill {i + 1}",
                "principle": (
                    "Break the problem into sub-cases and solve each "
                    "independently before recombining."
                ),
                "when_to_apply": (
                    "Problem contains piecewise or absolute-value terms."
                ),
            }
            for i in range(self.max_skills_per_pair)
        ]
        return json.dumps(payload, ensure_ascii=False)


def _extract_seed(prompt: str) -> str:
    m = re.search(r"Q:\s*(.+?)\n", prompt)
    return m.group(1).strip() if m else "<seed>"


# -----------------------------------------------------------------------------
# Parsers
# -----------------------------------------------------------------------------


def parse_pseudo_inputs(text: str, k: int) -> List[str]:
    """Split generated text into up to ``k`` pseudo questions.

    Tolerates missing numbering by also splitting on "Q:" markers.
    """
    if not text:
        return []
    # Prefer "New problem N:" blocks.
    blocks = re.split(r"New problem\s*\d+\s*:", text)
    candidates: List[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        # Strip leading "Q:" token if present.
        m = re.match(r"Q\s*:\s*(.*)", b, flags=re.DOTALL)
        body = (m.group(1) if m else b).strip()
        # Cut off at "New problem" fragments or trailing labels.
        body = re.split(r"\n\s*(?:Answer\s*:|A\s*:)", body, maxsplit=1)[0].strip()
        if body:
            candidates.append(body)
    # Fallback: split by "Q:" markers directly.
    if len(candidates) < k:
        extras = re.findall(r"Q\s*:\s*(.+?)(?=(?:\n\s*Q\s*:|\Z))", text,
                            flags=re.DOTALL)
        for e in extras:
            body = e.strip()
            if body and body not in candidates:
                candidates.append(body)
    return candidates[:k]


def parse_skill_json(text: str, max_skills: int) -> List[Dict[str, str]]:
    """Extract JSON list of skill dicts from model output; tolerate prefix/suffix."""
    if not text:
        return []
    # Greedy: pull the first "[...]" block.
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data[:max_skills]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        principle = str(item.get("principle", "")).strip()
        when = str(item.get("when_to_apply", "")).strip()
        if name and principle:
            out.append(
                {"name": name, "principle": principle, "when_to_apply": when}
            )
    return out


# -----------------------------------------------------------------------------
# Core orchestration
# -----------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bootstrap_pseudo_skills(
    seeds: List[Dict[str, Any]],
    generator: Any,
    pseudo_per_question: int,
    starting_idx: int,
    max_skills_per_pair: int = 2,
    max_skills_total: Optional[int] = None,
    verbose: bool = True,
) -> List[Skill]:
    """Run the 3-stage Self-ICL flow and return a list of ``Skill`` objects."""
    new_skills: List[Skill] = []
    running_idx = starting_idx

    for seed in seeds:
        sid = seed.get("question_id", f"seed_{seeds.index(seed)}")
        seed_q = seed.get("question", "").strip()
        if not seed_q:
            continue

        # ---- 1) Pseudo input generation ----
        prompt1 = PSEUDO_INPUT_TEMPLATE.format(
            seed_question=seed_q, k=pseudo_per_question
        )
        raw1 = generator.generate(prompt1, max_new_tokens=1024, temperature=0.8)
        pseudo_qs = parse_pseudo_inputs(raw1, pseudo_per_question)
        if verbose:
            print(f"[seed={sid}] pseudo questions generated: {len(pseudo_qs)}")
        if not pseudo_qs:
            continue

        for p_idx, pseudo_q in enumerate(pseudo_qs):
            # ---- 2) Zero-shot solve ----
            prompt2 = PSEUDO_PREDICT_TEMPLATE.format(pseudo_question=pseudo_q)
            pseudo_solution = generator.generate(
                prompt2, max_new_tokens=1024, temperature=0.7
            ).strip()
            if not pseudo_solution:
                continue

            # ---- 3) Skill extraction ----
            prompt3 = SKILL_EXTRACT_TEMPLATE.format(
                pseudo_question=pseudo_q,
                pseudo_solution=pseudo_solution,
                max_skills_per_pair=max_skills_per_pair,
            )
            raw3 = generator.generate(prompt3, max_new_tokens=512, temperature=0.3)
            skill_dicts = parse_skill_json(raw3, max_skills_per_pair)
            if verbose:
                print(
                    f"  [seed={sid} p{p_idx}] extracted {len(skill_dicts)} skill(s)"
                )

            for d in skill_dicts:
                skill = Skill(
                    skill_id=f"pseudo_{running_idx:03d}",
                    name=d["name"],
                    principle=d["principle"],
                    when_to_apply=d["when_to_apply"],
                    source="pseudo",
                    source_question_ids=[f"pseudo_{p_idx}_{sid}"],
                    usage_count=0,
                    success_count=0,
                    metric_score=0.5,
                    created_at=_now_iso(),
                )
                new_skills.append(skill)
                running_idx += 1
                if max_skills_total is not None and len(new_skills) >= max_skills_total:
                    return new_skills
    return new_skills


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Self-ICL cold-start bootstrap: generate pseudo skills when "
            "skill_library.jsonl is empty/too small (Task 3.1)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seed_questions",
        type=Path,
        required=True,
        help='JSONL file with {"question_id","question","ground_truth"} per line',
    )
    p.add_argument(
        "--skill_library",
        type=Path,
        default=Path(
            "/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/"
            "skill_library.jsonl"
        ),
        help="Output/append target (JSONL).",
    )
    p.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model id used for pseudo generation.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/config.yaml"
        ),
        help="Config YAML (reads self_icl.*)",
    )
    p.add_argument(
        "--pseudo_per_question",
        type=int,
        default=None,
        help="Overrides config.self_icl.pseudo_input_count",
    )
    p.add_argument(
        "--min_existing_skills",
        type=int,
        default=None,
        help=(
            "Skip bootstrap if library already has >= this many skills. "
            "Defaults to config.self_icl.target_pseudo_skill_min."
        ),
    )
    p.add_argument(
        "--max_new_skills",
        type=int,
        default=None,
        help=(
            "Stop early once this many new skills have been generated. "
            "Defaults to config.self_icl.target_pseudo_skill_max."
        ),
    )
    p.add_argument(
        "--max_skills_per_pair",
        type=int,
        default=2,
        help="Upper bound of skills extracted per pseudo problem.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Run even if the skill library already has enough skills.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Use a deterministic mock LLM (no transformers / GPU calls).",
    )
    p.add_argument(
        "--seed_limit",
        type=int,
        default=None,
        help="Only use the first N seed questions (useful for smoke tests).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress per seed.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # --- 1) Load config and resolve defaults ---
    cfg = load_config(args.config) if args.config.exists() else SelfICLConfig()
    k = args.pseudo_per_question or cfg.pseudo_input_count
    min_existing = (
        args.min_existing_skills
        if args.min_existing_skills is not None
        else cfg.target_pseudo_skill_min
    )
    max_new = (
        args.max_new_skills
        if args.max_new_skills is not None
        else cfg.target_pseudo_skill_max
    )

    # --- 2) Early-return guard on skill library size ---
    existing = read_jsonl(args.skill_library)
    n_existing = len(existing)
    if n_existing >= min_existing and not args.force:
        print(
            f"skill_library already has {n_existing} >= {min_existing} "
            f"skills, skipping self-ICL bootstrap"
        )
        return 0

    # --- 3) Load seeds ---
    seeds = read_jsonl(args.seed_questions)
    if args.seed_limit:
        seeds = seeds[: args.seed_limit]
    if not seeds:
        print(f"ERROR: no seed questions loaded from {args.seed_questions}",
              file=sys.stderr)
        return 2

    # --- 4) Pick generator ---
    if args.dry_run:
        generator: Any = MockLLMGenerator(
            pseudo_per_question=k,
            max_skills_per_pair=args.max_skills_per_pair,
        )
        print("[dry-run] using MockLLMGenerator (no LLM calls)")
    else:
        generator = LLMGenerator(args.llm_model)
        print(f"[live] using LLMGenerator(model_name={args.llm_model})")

    # --- 5) Generate ---
    start_idx = next_skill_id(existing, start=1)
    t0 = time.time()
    new_skills = bootstrap_pseudo_skills(
        seeds=seeds,
        generator=generator,
        pseudo_per_question=k,
        starting_idx=start_idx,
        max_skills_per_pair=args.max_skills_per_pair,
        max_skills_total=max_new,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0

    # --- 6) Persist ---
    if new_skills:
        atomic_append_jsonl(
            args.skill_library,
            existing=existing,
            to_append=[s.to_json() for s in new_skills],
        )

    print(
        f"[done] generated {len(new_skills)} pseudo skills in {elapsed:.1f}s "
        f"(library: {n_existing} -> {n_existing + len(new_skills)}); "
        f"wrote to {args.skill_library}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
