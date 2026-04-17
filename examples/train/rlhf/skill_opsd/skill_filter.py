"""Skill filter CLI for Skill-Informed OPSD (Task 2.2).

Scores each skill in `skill_library.jsonl` by the teacher-student Jensen-Shannon
divergence on a held-out validation batch. Skills whose mean KL falls below a
threshold (`skill_filter.kl_threshold` in config.yaml, default 0.01) are dropped
because the extra context fails to change the student's next-token distribution
and therefore carries no teaching signal.

Teacher forward:
    prompt = [Relevant Reasoning Skill] <skill_text> + [Problem] <question> + [Solution]
Student forward:
    prompt = [Problem] <question> + [Solution]

Both run under the SAME PEFT checkpoint; teacher uses ``student.disable_adapter()``
so we only ever hold one copy of the model in memory (mirrors the
self-distillation path in ``swift/rlhf_trainers/gkd_trainer.py``).

Logits alignment strategy (see ``compute_skill_kl``):
    * Teacher input = teacher_prefix + student_prompt (i.e. student_prompt is a
      suffix of teacher_input).
    * We slice the teacher logits tail with a length equal to
      ``len(student_input_ids) - 1`` and take the student logits tail with the
      same length. Both now correspond to the token positions that PREDICT the
      question-plus-generation-prefix region, giving token-wise aligned logits.
    * If due to special-token differences the two tails differ by a few tokens,
      we fall back to ``min(len_s, len_t)`` and truncate the front of the longer.
    * Only the last ``eval_tail_tokens`` positions (default: all of the tail)
      participate in the JSD; this mirrors OPSD's practice of evaluating the
      divergence on the response region, but because we have no gold
      completions here we use the end of the prompt instead (it is the part
      closest to a true generation step).

Refs:
    * ``generalized_jsd_loss`` from ``swift/rlhf_trainers/gkd_trainer.py``
      (Task 1.1 made it module-level).
    * EvolveR ``experience_manager._update_metric_scores`` — Laplace-smoothed
      ``(success+1)/(usage+2)``; here we overwrite ``metric_score`` with the
      KL value since KL replaces success-rate as the teaching-value proxy.
    * Skill schema: ``skill_extract.Skill`` dataclass (skill_id/name/principle/
      when_to_apply/source/source_question_ids/usage_count/success_count/
      metric_score/created_at).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import tempfile
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("skill_filter")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file tolerantly (empty dict if missing)."""
    if not path or not os.path.exists(path):
        return {}
    import yaml  # local import to keep `--help` lightweight

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
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
                LOGGER.warning(
                    "skipping malformed JSONL line %d in %s: %s", line_no, path, exc
                )
    return rows


def atomic_write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows to ``path`` atomically via tmp + os.replace."""
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
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_teacher_prompt(skill: Dict[str, Any], question: str) -> str:
    """Compose the teacher-side prompt that exposes the skill as in-context hint."""
    name = str(skill.get("name", "")).strip() or "(unnamed)"
    principle = str(skill.get("principle", "")).strip() or "(no principle)"
    when_to_apply = str(skill.get("when_to_apply", "")).strip() or "(no trigger)"
    return (
        "[Relevant Reasoning Skill]\n"
        f"Name: {name}\n"
        f"Principle: {principle}\n"
        f"When to apply: {when_to_apply}\n\n"
        "[Problem]\n"
        f"{question}\n\n"
        "[Solution]\n"
    )


def build_student_prompt(question: str) -> str:
    """Compose the student-side prompt (no skill context)."""
    return f"[Problem]\n{question}\n\n[Solution]\n"


def wrap_with_chat_template(tokenizer: Any, prompt: str) -> str:
    """Wrap ``prompt`` via the tokenizer's chat template when available."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as exc:  # noqa: BLE001 - tokenizer may not support templates
        LOGGER.debug("chat template failed, using raw prompt: %s", exc)
        return prompt


# ---------------------------------------------------------------------------
# Core KL computation
# ---------------------------------------------------------------------------


def _min_length_align(
    student_logits: Any, teacher_logits: Any
) -> Tuple[Any, Any]:
    """Clip front of longer logits tensor so that both have equal seq length."""
    # Shapes: [B, T, V]. We truncate along T from the front (keep tail == end of
    # the prompt; that is the region closest to a real generation step).
    len_s = student_logits.size(1)
    len_t = teacher_logits.size(1)
    common = min(len_s, len_t)
    if common <= 0:
        return student_logits[:, :0], teacher_logits[:, :0]
    return student_logits[:, -common:], teacher_logits[:, -common:]


def compute_skill_kl(
    *,
    skill: Dict[str, Any],
    questions: Sequence[str],
    tokenizer: Any,
    model: Any,
    jsd_fn: Any,
    torch_mod: Any,
    teacher_ctx_fn: Any,
    device: Any,
    max_length: int = 4096,
    eval_tail_tokens: Optional[int] = None,
    dtype: Any = None,
) -> float:
    """Return the average JSD between teacher and student on ``questions``.

    Parameters
    ----------
    skill : dict
        Skill record (must contain ``name``/``principle``/``when_to_apply``).
    questions : Sequence[str]
        Validation questions (plain text).
    tokenizer : AutoTokenizer
        HF tokenizer (already loaded).
    model : PeftModel
        Student model; the teacher is obtained via ``teacher_ctx_fn()``.
    jsd_fn : callable
        ``generalized_jsd_loss`` reference (Task 1.1).
    torch_mod : module
        Imported ``torch`` (passed so that this function can be dry-run without
        importing the real package at module load time).
    teacher_ctx_fn : callable
        Context manager factory that yields the teacher forward context. For a
        PEFT student we use ``lambda: model.disable_adapter()``.
    device : torch.device
        Device for logits.
    max_length : int
        Cap on teacher prompt length (tokens). Student prompt inherits the same
        cap but is almost always shorter.
    eval_tail_tokens : Optional[int]
        If set, only the last N aligned positions contribute to the JSD. None
        means "use the full aligned tail".
    dtype : torch.dtype
        Compute dtype for forward (usually bf16).

    Returns
    -------
    float
        Mean JSD across the question batch.
    """
    if not questions:
        return 0.0
    kls: List[float] = []
    for q in questions:
        teacher_text = wrap_with_chat_template(tokenizer, build_teacher_prompt(skill, q))
        student_text = wrap_with_chat_template(tokenizer, build_student_prompt(q))

        teacher_ids = tokenizer(
            teacher_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).input_ids.to(device)
        student_ids = tokenizer(
            student_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).input_ids.to(device)

        with torch_mod.no_grad():
            # Student forward (LoRA enabled).
            s_out = model(input_ids=student_ids, use_cache=False)
            s_logits = s_out.logits
            del s_out
            # Teacher forward (LoRA disabled) — self-distillation pattern.
            with teacher_ctx_fn():
                t_out = model(input_ids=teacher_ids, use_cache=False)
                t_logits = t_out.logits
                del t_out

        # Strategy: keep the shared tail. Both prompts end with the same
        # "[Solution]" + generation-prompt suffix and share `question` tokens,
        # so the tail of teacher logits corresponds to the same positions as
        # the tail of student logits.
        s_logits_aligned, t_logits_aligned = _min_length_align(s_logits, t_logits)
        if eval_tail_tokens is not None and eval_tail_tokens > 0:
            s_logits_aligned = s_logits_aligned[:, -eval_tail_tokens:]
            t_logits_aligned = t_logits_aligned[:, -eval_tail_tokens:]

        if s_logits_aligned.size(1) == 0:
            LOGGER.warning(
                "empty aligned logits tail for skill_id=%s question=%r; skipping",
                skill.get("skill_id"),
                q[:40],
            )
            continue

        # Cast to compute dtype; JSD kernel handles log_softmax internally.
        if dtype is not None:
            s_logits_aligned = s_logits_aligned.to(dtype)
            t_logits_aligned = t_logits_aligned.to(dtype)

        jsd = jsd_fn(
            s_logits_aligned,
            t_logits_aligned,
            labels=None,
            beta=0.5,
            temperature=1.0,
            token_clip=None,
        )
        kls.append(float(jsd.detach().float().item()))
    if not kls:
        return 0.0
    return sum(kls) / len(kls)


# ---------------------------------------------------------------------------
# Dry-run mock KL (no real model; just validates the whole pipeline)
# ---------------------------------------------------------------------------


def mock_skill_kl(skill: Dict[str, Any], questions: Sequence[str]) -> float:
    """Deterministic mock KL for ``--dry_run`` mode.

    Ordered so that ``avg_kl`` grows with ``skill_id`` suffix (numeric part);
    this lets the sorting/filtering path be validated without real logits.
    """
    num = 0
    sid = str(skill.get("skill_id", ""))
    for ch in sid:
        if ch.isdigit():
            num = num * 10 + int(ch)
    base = (num + 1) * 0.01
    # Small per-question jitter so the "batch mean" isn't trivially degenerate.
    rng = random.Random(hash(sid) & 0xFFFF)
    jitter = sum(rng.random() * 0.002 for _ in questions) / max(len(questions), 1)
    return float(base + jitter)


# ---------------------------------------------------------------------------
# Filter pipeline
# ---------------------------------------------------------------------------


def filter_skills(
    *,
    skills: List[Dict[str, Any]],
    val_questions: Sequence[Dict[str, Any]],
    kl_threshold: float,
    keep_ratio_min: float,
    keep_ratio_max: float,
    max_val: int,
    score_fn: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Score, filter and sort ``skills``. Returns ``(kept, dropped)``.

    ``score_fn`` is either ``compute_skill_kl``-bound (real) or ``mock_skill_kl``
    (dry-run). It must accept ``(skill, questions)``.
    """
    q_texts: List[str] = [str(q.get("question", "")).strip() for q in val_questions]
    q_texts = [q for q in q_texts if q]

    if len(q_texts) == 0:
        LOGGER.warning("no val_questions available; all skills will receive metric_score=0")

    scored: List[Dict[str, Any]] = []
    rng = random.Random(0xC0FFEE)
    for skill in skills:
        pool = q_texts
        if max_val and len(pool) > max_val:
            pool = rng.sample(pool, max_val)
        avg_kl = float(score_fn(skill, pool))
        new_skill = dict(skill)
        new_skill["metric_score"] = avg_kl
        scored.append(new_skill)
        LOGGER.info(
            "scored skill_id=%s name=%r avg_kl=%.6f (n_val=%d)",
            skill.get("skill_id"),
            skill.get("name"),
            avg_kl,
            len(pool),
        )

    kept = [s for s in scored if s["metric_score"] >= kl_threshold]
    dropped = [s for s in scored if s["metric_score"] < kl_threshold]

    if scored:
        ratio = len(kept) / len(scored)
        if ratio < keep_ratio_min or ratio > keep_ratio_max:
            LOGGER.warning(
                "keep_ratio=%.3f out of range [%.2f, %.2f]; threshold=%.4g may be off",
                ratio,
                keep_ratio_min,
                keep_ratio_max,
                kl_threshold,
            )

    kept.sort(key=lambda s: s["metric_score"], reverse=True)
    return kept, dropped


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skill_filter",
        description=(
            "Score skills by teacher-student JSD and drop those below a "
            "KL threshold."
        ),
    )
    p.add_argument(
        "--skill_library",
        required=True,
        help="Input JSONL skill library (also the default in-place output).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default: overwrite --skill_library after "
        "backing up to .bak.",
    )
    p.add_argument(
        "--val_questions",
        required=True,
        help="JSONL of validation questions; each line has a `question` field.",
    )
    p.add_argument(
        "--base_model",
        default="Qwen/Qwen3-4B",
        help="Base HF model name/path (default Qwen/Qwen3-4B).",
    )
    p.add_argument(
        "--adapter",
        default=None,
        help="Path to a PEFT LoRA adapter (serves as student; teacher is base "
        "via disable_adapter()). Required unless --dry_run.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="KL threshold; skills below this are dropped (default: config "
        "skill_filter.kl_threshold = 0.01).",
    )
    p.add_argument(
        "--config",
        default="",
        help="Path to config.yaml (optional; provides threshold/ratio defaults).",
    )
    p.add_argument(
        "--max_val",
        type=int,
        default=30,
        help="Max val questions evaluated per skill (default 30).",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Unused placeholder kept for CLI stability (JSD is computed on "
        "prompt logits, no generation).",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Max tokens per prompt (default 4096).",
    )
    p.add_argument(
        "--eval_tail_tokens",
        type=int,
        default=128,
        help="Number of aligned tail positions used for the JSD (default 128; "
        "0 means all aligned positions).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Mock model forward with deterministic KL; no torch/peft load.",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        help="Python logging level (default INFO).",
    )
    return p


def resolve_config(args: argparse.Namespace) -> Tuple[float, float, float]:
    """Merge CLI flags with config.yaml (CLI wins).

    Returns ``(kl_threshold, keep_ratio_min, keep_ratio_max)``.
    """
    cfg = load_yaml_config(args.config)
    sf = cfg.get("skill_filter", {}) if isinstance(cfg, dict) else {}
    kl_threshold = (
        args.threshold if args.threshold is not None else float(sf.get("kl_threshold", 0.01))
    )
    keep_ratio_min = float(sf.get("filter_keep_ratio_min", 0.5))
    keep_ratio_max = float(sf.get("filter_keep_ratio_max", 0.9))
    return kl_threshold, keep_ratio_min, keep_ratio_max


def _load_real_backend(
    *, base_model: str, adapter: Optional[str], dtype_str: str = "bfloat16"
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Lazy-import torch/transformers/peft and load the stack.

    Returns ``(torch, tokenizer, model, jsd_fn, teacher_ctx_fn, device, dtype)``.
    """
    import torch  # noqa: WPS433 - lazy by design
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

    from swift.rlhf_trainers.gkd_trainer import generalized_jsd_loss  # noqa: WPS433

    dtype = getattr(torch, dtype_str, torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("loading tokenizer from %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    LOGGER.info("loading base model from %s (dtype=%s, device=%s)", base_model, dtype, device)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base.to(device)
    base.eval()

    if adapter:
        from peft import PeftModel  # noqa: WPS433

        LOGGER.info("attaching PEFT adapter from %s", adapter)
        model = PeftModel.from_pretrained(base, adapter)
        model.to(device)
        model.eval()

        def teacher_ctx_fn():
            return model.disable_adapter()
    else:
        # No adapter provided: degenerate case — teacher == student.
        LOGGER.warning(
            "no --adapter provided; teacher == student, all KLs will be ~0"
        )
        model = base

        def teacher_ctx_fn():
            return nullcontext()

    return torch, tokenizer, model, generalized_jsd_loss, teacher_ctx_fn, device, dtype


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    kl_threshold, keep_min, keep_max = resolve_config(args)

    skills = load_jsonl(args.skill_library)
    val_questions = load_jsonl(args.val_questions)
    LOGGER.info(
        "loaded %d skills from %s; %d val questions from %s",
        len(skills),
        args.skill_library,
        len(val_questions),
        args.val_questions,
    )
    if not skills:
        LOGGER.warning("no skills to filter; exiting without touching outputs")
        return 0

    if args.dry_run:
        LOGGER.info("[dry_run] using mock_skill_kl (no torch/peft load)")
        score_fn = lambda skill, pool: mock_skill_kl(skill, pool)  # noqa: E731
    else:
        if not args.adapter:
            LOGGER.warning(
                "no --adapter in live mode; proceeding with teacher == student "
                "(debug only; expect ~0 KLs)"
            )
        (
            torch_mod,
            tokenizer,
            model,
            jsd_fn,
            teacher_ctx_fn,
            device,
            dtype,
        ) = _load_real_backend(base_model=args.base_model, adapter=args.adapter)

        def score_fn(skill: Dict[str, Any], pool: Sequence[str]) -> float:
            eval_tail = args.eval_tail_tokens if args.eval_tail_tokens > 0 else None
            return compute_skill_kl(
                skill=skill,
                questions=pool,
                tokenizer=tokenizer,
                model=model,
                jsd_fn=jsd_fn,
                torch_mod=torch_mod,
                teacher_ctx_fn=teacher_ctx_fn,
                device=device,
                max_length=args.max_length,
                eval_tail_tokens=eval_tail,
                dtype=dtype,
            )

    kept, dropped = filter_skills(
        skills=skills,
        val_questions=val_questions,
        kl_threshold=kl_threshold,
        keep_ratio_min=keep_min,
        keep_ratio_max=keep_max,
        max_val=args.max_val,
        score_fn=score_fn,
    )
    LOGGER.info(
        "filter: kept=%d dropped=%d (threshold=%.4g; keep_ratio=%.3f)",
        len(kept),
        len(dropped),
        kl_threshold,
        (len(kept) / max(len(skills), 1)),
    )
    for s in kept:
        LOGGER.info(
            "KEEP skill_id=%s name=%r metric_score=%.6f",
            s.get("skill_id"),
            s.get("name"),
            s["metric_score"],
        )
    for s in dropped:
        LOGGER.info(
            "DROP skill_id=%s name=%r metric_score=%.6f",
            s.get("skill_id"),
            s.get("name"),
            s["metric_score"],
        )

    output_path = args.output
    if not output_path:
        output_path = args.skill_library
        backup_path = args.skill_library + ".bak"
        if os.path.exists(args.skill_library):
            shutil.copyfile(args.skill_library, backup_path)
            LOGGER.info("backed up %s -> %s", args.skill_library, backup_path)

    atomic_write_jsonl(output_path, kept)
    LOGGER.info("wrote %d kept skills to %s", len(kept), output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
