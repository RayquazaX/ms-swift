"""Teacher-prompt context builder for Skill-Informed OPSD (Task 3.3).

Reads the per-question skill selections from Task 3.2, fetches each skill's
full definition from ``skill_library.jsonl`` (Task 2.1 + 2.2 + 3.1 output),
and emits a ``train_with_context.jsonl`` whose rows match the schema the
ms-swift GKD plugin (Task 3.4) expects:

.. code-block:: json

    {
      "messages": [
        {"role": "user", "content": "<question>"},
        {"role": "assistant", "content": "<reference solution>"}
      ],
      "teacher_prompt": "<layered markdown + OPSD transition + question>",
      "question_id": "<q_id>"
    }

The teacher prompt is formatted hierarchically by *source* (Task 2.1 schema):

* ``source=success``   -> ``## [Reasoning Framework]``
* ``source=failure``   -> ``## [Common Pitfalls]``
* ``source=pseudo``    -> ``## [Exploratory Heuristics]``

Design notes:
    * OPSD ``TRANSITION_PROMPT`` is imported *lazily* from
      ``opsd_plugin.py`` (file:line reference in module docstring below)
      so the wording stays the single source of truth.
    * ``--optimize_context`` enables a *simplified* PromptAgent loop:
      3 rounds of greedy candidate generation + validation. To keep the
      cost tractable we optimize the **three section-header templates**
      (Framework / Pitfalls / Heuristics) globally rather than per
      question. See ``PromptAgentGreedy`` below.
    * LLM / heavy deps are imported lazily inside the optimization path
      so ``--help`` and the default dry-run never need torch.

References:
    - OPSD transition prompt:
      /home/qzheng19/ms-swift/examples/train/rlhf/opsd/opsd_plugin.py L17-18
    - SkillRL markdown layout:
      /home/qzheng19/SkillRL/agent_system/memory/skills_only_memory.py L384-439
    - PromptAgent MCTS (simplified to greedy):
      /home/qzheng19/PromptAgent/src/prompt_optim_agent/search_algo/mcts.py
      /home/qzheng19/PromptAgent/src/prompt_optim_agent/world_model/gradient_descent.py
      /home/qzheng19/PromptAgent/src/prompt_optim_agent/world_model/world_model.py L138-155
    - IMPLEMENTATION_PLAN.md Task 3.3
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("context_builder")


# ---------------------------------------------------------------------------
# Constants / default templates
# ---------------------------------------------------------------------------
DEFAULT_FRAMEWORK_HEADER = "## [Reasoning Framework]"
DEFAULT_FRAMEWORK_INTRO = (
    "Apply the following high-success reasoning strategies. "
    "For each, check whether its `when_to_apply` condition matches "
    "before using it."
)
DEFAULT_PITFALLS_HEADER = "## [Common Pitfalls]"
DEFAULT_PITFALLS_INTRO = (
    "Avoid these failure modes observed in past incorrect attempts."
)
DEFAULT_HEURISTICS_HEADER = "## [Exploratory Heuristics]"
DEFAULT_HEURISTICS_INTRO = (
    "Optional heuristics synthesized from analogical problems; treat "
    "them as exploratory suggestions rather than hard rules."
)

_VALID_SOURCES = {"success", "failure", "pseudo"}
_SOURCE_TO_BUCKET = {
    "success": "framework",
    "failure": "pitfalls",
    "pseudo": "heuristics",
}


@dataclass
class SectionTemplates:
    """Per-section headers + intro lines. The PromptAgent greedy optimizer
    mutates this dataclass (not per-question prompts)."""

    framework_header: str = DEFAULT_FRAMEWORK_HEADER
    framework_intro: str = DEFAULT_FRAMEWORK_INTRO
    pitfalls_header: str = DEFAULT_PITFALLS_HEADER
    pitfalls_intro: str = DEFAULT_PITFALLS_INTRO
    heuristics_header: str = DEFAULT_HEURISTICS_HEADER
    heuristics_intro: str = DEFAULT_HEURISTICS_INTRO

    def clone(self) -> "SectionTemplates":
        return SectionTemplates(**self.__dict__)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOGGER.warning("skipping malformed JSONL %s:%d (%s)", path, ln, exc)
    return rows


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _load_yaml(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        # pylint: disable=import-outside-toplevel
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("pyyaml unavailable (%s); ignoring --config", exc)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# OPSD transition prompt (imported lazily, never hand-copied)
# ---------------------------------------------------------------------------
def _load_opsd_transition_prompt() -> str:
    """Import ``TRANSITION_PROMPT`` from the OPSD plugin.

    We import at call time (not at module import) so this module stays
    importable even if the OPSD example dir is relocated. Falls back to a
    hard-coded copy of the L17-18 wording only if the import completely
    fails — with a ``logging.warning`` so the mismatch is visible.
    """
    opsd_plugin = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "opsd", "opsd_plugin.py",
    )
    opsd_plugin = os.path.normpath(opsd_plugin)
    try:
        # pylint: disable=import-outside-toplevel
        import importlib.util
        spec = importlib.util.spec_from_file_location("_opsd_plugin_src", opsd_plugin)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build spec for {opsd_plugin}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return str(mod.TRANSITION_PROMPT)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning(
            "Could not import TRANSITION_PROMPT from %s (%s); "
            "falling back to hard-coded copy. Keep opsd_plugin.py in sync!",
            opsd_plugin, exc,
        )
        # Verbatim copy of opsd_plugin.py L17-18 as a safety net.
        return (
            "After understanding the reference solution and the rationale "
            "behind each step, now articulate your own step-by-step "
            "reasoning that derives the final answer."
        )


# ---------------------------------------------------------------------------
# Skill -> markdown block
# ---------------------------------------------------------------------------
def _normalize_source(raw: Optional[str]) -> Tuple[str, bool]:
    """Return (bucket_key, was_warning)."""
    if raw is None:
        return "framework", True
    src = str(raw).strip().lower()
    if src in _VALID_SOURCES:
        return _SOURCE_TO_BUCKET[src], False
    return "framework", True


def _render_skill(skill: Dict[str, Any], index: int) -> str:
    name = str(skill.get("name", "")).strip() or f"Skill {index}"
    principle = str(skill.get("principle", "")).strip()
    when = str(skill.get("when_to_apply", "")).strip()

    lines = [f"{index}. **{name}**"]
    if principle:
        lines.append(f"   - Principle: {principle}")
    if when:
        lines.append(f"   - When to apply: {when}")
    return "\n".join(lines)


def _render_section(
    header: str,
    intro: str,
    skills: List[Dict[str, Any]],
) -> str:
    if not skills:
        return ""
    body = "\n".join(_render_skill(s, i + 1) for i, s in enumerate(skills))
    return f"{header}\n{intro}\n\n{body}"


def build_teacher_prompt(
    question: str,
    selected_skills: List[Dict[str, Any]],
    transition_prompt: str,
    templates: Optional[SectionTemplates] = None,
) -> str:
    """Assemble the final teacher_prompt markdown string.

    Parameters
    ----------
    question:
        The raw problem text.
    selected_skills:
        Full skill dicts (already looked up from the library) for this
        question, in the order chosen by Task 3.2.
    transition_prompt:
        The OPSD "Let's think step by step..." / equivalent wording.
    templates:
        Optional :class:`SectionTemplates`. Defaults used if ``None``.
    """
    tpl = templates or SectionTemplates()

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "framework": [],
        "pitfalls": [],
        "heuristics": [],
    }
    warned_any = False
    for s in selected_skills:
        bucket, warn = _normalize_source(s.get("source"))
        if warn:
            warned_any = True
            LOGGER.warning(
                "skill %r has missing/unknown source=%r; defaulting to framework",
                s.get("skill_id"), s.get("source"),
            )
        buckets[bucket].append(s)

    sections: List[str] = []
    fw = _render_section(tpl.framework_header, tpl.framework_intro,
                         buckets["framework"])
    if fw:
        sections.append(fw)
    pt = _render_section(tpl.pitfalls_header, tpl.pitfalls_intro,
                         buckets["pitfalls"])
    if pt:
        sections.append(pt)
    hr = _render_section(tpl.heuristics_header, tpl.heuristics_intro,
                         buckets["heuristics"])
    if hr:
        sections.append(hr)

    header = "# Reasoning Guidance"
    body_parts: List[str] = []
    if sections:
        body_parts.append(header)
        body_parts.append("\n\n".join(sections))
        body_parts.append("---")
    # OPSD transition wording + problem restatement.
    body_parts.append(transition_prompt)
    body_parts.append(f"Problem: {question}")
    body_parts.append("Solution:")

    del warned_any  # silence lint; warnings already emitted

    return "\n\n".join(body_parts)


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------
def _index_skill_library(skills: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for s in skills:
        sid = s.get("skill_id")
        if not sid:
            continue
        idx[str(sid)] = s
    return idx


def _index_raw(raw: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for r in raw:
        qid = r.get("question_id")
        if qid is None:
            continue
        idx[str(qid)] = r
    return idx


def assemble_rows(
    selected: List[Dict[str, Any]],
    skill_index: Dict[str, Dict[str, Any]],
    raw_index: Dict[str, Dict[str, Any]],
    transition_prompt: str,
    templates: SectionTemplates,
) -> List[Dict[str, Any]]:
    """Build training rows in the ms-swift messages+teacher_prompt schema."""
    out: List[Dict[str, Any]] = []
    for row in selected:
        qid = str(row.get("question_id"))
        question = row.get("question", "")
        ids: Sequence[str] = row.get("selected_skill_ids", []) or []
        raw = raw_index.get(qid, {})
        if not question:
            question = raw.get("question", "")
        solution = str(raw.get("solution", ""))

        skills: List[Dict[str, Any]] = []
        for sid in ids:
            s = skill_index.get(str(sid))
            if s is None:
                LOGGER.warning(
                    "question %s: selected skill_id=%r not in library; skipping",
                    qid, sid,
                )
                continue
            skills.append(s)

        teacher_prompt = build_teacher_prompt(
            question=question,
            selected_skills=skills,
            transition_prompt=transition_prompt,
            templates=templates,
        )
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ]
        out.append({
            "messages": messages,
            "teacher_prompt": teacher_prompt,
            "question_id": qid,
        })
    return out


# ---------------------------------------------------------------------------
# PromptAgent — simplified 3-round greedy optimizer over SectionTemplates
# ---------------------------------------------------------------------------
@dataclass
class _ValProblem:
    question_id: str
    question: str
    solution: str
    is_correct: bool

    @classmethod
    def from_raw(cls, row: Dict[str, Any]) -> "_ValProblem":
        return cls(
            question_id=str(row.get("question_id", "")),
            question=str(row.get("question", "")),
            solution=str(row.get("solution", "")),
            is_correct=bool(row.get("is_correct", True)),
        )


# Type for the LLM callable: (prompt, kind) -> str.
# ``kind`` is one of {"student", "candidate"}. We split them so callers
# can route to different backends / sampling configs if desired.
LLMFunc = Callable[[str, str], str]


def _mock_llm(prompt: str, kind: str) -> str:
    """Deterministic mock used by dry-run.

    For ``kind='student'`` we emulate a student that mostly parrots the
    reference solution (so accuracy depends on whether we even include a
    solution in the prompt). For ``kind='candidate'`` we return a small
    text perturbation of one section intro so the greedy loop actually
    has candidates to choose between.
    """
    seed = (hash(prompt) & 0xFFFF) ^ (hash(kind) & 0xFFFF)
    rng = random.Random(seed)
    if kind == "student":
        # Pretend the student is correct 60% of the time (deterministic).
        return "CORRECT" if rng.random() < 0.6 else "WRONG"
    if kind == "candidate":
        # Suggest a minor intro rewrite.
        candidates = [
            "Work through these high-reliability patterns before committing to an answer.",
            "These strategies were distilled from correct solutions; apply the ones that fit.",
            "Review these reusable tactics; use each only where its trigger condition holds.",
        ]
        return rng.choice(candidates)
    return ""


class PromptAgentGreedy:
    """3-round greedy optimizer over global :class:`SectionTemplates`.

    Flow (mirrors PromptAgent's expand/evaluate/back-propagate but
    collapsed to greedy hill-climbing without children / tree):

    1. Evaluate the *current* ``SectionTemplates`` on a small validation
       set by rendering each problem's teacher_prompt then asking the LLM
       to "solve" (``kind='student'``).
    2. Collect error examples.
    3. Ask the LLM (``kind='candidate'``) to rewrite each intro line
       based on the errors; produce ``expand_width`` candidates per
       section.
    4. Evaluate each candidate and keep the single best new template
       **only if** it strictly improves on the incumbent (mirrors
       ``MCTSNode.is_terminal_with_min_threshold`` — never regress).
    5. Repeat ``iters`` rounds (default 3).

    The student / candidate LLM is injected via ``llm_func`` to keep the
    class testable; callers in production should wrap their real
    inference backend.
    """

    def __init__(
        self,
        llm_func: LLMFunc,
        iters: int = 3,
        expand_width: int = 3,
        seed: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm_func = llm_func
        self.iters = int(iters)
        self.expand_width = int(expand_width)
        self.rng = random.Random(seed)
        self.log = logger or LOGGER

    # ---------- evaluation ----------
    def _evaluate(
        self,
        templates: SectionTemplates,
        val: List[_ValProblem],
        question_to_skills: Dict[str, List[Dict[str, Any]]],
        transition_prompt: str,
    ) -> Tuple[float, List[_ValProblem]]:
        """Return (accuracy, list_of_error_problems)."""
        if not val:
            return 0.0, []
        correct = 0
        errors: List[_ValProblem] = []
        for p in val:
            tp = build_teacher_prompt(
                question=p.question,
                selected_skills=question_to_skills.get(p.question_id, []),
                transition_prompt=transition_prompt,
                templates=templates,
            )
            resp = self.llm_func(tp, "student")
            # Very tolerant correctness check: "CORRECT" token OR
            # substring match of first line of ground-truth solution
            # (avoids real answer verification in dry-run).
            resp_norm = (resp or "").strip().upper()
            is_ok = resp_norm.startswith("CORRECT") or (
                p.solution and p.solution.splitlines()[0].strip() in (resp or "")
            )
            if is_ok:
                correct += 1
            else:
                errors.append(p)
        return correct / len(val), errors

    # ---------- candidate generation ----------
    def _propose_candidates(
        self,
        cur: SectionTemplates,
        errors: List[_ValProblem],
    ) -> List[SectionTemplates]:
        """Generate ``expand_width`` mutated ``SectionTemplates``.

        We mutate only the *intro* of each bucket (not the header, so
        downstream regex / parsing keeps working). Each candidate picks
        one random section to rewrite via the LLM.
        """
        if not errors:
            return []
        error_digest = "\n".join(
            f"- Q{e.question_id}: solver failed." for e in errors[:5]
        )
        bucket_attrs = [
            ("framework_intro", cur.framework_intro),
            ("pitfalls_intro", cur.pitfalls_intro),
            ("heuristics_intro", cur.heuristics_intro),
        ]
        cands: List[SectionTemplates] = []
        for i in range(self.expand_width):
            attr, cur_intro = self.rng.choice(bucket_attrs)
            ask = (
                f"Current intro line: {cur_intro}\n"
                f"Observed failures:\n{error_digest}\n"
                f"Propose a single concise replacement intro that would reduce these failures."
            )
            new_intro = self.llm_func(ask, "candidate").strip()
            if not new_intro or new_intro == cur_intro:
                continue
            new = cur.clone()
            setattr(new, attr, new_intro)
            cands.append(new)
            self.log.debug("[PA cand %d] %s <- %s", i, attr, new_intro)
        return cands

    # ---------- main loop ----------
    def optimize(
        self,
        initial: SectionTemplates,
        val: List[_ValProblem],
        question_to_skills: Dict[str, List[Dict[str, Any]]],
        transition_prompt: str,
    ) -> SectionTemplates:
        best = initial.clone()
        best_acc, errors = self._evaluate(
            best, val, question_to_skills, transition_prompt
        )
        self.log.info("[PromptAgent] round 0 (baseline) acc=%.3f", best_acc)

        for rnd in range(1, self.iters + 1):
            cands = self._propose_candidates(best, errors)
            if not cands:
                self.log.info(
                    "[PromptAgent] round %d: no candidates; keeping baseline",
                    rnd,
                )
                continue
            improved = False
            for ci, cand in enumerate(cands):
                acc, errs = self._evaluate(
                    cand, val, question_to_skills, transition_prompt
                )
                self.log.info(
                    "[PromptAgent] round %d cand %d acc=%.3f (best=%.3f)",
                    rnd, ci, acc, best_acc,
                )
                if acc > best_acc:
                    best_acc = acc
                    best = cand
                    errors = errs
                    improved = True
            if not improved:
                self.log.info(
                    "[PromptAgent] round %d: no candidate beat baseline (acc=%.3f)",
                    rnd, best_acc,
                )
        return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="context_builder.py",
        description=(
            "Build teacher-prompt training JSONL for Skill-Informed "
            "OPSD (Task 3.3). Reads Task 3.2 selections + Task 2.x "
            "skill library + raw Q/A, emits messages+teacher_prompt "
            "rows. Optional 3-round greedy PromptAgent optimization "
            "over the section-header templates."
        ),
    )
    p.add_argument("--selected", required=True,
                   help="JSONL from Task 3.2 (question_id, question, selected_skill_ids).")
    p.add_argument("--skill_library", required=True,
                   help="JSONL of skills (skill_id, name, principle, when_to_apply, source, ...).")
    p.add_argument("--questions_with_answers", required=True,
                   help="JSONL of raw problems (question_id, question, solution, is_correct).")
    p.add_argument("--output", required=True,
                   help="Where to write train_with_context.jsonl.")
    p.add_argument("--config", default=None,
                   help="Optional YAML config (e.g. skill_opsd/config.yaml). "
                        "Values read: context_builder.optimize_context, "
                        "context_builder.promptagent_iters, "
                        "context_builder.num_validation_problems.")
    p.add_argument("--optimize_context", action="store_true",
                   help="Enable simplified 3-round PromptAgent greedy optimization "
                        "over the global section-header templates.")
    p.add_argument("--promptagent_iters", type=int, default=3,
                   help="Rounds of greedy optimization (default 3). Ignored unless "
                        "--optimize_context.")
    p.add_argument("--promptagent_expand_width", type=int, default=3,
                   help="Candidate prompts per round (default 3).")
    p.add_argument("--promptagent_val_size", type=int, default=10,
                   help="Validation-set size for optimization (default 10).")
    p.add_argument("--llm_model", default="Qwen/Qwen3-4B",
                   help="HF model id used to instantiate the LLM callable for "
                        "PromptAgent optimization. Only used when --optimize_context "
                        "AND not --dry_run.")
    p.add_argument("--dry_run", action="store_true",
                   help="Mock all LLM calls (deterministic); never load torch.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_level", default="INFO",
                   help="python logging level (DEBUG, INFO, WARNING, ...)")
    return p.parse_args(argv)


def _build_real_llm_func(model_name: str) -> LLMFunc:  # pragma: no cover
    """Lazy-build a real LLM callable via transformers.

    Guarded behind ``--optimize_context`` and not called under ``--dry_run``.
    """
    # pylint: disable=import-outside-toplevel
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Real LLM mode requires `transformers` + `torch`; install them or "
            "use --dry_run."
        ) from exc

    LOGGER.info("Loading LLM for PromptAgent optimization: %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def _call(prompt: str, kind: str) -> str:
        max_new = 64 if kind == "candidate" else 256
        with torch.no_grad():
            inputs = tok(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            new_ids = out[0, inputs.input_ids.shape[-1]:]
            return tok.decode(new_ids, skip_special_tokens=True)

    return _call


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_yaml(args.config) if args.config else {}
    cb_cfg = (cfg.get("context_builder") or {}) if isinstance(cfg, dict) else {}
    # CLI flags take precedence; YAML is fallback default.
    optimize_context = bool(args.optimize_context or cb_cfg.get("optimize_context", False))
    if not args.optimize_context and cb_cfg.get("optimize_context"):
        LOGGER.info("config.yaml enabled optimize_context=true")
    pa_iters = int(cb_cfg.get("promptagent_iters", args.promptagent_iters))
    if args.promptagent_iters != 3:
        pa_iters = args.promptagent_iters  # CLI explicit override
    val_size = int(cb_cfg.get("num_validation_problems", args.promptagent_val_size))
    if args.promptagent_val_size != 10:
        val_size = args.promptagent_val_size

    # Load inputs.
    selected = _read_jsonl(args.selected)
    skills = _read_jsonl(args.skill_library)
    raws = _read_jsonl(args.questions_with_answers)
    LOGGER.info(
        "loaded: %d selected rows, %d library skills, %d raw Q/A",
        len(selected), len(skills), len(raws),
    )

    skill_index = _index_skill_library(skills)
    raw_index = _index_raw(raws)
    transition_prompt = _load_opsd_transition_prompt()

    templates = SectionTemplates()

    # Optional PromptAgent optimization over global templates.
    if optimize_context:
        LOGGER.info(
            "PromptAgent optimization enabled (iters=%d, val_size=%d, "
            "expand_width=%d, dry_run=%s)",
            pa_iters, val_size, args.promptagent_expand_width, args.dry_run,
        )
        rng = random.Random(args.seed)
        val_pool = [_ValProblem.from_raw(r) for r in raws]
        rng.shuffle(val_pool)
        val = val_pool[:max(1, min(val_size, len(val_pool)))]

        # Pre-index question -> selected skills for the optimizer.
        sel_index = {str(r.get("question_id")): list(r.get("selected_skill_ids") or [])
                     for r in selected}
        q_to_skills: Dict[str, List[Dict[str, Any]]] = {}
        for p in val:
            ids = sel_index.get(p.question_id, [])
            q_to_skills[p.question_id] = [
                skill_index[str(i)] for i in ids if str(i) in skill_index
            ]

        llm_func: LLMFunc = _mock_llm if args.dry_run else _build_real_llm_func(args.llm_model)
        agent = PromptAgentGreedy(
            llm_func=llm_func,
            iters=pa_iters,
            expand_width=args.promptagent_expand_width,
            seed=args.seed,
        )
        optimized = agent.optimize(
            initial=templates,
            val=val,
            question_to_skills=q_to_skills,
            transition_prompt=transition_prompt,
        )
        if optimized.__dict__ == templates.__dict__:
            LOGGER.info("PromptAgent kept baseline templates unchanged.")
        else:
            LOGGER.info("PromptAgent updated templates: %s", optimized.__dict__)
        templates = optimized

    rows = assemble_rows(
        selected=selected,
        skill_index=skill_index,
        raw_index=raw_index,
        transition_prompt=transition_prompt,
        templates=templates,
    )
    n = _write_jsonl(args.output, rows)
    LOGGER.info("wrote %d rows -> %s", n, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
