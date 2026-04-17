"""TR / TL attribution diagnostic for Skill-Informed OPSD (Task 5.2).

Port of Pan et al. (2023) "Rethinking Label-Free In-Context Learning" to
the **context** dimension. The original paper toggles the *label* of a
demo among {gold / random / abstract}; here we instead toggle the
*context block* among:

    A  full        : layered skill markdown as produced by Task 3.3
    B  structure   : keep the section headers, replace each skill's
                     principle / when_to_apply with ``<REDACTED>`` (i.e.
                     equal structure, no useful content).
    C  content     : keep the skill principles as raw text (no markdown,
                     no headers), joined by blank lines.

plus a ``baseline`` setting with no skills (0-shot OPSD transition).

Accuracy deltas attribute the gain:

    TL = A - B              # CONTENT contribution (text beyond structure)
    TR = B - baseline       # STRUCTURE / format contribution

Design notes
------------
* We reuse the canonical ``build_teacher_prompt`` + ``SectionTemplates``
  from ``context_builder.py`` (Task 3.3) for setting A; we do NOT edit
  that module. Settings B/C are built in this file.
* For CEIL top-K we reuse ``select_skills`` from ``context_select.py``
  (Task 3.2).
* Heavy deps (torch / vllm / sentence-transformers) are imported lazily:
  ``--dry_run`` and ``--help`` must never load them.
* Inference (real mode) is delegated to ``collect_trajectories.py``
  (Task 5.1) via a best-effort import — see ``_resolve_inference_fn``.
  If Task 5.1 is not yet available at import time, we fall back to an
  in-module minimal HF-transformers ``generate`` + ``\\boxed{}``
  judge. This choice is documented in the report's ``notes`` field
  under ``inference_backend``.

References
----------
* Pan et al. 2023 gold/random/abstract label ablation.
* /home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/context_builder.py
* /home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/context_select.py
* /home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd/collect_trajectories.py (Task 5.1, optional)
* IMPLEMENTATION_PLAN.md Task 5.2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("tr_tl_diagnostic")

_HERE = os.path.dirname(os.path.abspath(__file__))

# default placeholder used in setting B (structure-only).
_REDACTED = "<REDACTED>"


# ---------------------------------------------------------------------------
# IO helpers (duplicated intentionally — we must not alter context_builder.py)
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
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


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("pyyaml unavailable (%s); ignoring --config", exc)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Load companion modules (Task 3.2, 3.3) via importlib.util.
# We avoid ``sys.path.insert`` so we don't pollute the caller's env, and we
# don't want to force Task 3.x to be importable as a package.
# ---------------------------------------------------------------------------
def _load_sibling_module(module_name: str, filename: str):
    """Load a sibling .py by absolute path using importlib.util.

    The module is registered in ``sys.modules`` **before** ``exec_module``
    runs, otherwise ``@dataclass`` inside the sibling would fail with
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``
    when it resolves type hints via ``sys.modules[cls.__module__]``.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = os.path.join(_HERE, filename)
    if not os.path.exists(path):
        raise ImportError(f"sibling module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def _get_context_builder():
    return _load_sibling_module("_skill_opsd_context_builder", "context_builder.py")


def _get_context_select():
    return _load_sibling_module("_skill_opsd_context_select", "context_select.py")


# ---------------------------------------------------------------------------
# Three-setting prompt constructors
# ---------------------------------------------------------------------------
@dataclass
class PromptBundle:
    """A prompt variant plus the skills it references (for logging)."""
    setting: str          # "A" / "B" / "C" / "baseline"
    k: int
    prompt: str
    question_id: str
    skill_ids: List[str]


def _rand_redacted(principle: str, rng: random.Random) -> str:
    """Placeholder body of the same rough shape as ``principle``.

    We keep length ~equal to the redacted text so tokenizer length bias
    doesn't sneak in as a confound. The placeholder is word-wise not
    character-wise (LLM tokenizers are usually word-ish), so we emit
    ``N`` copies of the sentinel where ``N = word_count(principle)``.
    """
    n_words = max(1, len(principle.split()))
    return " ".join([_REDACTED] * n_words)


def build_prompt_A(
    question: str,
    skills: List[Dict[str, Any]],
    transition_prompt: str,
    cb_mod,
) -> str:
    """Setting A: delegate to Task 3.3's ``build_teacher_prompt``."""
    return cb_mod.build_teacher_prompt(
        question=question,
        selected_skills=skills,
        transition_prompt=transition_prompt,
        templates=cb_mod.SectionTemplates(),
    )


def build_prompt_B(
    question: str,
    skills: List[Dict[str, Any]],
    transition_prompt: str,
    cb_mod,
    seed: int = 0,
) -> str:
    """Setting B: same structure as A but skill bodies are redacted.

    We rebuild the prompt by handing the upstream builder a **deep-copy**
    of each skill with its principle / when_to_apply replaced by
    ``<REDACTED> ...``. The headers + bullet skeleton are preserved
    because Task 3.3 emits them unconditionally when ``skills`` is
    non-empty, while the informational content is wiped.
    """
    rng = random.Random(seed)
    scrubbed: List[Dict[str, Any]] = []
    for s in skills:
        cp = dict(s)
        # keep the name but zero-out body text.
        pr = str(s.get("principle", "")).strip()
        wt = str(s.get("when_to_apply", "")).strip()
        cp["principle"] = _rand_redacted(pr, rng) if pr else ""
        cp["when_to_apply"] = _rand_redacted(wt, rng) if wt else ""
        scrubbed.append(cp)

    return cb_mod.build_teacher_prompt(
        question=question,
        selected_skills=scrubbed,
        transition_prompt=transition_prompt,
        templates=cb_mod.SectionTemplates(),
    )


def build_prompt_C(
    question: str,
    skills: List[Dict[str, Any]],
    transition_prompt: str,
) -> str:
    """Setting C: content-only — no section headers, no markdown.

    We keep the OPSD transition + restated question + "Solution:" tail
    so the task framing matches A/B; the ONLY difference is that the
    layered skill markdown is replaced by raw principle paragraphs
    joined by blank lines.
    """
    paragraphs: List[str] = []
    for s in skills:
        pr = str(s.get("principle", "")).strip()
        wt = str(s.get("when_to_apply", "")).strip()
        parts: List[str] = []
        if pr:
            parts.append(pr)
        if wt:
            parts.append(wt)
        if parts:
            paragraphs.append(" ".join(parts))
    body_parts: List[str] = []
    if paragraphs:
        body_parts.append("\n\n".join(paragraphs))
        body_parts.append("---")
    body_parts.append(transition_prompt)
    body_parts.append(f"Problem: {question}")
    body_parts.append("Solution:")
    return "\n\n".join(body_parts)


def build_prompt_baseline(
    question: str,
    transition_prompt: str,
) -> str:
    """Baseline: no skills — pure OPSD transition + question."""
    return "\n\n".join([
        transition_prompt,
        f"Problem: {question}",
        "Solution:",
    ])


# ---------------------------------------------------------------------------
# Skill selection for a single question
# ---------------------------------------------------------------------------
def _select_for_question(
    cs_mod,
    question: Dict[str, Any],
    skills: List[Dict[str, Any]],
    k: int,
    dry_run: bool,
    embedding_model: str,
) -> List[Dict[str, Any]]:
    """Select up to ``k`` skills for ``question`` using Task 3.2.

    If ``k`` exceeds ``len(skills)`` we silently fall back to the library
    size (this is a unit-tested behavior — see ``--ks 8`` on a library of
    2). We always pick with ``mode='ceil'`` in dry-run / non-dry-run for
    consistency; GenICL mode would require a separate selector ckpt and
    is out of scope for a diagnostic run.
    """
    if not skills:
        return []
    top_k = min(int(k), len(skills))
    rows = cs_mod.select_skills(
        questions=[question],
        skills=skills,
        mode="ceil",
        top_k=top_k,
        embedding_model=embedding_model,
        dry_run=dry_run,
        genicl_selector_path=None,
    )
    ids = rows[0].get("selected_skill_ids", []) if rows else []
    by_id = {str(s.get("skill_id")): s for s in skills}
    return [by_id[str(i)] for i in ids if str(i) in by_id]


# ---------------------------------------------------------------------------
# Inference reuse (Task 5.1)
# ---------------------------------------------------------------------------
GenerateJudgeFn = Callable[[List[str], List[str]], List[bool]]


def _boxed_answer(text: str) -> Optional[str]:
    """Extract ``\\boxed{...}`` content from a model response."""
    if not text:
        return None
    # Accept nested braces greedily up to line end, but strip trailing }.
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if not m:
        return None
    return m.group(1).strip()


def _answers_equal(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    p = pred.strip().rstrip(".")
    g = str(gold).strip().rstrip(".")
    if p == g:
        return True
    # numeric tolerance
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return False


def _try_import_collect_trajectories():
    """Best-effort import of Task 5.1's ``sample_and_judge`` helper.

    Returns the callable if available, else ``None``. We do NOT raise —
    the caller falls back to the in-module implementation.
    """
    path = os.path.join(_HERE, "collect_trajectories.py")
    if not os.path.exists(path):
        return None, "collect_trajectories.py not found on disk"
    try:
        mod = _load_sibling_module("_skill_opsd_collect_trajectories", "collect_trajectories.py")
    except Exception as exc:
        return None, f"import failed: {exc}"
    fn = getattr(mod, "sample_and_judge", None)
    if fn is None:
        return None, "collect_trajectories.sample_and_judge not exported"
    return fn, "reused collect_trajectories.sample_and_judge"


def _build_hf_infer_fn(
    model_name: str,
    adapters: Optional[str],
    temperature: float,
    max_new_tokens: int,
) -> GenerateJudgeFn:  # pragma: no cover - heavy, exercised only in real runs
    """Fallback inference path using raw HF transformers + PEFT.

    Used only when Task 5.1 is not importable. Kept minimal on purpose
    — the point of a diagnostic is to reuse the production inference
    path once it's landed.
    """
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    LOGGER.info("[inference] HF fallback: loading %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    if adapters:
        try:
            from peft import PeftModel  # type: ignore
        except Exception as exc:
            raise RuntimeError("--adapters requires peft") from exc
        LOGGER.info("[inference] attaching adapters: %s", adapters)
        model = PeftModel.from_pretrained(model, adapters)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def _fn(prompts: List[str], gts: List[str]) -> List[bool]:
        corrects: List[bool] = []
        for prompt, gt in zip(prompts, gts):
            enc = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    temperature=max(1e-3, float(temperature)),
                    do_sample=float(temperature) > 0.0,
                    pad_token_id=tok.pad_token_id,
                )
            new_ids = out[0, enc.input_ids.shape[-1]:]
            resp = tok.decode(new_ids, skip_special_tokens=True)
            corrects.append(_answers_equal(_boxed_answer(resp), gt))
        return corrects

    return _fn


def _build_mock_infer_fn(
    per_setting_accuracies: Optional[Dict[Tuple[str, int], float]] = None,
    seed: int = 0,
) -> GenerateJudgeFn:
    """Deterministic mock used by ``--dry_run`` / self-tests.

    Accepts a per-(setting, k) target accuracy map; when ``None``, picks
    is_correct uniformly at random with a fixed seed. This is what the
    self-tests use to verify TL / TR arithmetic.
    """
    rng = random.Random(seed)
    state: Dict[Tuple[str, int], int] = {}

    def _fn(prompts: List[str], gts: List[str], setting: str = "?", k: int = 0) -> List[bool]:
        out: List[bool] = []
        for _ in prompts:
            if per_setting_accuracies is not None:
                tgt = per_setting_accuracies.get((setting, k))
                if tgt is None:
                    # try setting-only
                    tgt = per_setting_accuracies.get((setting, -1))
                if tgt is None:
                    tgt = 0.5
                # deterministic threshold per call to approximate mean = tgt
                cnt = state.get((setting, k), 0)
                # schedule: first ceil(tgt*total) are True — total unknown
                # so we fall back to Bernoulli with the rng for stability.
                out.append(rng.random() < float(tgt))
                state[(setting, k)] = cnt + 1
            else:
                out.append(rng.random() < 0.5)
        return out

    return _fn  # type: ignore[return-value]


def _resolve_inference_fn(
    dry_run: bool,
    model_name: str,
    adapters: Optional[str],
    temperature: float,
    max_new_tokens: int,
    mock_acc_map: Optional[Dict[Tuple[str, int], float]] = None,
) -> Tuple[GenerateJudgeFn, str]:
    """Pick inference backend.

    Order of preference (per Task 5.2 §E):
      1. dry-run -> mock
      2. collect_trajectories.sample_and_judge (if importable)
      3. in-module HF transformers fallback
    """
    if dry_run:
        return _build_mock_infer_fn(mock_acc_map), "dry_run (mock)"
    fn, note = _try_import_collect_trajectories()
    if fn is not None:
        # we require fn(prompts, gts, setting=..., k=...) - but 5.1 may
        # not accept setting/k kwargs. Wrap it to accept & drop them.
        def _wrapped(prompts: List[str], gts: List[str], **_: Any) -> List[bool]:
            return list(fn(
                prompts=prompts,
                ground_truths=gts,
                model=model_name,
                adapters=adapters,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            ))
        return _wrapped, note
    LOGGER.warning("collect_trajectories.sample_and_judge unavailable; "
                   "falling back to in-module HF inference (%s)", note)
    return _build_hf_infer_fn(model_name, adapters, temperature, max_new_tokens), \
        f"hf_fallback ({note})"


# ---------------------------------------------------------------------------
# Driver: build prompts, run inference, aggregate accuracy
# ---------------------------------------------------------------------------
def _evaluate_setting(
    prompts: List[PromptBundle],
    gts: List[str],
    infer_fn: GenerateJudgeFn,
    setting: str,
    k: int,
) -> float:
    """Run inference over a bucket of prompts; return accuracy in [0,1]."""
    if not prompts:
        return 0.0
    # Some infer_fn implementations accept kwargs for logging; try both.
    try:
        corrects = infer_fn([p.prompt for p in prompts], gts, setting=setting, k=k)  # type: ignore
    except TypeError:
        corrects = infer_fn([p.prompt for p in prompts], gts)
    if len(corrects) != len(prompts):
        raise RuntimeError(
            f"inference returned {len(corrects)} judgments for {len(prompts)} prompts"
        )
    return sum(1 for c in corrects if bool(c)) / len(prompts)


def run_diagnostic(
    model: str,
    adapters: Optional[str],
    skill_library: List[Dict[str, Any]],
    eval_questions: List[Dict[str, Any]],
    ks: List[int],
    num_eval: int,
    temperature: float,
    max_new_tokens: int,
    dry_run: bool,
    embedding_model: str,
    mock_acc_map: Optional[Dict[Tuple[str, int], float]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the A/B/C + baseline sweep and return the report dict."""
    cb = _get_context_builder()
    cs = _get_context_select()
    transition_prompt = cb._load_opsd_transition_prompt()

    rng = random.Random(seed)
    eval_q = list(eval_questions)
    if num_eval < len(eval_q):
        rng.shuffle(eval_q)
        eval_q = eval_q[:num_eval]
    n_eval = len(eval_q)
    LOGGER.info("diagnostic: n_eval=%d ks=%s dry_run=%s",
                n_eval, ks, dry_run)

    # 1. Precompute per-k skill selections (one pass per k; reuses CEIL).
    per_k_selections: Dict[int, List[List[Dict[str, Any]]]] = {}
    for k in ks:
        sel_list: List[List[Dict[str, Any]]] = []
        for q in eval_q:
            sel = _select_for_question(
                cs_mod=cs,
                question=q,
                skills=skill_library,
                k=k,
                dry_run=dry_run,
                embedding_model=embedding_model,
            )
            sel_list.append(sel)
        per_k_selections[k] = sel_list
        if skill_library and int(k) > len(skill_library):
            LOGGER.info("k=%d > |library|=%d -> fell back to %d",
                        k, len(skill_library), len(skill_library))

    # 2. Build baseline prompts (independent of k).
    baseline_prompts: List[PromptBundle] = [
        PromptBundle(
            setting="baseline",
            k=0,
            prompt=build_prompt_baseline(q.get("question", ""), transition_prompt),
            question_id=str(q.get("question_id", "")),
            skill_ids=[],
        )
        for q in eval_q
    ]

    # 3. Build A/B/C prompts for each k.
    all_prompts: List[PromptBundle] = list(baseline_prompts)
    per_k_prompts: Dict[int, Dict[str, List[PromptBundle]]] = {}
    for k in ks:
        bundles: Dict[str, List[PromptBundle]] = {"A": [], "B": [], "C": []}
        for q, skills in zip(eval_q, per_k_selections[k]):
            qid = str(q.get("question_id", ""))
            qtxt = q.get("question", "")
            sids = [str(s.get("skill_id", "")) for s in skills]
            bundles["A"].append(PromptBundle(
                "A", k,
                build_prompt_A(qtxt, skills, transition_prompt, cb),
                qid, sids,
            ))
            bundles["B"].append(PromptBundle(
                "B", k,
                build_prompt_B(qtxt, skills, transition_prompt, cb, seed=seed),
                qid, sids,
            ))
            bundles["C"].append(PromptBundle(
                "C", k,
                build_prompt_C(qtxt, skills, transition_prompt),
                qid, sids,
            ))
        per_k_prompts[k] = bundles
        for sb in bundles.values():
            all_prompts.extend(sb)

    LOGGER.info("built %d total prompts (expected %d)",
                len(all_prompts), n_eval * (1 + 3 * len(ks)))

    # 4. Inference.
    infer_fn, backend_note = _resolve_inference_fn(
        dry_run=dry_run,
        model_name=model,
        adapters=adapters,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        mock_acc_map=mock_acc_map,
    )
    gts = [str(q.get("ground_truth", "")) for q in eval_q]

    # 4a. baseline accuracy
    baseline_acc = _evaluate_setting(baseline_prompts, gts, infer_fn, "baseline", 0)

    # 4b. per-k A/B/C accuracies
    results: List[Dict[str, Any]] = []
    for k in ks:
        per_setting: Dict[str, float] = {}
        for setting in ("A", "B", "C"):
            per_setting[setting] = _evaluate_setting(
                per_k_prompts[k][setting], gts, infer_fn, setting, k,
            )
        A = per_setting["A"]
        B = per_setting["B"]
        C = per_setting["C"]
        row = {
            "k": int(k),
            "effective_k": min(int(k), len(skill_library)) if skill_library else 0,
            "A": round(A, 4),
            "B": round(B, 4),
            "C": round(C, 4),
            "TL": round(A - B, 4),
            "TR": round(B - baseline_acc, 4),
        }
        results.append(row)

    return {
        "model": model,
        "adapters": adapters,
        "num_eval": n_eval,
        "ks": [int(k) for k in ks],
        "baseline_acc": round(baseline_acc, 4),
        "results": results,
        "notes": (
            "TL = A - B (content contribution); TR = B - baseline "
            "(structure contribution). Setting A = full layered markdown "
            "(Task 3.3 teacher_prompt); Setting B = same headers, skill "
            f"bodies replaced by '{_REDACTED}' tokens (equal-length); "
            "Setting C = skill principles joined by blank lines, no "
            "headers or markdown. "
            f"Inference backend: {backend_note}."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tr_tl_diagnostic.py",
        description=(
            "TR / TL attribution diagnostic for Skill-Informed OPSD "
            "(Task 5.2). Ports Pan et al. 2023's gold/random/abstract "
            "label ablation to the context dimension: full / "
            "structure-only / content-only contexts across demo counts "
            "k in {1, 3, 5, 8}."
        ),
    )
    p.add_argument("--model", required=True,
                   help="HF model id (e.g. Qwen/Qwen3-4B).")
    p.add_argument("--adapters", default=None,
                   help="Optional LoRA adapter dir for the student checkpoint.")
    p.add_argument("--skill_library", required=True,
                   help="JSONL of library skills (Task 2.1/2.2/3.1 output).")
    p.add_argument("--eval_questions", required=True,
                   help="JSONL with {question_id, question, ground_truth} per line.")
    p.add_argument("--output", required=True,
                   help="Where to write the diagnostic report JSON.")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 8],
                   help="Demo counts to sweep (default: 1 3 5 8). Values > library "
                        "size silently fall back to library size.")
    p.add_argument("--num_eval", type=int, default=30,
                   help="Max eval questions per (setting, k) pair (default: 30).")
    p.add_argument("--config", default=None,
                   help="Optional YAML config (e.g. skill_opsd/config.yaml). "
                        "Values read: training.temperature, "
                        "training.max_completion_length, "
                        "skill_extraction.embedding_model.")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Sampling temperature for diagnostic inference (default 0.3).")
    p.add_argument("--max_new_tokens", type=int, default=2048,
                   help="Max new tokens per generation (default 2048).")
    p.add_argument("--embedding_model",
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="sentence-transformers model for CEIL skill selection.")
    p.add_argument("--dry_run", action="store_true",
                   help="Mock all inference & embedding; never loads torch / vllm.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for deterministic dry-run / sub-sampling.")
    p.add_argument("--log_level", default="INFO",
                   help="python logging level (DEBUG, INFO, WARNING, ...)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_yaml(args.config)
    tr_cfg = (cfg.get("training") or {}) if isinstance(cfg, dict) else {}
    se_cfg = (cfg.get("skill_extraction") or {}) if isinstance(cfg, dict) else {}

    # CLI > YAML > argparse default. argparse already sets defaults, so we
    # only patch in YAML when the CLI flag is still at its default.
    if args.temperature == 0.3 and "temperature" in tr_cfg:
        try:
            args.temperature = float(tr_cfg["temperature"])
        except Exception:
            pass
    if args.max_new_tokens == 2048 and "max_completion_length" in tr_cfg:
        try:
            args.max_new_tokens = int(tr_cfg["max_completion_length"])
        except Exception:
            pass
    if args.embedding_model == "sentence-transformers/all-MiniLM-L6-v2" and \
            "embedding_model" in se_cfg:
        args.embedding_model = str(se_cfg["embedding_model"])

    skills = _read_jsonl(args.skill_library)
    eval_qs = _read_jsonl(args.eval_questions)
    LOGGER.info("loaded %d library skills, %d eval questions",
                len(skills), len(eval_qs))
    if not skills:
        LOGGER.warning("skill_library is empty — A/B settings will degenerate to baseline")
    if not eval_qs:
        raise SystemExit(f"eval_questions file is empty: {args.eval_questions}")

    report = run_diagnostic(
        model=args.model,
        adapters=args.adapters,
        skill_library=skills,
        eval_questions=eval_qs,
        ks=list(args.ks),
        num_eval=int(args.num_eval),
        temperature=float(args.temperature),
        max_new_tokens=int(args.max_new_tokens),
        dry_run=bool(args.dry_run),
        embedding_model=str(args.embedding_model),
        seed=int(args.seed),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    LOGGER.info("wrote diagnostic report -> %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
