"""Skill extraction CLI for Skill-Informed OPSD (Task 2.1).

Reads a JSONL of reasoning trajectories, generates structured skills per
trajectory (SUCCESSFUL / FAILED templates), deduplicates against the existing
skill library via sentence-transformer cosine similarity, and appends the
survivors to `skill_library.jsonl`.

References:
    - SkillRL/memory_data/alfworld/claude_style_skills.json (skill schema)
    - SkillRL/skill_generation/alfworld.py (extraction prompt design)
    - reasoning-bank/WebArena/prompts/memory_instruction.py (SUCCESSFUL_SI / FAILED_SI)
    - EvolveR/evolver/experience/prompts.py (Guiding Principle)
    - EvolveR/evolver/experience/experience_manager.py (`_deduplicate_potential_principles`)
    - EvolveR/evolver/experience/config.py (SIMILARITY_THRESHOLD = 0.85)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("skill_extract")

SUCCESSFUL_TEMPLATE = """You are an expert in distilling mathematical reasoning trajectories into reusable skills.
You will be shown a problem, the model's reasoning trajectory, and the ground-truth answer.
The trajectory was SUCCESSFUL (it reached the correct final answer).

## Guidelines
Extract 1-{max_skills} reusable skills that helped the trajectory succeed. Focus on:
  - Problem decomposition / planning patterns
  - Correct use of mathematical identities / theorems
  - Verification / sanity-check habits
  - Structural reasoning moves (substitution, induction, casework)
Each skill must be generalizable beyond this specific problem. Do not copy numbers or \
problem-specific phrasing; keep skills transferable across problems of the same family.

## Important
  - Return STRICT JSON only. No prose before or after.
  - Do not repeat overlapping skills.
  - Each `principle` is 1-3 sentences of core advice.
  - Each `when_to_apply` is a specific trigger condition.

## Output Format (STRICT JSON ARRAY)
[
  {{"name": "Short title (3-7 words)", "principle": "...", "when_to_apply": "..."}}
]

## Problem
{question}

## Successful Trajectory
{trajectory}

## Ground Truth
{ground_truth}

Return ONLY the JSON array (no markdown fences, no commentary):
"""


FAILED_TEMPLATE = """You are an expert in diagnosing failures in mathematical reasoning trajectories.
You will be shown a problem, the model's reasoning trajectory, and the ground-truth answer.
The trajectory FAILED (the final answer is wrong or missing).

## Guidelines
Extract 1-{max_skills} cautionary skills that could have prevented this failure. Focus on:
  - Common arithmetic / algebraic mistakes
  - Missing case analysis / edge conditions
  - Premature answer commitment without verification
  - Misreading problem constraints
  - Skipped justifications that propagated errors
Each skill must be generalizable beyond this specific problem. Phrase each as an \
actionable corrective principle (what the reasoner SHOULD do), not a description of \
the mistake itself.

## Important
  - Return STRICT JSON only. No prose before or after.
  - Do not repeat overlapping skills.
  - Each `principle` is 1-3 sentences of corrective advice.
  - Each `when_to_apply` is the trigger condition where this skill rescues the reasoner.

## Output Format (STRICT JSON ARRAY)
[
  {{"name": "Short title (3-7 words)", "principle": "...", "when_to_apply": "..."}}
]

## Problem
{question}

## Failed Trajectory
{trajectory}

## Ground Truth
{ground_truth}

Return ONLY the JSON array (no markdown fences, no commentary):
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """Normalized representation of a reasoning trajectory."""

    question_id: str
    question: str
    trajectory: str
    is_correct: bool
    ground_truth: str = ""

    @classmethod
    def from_raw(cls, obj: Dict[str, Any], fallback_idx: int) -> "Trajectory":
        """Build a Trajectory from a raw JSONL row, tolerating common field aliases."""
        qid = str(
            obj.get("question_id")
            or obj.get("id")
            or obj.get("idx")
            or f"traj_{fallback_idx:05d}"
        )
        question = str(
            obj.get("question")
            or obj.get("problem")
            or obj.get("prompt")
            or obj.get("query")
            or ""
        )
        trajectory = str(
            obj.get("trajectory")
            or obj.get("answer")
            or obj.get("response")
            or obj.get("solution")
            or obj.get("model_response")
            or obj.get("completion")
            or ""
        )
        is_correct = bool(
            obj.get("is_correct")
            if obj.get("is_correct") is not None
            else obj.get("correct")
            if obj.get("correct") is not None
            else obj.get("reward", 0) > 0
        )
        gt = str(obj.get("ground_truth") or obj.get("answer_gt") or obj.get("gold") or "")
        return cls(
            question_id=qid,
            question=question,
            trajectory=trajectory,
            is_correct=is_correct,
            ground_truth=gt,
        )


@dataclass
class Skill:
    """Structured skill persisted to `skill_library.jsonl`."""

    skill_id: str
    name: str
    principle: str
    when_to_apply: str
    source: str  # "success" | "failure"
    source_question_ids: List[str] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    metric_score: float = 0.5  # (success+1)/(usage+2) with usage=0, success=0
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
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

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Skill":
        return cls(
            skill_id=str(obj["skill_id"]),
            name=str(obj.get("name", "")),
            principle=str(obj.get("principle", "")),
            when_to_apply=str(obj.get("when_to_apply", "")),
            source=str(obj.get("source", "success")),
            source_question_ids=list(obj.get("source_question_ids", [])),
            usage_count=int(obj.get("usage_count", 0)),
            success_count=int(obj.get("success_count", 0)),
            metric_score=float(obj.get("metric_score", 0.5)),
            created_at=str(obj.get("created_at", "")),
        )


# ---------------------------------------------------------------------------
# Config / IO
# ---------------------------------------------------------------------------


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file, tolerant of absence."""
    if not path or not os.path.exists(path):
        return {}
    import yaml  # local import to keep `--help` lightweight

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts (empty list if missing)."""
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
    """Write all rows to `path` atomically via tmp + os.replace."""
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
# Prompt building & response parsing
# ---------------------------------------------------------------------------


def build_prompt(traj: Trajectory, max_skills: int, max_traj_chars: int = 4000) -> str:
    """Return the extraction prompt for a single trajectory."""
    template = SUCCESSFUL_TEMPLATE if traj.is_correct else FAILED_TEMPLATE
    traj_text = traj.trajectory
    if len(traj_text) > max_traj_chars:
        traj_text = traj_text[:max_traj_chars] + "\n... [truncated]"
    return template.format(
        max_skills=max_skills,
        question=traj.question.strip() or "[empty]",
        trajectory=traj_text.strip() or "[empty]",
        ground_truth=str(traj.ground_truth).strip() or "[unknown]",
    )


_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_llm_skills(raw: str) -> List[Dict[str, str]]:
    """Best-effort parse LLM output into a list of `{name, principle, when_to_apply}` dicts."""
    if not raw:
        return []

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    candidates: List[Any] = []
    try:
        parsed = json.loads(cleaned)
        candidates = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        array_match = _JSON_ARRAY_RE.search(cleaned)
        if array_match:
            try:
                candidates = json.loads(array_match.group(0))
            except json.JSONDecodeError:
                candidates = []
        if not candidates:
            candidates = []
            for m in _JSON_OBJECT_RE.finditer(cleaned):
                try:
                    candidates.append(json.loads(m.group(0)))
                except json.JSONDecodeError:
                    continue

    out: List[Dict[str, str]] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or c.get("title") or "").strip()
        principle = str(c.get("principle") or c.get("content") or c.get("description") or "").strip()
        when = str(c.get("when_to_apply") or c.get("trigger") or c.get("applies_when") or "").strip()
        if not name or not principle:
            continue
        out.append({"name": name, "principle": principle, "when_to_apply": when})
    return out


# ---------------------------------------------------------------------------
# LLM extractor (lazy-loaded)
# ---------------------------------------------------------------------------


class HFExtractor:
    """Causal-LM wrapper that generates skill JSON per trajectory."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # local heavy imports
        from transformers import AutoModelForCausalLM, AutoTokenizer

        LOGGER.info("loading extractor model %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()

    def generate(self, prompt: str) -> str:
        """Return raw generated text for a single prompt."""
        self._ensure_loaded()
        import torch  # local

        messages = [{"role": "user", "content": prompt}]
        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def cosine_similarity(a, b) -> float:
    """Cosine similarity of two 1D numpy vectors, safe against zero norms."""
    import numpy as np  # local

    denom = float((a @ a) ** 0.5) * float((b @ b) ** 0.5)
    if denom == 0.0:
        return 0.0
    return float(a @ b / denom)


class SkillLibrary:
    """In-memory, growable skill library backed by a JSONL file."""

    def __init__(
        self,
        existing: Sequence[Skill],
        embedding_model: str,
        dedup_threshold: float,
    ) -> None:
        self.skills: List[Skill] = list(existing)
        self.embedding_model = embedding_model
        self.dedup_threshold = dedup_threshold
        self._embedder = None
        self._embeddings: List[Any] = []  # parallel to self.skills; numpy arrays

    # ---- embedder lifecycle --------------------------------------------------

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer  # local

        LOGGER.info("loading embedding model %s", self.embedding_model)
        self._embedder = SentenceTransformer(self.embedding_model)
        texts = [self._skill_text(s) for s in self.skills]
        if texts:
            self._embeddings = list(self._embedder.encode(texts, convert_to_numpy=True))
        else:
            self._embeddings = []

    @staticmethod
    def _skill_text(s: Skill) -> str:
        return f"{s.name}. {s.principle} When: {s.when_to_apply}"

    # ---- id / dedup helpers --------------------------------------------------

    def _next_skill_id(self) -> str:
        nums: List[int] = []
        for s in self.skills:
            m = re.match(r"skill_(\d+)$", s.skill_id)
            if m:
                nums.append(int(m.group(1)))
        nxt = (max(nums) + 1) if nums else 1
        return f"skill_{nxt:04d}"

    def _nearest_same_source(self, emb, source: str) -> Tuple[Optional[int], float]:
        best_idx: Optional[int] = None
        best_sim = -1.0
        for i, existing in enumerate(self.skills):
            if existing.source != source:
                continue
            sim = cosine_similarity(emb, self._embeddings[i])
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx, max(best_sim, 0.0)

    # ---- public API ----------------------------------------------------------

    def try_add(self, candidate: Skill) -> Tuple[bool, float, Optional[str]]:
        """Add `candidate` if it is not a near-duplicate of an existing skill.

        Returns `(added, max_similarity, merged_into_skill_id)`:
          - If a same-source neighbour exceeds `dedup_threshold`, merge question ids
            into that neighbour and return `(False, sim, merged_id)`.
          - Otherwise append the candidate with a fresh id and return
            `(True, sim, None)`.
        """
        self._ensure_embedder()
        emb = self._embedder.encode([self._skill_text(candidate)], convert_to_numpy=True)[0]
        idx, sim = self._nearest_same_source(emb, candidate.source)
        if idx is not None and sim >= self.dedup_threshold:
            existing = self.skills[idx]
            for qid in candidate.source_question_ids:
                if qid not in existing.source_question_ids:
                    existing.source_question_ids.append(qid)
            return False, sim, existing.skill_id
        candidate.skill_id = self._next_skill_id()
        self.skills.append(candidate)
        self._embeddings.append(emb)
        return True, sim, None

    def as_rows(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.skills]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def extract_skills_for_trajectory(
    traj: Trajectory,
    extractor: HFExtractor,
    max_skills_per_traj: int,
) -> List[Skill]:
    """Run the extractor on a single trajectory and return parsed `Skill` objects."""
    prompt = build_prompt(traj, max_skills=max_skills_per_traj)
    raw = extractor.generate(prompt)
    parsed = parse_llm_skills(raw)
    if not parsed:
        LOGGER.warning("no skills parsed for question_id=%s (raw head=%r)", traj.question_id, raw[:160])
        return []

    source = "success" if traj.is_correct else "failure"
    skills: List[Skill] = []
    for item in parsed[:max_skills_per_traj]:
        skills.append(
            Skill(
                skill_id="",  # assigned later by library
                name=item["name"],
                principle=item["principle"],
                when_to_apply=item["when_to_apply"],
                source=source,
                source_question_ids=[traj.question_id],
                metric_score=0.5,
                created_at=_now_iso(),
            )
        )
    return skills


def run(
    trajectories_path: str,
    skill_library_path: str,
    extractor: HFExtractor,
    max_skills_per_traj: int,
    dedup_threshold: float,
    embedding_model: str,
    dry_run: bool,
) -> Dict[str, Any]:
    """End-to-end pipeline. Returns a summary dict."""
    raw_rows = load_jsonl(trajectories_path)
    trajectories = [Trajectory.from_raw(r, idx) for idx, r in enumerate(raw_rows)]
    if dry_run:
        trajectories = trajectories[:3]
        LOGGER.info("dry-run: limiting to %d trajectories", len(trajectories))

    existing_rows = load_jsonl(skill_library_path)
    existing_skills = [Skill.from_dict(r) for r in existing_rows]
    LOGGER.info("loaded %d existing skills from %s", len(existing_skills), skill_library_path)

    library = SkillLibrary(
        existing=existing_skills,
        embedding_model=embedding_model,
        dedup_threshold=dedup_threshold,
    )

    total_generated = 0
    total_added = 0
    total_merged = 0

    for i, traj in enumerate(trajectories):
        if dry_run:
            prompt = build_prompt(traj, max_skills=max_skills_per_traj)
            print(
                f"\n=== [dry-run] trajectory {i+1}/{len(trajectories)} "
                f"(qid={traj.question_id}, correct={traj.is_correct}) ==="
            )
            print(prompt[:1200] + ("\n...[prompt truncated]\n" if len(prompt) > 1200 else "\n"))

        skills = extract_skills_for_trajectory(traj, extractor, max_skills_per_traj)
        total_generated += len(skills)

        for skill in skills:
            added, sim, merged_into = library.try_add(skill)
            if added:
                total_added += 1
                if dry_run:
                    print(f"  + ADDED   id={skill.skill_id} sim={sim:.3f} name={skill.name!r}")
            else:
                total_merged += 1
                if dry_run:
                    print(f"  ~ MERGED  into={merged_into} sim={sim:.3f} name={skill.name!r}")

    summary = {
        "trajectories": len(trajectories),
        "generated": total_generated,
        "added": total_added,
        "merged": total_merged,
        "library_size_before": len(existing_skills),
        "library_size_after": len(library.skills),
        "dry_run": dry_run,
    }

    if dry_run:
        LOGGER.info("dry-run summary: %s", summary)
    else:
        atomic_write_jsonl(skill_library_path, library.as_rows())
        LOGGER.info("wrote %d skills to %s (summary=%s)", len(library.skills), skill_library_path, summary)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skill_extract",
        description=(
            "Extract structured skills from reasoning trajectories and append them "
            "(dedup-aware) to skill_library.jsonl."
        ),
    )
    p.add_argument("--trajectories", required=True, help="Input JSONL of trajectories.")
    p.add_argument(
        "--skill_library",
        required=True,
        help="Output (append) JSONL path for the skill library.",
    )
    p.add_argument(
        "--extractor_model",
        default="Qwen/Qwen3-4B",
        help="HF causal-LM used for skill extraction (default: Qwen/Qwen3-4B).",
    )
    p.add_argument("--config", default="", help="Path to config.yaml (optional).")
    p.add_argument(
        "--max_skills_per_traj",
        type=int,
        default=3,
        help="Max skills to extract per trajectory (default 3).",
    )
    p.add_argument(
        "--dedup_threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold for same-source dedup (default: config or 0.85).",
    )
    p.add_argument(
        "--embedding_model",
        default=None,
        help="Sentence-transformer model for dedup (default: config or all-MiniLM-L6-v2).",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max new tokens for the extractor LLM (default 1024).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the extractor LLM (default 0.7).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only process first 3 trajectories, print prompts & dedup decisions, no write.",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        help="Python logging level (default INFO).",
    )
    return p


def resolve_config(args: argparse.Namespace) -> Tuple[float, str]:
    """Merge CLI flags with config.yaml (CLI wins). Returns (dedup_threshold, embedding_model)."""
    cfg = load_yaml_config(args.config)
    se = cfg.get("skill_extraction", {}) if isinstance(cfg, dict) else {}
    dedup_threshold = (
        args.dedup_threshold
        if args.dedup_threshold is not None
        else float(se.get("dedup_similarity_threshold", 0.85))
    )
    embedding_model = (
        args.embedding_model
        or se.get("embedding_model")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    return dedup_threshold, embedding_model


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dedup_threshold, embedding_model = resolve_config(args)

    extractor = HFExtractor(
        model_name=args.extractor_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    summary = run(
        trajectories_path=args.trajectories,
        skill_library_path=args.skill_library,
        extractor=extractor,
        max_skills_per_traj=args.max_skills_per_traj,
        dedup_threshold=dedup_threshold,
        embedding_model=embedding_model,
        dry_run=args.dry_run,
    )
    print(json.dumps({"skill_extract_summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
