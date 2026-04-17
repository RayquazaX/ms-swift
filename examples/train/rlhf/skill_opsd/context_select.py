"""Context (skill) selection CLI for Skill-Informed OPSD (Task 3.2).

For every question, pick the ``top-K`` most helpful skills from
``skill_library.jsonl``. Two selection modes are supported:

* ``ceil`` (default)
    CEIL-style DPP MAP inference that balances *relevance* (cosine(q, s_i))
    and *diversity* (cosine(s_i, s_j)). The kernel is constructed as
    ``diag(rel) @ sim @ diag(rel)`` and solved greedily via
    ``fast_map_dpp``.

* ``genicl``
    Use a pre-trained selector (HF causal LM, optionally + PEFT/LoRA
    adapter) that scores ``(query, skill_candidate)`` pairs via average
    log-probability (GenICL style; see
    ``GenICL_preferred/src/llms/gpt2.py::batch_score``). Top-K highest-
    scoring skills are returned per question.

References:
    - /home/qzheng19/CEIL/src/utils/dpp_map.py (``fast_map_dpp``)
    - /home/qzheng19/CEIL/dense_retriever.py (``get_kernel``)
    - /home/qzheng19/GenICL_preferred/src/inference/inference_2_theta_peft.py
    - /home/qzheng19/GenICL_preferred/src/loaders/ktodataset_theta_lora.py
    - /home/qzheng19/GenICL_preferred/src/train_kto_lora.py
    - IMPLEMENTATION_PLAN.md §Task 3.2

Design notes:
    * ``fast_map_dpp`` is *embedded* locally (copied from CEIL) rather than
      imported via ``sys.path.insert``. Reasons: the CEIL source pulls in
      ``dppy`` at module import time, which is a heavy dep we do not need
      for pure MAP inference; and copying 25 lines avoids a hard
      cross-repo coupling.
    * sentence-transformers / torch are imported lazily (inside
      ``_encode_real``) so ``--help`` and ``--dry_run`` stay cheap and
      the script does not need CUDA to run its unit tests.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger("context_select")


# ---------------------------------------------------------------------------
# fast_map_dpp (verbatim from /home/qzheng19/CEIL/src/utils/dpp_map.py L9-34)
# ---------------------------------------------------------------------------
def fast_map_dpp(kernel_matrix: np.ndarray, max_length: int) -> List[int]:
    """Greedy MAP inference for a DPP; O(max_length * item_size).

    Paper: "Fast Greedy MAP Inference for Determinantal Point Process to
    Improve Recommendation Diversity" (Chen et al., NeurIPS 2018).

    Reference: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp_test.py
    """
    item_size = kernel_matrix.shape[0]
    max_length = min(max_length, item_size)
    if max_length <= 0:
        return []
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items: List[int] = []
    selected_item = int(np.argmax(di2s))
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(max(di2s[selected_item], 1e-20))
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        # numerical guard: avoid picking an already-selected item whose
        # residual is ~0 but negative due to float error
        di2s[selected_items] = -np.inf
        selected_item = int(np.argmax(di2s))
        if di2s[selected_item] <= 0:
            break
        selected_items.append(selected_item)
    return selected_items


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


def _skill_text(skill: Dict[str, Any]) -> str:
    """Concatenate ``name`` + ``principle`` + ``when_to_apply`` into one
    string for embedding (Task 2.1 skill schema)."""
    name = str(skill.get("name", "")).strip()
    principle = str(skill.get("principle", "")).strip()
    when = str(skill.get("when_to_apply", "")).strip()
    parts = [p for p in (name, principle, when) if p]
    return ". ".join(parts)


# ---------------------------------------------------------------------------
# Embedding (real vs dry)
# ---------------------------------------------------------------------------
def _encode_dry(texts: Sequence[str], dim: int = 64, seed: int = 0) -> np.ndarray:
    """Deterministic pseudo-embedding from text hashes (no heavy deps).

    This is ONLY used for ``--dry_run`` / unit tests. It intentionally
    avoids sentence-transformers so self-tests run without GPUs / network.
    """
    rng = np.random.default_rng(seed)
    embs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hash(t) & 0xFFFFFFFF
        rng_i = np.random.default_rng(h ^ seed)
        v = rng_i.standard_normal(dim).astype(np.float32)
        # small global noise so repeats of the same text still hit the
        # same vector (determinism) while different texts differ broadly
        embs[i] = v + 1e-6 * rng.standard_normal(dim)
    # L2 normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return embs / norms


def _encode_real(texts: Sequence[str], model_name: str, device: Optional[str] = None,
                 batch_size: int = 32) -> np.ndarray:
    """Lazy-import sentence-transformers and encode."""
    # pylint: disable=import-outside-toplevel
    from sentence_transformers import SentenceTransformer  # type: ignore
    import torch  # type: ignore

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Loading sentence-transformer: %s on %s", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embs, dtype=np.float32)


# ---------------------------------------------------------------------------
# CEIL mode
# ---------------------------------------------------------------------------
def _build_kernel(q_emb: np.ndarray, skill_embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build CEIL-style DPP kernel = diag(rel) @ sim @ diag(rel).

    ``q_emb``      : (d,) unit-normed
    ``skill_embs`` : (N, d) unit-normed
    Returns ``(rel, kernel)`` both in [0, 1]-ish range.
    """
    # relevance: cosine similarity, mapped to [0, 1]
    rel = skill_embs @ q_emb
    rel = (rel + 1.0) / 2.0
    rel = np.clip(rel, 1e-6, 1.0)

    # similarity: symmetric cosine matrix, mapped to [0, 1] so
    # the resulting kernel stays positive semi-definite after scaling
    sim = skill_embs @ skill_embs.T
    sim = (sim + 1.0) / 2.0

    # kernel = diag(rel) @ sim @ diag(rel)
    kernel = rel[:, None] * sim * rel[None, :]
    return rel, kernel


def _select_ceil(
    q_emb: np.ndarray,
    skill_embs: np.ndarray,
    top_k: int,
) -> List[int]:
    if skill_embs.shape[0] == 0 or top_k <= 0:
        return []
    _, kernel = _build_kernel(q_emb, skill_embs)
    return fast_map_dpp(kernel, max_length=min(top_k, skill_embs.shape[0]))


# ---------------------------------------------------------------------------
# GenICL mode
# ---------------------------------------------------------------------------
@dataclass
class GenICLSelector:
    """Thin wrapper around a HF causal LM + optional PEFT adapter.

    Scores ``(query, skill_text)`` pairs as mean log-prob of the skill
    text conditioned on the query. Mirrors
    ``GenICL_preferred/src/llms/gpt2.py::batch_score``.
    """

    selector_path: str
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if not os.path.isdir(self.selector_path) and not os.path.isfile(
            os.path.join(self.selector_path, "config.json")
        ):
            # accept either a directory with config.json or a full snapshot
            if not os.path.exists(self.selector_path):
                raise FileNotFoundError(
                    f"GenICL selector path does not exist: {self.selector_path}"
                )
        self._load()

    def _load(self) -> None:
        # pylint: disable=import-outside-toplevel
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "GenICL mode requires `transformers` + `torch` to be installed"
            ) from exc

        adapter_cfg = os.path.join(self.selector_path, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            # PEFT / LoRA adapter directory
            try:
                from peft import PeftModel  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    f"Selector at {self.selector_path} looks like a PEFT "
                    f"adapter (adapter_config.json present) but `peft` is not "
                    f"installed."
                ) from exc
            with open(adapter_cfg, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            base = cfg.get("base_model_name_or_path")
            if not base:
                raise RuntimeError(
                    f"adapter_config.json at {adapter_cfg} does not name a "
                    f"`base_model_name_or_path`."
                )
            LOGGER.info("Loading GenICL base model %s + adapter %s", base, self.selector_path)
            tok = AutoTokenizer.from_pretrained(base)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(base)
            model = PeftModel.from_pretrained(model, self.selector_path)
        else:
            LOGGER.info("Loading GenICL full model from %s", self.selector_path)
            tok = AutoTokenizer.from_pretrained(self.selector_path)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(self.selector_path)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(self.device).eval()
        self.tokenizer = tok
        self.model = model
        self._torch = torch

    def score_pairs(self, query: str, skill_texts: Sequence[str],
                    batch_size: int = 16) -> np.ndarray:
        """Return average log-prob of each ``skill_text`` given ``query``."""
        # pylint: disable=import-outside-toplevel
        import torch  # type: ignore
        tok = self.tokenizer
        model = self.model
        delim = "\n"

        scores: List[float] = []
        for start in range(0, len(skill_texts), batch_size):
            batch = list(skill_texts[start:start + batch_size])
            q_enc = tok([query] * len(batch), add_special_tokens=False)
            s_enc = tok(batch, add_special_tokens=False)
            delim_ids = tok(delim, add_special_tokens=False)["input_ids"]

            input_ids_list, labels_list = [], []
            for q_ids, s_ids in zip(q_enc["input_ids"], s_enc["input_ids"]):
                full = list(q_ids) + list(delim_ids) + list(s_ids)
                # only score the skill tokens
                lbl = ([-100] * (len(q_ids) + len(delim_ids))) + list(s_ids)
                input_ids_list.append(full)
                labels_list.append(lbl)

            max_len = max(len(x) for x in input_ids_list)
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
            input_ids = np.full((len(batch), max_len), pad_id, dtype=np.int64)
            attn = np.zeros((len(batch), max_len), dtype=np.int64)
            labels = np.full((len(batch), max_len), -100, dtype=np.int64)
            for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
                input_ids[i, : len(ids)] = ids
                attn[i, : len(ids)] = 1
                labels[i, : len(lbl)] = lbl

            device = self.device
            with torch.no_grad():
                out = model(
                    input_ids=torch.from_numpy(input_ids).to(device),
                    attention_mask=torch.from_numpy(attn).to(device),
                )
                logits = out.logits  # (B, T, V)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = torch.from_numpy(labels).to(device)[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                per_tok = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.shape)
                mask = (shift_labels != -100).float()
                nll = (per_tok * mask).sum(dim=1)
                n_valid = mask.sum(dim=1).clamp(min=1.0)
                avg_logp = (-nll / n_valid).cpu().tolist()
            scores.extend(avg_logp)
        return np.asarray(scores, dtype=np.float32)


def _select_genicl(
    selector: GenICLSelector,
    question: str,
    skill_texts: Sequence[str],
    top_k: int,
) -> List[int]:
    if not skill_texts or top_k <= 0:
        return []
    scores = selector.score_pairs(question, skill_texts)
    order = np.argsort(-scores)[: min(top_k, len(skill_texts))]
    return order.tolist()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def select_skills(
    questions: List[Dict[str, Any]],
    skills: List[Dict[str, Any]],
    mode: str,
    top_k: int,
    embedding_model: str,
    dry_run: bool,
    genicl_selector_path: Optional[str],
) -> List[Dict[str, Any]]:
    """Core driver; returns a list of output rows."""
    n_q, n_s = len(questions), len(skills)
    LOGGER.info("select_skills mode=%s n_questions=%d n_skills=%d top_k=%d",
                mode, n_q, n_s, top_k)

    # empty library -> empty selections (not an error)
    if n_s == 0:
        return [
            {
                "question_id": q.get("question_id"),
                "question": q.get("question", ""),
                "selected_skill_ids": [],
            }
            for q in questions
        ]

    skill_ids = [s.get("skill_id") for s in skills]
    skill_texts = [_skill_text(s) for s in skills]

    if mode == "ceil":
        q_texts = [q.get("question", "") for q in questions]
        if dry_run:
            all_emb = _encode_dry(q_texts + skill_texts, dim=64, seed=42)
        else:
            all_emb = _encode_real(q_texts + skill_texts, embedding_model)
        q_embs = all_emb[:n_q]
        skill_embs = all_emb[n_q:]

        out: List[Dict[str, Any]] = []
        for qi, q in enumerate(questions):
            idxs = _select_ceil(q_embs[qi], skill_embs, top_k=top_k)
            assert len(idxs) == len(set(idxs)), "CEIL must return unique indices"
            out.append({
                "question_id": q.get("question_id"),
                "question": q.get("question", ""),
                "selected_skill_ids": [skill_ids[i] for i in idxs],
            })
        return out

    if mode == "genicl":
        if not genicl_selector_path:
            raise ValueError("--genicl_selector_path is required when --mode genicl")
        if not os.path.exists(genicl_selector_path):
            raise FileNotFoundError(
                f"GenICL selector path not found: {genicl_selector_path}"
            )
        if dry_run:
            # dry-run still exercises fail-fast on bad path above; here we
            # mock scores with deterministic noise so top-K is well-defined.
            out = []
            for qi, q in enumerate(questions):
                rng = np.random.default_rng(abs(hash(q.get("question", ""))) & 0xFFFFFFFF)
                scores = rng.random(n_s)
                order = np.argsort(-scores)[: min(top_k, n_s)].tolist()
                out.append({
                    "question_id": q.get("question_id"),
                    "question": q.get("question", ""),
                    "selected_skill_ids": [skill_ids[i] for i in order],
                })
            return out
        selector = GenICLSelector(genicl_selector_path)
        out = []
        for q in questions:
            idxs = _select_genicl(selector, q.get("question", ""), skill_texts, top_k)
            out.append({
                "question_id": q.get("question_id"),
                "question": q.get("question", ""),
                "selected_skill_ids": [skill_ids[i] for i in idxs],
            })
        return out

    raise ValueError(f"unknown mode: {mode!r} (expected 'ceil' or 'genicl')")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="context_select.py",
        description=(
            "Select top-K skills per question via CEIL DPP (default) "
            "or a GenICL-style LM selector."
        ),
    )
    p.add_argument("--skill_library", required=True,
                   help="JSONL of skills (schema: skill_id, name, principle, when_to_apply, ...)")
    p.add_argument("--questions", required=True,
                   help="JSONL with {question_id, question, ...} per line")
    p.add_argument("--output", required=True,
                   help="Path to write JSONL of selections")
    p.add_argument("--mode", choices=["ceil", "genicl"], default="ceil",
                   help="Selection strategy (default: ceil)")
    p.add_argument("--top_k", type=int, default=3,
                   help="Number of skills to select per question (default: 3)")
    p.add_argument("--embedding_model",
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="sentence-transformers model (CEIL mode only)")
    p.add_argument("--genicl_selector_path", default=None,
                   help="Path to pre-trained GenICL selector (PEFT adapter dir "
                        "or full model dir). Required when --mode=genicl.")
    p.add_argument("--dry_run", action="store_true",
                   help="Use deterministic pseudo-embeddings / mock GenICL "
                        "scores; never loads sentence-transformers or heavy models.")
    p.add_argument("--log_level", default="INFO",
                   help="python logging level (DEBUG, INFO, WARNING, ...)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.mode == "genicl" and not args.dry_run and not args.genicl_selector_path:
        raise SystemExit("--genicl_selector_path is required when --mode=genicl")
    if (args.mode == "genicl" and args.genicl_selector_path
            and not os.path.exists(args.genicl_selector_path)):
        # fail fast even in dry-run: plan §E requires an explicit error
        raise SystemExit(
            f"GenICL selector path does not exist: {args.genicl_selector_path}"
        )

    skills = _read_jsonl(args.skill_library)
    questions = _read_jsonl(args.questions)
    LOGGER.info("loaded %d skills, %d questions", len(skills), len(questions))

    rows = select_skills(
        questions=questions,
        skills=skills,
        mode=args.mode,
        top_k=int(args.top_k),
        embedding_model=args.embedding_model,
        dry_run=bool(args.dry_run),
        genicl_selector_path=args.genicl_selector_path,
    )
    n = _write_jsonl(args.output, rows)
    LOGGER.info("wrote %d rows -> %s", n, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
