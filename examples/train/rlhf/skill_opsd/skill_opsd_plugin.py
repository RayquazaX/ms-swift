"""SkillRL-OPSD dataset plugin for locally-prepared JSONL files.

Reads a Task 3.3 ``train_with_context.jsonl`` and forwards the pre-built
``teacher_prompt`` field to the GKD trainer untouched.

Unlike :mod:`opsd_plugin` (which synthesises ``teacher_prompt`` by
concatenating ``problem`` and ``solution`` pulled from
``open-r1/OpenThoughts-114k-math``), this plugin assumes the JSONL already
carries a fully-assembled layered teacher prompt (section: library-skills +
OPSD transition + restated question) produced by
``context_builder.py``. The preprocessor therefore acts as a *pass-through*
that:

* normalises the row into the ms-swift standard ``messages`` schema,
* preserves ``teacher_prompt`` verbatim,
* drops malformed rows defensively.

Input schema (each JSONL line, per Task 3.3 spec)::

    {
      "messages": [
        {"role": "user", "content": "<question>"},
        {"role": "assistant", "content": "<reference solution>"}
      ],
      "teacher_prompt": "<layered markdown + OPSD transition + question>",
      "question_id": "<q_id>"
    }

Output schema (per row, consumed by ``GKDTrainer._build_opsd_teacher_data``)::

    {
      "messages":       <unchanged>,
      "teacher_prompt": <unchanged, verbatim>
    }

Usage::

    swift rlhf \\
        --rlhf_type gkd \\
        --external_plugins examples/train/rlhf/skill_opsd/skill_opsd_plugin.py \\
        --dataset /abs/path/to/train_with_context.jsonl \\
        ...

Registered dataset_path suffixes (the matcher uses basename):

    * ``train_with_context.jsonl``  (Task 3.3 default)
    * ``skill_opsd_train.jsonl``    (alias)
    * ``skill_opsd_val.jsonl``      (alias)

If you name your JSONL something else, either rename it to one of the above
or export ``SKILL_OPSD_JSONL_BASENAMES='my.jsonl,other.jsonl'`` before
launching ``swift rlhf`` to register extra basenames.

References:
    * Template: examples/train/rlhf/opsd/opsd_plugin.py
    * GKD trainer consumer: swift/rlhf_trainers/gkd_trainer.py
      (see ``_build_opsd_teacher_data`` at L467+)
    * Dataset registry: swift/dataset/register.py
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset
from swift.utils import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# Default JSONL basenames we will register as dataset_path entries.
# The ms-swift dataset-matcher (see
# swift/dataset/dataset_syntax.py ``_get_matched_dataset_meta``) falls back
# to matching by *filename suffix* when the literal absolute path is not
# already in DATASET_MAPPING, so registering by basename is the robust way
# to let users pass arbitrary absolute paths via ``--dataset``.
_DEFAULT_BASENAMES = (
    'train_with_context.jsonl',
    'skill_opsd_train.jsonl',
    'skill_opsd_val.jsonl',
)

# Canonical ms-swift "dataset_name" alias (rarely used directly; mostly
# useful for logging / documentation).
DATASET_NAME = 'skill_opsd_local'


class SkillOPSDPassthroughPreprocessor(RowPreprocessor):
    """Preprocessor that passes ``teacher_prompt`` through unchanged.

    The ``messages`` field is validated (must contain at least one user
    turn; the assistant turn is optional for pure on-policy training but
    kept when present so that SFT-style ``lmbda < 1`` blending still has a
    reference completion). Rows missing ``teacher_prompt`` are dropped
    with a warning since OPSD cannot proceed without privileged context.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        teacher_prompt = row.get('teacher_prompt')
        if not teacher_prompt or not isinstance(teacher_prompt, str):
            # Dropping silently would hide upstream bugs in context_builder.
            if self._traceback_counter < self.traceback_limit:
                logger.warning(
                    'skill_opsd row missing/empty `teacher_prompt`; dropping. '
                    'question_id=%r', row.get('question_id'))
                self._traceback_counter += 1
            return None

        messages = row.get('messages')
        if not messages or not isinstance(messages, list):
            if self._traceback_counter < self.traceback_limit:
                logger.warning(
                    'skill_opsd row missing/empty `messages`; dropping. '
                    'question_id=%r', row.get('question_id'))
                self._traceback_counter += 1
            return None

        # Defensive copy + role validation. The GKD trainer's
        # ``_build_opsd_teacher_data`` iterates messages in reverse looking
        # for a user turn to swap in the teacher_prompt, so we need at
        # least one user message.
        normalised: List[Dict[str, str]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get('role')
            content = m.get('content')
            if role not in {'system', 'user', 'assistant'} or content is None:
                continue
            normalised.append({'role': role, 'content': content})

        if not any(m['role'] == 'user' for m in normalised):
            if self._traceback_counter < self.traceback_limit:
                logger.warning(
                    'skill_opsd row has no user turn; dropping. '
                    'question_id=%r', row.get('question_id'))
                self._traceback_counter += 1
            return None

        return {
            'messages': normalised,
            'teacher_prompt': teacher_prompt,
        }


def _collect_basenames() -> List[str]:
    """Resolve the set of JSONL basenames this plugin should claim.

    Merges :data:`_DEFAULT_BASENAMES` with any comma-separated extras from
    the ``SKILL_OPSD_JSONL_BASENAMES`` env var (lets the user register
    custom filenames without editing this file).
    """
    basenames = list(_DEFAULT_BASENAMES)
    extra = os.environ.get('SKILL_OPSD_JSONL_BASENAMES', '').strip()
    if extra:
        for name in extra.split(','):
            name = name.strip()
            if name and name not in basenames:
                basenames.append(name)
    return basenames


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
# One DatasetMeta per basename so that ms-swift's suffix-matching resolver
# (``DatasetSyntax._get_matched_dataset_meta``) returns our preprocessor for
# any absolute path whose basename matches. All entries share the same
# preprocessor instance; registering them separately is cheap and keeps the
# public API simple.
_preprocessor = SkillOPSDPassthroughPreprocessor()

for _basename in _collect_basenames():
    # Use ``dataset_name`` + ``dataset_path`` so (a) the registry key is the
    # explicit dataset_name (avoids collision with opsd_plugin entries keyed
    # by ms/hf ids) and (b) suffix-matching still finds us via dataset_path.
    _name_alias = f'{DATASET_NAME}::{_basename}'
    try:
        register_dataset(
            DatasetMeta(
                dataset_name=_name_alias,
                dataset_path=_basename,  # basename-only; matched by suffix
                preprocess_func=_preprocessor,
                tags=['math', 'opsd', 'skill', 'local-jsonl'],
            ),
            exist_ok=True,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            'skill_opsd plugin: failed to register `%s` (%s)', _name_alias, exc)

logger.info(
    'skill_opsd plugin registered. Accepted JSONL basenames: %s. '
    'Override with $SKILL_OPSD_JSONL_BASENAMES=\'a.jsonl,b.jsonl\'.',
    _collect_basenames(),
)
