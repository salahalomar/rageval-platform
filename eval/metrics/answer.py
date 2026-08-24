"""Answer-quality metrics, judged by a model.

Everything here costs money, which is why it is separated from the retrieval metrics: the
retrieval half of every ablation arm runs free and offline, and the judge is opt-in.

**The judge shares a model family with the generator, and that is a bias, not a detail.**
A model asked to grade output from its own family scores it higher than an independent
judge would — self-preference bias is well documented. This module does not fix that; it
makes it measurable. `eval.judge_agreement` compares the judge against hand labels and
reports Cohen's kappa, and until that number exists every figure produced here should be
read as provisional. Reporting faithfulness without reporting kappa is reporting an
unvalidated instrument.

Judgements are cached on `(item_id, answer_hash, judge_model, prompt_version)`. Re-running
an evaluation after changing a retrieval parameter re-judges only the answers that
actually changed, which is what makes a nightly ablation affordable.
"""

import hashlib
import json
import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from eval.costs import CallEstimate, RunEstimate, estimate_run, estimate_tokens
from rag.config import GenerationConfig
from rag.generate.answer import Answer
from rag.generate.client import LLMClient
from rag.generate.prompt import context_block
from rag.retrieve.types import Candidate

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path("eval/results/.judge_cache.sqlite")
JUDGE_PROMPT_VERSION = "v1"

# Projected output sizes per judge call, for the dry-run estimate only.
FAITHFULNESS_OUTPUT_TOKENS = 300
CORRECTNESS_OUTPUT_TOKENS = 80
CITATION_OUTPUT_TOKENS = 200


@lru_cache(maxsize=8)
def judge_prompt(name: str) -> str:
    """Load a versioned judge prompt."""
    return resources.files("eval.prompts").joinpath(f"{name}.md").read_text("utf-8").strip()


def answer_hash(text: str) -> str:
    """Digest of an answer, used as part of the judgement cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    """What a judge concluded about one answer."""

    faithfulness: float | None = None
    correctness: float | None = None
    citation_accuracy: float | None = None
    claims_total: int = 0
    claims_supported: int = 0
    citations_total: int = 0
    citations_correct: int = 0

    def as_row(self) -> dict[str, float]:
        """Flattened for a results table, omitting anything not measured."""
        row: dict[str, float] = {}
        if self.faithfulness is not None:
            row["faithfulness"] = self.faithfulness
        if self.correctness is not None:
            row["correctness"] = self.correctness
        if self.citation_accuracy is not None:
            row["citation_accuracy"] = self.citation_accuracy
        return row


class JudgementCache:
    """Persistent cache keyed on what actually determines a judgement.

    SQLite rather than a JSON file because a nightly ablation writes thousands of rows and
    re-reading and rewriting a growing JSON document on every one of them is the kind of
    thing that turns a ten-minute job into an hour.
    """

    def __init__(self, path: Path = DEFAULT_CACHE) -> None:
        """Open, creating the file and schema if absent."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS judgements (
                item_id        TEXT NOT NULL,
                answer_hash    TEXT NOT NULL,
                judge_model    TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                metric         TEXT NOT NULL,
                payload        TEXT NOT NULL,
                PRIMARY KEY (item_id, answer_hash, judge_model, prompt_version, metric)
            )
            """
        )
        self._conn.commit()

    def get(self, key: tuple[str, str, str, str, str]) -> dict[str, object] | None:
        """A cached judgement, or None."""
        row = self._conn.execute(
            "SELECT payload FROM judgements WHERE item_id=? AND answer_hash=? "
            "AND judge_model=? AND prompt_version=? AND metric=?",
            key,
        ).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def put(self, key: tuple[str, str, str, str, str], payload: dict[str, object]) -> None:
        """Store a judgement."""
        self._conn.execute(
            "INSERT OR REPLACE INTO judgements VALUES (?, ?, ?, ?, ?, ?)",
            (*key, json.dumps(payload, sort_keys=True)),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()


def _context(candidates: Sequence[Candidate]) -> str:
    return "\n\n".join(
        context_block(index, candidate) for index, candidate in enumerate(candidates, start=1)
    )


def _parse_json(text: str) -> dict[str, object] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("judge reply was not JSON")
        return None
    return dict(parsed) if isinstance(parsed, dict) else None


def judge_faithfulness(
    llm: LLMClient, answer: Answer, generation: GenerationConfig
) -> tuple[float | None, int, int]:
    """Share of the answer's atomic claims that the retrieved context supports."""
    payload = _parse_json(
        llm.complete(
            judge_prompt("judge_faithfulness_v1"),
            f"Context blocks:\n\n{_context(answer.candidates)}\n\nAnswer:\n{answer.text}",
            generation,
        ).text
    )
    claims = payload.get("claims") if payload else None
    if not isinstance(claims, list) or not claims:
        return (None, 0, 0)
    supported = sum(
        1 for claim in claims if isinstance(claim, dict) and claim.get("verdict") == "supported"
    )
    return (supported / len(claims), len(claims), supported)


def judge_correctness(
    llm: LLMClient, answer: Answer, expected: str, generation: GenerationConfig
) -> float | None:
    """Score the answer against the reference on the 1-5 rubric."""
    payload = _parse_json(
        llm.complete(
            judge_prompt("judge_correctness_v1"),
            f"Reference answer:\n{expected}\n\nGenerated answer:\n{answer.text}",
            generation,
        ).text
    )
    score = payload.get("score") if payload else None
    return float(score) if isinstance(score, int | float) else None


def judge_citations(
    llm: LLMClient, answer: Answer, generation: GenerationConfig
) -> tuple[float | None, int, int]:
    """Share of citations whose cited block actually supports its sentence."""
    if answer.binding is None:
        return (None, 0, 0)
    cited = [s for s in answer.binding.sentences if s.markers]
    if not cited:
        return (None, 0, 0)

    listed = "\n".join(
        f"{index}. {sentence.text}  (cited: {list(sentence.markers)})"
        for index, sentence in enumerate(cited, start=1)
    )
    payload = _parse_json(
        llm.complete(
            judge_prompt("judge_citation_v1"),
            f"Context blocks:\n\n{_context(answer.candidates)}\n\nSentences:\n{listed}",
            generation,
        ).text
    )
    verdicts = payload.get("sentences") if payload else None
    if not isinstance(verdicts, list) or not verdicts:
        return (None, 0, 0)
    correct = sum(1 for v in verdicts if isinstance(v, dict) and v.get("verdict") == "correct")
    return (correct / len(verdicts), len(verdicts), correct)


def estimate_judging(
    answers: Sequence[Answer], expected: Sequence[str], generation: GenerationConfig
) -> RunEstimate:
    """Price a judging run without making any call.

    Projected from the text that would actually be sent, so the figure tracks the real
    answers rather than an assumed average.
    """
    faithfulness_in = correctness_in = citation_in = 0
    for answer, reference in zip(answers, expected, strict=True):
        context_tokens = estimate_tokens(_context(answer.candidates))
        answer_tokens = estimate_tokens(answer.text)
        faithfulness_in += (
            estimate_tokens(judge_prompt("judge_faithfulness_v1")) + context_tokens + answer_tokens
        )
        correctness_in += (
            estimate_tokens(judge_prompt("judge_correctness_v1"))
            + answer_tokens
            + estimate_tokens(reference)
        )
        citation_in += (
            estimate_tokens(judge_prompt("judge_citation_v1")) + context_tokens + answer_tokens
        )

    count = max(1, len(answers))
    return estimate_run(
        generation.model,
        [
            CallEstimate(
                "judge faithfulness",
                len(answers),
                faithfulness_in // count,
                FAITHFULNESS_OUTPUT_TOKENS,
            ),
            CallEstimate(
                "judge correctness",
                len(answers),
                correctness_in // count,
                CORRECTNESS_OUTPUT_TOKENS,
            ),
            CallEstimate(
                "judge citations", len(answers), citation_in // count, CITATION_OUTPUT_TOKENS
            ),
        ],
    )
