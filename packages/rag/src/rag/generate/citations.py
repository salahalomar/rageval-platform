"""Parsing `[n]` markers and binding them back to chunks.

This module is what turns "the model was asked to cite its sources" into "every claim in
this answer resolves to a chunk id and a page span". The difference matters: a model
instructed to cite will usually cite, and "usually" is not a property anyone can publish.
Here the answer is parsed, each marker is resolved against the blocks that were actually
sent, and any factual sentence without one is reported.

Three failure modes are handled explicitly because all three occur in practice:

* **Out-of-range markers.** The model writes [7] when five blocks were supplied. The
  marker is recorded as invalid rather than silently dropped, because an answer citing a
  block that does not exist is worse than an uncited one -- it looks sourced.
* **Uncited factual sentences.** Reported so the caller can retry once, and flagged on
  the answer if the retry also fails, rather than passed off as cited.
* **The refusal sentence.** Never requires a citation. Refusing is the correct behaviour
  when there is no evidence, and demanding a citation for it would make correct behaviour
  look like a violation.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag.generate.prompt import INSUFFICIENT_EVIDENCE
from rag.retrieve.types import Candidate

logger = logging.getLogger(__name__)

# Matches [1], and each bracket of [1][2] separately. Also matches the inner numbers of
# [1, 2] via the comma-separated group, because models write both forms freely.
MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Sentence terminator followed by whitespace. Markers sit *before* the full stop as often
# as after it, so splitting must not strand a marker into the following sentence.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A fragment shorter than this is a heading, a list label or a stray clause rather than a
# factual claim, and demanding a citation for it produces noise instead of signal.
MIN_WORDS_FOR_FACTUAL_SENTENCE = 5


@dataclass(frozen=True, slots=True)
class SentenceCitation:
    """One sentence of the answer, with the blocks it cited."""

    text: str
    markers: tuple[int, ...]
    chunk_ids: tuple[int, ...]

    @property
    def is_cited(self) -> bool:
        """Whether this sentence carries at least one resolvable citation."""
        return bool(self.chunk_ids)


@dataclass(frozen=True, slots=True)
class CitationBinding:
    """The result of binding an answer's markers to the context it was given."""

    sentences: tuple[SentenceCitation, ...]
    cited_chunk_ids: tuple[int, ...]
    invalid_markers: tuple[int, ...]
    uncited_sentences: tuple[str, ...]
    is_refusal: bool

    @property
    def uncited(self) -> bool:
        """Whether any factual sentence lacked a resolvable citation."""
        return bool(self.uncited_sentences) or bool(self.invalid_markers)


def parse_markers(text: str) -> list[int]:
    """Every citation number in `text`, in order of appearance, without deduplication.

    Handles `[1]`, `[1][2]` and `[1, 2]`, because models produce all three regardless of
    the form the prompt asks for.
    """
    numbers: list[int] = []
    for match in MARKER.finditer(text):
        numbers.extend(int(part) for part in match.group(1).split(","))
    return numbers


def strip_markers(text: str) -> str:
    """The answer with citation markers removed, for display or for a judge."""
    return re.sub(r"\s*" + MARKER.pattern, "", text).strip()


def split_sentences(answer: str) -> list[str]:
    """Split an answer into sentences, keeping markers attached to their own sentence."""
    return [part.strip() for part in SENTENCE_SPLIT.split(answer.strip()) if part.strip()]


def is_factual(sentence: str) -> bool:
    """Whether a sentence makes a claim that requires support.

    Deliberately a heuristic, and deliberately a conservative one: the refusal sentence
    and short fragments are exempt, everything else is not. A stricter rule would flag
    connectives as violations; a looser one would let real claims through uncited, which
    is the failure that matters.
    """
    if sentence.strip().rstrip(".") == INSUFFICIENT_EVIDENCE.rstrip("."):
        return False
    without_markers = strip_markers(sentence)
    return len(without_markers.split()) >= MIN_WORDS_FOR_FACTUAL_SENTENCE


def bind(answer: str, candidates: Sequence[Candidate]) -> CitationBinding:
    """Resolve an answer's markers against the blocks that were sent to the model.

    Markers are 1-based positions into `candidates`, matching the numbering in the
    assembled prompt, so binding is positional rather than by id -- the model never sees
    a chunk id and could not cite one.
    """
    refusal = INSUFFICIENT_EVIDENCE.rstrip(".") in answer
    sentences: list[SentenceCitation] = []
    invalid: list[int] = []
    cited: list[int] = []

    for sentence in split_sentences(answer):
        markers = parse_markers(sentence)
        chunk_ids: list[int] = []
        for marker in markers:
            if 1 <= marker <= len(candidates):
                chunk_id = candidates[marker - 1].chunk_id
                chunk_ids.append(chunk_id)
                if chunk_id not in cited:
                    cited.append(chunk_id)
            else:
                invalid.append(marker)
                logger.warning(
                    "answer cited block [%d] but only %d blocks were supplied",
                    marker,
                    len(candidates),
                )
        sentences.append(
            SentenceCitation(text=sentence, markers=tuple(markers), chunk_ids=tuple(chunk_ids))
        )

    uncited = tuple(
        sentence.text
        for sentence in sentences
        if is_factual(sentence.text) and not sentence.is_cited
    )

    return CitationBinding(
        sentences=tuple(sentences),
        cited_chunk_ids=tuple(cited),
        invalid_markers=tuple(invalid),
        uncited_sentences=uncited,
        is_refusal=refusal,
    )
