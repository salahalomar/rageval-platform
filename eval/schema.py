"""The golden set record, and reading and writing it as JSONL.

The golden set is the ground truth every published number is measured against, so its
records are validated rather than trusted. A malformed item -- a missing chunk id, a
`verified_by` that was never filled in -- would not crash anything; it would quietly
produce a metric computed over fewer items than the README claims.

JSONL rather than one JSON document because the file is reviewed by a human, edited by
hand, and diffed in code review. A one-line-per-item format makes a changed question a
one-line diff instead of a reformatted blob.
"""

import json
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# What kind of question this is. The four hard types come from the plan and exist to stop
# the set being all easy abstract lookups: each one probes a different failure.
QuestionType = Literal[
    "factual",  # answerable from a single chunk
    "multi_hop",  # requires combining two chunks, possibly across papers
    "numeric",  # the answer is a figure in a results table
    "unanswerable",  # plausible but genuinely absent -- correct behaviour is refusal
    "distractor",  # wording lexically matches the wrong paper
]

Difficulty = Literal["easy", "medium", "hard"]

# How the item came to exist. Recorded on every item because the README has to state the
# provenance mix honestly, and "human-verified" means nothing if the file cannot say
# which items a human actually looked at.
Provenance = Literal[
    "llm_generated",  # drafted by a model from a chunk, then filtered
    "llm_drafted_human_written",  # drafted as a starting point, intended to be rewritten
    "human_written",  # written from scratch
]


class GoldenItem(BaseModel):
    """One verified question in the golden set.

    `extra="forbid"` because a typo in a field name would otherwise be silently accepted
    and then silently ignored by every metric that reads the file.
    """

    model_config = {"extra": "forbid"}

    id: str = Field(pattern=r"^[a-z]+-\d{3}$")
    question: str = Field(min_length=10)
    expected_answer: str
    relevant_chunk_ids: list[int]
    difficulty: Difficulty
    type: QuestionType
    answerable: bool
    provenance: Provenance
    verified_by: str | None = None
    verified_at: date | None = None
    notes: str = ""

    def model_post_init(self, _: object) -> None:
        """Enforce the two invariants that make an item meaningful.

        An answerable question with no relevant chunks cannot score recall against
        anything; an unanswerable one with relevant chunks contradicts itself.
        """
        if self.answerable and not self.relevant_chunk_ids:
            raise ValueError(f"{self.id}: answerable items need at least one relevant chunk")
        if not self.answerable and self.relevant_chunk_ids:
            raise ValueError(f"{self.id}: unanswerable items must not name relevant chunks")

    @property
    def verified(self) -> bool:
        """Whether a human has actually confirmed this item."""
        return self.verified_by is not None and self.verified_at is not None


class GoldenCandidate(BaseModel):
    """A generated item before human verification.

    A superset of `GoldenItem`, carrying the evidence for why it survived generation: the
    chunk it came from, and what the model said when asked the question with no context
    at all. That second field is what makes the no-context filter auditable rather than a
    claim in a README -- a reviewer can see the parametric answer that was judged.
    """

    model_config = {"extra": "forbid"}

    id: str
    question: str
    expected_answer: str
    relevant_chunk_ids: list[int]
    difficulty: Difficulty
    type: QuestionType
    answerable: bool
    provenance: Provenance
    notes: str = ""

    source_chunk_id: int
    source_section_type: str
    source_paper_id: str
    no_context_answer: str = ""
    no_context_verdict: Literal["kept", "rejected", "not_run"] = "not_run"
    duplicate_of: str | None = None

    def to_item(self, verified_by: str, verified_at: date) -> GoldenItem:
        """Promote a reviewed candidate into a golden item, stamped with the reviewer."""
        return GoldenItem(
            id=self.id,
            question=self.question,
            expected_answer=self.expected_answer,
            relevant_chunk_ids=self.relevant_chunk_ids,
            difficulty=self.difficulty,
            type=self.type,
            answerable=self.answerable,
            provenance=self.provenance,
            verified_by=verified_by,
            verified_at=verified_at,
            notes=self.notes,
        )


def read_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    """Parse a JSONL file into validated records, naming the line that failed.

    The line number matters: a golden file is hand-edited, and "validation error" without
    a line is a bad experience at 11pm during a verification pass.
    """
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            records.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    return records


def read_items(path: Path) -> list[GoldenItem]:
    """Read a verified golden set."""
    return [item for item in read_jsonl(path, GoldenItem) if isinstance(item, GoldenItem)]


def read_candidates(path: Path) -> list[GoldenCandidate]:
    """Read a candidate set awaiting verification."""
    return [c for c in read_jsonl(path, GoldenCandidate) if isinstance(c, GoldenCandidate)]


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Write records one per line, sorted keys, returning how many were written.

    Sorted keys so that re-writing a file a reviewer has touched produces a diff of the
    lines that changed rather than of every line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(json.loads(record.model_dump_json()), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def iter_ids(prefix: str, start: int = 1) -> Iterator[str]:
    """Generate stable item ids: `g-001`, `g-002`, ..."""
    number = start
    while True:
        yield f"{prefix}-{number:03d}"
        number += 1
