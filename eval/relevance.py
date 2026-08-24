"""Resolving ground truth across chunkings.

The golden set names relevant chunks by `chunk_id`. That is fine for every arm that
shares the chunking the set was written against, and useless for the one arm that does
not: the Phase 7 chunk-size sweep re-chunks the corpus at 256, 512 and 1024 tokens, and
every id in the golden set belongs to exactly one of those. Scoring the 256-token arm
against 512-token ids would report recall of zero for every question -- a catastrophic
result that is entirely an artefact of the identifier scheme.

The fix is to stop treating a chunk id as the ground truth and start treating it as a
*pointer* to one. A relevant chunk is really a claim about a region of a paper: "the
answer lives in this paper, between these character offsets". That claim survives
re-chunking. So each golden id is resolved once to `(paper_id, char_start, char_end)`,
and under any other chunking a chunk counts as relevant when it overlaps that region
enough to plausibly contain the answer.

The overlap threshold is a judgement call and is stated rather than hidden: a chunk
covering a third of the original span is counted, because a 256-token chunk cannot
contain a 512-token span and demanding full coverage would score the small-chunk arm at
zero for structural reasons rather than quality ones.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

logger = logging.getLogger(__name__)

# Fraction of the ground-truth span a chunk must cover to count as relevant. Applied to
# the *shorter* of the two, so a large chunk containing a small span and a small chunk
# inside a large span both qualify.
MIN_OVERLAP_FRACTION = 0.33


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """A region of a paper that answers a question, independent of any chunking."""

    paper_id: str
    char_start: int
    char_end: int

    @property
    def length(self) -> int:
        """Characters covered."""
        return max(0, self.char_end - self.char_start)

    def overlaps(self, other: "EvidenceSpan", threshold: float = MIN_OVERLAP_FRACTION) -> bool:
        """Whether `other` covers enough of this span to plausibly contain the answer."""
        if self.paper_id != other.paper_id:
            return False
        shared = min(self.char_end, other.char_end) - max(self.char_start, other.char_start)
        if shared <= 0:
            return False
        shorter = min(self.length, other.length) or 1
        return shared / shorter >= threshold


def resolve_spans(conn: psycopg.Connection, chunk_ids: Sequence[int]) -> list[EvidenceSpan]:
    """Turn golden-set chunk ids into the paper regions they point at.

    Done once per golden item, against whatever chunking those ids belong to, so the
    result is stable regardless of which arm is being evaluated.
    """
    if not chunk_ids:
        return []
    rows = conn.execute(
        "SELECT paper_id, char_start, char_end FROM chunks WHERE id = ANY(%s)",
        (list(chunk_ids),),
    ).fetchall()
    if len(rows) != len(set(chunk_ids)):
        logger.warning(
            "%d of %d golden chunk ids did not resolve; has the corpus been re-ingested?",
            len(set(chunk_ids)) - len(rows),
            len(set(chunk_ids)),
        )
    return [EvidenceSpan(str(row[0]), int(row[1]), int(row[2])) for row in rows]


def relevant_ids_for_chunking(
    conn: psycopg.Connection,
    spans: Sequence[EvidenceSpan],
    chunk_config_sha256: str,
    threshold: float = MIN_OVERLAP_FRACTION,
) -> list[int]:
    """Which chunks of a given chunking count as relevant to these evidence spans.

    Returns ids in ascending order so that two runs produce identical ground truth and
    therefore identical metrics -- the determinism requirement reaches all the way down
    here, not just into retrieval.
    """
    if not spans:
        return []

    papers = sorted({span.paper_id for span in spans})
    rows = conn.execute(
        """
        SELECT id, paper_id, char_start, char_end
        FROM chunks
        WHERE chunk_config_sha256 = %s AND paper_id = ANY(%s)
        ORDER BY id
        """,
        (chunk_config_sha256, papers),
    ).fetchall()

    relevant = [
        int(row[0])
        for row in rows
        if any(
            span.overlaps(EvidenceSpan(str(row[1]), int(row[2]), int(row[3])), threshold)
            for span in spans
        )
    ]
    logger.debug("%d spans resolved to %d relevant chunks", len(spans), len(relevant))
    return relevant
