"""Stratified sampling of candidate chunks.

Sampling uniformly at random would produce a golden set shaped like the corpus, and the
corpus is 21% introductions. A set dominated by introductions measures whether the system
can find a paper's topic sentence, which every configuration can, and would report a
flattering number that moves for no arm in the ablation. Stratifying across section types
forces method and results chunks -- where the specific, checkable claims live -- into the
set in proportions a reviewer chooses rather than proportions the corpus happened to have.

Also stratified across papers: a set drawing ten questions from one paper measures that
paper, not the corpus.
"""

import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import psycopg

from rag.config import RetrievalConfig

logger = logging.getLogger(__name__)

# Coarse section types, matched against `section_path` in order. First match wins, so the
# more specific patterns come first: "Experimental Setup" is method, not results, and
# "Related Work" is introduction, not method.
#
# These are heuristics over free-text headings written by hundreds of different authors,
# and they are imperfect by construction. The stratification report prints the resulting
# distribution so the imperfection is visible rather than assumed away.
SECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("abstract", r"\babstract\b|\bfrontmatter\b"),
    ("limitations", r"\blimitat|\bfuture work|\bbroader impact|\bethic|\bthreats? to valid"),
    ("introduction", r"\bintroduc|\bbackground\b|\brelated work\b|\bmotivat|\bprelimin"),
    (
        "method",
        r"\bmethod|\bapproach\b|\barchitect|\balgorithm|\bframework\b|\bmodel\b"
        r"|\bimplement|\btraining\b|\bsetup\b|\bdesign\b|\bdata\b",
    ),
    (
        "results",
        r"\bresult|\bexperim|\bevaluat|\banalys|\bfinding|\bablation|\bbenchmark"
        r"|\bdiscussion\b|\bcomparison\b",
    ),
    ("conclusion", r"\bconclusion|\bsummary\b"),
)

# The literal section the Phase 1 chunker falls back to when heading detection fails.
# Kept as its own stratum rather than folded into "other": chunks with no section
# structure are a known quality signal, and a golden set drawn from them would be
# measuring the papers the parser handled worst.
UNSTRUCTURED_SECTION = "Body"

# Chunks shorter than this are stray headers, equation fragments or table dumps. They
# make poor questions -- there is often nothing in them to ask about.
MIN_TOKENS_FOR_CANDIDATE = 120

DEFAULT_SEED = 20260824


def classify_section(section_path: str) -> str:
    """Map a free-text section path onto a coarse type."""
    if section_path.strip() == UNSTRUCTURED_SECTION:
        return "unstructured"
    lowered = section_path.lower()
    for name, pattern in SECTION_PATTERNS:
        if re.search(pattern, lowered):
            return name
    return "other"


@dataclass(frozen=True, slots=True)
class ChunkSample:
    """One candidate chunk, with what is needed to write a question about it."""

    chunk_id: int
    paper_id: str
    paper_title: str
    section_path: str
    section_type: str
    token_count: int
    content: str


@dataclass(slots=True)
class Stratification:
    """What the sample actually contains, for the report the protocol requires."""

    requested: int
    returned: int
    by_section_type: Counter[str] = field(default_factory=Counter)
    papers: int = 0
    max_per_paper: int = 0

    def as_lines(self) -> list[str]:
        """Human-readable report."""
        lines = [
            f"  requested              {self.requested}",
            f"  returned               {self.returned}",
            f"  distinct papers        {self.papers}",
            f"  max chunks per paper   {self.max_per_paper}",
            "  by section type:",
        ]
        total = sum(self.by_section_type.values()) or 1
        for name, count in self.by_section_type.most_common():
            lines.append(f"    {name:<16} {count:>4}  {count / total:>5.1%}")
        return lines


def load_candidates(
    conn: psycopg.Connection, config: RetrievalConfig, min_tokens: int = MIN_TOKENS_FOR_CANDIDATE
) -> list[ChunkSample]:
    """Every chunk eligible to become a question, for the active chunking."""
    rows = conn.execute(
        """
        SELECT c.id, c.paper_id, p.title, c.section_path, c.token_count, c.content
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        WHERE c.chunk_config_sha256 = %s AND c.token_count >= %s
        ORDER BY c.id
        """,
        (config.chunking_sha256(), min_tokens),
    ).fetchall()
    return [
        ChunkSample(
            chunk_id=int(row[0]),
            paper_id=str(row[1]),
            paper_title=str(row[2]),
            section_path=str(row[3]),
            section_type=classify_section(str(row[3])),
            token_count=int(row[4]),
            content=str(row[5]),
        )
        for row in rows
    ]


def stratified_sample(
    chunks: list[ChunkSample],
    count: int,
    *,
    weights: dict[str, float] | None = None,
    max_per_paper: int = 3,
    seed: int = DEFAULT_SEED,
) -> tuple[list[ChunkSample], Stratification]:
    """Draw `count` chunks, spread across section types and across papers.

    `weights` are target shares per section type. The default deliberately over-samples
    method and results relative to the corpus, because that is where specific checkable
    claims live, and under-samples introductions, which the corpus has most of and which
    make the easiest possible questions.

    `max_per_paper` stops the set becoming a study of whichever three papers happened to
    be sampled first.

    Seeded, so the same corpus yields the same candidates. An unseeded sample makes it
    impossible to tell whether a protocol change altered the set or the dice did.
    """
    weights = weights or {
        "method": 0.30,
        "results": 0.30,
        "abstract": 0.15,
        "limitations": 0.10,
        "introduction": 0.10,
        "other": 0.05,
    }
    rng = random.Random(seed)

    by_type: dict[str, list[ChunkSample]] = defaultdict(list)
    for chunk in chunks:
        by_type[chunk.section_type].append(chunk)
    for bucket in by_type.values():
        rng.shuffle(bucket)

    selected: list[ChunkSample] = []
    per_paper: Counter[str] = Counter()

    def take(bucket: list[ChunkSample], wanted: int) -> None:
        for chunk in bucket:
            if wanted <= 0:
                return
            if per_paper[chunk.paper_id] >= max_per_paper:
                continue
            if any(existing.chunk_id == chunk.chunk_id for existing in selected):
                continue
            selected.append(chunk)
            per_paper[chunk.paper_id] += 1
            wanted -= 1

    for section_type, share in sorted(weights.items(), key=lambda pair: -pair[1]):
        take(by_type.get(section_type, []), round(count * share))

    # Top up from anything remaining if a stratum could not fill its quota -- a corpus
    # with few limitations sections should still yield the requested number of
    # candidates, with the shortfall visible in the report rather than silently dropped.
    if len(selected) < count:
        leftovers = [c for bucket in by_type.values() for c in bucket]
        rng.shuffle(leftovers)
        take(leftovers, count - len(selected))

    selected.sort(key=lambda chunk: chunk.chunk_id)
    stratification = Stratification(
        requested=count,
        returned=len(selected),
        by_section_type=Counter(chunk.section_type for chunk in selected),
        papers=len({chunk.paper_id for chunk in selected}),
        max_per_paper=max(per_paper.values()) if per_paper else 0,
    )
    logger.info("sampled %d chunks across %d papers", len(selected), stratification.papers)
    return selected, stratification
