"""Candidate generation for the golden set: sample, generate, filter, deduplicate.

The no-context filter is the step that makes an LLM-generated evaluation set defensible
rather than circular. Every generated question is asked again with **no retrieval context
at all**; anything the model answers correctly from parametric knowledge is discarded,
because such a question tests what the model already knows rather than whether the system
retrieved anything. The plan expects this to reject 30-40% of candidates, and the actual
rate is reported and belongs in the README.

Nothing here writes to `eval/golden/v1.jsonl`. That file is produced by human review in
`verify_cli.py` and by nothing else, because the whole value of "human-verified" is that
no automated step can put an item there.

Run with `--dry-run` first. It samples, stratifies and prices the run without making a
single API call.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from eval.costs import CallEstimate, estimate_run, estimate_tokens
from eval.sample import ChunkSample, load_candidates, stratified_sample
from eval.schema import GoldenCandidate, iter_ids, write_jsonl
from rag.config import GenerationConfig, RetrievalConfig
from rag.db import connect
from rag.generate.client import AnthropicClient, LLMClient
from rag.index.embed import Encoder, encoder_for

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES = 200
DEFAULT_OUTPUT = Path("eval/golden/candidates.jsonl")

# Two questions whose embeddings sit closer than this are treated as the same question.
# 0.9 comes from the plan. It is deliberately strict: dropping a genuine near-duplicate
# costs one item, while keeping two phrasings of one question silently doubles that
# question's weight in every metric.
DUPLICATE_COSINE = 0.90

# What the filter model must say to mean "I do not know this".
UNKNOWN_MARKER = "UNKNOWN"

# Rough per-call output sizes, for the dry-run projection only.
QUESTION_OUTPUT_TOKENS = 120
FILTER_OUTPUT_TOKENS = 80


@lru_cache(maxsize=4)
def prompt(name: str) -> str:
    """Load a versioned evaluation prompt."""
    return resources.files("eval.prompts").joinpath(f"{name}.md").read_text("utf-8").strip()


@dataclass(slots=True)
class GenerationReport:
    """What the run did, in the terms the protocol documentation has to state."""

    sampled: int = 0
    generated: int = 0
    malformed: int = 0
    rejected_by_filter: int = 0
    duplicates: int = 0
    kept: int = 0

    @property
    def filter_rejection_rate(self) -> float:
        """Share of generated questions the model could answer with no context at all."""
        return self.rejected_by_filter / self.generated if self.generated else 0.0

    def as_lines(self) -> list[str]:
        """The numbers the golden-set README must publish."""
        return [
            f"  chunks sampled         {self.sampled}",
            f"  questions generated    {self.generated}",
            f"  malformed responses    {self.malformed}",
            f"  rejected by no-context filter  {self.rejected_by_filter} "
            f"({self.filter_rejection_rate:.1%})",
            f"  near-duplicates removed        {self.duplicates}",
            f"  candidates kept        {self.kept}",
        ]


def project_cost(samples: list[ChunkSample], generation: GenerationConfig) -> object:
    """Price the run without making any call.

    Both stages are projected from the text that would actually be sent, so the estimate
    tracks the real corpus rather than a guessed average passage length.
    """
    question_system = prompt("generate_question_v1")
    filter_system = prompt("no_context_filter_v1")

    question_input = round(
        sum(estimate_tokens(question_system) + estimate_tokens(s.content) for s in samples)
        / max(1, len(samples))
    )
    # The filter sees only the question, which is why it is the cheap half.
    filter_input = estimate_tokens(filter_system) + 40

    return estimate_run(
        generation.model,
        [
            CallEstimate("generate question", len(samples), question_input, QUESTION_OUTPUT_TOKENS),
            CallEstimate("no-context filter", len(samples), filter_input, FILTER_OUTPUT_TOKENS),
        ],
    )


def generate_one(
    llm: LLMClient, sample: ChunkSample, generation: GenerationConfig
) -> dict[str, str] | None:
    """Ask for one question about one chunk. Returns None if the reply was unusable."""
    completion = llm.complete(prompt("generate_question_v1"), sample.content, generation)
    text = completion.text.strip()
    # Models wrap JSON in fences regardless of instructions.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("chunk %d: reply was not JSON", sample.chunk_id)
        return None
    if not isinstance(parsed, dict) or not parsed.get("question"):
        logger.warning("chunk %d: reply had no question", sample.chunk_id)
        return None
    return {str(key): str(value) for key, value in parsed.items()}


def survives_no_context(
    llm: LLMClient, question: str, generation: GenerationConfig
) -> tuple[bool, str]:
    """Ask the question with no context. True means it is worth keeping.

    A question the model answers from its own knowledge does not test retrieval, so it is
    discarded. The model's parametric answer is returned alongside the verdict and stored
    on the candidate, so a reviewer can audit the judgement rather than trust it.
    """
    completion = llm.complete(prompt("no_context_filter_v1"), question, generation)
    answer = completion.text.strip()
    return (UNKNOWN_MARKER in answer.upper(), answer)


def deduplicate(
    candidates: list[GoldenCandidate], encoder: Encoder, threshold: float = DUPLICATE_COSINE
) -> tuple[list[GoldenCandidate], int]:
    """Drop questions whose embeddings are near-identical to an earlier one.

    Uses the same local embedding model as retrieval, so deduplication costs nothing and
    needs no network.
    """
    if not candidates:
        return [], 0

    vectors = encoder.encode_documents([c.question for c in candidates])
    kept: list[GoldenCandidate] = []
    kept_vectors: list[list[float]] = []
    dropped = 0

    for candidate, vector in zip(candidates, vectors, strict=True):
        similarity = max(
            (sum(a * b for a, b in zip(vector, other, strict=True)) for other in kept_vectors),
            default=0.0,
        )
        if similarity >= threshold:
            dropped += 1
            logger.info("dropping %s as a near-duplicate (cosine %.3f)", candidate.id, similarity)
            continue
        kept.append(candidate)
        kept_vectors.append(vector)

    return kept, dropped


def build(
    count: int,
    output: Path,
    *,
    dry_run: bool,
    config: RetrievalConfig,
    generation: GenerationConfig,
    llm: LLMClient | None = None,
    encoder: Encoder | None = None,
) -> GenerationReport:
    """Sample, generate, filter and deduplicate candidates."""
    report = GenerationReport()

    with connect() as conn:
        chunks = load_candidates(conn, config)
    if not chunks:
        raise RuntimeError("no chunks in the corpus; run `rag ingest` and `rag embed` first")

    samples, stratification = stratified_sample(chunks, count)
    report.sampled = len(samples)

    print("stratification")
    for line in stratification.as_lines():
        print(line)
    print()
    print("projected cost" + ("  (DRY RUN — nothing will be called)" if dry_run else ""))
    estimate = project_cost(samples, generation)
    for line in estimate.as_lines():  # type: ignore[attr-defined]
        print(line)

    if dry_run:
        print()
        print("  dry run: no API calls were made, no candidates written")
        return report

    llm = llm or AnthropicClient()
    encoder = encoder or encoder_for(config.embedding_model)
    ids = iter_ids("g")
    candidates: list[GoldenCandidate] = []

    for index, sample in enumerate(samples, start=1):
        parsed = generate_one(llm, sample, generation)
        if parsed is None:
            report.malformed += 1
            continue
        report.generated += 1

        keep, parametric = survives_no_context(llm, parsed["question"], generation)
        if not keep:
            report.rejected_by_filter += 1
            logger.info("rejected: answerable without retrieval — %s", parsed["question"][:70])
            continue

        candidates.append(
            GoldenCandidate(
                id=next(ids),
                question=parsed["question"],
                expected_answer=parsed.get("expected_answer", ""),
                relevant_chunk_ids=[sample.chunk_id],
                difficulty=parsed.get("difficulty", "medium"),  # type: ignore[arg-type]
                type=parsed.get("type", "factual"),  # type: ignore[arg-type]
                answerable=True,
                provenance="llm_generated",
                source_chunk_id=sample.chunk_id,
                source_section_type=sample.section_type,
                source_paper_id=sample.paper_id,
                no_context_answer=parametric,
                no_context_verdict="kept",
                notes=f"survived no-context filter; section={sample.section_type}",
            )
        )
        if index % 25 == 0:
            logger.info("processed %d/%d", index, len(samples))

    candidates, report.duplicates = deduplicate(candidates, encoder)
    report.kept = write_jsonl(output, candidates)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="sample and price the run without calling the API or writing anything",
    )
    parser.add_argument("--model", default=GenerationConfig().model)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stderr)

    report = build(
        args.count,
        args.output,
        dry_run=args.dry_run,
        config=RetrievalConfig(),
        generation=GenerationConfig(model=args.model),
    )
    if not args.dry_run:
        print()
        print("generation report:")
        for line in report.as_lines():
            print(line)
        print()
        print(f"  wrote {report.kept} candidates to {args.output}")
        print("  NOTHING is verified yet — run `python -m eval.verify_cli` to review them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
