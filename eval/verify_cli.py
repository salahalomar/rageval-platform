"""Terminal review of generated candidates. The only thing that writes the golden set.

No automated step may promote a candidate into `v1.jsonl`, because "human-verified" means
nothing if a script can do it. Every item in the golden set passed through a keypress
here and carries the reviewer's name and the date they pressed it.

Resumable by construction: progress is the output file itself. Re-running skips whatever
has already been decided, so an eighty-item pass can be done in four sittings without
losing anything or reviewing anything twice.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from eval.schema import GoldenCandidate, GoldenItem, read_candidates, read_items, write_jsonl
from rag.config import RetrievalConfig
from rag.db import connect

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES = Path("eval/golden/candidates.jsonl")
DEFAULT_OUTPUT = Path("eval/golden/v1.jsonl")
DEFAULT_REJECTS = Path("eval/golden/rejected.jsonl")

HELP = """
  a  accept as written          e  edit the question
  x  reject                     w  edit the expected answer
  s  skip for now               c  show the full source chunk
  q  save and quit
"""


@dataclass(slots=True)
class Session:
    """Where a review pass has got to."""

    accepted: int = 0
    rejected: int = 0
    edited: int = 0
    skipped: int = 0

    def as_lines(self) -> list[str]:
        """Summary printed on exit."""
        return [
            f"  accepted   {self.accepted}",
            f"  edited     {self.edited}",
            f"  rejected   {self.rejected}",
            f"  skipped    {self.skipped}",
        ]


def fetch_chunk_text(chunk_id: int) -> str:
    """The source passage a candidate was written from."""
    with connect() as conn:
        row = conn.execute(
            "SELECT p.title, c.section_path, c.content FROM chunks c "
            "JOIN papers p ON p.id = c.paper_id WHERE c.id = %s",
            (chunk_id,),
        ).fetchone()
    if row is None:
        return f"(chunk {chunk_id} not found — has the corpus been re-ingested?)"
    return f"{row[0]}\n[{row[1]}]\n\n{row[2]}"


def prompt_line(label: str, current: str) -> str:
    """Read a replacement value, keeping the current one if the reviewer just hits enter."""
    print(f"\n  current {label}:\n    {current}")
    replacement = input(f"  new {label} (enter to keep): ").strip()
    return replacement or current


def show(candidate: GoldenCandidate, position: int, total: int, chunk_preview: str) -> None:
    """Render one candidate for review."""
    print("\n" + "=" * 78)
    print(
        f"  {candidate.id}   ({position} of {total})   "
        f"type={candidate.type}  difficulty={candidate.difficulty}  "
        f"section={candidate.source_section_type}"
    )
    print("=" * 78)
    print(f"\n  QUESTION\n    {candidate.question}")
    print(f"\n  EXPECTED ANSWER\n    {candidate.expected_answer}")
    print(f"\n  GROUND TRUTH CHUNKS  {candidate.relevant_chunk_ids}")
    if candidate.no_context_answer:
        # Shown because the filter's judgement is the thing most worth auditing: if the
        # model plainly knew the answer and the filter kept it anyway, that is a bug in
        # the protocol, not a good question.
        print(f"\n  WITH NO CONTEXT, THE MODEL SAID\n    {candidate.no_context_answer[:300]}")
    print(f"\n  SOURCE PASSAGE (first 600 chars)\n    {chunk_preview[:600]}")
    print(HELP)


def review(
    candidates: list[GoldenCandidate],
    accepted: list[GoldenItem],
    rejected: list[GoldenCandidate],
    reviewer: str,
    today: date,
) -> Session:
    """Run the interactive loop over undecided candidates."""
    session = Session()
    decided = {item.id for item in accepted} | {c.id for c in rejected}
    outstanding = [c for c in candidates if c.id not in decided]

    if not outstanding:
        print("nothing left to review — every candidate has been decided")
        return session

    print(
        f"{len(outstanding)} candidates to review "
        f"({len(decided)} already decided in a previous session)"
    )

    for position, candidate in enumerate(outstanding, start=1):
        current = candidate
        while True:
            show(current, position, len(outstanding), fetch_chunk_text(current.source_chunk_id))
            choice = input("  > ").strip().lower()

            if choice == "a":
                accepted.append(current.to_item(reviewer, today))
                session.accepted += 1
                break
            if choice == "x":
                rejected.append(current)
                session.rejected += 1
                break
            if choice == "s":
                session.skipped += 1
                break
            if choice == "e":
                current = current.model_copy(
                    update={"question": prompt_line("question", current.question)}
                )
                session.edited += 1
                continue
            if choice == "w":
                current = current.model_copy(
                    update={"expected_answer": prompt_line("answer", current.expected_answer)}
                )
                session.edited += 1
                continue
            if choice == "c":
                print("\n" + fetch_chunk_text(current.source_chunk_id))
                input("\n  (enter to continue) ")
                continue
            if choice == "q":
                return session
            print("  unrecognised key")

    return session


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rejects", type=Path, default=DEFAULT_REJECTS)
    parser.add_argument(
        "--reviewer",
        required=True,
        help="who is reviewing; stamped onto every accepted item as verified_by",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)-7s %(message)s", stream=sys.stderr
    )

    if not args.candidates.exists():
        print(f"no candidates at {args.candidates}; run eval.generate_golden first")
        return 1

    candidates = read_candidates(args.candidates)
    accepted = read_items(args.output) if args.output.exists() else []
    rejected = read_candidates(args.rejects) if args.rejects.exists() else []

    try:
        session = review(candidates, accepted, rejected, args.reviewer, datetime.now(UTC).date())
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted — saving progress")
        session = Session()

    write_jsonl(args.output, accepted)
    write_jsonl(args.rejects, rejected)

    print("\nsession:")
    for line in session.as_lines():
        print(line)
    print(f"\n  {len(accepted)} verified items in {args.output}")
    print(f"  {len(rejected)} rejected in {args.rejects}")
    print(f"  corpus chunking: {RetrievalConfig().chunking_sha256()[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
