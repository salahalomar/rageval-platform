"""Prompt assembly, from versioned files on disk.

The prompt text is not inline in this module on purpose. A prompt change moves every
answer metric exactly the way a chunk-size change moves every retrieval metric, which
makes it an ablation arm -- and an arm has to be nameable. `prompt_version` goes into the
generation config, the config goes into the result record, and a committed file makes
"which prompt produced this number" answerable by `git show` rather than by memory.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from importlib import resources

from rag.retrieve.types import Candidate

logger = logging.getLogger(__name__)

# The exact sentence the model is told to produce when the context is insufficient, and
# the exact sentence the answer path recognises as a refusal. Defined once: if these two
# ever drift apart, the system refuses and then fails to notice that it refused.
INSUFFICIENT_EVIDENCE = "I don't have enough evidence in the indexed papers to answer that."


@lru_cache(maxsize=8)
def system_prompt(version: str) -> str:
    """Load the versioned system prompt shipped alongside this module."""
    try:
        return (
            resources.files("rag.generate.prompts")
            .joinpath(f"answer_{version}.md")
            .read_text(encoding="utf-8")
            .strip()
        )
    except FileNotFoundError as exc:
        raise ValueError(f"no prompt file for version {version!r}") from exc


def context_block(index: int, candidate: Candidate) -> str:
    """One numbered context block.

    The paper title and section path are in the header rather than left implicit: they
    are what let the model distinguish two chunks that discuss the same method in
    different papers, and they are what a reader needs to judge a citation.
    """
    header = f"[{index}] ({candidate.paper_title} — {candidate.section_path})"
    return f"{header}\n{candidate.content}"


def assemble(question: str, candidates: Sequence[Candidate]) -> str:
    """Build the user message: numbered context blocks, then the question.

    The question goes last. Models attend most reliably to the end of a long input, and
    burying the question above several hundred lines of context is a well-known way to
    have it half-answered.
    """
    blocks = "\n\n".join(
        context_block(index, candidate) for index, candidate in enumerate(candidates, start=1)
    )
    return f"Context blocks:\n\n{blocks}\n\nQuestion: {question}"


def correction_prompt(answer: str, uncited: Sequence[str]) -> str:
    """The second-attempt message naming the sentences that lacked a citation.

    Quotes the offending sentences rather than restating the rule. A model that produced
    an uncited sentence has already read the rule once; what it has not seen is which of
    its own sentences broke it.
    """
    listed = "\n".join(f"- {sentence}" for sentence in uncited)
    return (
        "Your previous answer contained factual sentences with no citation marker:\n\n"
        f"{listed}\n\n"
        "Rewrite the answer so every factual sentence ends with one or more markers like "
        "[1] or [2][5], citing only blocks from the context above. Change nothing else. "
        "If a sentence cannot be supported by the context, remove it.\n\n"
        f"Your previous answer was:\n\n{answer}"
    )
