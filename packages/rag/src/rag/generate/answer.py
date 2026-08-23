"""End to end: retrieve, check the floor, generate, bind citations.

The order of the first two steps is the point of this module. Refusal happens *before*
the model is called, not after it produces something and a filter rejects it -- so an
out-of-corpus question costs zero tokens and zero dollars. That is a design property, and
`Answer.usage.calls == 0` is how it is checked rather than asserted.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import psycopg

from rag.config import GenerationConfig, RetrievalConfig
from rag.generate import citations as citation_binding
from rag.generate.client import Completion, GenerationError, LLMClient, Usage
from rag.generate.prompt import INSUFFICIENT_EVIDENCE, assemble, correction_prompt, system_prompt
from rag.index.embed import Encoder
from rag.retrieve import retrieve
from rag.retrieve.rerank import Reranker
from rag.retrieve.types import Candidate, RetrievalResult
from rag.telemetry import StageTimer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Answer:
    """One answered question, with everything needed to log, cite and judge it."""

    question: str
    text: str
    candidates: tuple[Candidate, ...]
    binding: citation_binding.CitationBinding | None
    usage: Usage
    retrieval: RetrievalResult
    refused: bool = False
    refusal_reason: str | None = None
    uncited: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def cited_chunk_ids(self) -> tuple[int, ...]:
        """Chunks the answer actually cited, in order of first citation."""
        return self.binding.cited_chunk_ids if self.binding else ()

    @property
    def retrieved_chunk_ids(self) -> list[int]:
        """Chunks that were retrieved, cited or not. What `query_logs` records."""
        return [candidate.chunk_id for candidate in self.candidates]


def refuse(question: str, result: RetrievalResult, reason: str, timer: StageTimer) -> Answer:
    """Build a refusal without calling the model.

    Constructed here rather than by asking the model to refuse, because asking costs
    tokens to produce a sentence already known in advance -- and because a refusal the
    model chose is a behaviour to measure, while a refusal the retrieval floor forced is
    a guarantee.
    """
    logger.info("refusing %r without an LLM call: %s", question[:60], reason)
    return Answer(
        question=question,
        text=INSUFFICIENT_EVIDENCE,
        candidates=(),
        binding=None,
        usage=Usage(),
        retrieval=result,
        refused=True,
        refusal_reason=reason,
        timings_ms=timer.as_dict(),
    )


def answer_question_stream(
    question: str,
    conn: psycopg.Connection,
    llm: LLMClient,
    *,
    config: RetrievalConfig | None = None,
    generation: GenerationConfig | None = None,
    timer: StageTimer | None = None,
    encoder: Encoder | None = None,
    reranker: Reranker | None = None,
) -> Iterator[str | Answer]:
    """Yield answer text as it is generated, then one final `Answer`.

    Lives here rather than in the API so the streaming path runs the same retrieval,
    refusal and citation-binding code as the blocking one. An endpoint that orchestrated
    this itself would be a second implementation, and the two would drift.

    One deliberate difference from `answer_question`: **streaming does not retry for
    citations.** The correction pass rewrites an answer the user has already watched
    arrive, and replacing streamed text is worse than flagging it. A streamed answer with
    uncited claims comes back with `uncited=True` instead.

    A refusal yields the refusal sentence and buys no tokens.
    """
    config = config or RetrievalConfig()
    generation = generation or GenerationConfig()
    timer = timer or StageTimer()

    result = retrieve(question, config, conn, timer=timer, encoder=encoder, reranker=reranker)
    if result.refused or not result.candidates:
        reason = "below_score_floor" if result.refused else "no_candidates"
        refusal = refuse(question, result, reason, timer)
        yield refusal.text
        yield refusal
        return

    system = system_prompt(generation.prompt_version)
    prompt = assemble(question, result.candidates)
    usage = Usage()
    completion: Completion | None = None

    with timer.stage("generation_ms"):
        for chunk in llm.stream(system, prompt, generation):
            if isinstance(chunk, Completion):
                completion = chunk
            else:
                yield chunk

    if completion is None:  # pragma: no cover - a client that never yielded a completion
        raise GenerationError("stream ended without a final completion")

    usage.add(completion)
    binding = citation_binding.bind(completion.text, result.candidates)
    yield Answer(
        question=question,
        text=completion.text,
        candidates=result.candidates,
        binding=binding,
        usage=usage,
        retrieval=result,
        refused=binding.is_refusal,
        refusal_reason="model_declined" if binding.is_refusal else None,
        uncited=binding.uncited,
        timings_ms=timer.as_dict(),
    )


def answer_question(
    question: str,
    conn: psycopg.Connection,
    llm: LLMClient,
    *,
    config: RetrievalConfig | None = None,
    generation: GenerationConfig | None = None,
    timer: StageTimer | None = None,
    encoder: Encoder | None = None,
    reranker: Reranker | None = None,
) -> Answer:
    """Answer `question` from the corpus, or refuse.

    Refuses in two cases, both before any model call: the reranker's best score fell
    below `score_floor`, or retrieval returned nothing at all. They are recorded as
    different reasons -- "the corpus has nothing good enough" and "the corpus has nothing
    matching" are different facts about the system.
    """
    config = config or RetrievalConfig()
    generation = generation or GenerationConfig()
    timer = timer or StageTimer()

    result = retrieve(question, config, conn, timer=timer, encoder=encoder, reranker=reranker)

    if result.refused:
        return refuse(question, result, "below_score_floor", timer)
    if not result.candidates:
        return refuse(question, result, "no_candidates", timer)

    system = system_prompt(generation.prompt_version)
    prompt = assemble(question, result.candidates)
    usage = Usage()

    with timer.stage("generation_ms"):
        completion = llm.complete(system, prompt, generation)
    usage.add(completion)
    binding = citation_binding.bind(completion.text, result.candidates)

    # One correction attempt. A model that ignored the citation rule once will sometimes
    # comply when shown its own offending sentences; one that ignores it twice will not
    # comply on a third pass, and the answer is flagged instead of retried into a bill.
    attempts = 0
    while binding.uncited and attempts < generation.max_citation_retries:
        attempts += 1
        logger.info(
            "retrying for citations (attempt %d): %d uncited, %d invalid markers",
            attempts,
            len(binding.uncited_sentences),
            len(binding.invalid_markers),
        )
        with timer.stage("generation_ms"):
            completion = llm.complete(
                system,
                f"{prompt}\n\n{correction_prompt(completion.text, binding.uncited_sentences)}",
                generation,
            )
        usage.add(completion)
        binding = citation_binding.bind(completion.text, result.candidates)

    if binding.uncited:
        logger.warning(
            "answer still uncited after %d retries; returning it flagged rather than "
            "presenting it as sourced",
            attempts,
        )

    return Answer(
        question=question,
        text=completion.text,
        candidates=result.candidates,
        binding=binding,
        usage=usage,
        retrieval=result,
        refused=binding.is_refusal,
        refusal_reason="model_declined" if binding.is_refusal else None,
        uncited=binding.uncited,
        timings_ms=timer.as_dict(),
    )
