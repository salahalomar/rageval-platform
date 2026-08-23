"""HTTP surface: health, and the query endpoints."""

import json
import logging
from collections.abc import Iterator
from typing import Literal

from fastapi import FastAPI, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag import __version__ as rag_version
from rag.config import GenerationConfig, RetrievalConfig
from rag.db import DatabaseHealth, check_health, connect
from rag.generate import answer_question
from rag.generate.answer import Answer, answer_question_stream
from rag.generate.client import AnthropicClient, GenerationError, LLMClient
from rag.generate.logging_store import record

logger = logging.getLogger(__name__)

app = FastAPI(
    title="rag-eval-platform",
    version=rag_version,
    description="RAG over arXiv ML papers. The evaluation harness is the product.",
)

# Constructed lazily and once. Building it at import time would make the whole API fail
# to start on a machine with no credentials -- including CI, where /health is checked and
# nothing is ever generated.
_llm: LLMClient | None = None


def llm_client() -> LLMClient:
    """The process-wide LLM client, created on first use."""
    global _llm
    if _llm is None:
        _llm = AnthropicClient()
    return _llm


class HealthResponse(BaseModel):
    """Payload for GET /health."""

    status: Literal["ok", "degraded"]
    db: DatabaseHealth
    version: str


class QueryRequest(BaseModel):
    """Body for POST /query.

    Retrieval and generation settings are accepted per request so the Phase 8 config
    panel can toggle reranking and fusion and show the effect live. They default to the
    same values the eval harness uses, so the demo and the measurements describe one
    system.
    """

    question: str = Field(min_length=1, max_length=2000)
    config: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()


class CitedSentence(BaseModel):
    """One sentence of the answer with the chunks it cited."""

    text: str
    chunk_ids: list[int]


class SourceBlock(BaseModel):
    """A retrieved chunk, in the numbering the answer's markers refer to."""

    marker: int
    chunk_id: int
    paper_id: str
    paper_title: str
    section_path: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    content: str
    score: float
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    """Payload for POST /query."""

    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    uncited: bool
    sentences: list[CitedSentence]
    sources: list[SourceBlock]
    cited_chunk_ids: list[int]
    timings_ms: dict[str, float]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    llm_calls: int
    query_log_id: int | None


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
def health(response: Response) -> HealthResponse:
    """Report process and database state.

    Defined with `def` rather than `async def` on purpose: FastAPI runs synchronous
    endpoints in a threadpool, which lets the whole library stay synchronous and keeps
    the API and the eval runner on one code path.
    """
    db = check_health()
    if not db.connected:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", db=db, version=rag_version)
    return HealthResponse(status="ok", db=db, version=rag_version)


def _sources(answer: Answer) -> list[SourceBlock]:
    """The retrieved chunks, numbered exactly as the answer's `[n]` markers refer to."""
    return [
        SourceBlock(
            marker=index,
            chunk_id=candidate.chunk_id,
            paper_id=candidate.paper_id,
            paper_title=candidate.paper_title,
            section_path=candidate.section_path,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            char_start=candidate.char_start,
            char_end=candidate.char_end,
            content=candidate.content,
            score=candidate.score,
            rerank_score=candidate.rerank_score,
        )
        for index, candidate in enumerate(answer.candidates, start=1)
    ]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, response: Response) -> QueryResponse:
    """Answer a question from the indexed corpus, or refuse.

    Synchronous, so it shares the library's single code path with the eval runner.
    """
    with connect() as conn:
        try:
            answer = answer_question(
                request.question,
                conn,
                llm_client(),
                config=request.config,
                generation=request.generation,
            )
        except GenerationError as exc:
            logger.exception("generation failed")
            response.status_code = status.HTTP_502_BAD_GATEWAY
            return QueryResponse(
                question=request.question,
                answer=f"The model could not be reached: {exc}",
                refused=True,
                refusal_reason="generation_error",
                uncited=False,
                sentences=[],
                sources=[],
                cited_chunk_ids=[],
                timings_ms={},
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                llm_calls=0,
                query_log_id=None,
            )

        log_id = record(conn, answer, request.config, request.generation)

    return QueryResponse(
        question=answer.question,
        answer=answer.text,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        uncited=answer.uncited,
        sentences=[
            CitedSentence(text=sentence.text, chunk_ids=list(sentence.chunk_ids))
            for sentence in (answer.binding.sentences if answer.binding else ())
        ],
        sources=_sources(answer),
        cited_chunk_ids=list(answer.cited_chunk_ids),
        timings_ms=answer.timings_ms,
        input_tokens=answer.usage.input_tokens,
        output_tokens=answer.usage.output_tokens,
        cost_usd=round(answer.usage.cost_usd, 6),
        llm_calls=answer.usage.calls,
        query_log_id=log_id,
    )


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/query/stream")
def query_stream(question: str) -> StreamingResponse:
    """Answer a question as a token stream, ending with a citations event.

    Retrieval runs to completion before the first token, because the citation markers a
    client renders are only meaningful once it knows which sources they number. The
    stages event is emitted first so the UI can show "searching / reranking / generating"
    against real timings rather than a spinner.

    A refusal streams the refusal sentence and no tokens are ever bought for it.
    """

    def events() -> Iterator[str]:
        config = RetrievalConfig()
        generation = GenerationConfig()
        with connect() as conn:
            final: Answer | None = None
            try:
                for chunk in answer_question_stream(
                    question, conn, llm_client(), config=config, generation=generation
                ):
                    if isinstance(chunk, Answer):
                        final = chunk
                        # Sources arrive before the citations event but after the tokens,
                        # because retrieval is what numbers them and a client cannot
                        # resolve a marker it has not been given the blocks for.
                        yield _sse("sources", [block.model_dump() for block in _sources(chunk)])
                        yield _sse("stages", chunk.timings_ms)
                    else:
                        yield _sse("token", {"text": chunk})
            except GenerationError as exc:
                yield _sse("error", {"message": str(exc)})
                return

            if final is None:  # pragma: no cover - defensive
                yield _sse("error", {"message": "stream ended without an answer"})
                return

            log_id = record(conn, final, config, generation)
            yield _sse(
                "citations",
                {
                    "cited_chunk_ids": list(final.cited_chunk_ids),
                    "refused": final.refused,
                    "refusal_reason": final.refusal_reason,
                    "uncited": final.uncited,
                    "cost_usd": round(final.usage.cost_usd, 6),
                    "query_log_id": log_id,
                },
            )
            yield _sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")
