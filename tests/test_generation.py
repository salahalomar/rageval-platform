"""Generation orchestration, against a live Postgres and a scripted LLM.

No test in this file spends money or touches a network. The LLM is a scripted fake, which
is the point of the `LLMClient` seam: the refusal guarantee, the citation retry and the
cost arithmetic are all checkable without a key.
"""

import hashlib
from collections.abc import Iterator
from datetime import date

import psycopg
import pytest

from conftest import ConstantReranker, FakeEncoder
from rag.config import GenerationConfig, RetrievalConfig
from rag.db import connect
from rag.generate import INSUFFICIENT_EVIDENCE, ScriptedLLMClient, answer_question
from rag.generate.answer import answer_question_stream
from rag.generate.client import Completion, UnknownModelRateError, Usage, cost_usd
from rag.generate.logging_store import record
from rag.generate.prompt import assemble, context_block, correction_prompt, system_prompt
from rag.index.embed import embed_corpus
from rag.ingest import store
from rag.ingest.arxiv import PaperMetadata
from rag.ingest.chunk import Chunk
from rag.retrieve.types import Candidate

PAPER_ID = "0000.77777v1"
MODEL = "BAAI/bge-small-en-v1.5"

TEXTS = [
    "Learning rate warmup stabilises transformer training during the first thousand steps.",
    "The batch size used for fine tuning was 512 sequences across eight accelerators.",
    "Reciprocal rank fusion combines ranked lists without normalising their scores.",
]


# --- cost arithmetic (no database needed) -----------------------------------


def test_cost_is_computed_from_the_published_rates() -> None:
    # Haiku 4.5: $1.00 per Mtok input, $5.00 per Mtok output.
    # 1,000,000 input + 1,000,000 output = $1.00 + $5.00 = $6.00
    assert cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.00)
    assert cost_usd("claude-haiku-4-5", 1_000, 500) == pytest.approx(0.001 + 0.0025)


def test_a_zero_token_call_costs_nothing() -> None:
    assert cost_usd("claude-haiku-4-5", 0, 0) == 0.0


def test_an_unpriced_model_raises_rather_than_costing_zero() -> None:
    # A silent zero would make the cost column wrong for exactly the arm someone added
    # without thinking about its price.
    with pytest.raises(UnknownModelRateError, match="MODEL_RATES"):
        cost_usd("some/unpriced-model", 100, 100)


def test_usage_accumulates_across_calls() -> None:
    # One answer can take two calls thanks to the citation retry, and the log records
    # what the answer cost rather than what its last attempt cost.
    usage = Usage()
    usage.add(Completion("a", "claude-haiku-4-5", input_tokens=1000, output_tokens=100))
    usage.add(Completion("b", "claude-haiku-4-5", input_tokens=2000, output_tokens=200))
    assert usage.calls == 2
    assert usage.input_tokens == 3000
    assert usage.output_tokens == 300
    assert usage.cost_usd == pytest.approx(cost_usd("claude-haiku-4-5", 3000, 300))


# --- prompt assembly --------------------------------------------------------


def test_context_blocks_are_numbered_from_one_with_provenance() -> None:
    block = context_block(
        1,
        Candidate(
            chunk_id=5,
            score=0.9,
            rank=1,
            content="the text",
            section_path="3 Method",
            paper_id="p",
            page_start=1,
            page_end=1,
            paper_title="A Paper",
        ),
    )
    assert block.startswith("[1] (A Paper — 3 Method)")
    assert "the text" in block


def test_the_question_comes_after_the_context() -> None:
    # Models attend most reliably to the end of a long input; a question buried above
    # several hundred lines of context gets half-answered.
    prompt = assemble(
        "What is X?",
        [
            Candidate(
                chunk_id=1,
                score=0.5,
                rank=1,
                content="c",
                section_path="s",
                paper_id="p",
                page_start=1,
                page_end=1,
            )
        ],
    )
    assert prompt.index("Context blocks:") < prompt.index("Question: What is X?")


def test_the_system_prompt_is_loaded_from_its_versioned_file() -> None:
    text = system_prompt("v1")
    assert INSUFFICIENT_EVIDENCE in text
    assert "[1]" in text


def test_an_unknown_prompt_version_fails_loudly() -> None:
    with pytest.raises(ValueError, match="no prompt file"):
        system_prompt("does-not-exist")


def test_the_correction_prompt_quotes_the_offending_sentences() -> None:
    # A model that produced an uncited sentence has already read the rule; what it has
    # not seen is which of its own sentences broke it.
    text = correction_prompt("Some answer.", ["Batch size was 512."])
    assert "Batch size was 512." in text


# --- the fixture corpus -----------------------------------------------------


def make_chunk(ordinal: int, text: str) -> Chunk:
    return Chunk(
        ordinal=ordinal,
        section_path=f"{ordinal} Section",
        content=text,
        embed_input=text,
        token_count=len(text.split()),
        page_start=1,
        page_end=1,
        char_start=ordinal * 200,
        char_end=ordinal * 200 + len(text),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        embed_input_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


@pytest.fixture
def corpus() -> Iterator[tuple[psycopg.Connection, RetrievalConfig]]:
    config = RetrievalConfig(
        embedding_model=MODEL,
        chunk_tokens=141,
        rerank_enabled=False,
        dense_top_k=3,
        lexical_top_k=3,
        final_top_k=3,
    )
    chunks = [make_chunk(i, text) for i, text in enumerate(TEXTS)]
    metadata = PaperMetadata(
        id=PAPER_ID,
        title="Generation Fixture Paper",
        authors=("A Author",),
        abstract="An abstract.",
        categories=("cs.LG",),
        published_at=date(2024, 1, 1),
        pdf_url="https://example.invalid/paper",
    )
    with connect() as conn:
        conn.execute("DELETE FROM papers WHERE id = %s", (PAPER_ID,))
        with conn.transaction():
            store.upsert_paper(conn, metadata, "fixturesha")
            store.insert_chunks(conn, PAPER_ID, chunks, config)
        embed_corpus(
            conn,
            model=MODEL,
            chunk_config_sha256=config.chunking_sha256(),
            encoder=FakeEncoder(),
        )
        try:
            yield conn, config
        finally:
            conn.execute("DELETE FROM papers WHERE id = %s", (PAPER_ID,))


pytestmark = pytest.mark.integration


# --- answering --------------------------------------------------------------


def test_a_cited_answer_binds_its_markers_to_chunks(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    llm = ScriptedLLMClient(["Warmup stabilises transformer training in early steps [1]."])
    answer = answer_question("warmup", conn, llm, config=config, encoder=FakeEncoder())
    assert not answer.refused
    assert answer.cited_chunk_ids
    assert answer.usage.calls == 1
    assert answer.usage.cost_usd > 0


def test_the_prompt_sent_to_the_model_contains_the_retrieved_context(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    llm = ScriptedLLMClient(["Warmup stabilises transformer training in early steps [1]."])
    answer_question("warmup", conn, llm, config=config, encoder=FakeEncoder())
    system, prompt = llm.prompts[0]
    assert INSUFFICIENT_EVIDENCE in system
    assert "[1] (Generation Fixture Paper" in prompt


def test_an_uncited_answer_triggers_exactly_one_retry(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    llm = ScriptedLLMClient(
        [
            "Batch size was 512 sequences on eight accelerators.",  # no marker
            "Batch size was 512 sequences on eight accelerators [2].",  # corrected
        ]
    )
    answer = answer_question("batch size", conn, llm, config=config, encoder=FakeEncoder())
    assert answer.usage.calls == 2
    assert not answer.uncited
    assert answer.cited_chunk_ids


def test_an_answer_still_uncited_after_the_retry_is_flagged_not_hidden(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # Returning it flagged beats presenting it as sourced, and beats retrying into a bill.
    conn, config = corpus
    llm = ScriptedLLMClient(["Batch size was 512 sequences on eight accelerators."])
    answer = answer_question("batch size", conn, llm, config=config, encoder=FakeEncoder())
    assert answer.usage.calls == 2  # one attempt plus one retry
    assert answer.uncited


def test_retries_can_be_disabled(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    llm = ScriptedLLMClient(["Batch size was 512 sequences on eight accelerators."])
    answer = answer_question(
        "batch size",
        conn,
        llm,
        config=config,
        encoder=FakeEncoder(),
        generation=GenerationConfig(max_citation_retries=0),
    )
    assert answer.usage.calls == 1
    assert answer.uncited


# --- refusal, the property that costs nothing -------------------------------


def test_a_question_below_the_score_floor_refuses_without_calling_the_model(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # The acceptance criterion for this phase: zero tokens, zero dollars, no call at all.
    conn, config = corpus
    llm = ScriptedLLMClient(["this must never be returned"])
    answer = answer_question(
        "What were Tesla's Q3 2025 delivery numbers?",
        conn,
        llm,
        config=config.model_copy(update={"rerank_enabled": True, "score_floor": 0.0}),
        encoder=FakeEncoder(),
        reranker=ConstantReranker(-5.0),
    )
    assert answer.refused
    assert answer.refusal_reason == "below_score_floor"
    assert answer.text == INSUFFICIENT_EVIDENCE
    assert answer.usage.calls == 0
    assert answer.usage.input_tokens == 0
    assert answer.usage.output_tokens == 0
    assert answer.usage.cost_usd == 0.0
    assert llm.prompts == []  # the model was never asked anything


def test_a_query_matching_nothing_refuses_for_a_different_reason(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # "The corpus has nothing good enough" and "the corpus has nothing matching" are
    # different facts and the golden set scores them separately.
    conn, config = corpus
    llm = ScriptedLLMClient(["never returned"])
    answer = answer_question(
        "warmup",
        conn,
        llm,
        config=config.model_copy(update={"chunk_tokens": 995}),
        encoder=FakeEncoder(),
    )
    assert answer.refused
    assert answer.refusal_reason == "no_candidates"
    assert answer.usage.calls == 0


def test_a_model_that_declines_is_recorded_as_a_different_refusal(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # A refusal the model chose is a behaviour to measure; a refusal the floor forced is
    # a guarantee. They must not be conflated in the logs.
    conn, config = corpus
    llm = ScriptedLLMClient([INSUFFICIENT_EVIDENCE])
    answer = answer_question("warmup", conn, llm, config=config, encoder=FakeEncoder())
    assert answer.refused
    assert answer.refusal_reason == "model_declined"
    assert answer.usage.calls == 1  # this one did cost a call


# --- streaming --------------------------------------------------------------


def test_streaming_yields_text_then_a_final_answer(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    from rag.generate.answer import Answer

    conn, config = corpus
    llm = ScriptedLLMClient(["Warmup stabilises transformer training in early steps [1]."])
    chunks = list(answer_question_stream("warmup", conn, llm, config=config, encoder=FakeEncoder()))
    assert isinstance(chunks[-1], Answer)
    assert all(isinstance(c, str) for c in chunks[:-1])
    assert "".join(str(c) for c in chunks[:-1]).strip() == chunks[-1].text.strip()


def test_streaming_a_refusal_costs_nothing(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    from rag.generate.answer import Answer

    conn, config = corpus
    llm = ScriptedLLMClient(["never returned"])
    chunks = list(
        answer_question_stream(
            "warmup",
            conn,
            llm,
            config=config.model_copy(update={"chunk_tokens": 994}),
            encoder=FakeEncoder(),
        )
    )
    final = chunks[-1]
    assert isinstance(final, Answer)
    assert final.refused
    assert final.usage.calls == 0
    assert llm.prompts == []


# --- query_logs -------------------------------------------------------------


def test_an_answer_is_recorded_in_query_logs(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    conn, config = corpus
    generation = GenerationConfig()
    llm = ScriptedLLMClient(["Warmup stabilises transformer training in early steps [1]."])
    answer = answer_question("warmup", conn, llm, config=config, encoder=FakeEncoder())

    log_id = record(conn, answer, config, generation)
    assert log_id is not None
    row = conn.execute(
        """
        SELECT question, answer, refused, refusal_reason, uncited, llm_calls,
               input_tokens, output_tokens, cost_usd, retrieved_ids, cited_chunk_ids
        FROM query_logs WHERE id = %s
        """,
        (log_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == "warmup"
    assert row[2] is False
    assert row[5] == 1
    assert row[9]  # retrieved ids
    assert row[10]  # cited chunk ids
    conn.execute("DELETE FROM query_logs WHERE id = %s", (log_id,))


def test_a_refusal_is_recorded_too(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # Especially refusals: refusal rate is the number that reveals a floor set so high
    # the system declines everything, and a log that skipped them would hide it.
    conn, config = corpus
    generation = GenerationConfig()
    llm = ScriptedLLMClient(["never returned"])
    answer = answer_question(
        "warmup",
        conn,
        llm,
        config=config.model_copy(update={"chunk_tokens": 993}),
        encoder=FakeEncoder(),
    )
    log_id = record(conn, answer, config, generation)
    assert log_id is not None
    row = conn.execute(
        "SELECT refused, refusal_reason, llm_calls, cost_usd FROM query_logs WHERE id = %s",
        (log_id,),
    ).fetchone()
    assert row is not None
    assert row[0] is True
    assert row[1] == "no_candidates"
    assert row[2] == 0
    assert float(row[3]) == 0.0
    conn.execute("DELETE FROM query_logs WHERE id = %s", (log_id,))


def test_the_log_records_both_configs(
    corpus: tuple[psycopg.Connection, RetrievalConfig],
) -> None:
    # A result whose prompt version and model cannot be reconstructed is not reproducible.
    conn, config = corpus
    generation = GenerationConfig()
    llm = ScriptedLLMClient(["Warmup stabilises transformer training in early steps [1]."])
    answer = answer_question("warmup", conn, llm, config=config, encoder=FakeEncoder())
    log_id = record(conn, answer, config, generation)
    assert log_id is not None
    row = conn.execute(
        "SELECT config, generation FROM query_logs WHERE id = %s", (log_id,)
    ).fetchone()
    assert row is not None
    assert row[0]["fusion"] == config.fusion
    assert row[1]["model"] == generation.model
    assert row[1]["prompt_version"] == generation.prompt_version
    conn.execute("DELETE FROM query_logs WHERE id = %s", (log_id,))
