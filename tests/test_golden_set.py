"""The golden set's schema, sampling and filtering — all checkable without spending.

Only the two generation steps need a paid API, and both are exercised here against a
scripted client. What is being tested is the *protocol*: that the no-context filter
discards what it should, that deduplication catches near-identical questions, that
stratification actually stratifies, and that nothing but human review can write the
verified file.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from conftest import FakeEncoder
from eval.costs import CallEstimate, estimate_run, estimate_tokens
from eval.generate_golden import deduplicate, generate_one, survives_no_context
from eval.sample import ChunkSample, Stratification, classify_section, stratified_sample
from eval.schema import GoldenCandidate, GoldenItem, read_items, write_jsonl
from rag.config import GenerationConfig
from rag.generate.client import ScriptedLLMClient

GENERATION = GenerationConfig()


def chunk(chunk_id: int, section: str, paper: str = "p1", tokens: int = 300) -> ChunkSample:
    return ChunkSample(
        chunk_id=chunk_id,
        paper_id=paper,
        paper_title=f"Paper {paper}",
        section_path=section,
        section_type=classify_section(section),
        token_count=tokens,
        content=f"content of chunk {chunk_id}",
    )


def candidate(item_id: str, question: str) -> GoldenCandidate:
    return GoldenCandidate(
        id=item_id,
        question=question,
        expected_answer="an answer",
        relevant_chunk_ids=[1],
        difficulty="medium",
        type="factual",
        answerable=True,
        provenance="llm_generated",
        source_chunk_id=1,
        source_section_type="method",
        source_paper_id="p1",
    )


# --- section classification -------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("Abstract", "abstract"),
        ("Frontmatter", "abstract"),
        ("1 Introduction", "introduction"),
        ("2 Related Work", "introduction"),  # related work is framing, not method
        ("3 Method > 3.2 Training", "method"),
        ("Experimental Setup", "method"),  # setup describes what was done, not what happened
        ("V. RESULTS", "results"),
        ("6 Discussion", "results"),
        ("Limitations", "limitations"),
        ("Broader Impact", "limitations"),
        ("Body", "unstructured"),  # the Phase 1 fallback, kept visible
        ("A Four-Component Decomposition", "other"),
    ],
)
def test_section_paths_map_to_coarse_types(path: str, expected: str) -> None:
    assert classify_section(path) == expected


def test_the_chunker_fallback_is_its_own_stratum() -> None:
    # Chunks with no detected structure are a known quality signal. Folding them into
    # "other" would let the set draw from the papers the parser handled worst without
    # anyone noticing.
    assert classify_section("Body") == "unstructured"
    assert classify_section("Body") != "other"


# --- stratified sampling ----------------------------------------------------


def corpus(size: int = 400) -> list[ChunkSample]:
    sections = ["Method", "Results", "Abstract", "Limitations", "Introduction", "Appendix"]
    return [chunk(i, sections[i % len(sections)], paper=f"p{i % 60}") for i in range(size)]


def test_sampling_hits_the_requested_size() -> None:
    selected, stats = stratified_sample(corpus(), 100)
    assert len(selected) == 100
    assert stats.returned == 100


def test_sampling_over_weights_method_and_results() -> None:
    # The corpus here is uniform across six sections; the sample must not be.
    _, stats = stratified_sample(corpus(), 100)
    assert stats.by_section_type["method"] == 30
    assert stats.by_section_type["results"] == 30
    assert stats.by_section_type["introduction"] == 10


def test_no_paper_dominates_the_sample() -> None:
    # A set drawing ten questions from one paper measures that paper, not the corpus.
    selected, stats = stratified_sample(corpus(), 100, max_per_paper=2)
    assert stats.max_per_paper <= 2
    counts: dict[str, int] = {}
    for sample in selected:
        counts[sample.paper_id] = counts.get(sample.paper_id, 0) + 1
    assert max(counts.values()) <= 2


def test_sampling_is_reproducible() -> None:
    # An unseeded sample makes it impossible to tell whether a protocol change altered
    # the set or the dice did.
    first, _ = stratified_sample(corpus(), 60)
    second, _ = stratified_sample(corpus(), 60)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_a_different_seed_gives_a_different_sample() -> None:
    first, _ = stratified_sample(corpus(), 60, seed=1)
    second, _ = stratified_sample(corpus(), 60, seed=2)
    assert [c.chunk_id for c in first] != [c.chunk_id for c in second]


def test_no_chunk_is_sampled_twice() -> None:
    selected, _ = stratified_sample(corpus(), 120)
    assert len({c.chunk_id for c in selected}) == len(selected)


def test_a_stratum_that_cannot_fill_its_quota_is_topped_up() -> None:
    # A corpus with no limitations sections should still yield the requested count, with
    # the shortfall visible in the report rather than silently dropped.
    only_method = [chunk(i, "Method", paper=f"p{i}") for i in range(50)]
    selected, stats = stratified_sample(only_method, 30)
    assert len(selected) == 30
    assert stats.by_section_type["method"] == 30


def test_stratification_report_is_printable() -> None:
    stats = Stratification(requested=10, returned=10)
    stats.by_section_type.update(["method", "method", "results"])
    assert any("method" in line for line in stats.as_lines())


# --- the no-context filter --------------------------------------------------


def test_a_question_the_model_cannot_answer_is_kept() -> None:
    llm = ScriptedLLMClient(["UNKNOWN"])
    keep, answer = survives_no_context(llm, "What batch size did paper X use?", GENERATION)
    assert keep
    assert answer == "UNKNOWN"


def test_a_question_answered_from_parametric_knowledge_is_discarded() -> None:
    # This is the step that makes an LLM-generated set defensible: a question the model
    # answers without any context tests what it already knows, not retrieval.
    llm = ScriptedLLMClient(["Transformers were introduced in Attention Is All You Need."])
    keep, _ = survives_no_context(llm, "What paper introduced the transformer?", GENERATION)
    assert not keep


def test_the_parametric_answer_is_kept_for_audit() -> None:
    # A reviewer must be able to check the filter's judgement rather than trust it.
    llm = ScriptedLLMClient(["It was 512 sequences."])
    _, answer = survives_no_context(llm, "q", GENERATION)
    assert answer == "It was 512 sequences."


def test_the_filter_only_sees_the_question() -> None:
    # If it saw the chunk it would answer everything, and the filter would reject the
    # entire candidate set.
    llm = ScriptedLLMClient(["UNKNOWN"])
    survives_no_context(llm, "the question text", GENERATION)
    _, prompt = llm.prompts[0]
    assert prompt == "the question text"


# --- question generation ----------------------------------------------------


def test_a_well_formed_reply_is_parsed() -> None:
    reply = json.dumps(
        {"question": "What is X?", "expected_answer": "Y", "difficulty": "easy", "type": "factual"}
    )
    parsed = generate_one(ScriptedLLMClient([reply]), chunk(1, "Method"), GENERATION)
    assert parsed is not None
    assert parsed["question"] == "What is X?"


def test_a_fenced_reply_is_still_parsed() -> None:
    # Models wrap JSON in code fences regardless of instructions.
    reply = '```json\n{"question": "What is X?", "expected_answer": "Y"}\n```'
    parsed = generate_one(ScriptedLLMClient([reply]), chunk(1, "Method"), GENERATION)
    assert parsed is not None
    assert parsed["question"] == "What is X?"


def test_an_unparseable_reply_is_dropped_not_crashed_on() -> None:
    parsed = generate_one(ScriptedLLMClient(["I'm afraid I can't"]), chunk(1, "M"), GENERATION)
    assert parsed is None


def test_a_reply_without_a_question_is_dropped() -> None:
    llm = ScriptedLLMClient(['{"expected_answer": "Y"}'])
    parsed = generate_one(llm, chunk(1, "M"), GENERATION)
    assert parsed is None


# --- deduplication ----------------------------------------------------------


def test_identical_questions_are_deduplicated() -> None:
    # Two phrasings of one question silently double its weight in every metric.
    candidates = [
        candidate("g-001", "What batch size was used?"),
        candidate("g-002", "What batch size was used?"),
    ]
    kept, dropped = deduplicate(candidates, FakeEncoder())
    assert len(kept) == 1
    assert dropped == 1


def test_distinct_questions_are_both_kept() -> None:
    candidates = [
        candidate("g-001", "What batch size was used?"),
        candidate("g-002", "Which optimiser did they choose?"),
    ]
    kept, dropped = deduplicate(candidates, FakeEncoder())
    assert len(kept) == 2
    assert dropped == 0


def test_deduplicating_nothing_is_safe() -> None:
    assert deduplicate([], FakeEncoder()) == ([], 0)


# --- the record schema ------------------------------------------------------


def test_an_answerable_item_must_name_its_evidence() -> None:
    # Otherwise recall cannot be scored against anything.
    with pytest.raises(ValueError, match="at least one relevant chunk"):
        GoldenItem(
            id="g-001",
            question="A question long enough",
            expected_answer="a",
            relevant_chunk_ids=[],
            difficulty="easy",
            type="factual",
            answerable=True,
            provenance="llm_generated",
        )


def test_an_unanswerable_item_must_not_name_evidence() -> None:
    with pytest.raises(ValueError, match="must not name relevant chunks"):
        GoldenItem(
            id="g-002",
            question="A question long enough",
            expected_answer="a",
            relevant_chunk_ids=[5],
            difficulty="easy",
            type="unanswerable",
            answerable=False,
            provenance="human_written",
        )


def test_an_unknown_field_is_rejected() -> None:
    # A typo would otherwise be accepted and then silently ignored by every metric.
    with pytest.raises(ValueError):
        GoldenItem.model_validate(
            {
                "id": "g-003",
                "question": "A question long enough",
                "expected_answer": "a",
                "relevant_chunk_ids": [1],
                "difficulty": "easy",
                "type": "factual",
                "answerable": True,
                "provenance": "llm_generated",
                "relevent_chunks": [2],
            }
        )


def test_verified_requires_both_a_name_and_a_date() -> None:
    item = GoldenItem(
        id="g-004",
        question="A question long enough",
        expected_answer="a",
        relevant_chunk_ids=[1],
        difficulty="easy",
        type="factual",
        answerable=True,
        provenance="llm_generated",
    )
    assert not item.verified
    stamped = item.model_copy(update={"verified_by": "someone", "verified_at": date(2026, 1, 1)})
    assert stamped.verified


def test_a_candidate_is_promoted_with_the_reviewer_stamped_on_it() -> None:
    item = candidate("g-005", "A question long enough").to_item("reviewer", date(2026, 1, 2))
    assert item.verified_by == "reviewer"
    assert item.verified_at == date(2026, 1, 2)
    assert item.verified


def test_records_round_trip_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "v1.jsonl"
    items = [
        candidate(f"g-{i:03d}", f"Question number {i} here").to_item("r", date(2026, 1, 1))
        for i in range(3)
    ]
    assert write_jsonl(path, items) == 3
    assert [item.id for item in read_items(path)] == [item.id for item in items]


def test_a_malformed_line_names_its_line_number(tmp_path: Path) -> None:
    # A golden file is hand-edited; "validation error" without a line number is a bad
    # experience at 11pm during a verification pass.
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "g-001"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        read_items(path)


# --- the committed hard cases -----------------------------------------------


HARD_CASES = Path(__file__).resolve().parents[1] / "eval" / "golden" / "hard_cases.jsonl"


def test_the_hard_cases_file_parses() -> None:
    assert len(read_items(HARD_CASES)) == 15


def test_the_hard_cases_cover_all_four_types() -> None:
    # Each type probes a different failure; a set missing one is blind to it.
    types = {item.type for item in read_items(HARD_CASES)}
    assert {"numeric", "multi_hop", "unanswerable", "distractor"} <= types


def test_the_hard_cases_include_unanswerable_items() -> None:
    # These are what expose hallucination, and most portfolio projects have none.
    unanswerable = [item for item in read_items(HARD_CASES) if not item.answerable]
    assert len(unanswerable) >= 4
    assert all(item.relevant_chunk_ids == [] for item in unanswerable)


def test_no_hard_case_claims_to_be_verified() -> None:
    # They are drafts. Marking them verified would forge the one property the golden set
    # sells, and no automated step is allowed to do that.
    assert not any(item.verified for item in read_items(HARD_CASES))


def test_multi_hop_items_name_more_than_one_chunk() -> None:
    for item in read_items(HARD_CASES):
        if item.type == "multi_hop":
            assert len(item.relevant_chunk_ids) >= 2, item.id


# --- cost projection --------------------------------------------------------


def test_token_estimation_scales_with_length() -> None:
    assert estimate_tokens("a" * 360) == 100
    assert estimate_tokens("") == 1  # never zero, so a cost is never projected as free


def test_a_run_estimate_sums_its_stages() -> None:
    run = estimate_run(
        "claude-haiku-4-5",
        [CallEstimate("a", 10, 100, 10), CallEstimate("b", 5, 200, 20)],
    )
    assert run.calls == 15
    assert run.input_tokens == 10 * 100 + 5 * 200
    assert run.output_tokens == 10 * 10 + 5 * 20
    assert run.cost_usd > 0


def test_the_estimate_states_that_it_is_an_estimate() -> None:
    # Nobody should read a projection as a quote.
    lines = estimate_run("claude-haiku-4-5", [CallEstimate("a", 1, 100, 10)]).as_lines()
    assert any("not measured" in line for line in lines)
    assert any("likely between" in line for line in lines)
