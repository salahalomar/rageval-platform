"""The single typed description of everything configurable about retrieval.

Why one frozen model rather than arguments threaded through the call stack: the
ablation runner works by instantiating variants of this object, and every evaluation
result embeds the exact instance that produced it. A measurement whose configuration
cannot be reconstructed afterwards is not evidence of anything.

Frozen because a mutable config could be altered by one stage after another stage had
already read it, which would make the copy recorded alongside the result a description
of something that never actually ran. Frozen also makes instances hashable, which the
Phase 7 judgement cache relies on.
"""

import hashlib
import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel


class GenerationConfig(BaseModel, frozen=True):
    """Everything configurable about turning retrieved context into an answer.

    Separate from `RetrievalConfig` deliberately. That model is the ablation axis for
    retrieval, and its every field is swept by the Phase 7 matrix; generation has its own
    axes -- model, temperature, prompt version -- which vary independently. Folding them
    together would make each retrieval arm carry fields that had nothing to do with it,
    and would misname the thing. An evaluation result embeds both.
    """

    # Which client implementation serves this config. "anthropic" uses the vendor SDK;
    # "openai_compatible" covers Groq, Cerebras, OpenRouter, Together and a local Ollama
    # or llama.cpp server, all of which speak the same wire format and differ only by
    # base_url. Recorded on every result, because two providers running nominally the
    # same weights do not necessarily produce the same output.
    provider: Literal["anthropic", "openai_compatible"] = "anthropic"

    # Pinned to an exact published model id.
    model: str = "claude-haiku-4-5"

    # Endpoint for the openai_compatible provider. A public URL, not a secret -- the API
    # key lives in Settings and never appears in a result record. Ignored by the
    # anthropic provider.
    base_url: str | None = None

    # Honoured by the openai_compatible provider and ignored by the anthropic one, which
    # is a real asymmetry rather than an oversight.
    #
    # The Anthropic SDK removed sampling controls entirely: temperature, top_p and top_k
    # are absent from messages.create in 1.0.0 and the current model family rejects them
    # with a 400. OpenAI-compatible endpoints still accept temperature, so routing through
    # one restores the determinism knob the plan asked for and the vendor SDK took away.
    #
    # Either way, ENGINEERING.md's determinism requirement covers *retrieval* metrics,
    # which involve no sampling at all. Answer metrics are LLM-judged and were never going
    # to be bit-identical; Phase 7 keeps re-runs stable by caching judgements rather than
    # by pretending generation is deterministic.
    temperature: float = 0.0

    # Deliberately small. Answers here are a few sentences bound to citations, not
    # essays; a large ceiling would only pay for the model to wander past the evidence.
    max_tokens: int = 1024

    # Prompt text lives in a versioned file rather than inline, because a prompt change
    # is an ablation arm exactly like a chunk size is, and a result that cannot name the
    # prompt that produced it is not reproducible.
    prompt_version: str = "v1"

    # One correction attempt when the model returns uncited claims. Beyond one, the cost
    # of retrying exceeds the value: a model that ignores the citation instruction twice
    # is not going to comply on the third pass, and the answer is flagged instead.
    max_citation_retries: int = 1

    def fingerprint(self) -> dict[str, Any]:
        """The generation settings recorded alongside every answer and eval result."""
        return self.model_dump()


class RetrievalConfig(BaseModel, frozen=True):
    """One point in the retrieval configuration space.

    Nothing in the retrieval path may read an environment variable; if a knob affects
    what gets retrieved, it belongs here so that it is captured in the result record.
    Process-level concerns such as the database URL live in `rag.settings` instead.
    """

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_tokens: int = 512
    chunk_overlap_pct: float = 0.15
    drop_references: bool = True  # references sections answer no question worth asking
    drop_figure_only_pages: bool = True  # pages whose extractable text is a caption or less
    contextual_headers: bool = True  # prepend paper+section title to chunk before embedding
    dense_enabled: bool = True
    dense_top_k: int = 50
    # HNSW's per-query search breadth. Not in the original specification, but it changes
    # which chunks come back -- it trades approximate-search recall against latency --
    # and nothing that changes retrieval output may live outside this model, or the
    # result it produced cannot be reconstructed from its record.
    #
    # 400 is measured, not guessed. Against exact-scan ground truth over 50 queries on
    # the 6,386-chunk corpus (`rag bench-index`):
    #
    #     ef_search   recall@10   recall@50   p50 ms   p95 ms
    #            40       0.830       0.860      6.9     11.6
    #           100       0.952       0.921      5.1      8.2
    #           200       0.980       0.973      6.7     12.4
    #           400       1.000       1.000      7.1      8.2
    #           800       1.000       1.000      7.4      8.0
    #
    # Chosen because approximation error here is indistinguishable from retrieval error
    # in the published metrics: at ef_search=100 roughly one true neighbour in twenty is
    # missed, and every ablation arm would carry that deficit while appearing to be a
    # property of the retrieval method. 400 removes it for 2ms.
    #
    # This value is tuned to a corpus of this size and must be re-measured if the corpus
    # grows; 400 will not stay exact at ten times the rows.
    hnsw_ef_search: int = 400
    lexical_enabled: bool = True
    lexical_top_k: int = 50
    fusion: Literal["rrf", "dense_only", "lexical_only"] = "rrf"
    rrf_k: int = 60
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    # How many fused candidates reach the cross-encoder. Not in the original field list,
    # but the architecture requires it: fusion can emit a hundred candidates and a
    # cross-encoder needs one CPU forward pass each. It changes what comes back, so it
    # belongs here rather than in a constant.
    #
    # Measured on this corpus, 4 queries, CPU (`rag search --compare`):
    #
    #     top_n   p50 ms   p95 ms   top-5 kept vs n=50   mean rank movement
    #         5      566      602              0.8 / 5                  1.1
    #        10    1,132    1,144              1.0 / 5                  2.8
    #        25    2,538    2,775              2.8 / 5                 10.1
    #        50    5,316    5,591              5.0 / 5                 23.4
    #
    # Cost is linear at roughly 106ms per pair, and there is no cheap cut: at top_n=25
    # only 2.8 of the 5 final results survive. Left at 50 -- the value the architecture
    # specifies -- deliberately rather than tuned down, because the number that should
    # decide this is Recall@5 at each setting and no golden set exists yet. Choosing it
    # on latency alone would be tuning against the one axis that is easy to measure.
    # Phase 7 sweeps it.
    rerank_top_n: int = 50
    final_top_k: int = 5

    # Below this reranker score, retrieval refuses and no LLM call is made.
    #
    # The specified default of 0.0 cannot work and is not used. The reranker's scores are
    # sigmoid probabilities in [0, 1] -- sentence-transformers applies the activation
    # itself -- so a floor of 0.0 refuses nothing at all.
    #
    # 0.05 is measured, over five questions this corpus can answer and five it cannot:
    #
    #     answerable        0.062  0.119  0.157  0.804  0.998
    #     unanswerable      0.000  0.002  0.011  0.152  0.674
    #
    # At 0.05 the three clearly-irrelevant questions are refused and nothing answerable
    # is falsely refused. It is a weak default and is labelled as one: the two
    # distributions **overlap** -- "What were Tesla's Q3 2025 delivery numbers?" scores
    # 0.674, above three of the five real questions -- so no threshold separates them
    # cleanly. A cross-encoder rates superficially topical text highly whether or not the
    # corpus contains the answer.
    #
    # This is the number Phase 6's `answerable: false` items exist to set properly, and
    # Phase 7 sweeps it against measured refusal accuracy. Ten hand-picked questions is
    # enough to reject 0.0; it is not enough to call 0.05 correct.
    score_floor: float = 0.05

    # Fields that change what a chunk *is*, as opposed to how chunks are searched.
    # Listed once, here, because two things depend on getting the set exactly right:
    # the `chunks.chunk_config` record, and the identity under which a chunking is
    # stored and later retrieved. Omitting a field that does affect chunking would let
    # two different chunkings collide under one identity.
    CHUNKING_FIELDS: ClassVar[tuple[str, ...]] = (
        "embedding_model",  # selects the tokenizer, so it moves every chunk boundary
        "chunk_tokens",
        "chunk_overlap_pct",
        "drop_references",
        "drop_figure_only_pages",
        "contextual_headers",  # changes embed_input, not content
    )

    # Bumped whenever the ingestion algorithm changes in a way that moves chunk
    # boundaries: a parser fix, a new heading heuristic, a different sentence splitter.
    #
    # Configuration alone cannot express "the code that produced this". Without this
    # field, a corpus re-ingested after a chunker fix is skipped as unchanged, and the
    # database quietly keeps serving chunks built by the old, broken logic.
    CHUNKING_PIPELINE_VERSION: ClassVar[int] = 1

    def chunking_params(self) -> dict[str, Any]:
        """The inputs that determine chunk boundaries and embed inputs.

        Persisted verbatim into `chunks.chunk_config` so a chunk can always be traced
        back to both the settings and the algorithm version that produced it.
        """
        params: dict[str, Any] = {field: getattr(self, field) for field in self.CHUNKING_FIELDS}
        params["pipeline_version"] = self.CHUNKING_PIPELINE_VERSION
        return params

    def chunking_sha256(self) -> str:
        """Stable identity of a chunking run, stored in `chunks.chunk_config_sha256`.

        Sorted keys and a canonical separator so the digest depends on the values alone
        and not on dict ordering; the Phase 7 chunk-size sweep selects chunk sets by
        this value, so it has to be reproducible across processes and machines.
        """
        canonical = json.dumps(self.chunking_params(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
