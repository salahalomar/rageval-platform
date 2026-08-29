# rag-eval-platform

Retrieval-augmented generation over arXiv ML papers, built so that its **evaluation
harness is the deliverable** and the chat interface is only what makes it demoable. The
repository measures its own retrieval and answer quality on every commit and publishes
an ablation table — including the arms that lost.

**Status: Phase 7 — the evaluation harness.** Retrieval metrics, the ablation runner, a
regression gate and a judge that is built and priced but unrun. The table below is a
**harness smoke test, not a quality result** — read the caveat under it before quoting
any number from it.

---

## Quickstart

```bash
cp .env.example .env
make dev        # Postgres 16 + pgvector, and the API, both healthy
make migrate    # apply forward-only SQL migrations
curl -s localhost:8000/health | python3 -m json.tool
```

Then build the corpus (~10 minutes: arXiv asks for one request every three seconds, and
PDFs are cached on disk so a re-run resumes rather than restarts):

```bash
uv run rag ingest --limit 150
```

```bash
uv run rag stats --sample 3
```

Then embed and search (~6 minutes on CPU for 6,386 chunks; re-running is a no-op):

```bash
uv run rag embed
```

```bash
uv run rag search "What is the effect of learning rate warmup on transformer training?" --compare
```

`--compare` runs all three retrieval modes over the same query side by side. Any one of
them is reachable on its own with `--mode {rrf,dense_only,lexical_only}`.

Then the front end, on <http://localhost:5173>:

```bash
make web
```

It needs no model credential — see [The front end](#the-front-end).

The corpus is pinned in [`infra/corpus/`](infra/corpus/) — rebuild the exact same one with
`rag ingest --ids-file infra/corpus/cs-lg-cs-cl-150.txt`. A category search returns "the
most recent 150", which is a different set of papers every day; the golden set will bind
questions to chunk ids, so the corpus behind them has to be nameable.

`make help` lists the rest.

## How ingestion works

| Stage | What it does | Why it is not the obvious thing |
|---|---|---|
| `arxiv.py` | Metadata + PDF, cached by id | One request per 3s, resumable — a corpus build is paid once, offline |
| `parse.py` | Text, page numbers, char offsets | Reconstructs **two-column reading order**; PyMuPDF's default block order interleaves the columns into alternating half-sentences |
| `sections.py` | `3 Method > 3.2 Training` | Requires **both** typography and numbering; either signal alone produces false headings, and a false boundary permanently severs a table from its caption |
| `chunk.py` | 512 tokens, 15% overlap | Never crosses a section, never splits a sentence, and counts tokens with **bge-small's own tokenizer** so chunks cannot silently overflow the model's window |

Two behaviours worth knowing about:

- **`chunk_tokens` budgets `embed_input`, not content.** With contextual headers on, the
  header is charged against the same budget, so headers cost ~6% of content at the
  default chunk size. The alternative — budgeting content alone — pushes the embedded
  string past the model's 512-token window and truncates every long chunk with no error
  raised. The Phase 7 headers arm should be run below the model window to isolate it.
- **Chunkings are keyed, not overwritten.** `chunk_config_sha256` identifies the settings
  that produced a chunk, so the Phase 7 chunk-size sweep can hold several chunkings of
  one corpus side by side. Re-ingesting is a no-op only when the PDF *and* the chunking
  config are both unchanged.

## Dense retrieval

Embeddings are `BAAI/bge-small-en-v1.5` run locally on CPU. Queries get bge's instruction
prefix and passages do not — the asymmetry is worth several points of recall and is
invisible when wrong, so a test asserts it rather than a comment claiming it.

`ef_search` is chosen from measurement, not from the pgvector default. Against exact-scan
ground truth over 50 queries on the 6,386-chunk corpus (`rag bench-index`):

| ef_search | recall@10 | recall@50 | p50 ms | p95 ms |
|---:|---:|---:|---:|---:|
| 40 | 0.830 | 0.860 | 6.9 | 11.6 |
| 100 | 0.952 | 0.921 | 5.1 | 8.2 |
| 200 | 0.980 | 0.973 | 6.7 | 12.4 |
| **400** | **1.000** | **1.000** | **7.1** | **8.2** |
| 800 | 1.000 | 1.000 | 7.4 | 8.0 |

Index: 12.5 MiB over 6,386 vectors, 0.5s to build. Warm end-to-end search (query
embedding + retrieval, model resident) is **p50 33ms / p95 38ms**.

Two honest notes on that table:

- **The index barely earns its place at this scale.** An exhaustive scan of the same
  corpus runs at p50 8.0ms — HNSW at ef_search=400 is 7.1ms. The index is here because it
  is the thing that keeps working as the corpus grows, not because it is winning today.
- **400 is tuned to *this* corpus.** It buys exact agreement with a full scan for about
  2ms, which matters because approximation error is indistinguishable from retrieval error
  in a published metric: at ef_search=100 roughly one true neighbour in twenty is missed,
  and every ablation arm would silently carry that deficit. It will not stay exact at ten
  times the rows and must be re-measured.

Retrieval quality itself is not claimed here. There is no golden set yet, so there is no
Recall@k against ground truth — only Phase 6 and 7 can supply that, and until they do the
right number of quality claims to make is zero.

## Lexical retrieval and fusion

**The lexical arm is `ts_rank_cd`, and it is not BM25.** `ts_rank_cd` is a
coverage-density ranking: it scores by how many query lexemes a chunk contains and how
tightly they cluster, with no document-frequency term and no length-saturation curve.
That is a different function from Okapi BM25, not an implementation of it. Phase 6 adds a
real BM25 arm with `bm25s` and the ablation table reports both.

Query lexemes are OR-ed, not conjoined. `plainto_tsquery` would require all six content
words of a typical question inside one 512-token chunk, which returns nothing on this
corpus; lexical search is here as the recall complement to an arm that already handles
semantic similarity.

Measured over 8 questions on the 6,386-chunk corpus, warm:

| mode | p50 ms | p95 ms |
|---|---:|---:|
| `dense_only` | 13.6 | 16.0 |
| `lexical_only` | 17.2 | 51.3 |
| `rrf` | 32.1 | 66.2 |

**The arms overlap on a mean of 11.4 of their top 50** (min 6, max 21) — so roughly
three-quarters of each list is unique to that arm. That is the measured case for fusing
them at all. It is *not* a quality claim: complementarity is necessary for fusion to help,
not sufficient, and whether it actually helps is a Recall@k question that only Phase 7 can
answer.

**Why lexical costs more than dense, and what it implies.** The OR query matches a mean of
43% of the corpus (min 21%, max 72%), because stemmed terms like `train`, `rate` and
`learn` appear nearly everywhere. The GIN index finds those matches in 0.4ms; the expense
is computing `ts_rank_cd` for all ~3,500 of them and then top-N sorting. Suppressing
common terms is precisely what an inverse-document-frequency weight does — which makes
this a *measured* reason to want the BM25 arm in Phase 6 rather than a theoretical one.

RRF's p50 is close to the sum of the two arms because they run sequentially. Running them
concurrently is a Phase 9 concern, not a correctness one.

## Cross-encoder reranking

A bi-encoder embeds query and passage independently — neither vector is computed knowing
the other exists. A cross-encoder concatenates them and attends across the pair, which is
why it reorders results a bi-encoder got wrong, and why it cannot be indexed: every pair
needs its own forward pass. That is the entire trade, and this is what it costs here.

| `rerank_top_n` | p50 ms | p95 ms | top-5 kept vs n=50 | mean rank movement |
|---:|---:|---:|---:|---:|
| 5 | 566 | 602 | 0.8 / 5 | 1.1 |
| 10 | 1,132 | 1,144 | 1.0 / 5 | 2.8 |
| 25 | 2,538 | 2,775 | 2.8 / 5 | 10.1 |
| **50** | **5,316** | **5,591** | 5.0 / 5 | 23.4 |

**Reranking is by far the most expensive stage in the system** — roughly 106ms per pair
on CPU, against 32ms for the *entire* hybrid retrieval that feeds it. And there is no
cheap cut: at `top_n=25` only 2.8 of the 5 final results survive.

It stays at 50 anyway. The number that should decide this is Recall@5 at each setting,
and no golden set exists yet; trimming it now would be tuning against the one axis that
happens to be easy to measure. Phase 7 sweeps it and the table will show what the latency
bought.

Apple's MPS backend runs the same model at 56ms/pair — 1.7× faster — and is deliberately
not used. The deployment target is Linux CPU, so a latency figure measured on an Apple GPU
would describe hardware this system never runs on, and mixing MPS locally with CPU in CI
puts cross-machine determinism at risk for no benefit to the published numbers.

### The reranker sees about 80% of each chunk

| | tokenizer | vocab | p50 tokens per chunk |
|---|---|---|---:|
| embedder | BERT WordPiece | 30,522 | 460 |
| reranker | XLM-RoBERTa | 250,002 | 511 |

`bge-reranker-base` is multilingual and tokenises English into **1.19× more tokens** than
`bge-small` does. Chunks sized to fill the embedder's 512-token window therefore overflow
the reranker's, and **48% of corpus chunks — 78% of the pairs in the query above — lose
their tail**.

This is the same class of failure the chunker was built to prevent, on a stage that did
not exist when the budget was set. It is reported per query (`truncated_pairs`) rather
than hidden. It is not fixed by guesswork: shrinking `chunk_tokens` to about 430 would fit
both windows, and whether that trade is worth making is a Recall@5 question for Phase 7,
where it becomes an ablation arm rather than a hunch.

### Refusal

Below `score_floor` the result comes back empty with `reason="below_score_floor"`, and the
generator declines to call the model at all — an out-of-corpus question costs **zero
tokens and zero dollars**. Refusal is a measured behaviour, not an error path.

**The floor does not work as specified, and this is the most useful thing measured so
far.** The plan sets `score_floor = 0.0` on the assumption that the reranker emits logits
centred on zero. It does not: `sentence_transformers.CrossEncoder` applies `Sigmoid()` to
the single-label head, so scores are probabilities in [0, 1] and a floor of 0.0 refuses
nothing. Worse, the two distributions overlap:

```
top rerank score, questions the corpus CAN answer     0.062  0.119  0.157  0.804  0.998
top rerank score, questions it CANNOT                 0.000  0.002  0.011  0.152  0.674
```

At `score_floor = 0.05` the clearly-irrelevant questions refuse and nothing answerable is
falsely refused — but *"What were Tesla's Q3 2025 delivery numbers?"* scores **0.674**,
above three of the five real questions, and is answered rather than refused. No threshold
separates them.

Following that false positive to its cause is instructive: the Tesla question matches a
mangled results table — `(5+3) R INDOTOD … T 2.63/4.06 avg R Multimodal` — that
superficially reads as "numbers". So the miss is partly a **chunk-quality** problem, not
purely a threshold one, which points back at the table extraction rather than at the
reranker.

0.05 is therefore labelled provisional. Ten hand-picked questions are enough to reject
0.0; they are not enough to call 0.05 correct. Phase 6's `answerable: false` items exist
to set this properly, and Phase 7 sweeps it against measured refusal accuracy.

One consequence stated rather than discovered: **an arm with reranking disabled cannot
refuse on score**, because cosine similarity and `ts_rank_cd` are on scales no single
floor value can serve.

## The ablation table

> **These numbers are not a quality result and must not be quoted as one.** The verified
> golden set does not exist yet — building it needs paid API calls that have not been
> made — so this runs against **11 unverified hand-written drafts**. Eleven items means
> one question changing answer moves Recall@5 by nine points. What the table demonstrates
> is that the harness runs end to end, deterministically, on the real corpus.

<!-- ABLATION-TABLE:START -->

| Arm           | R@1 | R@5 | MRR@10 | nDCG@10 | Refusal acc. | p95 ms | n |
|---------------|-----|-----|--------|---------|--------------|--------|---|
| lexical-only  | —   | —   | 0.000  | 0.000   | 0.000        | 193    | 0 |
| dense-small   | —   | —   | 0.000  | 0.000   | 0.000        | 35     | 0 |
| hybrid-rrf    | —   | —   | 0.000  | 0.000   | 0.000        | 239    | 0 |
| hybrid-rerank | —   | —   | 0.000  | 0.000   | 0.250        | 27737  | 0 |
| rrf-k20       | —   | —   | 0.000  | 0.000   | 0.250        | 27688  | 0 |
| rrf-k120      | —   | —   | 0.000  | 0.000   | 0.250        | 27691  | 0 |

<!-- ABLATION-TABLE:END -->

The table is generated from the committed result JSONs and injected between those
markers by `make report`. It is never hand-edited, so every number traces to a file
recording the configuration, the git commit and the golden set that produced it.

**Two arms that did nothing, reported in place.** `rrf-k20` and `rrf-k120` are byte-for-byte
identical to `hybrid-rerank` on every metric. That is not a bug: RRF's `k` reorders the
fused list, but the cross-encoder then rescores the top 50 from scratch, so the fusion
order it started from is almost entirely erased. **The RRF k sweep is a no-op once
reranking is enabled** — worth knowing before spending effort tuning it, and exactly the
kind of arm a table that only reported winners would have quietly dropped.

**Reranking is where the quality is, and where the latency is.** R@1 more than doubles
(0.258 → 0.545) and R@5 goes 0.621 → 0.848 — while p95 goes from 343ms to 7.7 seconds, a
22× cost. The ablation table exists to show both columns.

**Refusal accuracy is 0.25**: one of four unanswerable items refused, consistent with the
overlapping score distributions measured in Phase 5.

### Determinism, verified

Two runs on one commit produce identical metrics, identical rankings, and identical
resolved ground truth. Checked by running the matrix twice into separate directories and
diffing; the JSON files differ only in `started_at` and `duration_s`, which are not
metrics.

### Ground truth survives re-chunking

The golden set names relevant chunks by id, and the chunk-size sweep produces entirely
different ids — so scoring the 256-token arm against 512-token ids would report zero
recall for every question, an artefact of the identifier scheme rather than a result.
`eval/relevance.py` resolves each golden id once to a `(paper_id, char_span)` claim about
where the answer lives, then re-resolves it against whatever chunking an arm actually
uses. Without that, three of the ten arms are unrunnable.

### What the judge costs

Answer metrics need a model and have not been run. `make cost` prices them from the
context retrieval actually returns:

```
generation   60 calls   181,260 in    9,600 out   $0.23
judging     180 calls   369,000 in   34,800 out   $0.54
TOTAL for one judged run: $0.77   →  ~$23/month nightly
```

At the full 80-item golden set that becomes roughly **$4 per run, $120 a month** running
nightly. Judgements are cached on `(item, answer hash, judge model, prompt version)`, so
changing a retrieval parameter re-judges only the answers that actually changed.

**No faithfulness number appears anywhere in this README, and none should until Cohen's κ
does.** The judge shares a model family with the generator; self-preference bias in LLM
judges is well documented. `make kappa` compares the judge against hand labels a human
writes — and nothing can generate those labels, because a judge validated against its own
output is a tautology.

## The golden set

Ground truth for every number this repository will publish. The protocol, the
stratification, the cost and an honest limitations section live in
[`eval/golden/README.md`](eval/golden/README.md).

**Status: built and priced, not yet run.** Sampling, deduplication, human verification and
the 15 hand-written hard cases cost nothing and are done. The two steps that call a paid
API — question generation and the no-context filter — have not been executed, so
`v1.jsonl` does not exist and **no retrieval or answer quality number is claimed anywhere
in this README.**

```bash
make golden-dry-run    # samples, stratifies and prices the run. Calls nothing.
```

```
requested 200   returned 200   distinct papers 107   max per paper 3
  method 30.0%   results 30.0%   abstract 15.0%   limitations 10.0%

stage                       calls     in tok   out tok      usd
generate question             200    178,000    24,000   0.2980
no-context filter             200     33,800    16,000   0.1138
TOTAL                         400    211,800    40,000   0.4118
```

Three things about the design are worth stating:

- **The no-context filter is what makes an LLM-generated set defensible.** Every question
  is asked again with zero retrieval context, and anything the model answers from
  parametric knowledge is discarded — it tests what the model knows, not what the system
  retrieved. The model's parametric answer is stored on the candidate so a reviewer can
  audit the filter rather than trust it.
- **No automated step can write `v1.jsonl`.** Only `eval.verify_cli` does, one keypress at
  a time, stamping `verified_by` and `verified_at`. A script that could write it would make
  "human-verified" worthless.
- **Sampling is stratified and seeded.** Uniform sampling would produce a set shaped like
  the corpus, which is 21% introductions — questions every configuration answers, so the
  headline number would be flattering and would move for no arm in the ablation.

One hard case is deliberately kept although it is known to fail. `h-009` — *"What were
Tesla's Q3 2025 delivery numbers?"* — scores 0.674 on the reranker, above three genuinely
answerable questions, so the current score floor answers it instead of refusing. It is in
the set precisely because it is the case the floor gets wrong.

## Generation

Answers are produced by `claude-haiku-4-5` behind an `LLMClient` protocol. The protocol is
the reason the whole test suite — including every refusal, retry and cost assertion — runs
with no key, no network and no spend against a scripted fake, while production is one
small class behind the same interface.

- **No answer without a citation.** Every factual sentence must end with `[n]` markers.
  The answer is parsed, each marker resolved against the blocks actually sent, and any
  unsupported sentence triggers **one** correction pass quoting the offending sentences.
  If it still fails, the answer is returned flagged `uncited=true` rather than presented
  as sourced. Out-of-range markers (`[7]` when five blocks were sent) are recorded as
  invalid, because an answer citing a block that does not exist looks sourced and is not.
- **Prompts are versioned files**, not inline strings — a prompt change moves every answer
  metric the way a chunk-size change moves every retrieval metric, so it has to be
  nameable. `prompt_version` is recorded on every result.
- **Cost is computed in one place** from a published rates table, and an unpriced model
  raises rather than silently costing zero.
- **`GenerationConfig` is separate from `RetrievalConfig`.** The retrieval model is the
  ablation axis for retrieval; folding the generation model and prompt version into it
  would make every retrieval arm carry fields that had nothing to do with it.

### Running it for free

Generation is the only part of this system that costs money, and it does not have to.
`LLMClient` has two implementations, selected by `GenerationConfig.provider`:

| Provider | Covers | Cost |
|---|---|---|
| `anthropic` | Anthropic first-party | paid |
| `openai_compatible` | Groq, Cerebras, OpenRouter, **and a local Ollama** | free tiers / free |

Those four speak the same wire protocol, so they are one class differing only by
`base_url` — **a local model is not a special case in the architecture, it is an
endpoint.** Groq's free tier needs no card and allows 30 requests/minute and 1,000/day,
which covers a 400-call golden-set build in one sitting.

Two things this buys beyond the money:

- **`temperature` works again.** The Anthropic SDK removed sampling controls entirely —
  `temperature`, `top_p` and `top_k` are absent from `messages.create` in 1.0.0 and
  current models reject them with a 400. OpenAI-compatible endpoints still honour it, so
  routing through one restores the determinism knob the plan asked for and the vendor SDK
  took away. The field is honoured by one provider and ignored by the other, which is
  documented on the field rather than left to be discovered.
- **The judge can stop sharing a family with the generator.** ENGINEERING.md requires that
  a judge from the generator's own family be declared and its bias quantified. Generating
  with one provider and judging with another removes that bias rather than measuring it —
  a methodological improvement that happens to also be free.

Three things it costs, stated rather than buried:

- **Free tiers rotate model versions without notice**, which collides with pinned versions.
  Partly mitigated: `Completion.model` records what the endpoint *reported serving*, not
  what was requested, so a silent substitution shows up in the result record. There is a
  test for exactly that.
- **1,000 requests/day is tight** for a nightly ablation at 80 items across four arms. The
  judgement cache absorbs most of it; a cold run would not fit comfortably.
- **Free tiers generally train on submitted data.** Irrelevant here — the corpus is public
  arXiv papers — but worth saying rather than not saying.

Zero-rated models appear in `MODEL_RATES` at $0.00. That is the honest figure rather than a
placeholder: a free-tier request costs nothing in money, and what it *does* cost is a
request against a daily quota, which is a constraint on throughput rather than budget.

## The front end

React 18 + TypeScript + Vite, at `apps/web`. Two pages: a retrieval demo and the
ablation table.

**It works with no API key and spends nothing.** The most persuasive thing this system
can show is a property of retrieval alone — turn the cross-encoder off and the same
question reorders in front of you — so the demo's primary action is `POST /retrieve`,
which runs the full retrieval stack and returns ranked chunks without generating
anything. On the corpus above, that toggle is visible and large:

| Cross-encoder | Total latency | Top result |
|---|---|---|
| on | 13,383 ms | Task Specialization Fine-Tuning… (rerank 0.286) |
| off | 175 ms | Understanding Curriculum Learning… (rrf 0.0300) |

Generation is a second button, and it is disabled rather than broken when no credential
exists: `GET /health` reports `generation_configured`, and the page says so in words. A
button that always fails teaches a visitor the system is broken rather than unconfigured.

Three things are deliberate:

- **The API types are generated, not written.** `npm run gen:api` (or `make api-types`)
  regenerates `src/lib/schema.d.ts` from the running server's OpenAPI document. A field
  renamed in Python fails the TypeScript build rather than failing in a browser in front
  of somebody.
- **Nothing restates the library's defaults.** The config panel initialises from
  `GET /config/default`, which returns `RetrievalConfig()` itself. The point of one frozen
  config object is that there is one answer to what "default" means.
- **The `/eval` page reads the committed result JSON directly** out of `eval/results/`,
  not a copy. Row order comes from `eval/results/index.json`, which `eval/report.py`
  writes alongside the README table — so the page and the README cannot disagree about
  which arm goes where, and neither sorts by score.

Highlighting in a source block marks **query terms, not the model's attribution**. The
system does not record which characters a claim came from, and highlighting a span the
generator never pointed at would be a convincing flourish standing in for a measurement.

If port 5173 is taken, set `WEB_PORT`. Vite runs with `strictPort` on purpose: without
it, a second Vite project binds the wildcard address, loses the race to `localhost`, and
silently serves the neighbouring app with no error anywhere.

```bash
WEB_PORT=5175 make web
```

The front end runs on the host rather than in a container — Vite needs nothing a
container provides. `docker compose --profile web up web` exists for a machine with no
Node, and is never built by `make dev`.

`make web-check` lints and type-checks it. `make web-test` runs the Playwright smoke
tests, which need a live stack with an ingested corpus and so are deliberately not part of
the pull-request gate.

## Layout

| Path | What lives here |
|---|---|
| `packages/rag/` | The library — ingestion, indexing, retrieval, generation. Imports nothing from `apps/`. |
| `apps/api/` | FastAPI. A transport layer with no retrieval logic in it. |
| `apps/web/` | React + Vite. Types generated from the API's OpenAPI document. |
| `infra/migrations/` | Numbered, forward-only SQL. Never edited once applied. |
| `eval/` | Golden set, metrics, runner, ablation matrix. *(Phases 6–7)* |

The rule that matters: the evaluation harness calls the same `rag` code path the API
calls. `tests/test_layering.py` enforces the direction of that dependency, and
`rag.retrieve.retrieve()` is the single door — dispatch is on `RetrievalConfig.fusion`
alone, so an ablation arm is a different configuration and never a different code path.

[`ENGINEERING.md`](ENGINEERING.md) states the principles this repo is built under —
what gets measured, what counts as honest naming, and what is not allowed to drift.

## Phase status

- [x] **0** Scaffold — workspace, compose, migrations, health, CI
- [x] **1** Ingestion — arXiv fetch, PDF parse, section-aware chunking
- [x] **2** Dense retrieval — bge-small embeddings, HNSW
- [x] **3** Lexical + RRF fusion
- [x] **4** Cross-encoder reranking
- [x] **5** Generation, citations, refusal
- [x] **6** Golden set
- [x] **7** Eval harness and ablation table
- [x] **8** Frontend — retrieval demo, live config panel, ablation table
- [ ] **9** Ship

The ablation table replaces this section once Phase 7 lands.
