# The golden set

Ground truth for every number this repository publishes. Treat these files as read-only:
per the engineering guide, a change to any `*.jsonl` here requires its own commit
touching nothing else, so that a metric can never move in the same commit as the
yardstick it was measured against.

## Files

| File | What it is | Written by |
|---|---|---|
| `candidates.jsonl` | Generated questions awaiting review | `eval.generate_golden` |
| `v1.jsonl` | **Verified** items. The golden set. | `eval.verify_cli`, and nothing else |
| `rejected.jsonl` | Candidates a reviewer turned down, kept for the record | `eval.verify_cli` |
| `hard_cases.jsonl` | 15 hand-written hard items | a human, by hand |

## Status

**Not yet built.** `hard_cases.jsonl` contains 15 drafts, none verified. `candidates.jsonl`
and `v1.jsonl` do not exist yet: generation requires paid API calls and none have been
made. Every number below marked *(pending)* is unmeasured, and the README must not quote
any of them until it is.

The full pipeline is implemented and priced. `python -m eval.generate_golden --dry-run`
samples, stratifies and reports the exact cost without calling anything.

## Protocol

**1. Stratified sampling — free.** 200 chunks, drawn across section types *and* papers.
Uniform sampling would produce a set shaped like the corpus, and the corpus is 21%
introductions; a set of introduction lookups measures whether the system can find a topic
sentence, which every configuration can, so the number would be flattering and would move
for no arm in the ablation. Method and results are deliberately over-weighted because that
is where specific, checkable claims live. At most 3 chunks per paper, so the set measures
the corpus rather than three papers.

Measured on the current corpus:

```
requested 200   returned 200   distinct papers 107   max per paper 3
  method 30.0%   results 30.0%   abstract 15.0%   limitations 10.0%
  introduction 10.0%   other 5.0%
```

Chunks below 120 tokens are excluded — stray headers and table fragments have nothing
in them to ask about.

**2. Question generation — paid.** One question and expected answer per chunk, instructed
to be standalone, to avoid "according to the passage" phrasing, and never to quote the
chunk verbatim. A question findable only because it shares a rare string with its source
tests string matching, not retrieval.

**3. The no-context filter — paid, and the step that makes this defensible.** Every
question is asked again with **zero retrieval context**. Anything the model answers from
parametric knowledge is discarded, because it does not test retrieval at all. The model's
parametric answer is stored on the candidate (`no_context_answer`) so a reviewer can audit
the filter's judgement rather than trust it.

Expected rejection rate 30–40% *(pending — this number is a prediction from the plan, not
a measurement, and must be replaced with the observed rate before the README cites it).*

**4. Deduplication — free.** Questions whose embeddings exceed cosine 0.90 are dropped,
using the same local embedding model as retrieval. Two phrasings of one question silently
double that question's weight in every metric.

**5. Human verification — free, and mandatory.** `eval.verify_cli` shows each candidate
with its source passage and its no-context answer; accept, edit, reject. Accepted items
are stamped `verified_by` and `verified_at`. **No automated step can write `v1.jsonl`** —
that is the entire meaning of "human-verified", and a script that could do it would make
the claim worthless. Budget roughly 90 minutes for 120 candidates. Resumable across
sittings.

**6. Hard cases — free, hand-written.** 15 items across four types the generated set will
not produce on its own:

| Type | Count | What it probes |
|---|---:|---|
| `numeric` | 4 | The answer is a figure in a results table |
| `multi_hop` | 4 | Requires combining two chunks |
| `unanswerable` | 4 | Plausible but genuinely absent — correct behaviour is refusal |
| `distractor` | 3 | Wording lexically matches the wrong paper |

The unanswerable items are the ones most portfolio projects lack entirely, and they are
what expose hallucination. One of them (`h-009`) is kept deliberately even though it is
known to fail: it scores 0.674 on the reranker, above three genuinely answerable
questions, so the current score floor answers it rather than refusing. It is in the set
*because* it is the case the floor gets wrong.

## Cost

Priced from the actual corpus text, not estimated from an average:

```
stage                       calls     in tok   out tok      usd
generate question             200    178,000    24,000   0.2980
no-context filter             200     33,800    16,000   0.1138
TOTAL                         400    211,800    40,000   0.4118
```

Roughly **$0.41**, likely between $0.31 and $0.51. Token counts are approximated at 3.6
characters per token rather than measured.

## Limitations, stated rather than buried

- **The questions are model-written.** They inherit whatever the generating model finds
  salient, and will under-represent questions a human would think to ask but a model would
  not. Human verification catches bad questions; it does not fix a skewed distribution of
  question *kinds*.
- **The no-context filter uses the same model family as the generator and, at present, as
  the judge.** Self-preference bias in LLM judges is well documented. The judge-agreement
  work in Phase 7 exists to quantify this, and until it reports a κ the answer metrics
  should be read as provisional.
- **Ground truth is single-chunk for generated items.** A question generated from one
  chunk names that chunk as relevant, but another chunk may support the answer equally
  well. Recall@k is therefore a *lower* bound: the system may retrieve a genuinely correct
  chunk and be scored wrong for it. The hand-written multi-hop items are the only ones
  with genuinely multi-chunk ground truth.
- **The set is tied to one chunking.** Chunk ids belong to a specific
  `chunk_config_sha256`. Re-chunking the corpus invalidates every id here, which is why
  the corpus is pinned in `infra/corpus/` and the chunking identity is recorded.
- **80 items is small.** A difference of two or three items is inside the noise, so arms
  separated by a point or two of recall should not be reported as separated.

## Versioning

`v1.jsonl` is immutable once verification finishes. A change means `v2.jsonl`, with `v1`
kept, so that results referencing v1 stay meaningful. Never regenerate the golden set
after tuning a parameter — that inverts the entire point of having one.
