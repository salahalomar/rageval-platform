"""Pricing a full judged evaluation, without running one.

The retrieval half of this harness is free: local embeddings, a local cross-encoder, and
arithmetic. The answer half is not — it generates an answer per item and then judges it
three ways. This module prices that, using the context blocks retrieval *actually*
returns rather than an assumed average, so the figure tracks the corpus.

Nothing here calls a paid API. Retrieval runs for real, which costs CPU and no money.
"""

import argparse
import logging
import sys
from pathlib import Path

from eval.costs import CallEstimate, estimate_run
from eval.metrics.answer import estimate_judging
from eval.schema import read_items
from rag.config import GenerationConfig, RetrievalConfig
from rag.db import connect
from rag.generate.answer import Answer
from rag.generate.client import Usage
from rag.generate.prompt import assemble, system_prompt
from rag.retrieve import retrieve

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN = Path("eval/golden/v1.jsonl")
FALLBACK_GOLDEN = Path("eval/golden/hard_cases.jsonl")

# A typical grounded answer: three or four sentences with citation markers. Used to size
# the judge's input, since the real answers do not exist until generation is paid for.
TYPICAL_ANSWER = (
    "The authors report that the method improves over the baseline on the primary "
    "benchmark [1]. They attribute the gain to the training procedure described in the "
    "method section [2]. The effect is smaller on the held-out split [1][3]."
)
GENERATION_OUTPUT_TOKENS = 160


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--arms", type=int, default=1, help="how many arms will be judged")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)-7s %(message)s", stream=sys.stderr
    )

    golden = args.golden or (DEFAULT_GOLDEN if DEFAULT_GOLDEN.exists() else FALLBACK_GOLDEN)
    if not golden.exists():
        print(f"no golden set at {golden}")
        return 1

    items = read_items(golden)
    config = RetrievalConfig()
    generation = GenerationConfig()

    print(f"pricing a judged run over {len(items)} items x {args.arms} arm(s)")
    print(f"golden set: {golden}")
    print("(retrieval runs for real — CPU only, no API calls, no spend)")
    print()

    answers: list[Answer] = []
    generation_input = 0
    system = system_prompt(generation.prompt_version)

    with connect() as conn:
        for item in items:
            result = retrieve(item.question, config, conn)
            prompt = assemble(item.question, result.candidates)
            generation_input += len(system) + len(prompt)
            answers.append(
                Answer(
                    question=item.question,
                    text=TYPICAL_ANSWER,
                    candidates=result.candidates,
                    binding=None,
                    usage=Usage(),
                    retrieval=result,
                )
            )

    count = max(1, len(answers))
    generation_estimate = estimate_run(
        generation.model,
        [
            CallEstimate(
                "generate answer",
                len(answers) * args.arms,
                round(generation_input / count / 3.6),
                GENERATION_OUTPUT_TOKENS,
            )
        ],
    )
    judging_estimate = estimate_judging(
        answers * args.arms, [item.expected_answer for item in items] * args.arms, generation
    )

    print("generation")
    for line in generation_estimate.as_lines():
        print(line)
    print()
    print("judging")
    for line in judging_estimate.as_lines():
        print(line)
    print()
    total = generation_estimate.cost_usd + judging_estimate.cost_usd
    print(f"  TOTAL for one judged run: ${total:.2f}")
    print(f"  nightly, every night:     ${total * 30:.2f} per month")
    print()
    print("  Judgements are cached on (item, answer hash, judge model, prompt version),")
    print("  so a re-run after changing a retrieval parameter re-judges only the answers")
    print("  that actually changed. The figures above are for a cold cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
