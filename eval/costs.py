"""Projecting what a run would cost, before it is allowed to spend anything.

Every part of this project that spends money reports what it will spend first. That is
not caution for its own sake: the golden set and the ablation matrix are the two things
in the repository that call a paid API in bulk, and an unbounded loop over 200 chunks is
the kind of mistake that is obvious afterwards and invisible before.

The token counts here are estimates and are labelled as such wherever they are printed.
An exact count needs either the provider's tokenizer or a metered API call, and paying to
find out what something costs defeats the purpose.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rag.generate.client import MODEL_RATES, cost_usd

# Characters per token for English prose under a modern BPE vocabulary. Measured
# ratios for this kind of text cluster around 3.6-4.0; the low end is used so the
# estimate errs towards over-reporting cost rather than under-reporting it.
CHARS_PER_TOKEN = 3.6

# How far the real number may land from the estimate. Printed alongside every figure so
# nobody treats a projection as a quote.
ESTIMATE_TOLERANCE = 0.25


def estimate_tokens(text: str) -> int:
    """Approximate token count for `text`. Deliberately an over-estimate."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass(frozen=True, slots=True)
class CallEstimate:
    """The projected cost of one kind of call, repeated `count` times."""

    label: str
    count: int
    input_tokens_each: int
    output_tokens_each: int

    @property
    def input_tokens(self) -> int:
        """Total projected input tokens."""
        return self.count * self.input_tokens_each

    @property
    def output_tokens(self) -> int:
        """Total projected output tokens."""
        return self.count * self.output_tokens_each

    def cost(self, model: str) -> float:
        """Projected dollars for this group of calls."""
        return cost_usd(model, self.input_tokens, self.output_tokens)


@dataclass(frozen=True, slots=True)
class RunEstimate:
    """What a whole run would cost, broken down by stage."""

    model: str
    stages: tuple[CallEstimate, ...]

    @property
    def calls(self) -> int:
        """Total number of API calls the run would make."""
        return sum(stage.count for stage in self.stages)

    @property
    def input_tokens(self) -> int:
        """Total projected input tokens."""
        return sum(stage.input_tokens for stage in self.stages)

    @property
    def output_tokens(self) -> int:
        """Total projected output tokens."""
        return sum(stage.output_tokens for stage in self.stages)

    @property
    def cost_usd(self) -> float:
        """Total projected dollars."""
        return cost_usd(self.model, self.input_tokens, self.output_tokens)

    def as_lines(self) -> list[str]:
        """A cost table, with the uncertainty stated rather than implied."""
        low = self.cost_usd * (1 - ESTIMATE_TOLERANCE)
        high = self.cost_usd * (1 + ESTIMATE_TOLERANCE)
        rates = MODEL_RATES[self.model]
        lines = [
            f"  model                  {self.model} "
            f"(${rates.input_per_mtok:.2f}/${rates.output_per_mtok:.2f} per Mtok)",
            "",
            f"  {'stage':<26}{'calls':>7}{'in tok':>11}{'out tok':>10}{'usd':>9}",
            "  " + "-" * 62,
        ]
        for stage in self.stages:
            lines.append(
                f"  {stage.label:<26}{stage.count:>7}{stage.input_tokens:>11,}"
                f"{stage.output_tokens:>10,}{stage.cost(self.model):>9.4f}"
            )
        lines.append("  " + "-" * 62)
        lines.append(
            f"  {'TOTAL':<26}{self.calls:>7}{self.input_tokens:>11,}"
            f"{self.output_tokens:>10,}{self.cost_usd:>9.4f}"
        )
        lines.append("")
        lines.append(f"  estimated ${self.cost_usd:.2f}, likely between ${low:.2f} and ${high:.2f}")
        lines.append(
            f"  token counts are approximated at {CHARS_PER_TOKEN} chars/token, not measured"
        )
        return lines


def estimate_run(model: str, stages: Sequence[CallEstimate]) -> RunEstimate:
    """Build a run estimate from its stages."""
    return RunEstimate(model=model, stages=tuple(stages))
