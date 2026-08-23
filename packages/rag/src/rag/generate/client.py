"""The LLM boundary: one protocol, one Anthropic implementation, one cost table.

Why a Protocol rather than calling the SDK directly from the answer path: generation is
the only stage that costs money and the only stage that needs a network. Putting a seam
here means the entire test suite, and the retrieval half of the eval harness, run against
a fake with no key, no network and no spend -- while the production path is a single
small class behind the same interface. Swapping provider is then a file, not a refactor.

Cost is computed in exactly one place. Rates duplicated across call sites drift, and a
cost-per-query column in the ablation table is worthless if the arms were priced by
different copies of the same number.
"""

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rag.config import GenerationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelRates:
    """US dollars per million tokens, as published."""

    input_per_mtok: float
    output_per_mtok: float


# The single source of truth for pricing. Anthropic first-party API rates.
MODEL_RATES: dict[str, ModelRates] = {
    "claude-haiku-4-5": ModelRates(input_per_mtok=1.00, output_per_mtok=5.00),
    "claude-sonnet-5": ModelRates(input_per_mtok=3.00, output_per_mtok=15.00),
    "claude-opus-5": ModelRates(input_per_mtok=5.00, output_per_mtok=25.00),
}


class UnknownModelRateError(KeyError):
    """A model was used whose price is not in the rates table."""


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call.

    Raises rather than defaulting to zero for an unpriced model. A silent zero would
    make the cost column in the ablation table quietly wrong for exactly the arm someone
    added without thinking about its price.
    """
    try:
        rates = MODEL_RATES[model]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_RATES))
        raise UnknownModelRateError(
            f"no published rate for {model!r}; add it to rag.generate.client.MODEL_RATES "
            f"(known: {known})"
        ) from exc
    return (input_tokens * rates.input_per_mtok + output_tokens * rates.output_per_mtok) / 1_000_000


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response, with everything needed to log and price it."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        """Dollar cost of this call, from the single rates table."""
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


@dataclass(slots=True)
class Usage:
    """Running token and cost totals across the calls that produced one answer.

    An answer can take more than one call -- the citation-correction retry is a second
    one -- and the log has to record what the *answer* cost, not what its last attempt
    cost.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    models: list[str] = field(default_factory=list)

    def add(self, completion: Completion) -> None:
        """Accumulate one call into the totals."""
        self.calls += 1
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens
        self.cost_usd += completion.cost_usd
        if completion.model not in self.models:
            self.models.append(completion.model)


class LLMClient(Protocol):
    """The only thing the generation path is allowed to know about a model provider."""

    def complete(self, system: str, prompt: str, config: GenerationConfig) -> Completion:
        """Return a single completion for `prompt` under `system`."""
        ...

    def stream(
        self, system: str, prompt: str, config: GenerationConfig
    ) -> Iterator[str | Completion]:
        """Yield text deltas as they arrive, then one final `Completion`.

        The final object carries the usage totals, which are only known once the stream
        ends -- so a caller that needs cost must consume the iterator to completion.
        """
        ...


class AnthropicClient:
    """Anthropic implementation. The only module in the library that spends money."""

    def __init__(self, api_key: str | None = None, max_retries: int = 3) -> None:
        """Construct the SDK client.

        Retries are delegated to the SDK rather than reimplemented. It already backs off
        on 429, 408, 409 and 5xx, and hand-rolling that loop on top would double the
        wait on every transient failure while adding a second thing to get wrong.
        """
        import anthropic

        self._anthropic = anthropic
        # A bare constructor resolves credentials from the environment or a stored
        # profile, so an unset ANTHROPIC_API_KEY is not the same as having none.
        self._client = (
            anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
            if api_key
            else anthropic.Anthropic(max_retries=max_retries)
        )

    def complete(self, system: str, prompt: str, config: GenerationConfig) -> Completion:
        """Return a single completion, translating SDK errors into readable ones."""
        started = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except self._anthropic.AuthenticationError as exc:
            raise GenerationError("Anthropic rejected the credentials") from exc
        except self._anthropic.RateLimitError as exc:
            raise GenerationError("rate limited after the SDK exhausted its retries") from exc
        except self._anthropic.APIStatusError as exc:
            raise GenerationError(f"Anthropic returned HTTP {exc.status_code}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise GenerationError("could not reach the Anthropic API") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def stream(
        self, system: str, prompt: str, config: GenerationConfig
    ) -> Iterator[str | Completion]:
        """Yield text deltas, then the final completion carrying usage."""
        started = time.perf_counter()
        with self._client.messages.stream(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()

        yield Completion(
            text="".join(block.text for block in final.content if block.type == "text"),
            model=final.model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            stop_reason=final.stop_reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


class GenerationError(RuntimeError):
    """The model could not be reached or refused to respond."""


@dataclass(slots=True)
class ScriptedLLMClient:
    """A fake client that returns prepared responses. Never touches the network.

    Lives in the library rather than in the test tree because the eval harness needs it
    too: the retrieval half of the ablation matrix has no reason to spend money, and a
    smoke run in CI must be free and deterministic.
    """

    responses: Sequence[str]
    model: str = "claude-haiku-4-5"
    input_tokens: int = 100
    output_tokens: int = 20
    prompts: list[tuple[str, str]] = field(default_factory=list)
    _index: int = 0

    def complete(self, system: str, prompt: str, config: GenerationConfig) -> Completion:
        """Return the next scripted response, recording what it was asked."""
        self.prompts.append((system, prompt))
        text = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        return Completion(
            text=text,
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            stop_reason="end_turn",
        )

    def stream(
        self, system: str, prompt: str, config: GenerationConfig
    ) -> Iterator[str | Completion]:
        """Yield the scripted response word by word, then its completion."""
        completion = self.complete(system, prompt, config)
        for word in completion.text.split(" "):
            yield word + " "
        yield completion
