"""The OpenAI-compatible client and the provider factory.

No network. The SDK is replaced by a stub that records what it was asked for, because
what matters here is the shape of the request — the system prompt in the right place,
usage requested on streams, the reported model recorded rather than the requested one.
Those are the details that fail silently against a real endpoint.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from rag.config import GenerationConfig
from rag.generate.client import (
    MODEL_RATES,
    OPENAI_COMPATIBLE_ENDPOINTS,
    Completion,
    GenerationError,
    OpenAICompatibleClient,
    client_for,
    cost_usd,
)

CONFIG = GenerationConfig(
    provider="openai_compatible",
    model="llama-3.3-70b-versatile",
    base_url=OPENAI_COMPATIBLE_ENDPOINTS["groq"],
    max_tokens=256,
    temperature=0.0,
)


# --- a stub standing in for the SDK -----------------------------------------


@dataclass
class StubUsage:
    prompt_tokens: int = 120
    completion_tokens: int = 30


@dataclass
class StubMessage:
    content: str | None


@dataclass
class StubChoice:
    message: StubMessage
    finish_reason: str | None = "stop"


@dataclass
class StubResponse:
    choices: list[StubChoice]
    model: str = "llama-3.3-70b-versatile"
    usage: StubUsage | None = field(default_factory=StubUsage)


@dataclass
class StubDelta:
    content: str | None


@dataclass
class StubStreamChoice:
    delta: StubDelta
    finish_reason: str | None = None


@dataclass
class StubChunk:
    choices: list[StubStreamChoice]
    model: str = "llama-3.3-70b-versatile"
    usage: StubUsage | None = None


class StubCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    StubChunk([StubStreamChoice(StubDelta("Warmup "))]),
                    StubChunk([StubStreamChoice(StubDelta("stabilises [1]."), "stop")]),
                    StubChunk([], usage=StubUsage()),
                ]
            )
        return StubResponse([StubChoice(StubMessage("Warmup stabilises training [1]."))])


def stub_client() -> tuple[OpenAICompatibleClient, StubCompletions]:
    """A client whose SDK has been replaced, so nothing leaves the process."""
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    completions = StubCompletions()

    class Chat:
        pass

    chat = Chat()
    chat.completions = completions  # type: ignore[attr-defined]

    class Inner:
        pass

    inner = Inner()
    inner.chat = chat  # type: ignore[attr-defined]

    client._base_url = "https://api.groq.com/openai/v1"
    client._client = inner  # type: ignore[assignment]
    return client, completions


# --- request shape ----------------------------------------------------------


def test_the_system_prompt_is_sent_as_the_first_message() -> None:
    # Anthropic takes system as a top-level argument; these endpoints take it as
    # messages[0]. Sending it the wrong way is silently accepted by some gateways and
    # ignored, producing an uninstructed model and no error at all.
    client, completions = stub_client()
    client.complete("SYSTEM TEXT", "USER TEXT", CONFIG)
    messages = completions.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "SYSTEM TEXT"}
    assert messages[1] == {"role": "user", "content": "USER TEXT"}


def test_temperature_is_sent() -> None:
    # The knob the Anthropic SDK removed. These endpoints still honour it, which is what
    # restores the determinism the plan asked for.
    client, completions = stub_client()
    client.complete("s", "p", CONFIG)
    assert completions.calls[0]["temperature"] == 0.0


def test_the_configured_model_and_token_cap_are_sent() -> None:
    client, completions = stub_client()
    client.complete("s", "p", CONFIG)
    assert completions.calls[0]["model"] == "llama-3.3-70b-versatile"
    assert completions.calls[0]["max_tokens"] == 256


def test_streaming_requests_usage_explicitly() -> None:
    # Without include_usage these endpoints omit token counts from a streamed response
    # entirely, and the cost column would silently report zero for every streamed answer.
    client, completions = stub_client()
    list(client.stream("s", "p", CONFIG))
    assert completions.calls[0]["stream"] is True
    assert completions.calls[0]["stream_options"] == {"include_usage": True}


# --- response handling ------------------------------------------------------


def test_a_completion_carries_tokens_and_stop_reason() -> None:
    client, _ = stub_client()
    completion = client.complete("s", "p", CONFIG)
    assert completion.text == "Warmup stabilises training [1]."
    assert completion.input_tokens == 120
    assert completion.output_tokens == 30
    assert completion.stop_reason == "stop"


def test_the_model_recorded_is_the_one_the_server_reported() -> None:
    # A free tier that silently substitutes a different model is exactly the thing a
    # result record has to capture, so the response's model wins over the request's.
    client, _ = stub_client()
    swapped = CONFIG.model_copy(update={"model": "some-model-that-was-not-served"})
    assert client.complete("s", "p", swapped).model == "llama-3.3-70b-versatile"


def test_streaming_yields_deltas_then_a_completion() -> None:
    client, _ = stub_client()
    chunks = list(client.stream("s", "p", CONFIG))
    assert isinstance(chunks[-1], Completion)
    assert list(chunks[:-1]) == ["Warmup ", "stabilises [1]."]
    assert chunks[-1].text == "Warmup stabilises [1]."
    assert chunks[-1].output_tokens == 30


def test_streaming_survives_a_chunk_with_no_content() -> None:
    # The final usage-only chunk carries no choices at all.
    client, _ = stub_client()
    assert isinstance(list(client.stream("s", "p", CONFIG))[-1], Completion)


# --- pricing ----------------------------------------------------------------


@pytest.mark.parametrize(
    "model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen2.5:7b-instruct"]
)
def test_free_tier_and_local_models_cost_nothing(model: str) -> None:
    # Zero is the honest number for a free tier, not a placeholder. What a free tier does
    # cost is a request against a daily quota, which is throughput, not budget.
    assert cost_usd(model, 1_000_000, 1_000_000) == 0.0


def test_paid_models_still_cost_something() -> None:
    # The zero-rated entries must not have made every model free by accident.
    assert cost_usd("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.00)


def test_every_endpoint_in_the_registry_is_a_url() -> None:
    assert all(url.startswith("http") for url in OPENAI_COMPATIBLE_ENDPOINTS.values())
    assert "groq" in OPENAI_COMPATIBLE_ENDPOINTS
    assert "ollama" in OPENAI_COMPATIBLE_ENDPOINTS  # a local model is just an endpoint


def test_every_zero_rated_model_is_deliberate() -> None:
    free = [name for name, rates in MODEL_RATES.items() if rates.input_per_mtok == 0.0]
    assert free, "the free-tier entries went missing"
    assert not any(name.startswith("claude-") for name in free)


# --- the factory ------------------------------------------------------------


def test_an_openai_compatible_config_without_a_base_url_fails_helpfully() -> None:
    with pytest.raises(GenerationError, match="groq="):
        client_for(GenerationConfig(provider="openai_compatible", model="whatever"))


def test_the_config_carries_no_credential() -> None:
    # GenerationConfig is serialised verbatim into every answer log and every evaluation
    # result. A key in the config is a key committed to the repository.
    serialised = CONFIG.fingerprint()
    assert not any("key" in field.lower() for field in serialised)
    assert not any("secret" in field.lower() for field in serialised)


def test_the_provider_and_endpoint_are_recorded_in_the_fingerprint() -> None:
    # Two providers running nominally the same weights do not necessarily produce the
    # same output, so a result has to say which one answered.
    serialised = CONFIG.fingerprint()
    assert serialised["provider"] == "openai_compatible"
    assert serialised["base_url"] == OPENAI_COMPATIBLE_ENDPOINTS["groq"]


def test_switching_provider_changes_the_fingerprint() -> None:
    assert GenerationConfig().fingerprint() != CONFIG.fingerprint()
