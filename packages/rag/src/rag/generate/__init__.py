"""Answer generation: prompt assembly, citation binding, refusal.

The rule this package exists to enforce is that no answer leaves the system without its
claims bound to `chunk_id`s, and that a question the corpus cannot support is refused
rather than guessed at.
"""

from rag.generate.answer import Answer, answer_question, answer_question_stream
from rag.generate.citations import CitationBinding, bind, parse_markers
from rag.generate.client import (
    AnthropicClient,
    Completion,
    LLMClient,
    OpenAICompatibleClient,
    ScriptedLLMClient,
    Usage,
    client_for,
    cost_usd,
)
from rag.generate.prompt import INSUFFICIENT_EVIDENCE

__all__ = [
    "INSUFFICIENT_EVIDENCE",
    "Answer",
    "AnthropicClient",
    "CitationBinding",
    "Completion",
    "LLMClient",
    "OpenAICompatibleClient",
    "ScriptedLLMClient",
    "Usage",
    "answer_question",
    "answer_question_stream",
    "bind",
    "client_for",
    "cost_usd",
    "parse_markers",
]
