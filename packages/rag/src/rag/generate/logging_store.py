"""Persisting every answered question to `query_logs`.

Every request is recorded, refusals included -- especially refusals. Refusal rate is a
headline operational number and the only way to notice a score floor set so high the
system declines everything, or so low it never declines anything. A log that skipped the
refusals would make both look identical to a healthy system.

Writing is best-effort. A failure to log is reported and swallowed rather than raised: an
answer that was produced correctly should still reach the user when the observability
write fails behind it.
"""

import json
import logging

import psycopg

from rag.config import GenerationConfig, RetrievalConfig
from rag.generate.answer import Answer

logger = logging.getLogger(__name__)


def record(
    conn: psycopg.Connection,
    answer: Answer,
    config: RetrievalConfig,
    generation: GenerationConfig,
) -> int | None:
    """Write one answered question to `query_logs`, returning its id.

    Returns None if the write failed, having logged the reason.
    """
    try:
        with conn.transaction():
            row = conn.execute(
                """
                INSERT INTO query_logs (
                    question, config, generation, retrieved_ids, cited_chunk_ids,
                    stage_timings, input_tokens, output_tokens, cost_usd,
                    refused, refusal_reason, uncited, answer, llm_calls
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    answer.question,
                    json.dumps(config.model_dump()),
                    json.dumps(generation.fingerprint()),
                    answer.retrieved_chunk_ids,
                    list(answer.cited_chunk_ids),
                    json.dumps(answer.timings_ms),
                    answer.usage.input_tokens,
                    answer.usage.output_tokens,
                    round(answer.usage.cost_usd, 6),
                    answer.refused,
                    answer.refusal_reason,
                    answer.uncited,
                    answer.text,
                    answer.usage.calls,
                ),
            ).fetchone()
        return None if row is None else int(row[0])
    except psycopg.Error:
        logger.exception("failed to write query_logs row; the answer itself is unaffected")
        return None
