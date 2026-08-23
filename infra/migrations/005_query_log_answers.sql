-- 005_query_log_answers.sql — record what was answered, not only what was retrieved.
--
-- query_logs was defined in 002 for the retrieval path. Generation adds facts that the
-- original columns cannot hold, and every one of them is something the evaluation or the
-- ops page needs to read back:
--
--   answer          what was actually said, for citation-accuracy judging
--   cited_chunk_ids which chunks the claims resolved to, as distinct from which were
--                   retrieved -- the gap between the two is a real signal
--   uncited         the answer left a factual sentence unsupported after the retry
--   refusal_reason  *why* it refused; "below the floor" and "the model declined" are
--                   different behaviours and the golden set scores them separately
--   generation      the generation config, because a result whose prompt version and
--                   model cannot be reconstructed is not reproducible
--   llm_calls       one answer can cost more than one call, thanks to the citation retry

ALTER TABLE query_logs ADD COLUMN answer          TEXT;
ALTER TABLE query_logs ADD COLUMN cited_chunk_ids BIGINT[] NOT NULL DEFAULT '{}';
ALTER TABLE query_logs ADD COLUMN uncited         BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE query_logs ADD COLUMN refusal_reason  TEXT;
ALTER TABLE query_logs ADD COLUMN generation      JSONB;
ALTER TABLE query_logs ADD COLUMN llm_calls       INT NOT NULL DEFAULT 0;

-- The ops page in Phase 9 reports refusal rate over a time window, and scanning the
-- whole table for it gets slower every day the demo is up.
CREATE INDEX query_logs_refused_idx ON query_logs (refused, created_at DESC);
