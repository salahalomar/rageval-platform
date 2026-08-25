/**
 * The retrieval knobs, live.
 *
 * This is the part of the interface worth showing a technical visitor: the same question
 * re-run with the cross-encoder off reorders the results in front of them, and the arms
 * in the ablation table stop being abstract. The fields mirror `RetrievalConfig` exactly
 * and are sent to the server per request, so what happens here is the same code path the
 * eval harness measures -- not a demo mode.
 */
import type { RetrievalConfig } from "../lib/api";

type Fusion = NonNullable<RetrievalConfig["fusion"]>;

const FUSION_LABELS: Record<Fusion, string> = {
  rrf: "hybrid (RRF)",
  dense_only: "dense only",
  lexical_only: "lexical only",
};

export default function ConfigPanel({
  config,
  onChange,
  disabled,
}: {
  config: RetrievalConfig;
  onChange: (next: RetrievalConfig) => void;
  disabled: boolean;
}) {
  return (
    <section className="config" aria-label="Retrieval configuration">
      <h2>Retrieval configuration</h2>
      <div className="row">
        <label>
          Fusion
          <select
            value={config.fusion}
            disabled={disabled}
            onChange={(event) =>
              onChange({ ...config, fusion: event.target.value as Fusion })
            }
          >
            {Object.entries(FUSION_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <input
            type="checkbox"
            checked={config.rerank_enabled}
            disabled={disabled}
            onChange={(event) =>
              onChange({ ...config, rerank_enabled: event.target.checked })
            }
          />
          Cross-encoder rerank
        </label>

        <label>
          Final top-k
          <input
            type="number"
            min={1}
            max={20}
            value={config.final_top_k}
            disabled={disabled}
            onChange={(event) => {
              const parsed = Number(event.target.value);
              if (Number.isFinite(parsed) && parsed >= 1 && parsed <= 20) {
                onChange({ ...config, final_top_k: parsed });
              }
            }}
          />
        </label>
      </div>
      <p className="hint">
        Lexical search is Postgres <code>ts_rank_cd</code>, not BM25. Reranking is{" "}
        <code>bge-reranker-base</code> over the fused top 50; its scores are sigmoid
        outputs in [0, 1] and are not comparable to the RRF scores below them.
      </p>
    </section>
  );
}
