/**
 * The retrieved chunks, in the numbering the answer's markers refer to.
 *
 * Every block shows where it came from -- paper, section path, pages and the character
 * span within the extracted text -- because "cited" is only a meaningful claim if a reader
 * can go and check it. The character offsets are the same ones the golden set's evidence
 * spans are expressed in, so a source block and a ground-truth annotation are talking
 * about the same coordinates.
 */
import { Fragment, useEffect, useRef } from "react";

import type { SourceBlock } from "../lib/api";

// Short function words carry no retrieval signal and highlighting them would paint half
// the paragraph, so they are dropped before matching.
const STOPWORDS = new Set([
  "the", "and", "for", "are", "was", "were", "with", "that", "this", "from", "have",
  "has", "does", "did", "what", "which", "when", "how", "why", "who", "into", "than",
  "then", "they", "them", "you", "your", "its", "it's", "can", "could", "would",
  "should", "about", "over", "under", "between", "using", "use", "used", "does",
]);

function terms(question: string): string[] {
  return [
    ...new Set(
      question
        .toLowerCase()
        .split(/[^a-z0-9+#.-]+/)
        .map((word) => word.replace(/^[.-]+|[.-]+$/g, ""))
        .filter((word) => word.length >= 3 && !STOPWORDS.has(word)),
    ),
  ];
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Highlight the question's content words inside a chunk.
 *
 * Worth being precise about what this is: term matching, not the model's attribution. The
 * system does not record which characters a claim came from, and pretending otherwise by
 * highlighting a span the generator never pointed at would be exactly the kind of
 * convincing-but-unmeasured flourish this repository exists to argue against.
 */
function highlight(content: string, question: string): React.ReactNode {
  const words = terms(question);
  if (words.length === 0) return content;

  const pattern = new RegExp(`(${words.map(escapeRegExp).join("|")})`, "gi");
  return content
    .split(pattern)
    .map((part, index) =>
      index % 2 === 1 ? <mark key={index}>{part}</mark> : <Fragment key={index}>{part}</Fragment>,
    );
}

export default function Sources({
  sources,
  question,
  selected,
  onSelect,
}: {
  sources: SourceBlock[];
  question: string;
  selected: number | null;
  onSelect: (marker: number | null) => void;
}) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selected === null || !container.current) return;
    const target = container.current.querySelector(`[data-marker="${selected}"]`);
    target?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selected]);

  return (
    <div className="sources" ref={container} data-testid="sources">
      {sources.map((source) => (
        <article
          key={source.chunk_id}
          data-marker={source.marker}
          data-testid="source"
          className={`source${selected === source.marker ? " selected" : ""}`}
          onClick={() => onSelect(selected === source.marker ? null : source.marker)}
        >
          <div className="meta">
            <span className="marker">[{source.marker}]</span>
            <span className="title">{source.paper_title}</span>
            <a
              href={`https://arxiv.org/abs/${source.paper_id}`}
              target="_blank"
              rel="noreferrer"
              onClick={(event) => event.stopPropagation()}
            >
              arXiv:{source.paper_id}
            </a>
          </div>
          <div className="meta">
            <span>{source.section_path || "(no section)"}</span>
            <span>
              {source.page_start === source.page_end
                ? `p. ${source.page_start}`
                : `pp. ${source.page_start}–${source.page_end}`}
            </span>
            <span className="scores">
              chars {source.char_start.toLocaleString()}–{source.char_end.toLocaleString()}
            </span>
            <span className="scores">
              {source.rerank_score === null || source.rerank_score === undefined
                ? `rrf ${source.score.toFixed(4)}`
                : `rerank ${source.rerank_score.toFixed(3)}`}
            </span>
          </div>
          <div className="body">{highlight(source.content, question)}</div>
        </article>
      ))}
    </div>
  );
}
