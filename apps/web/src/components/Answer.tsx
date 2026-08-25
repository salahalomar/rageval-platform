/**
 * The answer, with its `[n]` markers turned into controls that select a source.
 *
 * The regex matches the server's -- `packages/rag/src/rag/generate/citations.py` accepts
 * `[1]` and `[1, 3]` -- because a marker the UI silently failed to recognise would render
 * as literal text and quietly break the one guarantee this system makes, that every claim
 * is traceable to a chunk.
 */
import { Fragment } from "react";

const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

export default function Answer({
  text,
  selected,
  onSelect,
}: {
  text: string;
  selected: number | null;
  onSelect: (marker: number) => void;
}) {
  const pieces: React.ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(MARKER)) {
    const at = match.index;
    if (at > cursor) pieces.push(text.slice(cursor, at));
    const markers = (match[1] ?? "").split(",").map((part) => Number(part.trim()));
    pieces.push(
      <Fragment key={`${at}`}>
        {markers.map((marker) => (
          <button
            key={marker}
            type="button"
            className={`chip${selected === marker ? " selected" : ""}`}
            onClick={() => onSelect(marker)}
            aria-label={`Show source ${marker}`}
          >
            {marker}
          </button>
        ))}
      </Fragment>,
    );
    cursor = at + match[0].length;
  }
  if (cursor < text.length) pieces.push(text.slice(cursor));

  return (
    <div className="answer" data-testid="answer">
      {pieces}
    </div>
  );
}
