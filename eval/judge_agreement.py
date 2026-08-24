"""Cohen's kappa between the judge and a human's own labels.

An unvalidated judge is an unmeasured system wearing a lab coat. A faithfulness score of
0.95 means nothing on its own — it is equally consistent with a genuinely faithful system
and with a judge that rubber-stamps everything, and those two possibilities are the whole
question. Kappa separates them, because it corrects for the agreement two raters would
reach by chance alone.

Below about 0.4 the judge is not measuring what it claims to and the prompt needs work.
That threshold is a convention, not a law, and the number is reported either way rather
than being used as a gate that quietly hides a bad judge.

The hand labels are a committed file that a human writes. Nothing here can generate them:
a judge validated against labels produced by the same judge is a tautology.
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LABELS = Path("eval/golden/judge_labels.jsonl")

# Landis & Koch's conventional bands, quoted rather than invented.
KAPPA_BANDS: tuple[tuple[float, str], ...] = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (-1.0, "poor — the judge is not measuring what it claims to"),
)


@dataclass(frozen=True, slots=True)
class Agreement:
    """Judge-versus-human agreement over one metric."""

    metric: str
    pairs: int
    observed: float
    expected: float
    kappa: float

    @property
    def interpretation(self) -> str:
        """The conventional band this kappa falls in."""
        return next(label for threshold, label in KAPPA_BANDS if self.kappa >= threshold)

    def as_lines(self) -> list[str]:
        """Human-readable report."""
        return [
            f"  metric                 {self.metric}",
            f"  labelled pairs         {self.pairs}",
            f"  observed agreement     {self.observed:.3f}",
            f"  chance agreement       {self.expected:.3f}",
            f"  Cohen's kappa          {self.kappa:.3f}  ({self.interpretation})",
        ]


def cohens_kappa(human: Sequence[str], judge: Sequence[str]) -> Agreement | None:
    """Cohen's kappa between two sequences of categorical labels.

    kappa = (observed - expected) / (1 - expected), where expected is the agreement two
    raters would reach by chance given their individual label distributions. Raw
    agreement alone is misleading: two raters who both answer "supported" 95% of the time
    agree 90% of the time while sharing no judgement at all.
    """
    if len(human) != len(judge) or not human:
        return None

    total = len(human)
    observed = sum(1 for a, b in zip(human, judge, strict=True) if a == b) / total

    categories = set(human) | set(judge)
    expected = sum(
        (sum(1 for label in human if label == category) / total)
        * (sum(1 for label in judge if label == category) / total)
        for category in categories
    )

    # Perfect chance agreement means the raters used one category between them; kappa is
    # undefined there, and 1.0 is the honest reading of two raters who never disagreed.
    kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1 - expected)
    return Agreement(metric="", pairs=total, observed=observed, expected=expected, kappa=kappa)


def load_labels(path: Path) -> list[dict[str, str]]:
    """Read hand labels: one JSON object per line with `metric`, `human` and `judge`."""
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            records.append({str(k): str(v) for k, v in json.loads(line).items()})
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    return records


def agreement_by_metric(records: Sequence[dict[str, str]]) -> list[Agreement]:
    """Kappa for each metric present in the labels."""
    metrics = sorted({record["metric"] for record in records if "metric" in record})
    results = []
    for metric in metrics:
        subset = [record for record in records if record.get("metric") == metric]
        computed = cohens_kappa(
            [record["human"] for record in subset], [record["judge"] for record in subset]
        )
        if computed is not None:
            results.append(
                Agreement(
                    metric=metric,
                    pairs=computed.pairs,
                    observed=computed.observed,
                    expected=computed.expected,
                    kappa=computed.kappa,
                )
            )
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stderr)

    if not args.labels.exists():
        print(f"no hand labels at {args.labels}.")
        print()
        print("This file is written by a human and by nothing else. Judge output validated")
        print("against labels produced by the same judge is a tautology, so there is no")
        print("command that can create it.")
        print()
        print("Format, one object per line:")
        print(
            '  {"item_id": "g-001", "metric": "faithfulness", '
            '"human": "supported", "judge": "supported"}'
        )
        return 1

    results = agreement_by_metric(load_labels(args.labels))
    if not results:
        print("no usable label pairs")
        return 1

    for agreement in results:
        print()
        for line in agreement.as_lines():
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
