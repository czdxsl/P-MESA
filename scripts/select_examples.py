"""Select qualitative example IDs from genuine JSONL run records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pmesa.selection import CandidateExample, select_representative_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--metric", default="faithfulness")
    parser.add_argument("--min-quality", type=float)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line]
    candidates = [CandidateExample(
        id=row["example_id"], metrics=row["metrics"], category=row.get("category", "default")
    ) for row in rows]
    selected = select_representative_examples(
        candidates, count=args.count, metric=args.metric, min_quality=args.min_quality
    )
    print(json.dumps({
        "source": str(Path(args.records).resolve()),
        "metric": args.metric,
        "min_quality": args.min_quality,
        "selected": [{"id": e.id, "category": e.category, "score": e.metrics[args.metric]} for e in selected],
    }, indent=2))


if __name__ == "__main__":
    main()

