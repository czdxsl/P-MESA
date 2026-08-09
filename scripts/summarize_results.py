"""Compute paper-ready aggregate metrics from per-example JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records")
    parser.add_argument("--output")
    args = parser.parse_args()
    source = Path(args.records)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    keys = sorted(set.intersection(*(set(row["metrics"]) for row in rows))) if rows else []
    summary = {
        "source": str(source.resolve()),
        "n": len(rows),
        "metrics": {
            key: {
                "mean": float(np.mean([row["metrics"][key] for row in rows])),
                "std": float(np.std([row["metrics"][key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
            for key in keys
        },
    }
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
