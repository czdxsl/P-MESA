"""Deterministically select genuine qualitative candidates from a run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--min-full-delta", type=float, default=0.015)
    parser.add_argument("--max-completeness-error", type=float, default=0.06)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.manifest.read_text(encoding="utf-8"))
    eligible = []
    for row in run["examples"]:
        full_delta = row["support_original"] - row["support_blur_baseline"]
        if (abs(full_delta) >= args.min_full_delta and
                row["pmesa_mean_midpoint_completeness_error"] <= args.max_completeness_error):
            eligible.append((row["selected_subset_delta_fraction"], row["example_id"], full_delta))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    selected = eligible[:args.count]
    if len(selected) < args.count:
        raise SystemExit(f"only {len(selected)} examples meet the minimum full-score delta")
    report = {
        "source_manifest": str(args.manifest.resolve()),
        "selection_rule": "filter by minimum full-score delta and maximum path-integration completeness error, then rank by selected-subset delta fraction",
        "min_full_delta": args.min_full_delta,
        "max_completeness_error": args.max_completeness_error,
        "selected": [
            {"example_id": example_id, "selected_subset_delta_fraction": fraction, "full_delta": full_delta}
            for fraction, example_id, full_delta in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
