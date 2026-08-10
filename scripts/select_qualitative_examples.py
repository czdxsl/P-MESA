from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--min-score-delta", type=float, default=0.02)
    parser.add_argument("--min-probability", type=float, default=0.7)
    parser.add_argument("--max-completeness-error", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = []
    for row in run["examples"]:
        complete = row["completeness_error"] <= args.max_completeness_error
        if run["dataset"] == "VQA-X":
            delta = row["score"] - row["baseline_score"]
            eligible = row["prediction_correct"] and delta >= args.min_score_delta and complete
            metric = delta
        elif run["dataset"] == "M-HalDetect":
            eligible = row["hallucination_probability"] >= args.min_probability and complete
            metric = row["hallucination_probability"]
        elif run["dataset"] == "TIIL":
            localization = row["localization"]
            pmesa = localization["pmesa"]
            baselines = [
                localization[name]["energy_in_mask"]
                for name in ("patch_similarity", "gradcam", "ig", "smoothgrad", "rise")
            ]
            gain = pmesa["energy_in_mask"] - max(baselines)
            eligible = row["consistency_margin"] > 0 and pmesa["pointing_game"] and gain > 0 and complete
            metric = gain
        else:
            raise SystemExit(f"unsupported dataset: {run['dataset']}")
        if eligible:
            selected.append({"example_id": row["example_id"], "selection_metric": metric})

    if len(selected) < args.count:
        raise SystemExit(f"only {len(selected)} examples pass the selection criteria")
    if run["dataset"] == "VQA-X":
        criteria = {"prediction_correct": True, "min_score_delta": args.min_score_delta}
    elif run["dataset"] == "M-HalDetect":
        criteria = {"min_hallucination_probability": args.min_probability}
    else:
        criteria = {"positive_margin": True, "pointing_game": True, "positive_mask_energy_gain": True}
    criteria["max_completeness_error"] = args.max_completeness_error
    report = {
        "source_manifest": str(args.manifest),
        "dataset": run["dataset"],
        "criteria": criteria,
        "selected": selected[:args.count],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
