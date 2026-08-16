"""Validate paths in a task YAML before an expensive run."""

from __future__ import annotations

import argparse
from pathlib import Path

from pmesa.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--check-artifacts", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    payload, method = load_config(config_path)
    if args.check_artifacts:
        required = {
            "dataset.root": Path(payload["dataset"]["root"]),
            "target_model.checkpoint": Path(payload["target_model"]["checkpoint"]),
        }
        missing = [f"{key}: {path}" for key, path in required.items() if not path.exists()]
        if missing:
            raise SystemExit("Missing required artifacts:\n  " + "\n  ".join(missing))
    print(
        f"OK: {payload['task']} / {payload['dataset']['name']} / "
        f"{payload['target_model']['architecture']} / "
        f"points={method.integration_points} paths={method.restoration_paths} K={method.evidence_budget}"
    )


if __name__ == "__main__":
    main()
