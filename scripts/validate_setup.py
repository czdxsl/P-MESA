"""Validate paths in a task YAML before an expensive run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {
        "dataset.root": Path(config["dataset"]["root"]),
        "model.checkpoint": Path(config["model"]["checkpoint"]),
    }
    missing = [f"{key}: {path}" for key, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit("Missing required artifacts:\n  " + "\n  ".join(missing))
    print(f"OK: {config['task']} / {config['dataset']['name']} / {config['model']['name']}")


if __name__ == "__main__":
    main()

