"""Run a manuscript task through a target-model integration module."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable

from pmesa.config import load_config
from pmesa.experiment import ExperimentIntegration, run_dataset
from pmesa.explainer import PMESAExplainer


def load_factory(specification: str) -> Callable[[dict, str], ExperimentIntegration]:
    if ":" not in specification:
        raise ValueError("integration must use module:function syntax")
    module_name, function_name = specification.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError("integration factory must be callable")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--integration", required=True, help="module:function target integration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload, method = load_config(args.config)
    integration = load_factory(args.integration)(payload, args.device)
    examples = list(integration.examples())
    for seed in method.seeds:
        explainer = PMESAExplainer(
            steps=method.integration_points,
            path_count=method.restoration_paths,
            budget=method.evidence_budget,
            weights=method.objective_weights,
            seed=seed,
            device=args.device,
        )
        destination = args.output / f"seed-{seed}.jsonl"
        run_dataset(
            examples,
            id_of=integration.example_id,
            category_of=integration.category,
            adapter=integration.adapter,
            explainer=explainer,
            evaluate=integration.evaluate,
            output=destination,
        )
        print(destination)


if __name__ == "__main__":
    main()
