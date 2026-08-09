"""Command-line entry points."""

from __future__ import annotations

import argparse

from .demo import run_demo
from .serialization import save_explanation


def main() -> None:
    parser = argparse.ArgumentParser(prog="pmesa")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the deterministic end-to-end demo")
    demo.add_argument("--output", default="outputs/demo/explanation.json")
    demo.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.command == "demo":
        explanation = run_demo(args.seed)
        save_explanation(explanation, args.output)
        print(f"selected: {', '.join(u.id for u in explanation.selected_units)}")
        print(f"max completeness error: {explanation.attribution.completeness_error.abs().max().item():.6g}")
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

