"""CLI entrypoint for the AI brain system."""

from __future__ import annotations

import argparse

from brain.controller import thinking_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run the thinking algorithm pipeline.")
    parser.add_argument("problem", nargs="?", help="Problem statement to process.")
    return parser


def main() -> None:
    """Run the CLI workflow."""
    parser = build_parser()
    args = parser.parse_args()
    problem = args.problem.strip() if args.problem else input("Enter problem: ").strip()

    if not problem:
        raise SystemExit("A problem statement is required.")

    result = thinking_pipeline(problem)
    print(result["final_answer"])


if __name__ == "__main__":
    main()
