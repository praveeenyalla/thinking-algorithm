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
    print(f"Controller strategy: {result['controller']['strategy']}")
    print(f"Controller reason: {result['controller']['reason']}")
    print(
        "Memory cleanup: "
        f"episodic_deleted={result['memory_cleanup']['episodic_deleted']} "
        f"case_deleted={result['memory_cleanup']['case_deleted']}"
    )
    if result["case_memory"]:
        print(
            "Case memory match: "
            f"similarity={result['case_memory']['similarity']} "
            f"score={result['case_memory']['score']} "
            f"weighted={result['case_memory']['weighted_score']}"
        )
    if result["case_reused"]:
        print("Case reuse: matched past solution used directly")
    print("Generated options:")
    for index, option in enumerate(result["options"], start=1):
        evaluation = result["evaluations"][index - 1]
        print(f"{index}. {option}")
        print(
            f"   score={evaluation['total_score']} "
            f"passed={evaluation['threshold_passed']} "
            f"mode={evaluation['judge_mode']}"
        )
        print(f"   reason={evaluation['summary_reason']}")
        print(
            "   retrieved="
            + ", ".join(item["source"] for item in evaluation["retrieved_context"])
        )
    print("\nSelected Answer Before Refinement:")
    print(result["best_option"])
    print(f"Initial Score: {result['best_evaluation']['total_score']}")

    print("\n--- REFINEMENT RESULT ---")
    if result["improved"]:
        print("Improved answer selected")
    else:
        print("Original answer retained")
    print(f"Refined Score: {result['refined_evaluation']['total_score']}")
    print(f"Final Score: {result['final_score']}")
    print(f"Stored Memory Chunks: {result['stored_memory_chunks']}")
    print(f"Stored Case Memory: {result['stored_case']}")
    print("\nFinal Answer:\n")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
