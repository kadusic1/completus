"""Entry point for the HumanEval evaluation.

Usage::

    uv run python src/run_eval.py
    uv run python src/run_eval.py --model merged

Produces data/eval/results/pass_at_k.json (baseline) or
data/eval/results/pass_at_k_merged.json (merged).
"""

import argparse

from eval import BaselineConfig, BaselineEvaluator


def main() -> None:
    """Run the evaluation with optional merged model flag."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on HumanEval",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "merged"],
        default="baseline",
        help="Model to evaluate (default: baseline)",
    )
    args = parser.parse_args()

    if args.model == "merged":
        cfg = BaselineConfig(
            model_name="merged/python",
            results_file="pass_at_k_merged.json",
        )
    else:
        cfg = BaselineConfig()

    evaluator = BaselineEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
