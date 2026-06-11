"""Entry point for the HumanEval baseline evaluation.

Usage::

    uv run python src/run_eval.py

Produces data/eval/results/pass_at_k.json.
"""

from eval import BaselineConfig, BaselineEvaluator


def main() -> None:
    """Run the baseline evaluation with default configuration."""
    cfg = BaselineConfig()
    evaluator = BaselineEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
