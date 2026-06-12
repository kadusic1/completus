"""BaselineEvaluator - OOP orchestrator for the HumanEval eval loop."""

import json
from pathlib import Path
import torch

from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from eval._compat import apply_windows_patches
from eval._config import BaselineConfig
from eval._generator import generate_all

apply_windows_patches()


class BaselineEvaluator:
    """Orchestrates the HumanEval baseline evaluation pipeline.

    Owns the model and tokenizer lifecycle.  Call :meth:`run` once.

    Usage::

        evaluator = BaselineEvaluator(BaselineConfig())
        results = evaluator.run()
    """

    def __init__(self, cfg: BaselineConfig | None = None) -> None:
        """Initialise the evaluator.

        The model is **not** loaded until :meth:`load` or :meth:`run`
        is called, so you can construct this object without a GPU.

        Args:
            cfg: Baseline configuration.  Uses defaults when ``None``.
        """
        self._cfg = cfg or BaselineConfig()
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> None:
        """Load the model and tokenizer onto the device.

        Idempotent - subsequent calls are no-ops.
        """
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self._cfg.model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            attn_implementation="sdpa",
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            pretrained_model_name_or_path=self._cfg.model_name,
        )

        model.eval()
        self._model = model
        self._tokenizer = tokenizer

    def run(self) -> dict:
        """Run the full evaluation pipeline.

        Steps: load model -> read problems -> generate completions
        -> write samples -> evaluate -> save results.

        Returns:
            Dict of pass@k metrics (e.g. ``{"pass@1": 0.439}``).
        """
        self.load()
        assert self._model is not None
        assert self._tokenizer is not None

        logger.info("Reading HumanEval problems...")
        problems = read_problems()

        logger.info(
            "Generating completions...",
            len(problems) * self._cfg.num_samples,
            self._cfg.batch_size,
        )
        samples = generate_all(self._model, self._tokenizer, problems, self._cfg)

        logger.info("Writing samples to %s", self._cfg.samples_file)
        write_jsonl(self._cfg.samples_file, samples)

        logger.info("Evaluating functional correctness...")
        results = evaluate_functional_correctness(
            self._cfg.samples_file,
            k=[1],
        )

        self._save_results(results)
        self._print_summary(results)

        return results

    def _save_results(self, results: dict) -> None:
        """Write evaluation results to ``pass_at_k.json``.

        Args:
            results: Dict of metric name -> score.
        """
        path = Path(self._cfg.results_dir)
        path.mkdir(parents=True, exist_ok=True)
        out = path / self._cfg.results_file
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to {}", out)

    @staticmethod
    def _print_summary(results: dict) -> None:
        logger.info("HumanEval Baseline Results")
        for metric, score in results.items():
            logger.info(f"{metric}: {score:.3f} ({score * 100:.1f}%)")
