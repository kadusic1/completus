from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the HumanEval baseline evaluation.

    All tuning parameters are centralised here to avoid magic
    numbers scattered across modules.

    Attributes:
        model_name: HuggingFace model ID for evaluation.
        greedy: Use greedy decoding (do_sample=False).
        max_new_tokens: Maximum tokens per completion.
        num_samples: Number of completions per problem.
        batch_size: Number of prompts per forward pass.
        results_dir: Directory for evaluation artifacts.
        samples_file: Path for generated samples JSONL.
    """

    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    greedy: bool = True
    max_new_tokens: int = 512
    num_samples: int = 1
    batch_size: int = 16
    results_dir: str = "data/eval/results"
    samples_file: str = "data/eval/samples.jsonl"
