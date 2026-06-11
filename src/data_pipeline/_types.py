from dataclasses import dataclass
from typing import TypedDict


class CodeSample(TypedDict, total=False):
    """A single sample from CodeSearchNet as processed by the pipeline.

    Attributes:
        func_code_string: The function body as a string.
        func_documentation_string: Docstring or comment.
        func_name: Name of the function.
        text: FIM-formatted or raw text after transformation.
    """

    func_code_string: str
    func_documentation_string: str
    func_name: str
    text: str


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the full data preparation pipeline.

    All tuning parameters are centralised here to avoid magic
    numbers scattered across stage files.

    Attributes:
        model_name: HuggingFace model ID for tokenization.
        max_tokens: Maximum sequence length.
        min_tokens: Drop sequences below this token count.
        fim_rate: Probability of applying FIM transform.
        context_ratio: Minimum ratio of context to completion.
        dedup_threshold: Jaccard threshold for MinHash LSH.
        shingle_size: N-gram size for MinHash shingling.
        max_line_length: Drop functions with any line exceeding this.
        docstring_min_tokens: Minimum tokens for docstrings.
        seed: Random seed for reproducibility.
    """

    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    max_tokens: int = 2048
    min_tokens: int = 30
    fim_rate: float = 0.8
    context_ratio: float = 3.0
    dedup_threshold: float = 0.8
    shingle_size: int = 5
    max_line_length: int = 120
    docstring_min_tokens: int = 15
    seed: int = 42
