"""Entry point for the CodeSearchNet data preparation pipeline.

Usage:
    python src/prepare_data.py

Produces data/processed/{python,go}_train.jsonl.
"""

import json
import warnings

from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer

from pipeline import (
    CodeSample,
    CodeSmellFilter,
    DocstringFilter,
    FIMTransform,
    LengthManagement,
    NearDedupStage,
    Pipeline,
    PipelineConfig,
    TRAINING_SAMPLE_SIZE,
)


def main() -> None:
    """Load CodeSearchNet and run the pipeline for Python and Go."""
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    logger.info("Starting data preparation pipeline...")
    # setup_logging()
    for lang in ["python", "go"]:
        ds = load_dataset("claudios/code_search_net", lang, split="train")
        raw = [
            CodeSample(
                func_code_string=ds[i]["func_code_string"],
                func_documentation_string=ds[i]["func_documentation_string"],
                func_name=ds[i]["func_name"],
                text="",
            )
            for i in range(min(TRAINING_SAMPLE_SIZE, len(ds)))
        ]
        config = PipelineConfig()
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        stages = [
            NearDedupStage(),
            CodeSmellFilter(tokenizer),
            DocstringFilter(tokenizer),
            FIMTransform(),
            LengthManagement(tokenizer),
        ]
        processed = Pipeline(stages).run(raw, config)
        out = f"data/processed/{lang}_train.jsonl"
        with open(out, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")
        logger.info("Saved to {}", out)


if __name__ == "__main__":
    main()
