"""Pure generation logic for HumanEval completions (no I/O)."""

import torch
from transformers import PreTrainedTokenizerBase
from transformers import PreTrainedModel

from eval._config import BaselineConfig


def generate_all(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problems: dict,
    cfg: BaselineConfig,
) -> list[dict]:
    """Generate completions for every problem using batched decoding.

    Left-padding is used internally so that decoder-only generation
    works correctly with batches.  Each sample dict follows the
    ``human_eval`` JSONL format: ``{"task_id": ..., "completion": ...}``.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer (padding_side is set to
            ``"left"`` inside this function).
        problems: Dict mapping task_id -> problem dict.
        cfg: Baseline configuration.

    Returns:
        List of sample dicts ready for ``write_jsonl``.
    """
    task_ids = list(problems.keys())
    prompts = [problems[tid]["prompt"] for tid in task_ids]
    samples: list[dict] = []

    for i in range(0, len(prompts), cfg.batch_size):
        batch = prompts[i : i + cfg.batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=not cfg.greedy,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for j, tid in enumerate(task_ids[i : i + cfg.batch_size]):
            completion = tokenizer.decode(
                outputs[j][prompt_len:],
                skip_special_tokens=True,
            )
            for _ in range(cfg.num_samples):
                samples.append({"task_id": tid, "completion": completion})

    return samples
