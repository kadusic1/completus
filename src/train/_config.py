from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for QLoRA fine-tuning runs.

    All hyperparameters are centralised here to avoid magic
    numbers scattered across modules.  Follows the Unsloth QLoRA
    best-practice guide and the QLoRA paper.

    Attributes:
        base_model: HuggingFace model ID for fine-tuning.
        max_seq_length: Maximum sequence length for tokenization.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling parameter.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: Linear layers to attach LoRA adapters to.
        batch_size: Per-device training batch size.
        grad_accum: Gradient accumulation steps.
        lr: Peak learning rate.
        epochs: Number of training epochs.
        warmup_steps: Linear warmup steps.
        optim: Optimizer type.
        scheduler: LR scheduler type.
        weight_decay: Weight decay for optimizer.
        max_grad_norm: Maximum gradient norm for clipping.
        seed: Random seed for reproducibility.
    """

    base_model: str = "unsloth/Qwen2.5-Coder-1.5B"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 2e-4
    epochs: int = 1
    warmup_steps: int = 10
    optim: str = "adamw_8bit"
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    seed: int = 3407
