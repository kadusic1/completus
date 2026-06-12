from dataclasses import dataclass


@dataclass(frozen=True)
class MergeConfig:
    """Configuration for LoRA adapter merging and GGUF export.

    Attributes:
        base_model: HuggingFace model ID of the base model.
        max_seq_length: Maximum sequence length for tokenization.
        adapters: Tuple of (lang, adapter_dir, gguf_output_path)
            per language.
        output_base: Base directory for merged safetensors output.
        gguf_dir: Directory for GGUF files.
        gguf_quant: GGUF quantization method.
    """

    base_model: str = "unsloth/Qwen2.5-Coder-1.5B"
    max_seq_length: int = 2048
    adapters: tuple[tuple[str, str, str], ...] = (
        ("python", "adapters/python_lora", "gguf/python_coder.gguf"),
        ("go", "adapters/go_lora", "gguf/go_coder.gguf"),
    )
    output_base: str = "merged"
    gguf_dir: str = "gguf"
    gguf_quant: str = "q4_k_m"
