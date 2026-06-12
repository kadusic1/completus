"""ModelMerger - OOP orchestrator for LoRA merging and GGUF export."""

from pathlib import Path

from unsloth import FastLanguageModel

import torch
from loguru import logger

from merge_export._config import MergeConfig


class ModelMerger:
    """Orchestrates LoRA adapter merging and GGUF export.

    Owns the model and tokenizer lifecycle.  Call ``load()`` once,
    then ``export()`` for safetensors + GGUF output.

    Usage::

        merger = ModelMerger(MergeConfig())
        merger.load("adapters/python_lora")
        merger.export("merged/python", "gguf/python_coder.gguf")
        merger.cleanup()
    """

    def __init__(self, cfg: MergeConfig) -> None:
        """Initialise the merger.

        The model is **not** loaded until :meth:`load` is called,
        so you can construct this object without a GPU.

        Args:
            cfg: Merge configuration.
        """
        self._cfg = cfg
        self._model: FastLanguageModel | None = None
        self._tokenizer: None = None

    def load(self, adapter_path: str) -> None:
        """Load the base model with merged LoRA adapter.

        Uses ``FastLanguageModel.from_pretrained`` which loads the
        base model and applies adapter weights in one step.
        Idempotent - subsequent calls are no-ops.

        Args:
            adapter_path: Path to the LoRA adapter directory.
        """
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info("Loading base model + adapter from {}", adapter_path)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=self._cfg.max_seq_length,
            load_in_4bit=True,
            dtype=None,
            trust_remote_code=True,
        )

        self._model = model
        self._tokenizer = tokenizer

    def export(self, output_dir: str, gguf_path: str) -> None:
        """Merge adapter weights and export safetensors + GGUF.

        Calls ``save_pretrained_gguf()`` which produces both
        safetensors (for transformers eval) and GGUF (for Ollama
        deployment) in a single step.  The GGUF file is then moved
        to the designated ``gguf_path``.

        Args:
            output_dir: Directory for merged safetensors output.
            gguf_path: Destination path for the GGUF file.
        """
        assert self._model is not None
        assert self._tokenizer is not None

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Exporting safetensors + GGUF to {}",
            output_dir,
        )

        self._model.save_pretrained_gguf(
            str(out),
            self._tokenizer,
            self._cfg.gguf_quant,
        )

        gguf_dir = out.parent / f"{out.name}_gguf"
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if not gguf_files:
            msg = f"No GGUF file found in {output_dir} after export"
            logger.error(msg)
            raise FileNotFoundError(msg)

        src = gguf_files[0]
        dst = Path(gguf_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        logger.info("GGUF moved to {}", dst)

    def cleanup(self) -> None:
        """Free GPU memory by deleting model and tokenizer."""
        logger.info("Cleaning up GPU memory...")
        model = getattr(self, "_model", None)
        if model is not None:
            del model
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is not None:
            del tokenizer
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()
