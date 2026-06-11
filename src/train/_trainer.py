"""ModelTrainer - OOP orchestrator for QLoRA fine-tuning."""

import json
from pathlib import Path

from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
from datasets import Dataset
from loguru import logger
from transformers import TrainingArguments
from trl import SFTTrainer
from train._config import TrainingConfig


class ModelTrainer:
    """Orchestrates a single QLoRA fine-tuning run.

    Owns the model, tokenizer, and trainer lifecycle.  Call
    ``load()`` once, then ``train()`` for each language dataset.

    Usage::

        trainer = ModelTrainer(TrainingConfig())
        trainer.load()
        trainer.train("data/processed/python_train.jsonl",
                       "adapters/python_lora")
        trainer.validate("adapters/python_lora")
        trainer.cleanup()
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        """Initialise the trainer.

        The model is **not** loaded until :meth:`load` is called,
        so you can construct this object without a GPU.

        Args:
            cfg: Training configuration.
        """
        self._cfg = cfg
        self._model: FastLanguageModel | None = None
        self._tokenizer: None = None
        self._trainer: SFTTrainer | None = None

    def load(self) -> None:
        """Load the base model, tokenizer, and attach LoRA adapters.

        Idempotent - subsequent calls are no-ops.
        """
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info("Loading base model and tokenizer...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self._cfg.base_model,
            max_seq_length=self._cfg.max_seq_length,
            load_in_4bit=True,
            dtype=None,
            trust_remote_code=True,
        )

        logger.info("Attaching LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=self._cfg.lora_r,
            lora_alpha=self._cfg.lora_alpha,
            lora_dropout=self._cfg.lora_dropout,
            target_modules=list(self._cfg.target_modules),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self._cfg.seed,
            max_seq_length=self._cfg.max_seq_length,
        )

        self._model = model
        self._tokenizer = tokenizer

    def train(self, dataset_path: str, output_dir: str) -> None:
        """Run the training loop on a JSONL dataset.

        Args:
            dataset_path: Path to the training JSONL file.
            output_dir: Directory to save the LoRA adapter.
        """
        assert self._model is not None
        assert self._tokenizer is not None

        logger.info("Loading dataset from {}", dataset_path)
        with open(dataset_path) as f:
            records = [json.loads(line) for line in f]
        dataset = Dataset.from_list(records)

        def _append_eos(examples: dict) -> dict:
            examples["text"] = [t + self._tokenizer.eos_token for t in examples["text"]]
            return examples

        dataset = dataset.map(_append_eos, batched=True)

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        bf16 = is_bfloat16_supported()

        args = TrainingArguments(
            output_dir=str(output),
            per_device_train_batch_size=self._cfg.batch_size,
            gradient_accumulation_steps=self._cfg.grad_accum,
            num_train_epochs=self._cfg.epochs,
            learning_rate=self._cfg.lr,
            warmup_steps=self._cfg.warmup_steps,
            optim=self._cfg.optim,
            lr_scheduler_type=self._cfg.scheduler,
            weight_decay=self._cfg.weight_decay,
            max_grad_norm=self._cfg.max_grad_norm,
            fp16=not bf16,
            bf16=bf16,
            seed=self._cfg.seed,
            data_seed=self._cfg.seed,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            ddp_find_unused_parameters=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        logger.info("Starting training...")
        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            args=args,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self._cfg.max_seq_length,
        )
        trainer.train()

        logger.info("Saving adapter to {}", output_dir)
        self._model.save_pretrained(str(output))
        self._tokenizer.save_pretrained(str(output))
        self._trainer = trainer

    def cleanup(self) -> None:
        """Free GPU memory by deleting model and trainer."""
        logger.info("Cleaning up GPU memory...")
        # Attempt to delete model and trainer attributes if they exist
        model = getattr(self, "_model", None)
        if model is not None:
            del model
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            del trainer
        self._model = None
        self._tokenizer = None
        self._trainer = None
        torch.cuda.empty_cache()

    def validate(self, adapter_dir: str) -> bool:
        """Verify that a saved adapter directory is complete.

        Args:
            adapter_dir: Path to the adapter directory to validate.

        Returns:
            True if the directory contains valid adapter files.
        """
        path = Path(adapter_dir)
        config_file = path / "adapter_config.json"
        model_file = path / "adapter_model.safetensors"

        if not config_file.exists():
            logger.error("Missing adapter_config.json in {}", adapter_dir)
            return False
        if not model_file.exists():
            logger.error("Missing adapter_model.safetensors in {}", adapter_dir)
            return False
        if config_file.stat().st_size == 0:
            logger.error("adapter_config.json is empty in {}", adapter_dir)
            return False
        if model_file.stat().st_size == 0:
            logger.error("adapter_model.safetensors is empty in {}", adapter_dir)
            return False

        logger.info("Validation passed for {}", adapter_dir)
        return True
