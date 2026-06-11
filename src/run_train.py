"""Entry point for QLoRA fine-tuning (Python + Go, sequential).

Produces adapters/python_lora/ and adapters/go_lora/.
"""

from loguru import logger

from train import ModelTrainer, TrainingConfig


def main() -> None:
    """Run sequential QLoRA fine-tuning for Python and Go."""
    cfg = TrainingConfig()
    runs = [
        ("python", "data/processed/python_train.jsonl", "adapters/python_lora"),
        ("go", "data/processed/go_train.jsonl", "adapters/go_lora"),
    ]

    for lang, data_path, out_dir in runs:
        logger.info("Starting {} fine-tuning", lang)
        trainer = ModelTrainer(cfg)
        trainer.load()
        trainer.train(data_path, out_dir)
        trainer.validate(out_dir)
        trainer.cleanup()
        logger.info("{} fine-tuning complete", lang)


if __name__ == "__main__":
    main()
