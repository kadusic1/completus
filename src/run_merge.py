"""Entry point for LoRA adapter merging and GGUF export.

Usage::

    uv run python src/run_merge.py

Produces merged/python/, merged/go/, gguf/python_coder.gguf, and
gguf/go_coder.gguf.
"""

from loguru import logger

from merge_export import MergeConfig, ModelMerger


def main() -> None:
    """Run sequential merge + export for Python and Go adapters."""
    cfg = MergeConfig()

    for lang, adapter_dir, gguf_path in cfg.adapters:
        logger.info("Merging {} adapter from {}", lang, adapter_dir)
        merger = ModelMerger(cfg)
        merger.load(adapter_dir)
        merger.export(f"{cfg.output_base}/{lang}", gguf_path)
        merger.cleanup()
        logger.info("{} merge complete", lang)


if __name__ == "__main__":
    main()
