# Completus

LoRA fine-tuning pipeline for `Qwen2.5-Coder-1.5B` on CodeSearchNet (Python & Go).

## Dataset statistics

`Raw` is the full CodeSearchNet train split. `Clean` is the subset kept
after applying the quality filter in `src/eda.py`.

| Language | Raw    | Clean  | Retention |
|----------|--------|--------|-----------|
| python   | 412178 | 384596 | 93.3%     |
| go       | 317832 | 278357 | 87.6%     |

### Filter criteria (raw to clean)

A sample is kept only if all of the following hold:

- Docstring longer than 20 characters.
- Code longer than 50 characters.
- Approximate token count (`len(code) // 4`) at most 2048.
- Code does not contain `TODO`.
- Code spans more than 2 newlines.
